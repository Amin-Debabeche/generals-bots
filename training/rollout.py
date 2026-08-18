"""Vectorized rollout collection (self-play and vs-heuristic-baseline) + GAE.

Only seat 0 is ever the trainable policy; seat 1 is either another set of
network params (self-play / league opponent) or a pure-JAX heuristic Agent
(RandomAgent/ExpanderAgent/HunterAgent from generals/agents/).

Trajectories store the COMPACT raw `Observation` AND `MemoryState` for seat 0
at each step (not a pre-expanded (19,H,W) tensor or legal-action mask) —
those are recomputed lazily inside training/ppo.py's minibatch loop. At
512envs x 256steps this is the difference between roughly 1GB and 2.7GB+ of
rollout buffer.

Memory (training/memory.py, per arXiv:2507.06825) is PER-SEAT, PER-EPISODE
state threaded through the scan carry alongside the GameState batch, and
reset to init_memory() for any env slot whose episode just ended (auto-reset
picks a fresh board from the pool; carrying stale memory into a new game
would leak information across episodes).

Reward uses `timestep.last_state` (the actual pre-auto-reset outcome of the
step), never `timestep.observation` (which reflects the FRESH pool board once
an episode ends) — using the latter would silently zero out the win/loss
signal on every terminal step, since a freshly reset board has a normal
owned/opponent army count rather than the 0-army signal
`composite_reward_fn`'s `game_done` check looks for.

PERFORMANCE NOTE (load-bearing, not a micro-optimization): on this JAX/XLA-CPU
build, any `lax.scan` whose body contains a conv op and that still lowers to
an actual `while_loop` (i.e. `unroll < length`) runs every conv call in that
body on a ~30-50x slower fallback algorithm — confirmed empirically: with a
32-step scan, unroll=1 and unroll=16 both cost ~93ms/step (16 envs), while
unroll=32 (fully unrolled, no while_loop at all) drops to ~1.7ms/step, a
~55x difference, on an otherwise-identical computation. Fully unrolling the
REAL env.step (far more complex than a bare conv stack — build-castle cost
grids, move-order resolution, scatter/gather) compiles acceptably fast only
for small step counts (~13s at unroll=8 for a 16-env/8x8 board; unroll=32
didn't finish compiling in over 10 minutes). The fix used throughout this
module: chunk `num_steps` into small fully-unrolled pieces (`_CHUNK_STEPS`)
run back-to-back from a plain Python loop — never nested inside another
lax.scan/while_loop, which would reintroduce the exact same pathology one
level up. Each distinct chunk size compiles once (cached by jax.jit) and is
reused for every chunk call and every training iteration thereafter.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game
from generals.core.observation import Observation

from training import action_space as asp
from training import augment as aug_mod
from training import memory as mem_mod
from training.augment import AugmentedObsState
from training.config import TransformerNetworkConfig
from training.memory import MemoryState
from training.network import PolicyValueNetwork
from training.network_transformer import TransformerPolicyValueNetwork
from training.rewards import shaped_reward_fn

_CHUNK_STEPS = 8  # see module docstring; small enough to compile in ~seconds,
                  # large enough to capture ~all of the achievable speedup


class StepData(NamedTuple):
    obs: Observation        # pre-action observation for seat 0, leading axis (num_steps, num_envs)
    mem: MemoryState        # memory state AS OF that same pre-action observation
    action_index: jnp.ndarray
    logprob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    winner: jnp.ndarray     # info.winner at this step (0/1/-1); only meaningful where done=True


def _get_obs_fn(env):
    return game.get_full_observation if env.perfect_info else game.get_observation


def _policy_forward(net: PolicyValueNetwork, obs: Observation, mem: MemoryState, key: jnp.ndarray,
                     build_castles_enabled: bool):
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, value = net(tensor)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    idx = asp.sample_action_index(key, logits, mask)
    logprob, _entropy = asp.logprob_and_entropy(logits, mask, idx)
    H, W = obs.armies.shape
    engine_action = asp.decode_action_index(idx, H, W)
    return idx, logprob, value, engine_action


def _value_forward(net: PolicyValueNetwork, obs: Observation, mem: MemoryState) -> jnp.ndarray:
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    _, value = net(tensor)
    return value


def _reset_memory_where_done(mem: MemoryState, done: jnp.ndarray) -> MemoryState:
    """mem: batched (num_envs, H, W, ...) fields. done: (num_envs,) bool.
    Resets any env slot whose episode just ended to a fresh init_memory()."""
    H, W = mem.revealed_structures.shape[-2:]
    init = mem_mod.init_memory(H, W)  # unbatched (H,W) fields, broadcasts against the batch dim below

    def select(cur, init_val):
        d = done.reshape((-1,) + (1,) * (cur.ndim - 1))
        return jnp.where(d, init_val, cur)

    return jax.tree.map(select, mem, init)


def _chunk_sizes(total_steps: int, chunk_steps: int) -> list[int]:
    full, remainder = divmod(total_steps, chunk_steps)
    sizes = [chunk_steps] * full
    if remainder:
        sizes.append(remainder)
    return sizes


@partial(jax.jit, static_argnames=("env", "chunk_steps", "build_castles_enabled"))
def _selfplay_chunk(env, states, mem0, mem1, pool, live_net, opp_net, key,
                     chunk_steps: int, build_castles_enabled: bool,
                     gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    get_obs = _get_obs_fn(env)

    def step_fn(carry, key_t):
        states, mem0, mem1 = carry
        num_envs = states.armies.shape[0]
        key_a, key_b = jrandom.split(key_t)
        keys_a = jrandom.split(key_a, num_envs)
        keys_b = jrandom.split(key_b, num_envs)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        idx0, logprob0, value0, action0 = jax.vmap(
            lambda o, m, k: _policy_forward(live_net, o, m, k, build_castles_enabled)
        )(obs0, mem0, keys_a)
        _idx1, _lp1, _v1, action1 = jax.vmap(
            lambda o, m, k: _policy_forward(opp_net, o, m, k, build_castles_enabled)
        )(obs1, mem1, keys_b)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        obs1_after = jax.vmap(lambda s: get_obs(s, 1))(timesteps.last_state)
        reward0 = jax.vmap(
            lambda po, a, o: shaped_reward_fn(po, a, o, gamma, contact_shaping_weight, general_safety_weight)
        )(obs0, action0, obs0_after)
        done0 = timesteps.terminated | timesteps.truncated
        winner0 = timesteps.info.winner

        mem0_updated = jax.vmap(mem_mod.update_memory)(mem0, obs0_after, action0)
        mem1_updated = jax.vmap(mem_mod.update_memory)(mem1, obs1_after, action1)
        mem0_next = _reset_memory_where_done(mem0_updated, done0)
        mem1_next = _reset_memory_where_done(mem1_updated, done0)

        step_data = StepData(obs0, mem0, idx0, logprob0, value0, reward0, done0, winner0)
        return (new_states, mem0_next, mem1_next), step_data

    keys = jrandom.split(key, chunk_steps)
    (final_states, final_mem0, final_mem1), trajectory = jax.lax.scan(
        step_fn, (states, mem0, mem1), keys, unroll=chunk_steps
    )
    return final_states, final_mem0, final_mem1, trajectory


@partial(jax.jit, static_argnames=("env", "heuristic_agent", "chunk_steps", "build_castles_enabled"))
def _vs_heuristic_chunk(env, states, mem0, pool, live_net, heuristic_agent, key,
                         chunk_steps: int, build_castles_enabled: bool,
                         gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    get_obs = _get_obs_fn(env)

    def step_fn(carry, key_t):
        states, mem0 = carry
        num_envs = states.armies.shape[0]
        key_a, key_b = jrandom.split(key_t)
        keys_a = jrandom.split(key_a, num_envs)
        keys_b = jrandom.split(key_b, num_envs)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        idx0, logprob0, value0, action0 = jax.vmap(
            lambda o, m, k: _policy_forward(live_net, o, m, k, build_castles_enabled)
        )(obs0, mem0, keys_a)
        action1 = jax.vmap(heuristic_agent.act)(obs1, keys_b)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        reward0 = jax.vmap(
            lambda po, a, o: shaped_reward_fn(po, a, o, gamma, contact_shaping_weight, general_safety_weight)
        )(obs0, action0, obs0_after)
        done0 = timesteps.terminated | timesteps.truncated
        winner0 = timesteps.info.winner

        mem0_updated = jax.vmap(mem_mod.update_memory)(mem0, obs0_after, action0)
        mem0_next = _reset_memory_where_done(mem0_updated, done0)

        step_data = StepData(obs0, mem0, idx0, logprob0, value0, reward0, done0, winner0)
        return (new_states, mem0_next), step_data

    keys = jrandom.split(key, chunk_steps)
    (final_states, final_mem0), trajectory = jax.lax.scan(
        step_fn, (states, mem0), keys, unroll=chunk_steps
    )
    return final_states, final_mem0, trajectory


def collect_selfplay_rollout(env, states, mem0, mem1, pool, live_net, opp_net, key,
                              num_steps: int, build_castles_enabled: bool,
                              gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    """seat 0 = live_net (trainable), seat 1 = opp_net (frozen params of the
    same architecture — either live_net again for true self-play, or a
    sampled league checkpoint's params). `mem0`/`mem1` are this bucket's
    persistent per-seat MemoryState carried in by the caller (train.py) across
    iterations, exactly like `states` — pass fresh `init_memory()` batches the
    first time a bucket is created. Runs as a plain Python loop over small
    fully-unrolled chunks — see module docstring for why. `gamma` must match
    PPOConfig.gamma (the same discount used in GAE) for the potential-based
    shaping term in training/rewards.py to stay policy-invariant."""
    get_obs = _get_obs_fn(env)
    chunks = []
    for size in _chunk_sizes(num_steps, _CHUNK_STEPS):
        key, chunk_key = jrandom.split(key)
        states, mem0, mem1, traj = _selfplay_chunk(
            env, states, mem0, mem1, pool, live_net, opp_net, chunk_key,
            size, build_castles_enabled, gamma, contact_shaping_weight, general_safety_weight,
        )
        chunks.append(traj)

    trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
    obs0_final = jax.vmap(lambda s: get_obs(s, 0))(states)
    bootstrap_value = jax.vmap(lambda o, m: _value_forward(live_net, o, m))(obs0_final, mem0)
    return states, mem0, mem1, trajectory, bootstrap_value


def collect_vs_heuristic_rollout(env, states, mem0, pool, live_net, heuristic_agent, key,
                                  num_steps: int, build_castles_enabled: bool,
                                  gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    """seat 0 = live_net (trainable), seat 1 = a pure-JAX heuristic Agent
    (e.g. RandomAgent()/ExpanderAgent()/HunterAgent() — reused across calls
    so it hits the jit cache; a fresh instance each call would recompile).
    `mem0` is this bucket's persistent MemoryState, same convention as
    collect_selfplay_rollout. Runs as a plain Python loop over small
    fully-unrolled chunks — see module docstring for why. `gamma` must match
    PPOConfig.gamma (the same discount used in GAE) for the potential-based
    shaping term in training/rewards.py to stay policy-invariant."""
    get_obs = _get_obs_fn(env)
    chunks = []
    for size in _chunk_sizes(num_steps, _CHUNK_STEPS):
        key, chunk_key = jrandom.split(key)
        states, mem0, traj = _vs_heuristic_chunk(
            env, states, mem0, pool, live_net, heuristic_agent, chunk_key,
            size, build_castles_enabled, gamma, contact_shaping_weight, general_safety_weight,
        )
        chunks.append(traj)

    trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
    obs0_final = jax.vmap(lambda s: get_obs(s, 0))(states)
    bootstrap_value = jax.vmap(lambda o, m: _value_forward(live_net, o, m))(obs0_final, mem0)
    return states, mem0, trajectory, bootstrap_value


@jax.jit
def compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray,
                 bootstrap_value: jnp.ndarray, gamma: float, lam: float) -> jnp.ndarray:
    """rewards/values/dones: (num_steps, num_envs). bootstrap_value: (num_envs,)
    — a real value estimate of the state after the rollout window, not zero,
    so credit assignment doesn't get truncated at the window boundary for
    episodes that outlive a single 256-step window (routine at 1200 turns).
    This scan has no conv op in its body, so it isn't subject to the
    while_loop pathology described in the module docstring — no unroll needed."""
    values_ext = jnp.concatenate([values, bootstrap_value[None, :]], axis=0)

    def gae_step(last_adv, inputs):
        reward, value, next_value, done = inputs
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * lam * nonterminal * last_adv
        return advantage, advantage

    rev = lambda x: x[::-1]
    init_adv = jnp.zeros(rewards.shape[1])
    _, advantages_rev = jax.lax.scan(
        gae_step, init_adv,
        (rev(rewards), rev(values), rev(values_ext[1:]), rev(dones)),
    )
    return rev(advantages_rev)


def init_memory_batch(num_envs: int, H: int, W: int) -> MemoryState:
    """A fresh, batched MemoryState for `num_envs` envs -- what train.py hands
    to collect_*_rollout the first time a bucket is created (or resized)."""
    single = mem_mod.init_memory(H, W)
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), single)


# ---------------------------------------------------------------------------
# Transformer path (training/network_transformer.py + training/augment.py),
# per ~/.claude/plans/drifting-questing-babbage.md's Step 2. Added alongside
# the CNN/MemoryState functions above, not replacing them -- the CNN path
# stays runnable while the transformer path goes through the plan's remaining
# steps (PPO additions, numpy export, BC rework, Colab smoke test) before
# train.py is ever pointed at it.
#
# Unlike MemoryState (updated in a separate post-action mem_mod.update_memory
# call using the POST-action observation, since it needs to know which cells
# the JUST-TAKEN action touched), AugmentedObsState has no dependency on the
# action taken -- aug_mod.augment_obs(obs, obs_state) is called ONCE per step
# with the PRE-action observation, and its second return value (new_obs_state)
# is directly the correct carry for the next step. This mirrors AverageJoe's
# own rollout (train/rollout_selfplay.py: augment_fn(obs_arr, osp) computed
# from the current pre-action obs, no separate update pass).
#
# Measured (not guessed) memory cost, single-sample: AugmentedObsState is
# 27,929 bytes at TransformerNetworkConfig defaults (grid_size=21,
# history_size=5, temporal_window=128) vs MemoryState's 5,292 bytes --
# ~5.3x, not the ~7-8x the plan's Design Decisions section estimated before
# this was actually measured (that estimate stands corrected here). A
# StepDataTransformer entry (obs + obs_state + 6 scalars) costs ~33.3KB/
# step/env in float32, or ~20.4KB/step/env with obs_state's float leaves
# stored as bf16 (bool leaves -- castles/generals/mountains/seen/enemy_seen --
# stay bool; there's nothing to gain converting a 1-byte dtype). At the CNN
# path's current 512envs x 256steps (131,072 samples): ~4.36GB float32 vs
# ~2.68GB with bf16 storage -- store as bf16 (see _to_bf16_floats below),
# matching the plan's mitigation. This keeps the SAME 512x256 the CNN path
# already uses as the starting point for the first real Colab smoke test
# (Step 6) rather than adopting AverageJoe's own larger num_envs blindly;
# scaling up is a later tuning knob once actual GPU headroom is confirmed on
# Colab itself, not something to guess on this CPU-only machine.
#
# The scan CARRY (obs_state fed into the next iteration) is kept in full
# float32 precision -- only the copy written into the trajectory output
# (accumulated num_steps times, the actual memory-heavy quantity) is cast to
# bf16. Recomputing augment_obs from a bf16-truncated obs_state would
# compound rounding error into the recurrent army_stack/temporal-history
# deltas step over step; truncating only the stored-for-later-replay copy
# avoids that while still shrinking the buffer. training/ppo.py's minibatch
# loop (Step 3) will need to upcast StepDataTransformer.obs_state back to
# float32 before calling augment_obs/normalize_augmented -- JAX auto-promotes
# mixed bf16/float32 arithmetic, but an explicit upcast keeps the intent
# clear at the call site.
# ---------------------------------------------------------------------------


class StepDataTransformer(NamedTuple):
    obs: Observation           # pre-action observation for seat 0, leading axis (num_steps, num_envs)
    obs_state: AugmentedObsState  # obs_state AS OF that same pre-action observation (bf16 float leaves)
    action_index: jnp.ndarray
    logprob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    winner: jnp.ndarray        # info.winner at this step (0/1/-1); only meaningful where done=True


def _to_bf16_floats(pytree):
    """Casts only floating-point leaves to bf16; bool/int leaves pass through
    unchanged (AugmentedObsState's castles/generals/mountains/seen/enemy_seen
    are already bool -- nothing to shrink there)."""
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, pytree
    )


def _policy_forward_transformer(net: TransformerPolicyValueNetwork, obs: Observation,
                                 obs_state: AugmentedObsState, key: jnp.ndarray,
                                 build_castles_enabled: bool):
    augmented, new_obs_state = aug_mod.augment_obs(obs, obs_state)
    normed = aug_mod.normalize_augmented(augmented)
    temporal = aug_mod.temporal_data(new_obs_state)
    logits, value = net(normed, temporal)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    idx = asp.sample_action_index(key, logits, mask)
    logprob, _entropy = asp.logprob_and_entropy(logits, mask, idx)
    H, W = obs.armies.shape
    engine_action = asp.decode_action_index(idx, H, W)
    return idx, logprob, value, engine_action, new_obs_state


def _value_forward_transformer(net: TransformerPolicyValueNetwork, obs: Observation,
                                obs_state: AugmentedObsState) -> jnp.ndarray:
    augmented, new_obs_state = aug_mod.augment_obs(obs, obs_state)
    normed = aug_mod.normalize_augmented(augmented)
    temporal = aug_mod.temporal_data(new_obs_state)
    _, value = net(normed, temporal)
    return value


@partial(jax.jit, static_argnames=("env", "chunk_steps", "build_castles_enabled"))
def _selfplay_chunk_transformer(env, states, obs_state0, obs_state1, pool, live_net, opp_net, key,
                                 chunk_steps: int, build_castles_enabled: bool,
                                 gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    get_obs = _get_obs_fn(env)

    def step_fn(carry, key_t):
        states, obs_state0, obs_state1 = carry
        num_envs = states.armies.shape[0]
        key_a, key_b = jrandom.split(key_t)
        keys_a = jrandom.split(key_a, num_envs)
        keys_b = jrandom.split(key_b, num_envs)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        idx0, logprob0, value0, action0, obs_state0_next = jax.vmap(
            lambda o, s, k: _policy_forward_transformer(live_net, o, s, k, build_castles_enabled)
        )(obs0, obs_state0, keys_a)
        _idx1, _lp1, _v1, action1, obs_state1_next = jax.vmap(
            lambda o, s, k: _policy_forward_transformer(opp_net, o, s, k, build_castles_enabled)
        )(obs1, obs_state1, keys_b)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        reward0 = jax.vmap(
            lambda po, a, o: shaped_reward_fn(po, a, o, gamma, contact_shaping_weight, general_safety_weight)
        )(obs0, action0, obs0_after)
        done0 = timesteps.terminated | timesteps.truncated
        winner0 = timesteps.info.winner

        obs_state0_next = aug_mod.reset_obs_state_where(obs_state0_next, done0)
        obs_state1_next = aug_mod.reset_obs_state_where(obs_state1_next, done0)

        step_data = StepDataTransformer(
            obs0, _to_bf16_floats(obs_state0), idx0, logprob0, value0, reward0, done0, winner0
        )
        return (new_states, obs_state0_next, obs_state1_next), step_data

    keys = jrandom.split(key, chunk_steps)
    (final_states, final_obs_state0, final_obs_state1), trajectory = jax.lax.scan(
        step_fn, (states, obs_state0, obs_state1), keys, unroll=chunk_steps
    )
    return final_states, final_obs_state0, final_obs_state1, trajectory


@partial(jax.jit, static_argnames=("env", "heuristic_agent", "chunk_steps", "build_castles_enabled"))
def _vs_heuristic_chunk_transformer(env, states, obs_state0, pool, live_net, heuristic_agent, key,
                                     chunk_steps: int, build_castles_enabled: bool,
                                     gamma: float, contact_shaping_weight: float, general_safety_weight: float):
    get_obs = _get_obs_fn(env)

    def step_fn(carry, key_t):
        states, obs_state0 = carry
        num_envs = states.armies.shape[0]
        key_a, key_b = jrandom.split(key_t)
        keys_a = jrandom.split(key_a, num_envs)
        keys_b = jrandom.split(key_b, num_envs)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        idx0, logprob0, value0, action0, obs_state0_next = jax.vmap(
            lambda o, s, k: _policy_forward_transformer(live_net, o, s, k, build_castles_enabled)
        )(obs0, obs_state0, keys_a)
        action1 = jax.vmap(heuristic_agent.act)(obs1, keys_b)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        reward0 = jax.vmap(
            lambda po, a, o: shaped_reward_fn(po, a, o, gamma, contact_shaping_weight, general_safety_weight)
        )(obs0, action0, obs0_after)
        done0 = timesteps.terminated | timesteps.truncated
        winner0 = timesteps.info.winner

        obs_state0_next = aug_mod.reset_obs_state_where(obs_state0_next, done0)

        step_data = StepDataTransformer(
            obs0, _to_bf16_floats(obs_state0), idx0, logprob0, value0, reward0, done0, winner0
        )
        return (new_states, obs_state0_next), step_data

    keys = jrandom.split(key, chunk_steps)
    (final_states, final_obs_state0), trajectory = jax.lax.scan(
        step_fn, (states, obs_state0), keys, unroll=chunk_steps
    )
    return final_states, final_obs_state0, trajectory


def collect_selfplay_rollout_transformer(env, states, obs_state0, obs_state1, pool, live_net, opp_net, key,
                                          num_steps: int, build_castles_enabled: bool,
                                          gamma: float, contact_shaping_weight: float,
                                          general_safety_weight: float):
    """Transformer-path twin of collect_selfplay_rollout -- see the module
    section docstring above. `obs_state0`/`obs_state1` are this bucket's
    persistent per-seat AugmentedObsState carried in by the caller across
    iterations, exactly like `mem0`/`mem1`; pass fresh init_obs_state_batch()
    the first time a bucket is created."""
    get_obs = _get_obs_fn(env)
    chunks = []
    for size in _chunk_sizes(num_steps, _CHUNK_STEPS):
        key, chunk_key = jrandom.split(key)
        states, obs_state0, obs_state1, traj = _selfplay_chunk_transformer(
            env, states, obs_state0, obs_state1, pool, live_net, opp_net, chunk_key,
            size, build_castles_enabled, gamma, contact_shaping_weight, general_safety_weight,
        )
        chunks.append(traj)

    trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
    obs0_final = jax.vmap(lambda s: get_obs(s, 0))(states)
    bootstrap_value = jax.vmap(
        lambda o, s: _value_forward_transformer(live_net, o, s)
    )(obs0_final, obs_state0)
    return states, obs_state0, obs_state1, trajectory, bootstrap_value


def collect_vs_heuristic_rollout_transformer(env, states, obs_state0, pool, live_net, heuristic_agent, key,
                                              num_steps: int, build_castles_enabled: bool,
                                              gamma: float, contact_shaping_weight: float,
                                              general_safety_weight: float):
    """Transformer-path twin of collect_vs_heuristic_rollout -- see the
    module section docstring above."""
    get_obs = _get_obs_fn(env)
    chunks = []
    for size in _chunk_sizes(num_steps, _CHUNK_STEPS):
        key, chunk_key = jrandom.split(key)
        states, obs_state0, traj = _vs_heuristic_chunk_transformer(
            env, states, obs_state0, pool, live_net, heuristic_agent, chunk_key,
            size, build_castles_enabled, gamma, contact_shaping_weight, general_safety_weight,
        )
        chunks.append(traj)

    trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)
    obs0_final = jax.vmap(lambda s: get_obs(s, 0))(states)
    bootstrap_value = jax.vmap(
        lambda o, s: _value_forward_transformer(live_net, o, s)
    )(obs0_final, obs_state0)
    return states, obs_state0, trajectory, bootstrap_value


def init_obs_state_batch(num_envs: int, cfg: TransformerNetworkConfig = TransformerNetworkConfig()) -> AugmentedObsState:
    """A fresh, batched AugmentedObsState for `num_envs` envs -- what
    train.py hands to collect_*_rollout_transformer the first time a bucket
    is created (or resized). Mirrors init_memory_batch."""
    single = aug_mod.init_obs_state(cfg)
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), single)
