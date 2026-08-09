"""Fast vectorized win-rate vs the built-in heuristic baselines
(RandomAgent/ExpanderAgent/HunterAgent), run under the CURRENT curriculum
stage's exact ruleset — not always full competition, so an early-stage policy
that hasn't learned castles/fog yet is judged against a comparable opponent
rather than penalized for not-yet-relevant skills. The trained policy always
plays greedily (argmax over legal logits), matching how it will actually be
deployed, not the stochastic sampling used during PPO rollout collection.

Each of `num_games` parallel envs plays exactly one game; outcome is recorded
the FIRST time a slot finishes (env.step's own pool-based auto-reset would
otherwise silently start a second game in that slot before the fixed-length
scan window ends).

Runs in small fully-unrolled chunks via a plain Python loop, exactly like
training/rollout.py — see that module's docstring for why (a lax.scan whose
body contains a conv op is ~30-50x slower per step unless fully unrolled, and
env.truncation can be up to 1200 here, far past what's practical to fully
unroll in one piece). Chunking also gets early-exit for free: most games
finish well before env.truncation, and this checks done_mask between chunks
so a finished batch doesn't keep paying for further (already fully unrolled
and now wasted) chunks.

Memory (training/memory.py) is tracked per-seat, per-episode, same convention
as training/rollout.py -- but since every game here runs to a single
conclusion (no auto-reset mid-batch), it's simply initialized once per batch
rather than threaded back out to a caller-owned bucket cache.
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.core import game

from training import action_space as asp
from training import memory as mem_mod
from training.network import PolicyValueNetwork
from training.rollout import init_memory_batch

BASELINE_AGENTS = {
    "vs_random": RandomAgent(),
    "vs_expander": ExpanderAgent(),
    "vs_hunter": HunterAgent(),
}

_CHUNK_STEPS = 8  # matches training/rollout.py's _CHUNK_STEPS


def _greedy_forward(net: PolicyValueNetwork, obs, mem, build_castles_enabled: bool):
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, _value = net(tensor)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    idx = asp.greedy_action_index(logits, mask)
    H, W = obs.armies.shape
    return asp.decode_action_index(idx, H, W)


def _chunk_sizes(total_steps: int, chunk_steps: int) -> list[int]:
    full, remainder = divmod(total_steps, chunk_steps)
    sizes = [chunk_steps] * full
    if remainder:
        sizes.append(remainder)
    return sizes


@partial(jax.jit, static_argnames=("env", "agent", "chunk_steps", "build_castles_enabled"))
def _play_chunk(env, agent, net, pool, carry, key, chunk_steps: int, build_castles_enabled: bool):
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    states, mem0, done_mask, first_winner = carry

    def step_fn(carry, key_t):
        states, mem0, done_mask, first_winner = carry
        num_envs = states.armies.shape[0]
        keys_b = jrandom.split(key_t, num_envs)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        action0 = jax.vmap(lambda o, m: _greedy_forward(net, o, m, build_castles_enabled))(obs0, mem0)
        action1 = jax.vmap(agent.act)(obs1, keys_b)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        just_finished = (timesteps.terminated | timesteps.truncated) & ~done_mask
        first_winner = jnp.where(just_finished, timesteps.info.winner, first_winner)
        done_mask = done_mask | timesteps.terminated | timesteps.truncated

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        mem0_next = jax.vmap(mem_mod.update_memory)(mem0, obs0_after, action0)

        return (new_states, mem0_next, done_mask, first_winner), None

    keys = jrandom.split(key, chunk_steps)
    (states, mem0, done_mask, first_winner), _ = jax.lax.scan(
        step_fn, (states, mem0, done_mask, first_winner), keys, unroll=chunk_steps
    )
    return states, mem0, done_mask, first_winner


@partial(jax.jit, static_argnames=("env", "chunk_steps", "build_castles_enabled"))
def _play_chunk_vs_network(env, net, frozen_net, pool, carry, key, chunk_steps: int,
                            build_castles_enabled: bool):
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    states, mem0, mem1, done_mask, first_winner = carry

    def step_fn(carry, key_t):
        states, mem0, mem1, done_mask, first_winner = carry

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        action0 = jax.vmap(lambda o, m: _greedy_forward(net, o, m, build_castles_enabled))(obs0, mem0)
        action1 = jax.vmap(lambda o, m: _greedy_forward(frozen_net, o, m, build_castles_enabled))(obs1, mem1)

        actions = jnp.stack([action0, action1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        just_finished = (timesteps.terminated | timesteps.truncated) & ~done_mask
        first_winner = jnp.where(just_finished, timesteps.info.winner, first_winner)
        done_mask = done_mask | timesteps.terminated | timesteps.truncated

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        obs1_after = jax.vmap(lambda s: get_obs(s, 1))(timesteps.last_state)
        mem0_next = jax.vmap(mem_mod.update_memory)(mem0, obs0_after, action0)
        mem1_next = jax.vmap(mem_mod.update_memory)(mem1, obs1_after, action1)

        return (new_states, mem0_next, mem1_next, done_mask, first_winner), None

    keys = jrandom.split(key, chunk_steps)
    (states, mem0, mem1, done_mask, first_winner), _ = jax.lax.scan(
        step_fn, (states, mem0, mem1, done_mask, first_winner), keys, unroll=chunk_steps
    )
    return states, mem0, mem1, done_mask, first_winner


def evaluate_vs_network(env, net: PolicyValueNetwork, frozen_net: PolicyValueNetwork, pool,
                         key: jnp.ndarray, num_games: int, build_castles_enabled: bool) -> dict:
    """Win-rate of `net` (seat 0, greedy) vs a frozen opponent network (seat 1,
    greedy) — e.g. Stage D's promotion metric vs the frozen Stage-C-final
    checkpoint."""
    init_keys = jrandom.split(key, num_games + 1)
    states = jax.vmap(env.init_state)(init_keys[:num_games])
    step_key = init_keys[num_games]
    mem0 = init_memory_batch(num_games, 21, 21)
    mem1 = init_memory_batch(num_games, 21, 21)
    done_mask = jnp.zeros(num_games, dtype=jnp.bool_)
    first_winner = jnp.full((num_games,), -1, dtype=jnp.int32)

    for size in _chunk_sizes(env.truncation, _CHUNK_STEPS):
        step_key, chunk_key = jrandom.split(step_key)
        states, mem0, mem1, done_mask, first_winner = _play_chunk_vs_network(
            env, net, frozen_net, pool, (states, mem0, mem1, done_mask, first_winner),
            chunk_key, size, build_castles_enabled,
        )
        if bool(jnp.all(done_mask)):
            break

    done_mask = jax.device_get(done_mask)
    first_winner = jax.device_get(first_winner)
    wins = int((first_winner == 0).sum())
    losses = int((first_winner == 1).sum())
    draws = int(((first_winner == -1) & done_mask).sum()) + int((~done_mask).sum())
    return {"winrate": wins / num_games if num_games else 0.0, "wins": wins,
            "losses": losses, "draws": draws, "n_games": num_games}


def evaluate_vs_baselines(env, net: PolicyValueNetwork, pool, key: jnp.ndarray,
                           num_games: int, build_castles_enabled: bool) -> dict:
    """Returns {"vs_random": {"winrate", "wins", "losses", "draws", "n_games"}, ...}."""
    results = {}
    for name, agent in BASELINE_AGENTS.items():
        key, subkey = jrandom.split(key)
        init_keys = jrandom.split(subkey, num_games + 1)
        states = jax.vmap(env.init_state)(init_keys[:num_games])
        step_key = init_keys[num_games]
        mem0 = init_memory_batch(num_games, 21, 21)
        done_mask = jnp.zeros(num_games, dtype=jnp.bool_)
        first_winner = jnp.full((num_games,), -1, dtype=jnp.int32)

        for size in _chunk_sizes(env.truncation, _CHUNK_STEPS):
            step_key, chunk_key = jrandom.split(step_key)
            states, mem0, done_mask, first_winner = _play_chunk(
                env, agent, net, pool, (states, mem0, done_mask, first_winner),
                chunk_key, size, build_castles_enabled,
            )
            if bool(jnp.all(done_mask)):
                break

        done_mask = jax.device_get(done_mask)
        first_winner = jax.device_get(first_winner)

        wins = int((first_winner == 0).sum())
        losses = int((first_winner == 1).sum())
        draws = int(((first_winner == -1) & done_mask).sum())
        undecided = int((~done_mask).sum())  # should be 0; env.truncation forces a decision

        results[name] = {
            "winrate": wins / num_games if num_games else 0.0,
            "wins": wins,
            "losses": losses,
            "draws": draws + undecided,
            "n_games": num_games,
        }
    return results
