"""Refine an existing (already RL-trained) checkpoint toward Hunter's
specific defensive behavior -- "garrison the general, send only the surplus
out, hunt the enemy general the moment it's reachable" -- via behavior
cloning against Hunter's actual chosen actions.

Why this exists, not training/pretrain_bc.py: that script's Hunter-vs-Hunter
demonstrations were collected once, from a random network, before any RL
happened, on Stage C's small-army ruleset. Direct behavioral inspection of
main9 (RL-trained through Stage D, general-safety reward shaping active)
showed it STILL loses to Hunter by getting its general one-shot captured
despite leading 2-5x on economy -- tracked one loss move-by-move: the
general's own army grows at almost exactly the passive structure-growth
rate (no active reinforcement at all), sitting at ~10 army against a
total empire of 131, both under the reward shaping's own target AND nowhere
near enough to survive a real attack. The reward-shaping pull just isn't
strong enough to overcome the competing expansion pressure. Rather than
keep escalating an abstract weight, imitate Hunter's actual garrison
behavior directly -- proven to work in this project twice already (the
original Hunter-BC broke a "do nothing" collapse; the leaderboard-replay
refinement measurably improved main9 over main8).

Key difference from pretrain_bc.py's data collection: demonstrations come
from Hunter playing against the CURRENT CHECKPOINT (not Hunter-vs-Hunter),
recording only Hunter's own (observation, action) pairs, so the visited
states -- especially large-army lategame states, exactly where main9's
garrison problem shows up -- match what the checkpoint actually
encounters, not Hunter-vs-Hunter's more modest economy. Hunter occupies
both seats across different env slots for symmetric state coverage. Uses
Stage D's ruleset (castles on) since that's where the failure was found,
though Hunter itself never builds regardless (compute_legal_action_mask
still keeps build_castles_enabled=False for the loss/mask, matching
pretrain_bc.py's convention -- Hunter's target is never a build action).

Usage:
    python -m training.pretrain_bc_hunter_refine \\
        --init-from training/runs/main9_ema_iter443.eqx \\
        --out training/runs/bc_hunter_refined.eqx
"""
import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.agents import HunterAgent
from generals.core import game

from training import action_space as asp
from training import augment as aug_mod
from training import curriculum as cur
from training import memory as mem_mod
from training import rollout as ro
from training.config import NetworkConfig, TransformerNetworkConfig
from training.network import PolicyValueNetwork
from training.network_transformer import TransformerPolicyValueNetwork
from training.pretrain_bc import make_bc_train_step, make_bc_train_step_transformer, run_bc_epochs, _flatten
from training.rollout import _reset_memory_where_done, init_memory_batch

_HUNTER = HunterAgent()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", required=True, help="checkpoint whose behavior main9 (etc.) needs refined")
    p.add_argument("--network", choices=("cnn", "transformer"), default="cnn",
                    help="must match --init-from's architecture; see "
                         "~/.claude/plans/drifting-questing-babbage.md")
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=128,
                    help="split evenly: half Hunter-as-seat0, half Hunter-as-seat1")
    p.add_argument("--num-steps", type=int, default=384)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=2e-4,
                    help="low -- refining an RL-trained checkpoint, not training from scratch")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _policy_action(net, obs, mem, key, build_castles_enabled):
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, _value = net(tensor)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    idx = asp.sample_action_index(key, logits, mask)
    H, W = obs.armies.shape
    return asp.decode_action_index(idx, H, W)


def collect_hunter_vs_policy_demonstrations(env, pool, net, num_envs: int, num_steps: int,
                                             build_castles_enabled: bool, key: jnp.ndarray):
    """First half of envs: Hunter is seat 0, `net` is seat 1. Second half:
    reversed. Only records Hunter's (observation, memory, action) each step,
    from whichever seat it's in -- `net` plays realistically but is never
    the demonstration target here."""
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    G = NetworkConfig().grid_size
    half = num_envs // 2

    init_keys = jrandom.split(key, num_envs + 1)
    states = jax.vmap(env.init_state)(init_keys[:num_envs])
    step_key = init_keys[num_envs]
    mem0 = init_memory_batch(num_envs, G, G)
    mem1 = init_memory_batch(num_envs, G, G)
    hunter_seat = jnp.concatenate([jnp.zeros(half, dtype=jnp.int32),
                                    jnp.ones(num_envs - half, dtype=jnp.int32)])

    def step_fn(carry, key_t):
        states, mem0, mem1 = carry
        n = states.armies.shape[0]
        k0, k1, kp = jrandom.split(key_t, 3)
        keys0 = jrandom.split(k0, n)
        keys1 = jrandom.split(k1, n)
        keysp = jrandom.split(kp, n)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        hunter_a0 = jax.vmap(_HUNTER.act)(obs0, keys0)
        hunter_a1 = jax.vmap(_HUNTER.act)(obs1, keys1)
        policy_a0 = jax.vmap(lambda o, m, k: _policy_action(net, o, m, k, build_castles_enabled))(obs0, mem0, keysp)
        policy_a1 = jax.vmap(lambda o, m, k: _policy_action(net, o, m, k, build_castles_enabled))(obs1, mem1, keysp)
        a0 = jnp.where((hunter_seat == 0)[:, None], hunter_a0, policy_a0)
        a1 = jnp.where((hunter_seat == 1)[:, None], hunter_a1, policy_a1)

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        obs1_after = jax.vmap(lambda s: get_obs(s, 1))(timesteps.last_state)
        done = timesteps.terminated | timesteps.truncated
        mem0_next = _reset_memory_where_done(jax.vmap(mem_mod.update_memory)(mem0, obs0_after, a0), done)
        mem1_next = _reset_memory_where_done(jax.vmap(mem_mod.update_memory)(mem1, obs1_after, a1), done)

        # Hunter's own (obs, mem, action) from whichever seat it's in this env.
        hunter_obs = jax.tree.map(lambda x, y: jnp.where((hunter_seat == 0).reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
                                    obs0, obs1)
        hunter_mem = jax.tree.map(lambda x, y: jnp.where((hunter_seat == 0).reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
                                    mem0, mem1)
        hunter_action = jnp.where((hunter_seat == 0)[:, None], a0, a1)

        return (new_states, mem0_next, mem1_next), (hunter_obs, hunter_mem, hunter_action)

    keys = jrandom.split(step_key, num_steps)
    _final, (h_obs, h_mem, h_action) = jax.lax.scan(step_fn, (states, mem0, mem1), keys)
    return h_obs, h_mem, h_action


def _policy_action_transformer(net, obs, obs_state, key, build_castles_enabled):
    augmented, new_state = aug_mod.augment_obs(obs, obs_state)
    normed = aug_mod.normalize_augmented(augmented)
    temporal = aug_mod.temporal_data(new_state)
    logits, _value = net(normed, temporal)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    idx = asp.sample_action_index(key, logits, mask)
    H, W = obs.armies.shape
    return asp.decode_action_index(idx, H, W)


def collect_hunter_vs_policy_demonstrations_transformer(env, pool, net, num_envs: int, num_steps: int,
                                                          build_castles_enabled: bool, key: jnp.ndarray,
                                                          cfg: TransformerNetworkConfig = TransformerNetworkConfig()):
    """Transformer-path twin of collect_hunter_vs_policy_demonstrations --
    same seat-split/where-select structure (both a "would-be Hunter" and a
    "would-be net" action computed for every env every step, then
    jnp.where-selected per the static hunter_seat split -- matches the CNN
    version's own vectorized-branch-then-select design, not a new pattern).
    AugmentedObsState for BOTH seats progresses every step regardless of
    which identity (Hunter or net) currently occupies that seat, via
    aug_mod.augment_obs called unconditionally on obs0/obs1 (see
    training/rollout.py's transformer-path section docstring -- no separate
    post-action update needed, unlike MemoryState)."""
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    half = num_envs // 2

    init_keys = jrandom.split(key, num_envs + 1)
    states = jax.vmap(env.init_state)(init_keys[:num_envs])
    step_key = init_keys[num_envs]
    obs_state0 = ro.init_obs_state_batch(num_envs, cfg)
    obs_state1 = ro.init_obs_state_batch(num_envs, cfg)
    hunter_seat = jnp.concatenate([jnp.zeros(half, dtype=jnp.int32),
                                    jnp.ones(num_envs - half, dtype=jnp.int32)])

    def step_fn(carry, key_t):
        states, obs_state0, obs_state1 = carry
        n = states.armies.shape[0]
        k0, k1, kp = jrandom.split(key_t, 3)
        keys0 = jrandom.split(k0, n)
        keys1 = jrandom.split(k1, n)
        keysp = jrandom.split(kp, n)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)

        _aug0, obs_state0_next = jax.vmap(aug_mod.augment_obs)(obs0, obs_state0)
        _aug1, obs_state1_next = jax.vmap(aug_mod.augment_obs)(obs1, obs_state1)

        hunter_a0 = jax.vmap(_HUNTER.act)(obs0, keys0)
        hunter_a1 = jax.vmap(_HUNTER.act)(obs1, keys1)
        policy_a0 = jax.vmap(
            lambda o, s, k: _policy_action_transformer(net, o, s, k, build_castles_enabled)
        )(obs0, obs_state0, keysp)
        policy_a1 = jax.vmap(
            lambda o, s, k: _policy_action_transformer(net, o, s, k, build_castles_enabled)
        )(obs1, obs_state1, keysp)
        a0 = jnp.where((hunter_seat == 0)[:, None], hunter_a0, policy_a0)
        a1 = jnp.where((hunter_seat == 1)[:, None], hunter_a1, policy_a1)

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)
        done = timesteps.terminated | timesteps.truncated

        obs_state0_next = aug_mod.reset_obs_state_where(obs_state0_next, done)
        obs_state1_next = aug_mod.reset_obs_state_where(obs_state1_next, done)

        hunter_obs = jax.tree.map(
            lambda x, y: jnp.where((hunter_seat == 0).reshape((-1,) + (1,) * (x.ndim - 1)), x, y), obs0, obs1
        )
        hunter_state = jax.tree.map(
            lambda x, y: jnp.where((hunter_seat == 0).reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
            obs_state0, obs_state1,
        )
        hunter_action = jnp.where((hunter_seat == 0)[:, None], a0, a1)

        return (new_states, obs_state0_next, obs_state1_next), (hunter_obs, hunter_state, hunter_action)

    keys = jrandom.split(step_key, num_steps)
    _final, (h_obs, h_state, h_action) = jax.lax.scan(step_fn, (states, obs_state0, obs_state1), keys)
    return h_obs, h_state, h_action


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)

    stage = cur.STAGE_D  # castles on -- the ruleset where the garrison failure was found
    env = cur.build_env(stage)
    key, pool_key, collect_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)

    net_key, key = jrandom.split(key)
    if args.network == "transformer":
        cfg = TransformerNetworkConfig()
        template = TransformerPolicyValueNetwork(net_key, cfg)
        net = eqx.tree_deserialise_leaves(args.init_from, template)
    else:
        template = PolicyValueNetwork(net_key, NetworkConfig())
        net = eqx.tree_deserialise_leaves(args.init_from, template)
    print(f"refining {args.init_from} ({args.network})")

    print(f"Collecting Hunter-vs-{Path(args.init_from).stem} demonstrations: "
          f"{args.num_envs} envs x {args.num_steps} steps on {stage.name}'s ruleset...")
    t0 = time.time()
    if args.network == "transformer":
        h_obs, h_state, h_action = collect_hunter_vs_policy_demonstrations_transformer(
            env, pool, net, args.num_envs, args.num_steps, stage.build_castles_enabled, collect_key, cfg
        )
    else:
        h_obs, h_state, h_action = collect_hunter_vs_policy_demonstrations(
            env, pool, net, args.num_envs, args.num_steps, stage.build_castles_enabled, collect_key
        )
    jax.block_until_ready(h_action)
    print(f"  collected in {time.time() - t0:.1f}s")

    H, W = h_obs.armies.shape[-2], h_obs.armies.shape[-1]
    all_obs = _flatten(h_obs)
    all_state = _flatten(h_state)
    all_actions = h_action.reshape(-1, 5)
    all_idx = jax.vmap(lambda a: asp.encode_action_index(a, H, W))(all_actions)

    num_samples = all_idx.shape[0]
    print(f"dataset: {num_samples} (observation, state, Hunter-action) pairs")
    pass_idx = H * W * asp.NUM_CELL_ACTIONS
    pass_frac = float(jnp.mean((all_idx == pass_idx).astype(jnp.float32)))
    print(f"  fraction of Hunter's actions that are PASS: {pass_frac:.3f}")

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    if args.network == "transformer":
        train_step = make_bc_train_step_transformer(optimizer, stage.build_castles_enabled)
    else:
        train_step = make_bc_train_step(optimizer)

    print(f"\nTraining {args.epochs} epochs, minibatch={args.minibatch_size}, lr={args.learning_rate}...")
    key, train_key = jrandom.split(key)
    net, opt_state = run_bc_epochs(
        net, opt_state, train_step, all_obs, all_state, all_idx,
        args.epochs, args.minibatch_size, train_key, label="Hunter-action pairs",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(out_path), net)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
