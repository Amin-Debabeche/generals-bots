"""Behavior-cloning warm start: imitate HunterAgent before ever running PPO.

Added after a full RL run (self-play + baseline mix) collapsed onto a
degenerate "shuffle the army next to my own general forever, never expand"
equilibrium — confirmed by loading the checkpoint and watching a real game:
land stuck at 5 tiles from turn 3 to turn 800 while army piled up to ~1000,
99.4% of actions being the same back-and-forth move. Zero wins across 584
iterations at the real board scale despite Hunter racking up real kills
against it — the policy never tasted the payoff of expanding, so pure
self-play/PPO from random init never discovered it here. Hunter's hand-coded
"garrison the general, send the surplus out, hunt the enemy general" strategy
is a good target to imitate directly, breaking the "doing nothing is safe"
trap before PPO ever starts, rather than hoping PPO stumbles into it.

This is imitation-only (supervised cross-entropy against Hunter's chosen
action, using the SAME legal-action masking as RL so the loss matches what
the policy will actually be evaluated on). The value head is untouched here
(cheap for PPO to recalibrate in the first few real iterations) -- only the
backbone + policy head get a meaningful gradient, since Hunter has no notion
of "value estimate" to imitate.

Records memory-augmented observations (training/memory.py) for both seats
even though Hunter itself never looks at memory to decide anything -- the
recorded (obs, mem, action) pairs are training data for OUR network, which
does use memory, so the input representation must match what it'll see
during real rollout collection.

Usage:
    python -m training.pretrain_bc --out training/runs/bc_hunter.eqx
"""
import argparse
import time
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.agents import HunterAgent
from generals.core import game

from training import action_space as asp
from training import curriculum as cur
from training import memory as mem_mod
from training.config import NetworkConfig
from training.network import PolicyValueNetwork
from training.rollout import _reset_memory_where_done, init_memory_batch

TRAINING_DIR = Path(__file__).resolve().parent
_HUNTER = HunterAgent()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output path for the pretrained network (.eqx)")
    p.add_argument("--num-envs", type=int, default=128, help="parallel Hunter-vs-Hunter games for data collection")
    p.add_argument("--num-steps", type=int, default=384, help="steps per game (with pool auto-reset, so this covers many episodes)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@partial(jax.jit, static_argnames=("env", "num_envs", "num_steps"))
def collect_hunter_demonstrations(env, pool, num_envs: int, num_steps: int, key: jnp.ndarray):
    """Hunter vs Hunter self-play; records BOTH seats' (observation, memory,
    action) pairs -- Hunter's logic is a pure function of its own observation
    regardless of which seat it plays, so this doubles the dataset for free
    and covers both sides of every matchup. No conv op anywhere in this path
    (Hunter has no network), so a plain lax.scan is fine here -- this isn't
    subject to the while_loop+conv pathology documented in rollout.py."""
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    G = NetworkConfig().grid_size

    init_keys = jrandom.split(key, num_envs + 1)
    states = jax.vmap(env.init_state)(init_keys[:num_envs])
    step_key = init_keys[num_envs]
    mem0 = init_memory_batch(num_envs, G, G)
    mem1 = init_memory_batch(num_envs, G, G)

    def step_fn(carry, key_t):
        states, mem0, mem1 = carry
        n = states.armies.shape[0]
        k0, k1 = jrandom.split(key_t)
        keys0 = jrandom.split(k0, n)
        keys1 = jrandom.split(k1, n)

        obs0 = jax.vmap(lambda s: get_obs(s, 0))(states)
        obs1 = jax.vmap(lambda s: get_obs(s, 1))(states)
        a0 = jax.vmap(_HUNTER.act)(obs0, keys0)
        a1 = jax.vmap(_HUNTER.act)(obs1, keys1)

        actions = jnp.stack([a0, a1], axis=1)
        timesteps, new_states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

        obs0_after = jax.vmap(lambda s: get_obs(s, 0))(timesteps.last_state)
        obs1_after = jax.vmap(lambda s: get_obs(s, 1))(timesteps.last_state)
        done = timesteps.terminated | timesteps.truncated
        mem0_next = _reset_memory_where_done(jax.vmap(mem_mod.update_memory)(mem0, obs0_after, a0), done)
        mem1_next = _reset_memory_where_done(jax.vmap(mem_mod.update_memory)(mem1, obs1_after, a1), done)

        return (new_states, mem0_next, mem1_next), (obs0, mem0, a0, obs1, mem1, a1)

    keys = jrandom.split(step_key, num_steps)
    _final, (obs0, mem0_traj, a0, obs1, mem1_traj, a1) = jax.lax.scan(step_fn, (states, mem0, mem1), keys)
    return obs0, mem0_traj, a0, obs1, mem1_traj, a1


def _flatten(x):
    return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), x)


def _bc_loss_one(net: PolicyValueNetwork, obs, mem, target_idx: jnp.ndarray) -> jnp.ndarray:
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, _value = net(tensor)
    # build_castles_enabled=False: Hunter never builds, and Stage C (used for
    # data collection) has castles off -- matches the demonstration data.
    mask = asp.compute_legal_action_mask(obs, False)
    log_probs = asp.masked_log_softmax(logits, mask)
    return -log_probs[target_idx]


def make_bc_train_step(optimizer):
    def loss_and_acc(net, obs, mem, target_idx):
        losses = jax.vmap(lambda o, m, i: _bc_loss_one(net, o, m, i))(obs, mem, target_idx)

        def predict(o, m):
            tensor = asp.obs_to_network_input_with_memory(o, m)
            logits, _ = net(tensor)
            mask = asp.compute_legal_action_mask(o, False)
            return asp.greedy_action_index(logits, mask)

        preds = jax.vmap(predict)(obs, mem)
        acc = jnp.mean((preds == target_idx).astype(jnp.float32))
        return jnp.mean(losses), acc

    @eqx.filter_jit
    def train_step(net, opt_state, obs, mem, target_idx):
        (loss, acc), grads = eqx.filter_value_and_grad(loss_and_acc, has_aux=True)(net, obs, mem, target_idx)
        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, acc

    return train_step


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)

    stage = cur.STAGE_C  # real 18-21 board, fog on, no castles -- the level where the collapse was diagnosed
    env = cur.build_env(stage)
    key, pool_key, collect_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)

    print(f"Collecting Hunter-vs-Hunter demonstrations: {args.num_envs} envs x {args.num_steps} steps "
          f"on {stage.name}'s ruleset...")
    t0 = time.time()
    obs0, mem0, a0, obs1, mem1, a1 = collect_hunter_demonstrations(
        env, pool, args.num_envs, args.num_steps, collect_key
    )
    jax.block_until_ready(a0)
    print(f"  collected in {time.time() - t0:.1f}s")

    H, W = obs0.armies.shape[-2], obs0.armies.shape[-1]
    all_obs = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), _flatten(obs0), _flatten(obs1))
    all_mem = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), _flatten(mem0), _flatten(mem1))
    a0_flat = a0.reshape(-1, 5)
    a1_flat = a1.reshape(-1, 5)
    all_actions = jnp.concatenate([a0_flat, a1_flat], axis=0)
    all_idx = jax.vmap(lambda a: asp.encode_action_index(a, H, W))(all_actions)

    num_samples = all_idx.shape[0]
    print(f"dataset: {num_samples} (observation, memory, Hunter-action) pairs")
    # Sanity: how much of the data is "pass" -- if Hunter itself passes constantly
    # on this board (it shouldn't, by design), BC would just learn to imitate that.
    pass_idx = H * W * asp.NUM_CELL_ACTIONS
    pass_frac = float(jnp.mean((all_idx == pass_idx).astype(jnp.float32)))
    print(f"  fraction of Hunter's actions that are PASS: {pass_frac:.3f} (sanity check -- should be low)")

    net_key, key = jrandom.split(key)
    net = PolicyValueNetwork(net_key, NetworkConfig())
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    train_step = make_bc_train_step(optimizer)

    print(f"\nTraining {args.epochs} epochs, minibatch={args.minibatch_size}...")
    for epoch in range(args.epochs):
        key, perm_key = jrandom.split(key)
        perm = jrandom.permutation(perm_key, num_samples)
        obs_shuf = jax.tree.map(lambda x: x[perm], all_obs)
        mem_shuf = jax.tree.map(lambda x: x[perm], all_mem)
        idx_shuf = all_idx[perm]

        num_batches = num_samples // args.minibatch_size
        epoch_loss, epoch_acc = 0.0, 0.0
        t0 = time.time()
        for i in range(num_batches):
            sl = slice(i * args.minibatch_size, (i + 1) * args.minibatch_size)
            batch_obs = jax.tree.map(lambda x: x[sl], obs_shuf)
            batch_mem = jax.tree.map(lambda x: x[sl], mem_shuf)
            batch_idx = idx_shuf[sl]
            net, opt_state, loss, acc = train_step(net, opt_state, batch_obs, batch_mem, batch_idx)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        print(f"epoch {epoch}: loss={epoch_loss:.4f} argmax-matches-Hunter accuracy={epoch_acc:.3f} "
              f"({time.time() - t0:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(out_path), net)
    print(f"\nSaved BC-pretrained network to {out_path}")


if __name__ == "__main__":
    main()
