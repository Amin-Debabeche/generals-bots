"""Behavior-cloning on real high-rated generals.io replays (see
training/human_replays.py for how the dataset -- strakammm/generals_io_replays
on Hugging Face -- gets converted into (Observation, MemoryState,
action_index) pairs).

Reuses training/pretrain_bc.py's loss function and train_step (identical
objective: masked cross-entropy against the demonstrated action) -- only the
data source differs (a fixed .npz of real human moves here, vs. live
Hunter-vs-Hunter self-play there).

`--init-from` optionally continues training from an existing checkpoint
(e.g. the Hunter-BC network) instead of a random init, for a sequential
curriculum: Hunter first for basic "don't just sit there" instincts, then
human replays for more sophisticated strategy on top.

Usage:
    python -m training.pretrain_bc_human --data training/runs/human_replays_stars80.npz \\
        --out training/runs/bc_human_warmstart.eqx [--init-from training/runs/bc_hunter_warmstart.eqx]
"""
import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from generals.core.observation import Observation
from training import action_space as asp
from training import augment as aug_mod
from training.augment import AugmentedObsState
from training.config import NetworkConfig, TransformerNetworkConfig
from training.memory import MemoryState
from training.network import PolicyValueNetwork
from training.network_transformer import TransformerPolicyValueNetwork
from training.pretrain_bc import make_bc_train_step, make_bc_train_step_transformer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help=".npz produced by training/human_replays.py "
                   "or training/leaderboard_replays.py")
    p.add_argument("--network", choices=("cnn", "transformer"), default="cnn",
                    help="must match the --network the .npz was generated with (mem_* vs obs_state_* "
                         "fields); see ~/.claude/plans/drifting-questing-babbage.md")
    p.add_argument("--out", required=True)
    p.add_argument("--init-from", type=str, default=None,
                    help="optional .eqx network-only checkpoint to continue training from")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=5e-4,
                    help="lower than pretrain_bc.py's 1e-3 by default when --init-from is set, "
                         "to refine rather than overwrite an existing warm start")
    p.add_argument("--val-fraction", type=float, default=0.05,
                    help="held-out fraction for a val accuracy check each epoch (BC-specific "
                         "overfitting risk: human replays repeat a much smaller pool of usernames "
                         "than Hunter-vs-Hunter's procedurally-endless self-play)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_dataset(path: str, network: str = "cnn"):
    npz = np.load(path)
    obs = Observation(**{f: jnp.asarray(npz[f]) for f in Observation._fields})
    action_index = jnp.asarray(npz["action_index"])
    if network == "transformer":
        # Stored as float16 (training/human_replays.py's/leaderboard_replays.py's
        # dataset-size mitigation) -- upcast to float32 here, matching every
        # other float32 array this file works with.
        fields = {}
        for f in AugmentedObsState._fields:
            arr = npz[f"obs_state_{f}"]
            fields[f] = jnp.asarray(arr.astype(np.float32) if arr.dtype != np.bool_ else arr)
        state = AugmentedObsState(**fields)
    else:
        state = MemoryState(**{f: jnp.asarray(npz[f"mem_{f}"]) for f in MemoryState._fields})
    return obs, state, action_index


def _tree_index(x, idx: jnp.ndarray):
    return jax.tree.map(lambda a: a[idx], x)


@eqx.filter_jit
def _eval_accuracy(net, obs: Observation, mem: MemoryState, target_idx: jnp.ndarray) -> jnp.ndarray:
    def predict(obs_i, mem_i):
        tensor = asp.obs_to_network_input_with_memory(obs_i, mem_i)
        logits, _ = net(tensor)
        mask = asp.compute_legal_action_mask(obs_i, False)
        return asp.greedy_action_index(logits, mask)

    preds = jax.vmap(predict)(obs, mem)
    return jnp.mean((preds == target_idx).astype(jnp.float32))


@eqx.filter_jit
def _eval_accuracy_transformer(net, obs: Observation, obs_state: AugmentedObsState,
                                target_idx: jnp.ndarray) -> jnp.ndarray:
    def predict(obs_i, state_i):
        augmented, new_state = aug_mod.augment_obs(obs_i, state_i)
        normed = aug_mod.normalize_augmented(augmented)
        temporal = aug_mod.temporal_data(new_state)
        logits, _ = net(normed, temporal)
        mask = asp.compute_legal_action_mask(obs_i, False)
        return asp.greedy_action_index(logits, mask)

    preds = jax.vmap(predict)(obs, obs_state)
    return jnp.mean((preds == target_idx).astype(jnp.float32))


def main():
    args = parse_args()
    key = jrandom.PRNGKey(args.seed)

    print(f"Loading {args.data} ({args.network}) ...")
    obs, state, action_index = load_dataset(args.data, args.network)
    num_samples = action_index.shape[0]
    print(f"  {num_samples} (observation, state, human-action) pairs")

    key, perm_key = jrandom.split(key)
    perm = jrandom.permutation(perm_key, num_samples)
    num_val = int(num_samples * args.val_fraction)
    val_idx, train_idx = perm[:num_val], perm[num_val:]

    train_obs = _tree_index(obs, train_idx)
    train_state = _tree_index(state, train_idx)
    train_actions = action_index[train_idx]
    val_obs = _tree_index(obs, val_idx)
    val_state = _tree_index(state, val_idx)
    val_actions = action_index[val_idx]
    print(f"  train: {train_actions.shape[0]}  val: {val_actions.shape[0]}")

    net_key, key = jrandom.split(key)
    if args.network == "transformer":
        net_cfg = TransformerNetworkConfig()
        net = TransformerPolicyValueNetwork(net_key, net_cfg)
        if args.init_from:
            template = TransformerPolicyValueNetwork(jrandom.PRNGKey(0), net_cfg)
            net = eqx.tree_deserialise_leaves(args.init_from, template)
            print(f"  continuing from {args.init_from}")
        optimizer = optax.adam(args.learning_rate)
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
        train_step = make_bc_train_step_transformer(optimizer, False)
        eval_fn = _eval_accuracy_transformer
    else:
        net_cfg = NetworkConfig()
        net = PolicyValueNetwork(net_key, net_cfg)
        if args.init_from:
            template = PolicyValueNetwork(jrandom.PRNGKey(0), net_cfg)
            net = eqx.tree_deserialise_leaves(args.init_from, template)
            print(f"  continuing from {args.init_from}")
        optimizer = optax.adam(args.learning_rate)
        opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
        train_step = make_bc_train_step(optimizer)
        eval_fn = _eval_accuracy

    print(f"\nTraining {args.epochs} epochs, minibatch={args.minibatch_size}, lr={args.learning_rate}...")
    num_train = train_actions.shape[0]
    for epoch in range(args.epochs):
        key, perm_key = jrandom.split(key)
        p = jrandom.permutation(perm_key, num_train)
        obs_shuf = _tree_index(train_obs, p)
        state_shuf = _tree_index(train_state, p)
        idx_shuf = train_actions[p]

        num_batches = num_train // args.minibatch_size
        epoch_loss, epoch_acc = 0.0, 0.0
        t0 = time.time()
        for i in range(num_batches):
            sl = slice(i * args.minibatch_size, (i + 1) * args.minibatch_size)
            batch_obs = _tree_index(obs_shuf, sl)
            batch_state = _tree_index(state_shuf, sl)
            batch_idx = idx_shuf[sl]
            net, opt_state, loss, acc = train_step(net, opt_state, batch_obs, batch_state, batch_idx)
            epoch_loss += float(loss)
            epoch_acc += float(acc)
        epoch_loss /= max(num_batches, 1)
        epoch_acc /= max(num_batches, 1)

        val_acc = float(eval_fn(net, val_obs, val_state, val_actions)) if num_val > 0 else float("nan")
        print(f"epoch {epoch}: loss={epoch_loss:.4f} train_acc={epoch_acc:.3f} "
              f"val_acc={val_acc:.3f} ({time.time() - t0:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(out_path), net)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
