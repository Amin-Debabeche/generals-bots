"""PPO update: clipped surrogate loss + value loss + masked entropy bonus.

Legal-action masks and normalized network inputs are recomputed here from the
compact `Observation`+`MemoryState` stored by training/rollout.py's
trajectory buffer, never carried through it as pre-expanded tensors — see
rollout.py's module docstring for the memory-footprint rationale. Entropy/
KL/clip-frac are always computed from the legal-masked, renormalized
distribution (training/action_space.py's
`logprob_and_entropy`/`masked_log_softmax`), never raw logits — illegal-action
mass would otherwise dominate these diagnostics on a 3970-logit action space
where only a handful of entries are ever legal.
"""
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from generals.core.observation import Observation
from training import action_space as asp
from training import augment as aug_mod
from training import magnet as mg
from training.augment import AugmentedObsState
from training.memory import MemoryState
from training.network import PolicyValueNetwork
from training.network_transformer import TransformerPolicyValueNetwork


class FlatBatch(NamedTuple):
    obs: Observation
    mem: MemoryState
    action_index: jnp.ndarray
    old_logprob: jnp.ndarray
    advantage: jnp.ndarray
    ret: jnp.ndarray


def make_optimizer(learning_rate: float, max_grad_norm: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


def flatten_batch(trajectory, advantages: jnp.ndarray, returns: jnp.ndarray) -> FlatBatch:
    """trajectory: rollout.StepData with leading axes (num_steps, num_envs, ...).
    Concatenate multiple buckets' trajectories along axis 1 (env axis) before
    calling this, then it flattens (steps, envs) -> one leading sample axis."""
    def flat(x):
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    return FlatBatch(
        obs=jax.tree.map(flat, trajectory.obs),
        mem=jax.tree.map(flat, trajectory.mem),
        action_index=flat(trajectory.action_index),
        old_logprob=flat(trajectory.logprob),
        advantage=flat(advantages),
        ret=flat(returns),
    )


def _index_batch(batch, idx):
    """Generic pytree indexing -- works for FlatBatch (or any other
    NamedTuple-of-arrays batch shape, e.g. a future transformer-path
    equivalent) without hand-listing fields, since JAX's tree.map already
    recurses through nested NamedTuples down to the leaf arrays."""
    return jax.tree.map(lambda x: x[idx], batch)


def _forward_one(net: PolicyValueNetwork, obs: Observation, mem: MemoryState, action_index: jnp.ndarray,
                  build_castles_enabled: bool):
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, value = net(tensor)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    logprob, entropy = asp.logprob_and_entropy(logits, mask, action_index)

    # KL(policy || magnet) toward a heuristic "sensible play" prior instead of
    # a plain entropy bonus toward uniform randomness -- see training/magnet.py.
    # KL(p||m) = -H(p) - sum(p * log(m)); reusing entropy_coef as the pull
    # weight (same trick as strakam/AverageJoe's train/ppo.py).
    log_probs = asp.masked_log_softmax(logits, mask)
    magnet_dist = mg.expander_magnet(obs, mask)
    magnet_kl = -entropy - jnp.sum(jnp.exp(log_probs) * jnp.log(magnet_dist + 1e-10))

    return logprob, entropy, value, magnet_kl


def make_train_step(optimizer: optax.GradientTransformation, build_castles_enabled: bool):
    """Bakes in `optimizer`/`build_castles_enabled` (a Python-level `if` inside
    action_space.compute_legal_action_mask needs the latter to be static, and
    optax's GradientTransformation is a namedtuple of plain functions, not a
    valid jit argument) — call once per curriculum stage (build_castles_enabled
    only changes at the Stage-D transition) rather than per iteration."""

    def loss_fn(net, batch: FlatBatch, clip_eps, value_coef, entropy_coef):
        logprob, entropy, value, magnet_kl = jax.vmap(
            lambda o, m, a: _forward_one(net, o, m, a, build_castles_enabled)
        )(batch.obs, batch.mem, batch.action_index)

        ratio = jnp.exp(logprob - batch.old_logprob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch.advantage, clipped_ratio * batch.advantage))

        value_loss = jnp.mean(0.5 * (value - batch.ret) ** 2)
        entropy_bonus = jnp.mean(entropy)
        magnet_kl_mean = jnp.mean(magnet_kl)

        loss = policy_loss + value_coef * value_loss + entropy_coef * magnet_kl_mean

        approx_kl = jnp.mean(batch.old_logprob - logprob)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

        metrics = {
            "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy_bonus, "magnet_kl": magnet_kl_mean,
            "approx_kl": approx_kl, "clip_frac": clip_frac,
        }
        return loss, metrics

    @eqx.filter_jit
    def train_step(net, opt_state, batch: FlatBatch, clip_eps, value_coef, entropy_coef):
        (_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            net, batch, clip_eps, value_coef, entropy_coef
        )
        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, metrics

    return train_step


class FlatBatchTransformer(NamedTuple):
    obs: Observation
    obs_state: AugmentedObsState
    action_index: jnp.ndarray
    old_logprob: jnp.ndarray
    advantage: jnp.ndarray
    ret: jnp.ndarray


def flatten_batch_transformer(trajectory, advantages: jnp.ndarray, returns: jnp.ndarray) -> FlatBatchTransformer:
    """Transformer-path twin of flatten_batch -- trajectory: rollout.py's
    StepDataTransformer with leading axes (num_steps, num_envs, ...)."""
    def flat(x):
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    return FlatBatchTransformer(
        obs=jax.tree.map(flat, trajectory.obs),
        obs_state=jax.tree.map(flat, trajectory.obs_state),
        action_index=flat(trajectory.action_index),
        old_logprob=flat(trajectory.logprob),
        advantage=flat(advantages),
        ret=flat(returns),
    )


def _forward_one_transformer(net: TransformerPolicyValueNetwork, obs: Observation, obs_state: AugmentedObsState,
                              action_index: jnp.ndarray, build_castles_enabled: bool):
    # obs_state arrives here as whatever FlatBatchTransformer carries -- bf16
    # for its float leaves, per training/rollout.py's trajectory-storage
    # mitigation (the scan CARRY used during actual rollout stayed float32;
    # only the copy written into the trajectory was downcast). Upcasting here
    # recomputes from that same stored (already bf16-rounded) copy, not a
    # phantom full-precision value that was never actually persisted -- so
    # old_logprob (computed rollout-time from the true float32 carry) and the
    # logprob recomputed here carry a small extra discrepancy beyond ordinary
    # float32 rounding. This is the accepted cost of the memory mitigation,
    # not a bug: PPO's clipped ratio and target_kl exist precisely to tolerate
    # bounded old-vs-new policy drift, and this adds a fixed, small amount of
    # it, the same way any bf16-activation training run does.
    obs_state = jax.tree.map(
        lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x, obs_state
    )
    augmented, new_state = aug_mod.augment_obs(obs, obs_state)
    normed = aug_mod.normalize_augmented(augmented)
    temporal = aug_mod.temporal_data(new_state)
    logits, value = net(normed, temporal)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    logprob, entropy = asp.logprob_and_entropy(logits, mask, action_index)

    # KL(policy || magnet) -- see training/ppo.py's _forward_one for the same
    # mechanism on the CNN path. expander_magnet is confirmed
    # architecture-agnostic (operates on the raw Observation + legal mask,
    # never the network tensor), so this is identical to the CNN version.
    log_probs = asp.masked_log_softmax(logits, mask)
    magnet_dist = mg.expander_magnet(obs, mask)
    magnet_kl = -entropy - jnp.sum(jnp.exp(log_probs) * jnp.log(magnet_dist + 1e-10))

    return logprob, entropy, value, magnet_kl


def make_train_step_transformer(optimizer: optax.GradientTransformation, build_castles_enabled: bool):
    """Transformer-path twin of make_train_step."""

    def loss_fn(net, batch: FlatBatchTransformer, clip_eps, value_coef, entropy_coef):
        logprob, entropy, value, magnet_kl = jax.vmap(
            lambda o, s, a: _forward_one_transformer(net, o, s, a, build_castles_enabled)
        )(batch.obs, batch.obs_state, batch.action_index)

        ratio = jnp.exp(logprob - batch.old_logprob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch.advantage, clipped_ratio * batch.advantage))

        value_loss = jnp.mean(0.5 * (value - batch.ret) ** 2)
        entropy_bonus = jnp.mean(entropy)
        magnet_kl_mean = jnp.mean(magnet_kl)

        loss = policy_loss + value_coef * value_loss + entropy_coef * magnet_kl_mean

        approx_kl = jnp.mean(batch.old_logprob - logprob)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

        metrics = {
            "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy_bonus, "magnet_kl": magnet_kl_mean,
            "approx_kl": approx_kl, "clip_frac": clip_frac,
        }
        return loss, metrics

    @eqx.filter_jit
    def train_step(net, opt_state, batch: FlatBatchTransformer, clip_eps, value_coef, entropy_coef):
        (_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            net, batch, clip_eps, value_coef, entropy_coef
        )
        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, metrics

    return train_step


def compute_top_k_advantage_indices(advantages: jnp.ndarray, adv_top_frac: float,
                                     minibatch_size: int) -> jnp.ndarray:
    """advantages: (total_samples,) flat, in the SAME order flatten_batch's
    output is in (so the returned indices index directly into a FlatBatch's
    fields via _index_batch/train_epoch's `sample_idx`). Returns indices of
    the adv_top_frac highest-|advantage| samples, rounded down to a multiple
    of minibatch_size so every epoch's minibatches stay a fixed size (no
    recompilation). Ported from strakam/AverageJoe's train/ppo.py
    (`n_keep`/`_compute_top_idx`) -- concentrates gradient steps on the
    highest-signal transitions instead of the (often near-zero-advantage,
    in a sparse-reward regime) majority."""
    total = advantages.shape[0]
    n_keep = int(total * adv_top_frac)
    n_keep = (n_keep // minibatch_size) * minibatch_size
    n_keep = max(min(minibatch_size, total), n_keep)  # never below one minibatch (or total, if smaller)
    _, top_idx = jax.lax.top_k(jnp.abs(advantages), n_keep)
    return top_idx


def train_epoch(net, opt_state, train_step, flat_batch: FlatBatch, key: jnp.ndarray,
                 minibatch_size: int, clip_eps: float, value_coef: float, entropy_coef: float,
                 sample_idx: jnp.ndarray | None = None):
    """One shuffled pass over `flat_batch`, or -- if `sample_idx` is given
    (see compute_top_k_advantage_indices) -- one shuffled pass over just that
    subset, reshuffled fresh each call so a multi-epoch loop sees a different
    minibatch grouping of the same filtered pool every epoch. Drops the last
    incomplete minibatch (keeps minibatch_size fixed -> no recompilation
    across epochs/iterations)."""
    pool_idx = jnp.arange(flat_batch.action_index.shape[0]) if sample_idx is None else sample_idx
    bs = pool_idx.shape[0]
    perm = jrandom.permutation(key, bs)
    shuffled = _index_batch(flat_batch, pool_idx[perm])

    # Normally bs is an exact multiple of minibatch_size (131072/1024=128 at
    # the real training defaults), so dropping a remainder never loses data.
    # But bs can be smaller than minibatch_size entirely (tiny num_envs, e.g.
    # in a smoke test) -- guard against silently running zero minibatches
    # (which would skip the gradient update AND return metrics=None).
    num_batches = max(1, bs // minibatch_size)
    effective_minibatch = min(minibatch_size, bs)
    metrics_sum = None
    for i in range(num_batches):
        sl = slice(i * effective_minibatch, (i + 1) * effective_minibatch)
        mb = _index_batch(shuffled, sl)
        net, opt_state, metrics = train_step(net, opt_state, mb, clip_eps, value_coef, entropy_coef)
        metrics_sum = metrics if metrics_sum is None else jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)

    metrics_avg = jax.tree.map(lambda x: x / max(num_batches, 1), metrics_sum)
    return net, opt_state, metrics_avg
