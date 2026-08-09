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
from training import magnet as mg
from training.memory import MemoryState
from training.network import PolicyValueNetwork


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


def _index_batch(batch: FlatBatch, idx) -> FlatBatch:
    return FlatBatch(
        obs=jax.tree.map(lambda x: x[idx], batch.obs),
        mem=jax.tree.map(lambda x: x[idx], batch.mem),
        action_index=batch.action_index[idx],
        old_logprob=batch.old_logprob[idx],
        advantage=batch.advantage[idx],
        ret=batch.ret[idx],
    )


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


def train_epoch(net, opt_state, train_step, flat_batch: FlatBatch, key: jnp.ndarray,
                 minibatch_size: int, clip_eps: float, value_coef: float, entropy_coef: float):
    """One shuffled pass over `flat_batch`. Drops the last incomplete minibatch
    (keeps minibatch_size fixed -> no recompilation across epochs/iterations)."""
    bs = flat_batch.action_index.shape[0]
    perm = jrandom.permutation(key, bs)
    shuffled = _index_batch(flat_batch, perm)

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
