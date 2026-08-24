"""Backend-agnostic core of the single-stage transformer PPO training loop --
extracted from colab/train_colab.py so modal/train_modal.py can reuse the
exact same (already GPU-battle-tested) per-iteration logic instead of a
second, independently-drifting copy. Everything backend-specific --
checkpoint paths, signal handling, git-push status, wandb, wall-clock/
iteration stopping conditions -- stays in the caller (colab/train_colab.py,
modal/train_modal.py); this module only knows how to build the initial
training state and advance it by one PPO iteration.

Still curriculum.STAGE_E only (full competition ruleset), no league/PFSP --
see colab/train_colab.py's module docstring for why.
"""
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents import ExpanderAgent, HunterAgent

from training import curriculum as cur
from training import ppo
from training import rollout as ro
from training.config import PPOConfig, TransformerNetworkConfig
from training.network_transformer import TransformerPolicyValueNetwork

HEURISTIC_AGENTS = {"hunter": HunterAgent(), "expander": ExpanderAgent()}

STAGE = cur.STAGE_E  # full competition ruleset -- the only stage this loop trains


class TrainState(NamedTuple):
    net: TransformerPolicyValueNetwork
    opt_state: object
    ema_net: TransformerPolicyValueNetwork
    buckets: dict
    key: jnp.ndarray
    env: object
    pool: object
    train_step: object
    global_iteration: int


def build_env_and_buckets(cfg: TransformerNetworkConfig, num_envs: int, hunter_frac: float,
                           expander_frac: float, key: jnp.ndarray):
    """Returns (env, pool, buckets, key). Bucket sizes are fractions of
    num_envs; whatever's left after hunter+expander goes to self-play."""
    env = cur.build_env(STAGE)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    n_hunter = int(num_envs * hunter_frac)
    n_expander = int(num_envs * expander_frac)
    n_selfplay = num_envs - n_hunter - n_expander
    print(f"opponent mix: hunter={n_hunter} expander={n_expander} selfplay={n_selfplay}")

    key, k1, k2, k3 = jrandom.split(key, 4)
    buckets = {}
    if n_hunter > 0:
        buckets["hunter"] = {
            "states": jax.vmap(env.init_state)(jrandom.split(k1, n_hunter)),
            "obs_state0": ro.init_obs_state_batch(n_hunter, cfg),
        }
    if n_expander > 0:
        buckets["expander"] = {
            "states": jax.vmap(env.init_state)(jrandom.split(k2, n_expander)),
            "obs_state0": ro.init_obs_state_batch(n_expander, cfg),
        }
    if n_selfplay > 0:
        buckets["selfplay"] = {
            "states": jax.vmap(env.init_state)(jrandom.split(k3, n_selfplay)),
            "obs_state0": ro.init_obs_state_batch(n_selfplay, cfg),
            "obs_state1": ro.init_obs_state_batch(n_selfplay, cfg),
        }
    return env, pool, buckets, key


def init_train_state(cfg: TransformerNetworkConfig, ppo_cfg: PPOConfig, seed: int,
                      hunter_frac: float = 0.15, expander_frac: float = 0.15) -> TrainState:
    """Fresh (not resumed) training state -- the caller is responsible for
    overwriting net/opt_state/ema_net/global_iteration from a checkpoint
    afterward if resuming (checkpoint I/O is backend-specific: a local path
    for colab/train_colab.py, a modal.Volume path for modal/train_modal.py)."""
    key = jrandom.PRNGKey(seed)
    net_key, key = jrandom.split(key)
    net = TransformerPolicyValueNetwork(net_key, cfg)
    ema_net = net
    optimizer = ppo.make_optimizer(ppo_cfg.learning_rate, ppo_cfg.max_grad_norm)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    train_step = ppo.make_train_step_transformer(optimizer, STAGE.build_castles_enabled)

    env, pool, buckets, key = build_env_and_buckets(
        cfg, ppo_cfg.num_envs, hunter_frac, expander_frac, key
    )

    return TrainState(
        net=net, opt_state=opt_state, ema_net=ema_net, buckets=buckets, key=key,
        env=env, pool=pool, train_step=train_step, global_iteration=0,
    )


def run_iteration(state: TrainState, ppo_cfg: PPOConfig, stage=STAGE) -> tuple[TrainState, dict]:
    """One PPO iteration: rollout across every bucket, GAE, flatten, top-k
    filter, PPO update (with target_kl early-stop), EMA update. Returns
    (new_state, metrics_row) -- metrics_row is plain-JSON-serializable,
    ready for append_jsonl/wandb.log/push_status in the caller."""
    net, opt_state, ema_net = state.net, state.opt_state, state.ema_net
    buckets, key, env, pool = state.buckets, state.key, state.env, state.pool

    import time
    iter_t0 = time.time()
    trajectories, bootstraps, bucket_stats = [], [], {}

    for name, bucket in buckets.items():
        key, rk = jrandom.split(key)
        if name == "selfplay":
            states, obs_state0, obs_state1, traj, boot = ro.collect_selfplay_rollout_transformer(
                env, bucket["states"], bucket["obs_state0"], bucket["obs_state1"], pool, net, net, rk,
                ppo_cfg.num_steps, stage.build_castles_enabled, ppo_cfg.gamma,
                ppo_cfg.contact_shaping_weight, ppo_cfg.general_safety_weight,
            )
            bucket["states"], bucket["obs_state0"], bucket["obs_state1"] = states, obs_state0, obs_state1
        else:
            states, obs_state0, traj, boot = ro.collect_vs_heuristic_rollout_transformer(
                env, bucket["states"], bucket["obs_state0"], pool, net, HEURISTIC_AGENTS[name], rk,
                ppo_cfg.num_steps, stage.build_castles_enabled, ppo_cfg.gamma,
                ppo_cfg.contact_shaping_weight, ppo_cfg.general_safety_weight,
            )
            bucket["states"], bucket["obs_state0"] = states, obs_state0

        trajectories.append(traj)
        bootstraps.append(boot)
        episodes = int(jnp.sum(traj.done))
        wins = int(jnp.sum(traj.done & (traj.winner == 0)))
        losses = int(jnp.sum(traj.done & (traj.winner == 1)))
        bucket_stats[name] = {"episodes": episodes, "wins": wins, "losses": losses,
                               "draws": episodes - wins - losses}

    trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *trajectories)
    bootstrap = jnp.concatenate(bootstraps, axis=0)

    advantages = ro.compute_gae(trajectory.reward, trajectory.value, trajectory.done, bootstrap,
                                 ppo_cfg.gamma, ppo_cfg.lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + trajectory.value
    flat = ppo.flatten_batch_transformer(trajectory, advantages, returns)
    sample_idx = ppo.compute_top_k_advantage_indices(
        flat.advantage, ppo_cfg.adv_top_frac, ppo_cfg.minibatch_size)

    epoch_metrics = None
    epochs_used = 0
    for _epoch in range(ppo_cfg.num_epochs):
        key, ek = jrandom.split(key)
        net, opt_state, epoch_metrics = ppo.train_epoch(
            net, opt_state, state.train_step, flat, ek, ppo_cfg.minibatch_size,
            ppo_cfg.clip_eps, ppo_cfg.value_coef, stage.entropy_coef, sample_idx=sample_idx,
        )
        epochs_used += 1
        if ppo_cfg.target_kl is not None and float(epoch_metrics["approx_kl"]) > ppo_cfg.target_kl:
            break

    decay = ppo_cfg.weight_ema_decay
    ema_net = jax.tree.map(lambda e, c: decay * e + (1 - decay) * c, ema_net, net)

    global_iteration = state.global_iteration + 1
    elapsed = time.time() - iter_t0
    metrics_row = {
        "ts": time.time(), "iteration": global_iteration, "wall_clock_iter_s": elapsed,
        "sps": (ppo_cfg.num_envs * ppo_cfg.num_steps) / elapsed,
        "mean_reward": float(trajectory.reward.mean()), "buckets": bucket_stats,
        "epochs_used": epochs_used,
        **{k: float(v) for k, v in epoch_metrics.items()},
    }

    new_state = state._replace(
        net=net, opt_state=opt_state, ema_net=ema_net, buckets=buckets, key=key,
        global_iteration=global_iteration,
    )
    return new_state, metrics_row
