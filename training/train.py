"""Main training orchestration: curriculum stage progression, self-play
rollout collection across opponent buckets, PPO updates, checkpointing,
league management, periodic evaluation, and crash-safe resume.

Everything a check-in needs lives under training/runs/<run_id>/:
  metrics.jsonl        one line per iteration (loss terms, per-bucket win/loss, wall-clock)
  stage.json           overwritten each iteration; status is set on every exit path
                        (running/stopped_ok/crashed: <exc>) so a dead run is obvious
  eval_baselines.jsonl  periodic vectorized win-rate vs Random/Expander/Hunter
  train.log             human-readable log (same content as stdout)
  checkpoints/          latest_full.eqx (net+opt_state, resume), league/*.eqx (snapshots)

Usage:
    python -m training.train --run-id my-run [--fresh] [--max-wall-clock-hours 8]
"""
import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent

from training import curriculum as cur
from training import eval_baselines as eb
from training import league as lg
from training import ppo
from training import rollout as ro
from training.config import LeagueConfig, NetworkConfig, PPOConfig
from training.network import PolicyValueNetwork

TRAINING_DIR = Path(__file__).resolve().parent

HEURISTIC_AGENTS = {
    "random": RandomAgent(),
    "expander": ExpanderAgent(),
    "hunter": HunterAgent(),
}

# Index of STAGE_C within curriculum.STAGES — its final network becomes the
# frozen opponent Stage D is promoted against ("beat where you started").
_FROZEN_REFERENCE_STAGE_IDX = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="run identifier; determines training/runs/<run-id>/")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint, start over")
    p.add_argument("--max-wall-clock-hours", type=float, default=None,
                    help="stop (with a clean checkpoint) after this many hours; default: run until killed")
    p.add_argument("--num-envs", type=int, default=None, help=f"default: {PPOConfig().num_envs}")
    p.add_argument("--num-steps", type=int, default=None, help=f"default: {PPOConfig().num_steps}")
    p.add_argument("--eval-interval", type=int, default=20, help="iterations between eval_baselines runs")
    p.add_argument("--eval-games", type=int, default=64, help="parallel games per baseline agent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-net-from", type=str, default=None,
                    help="path to a network-only .eqx checkpoint (e.g. from training/pretrain_bc.py) "
                         "to warm-start a brand-new run's weights from. Ignored if this run-id already "
                         "has a checkpoint to resume from -- only affects the very first launch.")
    p.add_argument("--start-stage", type=str, default=None,
                    help="curriculum stage NAME to start a brand-new run at (e.g. C_real_board_size), "
                         "instead of always A_foundations. Useful when --init-net-from was pretrained on "
                         "a later stage's exact ruleset -- starting at an earlier, mismatched stage can "
                         "actively fight a warm start (different observation distribution, and earlier "
                         "stages use a higher entropy_coef meant to encourage exploration from scratch, "
                         "which pulls a confident pretrained policy back toward randomness). Ignored if "
                         "this run-id already has a checkpoint to resume from.")
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def build_optimizer_and_state(net: PolicyValueNetwork, ppo_cfg: PPOConfig):
    optimizer = ppo.make_optimizer(ppo_cfg.learning_rate, ppo_cfg.max_grad_norm)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    return optimizer, opt_state


def main():
    args = parse_args()
    run_dir = TRAINING_DIR / "runs" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    league_dir = ckpt_dir / "league"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info(f"jax devices: {jax.devices()}")

    default_ppo = PPOConfig()
    ppo_cfg = PPOConfig(
        num_envs=args.num_envs or default_ppo.num_envs,
        num_steps=args.num_steps or default_ppo.num_steps,
    )
    net_cfg = NetworkConfig()
    league_cfg = LeagueConfig()
    logger.info(f"run_id={args.run_id} fresh={args.fresh} max_wall_clock_hours={args.max_wall_clock_hours} "
                f"num_envs={ppo_cfg.num_envs} num_steps={ppo_cfg.num_steps} num_epochs={ppo_cfg.num_epochs}")

    base_key = jrandom.PRNGKey(args.seed)
    net_key, key = jrandom.split(base_key)
    net = PolicyValueNetwork(net_key, net_cfg)

    resuming = not args.fresh and (run_dir / "checkpoints" / "train_meta.json").exists() and \
        (run_dir / "checkpoints" / "latest_full.eqx").exists()
    if args.init_net_from and not resuming:
        net = eqx.tree_deserialise_leaves(args.init_net_from, net)
        logger.info(f"warm-started network weights from {args.init_net_from}")

    # Exponential moving average of net's weights, updated every iteration
    # below -- used for eval/promotion decisions and the frozen Stage-D
    # reference instead of the raw, currently-being-gradient-updated live
    # net, to insulate those signals from single-iteration noise. Starts as
    # a copy of net (also serves as the deserialization template on resume).
    ema_net = net

    optimizer, opt_state = build_optimizer_and_state(net, ppo_cfg)
    league = lg.League(league_dir, league_cfg, net)

    stage_idx = 0
    if args.start_stage and not resuming:
        if args.start_stage not in cur.STAGE_BY_NAME:
            raise SystemExit(f"--start-stage {args.start_stage!r} not found; known stages: "
                              f"{[s.name for s in cur.STAGES]}")
        stage_idx = cur.STAGES.index(cur.STAGE_BY_NAME[args.start_stage])
        logger.info(f"starting fresh run directly at stage {args.start_stage} (idx={stage_idx}), "
                    f"skipping earlier stages")
    iters_in_stage = 0
    global_iteration = 0
    latest_eval = {}
    latest_vs_random = 0.0
    latest_vs_frozen = 0.0

    meta_path = ckpt_dir / "train_meta.json"
    full_ckpt_path = ckpt_dir / "latest_full.eqx"
    league_index_path = league_dir / "league_index.json"
    frozen_ref_path = ckpt_dir / "frozen_reference.eqx"

    if not args.fresh and meta_path.exists() and full_ckpt_path.exists():
        meta = json.loads(meta_path.read_text())
        stage_idx = meta["stage_idx"]
        iters_in_stage = meta["iters_in_stage"]
        global_iteration = meta["global_iteration"]
        latest_eval = meta.get("latest_eval_baselines", {})
        latest_vs_random = meta.get("latest_vs_random_winrate", 0.0)
        latest_vs_frozen = meta.get("latest_vs_frozen_winrate", 0.0)
        net, opt_state, ema_net = eqx.tree_deserialise_leaves(str(full_ckpt_path), (net, opt_state, ema_net))
        league.load_index(league_index_path)
        logger.info(f"resumed: stage_idx={stage_idx} ({cur.STAGES[stage_idx].name}) "
                    f"iters_in_stage={iters_in_stage} global_iteration={global_iteration} "
                    f"league_members={len(league.members)}")
    else:
        logger.info("starting fresh")

    stage = cur.STAGES[stage_idx]
    train_step = ppo.make_train_step(optimizer, stage.build_castles_enabled)
    env = cur.build_env(stage)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(jrandom.fold_in(pool_key, stage_idx))
    logger.info(f"stage {stage.name}: env built, pool_size={env.pool_size}")

    frozen_reference_net = None
    if frozen_ref_path.exists():
        frozen_reference_net = eqx.tree_deserialise_leaves(str(frozen_ref_path), net)
        logger.info("loaded frozen Stage-C reference checkpoint")

    bucket_cache: dict = {}
    prev_counts: dict = {}

    metrics_path = run_dir / "metrics.jsonl"
    stage_status_path = run_dir / "stage.json"
    eval_baselines_path = run_dir / "eval_baselines.jsonl"

    def write_stage_status(status: str) -> None:
        write_json(stage_status_path, {
            "stage_idx": stage_idx, "stage_name": stage.name,
            "iters_in_stage": iters_in_stage, "global_iteration": global_iteration,
            "promotion_metric": stage.promotion_metric,
            "promotion_threshold": stage.promotion_threshold,
            "latest_vs_random_winrate": latest_vs_random,
            "latest_vs_frozen_winrate": latest_vs_frozen,
            "status": status, "ts": time.time(),
        })

    def save_full_checkpoint() -> None:
        eqx.tree_serialise_leaves(str(full_ckpt_path), (net, opt_state, ema_net))
        write_json(meta_path, {
            "stage_idx": stage_idx, "iters_in_stage": iters_in_stage,
            "global_iteration": global_iteration,
            "latest_eval_baselines": latest_eval,
            "latest_vs_random_winrate": latest_vs_random,
            "latest_vs_frozen_winrate": latest_vs_frozen,
        })
        league.save_index(league_index_path)

    shutdown_requested = {"flag": False}

    def handle_signal(signum, _frame):
        logger.info(f"received signal {signum}; will stop after the current iteration")
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    write_stage_status("running")
    start_time = time.time()

    try:
        while True:
            wall_clock_hours = (time.time() - start_time) / 3600.0
            if args.max_wall_clock_hours and wall_clock_hours >= args.max_wall_clock_hours:
                logger.info(f"reached max_wall_clock_hours={args.max_wall_clock_hours}, stopping")
                break
            if shutdown_requested["flag"]:
                break

            iter_t0 = time.time()

            mix = cur.resolve_opponent_mix(stage, latest_vs_random, league.has_snapshots())
            counts = cur.allocate_envs(mix, ppo_cfg.num_envs)

            trajectories = []
            bootstraps = []
            bucket_stats = {}

            for name in cur.BUCKET_NAMES:
                count = counts.get(name, 0)
                if count == 0:
                    bucket_cache.pop(name, None)
                    prev_counts[name] = 0
                    continue

                needs_mem1 = name in ("selfplay_current", "selfplay_league")
                key, bucket_key, roll_key = jrandom.split(key, 3)
                if name not in bucket_cache or prev_counts.get(name) != count:
                    init_keys = jrandom.split(bucket_key, count)
                    bucket_cache[name] = {
                        "states": jax.vmap(env.init_state)(init_keys),
                        "mem0": ro.init_memory_batch(count, net_cfg.grid_size, net_cfg.grid_size),
                        "mem1": ro.init_memory_batch(count, net_cfg.grid_size, net_cfg.grid_size) if needs_mem1 else None,
                    }
                    prev_counts[name] = count
                bucket = bucket_cache[name]
                states, mem0, mem1 = bucket["states"], bucket["mem0"], bucket["mem1"]

                league_member_id = None
                if name in ("random", "expander", "hunter"):
                    states, mem0, traj, boot = ro.collect_vs_heuristic_rollout(
                        env, states, mem0, pool, net, HEURISTIC_AGENTS[name], roll_key,
                        ppo_cfg.num_steps, stage.build_castles_enabled,
                        ppo_cfg.gamma, ppo_cfg.contact_shaping_weight, ppo_cfg.general_safety_weight)
                elif name == "selfplay_current":
                    states, mem0, mem1, traj, boot = ro.collect_selfplay_rollout(
                        env, states, mem0, mem1, pool, net, net, roll_key,
                        ppo_cfg.num_steps, stage.build_castles_enabled,
                        ppo_cfg.gamma, ppo_cfg.contact_shaping_weight, ppo_cfg.general_safety_weight)
                elif name == "selfplay_league":
                    key, sample_key = jrandom.split(key)
                    member, opp_params = league.sample_opponent(sample_key)
                    league_member_id = member.id
                    states, mem0, mem1, traj, boot = ro.collect_selfplay_rollout(
                        env, states, mem0, mem1, pool, net, opp_params, roll_key,
                        ppo_cfg.num_steps, stage.build_castles_enabled,
                        ppo_cfg.gamma, ppo_cfg.contact_shaping_weight, ppo_cfg.general_safety_weight)
                else:
                    continue

                bucket_cache[name] = {"states": states, "mem0": mem0, "mem1": mem1}
                trajectories.append(traj)
                bootstraps.append(boot)

                episodes = int(jnp.sum(traj.done))
                wins = int(jnp.sum(traj.done & (traj.winner == 0)))
                losses = int(jnp.sum(traj.done & (traj.winner == 1)))
                bucket_stats[name] = {"episodes": episodes, "wins": wins,
                                       "losses": losses, "draws": episodes - wins - losses}
                if league_member_id is not None:
                    league.update_winrate(league_member_id, wins, episodes)

            trajectory = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *trajectories)
            bootstrap = jnp.concatenate(bootstraps, axis=0)

            advantages = ro.compute_gae(trajectory.reward, trajectory.value, trajectory.done,
                                         bootstrap, ppo_cfg.gamma, ppo_cfg.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + trajectory.value
            flat = ppo.flatten_batch(trajectory, advantages, returns)
            sample_idx = ppo.compute_top_k_advantage_indices(
                flat.advantage, ppo_cfg.adv_top_frac, ppo_cfg.minibatch_size)

            epoch_metrics = None
            epochs_used = 0
            for _epoch in range(ppo_cfg.num_epochs):
                key, ek = jrandom.split(key)
                net, opt_state, epoch_metrics = ppo.train_epoch(
                    net, opt_state, train_step, flat, ek, ppo_cfg.minibatch_size,
                    ppo_cfg.clip_eps, ppo_cfg.value_coef, stage.entropy_coef, sample_idx=sample_idx)
                epochs_used += 1
                if ppo_cfg.target_kl is not None and float(epoch_metrics["approx_kl"]) > ppo_cfg.target_kl:
                    break

            decay = ppo_cfg.weight_ema_decay
            ema_net = jax.tree.map(lambda e, c: decay * e + (1 - decay) * c, ema_net, net)

            iters_in_stage += 1
            global_iteration += 1
            elapsed = time.time() - iter_t0

            metrics_row = {
                "ts": time.time(), "iteration": global_iteration, "stage": stage.name,
                "stage_iter": iters_in_stage, "wall_clock_iter_s": elapsed,
                "sps": (ppo_cfg.num_envs * ppo_cfg.num_steps) / elapsed,
                "mean_reward": float(trajectory.reward.mean()),
                "buckets": bucket_stats,
                "epochs_used": epochs_used,
                **{k: float(v) for k, v in epoch_metrics.items()},
            }
            append_jsonl(metrics_path, metrics_row)
            logger.info(f"iter {global_iteration} stage={stage.name}({iters_in_stage}) "
                        f"loss={epoch_metrics['loss']:.4f} entropy={epoch_metrics['entropy']:.3f} "
                        f"magnet_kl={epoch_metrics['magnet_kl']:.4f} kl={epoch_metrics['approx_kl']:.4f} "
                        f"reward={metrics_row['mean_reward']:+.4f} "
                        f"sps={metrics_row['sps']:.0f} time={elapsed:.1f}s")

            if global_iteration % league_cfg.full_checkpoint_interval == 0:
                save_full_checkpoint()

            if global_iteration % league_cfg.snapshot_interval == 0:
                league.add_snapshot(net, stage.name, global_iteration)
                logger.info(f"league snapshot saved: {stage.name}_iter{global_iteration}")

            if global_iteration % args.eval_interval == 0:
                # Eval (and thus promotion decisions) read ema_net, not the
                # raw live net -- smooths out the iteration-to-iteration
                # noise that otherwise makes a single eval point unreliable.
                key, eval_key = jrandom.split(key)
                latest_eval = eb.evaluate_vs_baselines(env, ema_net, pool, eval_key,
                                                         args.eval_games, stage.build_castles_enabled)
                latest_vs_random = latest_eval["vs_random"]["winrate"]
                eval_row = {"ts": time.time(), "iteration": global_iteration, "stage": stage.name,
                            **latest_eval}

                if stage.promotion_metric == "vs_frozen_checkpoint" and frozen_reference_net is not None:
                    key, fk = jrandom.split(key)
                    vs_frozen = eb.evaluate_vs_network(env, ema_net, frozen_reference_net, pool, fk,
                                                        args.eval_games, stage.build_castles_enabled)
                    latest_vs_frozen = vs_frozen["winrate"]
                    eval_row["vs_frozen_checkpoint"] = vs_frozen

                append_jsonl(eval_baselines_path, eval_row)
                logger.info(f"eval @ iter {global_iteration}: " +
                            " ".join(f"{k}={v['winrate']:.2f}" for k, v in latest_eval.items()) +
                            (f" vs_frozen={latest_vs_frozen:.2f}" if "vs_frozen_checkpoint" in eval_row else ""))

            write_stage_status("running")

            promoted = False
            promotion_reason = None
            if stage.promotion_metric is not None:
                metric_value = {
                    "vs_random": latest_vs_random,
                    "vs_expander": latest_eval.get("vs_expander", {}).get("winrate", 0.0),
                    "vs_frozen_checkpoint": latest_vs_frozen,
                }.get(stage.promotion_metric, 0.0)

                hit_threshold = metric_value >= stage.promotion_threshold
                hit_cap = iters_in_stage >= stage.promotion_iter_cap
                if hit_threshold or hit_cap:
                    promoted = True
                    promotion_reason = "threshold" if hit_threshold else "iteration_cap"
                    logger.info(f"STAGE PROMOTION: {stage.name} -> next "
                                f"(reason={promotion_reason}, metric={metric_value:.3f}, "
                                f"iters_in_stage={iters_in_stage})")

            if promoted:
                league.add_snapshot(net, stage.name, global_iteration)

                if stage_idx == _FROZEN_REFERENCE_STAGE_IDX:
                    eqx.tree_serialise_leaves(str(frozen_ref_path), ema_net)
                    frozen_reference_net = ema_net
                    logger.info("saved frozen Stage-C reference checkpoint (EMA weights) "
                                "for Stage D's promotion metric")

                prev_stage_name = stage.name
                stage_idx = min(stage_idx + 1, len(cur.STAGES) - 1)
                stage = cur.STAGES[stage_idx]
                iters_in_stage = 0
                latest_eval = {}
                latest_vs_random = 0.0
                latest_vs_frozen = 0.0

                logger.info(f"stage_transition_start: {prev_stage_name} -> {stage.name}")
                t0 = time.time()
                league.prune_to_stages({prev_stage_name, stage.name})
                train_step = ppo.make_train_step(optimizer, stage.build_castles_enabled)
                env = cur.build_env(stage)
                key, pool_key = jrandom.split(key)
                pool, _ = env.reset(jrandom.fold_in(pool_key, stage_idx))
                bucket_cache.clear()
                prev_counts.clear()
                logger.info(f"stage_transition_recompile_done: {stage.name} "
                            f"(pool_size={env.pool_size}, took {time.time() - t0:.1f}s)")
                save_full_checkpoint()

        save_full_checkpoint()
        write_stage_status("stopped_ok")
        logger.info("training loop exited cleanly")
    except Exception as e:
        write_stage_status(f"crashed: {e!r}")
        logger.exception("training loop crashed")
        raise


if __name__ == "__main__":
    main()
