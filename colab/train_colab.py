"""Standalone single-stage PPO training loop for the transformer network
(training/network_transformer.py + training/augment.py), meant to run on a
Colab GPU runtime -- Step 6 of ~/.claude/plans/drifting-questing-babbage.md.

Deliberately NOT training/train.py's full multi-stage curriculum + league/
PFSP orchestration: training/league.py and training/eval_baselines.py
haven't been ported to the transformer path yet (a separate, not-yet-done
item in the plan's Files list). This runs a single fixed stage --
curriculum.STAGE_E's env_kwargs, the full competition ruleset -- against a
simple opponent mix (Hunter + Expander + self-play mirror, no league/PFSP),
matching the plan's own framing for this step: "smoke-test a handful of real
iterations... as early as practical", not a full production run. Every
per-step primitive (network, augment_obs, rollout, PPO update) is reused
directly from training/rollout.py and training/ppo.py, already verified
end-to-end on CPU earlier in this plan -- this script is orchestration only,
no new per-step math.

Resume/crash-safety reuses training/train.py's own train_meta.json /
latest_full.eqx convention (per the plan's explicit instruction) so a killed
Colab session restarts cleanly, minus the stage_idx/iters_in_stage fields
(there's only one stage here).

Runnable two ways:
  - Locally (CPU, tiny --num-envs/--num-steps) to smoke-test before ever
    touching Colab -- the same discipline every other change in this plan
    followed. No GPU, no Drive, no PAT, no wandb account needed for this
    (both --push-status and --wandb default off, and wandb itself is only
    imported when --wandb is actually passed -- matching how this script
    never imports `google.colab` either, so it stays portable/testable
    outside Colab without needing every optional integration installed).
  - On actual Colab GPU, from a notebook cell (see colab/train_colab.ipynb)
    after mounting Drive (for --ckpt-dir), setting a GITHUB_PAT secret (for
    --push-status), and/or a WANDB_API_KEY secret (for --wandb, live
    metrics charts at wandb.ai instead of tailing metrics.jsonl). The
    notebook cell is responsible for exporting these into the environment
    (`os.environ["GITHUB_PAT"] = userdata.get("GITHUB_PAT")`, etc.) and
    `drive.mount(...)` before invoking this.

Usage:
    python -m colab.train_colab --run-id colab1 \\
        --ckpt-dir /content/drive/MyDrive/generals-runs/colab1 \\
        --push-status --wandb --wandb-project generals-bots-transformer
"""
import os

# Must be set before jax (or anything that imports jax, e.g. equinox) is
# ever imported -- JAX reads this once at first initialization. Default
# behavior pre-allocates ~90% of GPU memory as one arena up front; as this
# script's rollout collection (3 differently-shaped opponent buckets) and
# PPO backward pass allocate/free different-shaped buffers each iteration,
# that arena can fragment into pieces too small to satisfy a later request
# even with plenty of total free memory -- confirmed on a real Colab run:
# minibatch_size=128 got through 2 clean iterations, then RESOURCE_EXHAUSTED
# on a mere ~110MB allocation at iteration 3, the fragmentation signature
# (a huge failed allocation would instead mean "still too big", not this).
# setdefault, not a hard assignment: don't override an explicit choice the
# user already made via the shell/notebook environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parent.parent

HEURISTIC_AGENTS = {"hunter": HunterAgent(), "expander": ExpanderAgent()}

# See --minibatch-size's help text: much lower than PPOConfig's own default
# (1024, tuned for the lighter CNN), chosen from a real Colab OOM at 1024 on
# a 15GB GPU -- a reasoned starting point, not a verified-safe number.
_DEFAULT_MINIBATCH_SIZE = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-dir", type=str, default=None,
                    help="where checkpoints/metrics.jsonl live -- default training/runs/<run-id> "
                         "(gitignored, local only); pass a Drive-mounted path on real Colab runs")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--num-envs", type=int, default=None, help=f"default: {PPOConfig().num_envs}")
    p.add_argument("--num-steps", type=int, default=None, help=f"default: {PPOConfig().num_steps}")
    p.add_argument("--minibatch-size", type=int, default=_DEFAULT_MINIBATCH_SIZE,
                    help=f"default: {_DEFAULT_MINIBATCH_SIZE} -- deliberately much lower than "
                         f"PPOConfig's own default ({PPOConfig().minibatch_size}, tuned for the "
                         "lighter CNN). A real Colab run OOM'd at 1024 trying to allocate ~3.5GiB "
                         "for a single op on a 15GB GPU; this is an untested-on-GPU but reasoned "
                         "starting point (~8x smaller), not a guarantee -- if it still OOMs, halve "
                         "it again. The direct lever for a GPU OOM during the PPO update "
                         "(train_step's forward+backward pass over one minibatch is the single "
                         "largest per-op allocation in this script) -- adjust this first, before "
                         "--num-envs/--num-steps, since those only change how many minibatches "
                         "there are, not each one's size")
    p.add_argument("--hunter-frac", type=float, default=0.15)
    p.add_argument("--expander-frac", type=float, default=0.15)
    p.add_argument("--max-wall-clock-hours", type=float, default=None)
    p.add_argument("--ckpt-every-iters", type=int, default=20)
    p.add_argument("--ckpt-every-minutes", type=float, default=15.0,
                    help="also checkpoint on this wall-clock cadence regardless of iteration count -- "
                         "well inside Colab's ~90min free-tier idle-disconnect window")
    p.add_argument("--push-status", action="store_true",
                    help="commit+push a small status file (colab/status/<run-id>.json) to origin every "
                         "--status-every-iters iterations, so progress is visible without direct Colab access")
    p.add_argument("--status-every-iters", type=int, default=10)
    p.add_argument("--status-path", type=str, default=None,
                    help="default: colab/status/<run-id>.json (repo-relative)")
    p.add_argument("--github-pat", type=str, default=None,
                    help="default: $GITHUB_PAT env var. Only needed with --push-status.")
    p.add_argument("--git-branch", type=str, default="master")
    p.add_argument("--wandb", action="store_true",
                    help="log every iteration's metrics to Weights & Biases -- a live-updating "
                         "dashboard (loss/entropy/kl/reward/per-bucket win-rate charts) instead of "
                         "tailing metrics.jsonl by hand. Requires the wandb package (pip install "
                         "wandb) and a logged-in account -- only imported/required when this flag "
                         "is passed, never otherwise.")
    p.add_argument("--wandb-project", type=str, default="generals-bots-transformer")
    p.add_argument("--wandb-entity", type=str, default=None, help="wandb team/username; default: your account's default")
    p.add_argument("--wandb-api-key", type=str, default=None,
                    help="default: $WANDB_API_KEY env var. Only needed with --wandb.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def push_status(status_path: Path, payload: dict, github_pat: str | None, branch: str) -> None:
    """Best-effort: a failed push must never kill a training run. Passes the
    PAT explicitly on the pull/push command lines rather than `git remote
    set-url` -- that would write the token into .git/config on disk, a real
    credential ending up somewhere it could be inspected/copied/shared by
    accident. git pull --rebase BEFORE writing/committing: a restarted Colab
    session (this run's own past incarnation, or the user re-running the
    same --run-id) must not clobber another still-running session's more
    recent status push.

    Defensively resets `status_path` out of the index before every pull and
    after any failed commit -- a fresh Colab clone has no git identity
    configured, so `git commit` fails outright the first time this runs;
    the original version of this function left the file `git add`-staged
    on that failure without ever undoing it, which then made every
    SUBSEQUENT call's `git pull --rebase` fail too ("index contains
    uncommitted changes"), permanently, for the rest of that clone's
    lifetime -- confirmed by hitting exactly this in a real Colab run."""
    import os

    status_path.parent.mkdir(parents=True, exist_ok=True)

    def run(cmd):
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    # Local to this one clone (not --global) -- a fresh Colab session has no
    # git identity configured at all, and `git commit` refuses to run
    # without one.
    run(["git", "config", "user.email", "colab-train-bot@users.noreply.github.com"])
    run(["git", "config", "user.name", "colab-train-bot"])

    pat = github_pat or os.environ.get("GITHUB_PAT")
    remote_url = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    if pat and remote_url.startswith("https://"):
        authed_url = remote_url.replace("https://", f"https://{pat}@", 1)
    else:
        authed_url = remote_url

    # Clear any staged-but-uncommitted leftovers from a prior call before
    # pulling -- git refuses to `pull --rebase` with a dirty index.
    run(["git", "reset", "--", str(status_path)])

    pull = run(["git", "pull", "--rebase", authed_url, branch])
    if pull.returncode != 0:
        print(f"[push_status] git pull --rebase failed, skipping this push: {pull.stderr[:500]}")
        run(["git", "rebase", "--abort"])  # leave the repo clean for the next call, win or lose
        return

    status_path.write_text(json.dumps(payload, indent=2))
    run(["git", "add", str(status_path)])
    commit = run(["git", "commit", "-m", f"colab status: {payload.get('run_id')} iter {payload.get('iteration')}"])
    if commit.returncode != 0:
        run(["git", "reset", "--", str(status_path)])  # undo the add -- keep the index clean either way
        return  # nothing changed since last push, or commit failed either way -- not fatal
    push = run(["git", "push", authed_url, f"HEAD:{branch}"])
    if push.returncode != 0:
        print(f"[push_status] git push failed, skipping: {push.stderr[:500]}")


def init_wandb(args, ppo_cfg: PPOConfig, cfg: TransformerNetworkConfig, stage, resuming: bool):
    """Lazy import -- wandb is only ever required when --wandb is actually
    passed, same reasoning as never importing `google.colab` unconditionally
    (see module docstring). `id=args.run_id` + `resume="allow"` makes a
    resumed Colab session (killed and restarted with the same --run-id)
    continue logging to the SAME wandb run instead of creating a new one
    each time, mirroring the checkpoint/metrics.jsonl resume convention
    everywhere else in this script."""
    import os
    import wandb

    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        config={
            "num_envs": ppo_cfg.num_envs, "num_steps": ppo_cfg.num_steps,
            "minibatch_size": ppo_cfg.minibatch_size, "num_epochs": ppo_cfg.num_epochs,
            "gamma": ppo_cfg.gamma, "lam": ppo_cfg.lam, "clip_eps": ppo_cfg.clip_eps,
            "learning_rate": ppo_cfg.learning_rate, "adv_top_frac": ppo_cfg.adv_top_frac,
            "target_kl": ppo_cfg.target_kl, "stage": stage.name,
            "grid_size": cfg.grid_size, "patch_size": cfg.patch_size, "depth": cfg.depth,
            "embed_dim": cfg.embed_dim, "n_head": cfg.n_head, "resuming": resuming,
        },
    )
    return wandb


def main():
    args = parse_args()

    ckpt_dir_root = Path(args.ckpt_dir) if args.ckpt_dir else REPO_ROOT / "training" / "runs" / args.run_id
    ckpt_dir = ckpt_dir_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir_root / "metrics.jsonl"
    status_path = REPO_ROOT / (args.status_path or f"colab/status/{args.run_id}.json")

    cfg = TransformerNetworkConfig()
    default_ppo = PPOConfig()
    ppo_cfg = PPOConfig(num_envs=args.num_envs or default_ppo.num_envs,
                         num_steps=args.num_steps or default_ppo.num_steps,
                         minibatch_size=args.minibatch_size)  # always set -- see _DEFAULT_MINIBATCH_SIZE
    print(f"run_id={args.run_id} num_envs={ppo_cfg.num_envs} num_steps={ppo_cfg.num_steps} "
          f"minibatch_size={ppo_cfg.minibatch_size} "
          f"num_epochs={ppo_cfg.num_epochs} ckpt_dir={ckpt_dir_root}")

    stage = cur.STAGE_E  # full competition ruleset -- the only stage this script trains

    key = jrandom.PRNGKey(args.seed)
    net_key, key = jrandom.split(key)
    net = TransformerPolicyValueNetwork(net_key, cfg)
    ema_net = net
    optimizer = ppo.make_optimizer(ppo_cfg.learning_rate, ppo_cfg.max_grad_norm)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    train_step = ppo.make_train_step_transformer(optimizer, stage.build_castles_enabled)

    meta_path = ckpt_dir / "train_meta.json"
    full_ckpt_path = ckpt_dir / "latest_full.eqx"

    global_iteration = 0
    resuming = not args.fresh and meta_path.exists() and full_ckpt_path.exists()
    if resuming:
        meta = json.loads(meta_path.read_text())
        global_iteration = meta["global_iteration"]
        net, opt_state, ema_net = eqx.tree_deserialise_leaves(str(full_ckpt_path), (net, opt_state, ema_net))
        print(f"resumed: global_iteration={global_iteration}")
    else:
        print("starting fresh")

    wb = init_wandb(args, ppo_cfg, cfg, stage, resuming) if args.wandb else None

    env = cur.build_env(stage)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)

    n = ppo_cfg.num_envs
    n_hunter = int(n * args.hunter_frac)
    n_expander = int(n * args.expander_frac)
    n_selfplay = n - n_hunter - n_expander
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

    shutdown_requested = {"flag": False}

    def handle_signal(signum, _frame):
        print(f"received signal {signum}; will stop after the current iteration")
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def save_checkpoint():
        eqx.tree_serialise_leaves(str(full_ckpt_path), (net, opt_state, ema_net))
        write_json(meta_path, {"global_iteration": global_iteration, "ts": time.time()})

    last_ckpt_time = time.time()
    train_start = time.time()

    try:
        while True:
            if shutdown_requested["flag"]:
                print("stopping (signal)")
                break
            if args.max_wall_clock_hours and (time.time() - train_start) / 3600 >= args.max_wall_clock_hours:
                print("reached max_wall_clock_hours, stopping")
                break

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
                    net, opt_state, train_step, flat, ek, ppo_cfg.minibatch_size,
                    ppo_cfg.clip_eps, ppo_cfg.value_coef, stage.entropy_coef, sample_idx=sample_idx,
                )
                epochs_used += 1
                if ppo_cfg.target_kl is not None and float(epoch_metrics["approx_kl"]) > ppo_cfg.target_kl:
                    break

            decay = ppo_cfg.weight_ema_decay
            ema_net = jax.tree.map(lambda e, c: decay * e + (1 - decay) * c, ema_net, net)

            global_iteration += 1
            elapsed = time.time() - iter_t0
            metrics_row = {
                "ts": time.time(), "iteration": global_iteration, "wall_clock_iter_s": elapsed,
                "sps": (ppo_cfg.num_envs * ppo_cfg.num_steps) / elapsed,
                "mean_reward": float(trajectory.reward.mean()), "buckets": bucket_stats,
                "epochs_used": epochs_used,
                **{k: float(v) for k, v in epoch_metrics.items()},
            }
            append_jsonl(metrics_path, metrics_row)
            if wb:
                wb.log(metrics_row, step=global_iteration)
            print(f"iter {global_iteration} loss={epoch_metrics['loss']:.4f} entropy={epoch_metrics['entropy']:.3f} "
                  f"kl={epoch_metrics['approx_kl']:.4f} reward={metrics_row['mean_reward']:+.4f} "
                  f"buckets={bucket_stats} sps={metrics_row['sps']:.0f} time={elapsed:.1f}s")

            due_by_iters = global_iteration % args.ckpt_every_iters == 0
            due_by_minutes = (time.time() - last_ckpt_time) / 60 >= args.ckpt_every_minutes
            if due_by_iters or due_by_minutes:
                save_checkpoint()
                last_ckpt_time = time.time()
                print(f"  checkpoint saved at iter {global_iteration}")

            if args.push_status and global_iteration % args.status_every_iters == 0:
                push_status(status_path, {
                    "run_id": args.run_id, "iteration": global_iteration, "ts": time.time(),
                    "status": "running", "recent_metrics": metrics_row,
                }, args.github_pat, args.git_branch)

    except Exception as e:
        save_checkpoint()
        if args.push_status:
            push_status(status_path, {
                "run_id": args.run_id, "iteration": global_iteration, "ts": time.time(),
                "status": f"crashed: {e!r}",
            }, args.github_pat, args.git_branch)
        if wb:
            wb.finish(exit_code=1)
        raise

    save_checkpoint()
    if args.push_status:
        push_status(status_path, {
            "run_id": args.run_id, "iteration": global_iteration, "ts": time.time(),
            "status": "stopped_ok",
        }, args.github_pat, args.git_branch)
    if wb:
        wb.finish()
    print(f"stopped cleanly at iter {global_iteration}")


if __name__ == "__main__":
    main()
