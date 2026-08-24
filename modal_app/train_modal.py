"""Modal-hosted single-stage PPO training for the transformer network --
the Modal-backend twin of colab/train_colab.py, sharing the exact same
per-iteration logic via training/transformer_train_loop.py.

Why Modal, not Colab: Colab requires a human in a browser tab to mount
Drive, load secrets, and re-run cells by hand every time a fix lands or a
session disconnects -- multiple real incidents this project hit were
literally "the fix was pushed but the open notebook tab never picked it up
because git-pulling files doesn't rewrite already-open cells." Modal
sidesteps that whole class of problem: this file *is* the source of truth
(no separate notebook copy to go stale), `modal run`/`modal deploy` builds
the container image directly from the local repo (no git clone/PAT/pull
inside the container at all), and it's invokable straight from a shell --
no browser step required once `modal setup` has been run once.

Named modal_app/, not modal/, specifically so this file's own `import
modal` resolves to the real pip-installed package rather than shadowing
itself with a same-named local directory.

Checkpoints/metrics live on a modal.Volume (`generals-bots-runs`), the
Modal equivalent of Colab's Drive mount -- persists across container
restarts, inspectable directly via `modal volume ls/get` without needing
the training run itself to still be alive.

Usage:
    modal run modal_app/train_modal.py --run-id modal1
    modal run modal_app/train_modal.py --run-id modal1 --fresh --num-envs 128
    modal run --detach modal_app/train_modal.py --run-id modal1   # keeps running after this shell exits
"""
import json
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("generals-bots-transformer-training")

# T4 is Modal's cheapest GPU tier -- the whole point of the aggressive
# num_envs/minibatch_size defaults below (already battle-tested against a
# real ~15GB-class GPU on Colab) is to fit comfortably on exactly this tier
# rather than needing anything pricier.
GPU = "T4"

VOLUME_NAME = "generals-bots-runs"
VOL_MOUNT = "/vol"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Everything the container needs to actually run training -- mirrors
# pyproject.toml's base + [training] extra exactly (not [modal]/[colab],
# which are for the machine invoking `modal run`, not the remote container).
# jax[cuda12], not bare jax/jaxlib: the pinned CPU-only jaxlib from
# pyproject.toml's base dependencies has no CUDA kernels at all.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=2.0.0",
        "pygame>=2.6.0",
        "python-socketio[client]>=5.11.4",
        "equinox>=0.13.8",
        "optax>=0.2.8",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "requests>=2.25.0",
    )
    .pip_install("jax[cuda12]")
    .env({
        # Same fragmentation mitigation as colab/train_colab.py, set here
        # (a container-level env var, applied before any Python even starts)
        # instead of os.environ.setdefault at import time -- Modal gives us
        # a cleaner place to guarantee this than "hope nothing imports jax
        # first", so use it.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    .add_local_dir(
        str(REPO_ROOT), "/root/generals-bots",
        ignore=[
            ".git", ".venv", ".claude", "__pycache__", "*.egg-info", "*.pyc",
            "training/runs", "modal_app/__pycache__",
            "competition/agents/my_bot/weights.npz",
        ],
    )
)


def _load_checkpoint_if_resuming(ckpt_dir: Path, fresh: bool, state):
    import equinox as eqx
    meta_path = ckpt_dir / "train_meta.json"
    full_ckpt_path = ckpt_dir / "latest_full.eqx"
    resuming = not fresh and meta_path.exists() and full_ckpt_path.exists()
    if resuming:
        meta = json.loads(meta_path.read_text())
        net, opt_state, ema_net = eqx.tree_deserialise_leaves(
            str(full_ckpt_path), (state.net, state.opt_state, state.ema_net)
        )
        state = state._replace(net=net, opt_state=opt_state, ema_net=ema_net,
                                global_iteration=meta["global_iteration"])
        print(f"resumed: global_iteration={state.global_iteration}")
    else:
        print("starting fresh")
    return state, resuming


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL_MOUNT: volume},
    timeout=23 * 3600,  # just under Modal's per-invocation ceiling; the loop
                        # stops itself cleanly at max_wall_clock_hours first
)
def train(run_id: str, num_envs: int = 128, num_steps: int = 256, minibatch_size: int = 128,
          hunter_frac: float = 0.15, expander_frac: float = 0.15,
          max_wall_clock_hours: float = 22.0, ckpt_every_iters: int = 20,
          ckpt_every_minutes: float = 10.0, fresh: bool = False, seed: int = 0):
    import sys
    sys.path.insert(0, "/root/generals-bots")

    import equinox as eqx
    from training.config import PPOConfig, TransformerNetworkConfig
    from training.transformer_train_loop import init_train_state, run_iteration

    ckpt_dir_root = Path(VOL_MOUNT) / run_id
    ckpt_dir = ckpt_dir_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir_root / "metrics.jsonl"
    meta_path = ckpt_dir / "train_meta.json"
    full_ckpt_path = ckpt_dir / "latest_full.eqx"

    cfg = TransformerNetworkConfig()
    ppo_cfg = PPOConfig(num_envs=num_envs, num_steps=num_steps, minibatch_size=minibatch_size)
    print(f"run_id={run_id} gpu={GPU} num_envs={ppo_cfg.num_envs} num_steps={ppo_cfg.num_steps} "
          f"minibatch_size={ppo_cfg.minibatch_size} ckpt_dir={ckpt_dir_root}")

    state = init_train_state(cfg, ppo_cfg, seed, hunter_frac, expander_frac)
    state, _resuming = _load_checkpoint_if_resuming(ckpt_dir, fresh, state)

    def save_checkpoint():
        eqx.tree_serialise_leaves(str(full_ckpt_path), (state.net, state.opt_state, state.ema_net))
        meta_path.write_text(json.dumps({"global_iteration": state.global_iteration, "ts": time.time()}))
        volume.commit()  # flush this container's writes so other containers/CLI inspection see them

    last_ckpt_time = time.time()
    train_start = time.time()

    try:
        while True:
            if (time.time() - train_start) / 3600 >= max_wall_clock_hours:
                print("reached max_wall_clock_hours, stopping")
                break

            state, metrics_row = run_iteration(state, ppo_cfg)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics_row) + "\n")
            print(f"iter {state.global_iteration} loss={metrics_row['loss']:.4f} "
                  f"entropy={metrics_row['entropy']:.3f} kl={metrics_row['approx_kl']:.4f} "
                  f"reward={metrics_row['mean_reward']:+.4f} buckets={metrics_row['buckets']} "
                  f"sps={metrics_row['sps']:.0f} time={metrics_row['wall_clock_iter_s']:.1f}s")

            due_by_iters = state.global_iteration % ckpt_every_iters == 0
            due_by_minutes = (time.time() - last_ckpt_time) / 60 >= ckpt_every_minutes
            if due_by_iters or due_by_minutes:
                save_checkpoint()
                last_ckpt_time = time.time()
                print(f"  checkpoint saved at iter {state.global_iteration}")

    except Exception:
        save_checkpoint()
        raise

    save_checkpoint()
    print(f"stopped cleanly at iter {state.global_iteration}")
    return state.global_iteration


@app.local_entrypoint()
def main(run_id: str = "modal1", num_envs: int = 128, num_steps: int = 256, minibatch_size: int = 128,
         hunter_frac: float = 0.15, expander_frac: float = 0.15,
         max_wall_clock_hours: float = 22.0, ckpt_every_iters: int = 20,
         ckpt_every_minutes: float = 10.0, fresh: bool = False, seed: int = 0):
    final_iter = train.remote(
        run_id=run_id, num_envs=num_envs, num_steps=num_steps, minibatch_size=minibatch_size,
        hunter_frac=hunter_frac, expander_frac=expander_frac,
        max_wall_clock_hours=max_wall_clock_hours, ckpt_every_iters=ckpt_every_iters,
        ckpt_every_minutes=ckpt_every_minutes, fresh=fresh, seed=seed,
    )
    print(f"finished at iteration {final_iter}")
