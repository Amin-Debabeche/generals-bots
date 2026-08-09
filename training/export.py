"""Export a trained JAX/equinox checkpoint to the pure-NumPy weights the
submitted bot (competition/agents/my_bot/) actually runs on, with a mandatory
parity check against the real wire protocol before anything is trusted.

Pipeline:
  1. Load the checkpoint (a league snapshot / pretrain_bc.py output -- a
     network-only .eqx -- or a full (net, opt_state) training checkpoint;
     tries network-only first, falls back to the tuple form).
  2. Dump every relevant leaf to weights.npz (explicit float32) and write
     weights_meta.json (fixed geometry + the exact normalization constants
     from training/config.py -- so competition/agents/my_bot/agent.py reads
     these rather than risking hand-copied literals drifting out of sync).
  3. Generate diverse competition-ruleset boards, round-trip each Observation
     through the REAL wire protocol (competition/protocol.py's encoder ->
     competition/agents/my_bot/main.py's parser -- the exact bytes a real
     bot process would receive), run both the JAX reference path
     (training/action_space.py + training/network.py) and the exported
     NumPy path (competition/agents/my_bot/numpy_infer.py) on the identical
     parsed input, and assert they agree. This is the single most important
     correctness gate: two independent implementations, only trustworthy if
     checked, never by inspection.

Usage:
    python -m training.export --checkpoint training/runs/main4/checkpoints/league/D_castles_iter100.eqx \\
        --out competition/agents/my_bot --build-castles
"""
import argparse
import io
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MY_BOT_DIR = REPO_ROOT / "competition" / "agents" / "my_bot"
sys.path.insert(0, str(MY_BOT_DIR))
sys.path.insert(0, str(REPO_ROOT / "competition"))
import numpy_infer as ninf  # noqa: E402 -- the self-contained module that ships with the bot
import protocol as wire_protocol  # noqa: E402 -- competition/protocol.py
import main as bot_main  # noqa: E402 -- competition/agents/my_bot/main.py (frame parsing)

from generals import GeneralsEnv  # noqa: E402
from generals.core import game  # noqa: E402
from generals.core.action import sample_valid_action  # noqa: E402
from training import action_space as asp  # noqa: E402
from training import curriculum as cur  # noqa: E402
from training import memory as mem_mod  # noqa: E402
from training.config import NetworkConfig, NORMALIZATION, PPOConfig  # noqa: E402
from training.network import PolicyValueNetwork  # noqa: E402
from training.train import build_optimizer_and_state  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .eqx checkpoint (network-only or full)")
    p.add_argument("--out", default=str(MY_BOT_DIR), help="output directory (default: competition/agents/my_bot)")
    p.add_argument("--build-castles", action="store_true",
                    help="whether the exported agent should treat build actions as legal "
                         "(true for Stage D/E checkpoints; the real competition ruleset always has this on)")
    p.add_argument("--parity-boards", type=int, default=200)
    p.add_argument("--parity-tolerance", type=float, default=1e-3)
    p.add_argument("--parity-min-agreement", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def load_net(checkpoint_path: str, net_cfg: NetworkConfig) -> PolicyValueNetwork:
    template = PolicyValueNetwork(jrandom.PRNGKey(0), net_cfg)
    try:
        return eqx.tree_deserialise_leaves(checkpoint_path, template)
    except Exception:
        pass
    optimizer, opt_state = build_optimizer_and_state(template, PPOConfig())
    net, _ = eqx.tree_deserialise_leaves(checkpoint_path, (template, opt_state))
    return net


def extract_weights(net: PolicyValueNetwork) -> dict:
    w = {}
    for i, conv in enumerate(net.convs):
        w[f"conv{i + 1}_w"] = np.asarray(conv.weight, dtype=np.float32)
        w[f"conv{i + 1}_b"] = np.asarray(conv.bias, dtype=np.float32)
    w["policy_conv_w"] = np.asarray(net.policy_conv.weight, dtype=np.float32)
    w["policy_conv_b"] = np.asarray(net.policy_conv.bias, dtype=np.float32)
    w["pass_w"] = np.asarray(net.pass_linear.weight, dtype=np.float32)
    w["pass_b"] = np.asarray(net.pass_linear.bias, dtype=np.float32)
    return w


def make_meta(net_cfg: NetworkConfig) -> dict:
    return {
        "grid_size": net_cfg.grid_size,
        "in_channels": net_cfg.in_channels,
        "num_cell_actions": net_cfg.num_cell_actions,
        "normalization": {
            "log1p_channels": list(NORMALIZATION["log1p_channels"]),
            "land_scale_channels": list(NORMALIZATION["land_scale_channels"]),
            "land_scale": NORMALIZATION["land_scale"],
            "timestep_channel": NORMALIZATION["timestep_channel"],
            "timestep_scale": NORMALIZATION["timestep_scale"],
        },
    }


def _obs_through_wire(obs) -> "bot_main.Observation":
    """Round-trip a JAX Observation through the REAL wire protocol: encode
    exactly as the engine would, parse exactly as the submitted bot's main.py
    would. Returns main.py's Observation (plain Python lists/ints)."""
    encoded = wire_protocol.encode_observation(obs)
    H, W = obs.armies.shape
    sio = io.StringIO(encoded)
    scalars_line = sio.readline()
    return bot_main._read_observation(sio, H, W, scalars_line)


def jax_reference(net, obs, mem, build_castles_enabled: bool):
    tensor = asp.obs_to_network_input_with_memory(obs, mem)
    logits, _value = net(tensor)
    mask = asp.compute_legal_action_mask(obs, build_castles_enabled)
    logits_np = np.asarray(logits, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=bool)
    penalized = np.where(mask_np, logits_np, logits_np - 1e9)
    return logits_np, mask_np, int(np.argmax(penalized))


def numpy_candidate(weights, meta, wire_obs, mem_np: dict, build_castles_enabled: bool):
    G = meta["grid_size"]
    raw = ninf.build_obs_tensor(
        wire_obs.type_grid, wire_obs.owner_grid, wire_obs.army_grid,
        wire_obs.my_land, wire_obs.my_army, wire_obs.opp_land, wire_obs.opp_army,
        wire_obs.turn, G,
    )
    mask_np = ninf.compute_legal_action_mask(
        raw[5], raw[0], raw[3], raw[2], raw[1], build_castles_enabled,
    )
    norm = ninf.normalize_obs_tensor(raw.copy(), meta)
    mem_channels = ninf.memory_to_channels(mem_np)
    network_input = np.concatenate([norm, mem_channels], axis=0)
    logits_np = ninf.forward_policy_logits(weights, network_input)
    idx = ninf.greedy_masked_argmax(logits_np, mask_np)
    return logits_np, mask_np, idx


def run_parity_check(net, weights, meta, build_castles_enabled: bool,
                      num_boards: int, tolerance: float, min_agreement: float, seed: int):
    print(f"\nRunning parity check on {num_boards} boards "
          f"(build_castles_enabled={build_castles_enabled})...")
    env = GeneralsEnv(mode="competition")
    key = jrandom.PRNGKey(seed)

    agreements = 0
    max_logit_diff = 0.0
    mask_mismatches = 0
    examples_checked = 0

    for i in range(num_boards):
        key, board_key, step_key = jrandom.split(key, 3)
        # Sample a variety of board sizes and game phases: play a random
        # number of no-op-ish steps first so armies/land aren't all turn-0.
        h = int(jrandom.randint(board_key, (), 18, 22))
        w = int(jrandom.randint(jrandom.fold_in(board_key, 1), (), 18, 22))
        from generals.core.grid import generate_grid
        from generals.modifiers import build_castles as _bc
        grid = generate_grid(
            board_key, grid_dims=(h, w), pad_to=env.pad_to,
            mountain_density_range=env.mountain_density_range,
            num_castles_range=env.num_castles_range,
            min_generals_distance=env.min_generals_distance,
            castle_val_range=env.castle_val_range,
        )
        grid = _bc.strip_neutral_castles(grid)
        state = game.create_initial_state(grid.astype(jnp.int32))

        G = meta["grid_size"]
        mem_j = [mem_mod.init_memory(G, G), mem_mod.init_memory(G, G)]
        mem_n = [ninf.init_memory(G), ninf.init_memory(G)]

        num_prelim_steps = int(jrandom.randint(step_key, (), 0, 60))
        for t in range(num_prelim_steps):
            k = jrandom.fold_in(step_key, t)
            k0, k1 = jrandom.split(k)
            # Occasionally take a REAL legal move (not just pass) so the
            # move-trail memory channels actually get exercised by this check
            # -- an all-pass prelim would leave own_move_age/opp_activity_age
            # permanently at their "never touched" sentinel, untested.
            obs0_pre = game.get_observation(state, 0)
            obs1_pre = game.get_observation(state, 1)
            a0 = sample_valid_action(k0, obs0_pre) if t % 3 == 0 else jnp.array([1, 0, 0, 0, 0])
            a1 = sample_valid_action(k1, obs1_pre) if t % 3 == 1 else jnp.array([1, 0, 0, 0, 0])
            state, _actions = _bc.apply_build_actions(state, jnp.stack([a0, a1]))
            state, _info = game.step(state, jnp.stack([a0, a1]))

            for p, a in ((0, a0), (1, a1)):
                obs_after = game.get_observation(state, p)
                mem_j[p] = mem_mod.update_memory(mem_j[p], obs_after, a)
                wire_after = _obs_through_wire(obs_after)
                raw_after = ninf.build_obs_tensor(
                    wire_after.type_grid, wire_after.owner_grid, wire_after.army_grid,
                    wire_after.my_land, wire_after.my_army, wire_after.opp_land, wire_after.opp_army,
                    wire_after.turn, G,
                )
                mem_n[p] = ninf.update_memory(mem_n[p], raw_after, [int(x) for x in a])

        for player in (0, 1):
            obs = game.get_observation(state, player)
            wire_obs = _obs_through_wire(obs)

            _logits_j, mask_j, idx_j = jax_reference(net, obs, mem_j[player], build_castles_enabled)
            _logits_n, mask_n, idx_n = numpy_candidate(weights, meta, wire_obs, mem_n[player], build_castles_enabled)

            if not np.array_equal(mask_j, mask_n):
                mask_mismatches += 1
            diff = float(np.max(np.abs(_logits_j - _logits_n)))
            max_logit_diff = max(max_logit_diff, diff)
            if idx_j == idx_n:
                agreements += 1
            examples_checked += 1

    agreement_rate = agreements / examples_checked
    print(f"  boards checked: {num_boards} ({examples_checked} (board,seat) examples)")
    print(f"  mask mismatches: {mask_mismatches}")
    print(f"  max abs logit diff: {max_logit_diff:.6f} (tolerance {tolerance})")
    print(f"  argmax agreement: {agreement_rate:.4f} (minimum required {min_agreement})")

    ok = (mask_mismatches == 0) and (max_logit_diff < tolerance) and (agreement_rate >= min_agreement)
    return ok, dict(agreement_rate=agreement_rate, max_logit_diff=max_logit_diff, mask_mismatches=mask_mismatches)


def main():
    args = parse_args()
    net_cfg = NetworkConfig()

    print(f"Loading checkpoint: {args.checkpoint}")
    net = load_net(args.checkpoint, net_cfg)

    weights = extract_weights(net)
    print("Extracted weight shapes:")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")

    meta = make_meta(net_cfg)

    ok, stats = run_parity_check(
        net, weights, meta, args.build_castles,
        args.parity_boards, args.parity_tolerance, args.parity_min_agreement, args.seed,
    )
    if not ok:
        print("\nPARITY CHECK FAILED -- refusing to write weights. See stats above.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    meta_path = out_dir / "weights_meta.json"
    np.savez(weights_path, **weights)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nPARITY CHECK PASSED ({stats['agreement_rate']:.1%} agreement, "
          f"max logit diff {stats['max_logit_diff']:.6f})")
    print(f"Wrote {weights_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
