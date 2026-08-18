"""Parity + latency check for the numpy transformer forward pass
(competition/agents/my_bot/numpy_infer_transformer.py) against the JAX
reference (training/network_transformer.py + training/augment.py) --
Step 4 of ~/.claude/plans/drifting-questing-babbage.md, flagged there as the
single biggest unaddressed risk in adopting a transformer architecture at
all: a real numpy attention forward pass, under the competition sandbox's
per-move latency budget, is unproven until measured.

Uses a RANDOM-initialized network -- not wired into training/export.py's CLI
(no trained checkpoint exists yet to export). The question this answers is
purely "does the numpy port compute the same function as the JAX one, and is
it fast enough", both true/false independent of what the weights are.

Run directly: python -m training.test_export_transformer
"""
import io
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MY_BOT_DIR = REPO_ROOT / "competition" / "agents" / "my_bot"
sys.path.insert(0, str(MY_BOT_DIR))
sys.path.insert(0, str(REPO_ROOT / "competition"))
import numpy_infer_transformer as ninf_t  # noqa: E402 -- self-contained module shipped with the bot
import protocol as wire_protocol  # noqa: E402 -- competition/protocol.py
import main as bot_main  # noqa: E402 -- competition/agents/my_bot/main.py (frame parsing)

from generals import GeneralsEnv  # noqa: E402
from generals.agents.random_agent import RandomAgent  # noqa: E402

from training import action_space as asp  # noqa: E402
from training import augment as aug_mod  # noqa: E402
from training.config import TransformerNetworkConfig  # noqa: E402
from training.network_transformer import TransformerPolicyValueNetwork  # noqa: E402
from training.rollout import _get_obs_fn  # noqa: E402


def _obs_through_wire(obs):
    """Round-trip a JAX Observation through the REAL wire protocol -- same
    helper as training/export.py's _obs_through_wire (duplicated rather than
    imported: export.py does heavy module-level setup for the CNN path we
    don't want to trigger here)."""
    encoded = wire_protocol.encode_observation(obs)
    H, W = obs.armies.shape
    sio = io.StringIO(encoded)
    scalars_line = sio.readline()
    return bot_main._read_observation(sio, H, W, scalars_line)


def extract_weights_transformer(net: TransformerPolicyValueNetwork, cfg: TransformerNetworkConfig):
    """Pulls every leaf numpy_infer_transformer.forward_policy_value needs
    out of a live TransformerPolicyValueNetwork. Not yet folded into
    training/export.py's CLI (see module docstring) -- when it is, this
    function is what moves there, alongside a weights.npz/weights_meta.json
    write and the CNN path's existing parity-gate-before-write discipline."""
    w = {}
    w["embedder_w"] = np.asarray(net.embedder.weight, dtype=np.float32)
    w["embedder_b"] = np.asarray(net.embedder.bias, dtype=np.float32)
    w["value_token"] = np.asarray(net.value_token, dtype=np.float32)
    w["pos_encoding"] = np.asarray(net.pos_encoding, dtype=np.float32)
    for i, layer in enumerate(net.transformer_layers):
        prefix = f"layer{i}"
        w[f"{prefix}_norm1_w"] = np.asarray(layer.norm1.weight, dtype=np.float32)
        w[f"{prefix}_norm1_b"] = np.asarray(layer.norm1.bias, dtype=np.float32)
        w[f"{prefix}_q_w"] = np.asarray(layer.attn.q_proj.weight, dtype=np.float32)
        w[f"{prefix}_q_b"] = np.asarray(layer.attn.q_proj.bias, dtype=np.float32)
        w[f"{prefix}_k_w"] = np.asarray(layer.attn.k_proj.weight, dtype=np.float32)
        w[f"{prefix}_k_b"] = np.asarray(layer.attn.k_proj.bias, dtype=np.float32)
        w[f"{prefix}_v_w"] = np.asarray(layer.attn.v_proj.weight, dtype=np.float32)
        w[f"{prefix}_v_b"] = np.asarray(layer.attn.v_proj.bias, dtype=np.float32)
        w[f"{prefix}_out_w"] = np.asarray(layer.attn.out_proj.weight, dtype=np.float32)
        w[f"{prefix}_out_b"] = np.asarray(layer.attn.out_proj.bias, dtype=np.float32)
        w[f"{prefix}_norm2_w"] = np.asarray(layer.norm2.weight, dtype=np.float32)
        w[f"{prefix}_norm2_b"] = np.asarray(layer.norm2.bias, dtype=np.float32)
        w[f"{prefix}_ff1_w"] = np.asarray(layer.ff_linear1.weight, dtype=np.float32)
        w[f"{prefix}_ff1_b"] = np.asarray(layer.ff_linear1.bias, dtype=np.float32)
        w[f"{prefix}_ff2_w"] = np.asarray(layer.ff_linear2.weight, dtype=np.float32)
        w[f"{prefix}_ff2_b"] = np.asarray(layer.ff_linear2.bias, dtype=np.float32)
    w["norm_out_w"] = np.asarray(net.norm_out.weight, dtype=np.float32)
    w["norm_out_b"] = np.asarray(net.norm_out.bias, dtype=np.float32)
    w["policy_head_w"] = np.asarray(net.policy_head.weight, dtype=np.float32)
    w["policy_head_b"] = np.asarray(net.policy_head.bias, dtype=np.float32)
    w["pass_head_w"] = np.asarray(net.pass_head.weight, dtype=np.float32)
    w["pass_head_b"] = np.asarray(net.pass_head.bias, dtype=np.float32)
    w["value_linear1_w"] = np.asarray(net.value_linear1.weight, dtype=np.float32)
    w["value_linear1_b"] = np.asarray(net.value_linear1.bias, dtype=np.float32)
    w["value_linear2_w"] = np.asarray(net.value_linear2.weight, dtype=np.float32)
    w["value_linear2_b"] = np.asarray(net.value_linear2.bias, dtype=np.float32)
    w["temporal_army_l1_w"] = np.asarray(net.temporal_encoder.army_l1.weight, dtype=np.float32)
    w["temporal_army_l1_b"] = np.asarray(net.temporal_encoder.army_l1.bias, dtype=np.float32)
    w["temporal_army_l2_w"] = np.asarray(net.temporal_encoder.army_l2.weight, dtype=np.float32)
    w["temporal_army_l2_b"] = np.asarray(net.temporal_encoder.army_l2.bias, dtype=np.float32)
    w["temporal_land_l1_w"] = np.asarray(net.temporal_encoder.land_l1.weight, dtype=np.float32)
    w["temporal_land_l1_b"] = np.asarray(net.temporal_encoder.land_l1.bias, dtype=np.float32)
    w["temporal_land_l2_w"] = np.asarray(net.temporal_encoder.land_l2.weight, dtype=np.float32)
    w["temporal_land_l2_b"] = np.asarray(net.temporal_encoder.land_l2.bias, dtype=np.float32)
    w["temporal_type_embed"] = np.asarray(net.temporal_type_embed, dtype=np.float32)

    meta = {
        "grid_size": cfg.grid_size, "patch_size": cfg.patch_size, "n_channels": cfg.n_channels,
        "num_cell_actions": cfg.num_cell_actions, "depth": cfg.depth, "n_head": cfg.n_head,
        "history_size": cfg.history_size, "temporal_window": cfg.temporal_window,
    }
    return w, meta


def test_parity_and_latency():
    cfg = TransformerNetworkConfig()
    key = jrandom.PRNGKey(0)
    key, net_key = jrandom.split(key)
    net = TransformerPolicyValueNetwork(net_key, cfg)
    weights, meta = extract_weights_transformer(net, cfg)

    env = GeneralsEnv(mode="competition")
    get_obs = _get_obs_fn(env)
    key, pool_key, state_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    state = env.init_state(state_key)
    agent = RandomAgent()

    obs_state_j = [aug_mod.init_obs_state(cfg), aug_mod.init_obs_state(cfg)]
    obs_state_n = [
        ninf_t.init_obs_state(cfg.grid_size, cfg.history_size, cfg.temporal_window),
        ninf_t.init_obs_state(cfg.grid_size, cfg.history_size, cfg.temporal_window),
    ]

    agreements = 0
    total = 0
    mask_mismatches = 0
    max_logit_diff = 0.0
    latencies = []
    actions_by_seat = {}

    for t in range(60):
        key, k0, k1 = jrandom.split(key, 3)
        for seat, k in ((0, k0), (1, k1)):
            obs = get_obs(state, seat)
            wire_obs = _obs_through_wire(obs)

            augmented_j, obs_state_j[seat] = aug_mod.augment_obs(obs, obs_state_j[seat])
            normed_j = aug_mod.normalize_augmented(augmented_j)
            temporal_j = aug_mod.temporal_data(obs_state_j[seat])
            logits_j, _value_j = net(normed_j, temporal_j)
            mask_j = asp.compute_legal_action_mask(obs, True)
            logits_j_np = np.asarray(logits_j, dtype=np.float32)
            mask_j_np = np.asarray(mask_j, dtype=bool)
            idx_j = int(np.argmax(np.where(mask_j_np, logits_j_np, logits_j_np - 1e9)))

            raw_n = ninf_t.build_obs_tensor(
                wire_obs.type_grid, wire_obs.owner_grid, wire_obs.army_grid,
                wire_obs.my_land, wire_obs.my_army, wire_obs.opp_land, wire_obs.opp_army,
                wire_obs.turn, cfg.grid_size,
            )
            augmented_n, obs_state_n[seat] = ninf_t.augment_obs(raw_n, obs_state_n[seat])
            normed_n = ninf_t.normalize_augmented(augmented_n)
            temporal_n = ninf_t.temporal_data(obs_state_n[seat])

            t0 = time.perf_counter()
            logits_n, _value_n = ninf_t.forward_policy_value(weights, meta, normed_n, temporal_n)
            latencies.append(time.perf_counter() - t0)

            mask_n = ninf_t.compute_legal_action_mask(raw_n[5], raw_n[0], raw_n[3], raw_n[2], raw_n[1], True)
            idx_n = ninf_t.greedy_masked_argmax(logits_n, mask_n)

            if not np.array_equal(mask_j_np, mask_n):
                mask_mismatches += 1
            diff = float(np.max(np.abs(logits_j_np - logits_n)))
            max_logit_diff = max(max_logit_diff, diff)
            if idx_j == idx_n:
                agreements += 1
            total += 1

            actions_by_seat[seat] = agent.act(obs, k)

        actions = np.stack([np.asarray(actions_by_seat[0]), np.asarray(actions_by_seat[1])])
        timestep, state = env.step(state, jnp.asarray(actions), pool)
        if bool(timestep.terminated | timestep.truncated):
            key, reset_key = jrandom.split(key)
            state = env.init_state(reset_key)
            obs_state_j = [aug_mod.init_obs_state(cfg), aug_mod.init_obs_state(cfg)]
            obs_state_n = [
                ninf_t.init_obs_state(cfg.grid_size, cfg.history_size, cfg.temporal_window),
                ninf_t.init_obs_state(cfg.grid_size, cfg.history_size, cfg.temporal_window),
            ]

    agreement_rate = agreements / total
    mean_latency_ms = 1000 * float(np.mean(latencies))
    p95_latency_ms = 1000 * float(np.percentile(latencies, 95))
    max_latency_ms = 1000 * float(np.max(latencies))

    print(f"  (board,seat,turn) samples checked: {total}")
    print(f"  mask mismatches: {mask_mismatches}")
    print(f"  max abs logit diff: {max_logit_diff:.6f}")
    print(f"  argmax agreement: {agreement_rate:.4f}")
    print(f"  numpy forward latency: mean={mean_latency_ms:.2f}ms p95={p95_latency_ms:.2f}ms max={max_latency_ms:.2f}ms")

    assert mask_mismatches == 0, f"{mask_mismatches} legal-mask mismatches between JAX and numpy paths"
    assert agreement_rate >= 0.98, f"argmax agreement {agreement_rate:.4f} below 0.98 threshold"
    # Measured empirically at ~2e-6 (float32 accumulation noise across a
    # 4-layer attention stack) -- matches the CNN path's own 1e-3 tolerance
    # (numpy_infer.py/export.py) comfortably despite the deeper/more
    # matmul-heavy architecture, so there's no reason to loosen it here.
    assert max_logit_diff < 1e-3, f"max logit diff {max_logit_diff:.6f} too large -- likely a real bug, not float noise"
    print("PARITY CHECK PASSED")


if __name__ == "__main__":
    test_parity_and_latency()
