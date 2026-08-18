"""Pure-NumPy inference for the transformer policy-value network
(training/network_transformer.py) + observation augmentation
(training/augment.py) -- the numpy twin for the AverageJoe-architecture
restart, see ~/.claude/plans/drifting-questing-babbage.md Step 4.

NOT wired into agent.py/main.py yet -- this is the export-path PROTOTYPE the
plan flagged as the single biggest unaddressed risk in adopting a transformer
architecture at all (a real numpy attention forward pass, under the sandbox's
10s import/first-move budget, is unproven until measured). Once a trained
checkpoint exists and training/export.py grows a transformer parity-check
mode (mirroring the CNN path's `run_parity_check`), agent.py/numpy_infer.py
get swapped to call into this module instead of numpy_infer.py's CNN path --
see training/test_export_transformer.py for the current parity+latency
harness.

Deliberately imports numpy_infer.py (a SIBLING file in this same submission
directory, not `training/`) for the network-architecture-independent pieces
that don't change with the network swap: build_obs_tensor (wire-frame ->
raw (14,G,G) tensor), compute_valid_move_mask/compute_build_cost_grid/
compute_legal_action_mask (the action-space's legality rules), and
decode_action_index/greedy_masked_argmax. Reusing these avoids duplicating
~150 already-parity-checked lines; this file cannot import `training` or
`generals` (nothing outside competition/agents/my_bot/ ships with the
submission), but importing a sibling file in the same directory is fine.
"""
import json
from pathlib import Path

import numpy as np

from numpy_infer import (  # noqa: E402 -- sibling file, see module docstring
    BUILD_KIND,
    NUM_CELL_ACTIONS,
    build_obs_tensor,
    compute_build_cost_grid,
    compute_legal_action_mask,
    compute_valid_move_mask,
    decode_action_index,
    greedy_masked_argmax,
)

NUM_BASE_CHANNELS = 24  # see training/augment.py -- 24 fixed channels + 2*history_size stacks


def load_weights(weights_path, meta_path):
    npz = np.load(weights_path)
    weights = {k: npz[k].astype(np.float32) for k in npz.files}
    meta = json.loads(Path(meta_path).read_text())
    return weights, meta


# ---------------------------------------------------------------------------
# Observation augmentation -- numpy twin of training/augment.py. State is a
# plain dict (not a NamedTuple; no jax here) threaded across turns by the
# caller, same convention numpy_infer.py's init_memory/update_memory use.
# ---------------------------------------------------------------------------


def init_obs_state(grid_size: int, history_size: int, temporal_window: int) -> dict:
    G = grid_size
    return {
        "army_stack": np.zeros((history_size, G, G), dtype=np.float32),
        "enemy_stack": np.zeros((history_size, G, G), dtype=np.float32),
        "last_army": np.zeros((G, G), dtype=np.float32),
        "last_enemy_army": np.zeros((G, G), dtype=np.float32),
        "castles": np.zeros((G, G), dtype=bool),
        "generals": np.zeros((G, G), dtype=bool),
        "mountains": np.zeros((G, G), dtype=bool),
        "seen": np.zeros((G, G), dtype=bool),
        "enemy_seen": np.zeros((G, G), dtype=bool),
        "last_enemy_army_seen_value": np.zeros((G, G), dtype=np.float32),
        "last_enemy_army_seen_timestep": np.zeros((G, G), dtype=np.float32),
        "opponent_army_history": np.zeros((temporal_window,), dtype=np.float32),
        "opponent_land_history": np.zeros((temporal_window,), dtype=np.float32),
    }


def augment_obs(raw_tensor: np.ndarray, obs_state: dict) -> tuple[np.ndarray, dict]:
    """raw_tensor: (14,G,G) float32 -- numpy_infer.build_obs_tensor's RAW
    (pre-normalize) output, channel order identical to Observation.as_tensor()
    (0 armies,1 generals,2 castles,3 mountains,4 neutral,5 owned,6 opponent,
    7 fog,8 structures_in_fog,9-13 broadcast scalars). Mirrors
    training/augment.py's augment_obs exactly, including its correction to
    AverageJoe's original ("seen" = ~fog_cells & ~structures_in_fog, the
    same visibility primitive training/memory.py established, not a
    max-pooled owned_cells proxy)."""
    (armies_ch, generals_ch, castles_ch, mountains_ch,
     neutral_cells, owned_cells, opponent_cells,
     fog_cells, structures_in_fog,
     owned_land_count, owned_army_count,
     opponent_land_count, opponent_army_count, timestep_ch) = range(14)

    arr = raw_tensor.astype(np.float32)
    G = obs_state["last_army"].shape[0]

    current_army = arr[armies_ch] * arr[owned_cells]
    current_enemy_army = arr[armies_ch] * arr[opponent_cells]

    new_army_stack = np.concatenate([
        (current_army - obs_state["last_army"])[None, :, :],
        obs_state["army_stack"][:-1, :, :],
    ], axis=0)
    new_enemy_stack = np.concatenate([
        (current_enemy_army - obs_state["last_enemy_army"])[None, :, :],
        obs_state["enemy_stack"][:-1, :, :],
    ], axis=0)

    visible = (arr[fog_cells] <= 0) & (arr[structures_in_fog] <= 0)
    enemy_visible = arr[opponent_cells] > 0
    new_seen = obs_state["seen"] | visible
    new_enemy_seen = obs_state["enemy_seen"] | enemy_visible

    new_castles = obs_state["castles"] | (arr[castles_ch] > 0)
    new_generals = obs_state["generals"] | (arr[generals_ch] > 0)
    new_mountains = obs_state["mountains"] | (arr[mountains_ch] > 0)

    new_last_enemy_army_seen_value = np.where(
        current_enemy_army > 0, current_enemy_army, obs_state["last_enemy_army_seen_value"]
    )
    new_last_enemy_army_seen_timestep = np.where(
        current_enemy_army > 0, 0.0, obs_state["last_enemy_army_seen_timestep"] + 1.0
    )

    opp_army_val = arr[opponent_army_count, 0, 0]
    opp_land_val = arr[opponent_land_count, 0, 0]
    new_opponent_army_history = np.roll(obs_state["opponent_army_history"], -1)
    new_opponent_army_history[-1] = opp_army_val
    new_opponent_land_history = np.roll(obs_state["opponent_land_history"], -1)
    new_opponent_land_history[-1] = opp_land_val

    coords_x = np.broadcast_to(np.arange(G, dtype=np.float32)[None, :] / (G - 1), (G, G))
    coords_y = np.broadcast_to(np.arange(G, dtype=np.float32)[:, None] / (G - 1), (G, G))
    ones = np.ones((G, G), dtype=np.float32)

    channels = np.stack([
        arr[armies_ch],                                   # 0
        current_army,                                      # 1
        current_enemy_army,                                 # 2
        arr[armies_ch] * arr[neutral_cells],                 # 3
        new_seen.astype(np.float32),                          # 4
        new_enemy_seen.astype(np.float32),                     # 5
        new_generals.astype(np.float32),                        # 6
        new_castles.astype(np.float32),                          # 7
        new_mountains.astype(np.float32),                         # 8
        arr[neutral_cells],                                        # 9
        arr[owned_cells],                                           # 10
        arr[opponent_cells],                                         # 11
        arr[fog_cells],                                               # 12
        arr[structures_in_fog],                                        # 13
        arr[timestep_ch] * ones,                                        # 14
        (arr[timestep_ch] % 50) * ones / 50,                             # 15
        arr[owned_land_count] * ones,                                     # 16
        arr[owned_army_count] * ones,                                      # 17
        arr[opponent_land_count] * ones,                                    # 18
        arr[opponent_army_count] * ones,                                     # 19
        new_last_enemy_army_seen_value,                                      # 20
        np.log1p(new_last_enemy_army_seen_timestep) / 5.0,                    # 21
        coords_x,                                                              # 22
        coords_y,                                                               # 23
    ], axis=0).astype(np.float32)
    assert channels.shape[0] == NUM_BASE_CHANNELS

    augmented = np.concatenate([channels, new_army_stack, new_enemy_stack], axis=0)

    new_state = {
        "army_stack": new_army_stack,
        "enemy_stack": new_enemy_stack,
        "last_army": current_army,
        "last_enemy_army": current_enemy_army,
        "castles": new_castles,
        "generals": new_generals,
        "mountains": new_mountains,
        "seen": new_seen,
        "enemy_seen": new_enemy_seen,
        "last_enemy_army_seen_value": new_last_enemy_army_seen_value,
        "last_enemy_army_seen_timestep": new_last_enemy_army_seen_timestep,
        "opponent_army_history": new_opponent_army_history,
        "opponent_land_history": new_opponent_land_history,
    }
    return augmented, new_state


def normalize_augmented(obs: np.ndarray) -> np.ndarray:
    """Mirrors training/augment.py's normalize_augmented exactly (divisor 50
    on the army-scale channels; everything else passes through unchanged)."""
    obs = obs.copy()
    army_channels = [0, 1, 2, 3, 17, 19, 20] + list(range(NUM_BASE_CHANNELS, obs.shape[0]))
    obs[army_channels] /= 50.0
    obs[14] /= 50.0
    obs[[16, 18]] /= 50.0
    return obs


def temporal_data(obs_state: dict) -> np.ndarray:
    return np.stack([obs_state["opponent_army_history"], obs_state["opponent_land_history"]])


# ---------------------------------------------------------------------------
# Network forward -- numpy twin of training/network_transformer.py's
# TransformerPolicyValueNetwork.__call__.
# ---------------------------------------------------------------------------


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Row-wise over the last axis -- matches eqx.nn.LayerNorm exactly
    (population variance/ddof=0, same as jnp.var's default)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (..., in). w: (out,in) (eqx.nn.Linear's own weight layout -- no
    transpose needed at extraction time, just here at use time). b: (out,)."""
    return x @ w.T + b


def _mhsa(x: np.ndarray, weights: dict, prefix: str, n_head: int) -> np.ndarray:
    """x: (seq_len, d_model). Mirrors MultiHeadSelfAttention.__call__."""
    seq_len, d_model = x.shape
    head_dim = d_model // n_head
    q = _linear(x, weights[f"{prefix}_q_w"], weights[f"{prefix}_q_b"]).reshape(seq_len, n_head, head_dim)
    k = _linear(x, weights[f"{prefix}_k_w"], weights[f"{prefix}_k_b"]).reshape(seq_len, n_head, head_dim)
    v = _linear(x, weights[f"{prefix}_v_w"], weights[f"{prefix}_v_b"]).reshape(seq_len, n_head, head_dim)
    q = q.transpose(1, 0, 2)  # (n_head, seq_len, head_dim)
    k = k.transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)

    scale = np.sqrt(head_dim)
    attn = np.matmul(q, k.transpose(0, 2, 1)) / scale  # (n_head, seq_len, seq_len)
    attn = _softmax(attn, axis=-1)
    out = np.matmul(attn, v)  # (n_head, seq_len, head_dim)
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return _linear(out, weights[f"{prefix}_out_w"], weights[f"{prefix}_out_b"])


def _self_attention_layer(x: np.ndarray, weights: dict, prefix: str, n_head: int) -> np.ndarray:
    """Pre-norm block, mirrors SelfAttentionLayer.__call__ exactly."""
    normed1 = _layer_norm(x, weights[f"{prefix}_norm1_w"], weights[f"{prefix}_norm1_b"])
    x = x + _mhsa(normed1, weights, prefix, n_head)
    normed2 = _layer_norm(x, weights[f"{prefix}_norm2_w"], weights[f"{prefix}_norm2_b"])
    h = _silu(_linear(normed2, weights[f"{prefix}_ff1_w"], weights[f"{prefix}_ff1_b"]))
    h = _linear(h, weights[f"{prefix}_ff2_w"], weights[f"{prefix}_ff2_b"])
    return x + h


def _temporal_encoder(temporal: np.ndarray, weights: dict) -> np.ndarray:
    """temporal: (2, temporal_window) -> (2, embed_dim). Mirrors
    TemporalEncoder.__call__."""
    army_hist = temporal[0] / 50.0
    land_hist = temporal[1] / 50.0
    army_token = _linear(
        _silu(_linear(army_hist, weights["temporal_army_l1_w"], weights["temporal_army_l1_b"])),
        weights["temporal_army_l2_w"], weights["temporal_army_l2_b"],
    )
    land_token = _linear(
        _silu(_linear(land_hist, weights["temporal_land_l1_w"], weights["temporal_land_l1_b"])),
        weights["temporal_land_l2_w"], weights["temporal_land_l2_b"],
    )
    return np.stack([army_token, land_token])


def _unpatchify_policy(patch_logits: np.ndarray, gp: int, M: int, num_cell_actions: int) -> np.ndarray:
    """(n_patches, K*M*M) -> (p,p,K), CELL-MAJOR. Identical transpose to
    training/network_transformer.py's _unpatchify_policy (see that module's
    docstring point 1) -- verified against training/test_network_transformer.py's
    exhaustive decode-roundtrip test in the JAX version; this is a
    line-for-line numpy port of the same reshape/transpose, so it inherits
    that guarantee as long as the two stay textually identical."""
    p = gp * M
    x = patch_logits.reshape(gp, gp, num_cell_actions, M, M)
    x = x.transpose(0, 3, 1, 4, 2)
    return x.reshape(p, p, num_cell_actions)


def forward_policy_value(weights: dict, meta: dict, obs_tensor: np.ndarray, temporal: np.ndarray):
    """obs_tensor: (n_channels,G,G) float32, already normalized+augmented
    (augment_obs + normalize_augmented above). temporal: (2,temporal_window)
    raw opponent army/land history. Mirrors
    TransformerPolicyValueNetwork.__call__ exactly.

    Returns:
        logits: (G*G*num_cell_actions + 1,) float32 flat, cell-major + trailing pass logit.
        value: float scalar.
    """
    p = meta["grid_size"]
    M = meta["patch_size"]
    gp = p // M
    n_channels = meta["n_channels"]
    K = meta["num_cell_actions"]
    depth = meta["depth"]
    n_head = meta["n_head"]

    # Patchify: (C,p,p) -> (n_patches, C*M*M) -- identical reshape/transpose
    # to the JAX version's patchify (self-consistent with _unpatchify_policy
    # above via matching patch index i = block_row*gp + block_col).
    x = obs_tensor.reshape(n_channels, gp, M, gp, M)
    x = x.transpose(1, 3, 0, 2, 4).reshape(gp * gp, -1)
    x = _linear(x, weights["embedder_w"], weights["embedder_b"])

    temporal_tokens = _temporal_encoder(temporal, weights) + weights["temporal_type_embed"]
    x = np.concatenate([weights["value_token"], temporal_tokens, x], axis=0)
    x = x + weights["pos_encoding"]

    for i in range(depth):
        x = _self_attention_layer(x, weights, f"layer{i}", n_head)
    x = _layer_norm(x, weights["norm_out_w"], weights["norm_out_b"])

    value_embedding = x[0]
    patch_embeddings = x[3:]  # skip value + 2 temporal tokens

    patch_logits = _linear(patch_embeddings, weights["policy_head_w"], weights["policy_head_b"])
    cell_logits = _unpatchify_policy(patch_logits, gp, M, K)
    cell_logits_flat = cell_logits.reshape(p * p * K)

    pass_logit = _linear(value_embedding, weights["pass_head_w"], weights["pass_head_b"])
    logits = np.concatenate([cell_logits_flat, pass_logit]).astype(np.float32)

    v_hidden = np.maximum(_linear(value_embedding, weights["value_linear1_w"], weights["value_linear1_b"]), 0.0)
    value = float(_linear(v_hidden, weights["value_linear2_w"], weights["value_linear2_b"])[0])

    return logits, value
