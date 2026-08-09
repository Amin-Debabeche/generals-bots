"""Pure-NumPy inference for the trained policy network -- no jax, no torch, no
imports outside the standard library + numpy, so importing this costs ~nothing
against the sandbox's 10s first-move budget (the competition docs note that
`import torch` alone costs ~2.3s, the full JAX stack ~3s).

This is a hand-transliterated NumPy twin of training/network.py's forward pass
and training/action_space.py's action encoding/masking/normalization --
training/export.py validates the two agree (argmax-under-identical-masking,
on many diverse boards) before ever writing weights.npz. Nothing outside this
directory ships with the submission, so this file cannot import `training` or
`generals` -- if the network architecture or action encoding in those modules
changes, this file's constants and math must be updated to match by hand, and
export.py's parity check re-run to confirm.
"""
import json
from pathlib import Path

import numpy as np

NUM_CELL_ACTIONS = 9
BUILD_KIND = 8
BUILD_BASE_COST = 35
BUILD_PROXIMITY_PENALTY = 14
BUILD_PROXIMITY_DECAY = 2
BUILD_RADIUS = (BUILD_PROXIMITY_PENALTY - 1) // BUILD_PROXIMITY_DECAY  # 6

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right -- matches generals/core/config.py

# Memory features (training/memory.py) -- a NumPy twin, since this directory
# can't import training/. See that module's docstring for the design
# rationale (per arXiv:2507.06825's "memory features"): 5 channels appended
# after the 14 base Observation channels, persisted as instance state on the
# Agent across turns within one game (see agent.py).
NUM_MEMORY_CHANNELS = 5
MOVE_HISTORY_WINDOW = 7
_MEM_AGE_SENTINEL = 999


def load_weights(weights_path, meta_path):
    npz = np.load(weights_path)
    weights = {k: npz[k].astype(np.float32) for k in npz.files}
    meta = json.loads(Path(meta_path).read_text())
    return weights, meta


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int) -> np.ndarray:
    """x: (Cin,H,W). weight: (Cout,Cin,KH,KW) (OIHW, matching eqx.nn.Conv2d).
    bias: (Cout,) or (Cout,1,1). Explicit zero-padding (never rely on a
    library default) to match eqx.nn.Conv2d(..., padding=padding)."""
    Cout, Cin, KH, KW = weight.shape
    _, H, W = x.shape
    if padding > 0:
        xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant", constant_values=0.0)
    else:
        xp = x
    # im2col: column order is (c outer, kh, kw inner), matching weight.reshape(Cout,-1)'s
    # default row-major flattening of a (Cout,Cin,KH,KW) array -- verified in export.py's parity check.
    cols = np.empty((Cin * KH * KW, H * W), dtype=np.float32)
    idx = 0
    for c in range(Cin):
        for kh in range(KH):
            for kw in range(KW):
                cols[idx] = xp[c, kh:kh + H, kw:kw + W].reshape(-1)
                idx += 1
    w_mat = weight.reshape(Cout, Cin * KH * KW)
    out = (w_mat @ cols).reshape(Cout, H, W)
    out = out + bias.reshape(Cout, 1, 1)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward_policy_logits(weights: dict, obs_tensor: np.ndarray) -> np.ndarray:
    """obs_tensor: (14,G,G) float32, already normalized (see normalize_obs_tensor).
    Returns flat logits of length G*G*9+1: per-cell blocks ordered
    [row][col][kind] (kind 0-3 move all-in, 4-7 move half, 8 build), with the
    single global pass logit appended last. Mirrors training/network.py's
    PolicyValueNetwork.__call__ exactly (value head omitted -- not needed for
    greedy action selection)."""
    h = obs_tensor
    for i in range(5):
        h = _conv2d(h, weights[f"conv{i + 1}_w"], weights[f"conv{i + 1}_b"], padding=1)
        h = _relu(h)
    # h: (48,G,G) backbone features

    cell_logits = _conv2d(h, weights["policy_conv_w"], weights["policy_conv_b"], padding=0)  # (9,G,G)
    cell_logits = np.transpose(cell_logits, (1, 2, 0))  # (G,G,9)
    cell_logits_flat = cell_logits.reshape(-1)

    pooled = h.mean(axis=(1, 2))  # (48,) global average pool
    pass_logit = weights["pass_w"] @ pooled + weights["pass_b"]  # (1,)

    return np.concatenate([cell_logits_flat, pass_logit])


def normalize_obs_tensor(tensor: np.ndarray, meta: dict) -> np.ndarray:
    """In-place-ish normalize (operates on and returns `tensor`; pass a copy
    if the raw tensor is still needed, e.g. for legal-mask computation)."""
    tensor = tensor.astype(np.float32)
    norm = meta["normalization"]
    for c in norm["log1p_channels"]:
        tensor[c] = np.log1p(tensor[c])
    for c in norm["land_scale_channels"]:
        tensor[c] = tensor[c] / norm["land_scale"]
    ts = norm["timestep_channel"]
    tensor[ts] = tensor[ts] / norm["timestep_scale"]
    return tensor


def build_obs_tensor(type_grid, owner_grid, army_grid, my_land, my_army, opp_land, opp_army,
                      turn, grid_size: int) -> np.ndarray:
    """Reconstruct the raw (14,G,G) tensor -- matching Observation.as_tensor()'s
    channel order exactly -- from the wire-protocol frame (competition/protocol.py),
    embedding the true HxW board at the TOP-LEFT of the fixed GxG canvas
    (G=21 in training; real competition boards are always <=21, and
    generals/core/grid.py's generate_grid pads bottom/right with mountains,
    which top-left placement mirrors).

    Padding-region behavior was EMPIRICALLY verified against the training
    observation builder (generals/core/game.get_observation), not assumed:
    every channel is exactly 0/False in the padding region EXCEPT
    structures_in_fog, which is True there -- NOT `mountains`. Getting this
    backwards would matter: the legal-move mask only checks `mountains`, so a
    wrong choice here would make this agent's mask stricter or looser than
    what the network actually trained under.
    """
    H, W = len(type_grid), len(type_grid[0])
    G = grid_size
    type_arr = np.array(type_grid, dtype=np.int32)
    owner_arr = np.array(owner_grid, dtype=np.int32)
    army_arr = np.array(army_grid, dtype=np.float32)

    armies = np.zeros((G, G), dtype=np.float32)
    armies[:H, :W] = army_arr

    generals = np.zeros((G, G), dtype=np.float32)
    generals[:H, :W] = (type_arr == 4)

    castles = np.zeros((G, G), dtype=np.float32)
    castles[:H, :W] = (type_arr == 3)

    mountains = np.zeros((G, G), dtype=np.float32)
    mountains[:H, :W] = (type_arr == 2)

    fog_cells = np.zeros((G, G), dtype=np.float32)
    fog_cells[:H, :W] = (type_arr == 0)

    structures_in_fog = np.zeros((G, G), dtype=np.float32)
    structures_in_fog[:H, :W] = (type_arr == 5)
    structures_in_fog[H:, :] = 1.0
    structures_in_fog[:, W:] = 1.0

    owned_cells = np.zeros((G, G), dtype=np.float32)
    owned_cells[:H, :W] = (owner_arr == 1)

    opponent_cells = np.zeros((G, G), dtype=np.float32)
    opponent_cells[:H, :W] = (owner_arr == 2)

    # "neutral" means visible, unclaimed, PLAIN land specifically -- not just
    # "unowned and visible" (that would wrongly include visible neutral
    # mountains, which also carry owner==0 on the wire; caught by export.py's
    # parity check against the JAX reference, which showed exactly this: 2
    # extra "neutral" cells matching the 2 visible mountains on the test board).
    neutral_cells = np.zeros((G, G), dtype=np.float32)
    neutral_cells[:H, :W] = (owner_arr == 0) & (type_arr == 1)

    owned_land_ch = np.full((G, G), my_land, dtype=np.float32)
    owned_army_ch = np.full((G, G), my_army, dtype=np.float32)
    opp_land_ch = np.full((G, G), opp_land, dtype=np.float32)
    opp_army_ch = np.full((G, G), opp_army, dtype=np.float32)
    timestep_ch = np.full((G, G), turn, dtype=np.float32)

    return np.stack([
        armies, generals, castles, mountains, neutral_cells,
        owned_cells, opponent_cells, fog_cells, structures_in_fog,
        owned_land_ch, owned_army_ch, opp_land_ch, opp_army_ch, timestep_ch,
    ], axis=0)


def init_memory(grid_size: int) -> dict:
    G = grid_size
    return {
        "revealed_structures": np.zeros((G, G), dtype=bool),
        "explored_cells": np.zeros((G, G), dtype=bool),
        "opponent_seen_cells": np.zeros((G, G), dtype=bool),
        "own_move_age": np.full((G, G), _MEM_AGE_SENTINEL, dtype=np.int32),
        "opp_activity_age": np.full((G, G), _MEM_AGE_SENTINEL, dtype=np.int32),
        "prev_opponent_cells": np.zeros((G, G), dtype=bool),
    }


def update_memory(mem: dict, raw_tensor: np.ndarray, own_action) -> dict:
    """`raw_tensor`: the RAW (pre-normalize_obs_tensor) (14,G,G) array from
    build_obs_tensor -- channel order 0 armies,1 generals,2 castles,
    3 mountains,4 neutral,5 owned,6 opponent,7 fog,8 structures_in_fog.
    `own_action`: the (pass,row,col,dir,split) tuple THIS agent just chose.
    Mirrors training/memory.py's update_memory exactly."""
    G = raw_tensor.shape[-1]
    fog = raw_tensor[7] > 0.5
    structures_in_fog = raw_tensor[8] > 0.5
    visible = (~fog) & (~structures_in_fog)  # excludes canvas padding too -- see build_obs_tensor's docstring
    castles = raw_tensor[2] > 0.5
    generals = raw_tensor[1] > 0.5
    opponent_cells = raw_tensor[6] > 0.5

    revealed = mem["revealed_structures"] | ((castles | generals) & visible)
    explored = mem["explored_cells"] | visible
    opponent_seen = mem["opponent_seen_cells"] | opponent_cells

    own_move_age = np.minimum(mem["own_move_age"] + 1, _MEM_AGE_SENTINEL)
    p, r, c, d, _s = own_action
    if p == 0:
        dr, dc = DIRECTIONS[d]
        dest_r = int(np.clip(r + dr, 0, G - 1))
        dest_c = int(np.clip(c + dc, 0, G - 1))
        own_move_age[r, c] = 0
        own_move_age[dest_r, dest_c] = 0

    opp_changed = opponent_cells != mem["prev_opponent_cells"]
    opp_activity_age = np.minimum(mem["opp_activity_age"] + 1, _MEM_AGE_SENTINEL)
    opp_activity_age[opp_changed] = 0

    return {
        "revealed_structures": revealed,
        "explored_cells": explored,
        "opponent_seen_cells": opponent_seen,
        "own_move_age": own_move_age,
        "opp_activity_age": opp_activity_age,
        "prev_opponent_cells": opponent_cells,
    }


def memory_to_channels(mem: dict) -> np.ndarray:
    """(NUM_MEMORY_CHANNELS, G, G) float32, appended after the normalized 14
    base channels. Trail channels are linearly decayed: 1.0 this turn, fading
    to 0 at MOVE_HISTORY_WINDOW turns ago. Mirrors training/memory.py's
    memory_to_channels exactly."""
    own_trail = np.maximum(0, MOVE_HISTORY_WINDOW - mem["own_move_age"]).astype(np.float32) / MOVE_HISTORY_WINDOW
    opp_trail = np.maximum(0, MOVE_HISTORY_WINDOW - mem["opp_activity_age"]).astype(np.float32) / MOVE_HISTORY_WINDOW
    return np.stack([
        mem["revealed_structures"].astype(np.float32),
        mem["explored_cells"].astype(np.float32),
        mem["opponent_seen_cells"].astype(np.float32),
        own_trail,
        opp_trail,
    ], axis=0)


def compute_valid_move_mask(armies: np.ndarray, owned_cells: np.ndarray, mountains: np.ndarray) -> np.ndarray:
    """(G,G,4) bool. Mirrors generals/core/action.py's compute_valid_move_mask exactly."""
    G = armies.shape[0]
    can_move_from = (owned_cells > 0.5) & (armies > 1)
    passable = ~(mountains > 0.5)
    mask = np.zeros((G, G, 4), dtype=bool)
    rows = np.arange(G)[:, None]
    cols = np.arange(G)[None, :]
    for d, (dr, dc) in enumerate(DIRECTIONS):
        dest_r = rows + dr
        dest_c = cols + dc
        in_bounds = (dest_r >= 0) & (dest_r < G) & (dest_c >= 0) & (dest_c < G)
        safe_r = np.clip(dest_r, 0, G - 1)
        safe_c = np.clip(dest_c, 0, G - 1)
        dest_passable = passable[safe_r, safe_c]
        mask[:, :, d] = can_move_from & in_bounds & dest_passable
    return mask


def compute_build_cost_grid(owned_cells: np.ndarray, castles: np.ndarray, generals: np.ndarray) -> np.ndarray:
    """Mirrors generals/modifiers/build_castles.py's build_cost_grid exactly."""
    G = owned_cells.shape[0]
    structures = (((castles > 0.5) | (generals > 0.5)) & (owned_cells > 0.5)).astype(np.int32)
    padded = np.pad(structures, BUILD_RADIUS)
    cost = np.full((G, G), BUILD_BASE_COST, dtype=np.int32)
    for di in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
        for dj in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
            surcharge = BUILD_PROXIMITY_PENALTY - BUILD_PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[BUILD_RADIUS + di:BUILD_RADIUS + di + G, BUILD_RADIUS + dj:BUILD_RADIUS + dj + G]
                cost = cost + surcharge * shifted
    return cost


def compute_legal_action_mask(owned_cells: np.ndarray, armies: np.ndarray, mountains: np.ndarray,
                               castles: np.ndarray, generals: np.ndarray,
                               build_castles_enabled: bool) -> np.ndarray:
    """(G*G*9 + 1,) bool. Mirrors training/action_space.py's compute_legal_action_mask."""
    G = armies.shape[0]
    move_mask = compute_valid_move_mask(armies, owned_cells, mountains)  # (G,G,4)

    if build_castles_enabled:
        cost = compute_build_cost_grid(owned_cells, castles, generals)
        plain = ~(castles > 0.5) & ~(generals > 0.5)
        build_mask = (owned_cells > 0.5) & plain & (armies >= cost)
    else:
        build_mask = np.zeros((G, G), dtype=bool)

    cell_mask = np.concatenate([move_mask, move_mask, build_mask[:, :, None]], axis=-1)  # (G,G,9)
    cell_mask_flat = cell_mask.reshape(-1)
    return np.concatenate([cell_mask_flat, np.array([True])])


def greedy_masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    penalized = np.where(mask, logits, logits - 1e9)
    return int(np.argmax(penalized))


def decode_action_index(index: int, H: int, W: int):
    """flat index -> engine action tuple (pass,row,col,dir,split). Mirrors
    training/action_space.py's decode_action_index. H, W here are the CANVAS
    size (always 21 in training), not the true board size -- matching how the
    mask/logits were computed."""
    num_cell = H * W * NUM_CELL_ACTIONS
    if index >= num_cell:
        return (1, 0, 0, 0, 0)
    row = index // (W * NUM_CELL_ACTIONS)
    rem = index % (W * NUM_CELL_ACTIONS)
    col = rem // NUM_CELL_ACTIONS
    kind = rem % NUM_CELL_ACTIONS
    if kind == BUILD_KIND:
        return (2, int(row), int(col), 0, 0)
    direction = kind if kind < 4 else kind - 4
    split = 1 if kind >= 4 else 0
    return (0, int(row), int(col), int(direction), int(split))
