"""The single reference for how (network logits) <-> (engine action) and
(Observation) <-> (network input) convert. `training/ppo.py`/`rollout.py`
import this module directly (JAX); `competition/agents/my_bot/agent.py`'s
numpy inference is a hand-transliterated twin of the pure-math functions here
(`compute_build_cost_grid`, the index<->(row,col,kind) encoding, the
normalization constants) — see training/export.py's parity check, which is
what actually guarantees the twin stays honest, since Python can't share this
module with a numpy-only, jax-free submission.

Action index layout: for an (H, W) board, index in [0, H*W*9) decodes to a
per-cell move/build:
    row = index // (W*9); rem = index % (W*9); col = rem // 9; kind = rem % 9
    kind 0-3: move all-in,  direction = kind
    kind 4-7: move half,    direction = kind - 4
    kind 8:   build a castle at (row, col)
index == H*W*9 (the last slot) is the single global "pass" action.
"""
import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation
from training.config import NORMALIZATION

BUILD_KIND = 8
NUM_CELL_ACTIONS = 9

# Mirrors generals/modifiers/build_castles.py's pricing exactly (BASE_COST,
# PROXIMITY_PENALTY, PROXIMITY_DECAY, _RADIUS) but computed from an
# Observation's own-visible fields instead of a GameState — a player's own
# structures are always inside their fog-of-war visibility, so this is exact,
# not an approximation.
BUILD_BASE_COST = 35
BUILD_PROXIMITY_PENALTY = 14
BUILD_PROXIMITY_DECAY = 2
BUILD_RADIUS = (BUILD_PROXIMITY_PENALTY - 1) // BUILD_PROXIMITY_DECAY  # 6


def num_logits(H: int, W: int) -> int:
    return H * W * NUM_CELL_ACTIONS + 1


def normalize_obs_tensor(tensor: jnp.ndarray) -> jnp.ndarray:
    """tensor: (14, H, W) raw Observation.as_tensor() output -> normalized float32."""
    tensor = tensor.astype(jnp.float32)
    channels = list(jnp.split(tensor, tensor.shape[0], axis=0))
    for c in NORMALIZATION["log1p_channels"]:
        channels[c] = jnp.log1p(channels[c])
    for c in NORMALIZATION["land_scale_channels"]:
        channels[c] = channels[c] / NORMALIZATION["land_scale"]
    ts = NORMALIZATION["timestep_channel"]
    channels[ts] = channels[ts] / NORMALIZATION["timestep_scale"]
    return jnp.concatenate(channels, axis=0)


def obs_to_network_input(obs: Observation) -> jnp.ndarray:
    return normalize_obs_tensor(obs.as_tensor())


def obs_to_network_input_with_memory(obs: Observation, mem: "memory.MemoryState") -> jnp.ndarray:
    """(19, H, W): the 14 normalized Observation channels plus the 5
    hand-crafted memory channels from training/memory.py, concatenated in
    that fixed order. Memory channels are already in [0,1] -- no
    normalization applied to them."""
    from training import memory  # local import: memory.py doesn't depend on action_space.py, avoids a cycle
    base = normalize_obs_tensor(obs.as_tensor())
    mem_channels = memory.memory_to_channels(mem)
    return jnp.concatenate([base, mem_channels], axis=0)


def compute_build_cost_grid(owned_cells: jnp.ndarray, castles: jnp.ndarray,
                             generals: jnp.ndarray) -> jnp.ndarray:
    """(H, W) castle-build price for the observing player, from own-visible
    structures only (own general + own castles are always visible)."""
    H, W = owned_cells.shape
    structures = ((castles | generals) & owned_cells).astype(jnp.int32)
    padded = jnp.pad(structures, BUILD_RADIUS)

    cost = jnp.full((H, W), BUILD_BASE_COST, dtype=jnp.int32)
    for di in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
        for dj in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
            surcharge = BUILD_PROXIMITY_PENALTY - BUILD_PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[BUILD_RADIUS + di:BUILD_RADIUS + di + H,
                                  BUILD_RADIUS + dj:BUILD_RADIUS + dj + W]
                cost = cost + surcharge * shifted
    return cost


def compute_legal_action_mask(obs: Observation, build_castles_enabled: bool) -> jnp.ndarray:
    """(H*W*9 + 1,) bool mask over the flat logit vector. Pass is always legal."""
    H, W = obs.armies.shape
    move_mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)  # (H,W,4)

    if build_castles_enabled:
        cost = compute_build_cost_grid(obs.owned_cells, obs.castles, obs.generals)
        plain = ~obs.generals & ~obs.castles
        build_mask = obs.owned_cells & plain & (obs.armies >= cost)  # (H,W)
    else:
        build_mask = jnp.zeros((H, W), dtype=jnp.bool_)

    # kinds 0-3 (all-in) and 4-7 (half) share the same legality as move_mask
    cell_mask = jnp.concatenate(
        [move_mask, move_mask, build_mask[:, :, None]], axis=-1
    )  # (H, W, 9)
    cell_mask_flat = cell_mask.reshape(H * W * NUM_CELL_ACTIONS)
    pass_mask = jnp.ones((1,), dtype=jnp.bool_)
    return jnp.concatenate([cell_mask_flat, pass_mask])


def encode_action_index(action: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """Inverse of decode_action_index: engine action [pass,row,col,dir,split] ->
    flat index. Used by training/pretrain_bc.py to convert a heuristic Agent's
    (e.g. HunterAgent's) chosen action into a supervised-learning target."""
    pass_field, row, col, direction, split = action[0], action[1], action[2], action[3], action[4]
    num_cell = H * W * NUM_CELL_ACTIONS
    kind_move = direction + jnp.where(split == 1, 4, 0)
    kind = jnp.where(pass_field == 2, BUILD_KIND, kind_move)
    cell_index = (row * W + col) * NUM_CELL_ACTIONS + kind
    return jnp.where(pass_field == 1, num_cell, cell_index).astype(jnp.int32)


def decode_action_index(index: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """flat index -> engine action array [pass, row, col, direction, split] (int32)."""
    num_cell = H * W * NUM_CELL_ACTIONS
    is_pass = index >= num_cell
    cell_idx = jnp.minimum(index, num_cell - 1)
    row = cell_idx // (W * NUM_CELL_ACTIONS)
    rem = cell_idx % (W * NUM_CELL_ACTIONS)
    col = rem // NUM_CELL_ACTIONS
    kind = rem % NUM_CELL_ACTIONS

    is_build = kind == BUILD_KIND
    direction = jnp.where(kind < 4, kind, kind - 4)
    split = ((kind >= 4) & (kind < BUILD_KIND)).astype(jnp.int32)

    pass_field = jnp.where(is_pass, 1, jnp.where(is_build, 2, 0)).astype(jnp.int32)
    row_out = jnp.where(is_pass, 0, row).astype(jnp.int32)
    col_out = jnp.where(is_pass, 0, col).astype(jnp.int32)
    dir_out = jnp.where(is_pass | is_build, 0, direction).astype(jnp.int32)
    split_out = jnp.where(is_pass | is_build, 0, split).astype(jnp.int32)

    return jnp.stack([pass_field, row_out, col_out, dir_out, split_out])


def masked_log_softmax(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    penalized = jnp.where(legal_mask, logits, logits - 1e9)
    return jax.nn.log_softmax(penalized)


def sample_action_index(key: jnp.ndarray, logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    penalized = jnp.where(legal_mask, logits, logits - 1e9)
    return jax.random.categorical(key, penalized)


def greedy_action_index(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Deterministic argmax over legal logits — used at eval/deployment time
    (converting a trained stochastic PPO policy into a competitive-play
    policy is standard practice), never during PPO rollout collection."""
    penalized = jnp.where(legal_mask, logits, logits - 1e9)
    return jnp.argmax(penalized)


def logprob_and_entropy(logits: jnp.ndarray, legal_mask: jnp.ndarray, index: jnp.ndarray):
    log_probs = masked_log_softmax(logits, legal_mask)
    logprob = log_probs[index]
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(jnp.where(legal_mask, probs * log_probs, 0.0))
    return logprob, entropy
