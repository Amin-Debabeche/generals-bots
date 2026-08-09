"""Hand-crafted memory features, per arXiv:2507.06825 ("Artificial Generals
Intelligence: Mastering Generals.io with Reinforcement Learning" -- the paper
behind this repo's engine): our network had none, theirs adds four persistent
features to combat fog-of-war amnesia (a plain per-turn Observation has no way
to remember a general it glimpsed five turns ago that's now fogged again).

Channels added (5, not 4 -- the paper's "last seven moves by each side" is
split into two single-side trail channels here rather than one, since a
single shared channel can't distinguish whose move it was):
  1. revealed_structures  -- castles/generals ever seen, even if now fogged
     again (terrain doesn't change once revealed, safe to remember forever)
  2. explored_cells       -- every cell ever visible (cumulative ~fog_cells)
  3. opponent_seen_cells  -- cells ever seen as opponent-owned (a proxy for
     "the opponent has definitely had vision here")
  4. own_move_trail       -- recency-weighted source+destination cells of our
     own last MOVE_HISTORY_WINDOW moves (1.0 = just now, fading to 0 at the
     window edge)
  5. opponent_activity_trail -- recency-weighted cells where the VISIBLE
     opponent-ownership mask changed (proxy for "opponent moved here
     recently"), inferred purely from observation deltas

Deliberately NOT using privileged ground-truth of the opponent's chosen
action for channel 5, even though that's freely available during self-play
training (we simulate both sides) -- using it would create a train/inference
mismatch, since the real deployed bot only ever sees its own fog-limited wire
frames and must infer opponent activity the same way in both places. This
mirrors the parity discipline the exported numpy agent already follows
elsewhere in this codebase.

Memory is inherently PER-SEAT (per-player fog-of-war perspective) and PER-
EPISODE (reset at the start of every game) -- callers own the MemoryState
carry themselves; nothing here is stored globally.
"""
from typing import NamedTuple

import jax.numpy as jnp

from generals.core.observation import Observation

MOVE_HISTORY_WINDOW = 7
NUM_MEMORY_CHANNELS = 5
_AGE_SENTINEL = 999  # "never touched" -- must stay > MOVE_HISTORY_WINDOW

_DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


class MemoryState(NamedTuple):
    revealed_structures: jnp.ndarray  # (H,W) bool
    explored_cells: jnp.ndarray       # (H,W) bool
    opponent_seen_cells: jnp.ndarray  # (H,W) bool
    own_move_age: jnp.ndarray         # (H,W) int32 -- turns since an own move touched this cell
    opp_activity_age: jnp.ndarray     # (H,W) int32 -- turns since visible opponent ownership changed here
    prev_opponent_cells: jnp.ndarray  # (H,W) bool -- bookkeeping only, never exported as a channel


def init_memory(H: int, W: int) -> MemoryState:
    zeros_bool = jnp.zeros((H, W), dtype=bool)
    return MemoryState(
        revealed_structures=zeros_bool,
        explored_cells=zeros_bool,
        opponent_seen_cells=zeros_bool,
        own_move_age=jnp.full((H, W), _AGE_SENTINEL, dtype=jnp.int32),
        opp_activity_age=jnp.full((H, W), _AGE_SENTINEL, dtype=jnp.int32),
        prev_opponent_cells=zeros_bool,
    )


def update_memory(mem: MemoryState, obs: Observation, own_action: jnp.ndarray) -> MemoryState:
    """`own_action`: the engine-format [pass,row,col,dir,split] action THIS
    seat just took (used to mark its own move trail -- always known exactly,
    no inference needed, unlike the opponent's)."""
    H, W = obs.armies.shape
    # NOT just ~fog_cells: the canvas padding beyond a real board's true H,W
    # (see numpy_infer.py's build_obs_tensor docstring) has fog_cells=False
    # there too, by the training pool's own convention -- ~fog_cells alone
    # would wrongly mark the entire padding region as "explored" on turn one.
    # structures_in_fog is True across that whole region, so excluding it
    # here excludes the padding (and, correctly, any real partially-seen
    # obstacle cells that aren't fully resolved yet either).
    visible = ~obs.fog_cells & ~obs.structures_in_fog

    revealed = mem.revealed_structures | ((obs.castles | obs.generals) & visible)
    explored = mem.explored_cells | visible
    opponent_seen = mem.opponent_seen_cells | obs.opponent_cells

    pass_field, r, c, d, _split = own_action[0], own_action[1], own_action[2], own_action[3], own_action[4]
    is_move = pass_field == 0
    delta = _DIRECTIONS[d]
    dest_r = jnp.clip(r + delta[0], 0, H - 1)
    dest_c = jnp.clip(c + delta[1], 0, W - 1)
    own_touch = jnp.zeros((H, W), dtype=bool).at[r, c].set(is_move).at[dest_r, dest_c].set(is_move)
    own_move_age = jnp.where(own_touch, 0, jnp.minimum(mem.own_move_age + 1, _AGE_SENTINEL))

    opp_changed = obs.opponent_cells != mem.prev_opponent_cells
    opp_activity_age = jnp.where(opp_changed, 0, jnp.minimum(mem.opp_activity_age + 1, _AGE_SENTINEL))

    return MemoryState(
        revealed_structures=revealed,
        explored_cells=explored,
        opponent_seen_cells=opponent_seen,
        own_move_age=own_move_age,
        opp_activity_age=opp_activity_age,
        prev_opponent_cells=obs.opponent_cells,
    )


def memory_to_channels(mem: MemoryState) -> jnp.ndarray:
    """(NUM_MEMORY_CHANNELS, H, W) float32, ready to concatenate onto
    Observation.as_tensor()'s 14 channels. Trail channels are linearly
    decayed: 1.0 this turn, fading to 0 at MOVE_HISTORY_WINDOW turns ago."""
    own_trail = jnp.maximum(0, MOVE_HISTORY_WINDOW - mem.own_move_age).astype(jnp.float32) / MOVE_HISTORY_WINDOW
    opp_trail = jnp.maximum(0, MOVE_HISTORY_WINDOW - mem.opp_activity_age).astype(jnp.float32) / MOVE_HISTORY_WINDOW
    return jnp.stack([
        mem.revealed_structures.astype(jnp.float32),
        mem.explored_cells.astype(jnp.float32),
        mem.opponent_seen_cells.astype(jnp.float32),
        own_trail,
        opp_trail,
    ], axis=0)
