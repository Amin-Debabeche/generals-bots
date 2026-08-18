"""Observation augmentation + history state for training/network_transformer.py,
adapted from strakam/AverageJoe's networks/common.py (AugmentedObsState/
augment_obs) -- replaces training/memory.py's role for the transformer path
(training/memory.py is untouched and still used by the CNN path,
training/network.py).

Differences from AverageJoe's original, each deliberate:

- Built directly on `Observation.as_tensor()` instead of duplicating their
  `obs_to_array` -- confirmed field-for-field identical in channel order (14
  channels: armies, generals, castles/their-"cities", mountains,
  neutral_cells, owned_cells, opponent_cells, fog_cells, structures_in_fog,
  owned_land_count, owned_army_count, opponent_land_count,
  opponent_army_count, timestep -- see training/config.py's own channel-index
  comment, which matches their augment_obs's local unpacking line for line).
- Their auto-padding logic (checking obs_arr's true H/W against a larger
  pad_to and filling the border with mountains) is dropped, not "ported
  inert": every Observation this module receives is already the canvas size
  in this repo's established convention (training/memory.py's
  init_memory/update_memory carry the same assumption; training/curriculum.py
  fixes pad_to=21 for every stage so the env itself never produces a
  differently-shaped observation, and training/human_replays.py /
  training/leaderboard_replays.py explicitly pad before calling any
  augmentation function). Padding here would be dead code, not a safety net.
- `seen` uses the exact `~fog_cells & ~structures_in_fog` signal
  training/memory.py's update_memory already established as the correct
  "currently visible" primitive in this repo (the `& ~structures_in_fog`
  term specifically excludes the canvas-padding region, which has
  fog_cells=False but structures_in_fog=True by the training pool's own
  convention -- see memory.py's docstring). AverageJoe's original instead
  max-pools `owned_cells` as a visibility proxy, a workaround needed only
  because of the auto-padding logic just above; since that's gone here,
  their approximation would also be strictly worse than the precise signal
  we already have (it marks a cell "seen" only once *owned*, never merely
  visible-but-unclaimed -- e.g. neutral or enemy territory just inside
  vision range).
- `history_size`/`temporal_window` default to the trimmed values in
  training/config.py's TransformerNetworkConfig (5 / 128, not AverageJoe's
  7 / 512) -- AugmentedObsState is inherently ~7-8x heavier per env-step than
  MemoryState even at these smaller sizes; budget for it explicitly in
  rollout.py's trajectory buffer sizing rather than copying their defaults.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.observation import Observation
from training.config import TransformerNetworkConfig

# 24 fixed channels (see _build_channels below) + 2*history_size stacked
# per-cell army-delta channels.
NUM_BASE_CHANNELS = 24


class AugmentedObsState(NamedTuple):
    army_stack: jnp.ndarray                     # (history_size, p, p) -- own army delta history
    enemy_stack: jnp.ndarray                     # (history_size, p, p) -- opponent army delta history
    last_army: jnp.ndarray                       # (p, p)
    last_enemy_army: jnp.ndarray                 # (p, p)
    castles: jnp.ndarray                         # (p, p) bool -- ever-seen castles (persistent)
    generals: jnp.ndarray                        # (p, p) bool -- ever-seen generals (persistent)
    mountains: jnp.ndarray                       # (p, p) bool -- ever-seen mountains (persistent)
    seen: jnp.ndarray                            # (p, p) bool -- ever-visible (own vision)
    enemy_seen: jnp.ndarray                      # (p, p) bool -- ever-visible (opponent-owned)
    last_enemy_army_seen_value: jnp.ndarray      # (p, p)
    last_enemy_army_seen_timestep: jnp.ndarray   # (p, p) -- turns since last seen (raw count)
    opponent_army_history: jnp.ndarray           # (temporal_window,)
    opponent_land_history: jnp.ndarray           # (temporal_window,)
    temporal_step: jnp.ndarray                   # () int32


def init_obs_state(cfg: TransformerNetworkConfig = TransformerNetworkConfig()) -> AugmentedObsState:
    p = cfg.grid_size
    return AugmentedObsState(
        army_stack=jnp.zeros((cfg.history_size, p, p)),
        enemy_stack=jnp.zeros((cfg.history_size, p, p)),
        last_army=jnp.zeros((p, p)),
        last_enemy_army=jnp.zeros((p, p)),
        castles=jnp.zeros((p, p), dtype=jnp.bool_),
        generals=jnp.zeros((p, p), dtype=jnp.bool_),
        mountains=jnp.zeros((p, p), dtype=jnp.bool_),
        seen=jnp.zeros((p, p), dtype=jnp.bool_),
        enemy_seen=jnp.zeros((p, p), dtype=jnp.bool_),
        last_enemy_army_seen_value=jnp.zeros((p, p)),
        last_enemy_army_seen_timestep=jnp.zeros((p, p)),
        opponent_army_history=jnp.zeros((cfg.temporal_window,)),
        opponent_land_history=jnp.zeros((cfg.temporal_window,)),
        temporal_step=jnp.int32(0),
    )


def reset_obs_state(obs_state: AugmentedObsState) -> AugmentedObsState:
    return jax.tree.map(jnp.zeros_like, obs_state)


def reset_obs_state_where(obs_state: AugmentedObsState, dones: jnp.ndarray) -> AugmentedObsState:
    """dones: (N,) bool, obs_state fields batched (N, ...). Resets any env
    slot whose episode just ended -- mirrors training/rollout.py's
    _reset_memory_where_done for MemoryState."""
    def reset_leaf(leaf):
        expand = dones.reshape(dones.shape[0], *([1] * (leaf.ndim - 1)))
        return jnp.where(expand, jnp.zeros_like(leaf), leaf)
    return jax.tree.map(reset_leaf, obs_state)


def augment_obs(obs: Observation, obs_state: AugmentedObsState) -> tuple[jnp.ndarray, AugmentedObsState]:
    """obs: single-sample Observation (already canvas-sized -- see module
    docstring). obs_state: single-sample AugmentedObsState.

    Returns:
        augmented: (NUM_BASE_CHANNELS + 2*history_size, p, p) float32
        new_obs_state: AugmentedObsState
    """
    # .astype(jnp.float32): as_tensor()'s spatial channels are int32/bool (army
    # counts are integers, masks are bool) -- matching training/action_space.py's
    # normalize_obs_tensor, which casts first for the same reason. Without this,
    # current_army (below) silently comes out int32, which breaks AugmentedObsState's
    # (float32-initialized) scan carry with a dtype mismatch the first time this
    # runs inside jax.lax.scan.
    arr = obs.as_tensor().astype(jnp.float32)  # (14, p, p) -- see training/config.py's channel-index comment
    (armies_ch, generals_ch, castles_ch, mountains_ch,
     neutral_cells, owned_cells, opponent_cells,
     fog_cells, structures_in_fog,
     owned_land_count, owned_army_count,
     opponent_land_count, opponent_army_count, timestep_ch) = range(14)

    p = obs_state.last_army.shape[0]

    current_army = arr[armies_ch] * arr[owned_cells]
    current_enemy_army = arr[armies_ch] * arr[opponent_cells]

    new_army_stack = jnp.concatenate([
        (current_army - obs_state.last_army)[None, :, :],
        obs_state.army_stack[:-1, :, :],
    ], axis=0)
    new_enemy_stack = jnp.concatenate([
        (current_enemy_army - obs_state.last_enemy_army)[None, :, :],
        obs_state.enemy_stack[:-1, :, :],
    ], axis=0)

    # ~fog_cells & ~structures_in_fog == "currently visible", the same primitive
    # training/memory.py's update_memory established (see module docstring).
    visible = (arr[fog_cells] <= 0) & (arr[structures_in_fog] <= 0)
    enemy_visible = arr[opponent_cells] > 0
    new_seen = obs_state.seen | visible
    new_enemy_seen = obs_state.enemy_seen | enemy_visible

    new_castles = obs_state.castles | (arr[castles_ch] > 0)
    new_generals = obs_state.generals | (arr[generals_ch] > 0)
    new_mountains = obs_state.mountains | (arr[mountains_ch] > 0)

    new_last_enemy_army_seen_value = jnp.where(
        current_enemy_army > 0, current_enemy_army, obs_state.last_enemy_army_seen_value
    )
    new_last_enemy_army_seen_timestep = jnp.where(
        current_enemy_army > 0, 0.0, obs_state.last_enemy_army_seen_timestep + 1.0
    )

    opp_army_val = arr[opponent_army_count, 0, 0]
    opp_land_val = arr[opponent_land_count, 0, 0]
    new_opponent_army_history = jnp.roll(obs_state.opponent_army_history, -1).at[-1].set(opp_army_val)
    new_opponent_land_history = jnp.roll(obs_state.opponent_land_history, -1).at[-1].set(opp_land_val)
    new_temporal_step = obs_state.temporal_step + 1

    coords_x = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[None, :] / (p - 1), (p, p))
    coords_y = jnp.broadcast_to(jnp.arange(p, dtype=jnp.float32)[:, None] / (p - 1), (p, p))
    ones = jnp.ones((p, p))

    channels = jnp.stack([
        arr[armies_ch],                               # 0
        current_army,                                  # 1
        current_enemy_army,                             # 2
        arr[armies_ch] * arr[neutral_cells],             # 3
        new_seen.astype(jnp.float32),                    # 4
        new_enemy_seen.astype(jnp.float32),               # 5
        new_generals.astype(jnp.float32),                  # 6
        new_castles.astype(jnp.float32),                    # 7
        new_mountains.astype(jnp.float32),                   # 8
        arr[neutral_cells],                                   # 9
        arr[owned_cells],                                      # 10
        arr[opponent_cells],                                    # 11
        arr[fog_cells],                                          # 12
        arr[structures_in_fog],                                   # 13
        arr[timestep_ch] * ones,                                   # 14
        (arr[timestep_ch] % 50) * ones / 50,                        # 15
        arr[owned_land_count] * ones,                                # 16
        arr[owned_army_count] * ones,                                 # 17
        arr[opponent_land_count] * ones,                               # 18
        arr[opponent_army_count] * ones,                                # 19
        new_last_enemy_army_seen_value,                                  # 20
        jnp.log1p(new_last_enemy_army_seen_timestep) / 5.0,               # 21
        coords_x,                                                         # 22
        coords_y,                                                         # 23
    ], axis=0)
    assert channels.shape[0] == NUM_BASE_CHANNELS

    augmented = jnp.concatenate([channels, new_army_stack, new_enemy_stack], axis=0)

    new_state = AugmentedObsState(
        army_stack=new_army_stack,
        enemy_stack=new_enemy_stack,
        last_army=current_army,
        last_enemy_army=current_enemy_army,
        castles=new_castles,
        generals=new_generals,
        mountains=new_mountains,
        seen=new_seen,
        enemy_seen=new_enemy_seen,
        last_enemy_army_seen_value=new_last_enemy_army_seen_value,
        last_enemy_army_seen_timestep=new_last_enemy_army_seen_timestep,
        opponent_army_history=new_opponent_army_history,
        opponent_land_history=new_opponent_land_history,
        temporal_step=new_temporal_step,
    )
    return augmented, new_state


def temporal_data(obs_state: AugmentedObsState) -> jnp.ndarray:
    """(2, temporal_window) -- [army_history, land_history], the raw input
    training/network_transformer.py's TemporalEncoder expects."""
    return jnp.stack([obs_state.opponent_army_history, obs_state.opponent_land_history])


def normalize_augmented(obs: jnp.ndarray) -> jnp.ndarray:
    """Normalize an augmented observation's army-scale channels (divisor 50,
    matching AverageJoe's own normalize_observations -- all their army-like
    channels/stacks use this one constant). Boolean/coordinate channels
    (4-9, 22-23) are left untouched; channels 14-19 have their own
    hand-picked divisors already applied inline in augment_obs (timestep/50,
    land counts *1 -- kept unnormalized here to match AverageJoe's own
    normalize_observations, which only rescales channels [0,1,2,3,17,19,20]
    + every history-stack channel, plus 14/16/18)."""
    army_channels = [0, 1, 2, 3, 17, 19, 20] + list(range(NUM_BASE_CHANNELS, obs.shape[0]))
    obs = obs.at[jnp.array(army_channels)].divide(50.0)
    obs = obs.at[14].divide(50.0)
    obs = obs.at[jnp.array([16, 18])].divide(50.0)
    return obs
