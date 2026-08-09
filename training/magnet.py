""""Magnet" policy: a lightweight heuristic action distribution used to
KL-anchor PPO's policy, replacing a plain entropy bonus (see training/ppo.py).

Adapted from strakam/AverageJoe's train/magnet.py (the #1-ranked real
generals.io bot, same author as the human-replay dataset this project also
uses) -- their insight: a plain entropy bonus only pulls the policy toward
*uniform randomness*, which is exactly what let this project's PPO runs drift
from a working, BC-warm-started policy into passivity over hundreds of
iterations (entropy climbing while win rate collapsed in lockstep -- see the
"main3" incident). A KL pull toward an already-sensible heuristic instead of
toward noise can provide the same exploration pressure without that failure
mode: at maximum pull, the policy converges toward "expand/capture
sensibly", never toward "do nothing" or random churn.

This is a *soft* heuristic, not a legal-move generator -- training/ppo.py
intersects its output with the real legal-action mask before using it, so
imprecision here (e.g. an approximate capture condition) only biases the KL
pull, never lets the network believe an illegal action is viable.
"""
import jax
import jax.numpy as jnp

from generals.core.observation import Observation
from training.action_space import NUM_CELL_ACTIONS


def _dest(arr: jnp.ndarray) -> jnp.ndarray:
    """(H, W) -> (H, W, 4): value at the destination cell of each of the 4
    directions (up, down, left, right -- see generals/core/action.py's
    DIRECTIONS, matched here via jnp.roll direction-for-direction)."""
    return jnp.stack([
        jnp.roll(arr, 1, axis=0),   # up:    dest = (r-1, c)
        jnp.roll(arr, -1, axis=0),  # down:  dest = (r+1, c)
        jnp.roll(arr, 1, axis=1),   # left:  dest = (r, c-1)
        jnp.roll(arr, -1, axis=1),  # right: dest = (r, c+1)
    ], axis=-1)


def expander_magnet(
    obs: Observation,
    legal_mask: jnp.ndarray,
    *,
    score_pass: float = 0.2,
    score_default: float = 1.0,
    score_neutral: float = 2.0,
    score_enemy: float = 3.0,
    score_castle: float = 5.0,
    score_build: float = 2.0,
    score_topk: float = 2.0,
    topk: int = 5,
) -> jnp.ndarray:
    """(H*W*9 + 1,) probability distribution favoring expansion and capture --
    same shape/layout as the network's flat action logits
    (training/action_space.py's index encoding).

    Args:
        obs: Observation for the seat being scored.
        legal_mask: (H*W*9+1,) bool, from action_space.compute_legal_action_mask
            -- illegal entries are always driven to ~0 probability regardless
            of heuristic score.
    """
    H, W = obs.armies.shape

    own_army = obs.armies
    dest_army = _dest(obs.armies)
    dest_neutral = _dest(obs.neutral_cells)
    dest_opponent = _dest(obs.opponent_cells)
    dest_owned = _dest(obs.owned_cells)
    dest_castle = _dest(obs.castles)

    can_capture = own_army[..., None] > dest_army + 1
    dest_capturable = ~dest_owned & can_capture

    dir_scores = jnp.full((H, W, 4), score_default)
    dir_scores = jnp.where(dest_neutral & dest_capturable, score_neutral, dir_scores)
    dir_scores = jnp.where(dest_opponent & dest_capturable, score_enemy, dir_scores)
    dir_scores = jnp.where(dest_castle & dest_capturable, score_castle, dir_scores)

    # Boost the top-k owned cells by army count -- mirrors human/heuristic
    # play concentrating force from its strongest stacks rather than
    # dribbling small amounts everywhere.
    top_thresh = jax.lax.sort(own_army.reshape(-1), is_stable=False)[-topk]
    is_top = (own_army >= top_thresh) & (own_army > 0)
    dir_scores = dir_scores + is_top[..., None] * score_topk

    build_scores = jnp.full((H, W, 1), score_build)
    cell_scores = jnp.concatenate([dir_scores, dir_scores, build_scores], axis=-1)  # (H, W, 9)
    assert cell_scores.shape[-1] == NUM_CELL_ACTIONS
    flat_scores = cell_scores.reshape(H * W * NUM_CELL_ACTIONS)
    scores = jnp.concatenate([flat_scores, jnp.array([score_pass])])

    penalized = jnp.where(legal_mask, scores, -1e9)
    return jax.nn.softmax(penalized)
