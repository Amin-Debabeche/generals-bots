"""Reward shaping layered on top of the engine's `composite_reward_fn`
(generals/core/rewards.py, left untouched — this module wraps it rather than
editing engine internals).

Added after Stage C (the real 18-21 board, generals ~17 tiles apart —
`min_generals_distance=17`, matching the competition's spawn rule) showed
literally zero wins across 347 iterations / tens of thousands of games,
despite Hunter/Expander still landing real kills against us — proof captures
work, just that our policy never learned to reach one. `composite_reward_fn`
only shapes land/army/castle *ratios*; nothing in it specifically points the
policy toward finding and closing on the opponent, which matters a lot more
once closing the distance takes a genuine multi-hundred-turn campaign than it
did on Stage A/B's small board.

The added term is potential-based shaping (Ng, Harada & Russell 1999):
F(s,a,s') = gamma*Phi(s') - Phi(s). This form is provably policy-invariant
(it changes the learning *speed*, not the optimal policy) for any bounded
potential Phi, as long as the same gamma is used here as in the return/GAE
computation — see `shaped_reward_fn`'s `gamma` argument, which callers must
pass PPOConfig.gamma, not a different value.
"""
import jax.numpy as jnp

from generals.core.observation import Observation
from generals.core.rewards import composite_reward_fn


def _masked_centroid(mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(mean_row, mean_col, any_true) of a boolean (H, W) mask."""
    H, W = mask.shape
    rows = jnp.arange(H, dtype=jnp.float32)[:, None]
    cols = jnp.arange(W, dtype=jnp.float32)[None, :]
    maskf = mask.astype(jnp.float32)
    count = jnp.sum(maskf)
    safe_count = jnp.maximum(count, 1.0)
    mean_row = jnp.sum(maskf * rows) / safe_count
    mean_col = jnp.sum(maskf * cols) / safe_count
    return mean_row, mean_col, count > 0


def _contact_potential(obs: Observation) -> jnp.ndarray:
    """In [-1, 0]: -1 when no enemy cell is currently visible (fog-of-war
    respected -- this only ever reads `obs`, never ground truth beyond what
    the policy itself can see), rising toward 0 as our territory's centroid
    nears the visible-enemy centroid. Deliberately -1 rather than 0 when no
    contact exists: a bounded, defined-everywhere potential means *first*
    spotting the enemy is a smooth improvement (Phi rises from -1), never a
    one-step penalty the way a 0-when-unseen/negative-when-seen potential
    would create right at the moment contact is made."""
    H, W = obs.armies.shape
    max_dist = float(H + W)
    or_, oc, has_owned = _masked_centroid(obs.owned_cells)
    er, ec, has_enemy = _masked_centroid(obs.opponent_cells)
    dist = jnp.abs(or_ - er) + jnp.abs(oc - ec)
    norm_dist = jnp.clip(dist / max_dist, 0.0, 1.0)
    return jnp.where(has_owned & has_enemy, -norm_dist, -1.0)


GARRISON_FRACTION = 0.12
MIN_GARRISON = 4.0


def _general_safety_potential(obs: Observation) -> jnp.ndarray:
    """In [0, 1]: how well-garrisoned our own general is, as a fraction of a
    target garrison size scaled to our current total army. 0 = general
    holds no army at all (maximally exposed), 1 = at or above target.

    Added after checking real generals.bot leaderboard replays (rank 105/116
    at the time): the bot was repeatedly *leading* on land/army against
    several opponents right up to the second-to-last tick, then losing
    everything in one turn -- an undefended general getting captured, not
    gradual strategic outplay. Neither composite_reward_fn nor
    training/magnet.py's expander_magnet had any defensive concept, only
    expansion/capture -- this term is meant to close that specific gap.
    `GARRISON_FRACTION`/`MIN_GARRISON` mirror generals/agents/hunter_agent.py's
    own `GARRISON = 4` (kept on its general; only surplus above 2x that is
    ever sent out) -- Hunter already defends this way, so this potential
    points the trained policy toward a similarly-sized home reserve, not a
    made-up one."""
    own_general = obs.generals & obs.owned_cells
    gen_army = jnp.sum(jnp.where(own_general, obs.armies, 0)).astype(jnp.float32)
    target = jnp.maximum(GARRISON_FRACTION * obs.owned_army_count.astype(jnp.float32), MIN_GARRISON)
    return jnp.clip(gen_army / target, 0.0, 1.0)


def shaped_reward_fn(prior_obs: Observation, prior_action: jnp.ndarray, obs: Observation,
                      gamma: float, contact_weight: float, general_safety_weight: float) -> jnp.ndarray:
    """composite_reward_fn plus the contact-distance and general-safety
    shaping terms above. Mirrors composite_reward_fn's own convention of not
    shaping the terminal step (a win/loss should stay a clean signal, not
    diluted by shaping noise)."""
    base = composite_reward_fn(prior_obs, prior_action, obs)
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    contact_shaping = gamma * _contact_potential(obs) - _contact_potential(prior_obs)
    safety_shaping = gamma * _general_safety_potential(obs) - _general_safety_potential(prior_obs)
    shaped = base + contact_weight * contact_shaping + general_safety_weight * safety_shaping
    return jnp.where(game_done, base, shaped)
