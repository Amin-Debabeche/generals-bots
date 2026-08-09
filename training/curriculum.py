"""The 5-stage curriculum: env ruleset ramps from a small perfect-info board
up to the exact competition ruleset. `pad_to=21` is fixed across every stage
so the network's input shape never changes and no stage transition ever
triggers a network resize (see training/network.py) — only the env's own
compiled kernels change, which is a real but bounded recompilation cost (see
the plan's risk #6).

Opponent mix is resolved per-iteration (not fixed at stage definition time)
because Stage A's mix depends on the running vs-Random win-rate, and every
stage's "self-play" fraction should fall back from league-checkpoint
opponents to mirror-self-play until at least one league snapshot exists.
"""
from dataclasses import dataclass, field
from typing import Callable

from generals import GeneralsEnv

BUCKET_NAMES = ("random", "expander", "hunter", "selfplay_current", "selfplay_league")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    env_kwargs: dict
    build_castles_enabled: bool
    entropy_coef: float
    # fraction of the total mix that is "self-play" (current + league combined),
    # split between current-mirror and league-PFSP by selfplay_league_fraction
    base_mix_fn: Callable[[float], dict]  # (vs_random_winrate) -> mix dict, pre-selfplay-split
    selfplay_league_fraction: float
    promotion_metric: str  # "vs_random" or "vs_expander" or "vs_frozen_checkpoint"
    promotion_threshold: float
    promotion_iter_cap: int


def _stage_a_mix(vs_random_winrate: float) -> dict:
    if vs_random_winrate >= 0.8:
        return {"random": 0.7, "expander": 0.3}
    return {"random": 1.0}


def _fixed_mix(mix: dict) -> Callable[[float], dict]:
    return lambda _vs_random_winrate: dict(mix)


STAGE_A = CurriculumStage(
    name="A_foundations",
    env_kwargs=dict(grid_dims=(12, 12), pad_to=21, truncation=200,
                     perfect_info=True, build_castles=False, deathtouch_turn=None,
                     pool_size=2000),
    build_castles_enabled=False,
    entropy_coef=0.02,
    base_mix_fn=_stage_a_mix,
    selfplay_league_fraction=0.5,  # irrelevant: stage A's mix never includes self-play
    promotion_metric="vs_random",
    promotion_threshold=0.90,
    promotion_iter_cap=300,
)

STAGE_B = CurriculumStage(
    name="B_fog",
    env_kwargs=dict(grid_dims=(12, 12), pad_to=21, truncation=300,
                     perfect_info=False, build_castles=False, deathtouch_turn=None,
                     pool_size=2000),
    build_castles_enabled=False,
    entropy_coef=0.02,
    base_mix_fn=_fixed_mix({"random": 0.2, "expander": 0.4, "hunter": 0.2, "selfplay": 0.2}),
    selfplay_league_fraction=0.5,
    promotion_metric="vs_expander",
    promotion_threshold=0.65,
    promotion_iter_cap=400,
)

_COMPETITION_BOARD_KWARGS = dict(
    min_grid_size=18, max_grid_size=21, pad_to=21,
    mountain_density_range=(0.24, 0.26), num_castles_range=(9, 11),
    min_generals_distance=17, castle_val_range=(20, 26),
)

STAGE_C = CurriculumStage(
    name="C_real_board_size",
    env_kwargs=dict(**_COMPETITION_BOARD_KWARGS, truncation=600,
                     perfect_info=False, build_castles=False, deathtouch_turn=None),
    build_castles_enabled=False,
    # 0.002, not 0.01: at 0.01, entropy climbed monotonically (0.82 -> 4.6 over
    # 191 iterations) and win-rate vs Hunter/Expander collapsed to zero in
    # lockstep -- confirmed directly on this exact stage (main3 run). The
    # entropy bonus's steady per-gradient-step pull toward higher entropy was
    # overwhelming the win-rate policy gradient over hundreds of iterations,
    # unlearning a working (BC-warm-started) policy rather than refining it.
    entropy_coef=0.002,
    base_mix_fn=_fixed_mix({"random": 0.1, "expander": 0.3, "hunter": 0.2, "selfplay": 0.4}),
    selfplay_league_fraction=0.6,
    promotion_metric="vs_expander",
    promotion_threshold=0.60,
    promotion_iter_cap=500,
)

STAGE_D = CurriculumStage(
    name="D_castles",
    env_kwargs=dict(**_COMPETITION_BOARD_KWARGS, truncation=900,
                     perfect_info=False, build_castles=True, deathtouch_turn=None),
    build_castles_enabled=True,
    # See Stage C's comment -- same fix applied preventively (D was never
    # reached with the old 0.01 in a healthy state, so this is untested here
    # specifically, but the mechanism isn't stage-specific).
    entropy_coef=0.002,
    base_mix_fn=_fixed_mix({"expander": 0.2, "hunter": 0.2, "selfplay": 0.6}),
    selfplay_league_fraction=0.7,
    promotion_metric="vs_frozen_checkpoint",
    promotion_threshold=0.55,
    promotion_iter_cap=600,
)

STAGE_E = CurriculumStage(
    name="E_full_competition",
    env_kwargs=dict(mode="competition"),
    build_castles_enabled=True,
    # See Stage C's comment for why this is much lower than the original 0.005.
    entropy_coef=0.001,
    base_mix_fn=_fixed_mix({"expander": 0.15, "hunter": 0.15, "selfplay": 0.7}),
    selfplay_league_fraction=0.85,
    promotion_metric=None,  # terminal stage
    promotion_threshold=None,
    promotion_iter_cap=None,
)

STAGES = [STAGE_A, STAGE_B, STAGE_C, STAGE_D, STAGE_E]
STAGE_BY_NAME = {s.name: s for s in STAGES}


def build_env(stage: CurriculumStage) -> GeneralsEnv:
    return GeneralsEnv(**stage.env_kwargs)


def resolve_opponent_mix(stage: CurriculumStage, vs_random_winrate: float,
                          league_has_snapshots: bool) -> dict:
    """Returns a mix dict over BUCKET_NAMES summing to ~1.0.

    `base_mix_fn` may return either a mix already keyed by BUCKET_NAMES
    (Stage A) or one with a combined "selfplay" key that gets split into
    selfplay_current/selfplay_league here (every other stage). The league
    portion folds into selfplay_current whenever no league snapshot exists
    yet (true self-play needs no history; PFSP-vs-checkpoint does).
    """
    raw = stage.base_mix_fn(vs_random_winrate)
    if "selfplay" not in raw:
        return {k: raw.get(k, 0.0) for k in BUCKET_NAMES}

    selfplay_total = raw.pop("selfplay")
    if league_has_snapshots:
        league_frac = selfplay_total * stage.selfplay_league_fraction
        current_frac = selfplay_total - league_frac
    else:
        league_frac = 0.0
        current_frac = selfplay_total

    mix = {k: raw.get(k, 0.0) for k in ("random", "expander", "hunter")}
    mix["selfplay_current"] = current_frac
    mix["selfplay_league"] = league_frac
    return mix


def allocate_envs(mix: dict, num_envs: int) -> dict:
    """Integer env-count per bucket, summing exactly to num_envs. Zero-weight
    buckets get 0 envs (never a stray 1-env bucket from rounding)."""
    raw_counts = {k: mix.get(k, 0.0) * num_envs for k in BUCKET_NAMES}
    counts = {k: int(v) for k, v in raw_counts.items()}
    shortfall = num_envs - sum(counts.values())
    # Distribute the rounding remainder to the buckets with the largest
    # fractional part (largest-remainder method), skipping zero-weight buckets.
    remainders = sorted(
        (k for k in BUCKET_NAMES if mix.get(k, 0.0) > 0.0),
        key=lambda k: raw_counts[k] - counts[k],
        reverse=True,
    )
    for k in remainders[:shortfall]:
        counts[k] += 1
    return counts
