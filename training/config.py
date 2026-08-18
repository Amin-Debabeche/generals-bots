"""Single source of tunables for the training pipeline.

Nothing in this module may ever set `jax.config.update("jax_enable_x64", True)`
or otherwise enable float64 — the exported bot does numpy-only float32
inference, and a silent float32/float64 mismatch is a real parity risk (see
training/export.py's parity check).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetworkConfig:
    """Fixed network geometry. `grid_size` matches the competition pad_to=21
    so the network never resizes across curriculum stages."""
    grid_size: int = 21
    # 14 from Observation.as_tensor() + 5 hand-crafted memory channels
    # (training/memory.py, per arXiv:2507.06825) -- changing this breaks
    # compatibility with any checkpoint trained under a different value.
    in_channels: int = 19
    backbone_channels: tuple[int, ...] = (48, 64, 64, 64, 48)
    kernel_size: int = 3
    value_hidden: int = 64
    # Per-cell action kinds: 4 directions x {all-in, half} + 1 build = 9
    num_cell_actions: int = 9


@dataclass(frozen=True)
class TransformerNetworkConfig:
    """Geometry for training/network_transformer.py -- adapted from
    strakam/AverageJoe's HistoryTransformer (the #1-ranked real generals.io
    ladder bot), kept as a separate dataclass from NetworkConfig rather than
    overloading it with irrelevant CNN-only/transformer-only fields. See
    ~/.claude/plans/drifting-questing-babbage.md for the full rationale.
    """
    grid_size: int = 21
    # 24 base augmented channels (training/augment.py, adapted from
    # AverageJoe's networks/common.py:augment_obs) + 2*history_size per-cell
    # army-delta stacks.
    history_size: int = 5  # AverageJoe default is 7; trimmed for CPU-smoke-test/Colab memory budget
    n_channels: int = 24 + 2 * history_size
    # patch_size=1 would preserve per-cell resolution but at ~300x the
    # attention cost of patch_size=3 for zero action-space benefit (per-cell
    # logits come from the per-patch head's 9*M*M output + unpatchify, not
    # from patch_size itself) -- see plan doc. 21 is divisible by 1,3,7,21.
    patch_size: int = 3
    depth: int = 4  # AverageJoe S config default; L uses 6-8, GPU-budget dependent
    embed_dim: int = 192  # trimmed from AverageJoe's 256-352 for a smaller first pass
    n_head: int = 8
    ff_factor: int = 4
    value_hidden: int = 64
    # Per-cell action kinds: 4 directions x {all-in, half} + 1 build = 9
    num_cell_actions: int = 9
    temporal_window: int = 128  # AverageJoe default 512; trimmed, see history_size note
    temporal_hidden: int = 256


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 512
    num_steps: int = 256
    # 1 epoch, not the more common 2-4: measured directly on this machine, the
    # PPO update (not rollout collection) dominates iteration wall-clock at
    # ~55s/epoch for a 256envx256step batch — with a 131072-sample batch at
    # the real 512-env default, 1 epoch already keeps iterations in the
    # ~1-1.5min range the training budget assumes; more epochs trades
    # iteration count (more distinct experience over a long run) for more
    # gradient steps per batch, and iteration count matters more here.
    num_epochs: int = 1
    minibatch_size: int = 1024
    gamma: float = 0.998
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01  # overridden per-stage by curriculum.py
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    # Weight on the contact-distance potential-based shaping term (see
    # training/rewards.py) added after Stage C showed zero wins across 347
    # iterations -- composite_reward_fn alone gives no signal toward finding
    # and closing on the opponent, which matters once that's a genuine
    # multi-hundred-turn campaign rather than a small-board skirmish.
    contact_shaping_weight: float = 0.15
    # Weight on the general-safety potential-based shaping term (see
    # training/rewards.py) -- added after real generals.bot leaderboard
    # replays showed the bot repeatedly leading on land/army against several
    # opponents right up to the second-to-last tick, then losing everything
    # in one turn to an undefended general.
    # Raised 0.2 -> 0.8: at 0.2, direct behavioral inspection (main9, real
    # vs-Hunter games) showed the general's own army barely rising above
    # passive structure growth -- the pull was too weak to change behavior at
    # all, not just insufficient. A Hunter-imitation BC refinement was tried
    # as an alternative fix and rejected: it improved garrison behavior in
    # some games but measurably regressed overall play (vs_random 90s%->59%,
    # vs_hunter itself 60-86%->31%) by pulling the whole policy toward
    # Hunter's simpler overall style, not just its defensive habit --
    # escalating this weight instead keeps the RL objective in charge of
    # everything except the one specific behavior being nudged.
    # Raised again 0.8 -> 2.0 (main11, Stage E): the 0.8 round was real
    # progress (garrison near losses ~8% -> ~14.6% of total army, per direct
    # inspection) but still well short of surviving Hunter reliably --
    # target vs_hunter is 80%, currently ~50%. Escalating the same
    # already-working lever further alongside GARRISON_FRACTION 0.25->0.4
    # in training/rewards.py.
    general_safety_weight: float = 2.0
    # Exponential moving average of network weights, updated every iteration
    # (see training/train.py's ema_net) and used for eval/promotion decisions
    # and as the frozen Stage-D reference -- insulates those signals from
    # single-iteration noise, unlike reading the raw live (currently-being-
    # gradient-updated) network directly. 0.999 matches strakam/AverageJoe's
    # default (the #1-ranked real generals.io bot, same technique).
    weight_ema_decay: float = 0.999


@dataclass(frozen=True)
class LeagueConfig:
    snapshot_interval: int = 50  # iterations between league snapshots
    full_checkpoint_interval: int = 20  # iterations between resume-checkpoints
    winrate_floor: float = 0.1
    winrate_ceil: float = 0.9
    uniform_floor: float = 0.2  # fraction of opponent draws that ignore PFSP weighting
    ema_alpha: float = 0.1  # exponential-moving-average rate for per-member win-rate


# Normalization constants applied to Observation.as_tensor() channels before
# they reach the network. Serialized verbatim into weights_meta.json at
# export time so the numpy inference agent never hand-copies these literals.
NORMALIZATION = {
    "log1p_channels": (0, 10, 12),      # armies, owned_army_count, opponent_army_count
    "land_scale_channels": (9, 11),      # owned_land_count, opponent_land_count
    "land_scale": 441.0,                 # 21*21
    "timestep_channel": 13,
    "timestep_scale": 1200.0,            # competition truncation
}

# Observation.as_tensor() channel order (see generals/core/observation.py docstring):
# 0 armies, 1 generals, 2 castles, 3 mountains, 4 neutral_cells, 5 owned_cells,
# 6 opponent_cells, 7 fog_cells, 8 structures_in_fog, 9 owned_land_count,
# 10 owned_army_count, 11 opponent_land_count, 12 opponent_army_count, 13 timestep
CH_ARMIES = 0
CH_GENERALS = 1
CH_CASTLES = 2
CH_MOUNTAINS = 3
CH_NEUTRAL = 4
CH_OWNED = 5
CH_OPPONENT = 6
CH_FOG = 7
CH_STRUCT_IN_FOG = 8
