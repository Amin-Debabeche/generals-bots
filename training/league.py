"""Checkpoint pool ('league') for self-play opponent diversity: periodic
params-only snapshots, sampled PFSP-lite (weighted toward opponents the
current policy is beating less), with a uniform floor so a snapshot never
drops out of rotation entirely. Restricted by the caller (training/train.py)
to snapshots from the current or immediately-previous curriculum stage via
`prune_to_stages` — an old Stage-A snapshot never learned castles and would
be a free, uninformative win once in Stage D/E.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from training.config import LeagueConfig
from training.network import PolicyValueNetwork


@dataclass
class LeagueMember:
    id: str
    stage_name: str
    iteration: int
    path: str
    winrate_ema: float = 0.5


class League:
    def __init__(self, checkpoints_dir: Path, cfg: LeagueConfig, template_net: PolicyValueNetwork):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._template = template_net
        self.members: list[LeagueMember] = []
        self._params_cache: dict[str, PolicyValueNetwork] = {}

    def has_snapshots(self) -> bool:
        return len(self.members) > 0

    def add_snapshot(self, net: PolicyValueNetwork, stage_name: str, iteration: int) -> LeagueMember:
        member_id = f"{stage_name}_iter{iteration}"
        path = self.checkpoints_dir / f"{member_id}.eqx"
        eqx.tree_serialise_leaves(str(path), net)
        member = LeagueMember(id=member_id, stage_name=stage_name, iteration=iteration, path=str(path))
        self.members.append(member)
        self._params_cache[member.id] = net
        return member

    def prune_to_stages(self, allowed_stage_names: set) -> None:
        self.members = [m for m in self.members if m.stage_name in allowed_stage_names]
        keep_ids = {m.id for m in self.members}
        self._params_cache = {k: v for k, v in self._params_cache.items() if k in keep_ids}

    def _load(self, member: LeagueMember) -> PolicyValueNetwork:
        if member.id not in self._params_cache:
            self._params_cache[member.id] = eqx.tree_deserialise_leaves(member.path, self._template)
        return self._params_cache[member.id]

    def sample_opponent(self, key: jnp.ndarray) -> tuple[LeagueMember, PolicyValueNetwork]:
        assert self.members, "sample_opponent called on an empty league"
        weights = jnp.array([
            min(max(1.0 - m.winrate_ema, self.cfg.winrate_floor), self.cfg.winrate_ceil)
            for m in self.members
        ])
        k_uniform, k_weighted, k_pick = jrandom.split(key, 3)
        use_uniform = jrandom.uniform(k_uniform) < self.cfg.uniform_floor
        weighted_idx = jrandom.choice(k_weighted, len(self.members), p=weights / weights.sum())
        uniform_idx = jrandom.choice(k_pick, len(self.members))
        idx = int(jnp.where(use_uniform, uniform_idx, weighted_idx))
        member = self.members[idx]
        return member, self._load(member)

    def update_winrate(self, member_id: str, wins: int, games: int) -> None:
        if games == 0:
            return
        for m in self.members:
            if m.id == member_id:
                observed = wins / games
                m.winrate_ema = (1 - self.cfg.ema_alpha) * m.winrate_ema + self.cfg.ema_alpha * observed
                return

    def save_index(self, path: Path) -> None:
        data = [dict(id=m.id, stage_name=m.stage_name, iteration=m.iteration,
                     path=m.path, winrate_ema=m.winrate_ema) for m in self.members]
        Path(path).write_text(json.dumps(data, indent=2))

    def load_index(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.members = [LeagueMember(**d) for d in data]
        self._params_cache = {}
