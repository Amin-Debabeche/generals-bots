"""RL-trained agent: a small CNN policy trained with self-play PPO (see
training/ at the repo root), exported to pure NumPy inference by
training/export.py. Weights live in weights.npz / weights_meta.json next to
this file, produced by:

    python -m training.export --checkpoint <path-to-.eqx> --out competition/agents/my_bot --build-castles

`Agent.act(obs)` is called once per turn. The `obs` argument is built by
`main.py` from the wire-protocol frame and has these fields:

    obs.H, obs.W            board dimensions (constant for the whole game)
    obs.turn                current turn number, increments each step
    obs.my_land              total cells you own
    obs.my_army              total armies summed over your cells
    obs.opp_land             opponent's land count (visible at all times)
    obs.opp_army             opponent's army total (visible at all times)
    obs.type_grid[r][c]     0=fog, 1=plain, 2=mountain, 3=castle, 4=general, 5=structure-in-fog
    obs.owner_grid[r][c]    0=neutral/unknown, 1=me, 2=opp  (perspective-relative)
    obs.army_grid[r][c]     army count, 0 in fog or empty

`act` must return a 5-tuple `(pass, row, col, direction, split)` -- see
numpy_infer.decode_action_index for the exact encoding.

Never imports torch or jax: pure NumPy keeps import cost near zero against
the sandbox's 10s first-move budget (the competition docs note `import torch`
alone costs ~2.3s, the full JAX stack ~3s). All the actual math (conv
forward pass, legal-move/build masking, wire-format reconstruction, memory
features) lives in numpy_infer.py, which is a hand-transliterated twin of
training/network.py + training/action_space.py + training/memory.py, checked
against them by training/export.py's parity gate before these weights were
ever written.

Memory (per arXiv:2507.06825's "memory features") is per-GAME state: one
Agent instance lives for the whole match (main.py constructs it once per
game), so it's carried as plain instance attributes, updated once per act()
call -- no external persistence needed.
"""
import sys
from pathlib import Path

import numpy as np

import numpy_infer as ninf

_DIR = Path(__file__).parent
_WEIGHTS, _META = ninf.load_weights(_DIR / "weights.npz", _DIR / "weights_meta.json")
_GRID_SIZE = _META["grid_size"]
_BUILD_CASTLES_ENABLED = True  # the real competition ruleset always has build-castles on

PASS = (1, 0, 0, 0, 0)


class Agent:
    """Greedy (argmax) inference over the trained policy network.

    Deterministic on purpose: converting a trained stochastic PPO policy into
    a competitive-play policy is standard practice -- the network still
    trained with exploration, this just always takes its best-scored legal
    move at deployment time.
    """

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.mem = ninf.init_memory(_GRID_SIZE)

    def act(self, obs):
        try:
            raw = ninf.build_obs_tensor(
                obs.type_grid, obs.owner_grid, obs.army_grid,
                obs.my_land, obs.my_army, obs.opp_land, obs.opp_army,
                obs.turn, _GRID_SIZE,
            )
            # Legal-move/build mask from the RAW (unnormalized) channels --
            # normalization (log1p, scaling) below must never leak into the
            # army-count/ownership comparisons the mask relies on.
            mask = ninf.compute_legal_action_mask(
                raw[5], raw[0], raw[3], raw[2], raw[1], _BUILD_CASTLES_ENABLED,
            )
            normalized = ninf.normalize_obs_tensor(raw.copy(), _META)
            mem_channels = ninf.memory_to_channels(self.mem)
            network_input = np.concatenate([normalized, mem_channels], axis=0)
            logits = ninf.forward_policy_logits(_WEIGHTS, network_input)
            idx = ninf.greedy_masked_argmax(logits, mask)
            action = ninf.decode_action_index(idx, _GRID_SIZE, _GRID_SIZE)

            # Memory update uses the RAW (unnormalized) tensor and the action
            # just chosen -- carried forward into next turn's act() call.
            self.mem = ninf.update_memory(self.mem, raw, action)

            p, r, c, d, s = action
            if p != 1 and (r >= self.H or c >= self.W):
                # Defensive: the network's action space spans the full
                # (padded) canvas; a move/build targeting the padding beyond
                # the real board would be silently no-op'd by the engine
                # anyway, but return a clean pass rather than rely on that.
                return PASS
            return action
        except Exception as e:
            print(f"[my_bot] act() failed, passing: {e!r}", file=sys.stderr)
            return PASS

    def reset(self):
        self.mem = ninf.init_memory(_GRID_SIZE)
