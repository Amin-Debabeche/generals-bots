"""Convert real generals.io replays (strakammm/generals_io_replays on Hugging
Face -- a curated high-rank dataset, Apache 2.0, from the same author as this
repo/paper) into (Observation, action_index) behavior-cloning pairs, filtered
to moves made by high-rated ("top ELO") players specifically.

Why this exists: training/pretrain_bc.py's Hunter-imitation warm start broke
the "never learns to expand" collapse, but Hunter is a simple hand-coded
heuristic -- real high-rated human play has far more sophisticated strategy
(multi-front pressure, feints, timing pushes, contesting cities) worth
imitating too.

Ruleset caveat (important, not swept under the rug): vanilla generals.io has
neutral CAPTURABLE cities, not the competition ruleset's player-BUILT castles,
and no deathtouch endgame. Human replays teach movement/economy/attack
fundamentals under the same base mechanics (verified below to match this
engine's create_initial_state/global_update exactly), but say nothing about
the build-castle action -- that's still learned from Hunter-BC/RL, not here.

Replay schema (confirmed against real downloaded data, not just the dataset
card): one row per 1v1 game --
    version, id, mapWidth, mapHeight, usernames (len 2), stars (len 2),
    cities (flat positions), cityArmies (parallel to cities, matches
    create_initial_state's castle-value-in-grid convention directly),
    generals (flat positions, generals[i] belongs to player i -- confirmed by
    cross-checking the first move of player i always originates there),
    mountains (flat positions),
    moves: list of [playerIndex, fromCell, toCell, isHalf(0/1), turn]
Positions are row-major flat indices (pos = row*mapWidth + col) -- confirmed
by observing move deltas of exactly +-1 (same row) and +-mapWidth (same col).

Version 13 is used (not 5): the two versions likely differ in some replay
mechanic and mixing them risks subtle tick misalignment; 8190 of the 18803
games are version 13, still ample.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from generals.core import game
from generals.core.observation import Observation
from training import action_space as asp
from training import augment as aug_mod
from training import memory as mem_mod
from training.config import TransformerNetworkConfig

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right -- matches generals/core/config.py
_DIR_LOOKUP = {d: i for i, d in enumerate(DIRECTIONS)}

VERSION = 13
CANVAS = 21  # matches NetworkConfig.grid_size -- fixed regardless of the replay's true board size


def pad_observation(obs: Observation, canvas: int = CANVAS) -> Observation:
    """Embed a true-HxW Observation (from game.get_observation on an
    unpadded replay board) at the top-left of a fixed canvas x canvas grid --
    same convention as training/rollout.py's pad_to=21 pool boards and
    competition/agents/my_bot/numpy_infer.py's wire-format embedding.
    Padding-region values were EMPIRICALLY verified (see numpy_infer.py's
    build_obs_tensor docstring): every spatial channel is 0/False in the
    padding except structures_in_fog, which is True there. Scalar fields
    (owned_land_count etc.) are plain 0-d values, unaffected by board size."""
    H, W = obs.armies.shape
    if H == canvas and W == canvas:
        return obs

    def pad(x):
        c = jnp.zeros((canvas, canvas), dtype=x.dtype)
        return c.at[:H, :W].set(x)

    structures_in_fog = pad(obs.structures_in_fog).at[H:, :].set(True).at[:, W:].set(True)

    return obs._replace(
        armies=pad(obs.armies), generals=pad(obs.generals), castles=pad(obs.castles),
        mountains=pad(obs.mountains), neutral_cells=pad(obs.neutral_cells),
        owned_cells=pad(obs.owned_cells), opponent_cells=pad(obs.opponent_cells),
        fog_cells=pad(obs.fog_cells), structures_in_fog=structures_in_fog,
    )


@dataclass
class ReplayMove:
    player: int
    from_rc: tuple[int, int]
    to_rc: tuple[int, int]
    is_half: bool
    turn: int


def load_dataframe(parquet_path: str, version: int = VERSION) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return df[df["version"] == version].reset_index(drop=True)


def build_grid(row: pd.Series) -> np.ndarray:
    """Mirrors generals/core/game.create_initial_state's expected grid encoding
    exactly: -2 mountain, 0 empty, 1/2 general (player 0/1), castle value in
    [3,255] = a castle worth that many starting armies."""
    H, W = int(row["mapHeight"]), int(row["mapWidth"])
    grid = np.zeros((H, W), dtype=np.int32)
    for pos in row["mountains"]:
        r, c = divmod(int(pos), W)
        grid[r, c] = -2
    gen0, gen1 = int(row["generals"][0]), int(row["generals"][1])
    grid[divmod(gen0, W)] = 1
    grid[divmod(gen1, W)] = 2
    for pos, army in zip(row["cities"], row["cityArmies"]):
        r, c = divmod(int(pos), W)
        grid[r, c] = int(army)
    return grid


def parse_moves(row: pd.Series) -> list[ReplayMove]:
    W = int(row["mapWidth"])
    out = []
    for m in row["moves"]:
        player, from_cell, to_cell, is_half, turn = (int(x) for x in m)
        out.append(ReplayMove(
            player=player,
            from_rc=divmod(from_cell, W),
            to_rc=divmod(to_cell, W),
            is_half=bool(is_half),
            turn=turn,
        ))
    return out


def _move_to_action(move: ReplayMove) -> jnp.ndarray | None:
    dr = move.to_rc[0] - move.from_rc[0]
    dc = move.to_rc[1] - move.from_rc[1]
    d = _DIR_LOOKUP.get((dr, dc))
    if d is None:
        return None  # malformed/non-adjacent move -- skip (should not happen on well-formed data)
    return jnp.array([0, move.from_rc[0], move.from_rc[1], d, int(move.is_half)], dtype=jnp.int32)


PASS_ACTION = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def simulate_replay(row: pd.Series, elite_players: set[int]):
    """Replays one game tick-by-tick through the REAL engine (game.step, no
    modifiers -- vanilla generals.io has neither build-castles nor
    deathtouch), yielding (Observation, MemoryState, action_index) pairs for
    moves made by players in `elite_players` (use {0,1} to collect both
    sides). Memory (training/memory.py) is tracked continuously for BOTH
    seats regardless of `elite_players` -- it has to be, to stay correct for
    whichever seat(s) we do end up sampling from -- and reflects only
    information available from that seat's own fog-limited observations,
    exactly like during RL rollout collection.

    Returns (samples, num_illegal) where num_illegal counts elite moves that
    turned out illegal in this simulation (wrong source ownership, army<=1,
    etc.) -- this should be ~0 on well-formed data; a high count means tick
    alignment or move parsing is wrong, not that the human misplayed.
    """
    grid = build_grid(row)
    state = game.create_initial_state(jnp.asarray(grid))
    moves = parse_moves(row)

    by_turn: dict[int, dict[int, ReplayMove]] = {}
    for mv in moves:
        by_turn.setdefault(mv.turn, {})[mv.player] = mv
    max_turn = max(by_turn) if by_turn else 0

    mem = {0: mem_mod.init_memory(CANVAS, CANVAS), 1: mem_mod.init_memory(CANVAS, CANVAS)}

    samples = []
    num_illegal = 0

    for t in range(max_turn + 1):
        turn_moves = by_turn.get(t, {})
        actions = []
        pending = {}  # player -> (obs_before, mem_before, action_index) to record after legality check
        for p in (0, 1):
            mv = turn_moves.get(p)
            if mv is None:
                actions.append(PASS_ACTION)
                continue
            action = _move_to_action(mv)
            if action is None:
                actions.append(PASS_ACTION)
                continue
            actions.append(action)
            if p in elite_players:
                obs = pad_observation(game.get_observation(state, p))
                # action's row/col are already within the true board, which sits
                # at the canvas's top-left -- encode at CANVAS size to match the
                # network's fixed action space, not the replay's true H,W.
                idx = int(asp.encode_action_index(action, CANVAS, CANVAS))
                pending[p] = (obs, mem[p], idx, mv)

        actions_arr = jnp.stack(actions)

        # Legality check against the mask BEFORE stepping (mirrors training's
        # compute_legal_action_mask exactly, since that's what the exported
        # bot/trained policy is held to as well).
        for p, (obs, mem_p, idx, mv) in pending.items():
            mask = asp.compute_legal_action_mask(obs, build_castles_enabled=False)
            if not bool(mask[idx]):
                num_illegal += 1
                continue
            samples.append((obs, mem_p, idx))

        state, _info = game.step(state, actions_arr)

        # Memory update for BOTH seats, every turn, regardless of whether we
        # sampled from them -- uses the post-step observation (what's visible
        # going into the NEXT turn) and this turn's actual action per side.
        for p in (0, 1):
            obs_after = pad_observation(game.get_observation(state, p))
            mem[p] = mem_mod.update_memory(mem[p], obs_after, actions_arr[p])

        if bool(state.winner >= 0):
            break

    return samples, num_illegal


def simulate_replay_transformer(row: pd.Series, elite_players: set[int],
                                 cfg: TransformerNetworkConfig = TransformerNetworkConfig()):
    """Transformer-path twin of simulate_replay -- same tick-by-tick replay
    through the real engine, same elite-move filtering and illegal-move
    guard, but tracks AugmentedObsState instead of MemoryState. Unlike
    memory's separate post-action update, aug_mod.augment_obs(obs, obs_state)
    is called ONCE per seat per turn with the PRE-action observation (see
    training/rollout.py's transformer-path section docstring) -- so state
    threading here is simpler: no separate "after" pass needed, just update
    obs_state[p] to augment_obs's second return value once, for every
    player, every turn (recording (obs, obs_state) as-carried-in for
    whichever seats are elite, exactly like simulate_replay records mem_p
    before its own post-hoc update)."""
    grid = build_grid(row)
    state = game.create_initial_state(jnp.asarray(grid))
    moves = parse_moves(row)

    by_turn: dict[int, dict[int, ReplayMove]] = {}
    for mv in moves:
        by_turn.setdefault(mv.turn, {})[mv.player] = mv
    max_turn = max(by_turn) if by_turn else 0

    obs_state = {0: aug_mod.init_obs_state(cfg), 1: aug_mod.init_obs_state(cfg)}

    samples = []
    num_illegal = 0

    for t in range(max_turn + 1):
        turn_moves = by_turn.get(t, {})
        actions = []
        obs_state_next = {}
        for p in (0, 1):
            mv = turn_moves.get(p)
            obs = pad_observation(game.get_observation(state, p), cfg.grid_size)
            _augmented, obs_state_next[p] = aug_mod.augment_obs(obs, obs_state[p])

            if mv is None:
                actions.append(PASS_ACTION)
                continue
            action = _move_to_action(mv)
            if action is None:
                actions.append(PASS_ACTION)
                continue
            actions.append(action)
            if p in elite_players:
                idx = int(asp.encode_action_index(action, cfg.grid_size, cfg.grid_size))
                mask = asp.compute_legal_action_mask(obs, build_castles_enabled=False)
                if not bool(mask[idx]):
                    num_illegal += 1
                else:
                    samples.append((obs, obs_state[p], idx))

        actions_arr = jnp.stack(actions)
        state, _info = game.step(state, actions_arr)
        obs_state = obs_state_next

        if bool(state.winner >= 0):
            break

    return samples, num_illegal


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True)
    p.add_argument("--network", choices=("cnn", "transformer"), default="cnn",
                    help="cnn: MemoryState-based samples (default, matches the currently-deployed "
                         "architecture). transformer: AugmentedObsState-based samples, see "
                         "~/.claude/plans/drifting-questing-babbage.md")
    p.add_argument("--min-stars", type=int, default=80)
    p.add_argument("--max-board", type=int, default=CANVAS,
                    help="skip replays with either map dimension above this (must fit the network's canvas)")
    p.add_argument("--max-games", type=int, default=None, help="cap for a quick test run")
    p.add_argument("--out", required=True, help="output .npz of stacked (obs fields, action_index)")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_dataframe(args.parquet)
    print(f"version {VERSION} games: {len(df)}")

    if args.max_games:
        df = df.iloc[:args.max_games]

    transformer_cfg = TransformerNetworkConfig()
    simulate_fn = (lambda row, elite: simulate_replay_transformer(row, elite, transformer_cfg)) \
        if args.network == "transformer" else simulate_replay

    all_samples = []
    total_illegal = 0
    games_used = 0
    for i, row in df.iterrows():
        if row["mapWidth"] > args.max_board or row["mapHeight"] > args.max_board:
            continue
        elite = {p for p, s in enumerate(row["stars"]) if s >= args.min_stars}
        if not elite:
            continue
        games_used += 1
        samples, illegal = simulate_fn(row, elite)
        all_samples.extend(samples)
        total_illegal += illegal
        if games_used % 50 == 0:
            print(f"  processed {games_used} elite games, {len(all_samples)} samples so far, "
                  f"{total_illegal} illegal moves encountered")

    print(f"\nDone: {games_used} games, {len(all_samples)} samples, {total_illegal} illegal moves "
          f"({total_illegal / max(len(all_samples), 1) * 100:.3f}% -- should be ~0)")

    # Stack into arrays matching Observation's and MemoryState's/
    # AugmentedObsState's field names for np.savez -- state fields prefixed
    # "mem_"/"obs_state_" to keep namespaces apart (Observation shares no
    # field names with either today, but prefixing makes that an invariant,
    # not an accident). AugmentedObsState's float leaves are stored as
    # float16, not float32: dataset size grows meaningfully with the richer
    # state (history stacks + temporal history vs. MemoryState's handful of
    # small fields), matching the mitigation training/rollout.py's trajectory
    # buffer already uses (bf16 there; float16 here since this is a
    # numpy/npz file, not a JAX array -- numpy has no native bf16).
    obs_fields = Observation._fields
    stacked = {f: np.stack([np.asarray(getattr(o, f)) for o, _s, _i in all_samples]) for f in obs_fields}
    if args.network == "transformer":
        for f in aug_mod.AugmentedObsState._fields:
            vals = np.stack([np.asarray(getattr(s, f)) for _o, s, _i in all_samples])
            stacked[f"obs_state_{f}"] = vals.astype(np.float16) if vals.dtype != np.bool_ else vals
    else:
        for f in mem_mod.MemoryState._fields:
            stacked[f"mem_{f}"] = np.stack([np.asarray(getattr(s, f)) for _o, s, _i in all_samples])
    stacked["action_index"] = np.array([idx for _o, _s, idx in all_samples], dtype=np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **stacked)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
