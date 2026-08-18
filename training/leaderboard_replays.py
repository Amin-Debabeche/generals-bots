"""Pull real generals.bot COMPETITION-ruleset replays from the actual
leaderboard (not vanilla generals.io like training/human_replays.py) and
convert strong players' moves into (Observation, MemoryState, action_index)
BC training pairs, in the same output schema training/pretrain_bc_human.py
already reads.

Why this exists: training/human_replays.py's vanilla generals.io data taught
movement/economy fundamentals but knows nothing about this competition's
actual ruleset (castles, deathtouch, this exact board-size range) since that
mechanic doesn't exist in vanilla generals.io. These leaderboard replays are
real games played under the exact ruleset our bot competes under, by
genuinely strong opponents (elo up to 3450+) -- directly relevant to the
general-safety gap found by hand-reading a handful of loss replays (see
training/rewards.py's docstring): several real losses showed the bot leading
on economy right up to the second-to-last tick, then losing everything in
one turn to an undefended general. Real skilled play is exactly what's
missing from the self-play/heuristic training mix.

API (reverse-engineered from the leaderboard frontend's own inline JS --
no official docs):
    GET /api/leaderboard                              -> {leaderboard: [...]}
    GET /api/leaderboard?matches=1&player=<name>       -> {matches: [...]}
    GET /api/leaderboard?replay=<game id>              -> full replay (below)

Replay format IMPORTANT DIFFERENCE from human_replays.py's data: no move
list, only a full (armies, owners) board snapshot per tick. Moves are
reconstructed here by diffing each tick against a "growth-only" projection
(both players pass) computed with the REAL engine (generals/core/game.step),
which gets tick/phase timing exactly right without hand-deriving it, then
searching the (typically tiny, diff-localized) candidate action space and
verifying each candidate by actually replaying it and checking for an exact
board match. Ticks that don't reconstruct to an exact match (rare -- see
`reconstruct_game`'s returned stats) are skipped entirely, never included in
the dataset. `castles` has been empty in every replay sampled while building
this (real players don't seem to use the build-castle mechanic much) --
handled gracefully if that's ever untrue, but don't expect castle-building
demonstrations from this data source.
"""
import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import requests

from generals.core import game
from training import action_space as asp
from training import augment as aug_mod
from training import memory as mem_mod
from training.config import TransformerNetworkConfig

API_BASE = "https://www.generals.bot/api/leaderboard"
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right -- matches generals/core/action.py
CANVAS = 21  # matches NetworkConfig.grid_size / training/human_replays.py's convention
PASS_ACTION = np.array([1, 0, 0, 0, 0], dtype=np.int32)
_REQUEST_DELAY_S = 0.15  # be a reasonable citizen of someone else's API


def _get(params: dict) -> dict:
    resp = requests.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY_S)
    return resp.json()


def fetch_leaderboard() -> list[dict]:
    return _get({})["leaderboard"]


def fetch_matches(player_name: str) -> list[dict]:
    return _get({"matches": 1, "player": player_name}).get("matches", [])


def fetch_replay(game_id: str) -> dict:
    return _get({"replay": game_id})


# ---------------------------------------------------------------------------
# Move reconstruction (verified against the real engine, see module docstring)
# ---------------------------------------------------------------------------

def build_initial_state(replay: dict):
    H, W = replay["dims"]["rows"], replay["dims"]["cols"]
    grid = np.zeros((H, W), dtype=np.int32)
    for r, c in replay["mountains"]:
        grid[r, c] = -2
    for castle in replay.get("castles", []):
        # Never observed nonempty while building this -- guard, don't assume shape.
        r, c, army = castle if len(castle) == 3 else (*castle, 40)
        grid[r, c] = army
    (gr0, gc0), (gr1, gc1) = replay["generals"]
    grid[gr0, gc0] = 1
    grid[gr1, gc1] = 2
    return game.create_initial_state(jnp.asarray(grid))


def _owners_to_ownership(owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return owners == 0, owners == 1


def _states_match(state, armies, own0, own1) -> bool:
    return (bool(jnp.all(state.armies == jnp.asarray(armies)))
            and bool(jnp.all(state.ownership[0] == jnp.asarray(own0)))
            and bool(jnp.all(state.ownership[1] == jnp.asarray(own1))))


def _candidate_actions(armies_t, owners_t, player_idx, anomalous_cells, H, W):
    cands = []
    for (r, c) in anomalous_cells:
        if owners_t[r, c] != player_idx or armies_t[r, c] <= 1:
            continue
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cands.append(np.array([0, r, c, d, 0], dtype=np.int32))  # all-in
                cands.append(np.array([0, r, c, d, 1], dtype=np.int32))  # half
    cands.append(PASS_ACTION.copy())
    return cands


def reconstruct_game(replay: dict, max_ticks: int | None = None):
    """Returns (list of (state_before, action0, action1) for each successfully
    reconstructed tick, {"matched": int, "total": int})."""
    H, W = replay["dims"]["rows"], replay["dims"]["cols"]
    state = build_initial_state(replay)
    ticks = replay["ticks"]
    n = len(ticks) - 1 if max_ticks is None else min(max_ticks, len(ticks) - 1)

    samples = []
    matched = 0
    for t in range(n):
        armies_t = np.array(ticks[t]["armies"], dtype=np.int32)
        owners_t = np.array(ticks[t]["owners"], dtype=np.int32)
        armies_next = np.array(ticks[t + 1]["armies"], dtype=np.int32)
        owners_next = np.array(ticks[t + 1]["owners"], dtype=np.int32)
        own0_t, own1_t = _owners_to_ownership(owners_t)

        if not _states_match(state, armies_t, own0_t, own1_t):
            # Drift from an earlier unmatched tick -- resync to ground truth
            # so later ticks in this game can still be attempted.
            state = state._replace(
                armies=jnp.asarray(armies_t),
                ownership=jnp.stack([jnp.asarray(own0_t), jnp.asarray(own1_t)]),
            )

        passed_state, _ = game.step(state, jnp.stack([jnp.asarray(PASS_ACTION), jnp.asarray(PASS_ACTION)]))
        proj_armies = np.asarray(passed_state.armies)
        proj_own0 = np.asarray(passed_state.ownership[0])
        proj_own1 = np.asarray(passed_state.ownership[1])
        real_own0, real_own1 = _owners_to_ownership(owners_next)
        anomalous = set(map(tuple, np.argwhere(
            (proj_armies != armies_next) | (proj_own0 != real_own0) | (proj_own1 != real_own1)
        )))

        found = None
        for a0 in _candidate_actions(armies_t, owners_t, 0, anomalous, H, W):
            for a1 in _candidate_actions(armies_t, owners_t, 1, anomalous, H, W):
                new_state, _info = game.step(state, jnp.stack([jnp.asarray(a0), jnp.asarray(a1)]))
                if _states_match(new_state, armies_next, real_own0, real_own1):
                    found = (a0, a1, new_state)
                    break
            if found:
                break

        if found is None:
            state = state._replace(
                armies=jnp.asarray(armies_next),
                ownership=jnp.stack([jnp.asarray(real_own0), jnp.asarray(real_own1)]),
                time=state.time + 1,
            )
            continue

        a0, a1, new_state = found
        samples.append((state, a0, a1))
        state = new_state
        matched += 1

    return samples, {"matched": matched, "total": n}


# ---------------------------------------------------------------------------
# Convert reconstructed (state, action0, action1) triples into BC samples
# ---------------------------------------------------------------------------

def collect_bc_samples(reconstructed, elite_seats: set[int]):
    """reconstructed: output of reconstruct_game. Tracks memory continuously
    for BOTH seats (same reasoning as training/human_replays.py) but only
    yields samples for seats in `elite_seats`.

    Drops any sample whose target action is illegal under
    action_space.compute_legal_action_mask(obs, build_castles_enabled=False)
    -- the same convention training/pretrain_bc.py's loss function assumes.
    Rare (~0.1% of ticks in practice) but real: found by tracing an
    anomalously huge BC training loss back to exactly these samples, all of
    which showed a *visible* (non-fogged) mountain at the reconstructed
    move's destination despite the move having actually succeeded when
    verified against the real engine during reconstruction -- root cause not
    fully pinned down (a rare reconstruction ambiguity picking a
    board-matching but not-the-true candidate is the leading guess), but
    filtering here is the same defensive move training/human_replays.py
    already makes for its own illegal-move edge cases, and keeps a single
    corrupted sample from ever contributing a ~1e9 loss spike mid-training
    again."""
    mem = {0: mem_mod.init_memory(CANVAS, CANVAS), 1: mem_mod.init_memory(CANVAS, CANVAS)}
    samples = []
    num_illegal = 0
    actions_by_seat = {0: PASS_ACTION, 1: PASS_ACTION}

    for state, a0, a1 in reconstructed:
        actions_by_seat[0], actions_by_seat[1] = a0, a1
        for p in (0, 1):
            if p not in elite_seats:
                continue
            obs = _pad_observation(game.get_observation(state, p))
            idx = int(asp.encode_action_index(jnp.asarray(actions_by_seat[p]), CANVAS, CANVAS))
            mask = asp.compute_legal_action_mask(obs, build_castles_enabled=False)
            if not bool(mask[idx]):
                num_illegal += 1
                continue
            samples.append((obs, mem[p], idx))

        new_state, _info = game.step(state, jnp.stack([jnp.asarray(a0), jnp.asarray(a1)]))
        for p in (0, 1):
            obs_after = _pad_observation(game.get_observation(new_state, p))
            mem[p] = mem_mod.update_memory(mem[p], obs_after, jnp.asarray(actions_by_seat[p]))

    return samples, num_illegal


def collect_bc_samples_transformer(reconstructed, elite_seats: set[int],
                                    cfg: TransformerNetworkConfig = TransformerNetworkConfig()):
    """Transformer-path twin of collect_bc_samples -- same elite-seat
    filtering and illegal-target guard, but tracks AugmentedObsState instead
    of MemoryState. aug_mod.augment_obs(obs, obs_state) is called once per
    seat per tick with the PRE-step observation already computed here (no
    separate post-step pass needed, unlike memory's update_memory -- see
    training/rollout.py's transformer-path section docstring)."""
    obs_state = {0: aug_mod.init_obs_state(cfg), 1: aug_mod.init_obs_state(cfg)}
    samples = []
    num_illegal = 0
    actions_by_seat = {0: PASS_ACTION, 1: PASS_ACTION}

    for state, a0, a1 in reconstructed:
        actions_by_seat[0], actions_by_seat[1] = a0, a1
        obs_state_next = {}
        for p in (0, 1):
            obs = _pad_observation(game.get_observation(state, p), cfg.grid_size)
            _augmented, obs_state_next[p] = aug_mod.augment_obs(obs, obs_state[p])
            if p not in elite_seats:
                continue
            idx = int(asp.encode_action_index(jnp.asarray(actions_by_seat[p]), cfg.grid_size, cfg.grid_size))
            mask = asp.compute_legal_action_mask(obs, build_castles_enabled=False)
            if not bool(mask[idx]):
                num_illegal += 1
                continue
            samples.append((obs, obs_state[p], idx))

        obs_state = obs_state_next

    return samples, num_illegal


def _pad_observation(obs, canvas: int = CANVAS):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--network", choices=("cnn", "transformer"), default="cnn",
                    help="cnn: MemoryState-based samples (default, matches the currently-deployed "
                         "architecture). transformer: AugmentedObsState-based samples, see "
                         "~/.claude/plans/drifting-questing-babbage.md")
    p.add_argument("--min-elo", type=float, default=1800.0,
                    help="only extract moves from players at or above this elo")
    p.add_argument("--min-games", type=int, default=10,
                    help="skip players with fewer rated games (noisy/provisional ratings)")
    p.add_argument("--max-games", type=int, default=None, help="cap number of unique replay games processed")
    p.add_argument("--max-board", type=int, default=CANVAS)
    return p.parse_args()


def main():
    args = parse_args()

    print("Fetching leaderboard...")
    leaderboard = fetch_leaderboard()
    elite_players = {
        row["label"] for row in leaderboard
        if row["elo"] >= args.min_elo and row["games"] >= args.min_games and not row.get("provisional")
    }
    print(f"  {len(leaderboard)} total players, {len(elite_players)} at elo>={args.min_elo} "
          f"(min {args.min_games} games, non-provisional)")

    print("Fetching match lists...")
    games_by_id: dict[str, dict] = {}
    for i, name in enumerate(sorted(elite_players)):
        for m in fetch_matches(name):
            games_by_id[m["id"]] = m
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(elite_players)} players, {len(games_by_id)} unique games so far")
    print(f"  {len(games_by_id)} unique games touch an elite player")

    game_ids = list(games_by_id.keys())
    if args.max_games:
        game_ids = game_ids[:args.max_games]

    transformer_cfg = TransformerNetworkConfig()
    collect_fn = (lambda reconstructed, elite_seats: collect_bc_samples_transformer(reconstructed, elite_seats, transformer_cfg)) \
        if args.network == "transformer" else collect_bc_samples

    all_samples = []
    total_matched, total_ticks, total_illegal = 0, 0, 0
    games_used = 0
    for i, gid in enumerate(game_ids):
        m = games_by_id[gid]
        elite_seats = set()
        if m["a_name"] in elite_players:
            elite_seats.add(0)
        if m["b_name"] in elite_players:
            elite_seats.add(1)
        if not elite_seats:
            continue

        try:
            replay = fetch_replay(gid)
        except Exception as e:
            print(f"  game {gid}: fetch failed ({e!r}), skipping")
            continue
        if replay["dims"]["rows"] > args.max_board or replay["dims"]["cols"] > args.max_board:
            continue

        reconstructed, stats = reconstruct_game(replay)
        total_matched += stats["matched"]
        total_ticks += stats["total"]
        games_used += 1

        samples, num_illegal = collect_fn(reconstructed, elite_seats)
        all_samples.extend(samples)
        total_illegal += num_illegal

        if (i + 1) % 20 == 0 or i == len(game_ids) - 1:
            rate = total_matched / max(total_ticks, 1) * 100
            print(f"  {i + 1}/{len(game_ids)} games ({games_used} used), "
                  f"{len(all_samples)} samples so far, "
                  f"tick reconstruction rate {rate:.1f}%")

    print(f"\nDone: {games_used} games, {len(all_samples)} samples, "
          f"tick reconstruction rate {total_matched / max(total_ticks, 1) * 100:.1f}% "
          f"({total_matched}/{total_ticks}), {total_illegal} illegal-target samples dropped")

    from generals.core.observation import Observation
    obs_fields = Observation._fields
    stacked = {f: np.stack([np.asarray(getattr(o, f)) for o, _s, _i in all_samples]) for f in obs_fields}
    if args.network == "transformer":
        # float16, not float32 -- see training/human_replays.py's identical note
        # (AugmentedObsState is meaningfully heavier than MemoryState per sample).
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
