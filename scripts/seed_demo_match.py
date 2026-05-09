"""Generate a synthetic 30-race FFA match for UI testing.

Useful for exercising the per-race nav strip when the chip count overflows
the detail pane.  Drops a `match.json` (no debug frames) into `matches/`
under a far-future timestamp so it sorts to the top of the history list
without polluting real recorded matches.

Run: ``uv run python -m scripts.seed_demo_match``
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from mktracker.detection.tracks import TRACK_NAMES
from mktracker.match_record import (
    DEFAULT_MATCHES_DIR,
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    RaceRecord,
)
from mktracker.team_scoring import points_for_place


def main() -> None:
    rng = random.Random(0xC0FFEE)
    match_id = "99991231_235930"  # sorts to top of newest-first list
    started = datetime(9999, 12, 31, 21, 0, 0)
    settings = MatchSettingsRecord(
        cc_class="150cc",
        teams="No Teams",
        items="Normal",
        com_difficulty="Hard",
        race_count=30,
        intermission="Off",
    )
    players = [f"DemoP{i:02d}" for i in range(1, 13)]

    races: list[RaceRecord] = []
    cumulative: dict[str, int] = {p: 0 for p in players}
    tracks = list(TRACK_NAMES)
    rng.shuffle(tracks)

    for race_n in range(1, 31):
        order = list(players)
        rng.shuffle(order)
        placements = [
            PlayerPlacement(place=i + 1, name=name) for i, name in enumerate(order)
        ]
        for p in placements:
            cumulative[p.name] += points_for_place(p.place, 12)
        user_rank = next(p.place for p in placements if p.name == players[0])
        races.append(RaceRecord(
            race_number=race_n,
            track_name=tracks[(race_n - 1) % len(tracks)],
            players=list(players),
            user_rank=user_rank,
            mode="no_teams",
            placements=placements,
            teams=None,
        ))

    ranked = sorted(cumulative.items(), key=lambda kv: kv[1], reverse=True)
    final_players = [
        PlayerPlacement(place=i + 1, name=name, score=score)
        for i, (name, score) in enumerate(ranked)
    ]
    final_standings = FinalStandings(
        mode="no_teams", players=final_players, teams=None,
    )

    completed = started + timedelta(minutes=90)
    record = MatchRecord(
        match_id=match_id,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        settings=settings,
        races=races,
        final_standings=final_standings,
    )

    matches_dir = Path(__file__).resolve().parents[1] / DEFAULT_MATCHES_DIR
    match_dir = matches_dir / match_id
    record.save(match_dir)
    print(f"Wrote {match_dir / 'match.json'}")


if __name__ == "__main__":
    main()
