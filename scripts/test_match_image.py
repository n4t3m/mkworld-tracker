"""One-shot test: run Gemini match-results on a single image and generate a table.

Usage:
    uv run python -m scripts.test_match_image <image_path>
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force stdout to UTF-8 so Unicode player names (↔, ⓑ, etc.) don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from mktracker.gemini_match_results import (
    _encode_frame,
    _parse_results,
    _query_gemini,
)
from mktracker.gemini_client import load_api_key, load_model
from mktracker.match_record import (
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    TeamGroup,
)
from mktracker.table_generator import generate_table


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run python -m scripts.test_match_image <image_path>")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"File not found: {image_path}")

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("No Gemini API key configured — add one in the Settings tab or .env")

    model = load_model()
    print(f"Model: {model}")
    print(f"Image: {image_path}")
    print("Querying Gemini…")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"cv2 could not read image: {image_path}")

    image_b64 = _encode_frame(frame)
    raw = _query_gemini(image_b64, api_key, model)

    print("\n=== Raw Gemini JSON ===")
    try:
        pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        print(pretty)
    except Exception:
        print(raw)

    print("\n=== Parsed result ===")
    parsed = _parse_results(raw)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    # Build a minimal MatchRecord so we can run generate_table
    mode = parsed.get("mode", "no_teams")
    teams_data = parsed.get("teams", [])

    all_players: list[PlayerPlacement] = []
    team_groups: list[TeamGroup] = []

    for team in teams_data:
        players = [
            PlayerPlacement(
                place=int(p["place"]),
                name=str(p["name"]),
                score=int(p.get("score", 0)),
            )
            for p in team.get("players", [])
        ]
        all_players.extend(players)
        team_groups.append(
            TeamGroup(
                name=team.get("name"),
                tag=team.get("tag"),
                points=team.get("points"),
                winner=team.get("winner"),
                players=players,
            )
        )

    all_players.sort(key=lambda p: p.place)

    final_standings = FinalStandings(
        mode=mode,
        players=all_players,
        teams=team_groups if mode != "no_teams" else None,
    )

    now = datetime.now(timezone.utc).isoformat()
    match_id = f"test_{image_path.stem}"

    record = MatchRecord(
        match_id=match_id,
        started_at=now,
        completed_at=now,
        settings=MatchSettingsRecord(
            cc_class="150cc",
            teams=mode,
            items="All Items",
            com_difficulty="Normal",
            race_count=0,
            intermission="No",
        ),
        races=[],
        final_standings=final_standings,
    )

    print("\nGenerating table…")
    png = generate_table(record)

    out = image_path.parent / f"{image_path.stem}_table.png"
    out.write_bytes(png)
    print(f"Table saved: {out}")


if __name__ == "__main__":
    main()
