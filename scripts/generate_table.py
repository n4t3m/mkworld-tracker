"""Generate a Lorenzi-style table image from a match record.

Usage:
    uv run python -m scripts.generate_table [match_id]

If match_id is omitted, uses the most recently completed match with final
standings.  Saves the PNG alongside the match folder as ``table.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/generate_table.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mktracker.match_record import list_matches, MatchRecord
from mktracker.table_generator import generate_table

MATCHES_DIR = Path("matches")


def _find_record(match_id: str | None) -> tuple[MatchRecord, Path]:
    records = list_matches(MATCHES_DIR)
    if not records:
        raise SystemExit("No match records found in matches/")

    if match_id:
        for r in records:
            if r.match_id == match_id:
                return r, MATCHES_DIR / r.match_id
        raise SystemExit(f"Match '{match_id}' not found.")

    # Pick the first record that has final standings with players
    for r in records:
        if r.final_standings and r.final_standings.players:
            return r, MATCHES_DIR / r.match_id

    raise SystemExit("No completed match with final standings found.")


def main() -> None:
    match_id = sys.argv[1] if len(sys.argv) > 1 else None
    record, match_dir = _find_record(match_id)

    print(f"Generating table for match {record.match_id} …")
    print(f"  Players in standings: {len(record.final_standings.players)}")
    print(f"  Races: {len(record.races)}")

    png = generate_table(record)

    out = match_dir / "table.png"
    out.write_bytes(png)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
