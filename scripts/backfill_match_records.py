"""Rebuild ``match.json`` for legacy matches in ``matches/``.

Walks every subfolder of ``matches/`` that lacks ``match.json`` and
reconstructs one from the saved frames.  Re-runs Gemini synchronously for
race rank, race results, and final match results (mirroring the live app),
and uses the normal OCR detectors for match settings, track names, and
player lists.  Existing ``gemini_*.txt`` logs are NOT reused — every race
gets a fresh Gemini call so broken/errored logs don't poison the output.

Usage:
    python -m scripts.backfill_match_records [--matches-dir matches]
                                             [--force]
                                             [--max-placement-frames 12]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mktracker.detection.match_settings import (
    MatchSettings,
    MatchSettingsDetector,
)
from mktracker.detection.player_reader import PlayerReader
from mktracker.detection.track_select import TrackSelectDetector
from mktracker.gemini_client import load_api_key, load_model
from mktracker.gemini_match_results import (
    _PROMPT as MATCH_RESULTS_PROMPT,
    _encode_frame as encode_match_results_frame,
    _parse_results as parse_match_results,
    _query_gemini as query_match_results,
)
from mktracker.gemini_rank import (
    _PROMPT as RANK_PROMPT,
    _encode_frame as encode_rank_frame,
    _parse_rank,
    _query_gemini as query_rank,
)
from mktracker.gemini_results import (
    _build_prompt as build_results_prompt,
    _PROMPT as RESULTS_PROMPT,
    _encode_frame as encode_results_frame,
    _parse_results as parse_race_results,
    _query_gemini as query_race_results,
)
from mktracker.match_record import (
    DEFAULT_MATCHES_DIR,
    FinalStandings,
    MATCH_FILE,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    RaceRecord,
    TeamGroup,
)

logger = logging.getLogger(__name__)

_FALLBACK_SETTINGS = MatchSettings(
    cc_class="150cc",
    teams="No Teams",
    items="Normal",
    com_difficulty="Normal",
    race_count=12,
    intermission="10 seconds",
)

_RACE_DIR_RE = re.compile(r"^race_(\d+)$")
_PLACEMENT_FRAME_RE = re.compile(r"^placement_(\d+)\.png$")
_TIMESTAMP_RE = re.compile(r"^(\d{8})_(\d{6})$")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _read_frame(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = cv2.imread(str(path))
    return frame if frame is not None else None


def _sorted_placement_frames(race_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for p in race_dir.iterdir():
        m = _PLACEMENT_FRAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]


def _sample_evenly(items: list[Any], max_items: int) -> list[Any]:
    if len(items) <= max_items:
        return list(items)
    step = (len(items) - 1) / (max_items - 1)
    return [items[round(i * step)] for i in range(max_items)]


# ---------------------------------------------------------------------------
# Synchronous Gemini wrappers
# ---------------------------------------------------------------------------

def _log_gemini(
    log_dir: Path, filename: str, sections: list[tuple[str, str]],
) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for heading, body in sections:
            lines.append(f"--- {heading} ---")
            lines.append(body)
            lines.append("")
        (log_dir / filename).write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        logger.debug("Failed to write %s", filename, exc_info=True)


def _call_rank(
    frame: np.ndarray, api_key: str, model: str, log_dir: Path,
) -> int | None:
    try:
        b64 = encode_rank_frame(frame)
        text = query_rank(b64, api_key, model)
    except Exception:
        _log_gemini(log_dir, "gemini_rank.txt", [
            ("Prompt", RANK_PROMPT),
            ("Model", model),
            ("Error", traceback.format_exc()),
        ])
        return None

    try:
        rank = _parse_rank(text)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        _log_gemini(log_dir, "gemini_rank.txt", [
            ("Prompt", RANK_PROMPT),
            ("Model", model),
            ("Response", text),
            ("Result", "INVALID — could not parse rank"),
        ])
        return None

    _log_gemini(log_dir, "gemini_rank.txt", [
        ("Prompt", RANK_PROMPT),
        ("Model", model),
        ("Response", text),
        ("Result", str(rank)),
    ])
    return rank


def _call_race_results(
    frames: list[np.ndarray], api_key: str, model: str, log_dir: Path,
    *, teams_setting: str | None = None,
) -> dict | None:
    prompt = build_results_prompt(teams_setting)
    try:
        frames_b64 = [encode_results_frame(f) for f in frames]
        text = query_race_results(frames_b64, api_key, model, prompt=prompt)
    except Exception:
        _log_gemini(log_dir, "gemini_results.txt", [
            ("Prompt", prompt),
            ("Model", model),
            ("Frames", str(len(frames))),
            ("Error", traceback.format_exc()),
        ])
        return None

    try:
        parsed = parse_race_results(text)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        _log_gemini(log_dir, "gemini_results.txt", [
            ("Prompt", prompt),
            ("Model", model),
            ("Frames", str(len(frames))),
            ("Response", text),
            ("Result", "INVALID — could not parse results"),
        ])
        return None

    _log_gemini(log_dir, "gemini_results.txt", [
        ("Prompt", prompt),
        ("Model", model),
        ("Frames", str(len(frames))),
        ("Response", text),
        ("Parsed", json.dumps(parsed, indent=2)),
    ])
    return parsed


def _call_match_results(
    frame: np.ndarray, api_key: str, model: str, log_dir: Path,
) -> dict | None:
    try:
        b64 = encode_match_results_frame(frame)
        text = query_match_results(b64, api_key, model)
    except Exception:
        _log_gemini(log_dir, "gemini_match_results.txt", [
            ("Prompt", MATCH_RESULTS_PROMPT),
            ("Model", model),
            ("Error", traceback.format_exc()),
        ])
        return None

    try:
        parsed = parse_match_results(text)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        _log_gemini(log_dir, "gemini_match_results.txt", [
            ("Prompt", MATCH_RESULTS_PROMPT),
            ("Model", model),
            ("Response", text),
            ("Result", "INVALID — could not parse results"),
        ])
        return None

    _log_gemini(log_dir, "gemini_match_results.txt", [
        ("Prompt", MATCH_RESULTS_PROMPT),
        ("Model", model),
        ("Response", text),
        ("Parsed", json.dumps(parsed, indent=2)),
    ])
    return parsed


# ---------------------------------------------------------------------------
# OCR detectors (same as live app)
# ---------------------------------------------------------------------------

def _detect_settings(path: Path) -> MatchSettings:
    frame = _read_frame(path)
    if frame is None:
        return _FALLBACK_SETTINGS
    settings = MatchSettingsDetector().detect(frame)
    if settings is None:
        logger.info("    settings: detector returned None → using fallback")
        return _FALLBACK_SETTINGS
    return settings


def _detect_track_name(path: Path) -> str | None:
    frame = _read_frame(path)
    if frame is None:
        return None
    detector = TrackSelectDetector()
    if not detector.is_active(frame):
        return None
    raw = detector.read_track_name(frame)
    if raw is None:
        return None
    return detector.match_track_name(raw)


def _detect_players(path: Path, *, teams: bool) -> list[str]:
    frame = _read_frame(path)
    if frame is None:
        return []
    return [p.name for p in PlayerReader().read_players(frame, teams=teams)]


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------

def _race_structures_from_results(
    data: dict[str, Any],
) -> tuple[str | None, list[PlayerPlacement], list[TeamGroup] | None]:
    mode = data.get("mode")
    teams_data = data.get("teams") or []

    team_groups: list[TeamGroup] = []
    flat: list[PlayerPlacement] = []
    for team in teams_data:
        players = [
            PlayerPlacement(place=int(p["place"]), name=str(p["name"]))
            for p in team.get("players", [])
            if "place" in p and "name" in p
        ]
        flat.extend(players)
        team_groups.append(
            TeamGroup(
                name=team.get("name"),
                points=team.get("race_points", team.get("points")),
                winner=team.get("race_winner", team.get("winner")),
                players=players,
            )
        )

    flat.sort(key=lambda p: p.place)
    has_real_teams = bool(mode and mode != "no_teams" and team_groups)
    return mode, flat, team_groups if has_real_teams else None


def _final_standings_from_dict(data: dict[str, Any]) -> FinalStandings:
    mode = data.get("mode")
    teams_data = data.get("teams") or []

    team_groups: list[TeamGroup] = []
    flat: list[PlayerPlacement] = []
    for team in teams_data:
        players = [
            PlayerPlacement(
                place=int(p["place"]),
                name=str(p["name"]),
                score=p.get("score"),
            )
            for p in team.get("players", [])
            if "place" in p and "name" in p
        ]
        flat.extend(players)
        team_groups.append(
            TeamGroup(
                name=team.get("name"),
                points=team.get("points"),
                winner=team.get("winner"),
                players=players,
            )
        )

    flat.sort(key=lambda p: p.place)
    has_real_teams = bool(mode and mode != "no_teams" and team_groups)
    return FinalStandings(
        mode=mode,
        players=flat,
        teams=team_groups if has_real_teams else None,
    )


def _iso_from_folder(folder: Path) -> str:
    m = _TIMESTAMP_RE.match(folder.name)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S",
            ).isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(folder.stat().st_mtime).isoformat()


def _build_race_record(
    race_dir: Path, race_number: int, teams_setting: str,
    api_key: str, model: str, max_frames: int,
) -> RaceRecord:
    teams_mode_ocr = teams_setting != "No Teams"

    track_name = _detect_track_name(race_dir / "track.png")
    players = _detect_players(race_dir / "players.png", teams=teams_mode_ocr)

    # Race rank — Gemini on rank.png.
    user_rank = None
    rank_frame = _read_frame(race_dir / "rank.png")
    if rank_frame is not None:
        user_rank = _call_rank(rank_frame, api_key, model, race_dir)

    # Race results — Gemini on placement frames.
    mode: str | None = None
    placements: list[PlayerPlacement] = []
    teams: list[TeamGroup] | None = None
    placement_paths = _sorted_placement_frames(race_dir)
    if placement_paths:
        sampled = _sample_evenly(placement_paths, max_frames)
        frames = [f for f in (_read_frame(p) for p in sampled) if f is not None]
        logger.info(
            "    race %d: %d placement frames → sampling %d → sending to Gemini",
            race_number, len(placement_paths), len(frames),
        )
        if frames:
            data = _call_race_results(
                frames, api_key, model, race_dir,
                teams_setting=teams_setting,
            )
            if data is not None:
                mode, placements, teams = _race_structures_from_results(data)

    return RaceRecord(
        race_number=race_number,
        track_name=track_name,
        players=players,
        user_rank=user_rank,
        mode=mode,
        placements=placements,
        teams=teams,
    )


def _build_match_record(
    folder: Path, api_key: str, model: str, max_frames: int,
) -> MatchRecord:
    logger.info("  reading match_settings.png")
    settings = _detect_settings(folder / "match_settings.png")

    race_dirs: list[tuple[int, Path]] = []
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        m = _RACE_DIR_RE.match(sub.name)
        if m:
            race_dirs.append((int(m.group(1)), sub))
    race_dirs.sort(key=lambda x: x[0])
    logger.info("  found %d race folder(s)", len(race_dirs))

    races = [
        _build_race_record(rdir, num, settings.teams, api_key, model, max_frames)
        for num, rdir in race_dirs
    ]

    final_standings = None
    completed_at = None
    mr_frame = _read_frame(folder / "match_results.png")
    if mr_frame is not None:
        logger.info("  match_results.png → Gemini")
        data = _call_match_results(mr_frame, api_key, model, folder)
        if data is not None:
            final_standings = _final_standings_from_dict(data)
            completed_at = _iso_from_folder(folder)

    return MatchRecord(
        match_id=folder.name,
        started_at=_iso_from_folder(folder),
        completed_at=completed_at,
        settings=MatchSettingsRecord(
            cc_class=settings.cc_class,
            teams=settings.teams,
            items=settings.items,
            com_difficulty=settings.com_difficulty,
            race_count=settings.race_count,
            intermission=settings.intermission,
        ),
        races=races,
        final_standings=final_standings,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR)
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if match.json already exists.",
    )
    parser.add_argument(
        "--max-placement-frames", type=int, default=12,
        help="Max placement frames sent per race (evenly sampled).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api_key = load_api_key()
    model = load_model()
    if not api_key:
        logger.error(
            "No Gemini API key configured. Set one via the Settings tab or .env.",
        )
        return
    logger.info("Using Gemini model: %s", model)

    if not args.matches_dir.exists():
        logger.error("Directory not found: %s", args.matches_dir)
        return

    folders = sorted(d for d in args.matches_dir.iterdir() if d.is_dir())
    logger.info("Scanning %d folder(s) under %s", len(folders), args.matches_dir)

    built = skipped = 0
    for folder in folders:
        match_path = folder / MATCH_FILE
        if match_path.exists() and not args.force:
            logger.info("[skip] %s (already has %s)", folder.name, MATCH_FILE)
            skipped += 1
            continue
        logger.info("[build] %s", folder.name)
        record = _build_match_record(folder, api_key, model, args.max_placement_frames)
        record.save(folder)
        logger.info(
            "  → saved: %s/%s · %d race(s) · final_standings=%s",
            record.settings.cc_class, record.settings.teams,
            len(record.races),
            "yes" if record.final_standings else "no",
        )
        built += 1

    logger.info("Done. built=%d, skipped=%d", built, skipped)


if __name__ == "__main__":
    main()
