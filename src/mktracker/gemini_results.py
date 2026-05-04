"""Async Gemini API call to read race results from multiple frames.

Encodes several BGR frames as PNGs, sends them all in a single Gemini request
with the race-results prompt, and parses the structured JSON response.  The
call runs in a daemon thread so it never blocks the main/UI thread.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from mktracker.gemini_client import load_api_key, load_model

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_PROMPT = (
    "These are sequential frames from a Mario Kart results screen. The screen transitions "
    "through two phases that look nearly identical: first RACE RESULTS (placements from "
    "this race only), then OVERALL STANDINGS (cumulative match totals). You must only read "
    "the RACE RESULTS phase and ignore OVERALL STANDINGS. "
    "To tell them apart: in race results the +points column follows a fixed scale "
    "(1st=+15, 2nd=+12, 3rd=+10, 4th=+9, 5th=+8, 6th=+7, 7th=+6, 8th=+5, 9th=+4, 10th=+3, 11th=+2, 12th=+1). "
    "In overall standings the +points values do NOT follow this pattern. Discard those frames. "
    "The list may scroll upward within the race results phase, so earlier frames show lower "
    "placements (higher numbers) and later frames show higher placements (lower numbers). "
    "Each row, left-to-right: placement | racer icon | racer sticker | racer name | +race_points | total score (ignore the total). "
    "Read the full racer name including special characters, clan tags, and symbols. "
    "Pay close attention to star characters: a filled star is ★ and an outline/blank star is ☆. "
    "If a name is partially obscured in one frame, use other frames where the same player "
    "appears more clearly. A gold-highlighted bar marks the current player and can appear on "
    "any placement in either phase — it is NOT a signal for distinguishing the phases. "
    "Determine the team mode from bar colours (no_teams, two_teams, three_teams, or four_teams). "
    "Players on the same team often share a common tag — a short prefix or suffix in their name "
    "(e.g. '[ABC]' or '|XYZ'). "
    "For each team give race_points (sum of the +race_points values for its members) and "
    "race_winner (true for the team with the most race_points, false for others). For no_teams "
    "mode return a single team with name, race_points, and race_winner all null. "
    "Using all frames together, reconstruct the complete placement list: every player appears "
    "exactly once, ordered by placement ascending, with no duplicates and no gaps."
)

# Maps the MatchSettings.teams string to the schema-level mode hint.
_TEAMS_MODE_MAP = {
    "No Teams": ("no_teams", 1),
    "Two Teams": ("two_teams", 2),
    "Three Teams": ("three_teams", 3),
    "Four Teams": ("four_teams", 4),
}


def _build_prompt(teams_setting: str | None) -> str:
    """Return the base prompt, optionally with an authoritative team-mode hint."""
    if not teams_setting:
        return _PROMPT
    hint = _TEAMS_MODE_MAP.get(teams_setting)
    if hint is None:
        return _PROMPT
    mode, count = hint
    if count == 1:
        suffix = (
            f" The match is configured in '{teams_setting}' mode — set mode='{mode}' "
            "and return a single team containing every player with name, race_points, "
            "and race_winner all null. Do not split players into multiple teams."
        )
    else:
        suffix = (
            f" The match is configured in '{teams_setting}' mode — set mode='{mode}' "
            f"and split the 12 players into exactly {count} teams of 3 by bar colour. "
            "Trust this configuration over your own colour analysis if the frames are "
            "ambiguous; do not return no_teams or any other team count."
        )
    return _PROMPT + suffix

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["no_teams", "two_teams", "three_teams", "four_teams"],
        },
        "teams": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "nullable": True},
                    "race_points": {"type": "integer", "nullable": True},
                    "race_winner": {"type": "boolean", "nullable": True},
                    "players": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "place": {"type": "integer"},
                                "name": {"type": "string"},
                            },
                            "required": ["place", "name"],
                        },
                    },
                },
                "required": ["players"],
            },
        },
    },
    "required": ["mode", "teams"],
}


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _query_gemini(
    frames_b64: list[str],
    api_key: str,
    model: str,
    *,
    prompt: str = _PROMPT,
) -> str:
    """Send the frames to Gemini and return the raw response text."""
    url = f"{_BASE_URL}/{model}:generateContent?key={api_key}"

    parts: list[dict] = []
    for b64 in frames_b64:
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": b64,
            }
        })
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return _strip_markdown(text)


def _strip_markdown(text: str) -> str:
    """Strip leading and/or trailing markdown code fences if present.

    Handles three malformed shapes the model has been observed emitting:
    a properly fenced block, a leading fence with no closing fence, and
    raw JSON followed by a stray closing fence with no opening one.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].strip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    return text


def _parse_results(text: str) -> dict:
    """Parse the Gemini response into the structured results dict.

    Tries raw JSON first, then strips markdown fences and retries.
    Returns a dict with keys ``mode``, ``teams``.
    Raises ``ValueError`` / ``json.JSONDecodeError`` on bad input.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_strip_markdown(text))

    if "teams" not in parsed or "mode" not in parsed:
        raise ValueError("Response missing required 'teams' or 'mode' key")
    return parsed


def _placements_from_parsed(parsed: dict) -> list[tuple[int, str]]:
    """Extract a flat ``[(placement, name), ...]`` list from the parsed dict."""
    placements: list[tuple[int, str]] = []
    for team in parsed.get("teams", []):
        for player in team.get("players", []):
            place = player.get("place")
            name = player.get("name", "")
            if place is not None:
                placements.append((int(place), str(name)))
    placements.sort(key=lambda x: x[0])
    return placements


def request_race_results(
    frames: list[np.ndarray],
    race_num: int,
    callback: Callable[[dict | None, list[tuple[int, str]]], None],
    log_dir: Path | None = None,
    *,
    teams_setting: str | None = None,
) -> None:
    """Fire-and-forget: query Gemini for race results from *frames*.

    *race_num* is the 1-based race number used for logging.
    *log_dir* is an optional directory where a ``gemini_results.txt`` file
    will be written with the prompt, raw response, and parsed result.
    *teams_setting* is the authoritative ``MatchSettings.teams`` value
    (e.g. ``"Four Teams"``); when provided it is folded into the prompt as
    a hard constraint so Gemini doesn't have to guess team mode from bar
    colours alone. The *callback* is invoked from a background thread with
    ``(parsed_dict, placements)`` on success, or ``(None, [])`` on failure.
    The caller is responsible for thread-safe handling in the callback.
    """
    api_key = load_api_key()
    if not api_key:
        logger.warning("Gemini results request skipped — no API key configured")
        callback(None, [])
        return

    model = load_model()
    prompt = _build_prompt(teams_setting)
    logger.info("Requesting race results for race %d (%d frames)", race_num, len(frames))

    def _write_log(sections: list[tuple[str, str]]) -> None:
        if log_dir is None:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            for heading, body in sections:
                lines.append(f"--- {heading} ---")
                lines.append(body)
                lines.append("")
            (log_dir / "gemini_results.txt").write_text(
                "\n".join(lines), encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to write gemini results log", exc_info=True)

    def _worker() -> None:
        try:
            frames_b64 = [_encode_frame(f) for f in frames]
            text = _query_gemini(frames_b64, api_key, model, prompt=prompt)
        except Exception:
            logger.exception("Gemini results request failed for race %d", race_num)
            _write_log([
                ("Prompt", prompt),
                ("Model", model),
                ("Frames", str(len(frames))),
                ("Error", traceback.format_exc()),
            ])
            callback(None, [])
            return

        try:
            parsed = _parse_results(text)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            logger.warning(
                "Received invalid results response for race %d. Response: %s",
                race_num, text,
            )
            _write_log([
                ("Prompt", prompt),
                ("Model", model),
                ("Frames", str(len(frames))),
                ("Response", text),
                ("Result", "INVALID — could not parse results"),
            ])
            callback(None, [])
            return

        placements = _placements_from_parsed(parsed)
        logger.info(
            "Received valid results response for race %d: %d placements",
            race_num, len(placements),
        )
        _write_log([
            ("Prompt", _PROMPT),
            ("Model", model),
            ("Frames", str(len(frames))),
            ("Response", text),
            ("Parsed", json.dumps(parsed, indent=2)),
            ("Placements", "\n".join(f"  {p}. {n}" for p, n in placements)),
        ])
        callback(parsed, placements)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
