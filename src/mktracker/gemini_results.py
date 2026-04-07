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
    "These are sequential frames captured from a Mario Kart results screen. "
    "IMPORTANT: The screen transitions through two phases that look nearly identical — "
    "first the RACE RESULTS (placements from this race only), then the OVERALL STANDINGS (cumulative match totals). "
    "You must only read the RACE RESULTS phase and ignore the OVERALL STANDINGS phase. "
    "To tell them apart: in race results, the +points column follows a fixed Mario Kart points scale "
    "(1st=+15, 2nd=+12, 3rd=+10, 4th=+9, 5th=+8, 6th=+7, 7th=+6, 8th=+5, 9th=+4, 10th=+3, 11th=+2, 12th=+1). "
    "In overall standings, the +points values do not follow this pattern. "
    "Discard any frames that show overall standings. "
    "The list may scroll upward within the race results phase, so earlier frames show lower placements (higher numbers) "
    "and later frames show higher placements (lower numbers). "
    "Each row follows this layout from left to right: "
    "placement number | racer icon | racer sticker | racer name | race delta (+points) | total score. "
    "The placement number is on the far left. The racer name comes after two small images (icon and sticker). "
    "Read the full racer name — it may contain special characters, clan tags, or symbols. "
    "Pay close attention to star characters: a filled star is ★ and an outline/blank star is ☆. Both may appear in player names. "
    "If a name is partially obscured in one frame, use other frames where the same player appears more clearly to complete it. "
    "The number on the far right is the cumulative total score — ignore it. "
    "A gold highlighted bar indicates the current player's row — it can appear on any placement in both phases, so it is not a useful signal for distinguishing race results from overall standings. "
    "The same player may appear in multiple race results frames as the list scrolls — use the clearest reading for their name. "
    "Determine whether the race used no teams, two teams, three teams, or four teams. "
    "The color of each player's result bar indicates which team they belong to. "
    "Players on the same team often share a common tag — a short prefix or suffix in their name (e.g. '[ABC]' or '|XYZ'). "
    "Using all frames together, reconstruct the complete placement list with no duplicates and no gaps. "
    "Return ONLY a raw JSON object. Do not wrap it in markdown, code fences, or any other formatting. No extra text before or after:\n"
    "{\n"
    '  "mode": "no_teams" | "two_teams" | "three_teams" | "four_teams",\n'
    '  "teams": [\n'
    "    {\n"
    '      "name": "Red Team",\n'
    '      "race_points": 42,\n'
    '      "race_winner": true,\n'
    '      "players": [\n'
    '        { "place": 1, "name": "PlayerA" }\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "For no_teams mode, use a single entry in 'teams' with name, race_points, and race_winner all set to null. "
    "When teams are present, 'race_points' is the sum of the +race_points values shown next to each team member's name, "
    "and 'race_winner' is true for the team with the most race_points (false for all others). "
    "Every player must appear exactly once, ordered by placement ascending."
)


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _query_gemini(frames_b64: list[str], api_key: str, model: str) -> str:
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
    parts.append({"text": _PROMPT})

    payload = {"contents": [{"parts": parts}]}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()

    return text


def _strip_markdown(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
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
) -> None:
    """Fire-and-forget: query Gemini for race results from *frames*.

    *race_num* is the 1-based race number used for logging.
    *log_dir* is an optional directory where a ``gemini_results.txt`` file
    will be written with the prompt, raw response, and parsed result.
    The *callback* is invoked from a background thread with
    ``(parsed_dict, placements)`` on success, or ``(None, [])`` on failure.
    The caller is responsible for thread-safe handling in the callback.
    """
    api_key = load_api_key()
    if not api_key:
        logger.warning("Gemini results request skipped — no API key configured")
        callback(None, [])
        return

    model = load_model()
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
            text = _query_gemini(frames_b64, api_key, model)
        except Exception:
            logger.exception("Gemini results request failed for race %d", race_num)
            _write_log([
                ("Prompt", _PROMPT),
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
                ("Prompt", _PROMPT),
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
