"""Async Gemini API call to read final match results from a single frame.

Encodes a BGR frame as PNG, sends it to the Gemini API with the match-results
prompt, and parses the structured JSON response.  The call runs in a daemon
thread so it never blocks the main/UI thread.
"""

from __future__ import annotations

import base64
import json
import logging
import re
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
    "Analyse this Mario Kart World results screen and return the structured results. "
    "Determine the team mode (no_teams, two_teams, three_teams, or four_teams) from how "
    "players are split into coloured columns or panels. Each player's team is decided by "
    "the COLOUR of their result bar, not by their name. A shared name tag (e.g. 'Ly ...' "
    "or '[ABC]') is only a hint — the bar colour is authoritative. "
    "CRITICAL PARTITION RULE: every visible player belongs to EXACTLY ONE team. The "
    "teams[*].players arrays must be disjoint — a given player (by name and by placement) "
    "must appear in one and only one team's players list. The sum of len(team.players) "
    "across all teams must equal the total number of result bars visible on screen. Never "
    "duplicate a player across teams. "
    "CLAN TAG RULE: inspect the player names within each team. If the majority share a "
    "common prefix or bracket tag (e.g. 'GK', 'Zog Zhit', '[ABC]'), extract the LONGEST "
    "common prefix shared by those players as the team's 'tag' field, then strip that "
    "prefix — including any trailing space or separator — from every player's 'name' so "
    "that 'name' contains only the player-unique part (e.g. 'Zog Zhit A' → tag='Zog Zhit', "
    "name='A'; 'GK Breezy' → tag='GK', name='Breezy'). If players on a team have no "
    "common prefix, set 'tag' to null and leave 'name' unchanged. "
    "For each team give its displayed total points, whether it won (winner=true for the "
    "team with the highest points, false for all others), and every player with their final "
    "OVERALL placement (1..N across ALL players, not per-team), display name, and score. "
    "For no_teams mode return a single team entry with name, points, and winner all null. "
    "If no player result bars are visible (only the CONGRATULATIONS! or NICE TRY! banner "
    "with confetti and no player names loaded), return an empty teams array. "
    "Do NOT invent player names, scores, or placements — only report what you can read."
)

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
                    "tag": {"type": "string", "nullable": True},
                    "points": {"type": "integer", "nullable": True},
                    "winner": {"type": "boolean", "nullable": True},
                    "players": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "place": {"type": "integer"},
                                "name": {"type": "string"},
                                "score": {"type": "integer"},
                            },
                            "required": ["place", "name", "score"],
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


def _query_gemini(image_b64: str, api_key: str, model: str) -> str:
    """Send the image to Gemini and return the raw response text."""
    url = f"{_BASE_URL}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"text": _PROMPT},
                ],
            },
        ],
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

    candidate = result["candidates"][0]
    parts = candidate.get("content", {}).get("parts")
    if not parts:
        finish = candidate.get("finishReason", "UNKNOWN")
        raise RuntimeError(f"Gemini returned no content (finishReason={finish})")
    text = parts[0]["text"]
    return _strip_markdown(text)


def _repair_json(text: str) -> str:
    """Apply forgiving repairs to common model-emitted JSON glitches.

    Covers observed failure modes of gemma-4-31b-it and similar:
      * Duplicated consecutive `"key": "key":` sequences → collapse to one.
      * Trailing commas before `]` or `}`.
      * Stray text before `{` / after the matching `}`.

    NOTE: with structured outputs (responseSchema + responseMimeType) this
    path is no longer exercised on the happy path. Kept as a fallback for
    models/configs that don't support structured outputs, and for future
    reuse in the other gemini_* modules which still go through the plain
    JSON path.
    """
    repaired = text

    # Trim to the outermost {...} block if there's leading/trailing prose.
    first = repaired.find("{")
    last = repaired.rfind("}")
    if first != -1 and last != -1 and last > first:
        repaired = repaired[first : last + 1]

    # Collapse duplicated `"<key>": "<key>":` runs (model sometimes emits the
    # key twice in a row, e.g. `"name": "name": "value"`). Handles any key.
    repaired = re.sub(
        r'("[A-Za-z_][A-Za-z0-9_]*")\s*:\s*\1\s*:',
        r"\1:",
        repaired,
    )

    # Strip trailing commas before ] or }.
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    return repaired


def _parse_results(text: str) -> dict:
    """Parse the Gemini response into the structured results dict.

    Tries raw JSON, then markdown-stripped JSON, then forgiving-repaired JSON.
    Returns a dict with keys ``mode``, ``teams``.
    Raises ``ValueError`` / ``json.JSONDecodeError`` on bad input.
    """
    attempts = (
        text,
        _strip_markdown(text),
        _repair_json(text),
    )

    parsed = None
    last_err: Exception | None = None
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as err:
            last_err = err
            continue

    if parsed is None:
        assert last_err is not None
        raise last_err

    if "teams" not in parsed or "mode" not in parsed:
        raise ValueError("Response missing required 'teams' or 'mode' key")
    return parsed


def _final_results_from_parsed(parsed: dict) -> list[tuple[str, int]]:
    """Extract ``[(name, score), ...]`` sorted by placement from the parsed dict."""
    results: list[tuple[int, str, int]] = []
    for team in parsed.get("teams", []):
        for player in team.get("players", []):
            place = player.get("place")
            name = player.get("name", "")
            score = player.get("score", 0)
            if place is not None:
                results.append((int(place), str(name), int(score)))
    results.sort(key=lambda x: x[0])
    return [(name, score) for _, name, score in results]


def request_match_results(
    frame: np.ndarray,
    callback: Callable[[dict | None, list[tuple[str, int]]], None],
    log_dir: Path | None = None,
) -> None:
    """Fire-and-forget: query Gemini for match results from *frame*.

    *log_dir* is an optional directory where a ``gemini_match_results.txt``
    file will be written with the prompt, raw response, and parsed result.
    The *callback* is invoked from a background thread with
    ``(parsed_dict, results)`` on success, or ``(None, [])`` on failure.
    The caller is responsible for thread-safe handling in the callback.
    """
    api_key = load_api_key()
    if not api_key:
        logger.warning("Gemini match results request skipped — no API key configured")
        callback(None, [])
        return

    model = load_model()
    logger.info("Requesting match results from Gemini")

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
            (log_dir / "gemini_match_results.txt").write_text(
                "\n".join(lines), encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to write gemini match results log", exc_info=True)

    def _worker() -> None:
        try:
            image_b64 = _encode_frame(frame)
            text = _query_gemini(image_b64, api_key, model)
        except Exception:
            logger.exception("Gemini match results request failed")
            _write_log([
                ("Prompt", _PROMPT),
                ("Model", model),
                ("Error", traceback.format_exc()),
            ])
            callback(None, [])
            return

        try:
            parsed = _parse_results(text)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            logger.warning(
                "Received invalid match results response. Response: %s", text,
            )
            _write_log([
                ("Prompt", _PROMPT),
                ("Model", model),
                ("Response", text),
                ("Result", "INVALID — could not parse match results"),
            ])
            callback(None, [])
            return

        results = _final_results_from_parsed(parsed)
        logger.info(
            "Received valid match results response: %d players", len(results),
        )
        _write_log([
            ("Prompt", _PROMPT),
            ("Model", model),
            ("Response", text),
            ("Parsed", json.dumps(parsed, indent=2, ensure_ascii=False)),
            ("Results", "\n".join(f"  {n}: {s} pts" for n, s in results)),
        ])
        callback(parsed, results)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
