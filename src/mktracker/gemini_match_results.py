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
    "This is a Mario Kart World results screen. "
    "Determine whether the match used no teams, two teams, three teams, or four teams. "
    "If players are sorted into separate columns by color, each column represents one team. "
    "Players on the same team often share a common tag — a short prefix or suffix in their name (e.g. '[ABC]' or '|XYZ'). "
    "Return ONLY a raw JSON object with no markdown formatting, no code fences, and no extra text. Use exactly this structure:\n"
    "{\n"
    '  "mode": "no_teams" | "two_teams" | "three_teams" | "four_teams",\n'
    '  "teams": [\n'
    "    {\n"
    '      "name": "Red Team",\n'
    '      "points": 42,\n'
    '      "winner": true,\n'
    '      "players": [\n'
    '        { "place": 1, "name": "PlayerA", "score": 100 }\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "For no_teams mode, use a single entry in 'teams' with name, points, and winner all set to null. "
    "When teams are present, 'points' is the team's total point tally shown on screen, and 'winner' is true for the team with the highest points (false for all others). "
    "Every player must appear exactly once with their final placement number, display name, and total score. "
    "CRITICAL: If no player result bars are visible in the image — for example, if only the banner text "
    "(CONGRATULATIONS! or NICE TRY!) is shown with confetti but no player names or scores have loaded yet — "
    "you MUST return {\"mode\": \"no_teams\", \"teams\": []} with an empty teams array. "
    "Do NOT hallucinate or invent player names, scores, or placements. Only report data you can actually read from the image. "
    "VALID JSON RESPONSE ONLY. VALID JSON RESPONSE ONLY. VALID JSON RESPONSE ONLY. DO NOT GIVE ME MARKDOWN. DO NOT DARE GIVE ME MARKDOWN"
)


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _strip_markdown(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (may include language tag like ```json)
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    return text


def _query_gemini(image_b64: str, api_key: str, model: str) -> str:
    """Send the image to Gemini and return the raw response text."""
    url = f"{_BASE_URL}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"text": _PROMPT},
                ]
            }
        ]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return _strip_markdown(text)


def _parse_results(text: str) -> dict:
    """Parse the Gemini response into the structured results dict.

    Tries raw JSON first, then strips markdown fences and retries.
    Returns a dict with keys ``mode``, ``teams``.
    Raises ``ValueError`` / ``json.JSONDecodeError`` on bad input.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Second attempt: strip any remaining markdown formatting.
        stripped = _strip_markdown(text)
        parsed = json.loads(stripped)

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
