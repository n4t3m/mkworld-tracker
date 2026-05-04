"""Async Gemini API call to read race placement rank from a frame.

Encodes a BGR frame as PNG, sends it to the Gemini API with the rank-reading
prompt, and parses the JSON response.  The call runs in a daemon thread so
it never blocks the main/UI thread.
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
    "This is a cropped image from the bottom-right of a Mario Kart gameplay screen. "
    "It may show a race placement rank indicator (e.g. 1st, 2nd, 12th, 24th) — a large "
    "styled number with an ordinal suffix (st, nd, rd, th) in a bold italic 3D font with "
    "a dark outline. Text colour varies: yellow/lime for 1st, blue-white for 2nd, "
    "red/salmon for 3rd, orange for 4th-24th. "
    "Return the integer placement (1-24) the image shows, or null if no rank indicator is visible."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"rank": {"type": "integer", "nullable": True}},
    "required": ["rank"],
}


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


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

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    return _strip_markdown(text)


def _strip_markdown(text: str) -> str:
    """Strip leading and/or trailing markdown code fences if present.

    Handles three malformed shapes the model has been observed emitting:
    a properly fenced block, a leading fence with no closing fence, and
    raw text followed by a stray closing fence with no opening one.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].strip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    return text


def _parse_rank(text: str) -> int | None:
    """Parse a rank integer from the Gemini response text.

    Tries raw JSON first, then strips markdown fences and retries.
    Returns the rank (1-24) or ``None``.
    Raises ``ValueError`` if the text is not valid JSON or has unexpected shape.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_strip_markdown(text))

    rank = parsed.get("rank")
    if rank is not None:
        rank = int(rank)
    return rank


def request_race_rank(
    frame: np.ndarray,
    race_num: int,
    callback: Callable[[int | None], None],
    log_dir: Path | None = None,
) -> None:
    """Fire-and-forget: query Gemini for the race rank in *frame*.

    *race_num* is the 1-based race number used for logging.
    *log_dir* is an optional directory where a ``gemini_rank.txt`` file will
    be written with the prompt, raw response, and parsed result.
    The *callback* is invoked from a background thread with the integer rank
    (1-24) or ``None`` if detection failed or the API returned null.
    The caller is responsible for thread-safe handling in the callback.
    """
    api_key = load_api_key()
    if not api_key:
        logger.warning("Gemini rank request skipped — no API key configured")
        callback(None)
        return

    model = load_model()
    logger.info("Requesting rank for race %d", race_num)

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
            (log_dir / "gemini_rank.txt").write_text(
                "\n".join(lines), encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to write gemini rank log", exc_info=True)

    def _worker() -> None:
        try:
            image_b64 = _encode_frame(frame)
            text = _query_gemini(image_b64, api_key, model)
        except Exception:
            logger.exception("Gemini rank request failed for race %d", race_num)
            _write_log([
                ("Prompt", _PROMPT),
                ("Model", model),
                ("Error", traceback.format_exc()),
            ])
            callback(None)
            return

        try:
            rank = _parse_rank(text)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            logger.warning(
                "Received invalid rank response for race %d. Response: %s",
                race_num, text,
            )
            _write_log([
                ("Prompt", _PROMPT),
                ("Model", model),
                ("Response", text),
                ("Result", "INVALID — could not parse rank"),
            ])
            callback(None)
            return

        logger.info("Received valid rank response for race %d: %s", race_num, rank)
        _write_log([
            ("Prompt", _PROMPT),
            ("Model", model),
            ("Response", text),
            ("Result", str(rank)),
        ])
        callback(rank)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
