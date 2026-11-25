"""Recognize the complete character set of an optional local game font."""
from __future__ import annotations

from functools import lru_cache
import logging
import os
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont
from dotenv import dotenv_values, set_key

from mktracker.detection.font_decoder import FontDecoder

logger = logging.getLogger(__name__)
FONT_KEY = "GAME_FONT_PATH"


def save_font_path(path: str) -> None:
    # Keep the user's font external; it is not bundled in the repository.
    if path:
        ImageFont.truetype(path, 64)  # validate before saving
    env = Path(".env")
    env.touch(exist_ok=True)
    set_key(str(env), FONT_KEY, str(Path(path).resolve()) if path else "")


@lru_cache(maxsize=4)
def _configured_path(env_path: str, modified: int) -> str:
    return dotenv_values(env_path).get(FONT_KEY, "") or ""


def configured_font_path() -> str:
    if os.environ.get(FONT_KEY):
        return os.environ[FONT_KEY]
    env = Path(".env").resolve()
    return _configured_path(str(env), env.stat().st_mtime_ns if env.exists() else 0)


class FontSymbolMatcher(FontDecoder):
    """Compatibility name for the complete font recognizer."""


@lru_cache(maxsize=3)
def _load_matcher(path: str, modified: int):
    try:
        return FontSymbolMatcher(path)
    except (OSError, ValueError):
        logger.warning("Unable to load game font: %s", path)
        return None


def correct_symbol_words(image: np.ndarray, words: list[dict], *, deskewed: bool = True) -> list[dict]:
    path = configured_font_path()
    if not path or not Path(path).is_file():
        return words
    matcher = _load_matcher(path, Path(path).stat().st_mtime_ns)
    if matcher is None:
        return words
    if not deskewed:
        h, w = image.shape[:2]
        image = cv2.warpAffine(image, np.float32([[1, .18, -.09*h], [0, 1, 0]]), (w, h), borderValue=255)
    return matcher.correct_words(image, words)


def canonical_font_text(text: str) -> str:
    """Use the same visual families for identity comparison as recognition."""
    text = unicodedata.normalize("NFC", text)
    path = configured_font_path()
    if not path or not Path(path).is_file():
        return text
    matcher = _load_matcher(path, Path(path).stat().st_mtime_ns)
    return matcher.canonical(text) if matcher is not None else text
