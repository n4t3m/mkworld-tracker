"""Gemini API client: key/model persistence and health check."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values, set_key

_ENV_PATH = Path(".env")
_ENV_KEY = "GEMINI_API_KEY"
_ENV_MODEL = "GEMINI_MODEL"
_DEFAULT_MODEL = "gemma-4-31b-it"
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

SUGGESTED_MODELS: tuple[str, ...] = (
    "gemma-4-31b-it",
    "gemma-3-27b-it",
    "gemini-3.1-flash-lite-preview",
)


def load_api_key() -> str:
    """Return the stored API key, or an empty string if not set."""
    return dotenv_values(_ENV_PATH).get(_ENV_KEY, "") or ""


def save_api_key(key: str) -> None:
    """Persist *key* to .env (creates the file if needed)."""
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), _ENV_KEY, key)


def load_model() -> str:
    """Return the stored model name, falling back to the default."""
    return dotenv_values(_ENV_PATH).get(_ENV_MODEL, "") or _DEFAULT_MODEL


def save_model(model: str) -> None:
    """Persist *model* to .env."""
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), _ENV_MODEL, model)


def verify_api_key(key: str, model: str) -> tuple[bool, str]:
    """Verify *key* by fetching model metadata (GET — no tokens consumed).

    Returns ``(True, "")`` on success, or ``(False, error_message)`` on failure.
    """
    if not key.strip():
        return False, "No API key entered."

    url = f"{_BASE_URL}/{model}?key={key}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, ""
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            msg = body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return False, msg
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"
    except Exception as e:
        return False, str(e)
