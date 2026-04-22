"""Debug-mode flag persisted in .env."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, set_key

_ENV_PATH = Path(".env")
_ENV_DEBUG = "DEBUG_MODE"
_TRUTHY = {"1", "true", "yes", "on"}


def load_debug_mode() -> bool:
    """Return the persisted debug-mode flag (default False)."""
    raw = dotenv_values(_ENV_PATH).get(_ENV_DEBUG, "") or ""
    return raw.strip().lower() in _TRUTHY


def save_debug_mode(enabled: bool) -> None:
    """Persist the debug-mode flag to .env."""
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), _ENV_DEBUG, "true" if enabled else "false")
