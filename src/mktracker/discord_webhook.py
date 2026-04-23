"""Discord webhook URL persisted in .env."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, set_key

_ENV_PATH = Path(".env")
_ENV_WEBHOOK = "DISCORD_WEBHOOK_URL"


def load_webhook_url() -> str:
    """Return the stored Discord webhook URL, or an empty string if not set."""
    return dotenv_values(_ENV_PATH).get(_ENV_WEBHOOK, "") or ""


def save_webhook_url(url: str) -> None:
    """Persist *url* to .env (creates the file if needed)."""
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), _ENV_WEBHOOK, url)
