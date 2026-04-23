"""Discord webhook URL persisted in .env."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
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


def send_message(
    url: str,
    content: str = "",
    *,
    embeds: list[dict] | None = None,
    username: str | None = None,
) -> tuple[bool, str]:
    """POST a message to the Discord webhook at *url*.

    Returns ``(True, "")`` on success, or ``(False, error_message)`` on failure.
    """
    if not url.strip():
        return False, "No webhook URL entered."

    body: dict = {}
    if content:
        body["content"] = content
    if embeds:
        body["embeds"] = embeds
    if username:
        body["username"] = username
    if not body:
        return False, "Empty message payload."

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            # Discord's Cloudflare edge returns 403 for the default
            # "Python-urllib/x.y" UA — send a descriptive one instead.
            "User-Agent": "MKWorldTracker (https://github.com/n4t3m/mkworld-tracker, 0.1)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, ""
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            msg = body.get("message", str(e))
        except Exception:
            msg = str(e)
        return False, msg
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"
    except Exception as e:
        return False, str(e)
