"""Discord webhook URL persisted in .env."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from dotenv import dotenv_values, set_key

_ENV_PATH = Path(".env")
_ENV_WEBHOOK = "DISCORD_WEBHOOK_URL"
_EVENT_ENV_PREFIX = "DISCORD_NOTIFY_"
_TRUTHY = {"1", "true", "yes", "on"}

# Event keys used across the app. Adding a new event = register it here and
# add a checkbox in the Settings tab + a gated call in the state machine.
EVENT_MATCH_START = "MATCH_START"


def load_webhook_url() -> str:
    """Return the stored Discord webhook URL, or an empty string if not set."""
    return dotenv_values(_ENV_PATH).get(_ENV_WEBHOOK, "") or ""


def save_webhook_url(url: str) -> None:
    """Persist *url* to .env (creates the file if needed)."""
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), _ENV_WEBHOOK, url)


def load_event_enabled(event: str) -> bool:
    """Return whether the webhook notification for *event* is enabled.

    Unset / empty values default to ``True`` — new events are opt-out.
    """
    key = _EVENT_ENV_PREFIX + event.upper()
    raw = dotenv_values(_ENV_PATH).get(key)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in _TRUTHY


def save_event_enabled(event: str, enabled: bool) -> None:
    """Persist the enabled flag for *event* to .env."""
    key = _EVENT_ENV_PREFIX + event.upper()
    _ENV_PATH.touch(exist_ok=True)
    set_key(str(_ENV_PATH), key, "true" if enabled else "false")


def _encode_multipart(
    body: dict,
    files: list[tuple[str, bytes]],
    boundary: str,
) -> tuple[bytes, str]:
    """Encode a Discord webhook payload as multipart/form-data.

    Returns ``(body_bytes, content_type_header)``. The JSON payload rides
    in the ``payload_json`` part; each file is sent as ``files[i]`` with
    the caller-supplied filename so embeds can reference it via
    ``attachment://<filename>``.
    """
    lines: list[bytes] = []
    dash = f"--{boundary}".encode()

    lines.append(dash)
    lines.append(b'Content-Disposition: form-data; name="payload_json"')
    lines.append(b"Content-Type: application/json")
    lines.append(b"")
    lines.append(json.dumps(body).encode("utf-8"))

    for i, (filename, data) in enumerate(files):
        safe = filename.replace('"', "")
        lines.append(dash)
        lines.append(
            f'Content-Disposition: form-data; name="files[{i}]"; '
            f'filename="{safe}"'.encode()
        )
        lines.append(b"Content-Type: application/octet-stream")
        lines.append(b"")
        lines.append(data)

    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    return b"\r\n".join(lines), f"multipart/form-data; boundary={boundary}"


def send_message(
    url: str,
    content: str = "",
    *,
    embeds: list[dict] | None = None,
    username: str | None = None,
    files: list[tuple[str, bytes]] | None = None,
) -> tuple[bool, str]:
    """POST a message to the Discord webhook at *url*.

    *files* is a list of ``(filename, bytes)`` pairs. When supplied the
    request is encoded as multipart/form-data and each file can be
    referenced from an embed via ``attachment://<filename>``.

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
    if not body and not files:
        return False, "Empty message payload."

    ua = "MKWorldTracker (https://github.com/n4t3m/mkworld-tracker, 0.1)"

    if files:
        boundary = uuid.uuid4().hex
        payload, content_type = _encode_multipart(body, files, boundary)
        headers = {"Content-Type": content_type, "User-Agent": ua}
    else:
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": ua}

    req = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
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
