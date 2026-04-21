from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest
from PIL import ImageFont as _PILImageFont

# Create the stub font once, before any patching, using the real truetype path
# that load_default bundles internally (base64-encoded Aileron Regular).
# After the patch is active, truetype calls would recurse; this avoids that.
_STUB_FONT = _PILImageFont.load_default(size=20)


@pytest.fixture(scope="session")
def stub_fonts():
    """Patch font loading so generate_table tests run without real font files.

    _ensure_font normally downloads TTF files from Google Fonts on first run.
    ImageFont.truetype normally opens those files.  Both are replaced here:
    - _ensure_font returns a dummy Path (never opened as a file)
    - ImageFont.truetype returns a pre-built Pillow built-in font

    Tests checking PNG dimensions/format/mode are unaffected by render fidelity.
    """

    def _stub_truetype(path, size=12, **kwargs):
        return _STUB_FONT

    with (
        unittest.mock.patch(
            "mktracker.table_generator._ensure_font", lambda name: Path(name)
        ),
        unittest.mock.patch("PIL.ImageFont.truetype", _stub_truetype),
    ):
        yield
