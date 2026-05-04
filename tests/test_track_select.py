"""Tests for TrackSelectDetector against captured frames."""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.track_select import TrackSelectDetector

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "track_select")
_TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "trackselected"
)


def _load(path: str):
    frame = cv2.imread(path)
    assert frame is not None, f"Could not load {path}"
    return frame


@pytest.mark.parametrize(
    "filename, expected",
    [
        (os.path.join(_TESTDATA_DIR, "acornheights.png"), "Acorn Heights"),
        (os.path.join(_TESTDATA_DIR, "dinodinojungle.png"), "Dino Dino Jungle"),
        (os.path.join(_TESTDATA_DIR, "drybonesburnout.png"), "Dry Bones Burnout"),
        (os.path.join(_TESTDATA_DIR, "farawayoasis.png"), "Faraway Oasis"),
        (os.path.join(_TESTDATA_DIR, "shyguybazaar.png"), "Shy Guy Bazaar"),
        # Regression: "Wario Shipyard" used to fuzzy-match to "Wario Stadium"
        # because the canonical list had "Wario's Galleon" instead of the
        # actual in-game name.
        (os.path.join(_FIXTURE_DIR, "wario_shipyard.png"), "Wario Shipyard"),
        # Regression: bright team-color bars on the left panel + dark map on
        # the right defeated the old left/right brightness pre-check.
        (os.path.join(_FIXTURE_DIR, "acorn_heights_team_mode.png"), "Acorn Heights"),
        (os.path.join(_FIXTURE_DIR, "wario_stadium_team_mode.png"), "Wario Stadium"),
        # Bowser's Castle: 24-player layout widens the player panel beyond
        # the old 42% boundary, also breaking the pre-check.
        (os.path.join(_TESTDATA_DIR, "bowserscastle.png"), "Bowser's Castle"),
    ],
)
def test_detects_track_name(filename, expected):
    detector = TrackSelectDetector()
    result = detector.detect(_load(filename))
    assert result is not None, f"No track detected for {filename}"
    assert result["track_name"] == expected
