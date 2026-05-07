"""Tests for TrackSelectDetector against captured frames."""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.track_select import TrackSelectDetector

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "track_select")
_NEGATIVE_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "track_select_negative"
)
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
        # Original testdata fixtures (no-team layouts).
        (os.path.join(_TESTDATA_DIR, "acornheights.png"), "Acorn Heights"),
        (os.path.join(_TESTDATA_DIR, "dinodinojungle.png"), "Dino Dino Jungle"),
        (os.path.join(_TESTDATA_DIR, "drybonesburnout.png"), "Dry Bones Burnout"),
        (os.path.join(_TESTDATA_DIR, "farawayoasis.png"), "Faraway Oasis"),
        (os.path.join(_TESTDATA_DIR, "shyguybazaar.png"), "Shy Guy Bazaar"),
        # Bowser's Castle: 24-player layout widens the player panel beyond
        # the old 42% boundary.
        (os.path.join(_TESTDATA_DIR, "bowserscastle.png"), "Bowser's Castle"),
        # Team-color bars on the left panel + dark map on the right.
        (os.path.join(_FIXTURE_DIR, "acorn_heights_team_mode.png"), "Acorn Heights"),
        (os.path.join(_FIXTURE_DIR, "wario_stadium_team_mode.png"), "Wario Stadium"),
        # Successful detections recovered from past matches — diverse layouts
        # (no-team / team-mode / 24p) and tracks not covered above.
        (os.path.join(_FIXTURE_DIR, "cheep_cheep_falls.png"), "Cheep Cheep Falls"),
        (os.path.join(_FIXTURE_DIR, "peach_stadium.png"), "Peach Stadium"),
        (os.path.join(_FIXTURE_DIR, "boo_cinema.png"), "Boo Cinema"),
        (os.path.join(_FIXTURE_DIR, "moo_moo_meadows.png"), "Moo Moo Meadows"),
        (os.path.join(_FIXTURE_DIR, "mario_bros_circuit.png"), "Mario Bros. Circuit"),
        (os.path.join(_FIXTURE_DIR, "bowsers_castle_24p.png"), "Bowser's Castle"),
        (os.path.join(_FIXTURE_DIR, "salty_salty_speedway.png"), "Salty Salty Speedway"),
        (os.path.join(_FIXTURE_DIR, "airship_fortress.png"), "Airship Fortress"),
    ],
)
def test_detects_track_name(filename, expected):
    detector = TrackSelectDetector()
    result = detector.detect(_load(filename))
    assert result is not None, f"No track detected for {filename}"
    assert result["track_name"] == expected


@pytest.mark.parametrize(
    "filename",
    [
        # Voting screen (intermission) — multiple candidate banners visible
        # over the world map. Used to fuzzy-match one of the candidates.
        os.path.join(_NEGATIVE_DIR, "voting_screen.png"),
        os.path.join(_NEGATIVE_DIR, "voting_screen_2.png"),
        # In-race "track name" overlay shown briefly at race start. The OCR
        # would happily read "Salty Salty Speedway" / "Bowser's Castle" /
        # "Sky-High Sundae" off these gameplay frames; the panel pre-check
        # rejects them because the left side is 3D scenery, not a player list.
        os.path.join(_NEGATIVE_DIR, "gameplay_overlay_salty.png"),
        os.path.join(_NEGATIVE_DIR, "gameplay_overlay_bowsers.png"),
        os.path.join(_NEGATIVE_DIR, "gameplay_overlay_skyhigh.png"),
    ],
)
def test_rejects_non_track_select_screens(filename):
    detector = TrackSelectDetector()
    assert detector.detect(_load(filename)) is None


# Regression: "Wario Shipyard" used to fuzzy-match to "Wario Stadium" because
# the canonical list had "Wario's Galleon" instead of the actual in-game
# name. The fixture is a cropped map-only frame with no player panel, so it
# exercises read_track_name() (canonical-name correctness) directly rather
# than detect() (which now also requires the panel pre-check).
def test_canonical_match_distinguishes_wario_shipyard_from_stadium():
    detector = TrackSelectDetector()
    frame = _load(os.path.join(_FIXTURE_DIR, "wario_shipyard.png"))
    result = detector.read_track_name(frame)
    assert result is not None
    name, _ratio = result
    assert name == "Wario Shipyard"
