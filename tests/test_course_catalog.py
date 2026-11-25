"""Course updates must work in both OCR pipelines without losing variants."""
import numpy as np
import pytest

from mktracker.detection.tracks import TRACK_NAMES, match_track_name
from mktracker.detection.track_select import TrackSelectDetector
from mktracker.video_analysis import read_course


ADDED_COURSES = (
    "SNES Mario Circuit 1", "SNES Mario Circuit 2", "SNES Mario Circuit 3",
    "SNES Ghost Valley 1", "SNES Ghost Valley 2", "SNES Ghost Valley 3",
    "SNES Choco Island 1", "SNES Choco Island 2", "SNES Vanilla Lake 1",
    "SNES Koopa Beach 1", "Airship Fortress 2", "Desert Hills 2",
    "Salty Salty Speedway 2", "DK Pass 2", "Koopa Troopa Beach 2",
    "Shy Guy Bazaar 2", "Mario Bros. Circuit 2", "Wario Stadium 2",
)


@pytest.mark.parametrize("name", ADDED_COURSES)
@pytest.mark.parametrize("wrapped", [False, True])
def test_added_courses_in_live_and_video_ocr(monkeypatch, name, wrapped):
    lines = name.rsplit(" ", 1) if wrapped else [name]
    monkeypatch.setattr("pytesseract.image_to_string", lambda *a, **kw: "\n".join(lines))
    monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **kw: {
        "text": lines, "conf": [95] * len(lines),
        "block_num": [1] * len(lines), "par_num": [1] * len(lines),
        "line_num": list(range(len(lines))),
    })
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    detector = TrackSelectDetector()
    monkeypatch.setattr(detector, "has_player_panel", lambda f: True)
    assert detector.detect(frame) == {"track_name": name}
    assert read_course(frame) == name


@pytest.mark.parametrize("name", TRACK_NAMES)
def test_catalog_names_remain_distinct(name):
    assert match_track_name(name.lower()) == (name, 1)


@pytest.mark.parametrize("label", ["Airship Fortress 3", "SNES Mario Circuit 4", "SNES Mario Circuit"])
def test_unknown_numbers_are_not_replaced_with_known_courses(label):
    assert match_track_name(label) is None


def test_voting_screen_with_original_and_variant_is_rejected(monkeypatch):
    monkeypatch.setattr("pytesseract.image_to_string",
                        lambda *a, **kw: "Airship Fortress\nAirship Fortress 2")
    assert TrackSelectDetector().read_track_name(np.zeros((180, 320, 3), dtype=np.uint8)) is None


def test_catalog_has_all_courses_without_duplicates():
    assert len(TRACK_NAMES) == len(set(TRACK_NAMES)) == 48
