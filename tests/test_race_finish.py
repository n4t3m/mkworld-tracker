"""Tests for RaceFinishDetector against captured frames."""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.race_finish import RaceFinishDetector

_DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "race_finish")

_detector = RaceFinishDetector()


def _load(name: str):
    path = os.path.join(_DATA_DIR, name)
    frame = cv2.imread(path)
    assert frame is not None, f"Could not load {path}"
    return frame


@pytest.mark.parametrize(
    "filename",
    [
        "finish_good.png",
    ],
)
def test_detects_finish_screen(filename):
    assert _detector.is_active(_load(filename))


@pytest.mark.parametrize(
    "filename",
    [
        "bad_go.png",
        "bad_go_warm.png",
        "bad_track_select.png",
        "bad_results.png",
        "bad_partial_finish.png",
        "bad_gameplay_boost.png",
        "bad_gameplay_desert.png",
        "bad_gameplay_desert2.png",
        "bad_gameplay_spaceport.png",
        "bad_gameplay_lava.png",
        "bad_gameplay_yellow_road.png",
        "bad_false_finish_1.png",
        "bad_false_finish_2.png",
    ],
)
def test_rejects_non_finish(filename):
    assert not _detector.is_active(_load(filename))
