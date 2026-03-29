"""Tests for RaceFinishDetector against captured frames."""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.race_finish import RaceFinishDetector

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "testdata", "race_finish")

_detector = RaceFinishDetector()


def _load(name: str):
    path = os.path.join(_DATA_DIR, name)
    frame = cv2.imread(path)
    assert frame is not None, f"Could not load {path}"
    return frame


def test_detects_finish_screen():
    assert _detector.is_active(_load("finish_good.png"))


@pytest.mark.parametrize(
    "filename",
    [
        "bad_go.png",
        "bad_partial_finish.png",
        "bad_track_select.png",
        "bad_results.png",
    ],
)
def test_rejects_non_finish(filename):
    assert not _detector.is_active(_load(filename))
