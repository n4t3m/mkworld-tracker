"""Tests for RaceResultDetector placement reading and accumulation.

The fixture frames come from real 12-player no-teams races where the
results scroll in over several frames.  The key regression this file
guards against is an off-by-one in placement assignment that caused
first place to be dropped entirely.
"""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.race_results import RaceResultDetector

# ---- PB race fixtures (first-place player name starts with ">€>") ----
_PB_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "race_results")
_PB_FRAMES = [
    "frame_07.png",   # rows 6-12 visible
    "frame_10a.png",  # rows 1-12 visible (race results, +N)
    "frame_10b.png",  # rows 1-12 visible (race results, +N, diff bg)
    "frame_11.png",   # rows 1-12 visible (overall standings)
]
_PB_EXPECTED = {
    1: "Spritz",
    2: "Aho",
    3: "B",
    6: "Beef Boss",
    12: "Yae",
}

# ---- Dandelion race fixtures (first-place name "KT★" with icon overlap) ----
_DAN_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "race_results_dandelion",
)
_DAN_FRAMES = [
    "frame_06.png",   # rows 6-12 visible
    "frame_09.png",   # rows 1-12 visible (race results, +N)
    "frame_10.png",   # rows 1-12 visible (race results, +N, diff bg)
    "frame_11.png",   # rows 1-12 visible (overall standings)
]
_DAN_EXPECTED = {
    1: "KT",
    3: "Yae",
    5: "Beef Boss",
    7: "Spritz",
    10: "B",
    11: "Aho",
    12: "ProTime",
}


@pytest.fixture(scope="module")
def detector():
    return RaceResultDetector()


def _load_frames(data_dir, filenames):
    loaded = []
    for name in filenames:
        path = os.path.join(data_dir, name)
        img = cv2.imread(path)
        if img is None:
            pytest.skip(f"fixture not found: {path}")
        loaded.append(img)
    return loaded


@pytest.fixture(scope="module")
def pb_frames():
    return _load_frames(_PB_DIR, _PB_FRAMES)


@pytest.fixture(scope="module")
def dan_frames():
    return _load_frames(_DAN_DIR, _DAN_FRAMES)


def _accumulate(detector: RaceResultDetector, frames):
    """Simulate the state-machine accumulation loop."""
    placements: dict[int, str] = {}
    quality: dict[int, int] = {}

    for frame in frames:
        result = detector.detect(frame, teams=False)
        if result is None or result["type"] != "race":
            continue
        results = result["results"]
        frame_quality = len(results)
        for placement, name in results:
            if not placement:
                continue
            prev_q = quality.get(placement, 0)
            if placement not in placements:
                placements[placement] = name
                quality[placement] = frame_quality
            elif frame_quality > prev_q:
                placements[placement] = name
                quality[placement] = frame_quality

    return placements


# ------------------------------------------------------------------
# PB race tests
# ------------------------------------------------------------------


class TestPBAccumulation:
    """End-to-end accumulation for the PB race (first-place '>€> Spritz')."""

    @pytest.fixture(autouse=True, scope="class")
    def _accumulated(self, detector, pb_frames, request):
        request.cls.placements = _accumulate(detector, pb_frames)

    def test_first_place_detected(self):
        assert 1 in self.placements, "first place must not be missing"

    def test_placements_start_at_one(self):
        assert min(self.placements) == 1

    def test_no_placement_exceeds_twelve(self):
        assert max(self.placements) <= 12

    def test_all_twelve_placements(self):
        assert len(self.placements) >= 12

    @pytest.mark.parametrize("placement,expected", list(_PB_EXPECTED.items()))
    def test_expected_name(self, placement, expected):
        actual = self.placements.get(placement, "")
        assert expected.lower() in actual.lower(), (
            f"placement {placement}: expected '{expected}' "
            f"in '{actual}'"
        )


class TestPBSingleFrame:
    """Sanity checks on individual PB frame detection."""

    def test_race_result_frame_returns_race(self, detector, pb_frames):
        result = detector.detect(pb_frames[1], teams=False)
        assert result is not None
        assert result["type"] == "race"

    def test_overall_standings_detected(self, detector, pb_frames):
        result = detector.detect(pb_frames[3], teams=False)
        assert result is not None

    def test_scrolled_frame_has_lower_rows(self, detector, pb_frames):
        result = detector.detect(pb_frames[0], teams=False)
        assert result is not None and result["type"] == "race"
        placements = [p for p, _ in result["results"] if p]
        assert all(p >= 5 for p in placements), (
            f"expected only lower rows, got {placements}"
        )


# ------------------------------------------------------------------
# Dandelion race tests
# ------------------------------------------------------------------


class TestDandelionAccumulation:
    """End-to-end accumulation for the Dandelion race (first-place 'KT★')."""

    @pytest.fixture(autouse=True, scope="class")
    def _accumulated(self, detector, dan_frames, request):
        request.cls.placements = _accumulate(detector, dan_frames)

    def test_first_place_detected(self):
        assert 1 in self.placements, "first place must not be missing"

    def test_placements_start_at_one(self):
        assert min(self.placements) == 1

    def test_no_placement_exceeds_twelve(self):
        assert max(self.placements) <= 12

    def test_all_twelve_placements(self):
        assert len(self.placements) >= 12

    @pytest.mark.parametrize("placement,expected", list(_DAN_EXPECTED.items()))
    def test_expected_name(self, placement, expected):
        actual = self.placements.get(placement, "")
        assert expected.lower() in actual.lower(), (
            f"placement {placement}: expected '{expected}' "
            f"in '{actual}'"
        )
