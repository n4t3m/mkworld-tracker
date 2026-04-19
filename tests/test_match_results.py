"""Tests for MatchResultDetector — no-teams ≤12 player layout."""
from __future__ import annotations

import os

import cv2
import pytest

from mktracker.detection.match_results import MatchResultDetector

_DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "match_results")

_detector = MatchResultDetector()


def _load(name: str):
    path = os.path.join(_DATA_DIR, name)
    frame = cv2.imread(path)
    assert frame is not None, f"Could not load {path}"
    return frame


# ---------------------------------------------------------------------------
# True positives — no-teams ≤12 player result screens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "nicetryvariant.png",       # NICE TRY!, 8 players, silver cup (player got 7th)
        "12presultscreen.png",      # CONGRATULATIONS!, 8 players, silver cup (player got 2nd)
        "20260404_151819.png",      # CONGRATULATIONS!, 8 players, same match as above
    ],
)
def test_detects_no_teams_result_screen(filename):
    frame = _load(filename)
    result = _detector.detect(frame, teams="No Teams", player_count=8)
    assert result is not None, f"Expected positive detection for {filename}"
    assert "results" in result
    assert len(result["results"]) > 0


# ---------------------------------------------------------------------------
# False positives (gameplay) — must return None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "20260329_131323.png",  # Pure gameplay — golden temple track with yellow scene
        "20260329_144018.png",  # Pure gameplay — race with yellow "1st" badge
    ],
)
def test_rejects_gameplay_frame(filename):
    frame = _load(filename)
    result = _detector.detect(frame, teams="No Teams", player_count=8)
    assert result is None, f"Expected None for gameplay frame {filename}"


# ---------------------------------------------------------------------------
# >12 player result screens — no-teams two-column layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,player_count",
    [
        ("20260329_162559.png", 24),  # CONGRATULATIONS!, 24 players, two-column
        ("20260329_162602.png", 24),  # CONGRATULATIONS!, 24 players, two-column
        ("20260330_190417.png", 24),  # CONGRATULATIONS!, 24 players, two-column (silver cup)
        ("20260403_172613.png", 23),  # NICE TRY!, 23 players, score-only right column
        ("20260403_175939.png", 22),  # NICE TRY!, 22 players, score-only right column
    ],
)
def test_detects_no_teams_24_result_screen(filename, player_count):
    frame = _load(filename)
    result = _detector.detect(frame, teams="No Teams", player_count=player_count)
    assert result is not None, f"Expected positive detection for {filename}"
    assert "results" in result


# ---------------------------------------------------------------------------
# Two-teams ≤12 player result screens — must fire the two-teams detector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,player_count",
    [
        ("twoteams_result.png", 12),       # CONGRATULATIONS!, 12 players, two teams
        ("20260329_163426.png", 12),        # DRAW!, 12 players, two teams (216 vs 216)
        ("20260329_174714.png", 12),        # CONGRATULATIONS!, 12 players, two teams
        ("20260329_174922.png", 12),        # CONGRATULATIONS!, 12 players, two teams (duplicate frame)
    ],
)
def test_detects_two_teams_result_screen(filename, player_count):
    frame = _load(filename)
    result = _detector.detect(frame, teams="Two Teams", player_count=player_count)
    assert result is not None, f"Expected positive detection for {filename}"
    assert "results" in result
    assert len(result["results"]) > 0


# ---------------------------------------------------------------------------
# Wrong teams mode — two-team screens must not match no-teams path
# ---------------------------------------------------------------------------

def test_rejects_two_teams_screen_on_no_teams_path():
    """A two-team result screen must not fire the no-teams detector."""
    frame = _load("twoteams_result.png")
    result = _detector.detect(frame, teams="No Teams", player_count=12)
    assert result is None


# ---------------------------------------------------------------------------
# Two-teams banner readiness — Gemini path uses _has_result_banner directly.
# In team mode the banner background is the winning team's colour (red or
# blue) with white text — not the no-teams "yellow text on red stripe" — so
# the readiness check must dispatch to a team-specific detector.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "twoteams_banner_red_win.png",   # CONGRATULATIONS!, red team won (red banner)
        "twoteams_banner_blue_win.png",  # CONGRATULATIONS!, blue team won (blue banner)
        "twoteams_result.png",           # CONGRATULATIONS!, two teams
        "20260329_174714.png",           # CONGRATULATIONS!, two teams
        "20260329_174922.png",           # CONGRATULATIONS!, two teams
        "20260329_163426.png",           # DRAW!, two teams (dark banner)
    ],
)
def test_two_teams_banner_detected(filename):
    frame = _load(filename)
    assert _detector._has_result_banner(frame, teams="Two Teams"), (
        f"Expected two-team banner to be detected on {filename}"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "20260329_131323.png",   # Pure gameplay — golden temple
        "20260329_144018.png",   # Pure gameplay — yellow "1st" badge
        "12presultscreen.png",   # No-teams result screen — wrong layout
        "nicetryvariant.png",    # No-teams result screen — wrong layout
        "20260329_162559.png",   # No-teams 24-player result screen
    ],
)
def test_two_teams_banner_rejects_non_team_screens(filename):
    frame = _load(filename)
    assert not _detector._has_result_banner(frame, teams="Two Teams"), (
        f"Two-team banner check should reject {filename}"
    )
