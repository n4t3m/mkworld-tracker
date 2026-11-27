"""Held-out recording crops, including absent HUDs and independent seeks."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from mktracker.detection.hud_coins import HudCoinReader
from mktracker.ui.main_window import MainWindow

FIXTURES = Path(__file__).parent / "fixtures" / "hud_coins"
CASES = json.loads((FIXTURES / "labels.json").read_text())

# Original full capture-card frames, without the recording's side bars.
CAPTURE_CASES = {
    "bad_false_finish_1": 3, "bad_false_finish_2": 3,
    "bad_gameplay_boost": 6, "bad_gameplay_desert": 6,
    "bad_gameplay_desert2": 0, "bad_gameplay_glider": 4,
    "bad_gameplay_lava": 6, "bad_gameplay_lava2": 11,
    "bad_gameplay_spaceport": 2, "bad_gameplay_yellow_road": 13,
    "bad_gameplay_yoshi_sign": None, "bad_go": 0, "bad_go_warm": 3,
    "bad_partial_finish": 3, "bad_results": None, "bad_track_select": None,
    "finish_good": 7, "finish_good_0418_02": 1, "finish_good_0418_03": 7,
    "finish_good_0418_04": 6, "finish_good_0418_06": 9, "finish_good_0418_07": 7,
    "finish_good_0419_01": 7, "finish_good_0419_02": 2, "finish_good_0419_04": 5,
    "finish_good_0419_05": 16, "finish_good_0419_06": 13,
    "finish_good_0419_07": 20, "finish_good_0419_08": 17,
    "finish_good_chain_chomp": 20, "finish_good_dockyard": 1,
    "finish_good_lava": 10, "finish_good_temple": 18,
}


@pytest.mark.parametrize("name,expected", CAPTURE_CASES.items())
def test_original_capture_card_frames(name, expected):
    frame = cv2.imread(str(FIXTURES.parent / "race_finish" / f"{name}.png"))
    assert frame is not None
    assert HudCoinReader().read(frame) == expected


@pytest.mark.parametrize("name,expected", [("gameplay_daisy_rank_badge", 8),
                                            ("gameplay_mario_kart_arch", 2)])
def test_other_full_capture_frames(name, expected):
    frame = cv2.imread(str(FIXTURES.parent / "race_results_negative" / f"{name}.png"))
    assert HudCoinReader().read(frame) == expected


@pytest.mark.parametrize("padding", [(0, 0, 48, 48), (60, 60, 0, 0),
                                     (37, 83, 71, 23), (108, 108, 192, 192)])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (960, 540)])
def test_capture_padding_and_scaling(padding, size):
    reader = HudCoinReader()
    for name, expected in [("bad_gameplay_lava", 6), ("bad_gameplay_desert2", 0),
                           ("finish_good_0419_07", 20), ("finish_good_temple", 18)]:
        frame = cv2.imread(str(FIXTURES.parent / "race_finish" / f"{name}.png"))
        frame = cv2.copyMakeBorder(frame, *padding, cv2.BORDER_CONSTANT, value=0)
        assert reader.read(cv2.resize(frame, size)) == expected


OTHER_FRAMES = [p for folder in ["track_select", "player_reader", "race_results",
                             "match_results", "team_results", "video_analysis"]
             for p in (FIXTURES.parent / folder).glob("*")
             if p.suffix.lower() in (".png", ".jpg")]


@pytest.mark.parametrize("path", OTHER_FRAMES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_other_screens_and_remaining_visible_huds(path):
    # These files still show the in-race HUD despite their fixture folder.
    expected = {"20260329_144018.png": 4, "bad_finish_banner_blue_sky.png": 7,
                "result-animation.jpg": 4,
                "ghost-valley-false-results.jpg": 20}.get(path.name)
    assert HudCoinReader().read(cv2.imread(str(path))) == expected


def frame_for(name):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[940:, :480] = cv2.imread(str(FIXTURES / name))
    return frame


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["image"])
def test_recorded_coin_count(case):
    assert HudCoinReader().read(frame_for(case["image"])) == case["coins"]


@pytest.mark.parametrize("size", [(1280, 720), (960, 540), (640, 360)])
def test_resized_hud(size):
    reader = HudCoinReader()
    for name, expected in [("oct-75.png", 0), ("oct-184.png", 14), ("oct-555.6.png", 8)]:
        assert reader.read(cv2.resize(frame_for(name), size)) == expected


def test_missing_or_incomplete_hud():
    reader = HudCoinReader()
    for frame in [None, np.empty((0, 0, 3), np.uint8),
                  np.zeros((1080, 1920, 3), np.uint8),
                  np.full((1080, 1920, 3), 255, np.uint8)]:
        assert reader.read(frame) is None
    frame = frame_for("oct-75.png")
    frame[970:1040, 100:160] = 0  # Digits alone must not count as a HUD.
    assert reader.read(frame) is None


def test_coin_count_updates_on_seek_and_mode_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    path = tmp_path / "coins.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (1920, 1080))
    assert writer.isOpened()
    for name in ["oct-75.png", "oct-184.png", "oct-210.png", "oct-555.6.png"]:
        for _ in range(4):
            writer.write(frame_for(name))
    writer.release()
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(path))
    try:
        assert window._review_coins.text() == "Coins: 0"
        for seconds, expected in [(1, "14"), (2, "—"), (3, "8"), (0, "0")]:
            window._seek_video(seconds)
            assert window._review_coins.text() == f"Coins: {expected}"
        window._review_mode_combo.setCurrentText("Time Trial")
        assert window._review_coins.text() == "Coins: —"
        window._review_mode_combo.setCurrentText("Custom Room")
        assert window._review_coins.text() == "Coins: 0"
    finally:
        window.close()
