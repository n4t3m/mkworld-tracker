"""Real HUD crops held out from the reference templates, plus playback checks."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from mktracker.detection.hud_position import HudPositionReader
from mktracker.ui.main_window import MainWindow

FIXTURES = Path(__file__).parent / "fixtures" / "hud_position"
CASES = json.loads((FIXTURES / "labels.json").read_text())
FADE_CASES = json.loads((FIXTURES / "fades.json").read_text())

CAPTURE_CASES = {
    "bad_false_finish_1": None, "bad_false_finish_2": None,  # Hidden / 14th.
    "bad_gameplay_boost": 1, "bad_gameplay_desert": 1,
    "bad_gameplay_desert2": None,  # 21st is outside the supported 12-player set.
    "bad_gameplay_glider": 3, "bad_gameplay_lava": None,  # Heavy glow/fade.
    "bad_gameplay_lava2": 1, "bad_gameplay_spaceport": 5,
    "bad_gameplay_yellow_road": None,  # Gradient fill remains ambiguous.
    "bad_gameplay_yoshi_sign": None, "bad_go": None, "bad_go_warm": None,
    "bad_partial_finish": 1, "bad_results": None, "bad_track_select": None,
    "finish_good": 1, "finish_good_0418_02": None, "finish_good_0418_03": None,
    "finish_good_0418_04": 12, "finish_good_0418_06": None,
    "finish_good_0418_07": None, "finish_good_0419_01": None,
    "finish_good_0419_02": None, "finish_good_0419_04": None,
    "finish_good_0419_05": None, "finish_good_0419_06": None,
    "finish_good_0419_07": None, "finish_good_0419_08": None,
    "finish_good_chain_chomp": None, "finish_good_dockyard": None,
    "finish_good_lava": None, "finish_good_temple": None,
}


@pytest.mark.parametrize("name,expected", CAPTURE_CASES.items())
def test_direct_capture_frames(reader, name, expected):
    frame = cv2.imread(str(FIXTURES.parent / "race_finish" / f"{name}.png"))
    assert reader.read(frame) == expected


@pytest.mark.parametrize("padding", [(0, 0, 48, 48), (60, 60, 0, 0),
                                     (37, 83, 71, 23), (108, 108, 192, 192)])
@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720), (960, 540)])
def test_capture_padding(reader, padding, size):
    for name, expected in [("bad_gameplay_boost", 1), ("bad_gameplay_glider", 3),
                           ("bad_gameplay_spaceport", 5), ("finish_good_0418_04", 12)]:
        frame = cv2.imread(str(FIXTURES.parent / "race_finish" / f"{name}.png"))
        frame = cv2.copyMakeBorder(frame, *padding, cv2.BORDER_CONSTANT, value=0)
        assert reader.read(cv2.resize(frame, size)) == expected


@pytest.mark.parametrize("padding", [(0, 0, 48, 48), (60, 60, 0, 0), (37, 83, 71, 23)])
def test_recording_huds_translate_with_padding(reader, padding):
    for name, expected in [("oct-80.png", 12), ("oct-130.png", 8),
                           ("sep-105.png", 1), ("fade-oct-70.00.png", 4)]:
        frame = cv2.copyMakeBorder(frame_for(name), *padding, cv2.BORDER_CONSTANT, value=0)
        assert reader.read(frame) == expected


OTHER_FRAMES = [p for folder in ["track_select", "player_reader", "race_results",
                               "match_results", "team_results", "video_analysis"]
               for p in (FIXTURES.parent / folder).glob("*")
               if p.suffix.lower() in (".png", ".jpg")]


@pytest.mark.parametrize("path", OTHER_FRAMES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_other_full_frames(reader, path):
    expected = {"20260329_144018.png": 1, "result-animation.jpg": 2}.get(path.name)
    assert reader.read(cv2.imread(str(path))) == expected


def frame_for(name):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[810:, 1574:] = cv2.imread(str(FIXTURES / name))
    return frame


@pytest.fixture(scope="module")
def reader():
    return HudPositionReader()


@pytest.mark.parametrize("case", CASES + FADE_CASES, ids=lambda c: c["image"])
def test_real_hud_position(reader, case):
    assert reader.read(frame_for(case["image"])) == case["position"]


@pytest.mark.parametrize("size", [(1280, 720), (960, 540), (640, 360)])
def test_position_scales_with_video_resolution(reader, size):
    for name, expected in [("oct-80.png", 12), ("oct-130.png", 8), ("sep-105.png", 1)]:
        assert reader.read(cv2.resize(frame_for(name), size)) == expected


def test_blank_frame_and_missing_hud_are_unknown(reader):
    assert reader.read(np.zeros((180, 320, 3), np.uint8)) is None
    assert reader.read(np.full((180, 320, 3), 255, np.uint8)) is None
    assert reader.read(np.empty((0, 0, 3), np.uint8)) is None


def test_position_follows_seek_playback_and_mode_without_affecting_live(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    path = tmp_path / "positions.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (1920, 1080))
    assert writer.isOpened()
    for name in ["sep-105.png", "oct-140.png", "oct-80.png", "oct-240.png"]:
        for _ in range(4):
            writer.write(frame_for(name))
    writer.release()
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(path))
    try:
        assert window._review_position.text() == "Race position: 1st"
        assert window._video_review_tab.isAncestorOf(window._review_position)
        window._seek_video(1)
        assert window._review_position.text() == "Race position: 7th"
        assert window._paused
        window._seek_video(2)
        assert window._review_position.text() == "Race position: 12th"
        window._seek_video(3)
        assert window._review_position.text() == "Race position: —"
        window._seek_video(0)
        now = [100.]
        monkeypatch.setattr("mktracker.ui.main_window.time.monotonic", lambda: now[0])
        window._toggle_playback()
        now[0] += 1.1
        window._update_frame()
        assert window._review_position.text() == "Race position: 7th"
        window._toggle_playback()
        window._review_mode_combo.setCurrentText("Time Trial")
        assert window._review_position.text() == "Race position: —"
        window._review_mode_combo.setCurrentText("Custom Room")
        assert window._review_position.text() == "Race position: 7th"
        # A live frame cannot replace the position shown for recorded footage.
        monkeypatch.setattr(window._live_capture, "read_frame", lambda: frame_for("oct-80.png"))
        window._update_live_frame()
        assert window._review_position.text() == "Race position: 7th"
        assert window.open_video(str(path))
        assert window._review_position.text() == "Race position: 1st"
    finally:
        window.close()


def test_direct_seek_reads_faded_numbers_and_clears_hidden_hud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    path = tmp_path / "fades.mp4"
    samples = [("fade-oct-70.00.png", "4th"), ("fade-oct-71.25.png", "2nd"),
               ("fade-oct-820.00.png", "9th"), ("fade-sep-359.00.png", "9th"),
               ("fade-sep-359.50.png", "10th"), ("fade-oct-69.00.png", "—")]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (1920, 1080))
    assert writer.isOpened()
    for name, _ in samples:
        for _ in range(4):
            writer.write(frame_for(name))
    writer.release()
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(path))
    try:
        # Start and seek directly into fades, with no previous solid reading.
        assert window._review_position.text() == "Race position: 4th"
        for index in [2, 5, 1, 4, 3, 5, 0]:
            window._seek_video(index)
            assert window._review_position.text() == f"Race position: {samples[index][1]}"
            assert window._paused
    finally:
        window.close()
