"""Exercise actual decoding, timeline persistence, and click-to-seek playback."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mktracker.capture.video_source import VideoCapture
from mktracker.video_timeline import VideoRace, VideoTimeline
from mktracker.ui.main_window import MainWindow


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "game.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    assert writer.isOpened()
    for i in range(40):
        writer.write(np.full((180, 320, 3), i * 6, np.uint8))
    writer.release()
    return path


def test_file_capture_seeks_and_releases(video):
    capture = VideoCapture()
    assert capture.open_file(str(video))
    assert capture.duration == pytest.approx(4)
    capture.seek(2)
    frame = capture.read_frame()
    assert 110 < frame.mean() < 135
    capture.close()
    assert capture.file_path is None
    assert not capture.is_open


def test_playback_follows_elapsed_time_after_render_stall(video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    now = [100.0]
    monkeypatch.setattr("mktracker.ui.main_window.time.monotonic", lambda: now[0])
    try:
        window._toggle_playback()
        # Simulate a blocked/slow UI: only one render callback in 1.8 seconds.
        now[0] += 1.8
        window._update_frame()
        assert window._capture.position == pytest.approx(1.8, abs=.11)
        window._toggle_playback()
        paused_position = window._capture.position
        now[0] += 20
        window._update_frame()
        assert window._capture.position == paused_position
        window._toggle_playback()
        now[0] += .4
        window._update_frame()
        assert window._capture.position == pytest.approx(paused_position + .4, abs=.11)
    finally:
        window.close()


def test_timeline_cache_tracks_source_and_survives_reload(video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    timeline = VideoTimeline.create(str(video), 4)
    timeline.races = [VideoRace(1, "Acorn Heights", finish=2, results=3)]
    timeline.complete = True
    timeline.save()
    loaded = VideoTimeline.load(str(video), 4)
    assert loaded == timeline
    # A changed source must not inherit old race detections.
    video.write_bytes(video.read_bytes() + b"changed")
    assert VideoTimeline.load(str(video), 4).races == []


def test_open_pause_and_click_timeline_preserves_analysis(video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mktracker.ui.main_window.enumerate_sources", lambda: [])
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    try:
        assert window._mode_combo.currentText() == "Custom Room"
        assert window.open_video(str(video))
        assert window._paused
        window._timeline.races = [VideoRace(1, "Acorn Heights", finish=2)]
        window._timeline.complete = True
        window._render_video_timeline()
        app.processEvents()
        item = window._timeline_list.item(1)
        QTest.mouseClick(window._timeline_list.viewport(), Qt.MouseButton.LeftButton,
                         pos=window._timeline_list.visualItemRect(item).center())
        assert window._capture.position == pytest.approx(2, abs=.2)
        assert window._paused
        assert 110 < window._last_frame.mean() < 135
        assert len(window._timeline.races) == 1
        window._toggle_playback()
        assert not window._paused
        window._update_frame()
        window._toggle_playback()
        assert window._paused
        paused_at = window._capture.position
        window._update_frame()
        assert window._capture.position == paused_at
        assert len(window._timeline.races) == 1
    finally:
        window.close()


def test_course_banner_excludes_previous_course_in_free_roam():
    from pathlib import Path
    from mktracker.video_analysis import read_course, has_result_table
    fixture_dir = Path(__file__).parent / "fixtures" / "video_analysis"
    frame = lambda name: cv2.resize(cv2.imread(str(fixture_dir / name)), (1920, 1080))
    assert read_course(frame("course-preview.jpg")) == "Dandelion Depths"
    assert read_course(frame("previous-course.jpg")) is None
    assert read_course(frame("preset-course.png")) == "SNES Mario Circuit 2"
    assert has_result_table(frame("result-table.jpg"))
    assert has_result_table(frame("race-results-dark-24p.jpg"))
    assert has_result_table(frame("race-results-dark-24p-partial.jpg"))
    assert has_result_table(frame("race-results-dark-24p-tail.jpg"))
    assert not has_result_table(frame("result-animation.jpg"))
    assert not has_result_table(frame("ghost-valley-false-results.jpg"))
    assert not has_result_table(frame("previous-course.jpg"))


def test_analysis_sequences_two_races_without_duplicate_finishes(video, tmp_path, monkeypatch):
    import mktracker.video_analysis as analysis
    monkeypatch.chdir(tmp_path)

    class Frames:
        index = -1
        def isOpened(self): return True
        def get(self, prop):
            return 2 if prop == cv2.CAP_PROP_FPS else 200
        def grab(self):
            self.index += 1
            return self.index < 200
        def retrieve(self):
            t = self.index / 2
            marker = 1 if 2 <= t < 5 else 3 if 60 <= t < 63 else 2 if 10 <= t < 12 or 80 <= t < 82 else 0
            return True, np.full((18, 32, 3), marker, np.uint8)
        def release(self): pass

    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: Frames())
    monkeypatch.setattr(analysis.TrackSelectDetector, "has_player_panel", lambda f: int(f[0, 0, 0]) in (1, 3))
    monkeypatch.setattr(analysis, "read_course", lambda f: "Acorn Heights" if f[0, 0, 0] == 1 else "Peach Stadium")
    monkeypatch.setattr(analysis.RaceFinishDetector, "is_active", lambda self, f: f[0, 0, 0] == 2)
    monkeypatch.setattr(analysis, "has_result_table", lambda f: False)
    monkeypatch.setattr(analysis.RaceResultDetector, "has_race_results", lambda self, f: False)
    timeline = analysis.analyze_video(str(video))
    assert timeline.complete
    assert [(r.track, r.start, r.finish) for r in timeline.races] == [
        ("Acorn Heights", 2, 10), ("Peach Stadium", 60, 80)]
    assert VideoTimeline.load(str(video), 100) == timeline


def test_cancelled_analysis_is_not_marked_complete(video, tmp_path, monkeypatch):
    from mktracker.video_analysis import analyze_video
    monkeypatch.chdir(tmp_path)
    timeline = analyze_video(str(video), cancelled=lambda: True)
    assert not timeline.complete


def test_background_analysis_updates_window(video, tmp_path, monkeypatch):
    import time
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mktracker.ui.main_window.enumerate_sources", lambda: [])
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    try:
        for _ in range(500):
            time.sleep(.02)
            app.processEvents()
            if not window._analysis_thread.isRunning():
                break
        app.processEvents()
        assert not window._analysis_thread.isRunning()
        assert window._timeline.complete
        assert window._analysis_progress.value() == 1000
        assert "Analysis complete" in window._analysis_progress.format()
        assert window._mode_combo.isEnabled()
        assert window._analyze_btn.isEnabled()
        assert "Analysis complete" in window._analysis_status.text()
    finally:
        window.close()


def test_analysis_progress_uses_video_time_and_resets_for_new_file(video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    try:
        window._timeline.analyzed_until = 1
        window._on_video_progress(window._timeline)
        assert window._analysis_progress.value() == 250
        window._timeline.complete = True
        window._on_video_progress(window._timeline)
        assert window._analysis_progress.value() == 1000
        other = tmp_path / "other.mp4"
        other.write_bytes(video.read_bytes())
        assert window.open_video(str(other))
        assert window._analysis_progress.value() == 0
        window._on_video_analysis_failed("Test failure")
        assert "Analysis failed" in window._analysis_progress.format()
    finally:
        window.close()


def test_live_and_review_tabs_keep_independent_sources_and_modes(video, tmp_path, monkeypatch):
    from unittest.mock import Mock
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mktracker.ui.main_window.enumerate_sources", lambda: [])
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    try:
        assert window._main_tabs.currentWidget() is window._video_review_tab
        assert window._video_review_tab.isAncestorOf(window._timeline_list)
        assert window._video_review_tab.isAncestorOf(window._play_btn)
        assert window._live_view_tab.isAncestorOf(window._combo)
        assert not window._live_view_tab.isAncestorOf(window._timeline_list)
        window._timeline.races = [VideoRace(1, "Acorn Heights", finish=2)]
        window._render_video_timeline()
        timeline = window._timeline
        window._seek_video(1)
        position = window._capture.position
        window._toggle_playback()
        window._main_tabs.setCurrentWidget(window._live_view_tab)
        assert window._paused
        app.processEvents()
        window._mode_combo.setCurrentText("Time Trial")
        assert window._review_mode_combo.currentText() == "Custom Room"
        live = Mock()
        live.read_frame.return_value = np.full((180, 320, 3), 220, np.uint8)
        window._live_capture = live
        window._sources = [{"index": 2, "label": "Test capture"}]
        window._on_source_changed(0)
        live.open.assert_called_once_with(2)
        window._update_live_frame()
        assert window._capture.position == position
        assert window._timeline is timeline
        assert window._timeline_list.count() == 2
        assert window._live_last_frame.mean() == 220
        window._main_tabs.setCurrentWidget(window._video_review_tab)
        assert window._capture.position == position
        assert window._paused
        # Opening another recording must not reset an ongoing live match.
        reset = Mock(side_effect=AssertionError("Video review reset live state"))
        window._state_machine.reset = reset
        assert window.open_video(str(video))
        reset.assert_not_called()
    finally:
        window.close()


def test_refresh_names_updates_visible_results_without_rescanning(video, tmp_path, monkeypatch):
    import time
    monkeypatch.chdir(tmp_path)
    timeline = VideoTimeline.create(str(video), 4)
    timeline.races = [VideoRace(1, "Acorn Heights", finish=2, results=3)]
    timeline.complete = True
    timeline.save()
    calls = []
    def refreshed(timeline, *, progress, cancelled):
        calls.append(timeline.source)
        timeline.races[0].placements = [[1, "Player1"], [2, "Player2"]]
        timeline.save()
        progress(timeline)
    monkeypatch.setattr("mktracker.ui.main_window.refresh_names", refreshed)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    window.show()
    app.processEvents()
    try:
        position = window._capture.position
        QTest.mouseClick(window._refresh_names_btn, Qt.MouseButton.LeftButton)
        for _ in range(300):
            time.sleep(.01)
            app.processEvents()
            if window._analysis_thread and not window._analysis_thread.isRunning():
                break
        app.processEvents()
        assert calls == [str(video.resolve())]
        assert window._timeline.races[0].start == 1
        assert window._capture.position == position
        assert window._names_list.item(0).text() == "1. Player1"
        assert window._names_list.item(1).text() == "2. Player2"
        assert window._analysis_progress.value() == 1000
        assert "Names refreshed" in window._analysis_progress.format()
        assert window._refresh_names_btn.isEnabled()
    finally:
        window.close()


def test_name_refresh_waits_through_one_misclassified_standings_frame(video, tmp_path, monkeypatch):
    from unittest.mock import Mock
    import mktracker.video_analysis as analysis
    monkeypatch.chdir(tmp_path)
    timeline = VideoTimeline.create(str(video), 20)
    timeline.races = [VideoRace(0, "Course", finish=0)]
    class Capture:
        frame = 0
        def isOpened(self): return True
        def get(self, prop): return 4
        def set(self, *args): return True
        def read(self):
            self.frame += 1
            return True, np.full((18, 32, 3), self.frame, np.uint8)
        def release(self): pass
    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(analysis, "has_result_table", lambda _: True)
    partial = {"type": "race", "results": [(1, "Player1")], "confidences": {1: 90}}
    full = {"type": "race", "results": [(1, "Player1"), (2, "Player2")], "confidences": {1: 90, 2: 90}}
    overall = {"type": "overall", "results": [(1, "Wrong standings order")]}
    detector = Mock()
    detector.detect.side_effect = [partial, overall, full, full, overall, overall, overall]
    monkeypatch.setattr(analysis, "RaceResultDetector", lambda: detector)
    analysis.refresh_names(timeline)
    assert timeline.races[0].placements == [[1, "Player1"], [2, "Player2"]]
    assert timeline.races[0].placement_evidence["2"]["frames"] == 2
    assert detector.detect.call_count == 7


def test_review_displays_frame_agreement_and_uncertainty(video, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MainWindow, "_start_video_analysis", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_video=str(video))
    try:
        window._timeline.races = [VideoRace(0, "Course", placements=[[1, "Player1"], [2, "Player2"]],
            placement_evidence={"1": {"status": "confirmed", "frames": 5, "agreement": 1, "spelling_agreement": .9},
                                "2": {"status": "review", "frames": 2, "agreement": .55, "spelling_agreement": .5}})]
        window._render_video_timeline()
        assert window._names_list.item(0).text() == "1. Player1"
        assert window._names_list.item(1).text().endswith("[review]")
        assert "5 frames" in window._names_list.item(0).toolTip()
        assert "1/2 corroborated" in window._names_group.title()
        window._timeline.save()
        loaded = VideoTimeline.load(str(video), 4)
        assert loaded.races[0].placement_evidence["1"]["frames"] == 5
    finally:
        window.close()
