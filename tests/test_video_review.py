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
    from mktracker.video_analysis import read_course, has_result_table, has_race_point_differentials
    fixture_dir = Path(__file__).parent / "fixtures" / "video_analysis"
    frame = lambda name: cv2.resize(cv2.imread(str(fixture_dir / name)), (1920, 1080))
    assert read_course(frame("course-preview.jpg")) == "Dandelion Depths"
    assert read_course(frame("previous-course.jpg")) is None
    assert read_course(frame("preset-course.png")) == "SNES Mario Circuit 2"
    assert has_result_table(frame("overall-standings-colored-12p.jpg"))
    assert has_result_table(frame("race-results-dark-24p.jpg"))
    assert has_result_table(frame("race-results-dark-24p-partial.jpg"))
    assert has_result_table(frame("race-results-dark-24p-tail.jpg"))
    assert has_race_point_differentials(frame("race-results-dark-24p.jpg"))
    assert not has_race_point_differentials(frame("overall-standings-colored-12p.jpg"))
    assert not has_result_table(frame("result-animation.jpg"))
    assert not has_result_table(frame("ghost-valley-false-results.jpg"))
    assert not has_result_table(frame("previous-course.jpg"))


def test_model_result_grid_reads_both_24p_pages_without_legacy_name_ocr(monkeypatch):
    from pathlib import Path
    import pytesseract
    from mktracker.detection.race_results import RaceResultDetector

    fixture_dir = Path(__file__).parent / "fixtures" / "video_analysis"
    reads = []
    original = pytesseract.image_to_data
    def counted(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pytesseract, "image_to_data", counted)
    detector = RaceResultDetector()
    for file, expected in (("race-results-dark-24p.jpg", range(1, 14)),
                           ("race-results-dark-24p-partial.jpg", range(14, 25))):
        frame = cv2.resize(cv2.imread(str(fixture_dir / file)), (1920, 1080))
        result = detector.detect(frame, teams=True, model_only=True)
        assert result["type"] == "race"
        assert list(result["row_centers"]) == list(expected)
        assert result["results"] == []
    assert len(reads) <= 4  # One rank sheet and at most one pinned-row digit per page.
    pinned = cv2.imread(str(fixture_dir / "race-results-dark-12p-pinned.jpg"))
    pinned_result = detector.detect(pinned, teams=True, model_only=True)
    assert pinned_result["type"] == "race"
    assert 4 in pinned_result["row_centers"]
    for file in ("result-animation.jpg", "ghost-valley-false-results.jpg"):
        frame = cv2.resize(cv2.imread(str(fixture_dir / file)), (1920, 1080))
        assert detector.detect(frame, teams=True, model_only=True) is None


def test_model_result_grid_reads_colored_12p_without_legacy_name_ocr(monkeypatch):
    from pathlib import Path
    import pytesseract
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import _colored_room_size

    fixture_dir = Path(__file__).parent / "fixtures"
    reads = []
    original = pytesseract.image_to_data
    def counted(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pytesseract, "image_to_data", counted)
    frame = cv2.imread(str(fixture_dir / "team_results" / "race-1.jpg"))
    result = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert result["type"] == "race"
    assert list(result["row_centers"]) == list(range(1, 13))
    assert result["results"] == []
    assert result["name_crop_x"] == (.68, .89)
    assert _colored_room_size(result) == 12
    assert len(reads) == 1


def test_dark_scroll_gap_recovers_tail_seen_between_half_second_samples():
    from collections import deque
    from pathlib import Path
    from mktracker.detection.name_consensus import PlacementConsensus
    from mktracker.detection.name_model import NameModelReader
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import _fill_dark_scroll_gaps

    if not NameModelReader.available():
        pytest.skip("No local dark checkpoint configured")
    fixture_dir = Path(__file__).parent / "fixtures" / "video_analysis"
    middle = cv2.imread(str(fixture_dir / "dandelion-24p-scroll-middle.jpg"))
    tail = cv2.imread(str(fixture_dir / "dandelion-24p-scroll-tail.jpg"))
    assert middle is not None and tail is not None
    reader = NameModelReader()
    detector = RaceResultDetector()
    try:
        grid = detector.detect(middle, teams=True, model_only=True,
                               dark_model=reader, rank_only=True)
        votes = PlacementConsensus()
        votes.add(reader.read(middle, grid), frame_id=1)
        assert 24 not in dict(votes.placements())
        assert _fill_dark_scroll_gaps(
            votes, deque([(2, tail)], maxlen=8), set(), detector, reader)
        names = dict(votes.placements())
        assert names[23] == "SKaR"
        assert names[24] == "luce"
    finally:
        reader.close()


def test_model_rank_reader_avoids_tesseract_on_result_grids(monkeypatch):
    from pathlib import Path
    import pytesseract
    from mktracker.detection.race_results import RaceResultDetector

    class RankReader:
        def predict_crops(self, crops):
            return [(str(index), .99) for index in range(1, len(crops) + 1)]

    def unexpected_tesseract(*args, **kwargs):
        raise AssertionError("Result rank OCR called Tesseract")

    monkeypatch.setattr(pytesseract, "image_to_data", unexpected_tesseract)
    fixture_dir = Path(__file__).parent / "fixtures" / "video_analysis"
    for filename, model in (("race-results-dark-24p.jpg", "dark_model"),
                            ("race-results-colored-12p-vy-tags.jpg", "color_model")):
        frame = cv2.imread(str(fixture_dir / filename))
        result = RaceResultDetector().detect(frame, teams=True, model_only=True,
                                             **{model: RankReader()})
        assert result is not None and result["type"] == "race"
        assert list(result["row_centers"]) == list(range(1, len(result["row_centers"]) + 1))


def test_three_team_dark_24p_result_uses_right_score_column():
    from pathlib import Path
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import has_race_point_differentials, has_result_table

    # EXwtxrNsdvE at 3:31; this room's 81px grid and right-side +points
    # differ from the bundled 77px dark-result fixtures.
    fixture = Path(__file__).parent / "fixtures" / "video_analysis" / "race-results-dark-3team-24p.jpg"
    frame = cv2.imread(str(fixture))
    result = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert result is not None and result["type"] == "race"
    assert list(result["row_centers"]) == list(range(1, 14))
    assert has_result_table(frame)
    assert has_race_point_differentials(frame)


def test_four_team_colored_24p_result_keeps_yellow_and_green_rows():
    from pathlib import Path
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import _colored_room_size, has_race_point_differentials, has_result_table

    # t2MFiGRmP-g at 2:41: red, blue, yellow and green rows all contribute
    # to the 24-player result grid. Its bars are shifted left of the 12p table.
    fixture = Path(__file__).parent / "fixtures" / "video_analysis" / "race-results-colored-4team-24p.jpg"
    frame = cv2.imread(str(fixture))
    result = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert result is not None and result["type"] == "race"
    assert list(result["row_centers"]) == list(range(1, 13))
    assert result["name_crop_x"][0] < .66
    assert _colored_room_size(result) == 24
    assert has_result_table(frame)
    assert has_race_point_differentials(frame)


def test_cached_quarter_frame_fills_one_scrolling_24p_rank():
    from collections import deque
    from pathlib import Path
    from mktracker.detection.name_consensus import PlacementConsensus
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import _fill_colored_scroll_gaps

    fixture = Path(__file__).parent / "fixtures" / "video_analysis" / "race-results-colored-4team-rank15.jpg"
    frame = cv2.imread(str(fixture))
    votes = PlacementConsensus()
    votes.add({"type": "race", "results": [(rank, f"Name{rank}")
                                          for rank in range(1, 25) if rank != 15]})

    class Reader:
        calls = 0
        def read(self, frame, grid):
            self.calls += 1
            return {"type": "race", "results": [(15, "Monty Mole")],
                    "confidences": {15: 90}}

    reader = Reader()
    buffer = deque([(1, frame)], maxlen=8)
    assert _fill_colored_scroll_gaps(votes, buffer, RaceResultDetector(), reader)
    assert dict(votes.placements())[15] == "Monty Mole"
    assert reader.calls == 1
    assert not _fill_colored_scroll_gaps(votes, buffer, RaceResultDetector(), reader)
    assert reader.calls == 1


def test_dark_quarter_frame_recovers_tail_before_recording_cuts_away():
    from collections import deque
    from pathlib import Path
    from mktracker.detection.name_consensus import PlacementConsensus
    from mktracker.detection.race_results import RaceResultDetector
    from mktracker.video_analysis import _fill_dark_scroll_gaps

    # In the full 24p recording, ranks 23–24 are visible at 9:20.75 but the
    # next half-second sample has already scrolled past them.
    fixture = Path(__file__).parent / "fixtures" / "video_analysis" / "race-results-dark-24p-quarter-tail.jpg"
    frame = cv2.imread(str(fixture))
    votes = PlacementConsensus()
    votes.add({"type": "race", "results": [(rank, f"Name{rank}")
                                          for rank in range(1, 23)]})

    class Reader:
        calls = 0
        def read(self, frame, grid):
            self.calls += 1
            return {"type": "race", "results": [(23, "SKaR"), (24, "luce")],
                    "confidences": {23: 95, 24: 95}}

    reader = Reader()
    examined = set()
    buffer = deque([(1, frame)], maxlen=8)
    assert _fill_dark_scroll_gaps(votes, buffer, examined, RaceResultDetector(), reader)
    assert dict(votes.placements())[23] == "SKaR"
    assert dict(votes.placements())[24] == "luce"
    assert not _fill_dark_scroll_gaps(votes, buffer, examined, RaceResultDetector(), reader)
    assert reader.calls == 1


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
    monkeypatch.setattr(analysis, "read_course", lambda f, reader=None: "Acorn Heights" if f[0, 0, 0] == 1 else "Peach Stadium")
    monkeypatch.setattr(analysis.RaceFinishDetector, "is_active", lambda self, f: f[0, 0, 0] == 2)
    monkeypatch.setattr(analysis, "has_result_table", lambda f: False)
    monkeypatch.setattr(analysis.RaceResultDetector, "has_race_results", lambda self, f: False)
    timeline = analysis.analyze_video(str(video))
    assert timeline.complete
    assert [(r.track, r.start, r.finish) for r in timeline.races] == [
        ("Acorn Heights", 2, 10), ("Peach Stadium", 60, 80)]
    assert VideoTimeline.load(str(video), 100) == timeline


def test_analysis_starts_next_race_from_trained_countdown_course_card(
    video, tmp_path, monkeypatch,
):
    import mktracker.video_analysis as analysis

    monkeypatch.chdir(tmp_path)

    class Frames:
        index = -1

        def isOpened(self): return True
        def get(self, prop):
            return 2 if prop == cv2.CAP_PROP_FPS else 160
        def grab(self):
            self.index += 1
            return self.index < 160
        def retrieve(self):
            t = self.index / 2
            marker = 2 if 5 <= t < 6 else 3 if 30 <= t < 31 else 0
            return True, np.full((18, 32, 3), marker, np.uint8)
        def release(self): pass

    class Reader:
        disabled = False

        @staticmethod
        def available(): return True
        @staticmethod
        def color_available(): return False
        def close(self): pass

    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: Frames())
    monkeypatch.setattr(analysis, "NameModelReader", Reader)
    monkeypatch.setattr(analysis.TrackSelectDetector, "has_player_panel",
                        lambda _frame: False)
    monkeypatch.setattr(analysis, "read_race_start_card_model",
                        lambda frame, _reader: (
                            ("SNES Ghost Valley 2", .99)
                            if frame[0, 0, 0] == 3 else None))
    monkeypatch.setattr(analysis.RaceFinishDetector, "is_active",
                        lambda _self, frame: frame[0, 0, 0] == 2)
    monkeypatch.setattr(analysis.RaceResultDetector, "detect",
                        lambda *_args, **_kwargs: None)
    timeline = analysis.analyze_video(str(video))
    assert [(race.track, race.start, race.finish) for race in timeline.races] == [
        ("Unknown course", 0, 5),
        ("SNES Ghost Valley 2", 30, None),
    ]


def test_cancelled_analysis_is_not_marked_complete(video, tmp_path, monkeypatch):
    from mktracker.video_analysis import analyze_video
    monkeypatch.chdir(tmp_path)
    timeline = analyze_video(str(video), cancelled=lambda: True)
    assert not timeline.complete


def test_short_result_clip_creates_review_race_without_finish(video, tmp_path, monkeypatch):
    """A downloaded result-only clip has no finish banner to anchor a race."""
    from types import SimpleNamespace
    import mktracker.video_analysis as analysis

    monkeypatch.chdir(tmp_path)
    class Frames:
        index = -1
        def isOpened(self): return True
        def get(self, prop):
            return 10 if prop == cv2.CAP_PROP_FPS else 80 if prop == cv2.CAP_PROP_FRAME_COUNT else 0
        def grab(self):
            self.index += 1
            return self.index < 40
        def retrieve(self): return True, np.full((180, 320, 3), 120, np.uint8)
        def release(self): pass

    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: Frames())
    monkeypatch.setattr(analysis.RaceFinishDetector, "is_active", lambda self, _: False)
    monkeypatch.setattr(analysis.TrackSelectDetector, "has_player_panel", lambda _: False)
    monkeypatch.setattr(analysis, "has_result_table", lambda _: True)
    monkeypatch.setattr(analysis, "has_race_point_differentials", lambda _: True)
    monkeypatch.setattr(analysis, "RaceResultDetector", lambda: SimpleNamespace(
        detect=lambda frame, teams: {"type": "race", "results": [(1, "Player1"), (2, "Player2")]}))
    monkeypatch.setattr(analysis.NameModelReader, "available", lambda: False)

    timeline = analysis.analyze_video(str(video))
    assert timeline.complete
    assert timeline.duration == pytest.approx(4)
    assert len(timeline.races) == 1
    assert timeline.races[0].finish is None
    assert timeline.races[0].results is not None
    assert timeline.races[0].placements == [[1, "Player1"], [2, "Player2"]]


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
    monkeypatch.setattr(analysis.NameModelReader, "available", lambda: False)
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


def test_refresh_names_reads_result_only_clip(video, tmp_path, monkeypatch):
    import mktracker.video_analysis as analysis
    monkeypatch.chdir(tmp_path)
    timeline = VideoTimeline.create(str(video), 4)
    timeline.races = [VideoRace(0, "Result clip", results=1.5)]
    samples = []
    def sample(cap, start, end, *, cancelled):
        samples.append((start, end))
        for t in (1.5, 2.0):
            yield t, np.full((180, 320, 3), 120, np.uint8)
    monkeypatch.setattr(analysis, "_sample_window", sample)
    monkeypatch.setattr(analysis, "has_result_table", lambda _: True)
    monkeypatch.setattr(analysis.TrackSelectDetector, "has_player_panel", lambda _: False)
    monkeypatch.setattr(analysis.RaceResultDetector, "detect", lambda self, frame, teams: {
        "type": "race", "results": [(1, "Player1"), (2, "Player2")]})
    monkeypatch.setattr(analysis.NameModelReader, "available", lambda: False)

    analysis.refresh_names(timeline)
    assert samples == [(1.0, 4)]
    assert timeline.races[0].finish is None
    assert timeline.races[0].placements == [[1, "Player1"], [2, "Player2"]]


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
