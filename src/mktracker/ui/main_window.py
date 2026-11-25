from __future__ import annotations

import logging
import copy
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QSlider,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mktracker.capture.video_source import VideoCapture, enumerate_sources
from mktracker.debug_config import load_debug_mode, save_debug_mode
from mktracker.detection.match_settings import MatchSettings
from mktracker.discord_webhook import (
    EVENT_MATCH_END,
    EVENT_MATCH_START,
    EVENT_RACE_RESULTS,
    load_event_enabled,
    load_webhook_url,
    save_event_enabled,
    save_webhook_url,
    send_message,
)
from mktracker.gemini_client import (
    SUGGESTED_MODELS,
    load_api_key,
    load_model,
    save_api_key,
    save_model,
    verify_api_key,
)
from mktracker.state_machine import GameState, GameStateMachine
from mktracker.ui.match_history import MatchHistoryView
from mktracker.video_analysis import analyze_video, refresh_names
from mktracker.video_timeline import VideoTimeline, timestamp
from mktracker.detection.font_symbols import configured_font_path, save_font_path

_CAPTURE_DIR = "captured_frames"


class _VerifyThread(QThread):
    """Runs the Gemini API key health check off the main thread."""
    finished = Signal(bool, str)  # (ok, message)

    def __init__(self, key: str, model: str) -> None:
        super().__init__()
        self._key = key
        self._model = model

    def run(self) -> None:
        ok, msg = verify_api_key(self._key, self._model)
        self.finished.emit(ok, msg)


class _WebhookPingThread(QThread):
    """Posts a test message to a Discord webhook off the main thread."""
    finished = Signal(bool, str)  # (ok, message)

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url

    def run(self) -> None:
        import time

        unix_ts = int(time.time())
        embed = {
            "title": "🏁 Webhook Connected",
            "description": (
                "MKWorld Tracker is successfully connected to this channel.\n"
                "Live race updates will appear here during matches."
            ),
            "color": 0x2ECC71,
            "fields": [
                {"name": "Status", "value": "✅ Online", "inline": True},
                {"name": "Source", "value": "Settings → Send Test", "inline": True},
                {
                    "name": "Sent",
                    "value": f"<t:{unix_ts}:F> (<t:{unix_ts}:R>)",
                    "inline": False,
                },
            ],
            "footer": {"text": "MKWorld Tracker"},
        }
        ok, msg = send_message(
            self._url,
            embeds=[embed],
            username="MKWorld Tracker",
        )
        self.finished.emit(ok, msg)

logger = logging.getLogger(__name__)

_FRAME_INTERVAL_MS = 33  # ~30 fps
_DETECT_EVERY_N_FRAMES = 15  # run detection every ~500 ms
_DETECT_FAST_N_FRAMES = 3   # faster sampling during race results reading


class _VideoAnalysisThread(QThread):
    progress = Signal(object)
    failed = Signal(str)
    name_progress = Signal(int, int)

    def __init__(self, source: str, *, names_only: bool = False) -> None:
        super().__init__()
        self.source = source
        self.names_only = names_only
        self.had_error = False

    def run(self) -> None:
        try:
            if self.names_only:
                capture = VideoCapture()
                if not capture.open_file(self.source):
                    raise ValueError(f"Cannot open video: {self.source}")
                timeline = VideoTimeline.load(self.source, capture.duration)
                capture.close()
                completed = 0
                def updated(timeline):
                    nonlocal completed
                    completed += 1
                    self.progress.emit(copy.deepcopy(timeline))
                    self.name_progress.emit(completed, len(timeline.races))
                refresh_names(timeline, progress=updated, cancelled=self.isInterruptionRequested)
                return
            analyze_video(self.source,
                          cancelled=self.isInterruptionRequested,
                          progress=lambda timeline: self.progress.emit(copy.deepcopy(timeline)))
        except Exception as exc:
            self.had_error = True
            logger.exception("Video analysis failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, initial_video: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("MKWorld Tracker")
        self.resize(1280, 800)

        self._capture = VideoCapture()
        self._live_capture = VideoCapture()
        self._live_last_frame: np.ndarray | None = None
        self._sources: list[dict] = []
        self._sources_loaded = False

        self._state_machine = GameStateMachine()
        self._frame_count = 0
        self._last_frame: np.ndarray | None = None
        self._paused = False
        self._playback_clock: tuple[float, float] | None = None
        self._timeline: VideoTimeline | None = None
        self._analysis_thread: _VideoAnalysisThread | None = None
        self._selected_video_race = 0

        self._build_ui()
        if not initial_video:
            self._refresh_sources()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(_FRAME_INTERVAL_MS)
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._update_live_frame)
        self._live_timer.start(_FRAME_INTERVAL_MS)
        if initial_video:
            self.open_video(initial_video)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- top-level tabs: Live View / Match History / Settings ---
        self._main_tabs = QTabWidget()
        self._live_view_tab = self._build_live_view()
        self._main_tabs.addTab(self._live_view_tab, "Live View")
        self._video_review_tab = self._build_video_review()
        self._main_tabs.addTab(self._video_review_tab, "Video Review")
        self._history_view = MatchHistoryView(state_machine=self._state_machine)
        self._main_tabs.addTab(self._history_view, "Match History")
        self._main_tabs.addTab(self._build_api_settings_panel(), "Settings")
        self._main_tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._main_tabs, stretch=1)

        self.setStatusBar(QStatusBar())

    def _build_live_view(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # --- toolbar row (lives inside the Live View tab) ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        source_label = QLabel("Video source:")
        toolbar.addWidget(source_label)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(200)
        self._combo.currentIndexChanged.connect(self._on_source_changed)
        toolbar.addWidget(self._combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_sources)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()

        self._race_label = QLabel()
        self._race_label.setStyleSheet("QLabel { font-weight: bold; font-size: 13px; }")
        toolbar.addWidget(self._race_label)

        self._state_label = QLabel()
        self._state_label.setStyleSheet("QLabel { font-weight: bold; font-size: 13px; }")
        self._update_state_label()
        toolbar.addWidget(self._state_label)

        self._gemini_indicator = QLabel("●")
        self._gemini_indicator.setToolTip("Gemini API: unknown")
        self._gemini_indicator.setStyleSheet("QLabel { color: #888; font-size: 16px; }")
        toolbar.addWidget(self._gemini_indicator)

        actionbar = QHBoxLayout()
        actionbar.addStretch()
        start_manual_btn = QPushButton("Start Manual Match")
        start_manual_btn.setToolTip(
            "Begin a manual match session using the current Match settings. "
            "Required to persist data when advancing past WAITING_FOR_MATCH "
            "without real settings detection."
        )
        start_manual_btn.clicked.connect(self._on_start_manual_match)
        actionbar.addWidget(start_manual_btn)

        advance_btn = QPushButton("Advance State")
        advance_btn.clicked.connect(self._on_advance)
        actionbar.addWidget(advance_btn)

        reset_btn = QPushButton("Reset State")
        reset_btn.clicked.connect(self._on_reset)
        actionbar.addWidget(reset_btn)

        capture_btn = QPushButton("Capture Frame")
        capture_btn.clicked.connect(self._on_capture)
        actionbar.addWidget(capture_btn)

        outer.addLayout(toolbar)
        outer.addLayout(actionbar)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Worldwides", "Time Trial", "Custom Room"])
        self._mode_combo.setCurrentText("Custom Room")
        self._mode_combo.setToolTip("Choose the mode before starting a session.")
        mode_row.addWidget(self._mode_combo)
        self._mode_status = QLabel()
        self._mode_status.setWordWrap(True)
        mode_row.addWidget(self._mode_status, stretch=1)
        outer.addLayout(mode_row)
        self._match_controls = (start_manual_btn, advance_btn, reset_btn)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)

        # --- video display ---
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self._video_label.setMinimumSize(1, 1)
        self._video_label.setText("No source selected")
        self._video_label.setStyleSheet(
            "background-color: #111; color: #555; font-size: 16px;"
        )
        content.addWidget(self._video_label, stretch=1)

        # --- right panel: match settings editor ---
        match_panel = self._build_settings_panel()
        match_panel.setFixedWidth(270)
        self._match_panel = match_panel
        content.addWidget(match_panel)

        outer.addLayout(content, stretch=1)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed()
        return container

    def _build_video_review(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        toolbar = QHBoxLayout()
        open_btn = QPushButton("Open Video…")
        open_btn.clicked.connect(self._choose_video)
        toolbar.addWidget(open_btn)
        toolbar.addWidget(QLabel("Mode:"))
        self._review_mode_combo = QComboBox()
        self._review_mode_combo.addItems(["Worldwides", "Time Trial", "Custom Room"])
        self._review_mode_combo.setCurrentText("Custom Room")
        self._review_mode_combo.currentTextChanged.connect(self._on_review_mode_changed)
        toolbar.addWidget(self._review_mode_combo)
        self._review_filename = QLabel("No video selected")
        self._review_filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(self._review_filename, stretch=1)
        outer.addLayout(toolbar)
        self._review_video_label = QLabel("Open a video to review recorded gameplay")
        self._review_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._review_video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._review_video_label.setMinimumSize(1, 1)
        self._review_video_label.setStyleSheet("background-color: #111; color: #888; font-size: 16px;")
        outer.addWidget(self._review_video_label, stretch=1)

        playback = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_playback)
        playback.addWidget(self._play_btn)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.sliderReleased.connect(self._seek_from_slider)
        playback.addWidget(self._seek_slider, stretch=1)
        self._video_time = QLabel("0:00:00 / 0:00:00")
        playback.addWidget(self._video_time)
        self._analyze_btn = QPushButton("Analyze Video")
        self._analyze_btn.clicked.connect(self._start_video_analysis)
        playback.addWidget(self._analyze_btn)
        self._refresh_names_btn = QPushButton("Refresh Names")
        self._refresh_names_btn.setToolTip("Re-read names from the detected race results without rescanning the full video.")
        self._refresh_names_btn.clicked.connect(lambda: self._start_video_analysis(names_only=True))
        playback.addWidget(self._refresh_names_btn)
        outer.addLayout(playback)
        self._analysis_status = QLabel("Open a video to build a reviewable race timeline.")
        self._analysis_status.setWordWrap(True)
        outer.addWidget(self._analysis_status)
        self._analysis_progress = QProgressBar()
        self._analysis_progress.setRange(0, 1000)
        self._analysis_progress.setValue(0)
        self._analysis_progress.setFormat("%p% analyzed")
        self._analysis_progress.hide()
        outer.addWidget(self._analysis_progress)
        self._timeline_list = QListWidget()
        self._timeline_list.setMaximumHeight(180)
        self._timeline_list.setToolTip("Click a race to review its course preview; click a Finish or Results entry to jump there.")
        self._timeline_list.itemClicked.connect(self._review_timeline_item)
        result_row = QHBoxLayout()
        result_row.addWidget(self._timeline_list, stretch=3)
        self._names_group = QGroupBox("Detected results (OCR)")
        self._names_group.setMaximumHeight(180)
        names_layout = QVBoxLayout(self._names_group)
        self._names_list = QListWidget()
        names_layout.addWidget(self._names_list)
        result_row.addWidget(self._names_group, stretch=1)
        outer.addLayout(result_row)
        self._set_playback_enabled(False)
        return container

    def _set_playback_enabled(self, enabled: bool) -> None:
        self._play_btn.setEnabled(enabled)
        self._seek_slider.setEnabled(enabled)
        self._analyze_btn.setEnabled(enabled and self._review_mode_combo.currentText() == "Custom Room")
        self._refresh_names_btn.setEnabled(enabled and bool(self._timeline and self._timeline.races)
                                          and self._review_mode_combo.currentText() == "Custom Room")

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open gameplay video", "", "Videos (*.mp4 *.mkv *.avi *.mov *.webm);;All files (*)")
        if path:
            self.open_video(path)

    def _stop_video_analysis(self) -> None:
        if self._analysis_thread and self._analysis_thread.isRunning():
            self._analysis_thread.requestInterruption()
            self._analysis_thread.wait()

    def open_video(self, path: str) -> bool:
        self._stop_video_analysis()
        if not self._capture.open_file(path):
            self._set_playback_enabled(False)
            self._analysis_progress.hide()
            self.statusBar().showMessage(f"Could not open video: {path}")
            return False
        self._main_tabs.setCurrentWidget(self._video_review_tab)
        self._review_filename.setText(Path(path).name)
        self._review_filename.setToolTip(str(Path(path).resolve()))
        self._timeline = VideoTimeline.load(path, self._capture.duration)
        self._selected_video_race = 0
        self._seek_slider.setRange(0, round(self._capture.duration * 1000))
        self._timer.setInterval(max(1, round(1000 / self._capture.fps)))
        self._set_playback_enabled(True)
        self._playback_clock = None
        self._paused = False
        self._update_frame()
        self._paused = True
        self._play_btn.setText("Play")
        self._render_video_timeline()
        self._on_review_mode_changed()
        self.statusBar().showMessage(f"Video: {path}")
        if self._review_mode_combo.currentText() == "Custom Room" and not self._timeline.complete:
            self._start_video_analysis()
        return True

    def _toggle_playback(self) -> None:
        if self._capture.file_path:
            if self._capture.position >= self._capture.duration - 1 / self._capture.fps:
                self._capture.seek(0)
            self._paused = not self._paused
            self._playback_clock = None if self._paused else (time.monotonic(), self._capture.position)
            self._play_btn.setText("Play" if self._paused else "Pause")

    def _seek_from_slider(self) -> None:
        self._seek_video(self._seek_slider.value() / 1000)

    def _seek_video(self, seconds: float) -> None:
        self._capture.seek(seconds)
        self._playback_clock = None
        was_paused = self._paused
        self._paused = False
        self._update_frame()
        self._paused = was_paused
        if not was_paused:
            self._playback_clock = (time.monotonic(), self._capture.position)

    def _start_video_analysis(self, *, names_only: bool = False) -> None:
        if not self._capture.file_path or self._review_mode_combo.currentText() != "Custom Room":
            return
        if self._analysis_thread and self._analysis_thread.isRunning():
            return
        self._analysis_thread = _VideoAnalysisThread(self._capture.file_path, names_only=names_only)
        self._analysis_thread.progress.connect(self._on_video_progress)
        self._analysis_thread.failed.connect(self._on_video_analysis_failed)
        self._analysis_thread.name_progress.connect(self._on_name_progress)
        self._analysis_thread.finished.connect(self._on_video_analysis_finished)
        self._analysis_thread.start()
        self._analysis_progress.setRange(0, 1000)
        self._analysis_progress.setValue(0)
        self._analysis_progress.setFormat("%p% · Refreshing names…" if names_only else "%p% analyzed · Starting…")
        self._analysis_progress.show()
        self._review_mode_combo.setEnabled(False)
        self._analyze_btn.setEnabled(False)
        self._refresh_names_btn.setEnabled(False)
        self._analysis_status.setText("Refreshing saved race names…" if names_only else "Analyzing video locally… You can play, pause, and seek while analysis runs.")

    def _on_name_progress(self, completed: int, total: int) -> None:
        if self.sender() is not self._analysis_thread:
            return
        self._analysis_progress.setRange(0, max(1, total))
        self._analysis_progress.setValue(completed)
        self._analysis_progress.setFormat(f"%p% · Names refreshed for {completed} / {total} races")

    def _on_video_analysis_failed(self, error: str) -> None:
        if self.sender() is not None and self.sender() is not self._analysis_thread:
            return
        self._analysis_status.setText(f"Analysis failed: {error}")
        if self._analysis_progress.maximum() == 0:
            self._analysis_progress.setRange(0, 1000)
            self._analysis_progress.setValue(0)
        self._analysis_progress.setFormat("Analysis failed · %p% analyzed")

    def _on_video_analysis_finished(self) -> None:
        if self.sender() is not self._analysis_thread:
            return
        self._analyze_btn.setEnabled(bool(self._capture.file_path) and self._review_mode_combo.currentText() == "Custom Room")
        self._review_mode_combo.setEnabled(True)
        self._refresh_names_btn.setEnabled(bool(self._timeline and self._timeline.races)
                                          and self._review_mode_combo.currentText() == "Custom Room")
        if self._analysis_thread.names_only and not self._analysis_thread.had_error and not self._analysis_thread.isInterruptionRequested():
            self._analysis_progress.setRange(0, 1000)
            self._analysis_progress.setValue(1000)
            self._analysis_progress.setFormat("100% · Names refreshed")

    def _on_video_progress(self, timeline: VideoTimeline) -> None:
        if self.sender() is not None and self.sender() is not self._analysis_thread:
            return
        if timeline.source != str(Path(self._capture.file_path or ".").resolve()):
            return
        self._timeline = timeline
        self._render_video_timeline()

    def _render_video_timeline(self) -> None:
        self._timeline_list.clear()
        if not self._timeline:
            return
        t = self._timeline
        self._analysis_progress.setVisible(self._review_mode_combo.currentText() == "Custom Room")
        if t.duration > 0 or t.complete:
            self._analysis_progress.setRange(0, 1000)
            value = 1000 if t.complete else min(999, max(0, int(1000 * t.analyzed_until / t.duration)))
            self._analysis_progress.setValue(value)
            self._analysis_progress.setFormat(
                "100% · Analysis complete" if t.complete else
                f"%p% analyzed · {timestamp(t.analyzed_until)} / {timestamp(t.duration)}"
            )
        else:
            self._analysis_progress.setRange(0, 0)
            self._analysis_progress.setFormat("Analyzing…")
        state = "Analysis complete" if t.complete else f"Analyzed through {timestamp(t.analyzed_until)} / {timestamp(t.duration)}"
        self._analysis_status.setText(f"{state} · {len(t.races)} races · Click an entry to review. Course preview times are sampled; OCR results need review.")
        for number, race in enumerate(t.races, 1):
            entries = [(race.start, f"Race {number} · {race.track} · Course preview")]
            if race.finish is not None:
                entries.append((race.finish, f"    Race {number} · Finish"))
            if race.results is not None:
                entries.append((race.results, f"    Race {number} · Results (OCR draft: {len(race.placements)} names)"))
            for seconds, label in entries:
                item = QListWidgetItem(f"{timestamp(seconds)}  {label}")
                item.setData(Qt.ItemDataRole.UserRole, seconds)
                item.setData(Qt.ItemDataRole.UserRole + 1, number - 1)
                item.setToolTip(race.review_note + "\n" + "\n".join(f"{place}. {name}" for place, name in race.placements))
                self._timeline_list.addItem(item)
        self._show_video_names(self._selected_video_race)

    def _show_video_names(self, race_index: int) -> None:
        self._names_list.clear()
        if not self._timeline or not 0 <= race_index < len(self._timeline.races):
            self._names_group.setTitle("Detected results (OCR)")
            self._names_list.addItem("Select a race to inspect its names.")
            return
        self._selected_video_race = race_index
        race = self._timeline.races[race_index]
        self._names_group.setToolTip("Course-preview roster:\n" + "\n".join(race.preview_players)
                                     if race.preview_players else "No corroborated preview roster was available.")
        confirmed = sum(e.get("status") == "confirmed" for e in race.placement_evidence.values())
        self._names_group.setTitle(f"Race {race_index + 1} · results ({confirmed}/{len(race.placements)} corroborated)"
                                   if race.placement_evidence else f"Race {race_index + 1} · detected results (OCR)")
        for rank, name in race.placements:
            evidence = race.placement_evidence.get(str(rank))
            suffix = " [review]" if evidence and evidence.get("status") != "confirmed" else ""
            item = QListWidgetItem(f"{rank}. {name}{suffix}")
            # The supplied font also supplies private-use icons that system
            # fonts cannot display. Keep registration on the GUI thread.
            font_path = configured_font_path()
            if font_path and Path(font_path).is_file():
                if getattr(self, "_registered_game_font", None) != font_path:
                    font_id = QFontDatabase.addApplicationFont(font_path)
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    self._game_font_family = families[0] if families else None
                    self._registered_game_font = font_path
                if self._game_font_family:
                    item.setFont(QFont(self._game_font_family, self._names_list.font().pointSize()))
            if evidence:
                item.setToolTip(
                    f"Read across {evidence.get('frames', 0)} frames.\n"
                    f"Identity agreement: {evidence.get('agreement', 0):.0%}.\n"
                    f"Spelling agreement: {evidence.get('spelling_agreement', 0):.0%}.\n"
                    + (f"Course preview: {evidence.get('preview_match') or 'no clear match'}\n"
                       if evidence.get('preview_checked') else "") +
                    ("A glyph is ambiguous; the original OCR spelling was retained.\n"
                     if evidence.get('glyph_review') else "") +
                    "Agreement measures consistency between OCR reads, not guaranteed accuracy."
                )
            self._names_list.addItem(item)
        if not race.placements:
            self._names_list.addItem("No reliable name reads yet.")

    def _review_timeline_item(self, item: QListWidgetItem) -> None:
        race_index = item.data(Qt.ItemDataRole.UserRole + 1)
        if race_index is not None:
            self._show_video_names(race_index)
        self._paused = True
        self._play_btn.setText("Play")
        self._seek_video(item.data(Qt.ItemDataRole.UserRole))

    def _on_mode_changed(self, _text: str = "") -> None:
        custom_room = self._mode_combo.currentText() == "Custom Room"
        self._match_panel.setEnabled(custom_room)
        for control in self._match_controls:
            control.setEnabled(custom_room)
        self._mode_status.setText(
            "Track room settings, race results, and match standings."
            if custom_room else
            "Video preview available. Analysis for this mode is not implemented yet."
        )
        self._update_state_label()

    def _on_review_mode_changed(self, _text: str = "") -> None:
        analyzing = self._analysis_thread is not None and self._analysis_thread.isRunning()
        custom_room = self._review_mode_combo.currentText() == "Custom Room"
        self._analyze_btn.setEnabled(bool(self._capture.file_path) and custom_room and not analyzing)
        self._refresh_names_btn.setEnabled(bool(self._capture.file_path and self._timeline and self._timeline.races)
                                          and custom_room and not analyzing)
        if not custom_room:
            self._analysis_progress.hide()
            self._analysis_status.setText("Video preview available. Analysis for this mode is not implemented yet.")
        elif self._timeline:
            self._render_video_timeline()

    def _build_settings_panel(self) -> QGroupBox:
        group = QGroupBox("Match Settings")
        form = QFormLayout(group)
        form.setSpacing(8)

        self._cc_combo = QComboBox()
        self._cc_combo.addItems(["50cc", "100cc", "150cc"])
        self._cc_combo.setCurrentText("150cc")
        self._cc_combo.currentTextChanged.connect(self._on_settings_changed)
        form.addRow("Class:", self._cc_combo)

        self._teams_combo = QComboBox()
        self._teams_combo.addItems([
            "No Teams", "Two Teams", "Three Teams", "Four Teams",
        ])
        self._teams_combo.currentTextChanged.connect(self._on_settings_changed)
        form.addRow("Teams:", self._teams_combo)

        self._items_combo = QComboBox()
        self._items_combo.addItems([
            "Normal", "Frantic", "Custom Items", "Mushrooms Only",
        ])
        self._items_combo.currentTextChanged.connect(self._on_settings_changed)
        form.addRow("Items:", self._items_combo)

        self._com_combo = QComboBox()
        self._com_combo.addItems(["No COM", "Easy", "Normal", "Hard"])
        self._com_combo.setCurrentText("Normal")
        self._com_combo.currentTextChanged.connect(self._on_settings_changed)
        form.addRow("COM:", self._com_combo)

        self._race_count_spin = QSpinBox()
        self._race_count_spin.setRange(1, 48)
        self._race_count_spin.setValue(12)
        self._race_count_spin.valueChanged.connect(self._on_settings_changed)
        form.addRow("Race count:", self._race_count_spin)

        self._intermission_combo = QComboBox()
        self._intermission_combo.addItems(["10 seconds", "One minute"])
        self._intermission_combo.currentTextChanged.connect(
            self._on_settings_changed,
        )
        form.addRow("Intermission:", self._intermission_combo)

        self._settings_status = QLabel("Manual")
        self._settings_status.setStyleSheet(
            "QLabel { color: #888; font-style: italic; margin-top: 4px; }"
        )
        form.addRow("Source:", self._settings_status)

        # Push the UI values into the state machine on startup.
        self._push_settings_to_state_machine()

        return group

    def _build_api_settings_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # --- API key group ---------------------------------------------------
        key_group = QGroupBox("Gemini API Key")
        key_layout = QVBoxLayout(key_group)
        key_layout.setSpacing(6)

        key_row = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Paste API key here")
        key_row.addWidget(self._api_key_edit, 1)

        self._eye_btn = QPushButton("👁")
        self._eye_btn.setFixedWidth(30)
        self._eye_btn.setCheckable(True)
        self._eye_btn.setToolTip("Show/hide API key")
        self._eye_btn.toggled.connect(self._on_toggle_key_visibility)
        key_row.addWidget(self._eye_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save_api_key)
        key_row.addWidget(save_btn)

        verify_btn = QPushButton("Verify")
        verify_btn.clicked.connect(self._on_verify_api_key)
        key_row.addWidget(verify_btn)

        key_layout.addLayout(key_row)

        self._api_status_label = QLabel()
        self._api_status_label.setWordWrap(True)
        key_layout.addWidget(self._api_status_label)

        layout.addWidget(key_group)

        # --- Model group -----------------------------------------------------
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(6)

        self._api_model_combo = QComboBox()
        self._api_model_combo.setEditable(True)
        self._api_model_combo.addItems(SUGGESTED_MODELS)
        self._api_model_combo.lineEdit().setPlaceholderText("e.g. gemma-4-31b-it")
        model_layout.addWidget(self._api_model_combo)

        model_hint = QLabel("Saved automatically when changed.")
        model_hint.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        model_layout.addWidget(model_hint)

        layout.addWidget(model_group)

        # --- Discord webhook group ------------------------------------------
        webhook_group = QGroupBox("Discord Webhook")
        webhook_layout = QVBoxLayout(webhook_group)
        webhook_layout.setSpacing(6)

        webhook_row = QHBoxLayout()
        self._webhook_edit = QLineEdit()
        self._webhook_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._webhook_edit.setPlaceholderText("https://discord.com/api/webhooks/…")
        webhook_row.addWidget(self._webhook_edit, 1)

        self._webhook_eye_btn = QPushButton("👁")
        self._webhook_eye_btn.setFixedWidth(30)
        self._webhook_eye_btn.setCheckable(True)
        self._webhook_eye_btn.setToolTip("Show/hide webhook URL")
        self._webhook_eye_btn.toggled.connect(self._on_toggle_webhook_visibility)
        webhook_row.addWidget(self._webhook_eye_btn)

        webhook_save_btn = QPushButton("Save")
        webhook_save_btn.clicked.connect(self._on_save_webhook_url)
        webhook_row.addWidget(webhook_save_btn)

        self._webhook_ping_btn = QPushButton("Send Test")
        self._webhook_ping_btn.setToolTip(
            "Posts a test message to the webhook to confirm it works."
        )
        self._webhook_ping_btn.clicked.connect(self._on_ping_webhook)
        webhook_row.addWidget(self._webhook_ping_btn)

        webhook_layout.addLayout(webhook_row)

        self._webhook_status_label = QLabel()
        self._webhook_status_label.setWordWrap(True)
        webhook_layout.addWidget(self._webhook_status_label)

        stored_webhook = load_webhook_url()
        if stored_webhook:
            self._webhook_edit.setText(stored_webhook)
            self._set_webhook_status(True, "Webhook URL loaded from .env.")
        else:
            self._set_webhook_status(None, "No webhook URL set.")

        layout.addWidget(webhook_group)

        # --- Webhook events group -------------------------------------------
        events_group = QGroupBox("Webhook Events")
        events_layout = QVBoxLayout(events_group)
        events_layout.setSpacing(6)

        events_hint = QLabel(
            "Choose which events post to the webhook. "
            "Defaults to enabled for new events."
        )
        events_hint.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        events_hint.setWordWrap(True)
        events_layout.addWidget(events_hint)

        self._event_match_start_check = QCheckBox("Match start")
        self._event_match_start_check.setToolTip(
            "Post an embed with the match rules (and settings screenshot) "
            "when a new match begins."
        )
        self._event_match_start_check.setChecked(
            load_event_enabled(EVENT_MATCH_START)
        )
        self._event_match_start_check.toggled.connect(
            self._on_event_match_start_toggled
        )
        events_layout.addWidget(self._event_match_start_check)

        self._event_race_results_check = QCheckBox("Race results (team mode only)")
        self._event_race_results_check.setToolTip(
            "Post a per-race embed with the winning team and per-team "
            "scores after each race. Only fires when the match is using a "
            "team mode and team scores can be computed from the results."
        )
        self._event_race_results_check.setChecked(
            load_event_enabled(EVENT_RACE_RESULTS)
        )
        self._event_race_results_check.toggled.connect(
            self._on_event_race_results_toggled
        )
        events_layout.addWidget(self._event_race_results_check)

        self._event_match_end_check = QCheckBox("Match end (results table)")
        self._event_match_end_check.setToolTip(
            "Post an embed with the winner and the generated results "
            "table image when a match completes."
        )
        self._event_match_end_check.setChecked(
            load_event_enabled(EVENT_MATCH_END)
        )
        self._event_match_end_check.toggled.connect(
            self._on_event_match_end_toggled
        )
        events_layout.addWidget(self._event_match_end_check)

        layout.addWidget(events_group)

        # --- Debug group -----------------------------------------------------
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.setSpacing(6)

        self._debug_mode_check = QCheckBox("Enable debug mode")
        self._debug_mode_check.setToolTip(
            "Emit additional logging during a race. Persisted in .env."
        )
        self._debug_mode_check.setChecked(self._state_machine.debug_mode)
        self._debug_mode_check.toggled.connect(self._on_debug_mode_toggled)
        debug_layout.addWidget(self._debug_mode_check)

        layout.addWidget(debug_group)

        font_group = QGroupBox("Game font (optional)")
        font_layout = QVBoxLayout(font_group)
        font_row = QHBoxLayout()
        self._game_font_label = QLabel()
        self._game_font_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        font_row.addWidget(self._game_font_label, 1)
        select_font = QPushButton("Select Font…")
        select_font.clicked.connect(self._choose_game_font)
        font_row.addWidget(select_font)
        clear_font = QPushButton("Clear")
        clear_font.clicked.connect(self._clear_game_font)
        font_row.addWidget(clear_font)
        font_layout.addLayout(font_row)
        self._game_font_status = QLabel("Checks names against every visible glyph in your local game font. Visually equivalent characters share an identity; ambiguous reads remain marked for review.")
        self._game_font_status.setWordWrap(True)
        font_layout.addWidget(self._game_font_status)
        layout.addWidget(font_group)
        self._refresh_game_font_label()

        layout.addStretch()

        # Load stored values and auto-verify if a key exists
        stored_key = load_api_key()
        stored_model = load_model()
        self._api_model_combo.setCurrentText(stored_model)
        # Hook up auto-save AFTER initial population so we don't write back on load.
        self._api_model_combo.currentTextChanged.connect(self._on_model_changed)
        if stored_key:
            self._api_key_edit.setText(stored_key)
            self._set_api_status(None, "Verifying API key...")
            self._run_verify(stored_key, stored_model)
        else:
            self._set_api_status(False, "No API key set. Gemini features are unavailable.")

        return widget

    def _refresh_game_font_label(self) -> None:
        path = configured_font_path()
        self._game_font_label.setText(Path(path).name if path else "No font selected")
        self._game_font_label.setToolTip(path)

    def _choose_game_font(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select game font", "", "Fonts (*.ttf *.otf);;All files (*)")
        if not path:
            return
        try:
            save_font_path(path)
        except (OSError, ValueError) as error:
            self._game_font_status.setText(f"Could not load font: {error}")
            return
        self._refresh_game_font_label()
        self._game_font_status.setText("Font selected. Use Refresh Names to update saved results.")

    def _clear_game_font(self) -> None:
        save_font_path("")
        self._refresh_game_font_label()
        self._game_font_status.setText("Game-font verification disabled.")

    def _set_api_status(self, ok: bool | None, message: str) -> None:
        """Update the status label. ok=True=green, ok=False=red, ok=None=grey."""
        if ok is True:
            color = "#4a4"
        elif ok is False:
            color = "#c44"
        else:
            color = "#888"
        self._api_status_label.setStyleSheet(f"QLabel {{ color: {color}; font-style: italic; }}")
        self._api_status_label.setText(message)

    def _on_save_api_key(self) -> None:
        key = self._api_key_edit.text().strip()
        if not key:
            self._set_api_status(False, "Cannot save an empty key.")
            return
        save_api_key(key)
        self._set_api_status(None, "API key saved. Use Verify Key to check it.")

    def _on_model_changed(self, model: str) -> None:
        model = model.strip()
        if model:
            save_model(model)

    def _on_debug_mode_toggled(self, checked: bool) -> None:
        save_debug_mode(checked)
        self._state_machine.debug_mode = checked
        self.statusBar().showMessage(
            f"Debug mode {'enabled' if checked else 'disabled'}", 2000,
        )
        logger.info("Debug mode %s", "enabled" if checked else "disabled")

    def _on_toggle_key_visibility(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self._api_key_edit.setEchoMode(mode)

    def _set_webhook_status(self, ok: bool | None, message: str) -> None:
        """Update the webhook status label. ok=True=green, ok=False=red, ok=None=grey."""
        if ok is True:
            color = "#4a4"
        elif ok is False:
            color = "#c44"
        else:
            color = "#888"
        self._webhook_status_label.setStyleSheet(
            f"QLabel {{ color: {color}; font-style: italic; }}"
        )
        self._webhook_status_label.setText(message)

    def _on_save_webhook_url(self) -> None:
        url = self._webhook_edit.text().strip()
        if not url:
            save_webhook_url("")
            self._set_webhook_status(None, "Webhook URL cleared.")
            logger.info("Discord webhook URL cleared.")
            return
        save_webhook_url(url)
        self._set_webhook_status(True, "Webhook URL saved.")
        self.statusBar().showMessage("Discord webhook URL saved", 2000)
        logger.info("Discord webhook URL saved.")

    def _on_toggle_webhook_visibility(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self._webhook_edit.setEchoMode(mode)

    def _on_event_match_start_toggled(self, checked: bool) -> None:
        save_event_enabled(EVENT_MATCH_START, checked)
        self.statusBar().showMessage(
            f"Match-start webhook {'enabled' if checked else 'disabled'}", 2000,
        )
        logger.info(
            "Discord webhook event MATCH_START %s",
            "enabled" if checked else "disabled",
        )

    def _on_event_race_results_toggled(self, checked: bool) -> None:
        save_event_enabled(EVENT_RACE_RESULTS, checked)
        self.statusBar().showMessage(
            f"Race-results webhook {'enabled' if checked else 'disabled'}", 2000,
        )
        logger.info(
            "Discord webhook event RACE_RESULTS %s",
            "enabled" if checked else "disabled",
        )

    def _on_event_match_end_toggled(self, checked: bool) -> None:
        save_event_enabled(EVENT_MATCH_END, checked)
        self.statusBar().showMessage(
            f"Match-end webhook {'enabled' if checked else 'disabled'}", 2000,
        )
        logger.info(
            "Discord webhook event MATCH_END %s",
            "enabled" if checked else "disabled",
        )

    def _on_ping_webhook(self) -> None:
        url = self._webhook_edit.text().strip()
        if not url:
            self._set_webhook_status(False, "No webhook URL entered.")
            return
        if hasattr(self, "_webhook_ping_thread") and self._webhook_ping_thread.isRunning():
            return
        self._set_webhook_status(None, "Sending test message…")
        self._webhook_ping_btn.setEnabled(False)
        self._webhook_ping_thread = _WebhookPingThread(url)
        self._webhook_ping_thread.finished.connect(self._on_ping_done)
        self._webhook_ping_thread.start()

    def _on_ping_done(self, ok: bool, message: str) -> None:
        self._webhook_ping_btn.setEnabled(True)
        if ok:
            self._set_webhook_status(True, "Test message delivered — check your Discord channel.")
            self.statusBar().showMessage("Discord webhook test sent", 2500)
            logger.info("Discord webhook test message delivered.")
        else:
            self._set_webhook_status(False, f"Test failed: {message}")
            logger.warning("Discord webhook test failed: %s", message)

    def _on_verify_api_key(self) -> None:
        key = self._api_key_edit.text().strip()
        model = self._api_model_combo.currentText().strip()
        if not key:
            self._set_api_status(False, "No API key entered.")
            return
        self._set_api_status(None, "Verifying...")
        self._run_verify(key, model)

    def _run_verify(self, key: str, model: str) -> None:
        if hasattr(self, "_verify_thread") and self._verify_thread.isRunning():
            return
        self._verify_thread = _VerifyThread(key, model)
        self._verify_thread.finished.connect(self._on_verify_done)
        self._verify_thread.start()

    def _on_verify_done(self, ok: bool, message: str) -> None:
        if ok:
            self._set_api_status(True, "API key verified successfully.")
            self._gemini_indicator.setStyleSheet("QLabel { color: #4a4; font-size: 16px; }")
            self._gemini_indicator.setToolTip("Gemini API: connected")
            logger.info("Gemini API key verified successfully.")
        else:
            self._set_api_status(False, f"Verification failed: {message}")
            self._gemini_indicator.setStyleSheet("QLabel { color: #c44; font-size: 16px; }")
            self._gemini_indicator.setToolTip(f"Gemini API: {message}")
            logger.warning("Gemini API key verification failed: %s", message)

    # ------------------------------------------------------------------
    # Settings ↔ state machine synchronisation
    # ------------------------------------------------------------------

    def _settings_from_ui(self) -> MatchSettings:
        """Build a ``MatchSettings`` from the current widget values."""
        return MatchSettings(
            cc_class=self._cc_combo.currentText(),
            teams=self._teams_combo.currentText(),
            items=self._items_combo.currentText(),
            com_difficulty=self._com_combo.currentText(),
            race_count=self._race_count_spin.value(),
            intermission=self._intermission_combo.currentText(),
        )

    def _load_settings_into_ui(self, settings: MatchSettings) -> None:
        """Update every widget to reflect *settings*, without triggering
        ``_on_settings_changed`` recursively."""
        for widget in (
            self._cc_combo, self._teams_combo, self._items_combo,
            self._com_combo, self._intermission_combo, self._race_count_spin,
        ):
            widget.blockSignals(True)

        self._cc_combo.setCurrentText(settings.cc_class)
        self._teams_combo.setCurrentText(settings.teams)
        self._items_combo.setCurrentText(settings.items)
        self._com_combo.setCurrentText(settings.com_difficulty)
        self._race_count_spin.setValue(settings.race_count)
        self._intermission_combo.setCurrentText(settings.intermission)

        for widget in (
            self._cc_combo, self._teams_combo, self._items_combo,
            self._com_combo, self._intermission_combo, self._race_count_spin,
        ):
            widget.blockSignals(False)

    def _push_settings_to_state_machine(self) -> None:
        self._state_machine.match_settings = self._settings_from_ui()

    def _on_settings_changed(self) -> None:
        """Called when the user edits any setting widget."""
        self._push_settings_to_state_machine()
        self._settings_status.setText("Manual")
        self.statusBar().showMessage("Match settings updated", 1500)
        self._settings_status.setStyleSheet(
            "QLabel { color: #888; font-style: italic; margin-top: 4px; }"
        )

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        if self._main_tabs.widget(index) is self._live_view_tab and not self._sources_loaded:
            QTimer.singleShot(0, self._refresh_sources)
        if self._main_tabs.widget(index) is not self._video_review_tab:
            self._paused = True
            self._playback_clock = None
            self._play_btn.setText("Play")
        elif self._last_frame is not None:
            QTimer.singleShot(0, lambda: self._paint_frame(self._last_frame))
        if self._main_tabs.widget(index) is self._history_view:
            self._history_view.refresh()

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _refresh_sources(self) -> None:
        self._sources_loaded = True
        self.statusBar().showMessage("Scanning for video sources...")
        self._sources = enumerate_sources()

        self._combo.blockSignals(True)
        self._combo.clear()
        if self._sources:
            for src in self._sources:
                self._combo.addItem(src["label"], userData=src["index"])
            self._combo.blockSignals(False)
            self._combo.setCurrentIndex(0)
            self._on_source_changed(0)
        else:
            self._combo.addItem("No sources found")
            self._combo.blockSignals(False)
            self._live_capture.close()
            self.statusBar().showMessage("No video sources detected.")

    def _on_source_changed(self, combo_index: int) -> None:
        if combo_index < 0 or combo_index >= len(self._sources):
            return
        index = self._sources[combo_index]["index"]
        self._state_machine.reset()
        self._push_settings_to_state_machine()
        ok = self._live_capture.open(index)
        self._on_mode_changed()
        if ok:
            self.statusBar().showMessage(
                f"Opened {self._sources[combo_index]['label']}"
            )
        else:
            self.statusBar().showMessage(
                f"Failed to open {self._sources[combo_index]['label']}"
            )

    # ------------------------------------------------------------------
    # Frame update loop
    # ------------------------------------------------------------------

    def _update_frame(self) -> None:
        if not self._capture.file_path or self._paused:
            return
        if self._capture.file_path and self._playback_clock is not None:
            started_at, started_position = self._playback_clock
            target = started_position + time.monotonic() - started_at
            if target >= self._capture.duration:
                self._paused = True
                self._playback_clock = None
                self._play_btn.setText("Play")
                return
            if int(target * self._capture.fps) < self._capture.next_frame_index:
                return
            frame = self._capture.read_frame_at(target)
        else:
            frame = self._capture.read_frame()
        if frame is None:
            if self._capture.file_path:
                self._paused = True
                self._play_btn.setText("Play")
            return

        if self._capture.file_path:
            if not self._seek_slider.isSliderDown():
                self._seek_slider.setValue(round(self._capture.position * 1000))
            self._video_time.setText(f"{timestamp(self._capture.position)} / {timestamp(self._capture.duration)}")

        self._last_frame = frame
        self._paint_frame(frame)

    def _update_live_frame(self) -> None:
        frame = self._live_capture.read_frame()
        if frame is None:
            return
        self._live_last_frame = frame
        self._frame_count += 1

        if self._state_machine.state in (
            GameState.WAITING_FOR_RACE_END,
            GameState.DETECTING_RACE_RANK,
            GameState.READING_RACE_RESULTS,
        ):
            detect_interval = _DETECT_FAST_N_FRAMES
        else:
            detect_interval = _DETECT_EVERY_N_FRAMES

        if (
            self._mode_combo.currentText() == "Custom Room"
            and self._frame_count % detect_interval == 0
        ):
            prev_settings = self._state_machine.match_settings
            self._state_machine.update(frame)
            self._update_state_label()

            # If the state machine just detected match settings from the
            # video feed, sync them into the UI widgets.
            new_settings = self._state_machine.match_settings
            if (
                new_settings is not None
                and new_settings is not prev_settings
                and prev_settings is not new_settings
            ):
                self._load_settings_into_ui(new_settings)
                self._settings_status.setText("Detected")
                self._settings_status.setStyleSheet(
                    "QLabel { color: #4a4; font-weight: bold; margin-top: 4px; }"
                )

        # Skip the expensive scale/paint when the history tab is showing —
        # but tick the history view so the live match timeline refreshes.
        if self._main_tabs.currentWidget() is self._history_view:
            self._history_view.tick()
            return

        # Convert BGR -> RGB then to QImage
        if self._main_tabs.currentWidget() is self._live_view_tab:
            self._paint_frame(frame, self._video_label)

    def _paint_frame(self, frame: np.ndarray, label: QLabel | None = None) -> None:
        label = label if label is not None else self._review_video_label
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Scale to label size, keeping aspect ratio
        pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_last_frame", None) is not None:
            QTimer.singleShot(0, lambda: self._paint_frame(self._last_frame))
        if getattr(self, "_live_last_frame", None) is not None:
            QTimer.singleShot(0, lambda: self._paint_frame(self._live_last_frame, self._video_label))

    def _update_state_label(self) -> None:
        if hasattr(self, "_mode_combo"):
            self._mode_combo.setEnabled(not self._state_machine.is_match_active)
            if self._mode_combo.currentText() != "Custom Room":
                self._state_label.setText("PREVIEW ONLY")
                self._race_label.clear()
                return
        self._state_label.setText(self._state_machine.state.name)
        race = self._state_machine.current_race
        if race > 0:
            total = self._state_machine.match_settings.race_count if self._state_machine.match_settings else "?"
            self._race_label.setText(f"Race {race}/{total}")
        else:
            self._race_label.setText("")

    def _on_advance(self) -> None:
        self._state_machine.advance()
        self._update_state_label()

    def _on_start_manual_match(self) -> None:
        # Push current UI settings into the state machine first so the
        # manual match captures whatever the user currently has selected.
        self._push_settings_to_state_machine()
        started = self._state_machine.start_manual_match()
        self._update_state_label()
        if started:
            self.statusBar().showMessage("Manual match started", 2500)
        else:
            self.statusBar().showMessage(
                "Manual match not started — already in progress (use Reset first)",
                3000,
            )

    def _on_reset(self) -> None:
        self._state_machine.reset()
        self._update_state_label()
        # Wipe detected settings — push the current UI values as manual.
        self._settings_status.setText("Manual")
        self._settings_status.setStyleSheet(
            "QLabel { color: #888; font-style: italic; margin-top: 4px; }"
        )
        self._push_settings_to_state_machine()

    def _on_capture(self) -> None:
        if self._live_last_frame is None:
            self.statusBar().showMessage("No frame to capture.")
            return
        out = Path(_CAPTURE_DIR)
        out.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        path = out / name
        cv2.imwrite(str(path), self._live_last_frame)
        self.statusBar().showMessage(f"Saved {path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._live_timer.stop()
        self._stop_video_analysis()
        self._capture.close()
        self._live_capture.close()
        super().closeEvent(event)
