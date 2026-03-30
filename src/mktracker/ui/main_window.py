from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mktracker.capture.video_source import VideoCapture, enumerate_sources
from mktracker.detection.match_settings import MatchSettings
from mktracker.state_machine import GameState, GameStateMachine

_CAPTURE_DIR = "captured_frames"

logger = logging.getLogger(__name__)

_FRAME_INTERVAL_MS = 33  # ~30 fps
_DETECT_EVERY_N_FRAMES = 15  # run detection every ~500 ms
_DETECT_FAST_N_FRAMES = 3   # faster sampling during race results reading


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MK Tracker")
        self.resize(1280, 800)

        self._capture = VideoCapture()
        self._sources: list[dict] = []

        self._state_machine = GameStateMachine()
        self._frame_count = 0
        self._last_frame: np.ndarray | None = None

        self._build_ui()
        self._refresh_sources()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(_FRAME_INTERVAL_MS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- toolbar row ---
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
        self._race_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        toolbar.addWidget(self._race_label)

        self._state_label = QLabel()
        self._state_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._update_state_label()
        toolbar.addWidget(self._state_label)

        advance_btn = QPushButton("Advance State")
        advance_btn.clicked.connect(self._on_advance)
        toolbar.addWidget(advance_btn)

        reset_btn = QPushButton("Reset State")
        reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(reset_btn)

        capture_btn = QPushButton("Capture Frame")
        capture_btn.clicked.connect(self._on_capture)
        toolbar.addWidget(capture_btn)

        layout.addLayout(toolbar)

        # --- main content: video + settings side panel ---
        content = QHBoxLayout()
        content.setSpacing(8)

        # --- video display ---
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setText("No source selected")
        self._video_label.setStyleSheet(
            "background-color: #111; color: #555; font-size: 16px;"
        )
        content.addWidget(self._video_label, stretch=1)

        # --- match settings panel ---
        content.addWidget(self._build_settings_panel())

        layout.addLayout(content, stretch=1)

        self.setStatusBar(QStatusBar())

    def _build_settings_panel(self) -> QGroupBox:
        group = QGroupBox("Match Settings")
        group.setFixedWidth(250)
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
            "color: #888; font-style: italic; margin-top: 4px;"
        )
        form.addRow("Source:", self._settings_status)

        # Push the UI values into the state machine on startup.
        self._push_settings_to_state_machine()

        return group

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
            "color: #888; font-style: italic; margin-top: 4px;"
        )

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _refresh_sources(self) -> None:
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
            self._capture.close()
            self.statusBar().showMessage("No video sources detected.")

    def _on_source_changed(self, combo_index: int) -> None:
        if combo_index < 0 or combo_index >= len(self._sources):
            return
        index = self._sources[combo_index]["index"]
        ok = self._capture.open(index)
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
        frame = self._capture.read_frame()
        if frame is None:
            return

        self._last_frame = frame
        self._frame_count += 1

        if self._state_machine.state is GameState.READING_RACE_RESULTS:
            detect_interval = _DETECT_FAST_N_FRAMES
        else:
            detect_interval = _DETECT_EVERY_N_FRAMES

        if self._frame_count % detect_interval == 0:
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
                    "color: #4a4; font-weight: bold; margin-top: 4px;"
                )

        # Convert BGR -> RGB then to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Scale to label size, keeping aspect ratio
        pixmap = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(pixmap)

    def _update_state_label(self) -> None:
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

    def _on_reset(self) -> None:
        self._state_machine.reset()
        self._update_state_label()
        # Wipe detected settings — push the current UI values as manual.
        self._settings_status.setText("Manual")
        self._settings_status.setStyleSheet(
            "color: #888; font-style: italic; margin-top: 4px;"
        )
        self._push_settings_to_state_machine()

    def _on_capture(self) -> None:
        if self._last_frame is None:
            self.statusBar().showMessage("No frame to capture.")
            return
        out = Path(_CAPTURE_DIR)
        out.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        path = out / name
        cv2.imwrite(str(path), self._last_frame)
        self.statusBar().showMessage(f"Saved {path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._capture.close()
        super().closeEvent(event)
