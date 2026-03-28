from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mktracker.capture.video_source import VideoCapture, enumerate_sources
from mktracker.state_machine import GameStateMachine

logger = logging.getLogger(__name__)

_FRAME_INTERVAL_MS = 33  # ~30 fps
_DETECT_EVERY_N_FRAMES = 15  # run detection every ~500 ms


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MK Tracker")
        self.resize(1280, 800)

        self._capture = VideoCapture()
        self._sources: list[dict] = []

        self._state_machine = GameStateMachine()
        self._frame_count = 0

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
        layout.addLayout(toolbar)

        # --- video display ---
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setStyleSheet("background-color: #111;")
        self._video_label.setText("No source selected")
        self._video_label.setStyleSheet(
            "background-color: #111; color: #555; font-size: 16px;"
        )
        layout.addWidget(self._video_label, stretch=1)

        self.setStatusBar(QStatusBar())

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

        # Run detection at a reduced rate to avoid OCR overhead every frame
        self._frame_count += 1
        if self._frame_count % _DETECT_EVERY_N_FRAMES == 0:
            self._run_detection(frame)

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

    def _run_detection(self, frame: np.ndarray) -> None:
        self._state_machine.update(frame)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._capture.close()
        super().closeEvent(event)
