from __future__ import annotations

import sys

import cv2
import numpy as np


def _select_capture_backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


_CAPTURE_BACKEND = _select_capture_backend()


def enumerate_sources(max_index: int = 10) -> list[dict]:
    """Probe camera indices and return those that open successfully."""
    sources = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _CAPTURE_BACKEND)
        if cap.isOpened():
            sources.append({"index": i, "label": f"Camera {i}"})
            cap.release()
    return sources


class VideoCapture:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._index: int | None = None
        self.file_path: str | None = None

    @property
    def fps(self) -> float:
        return (self._cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_open else 30.0

    @property
    def duration(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps if self.is_open and self.file_path else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        if not self.is_open:
            return (0, 0)
        return (round(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                round(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def position(self) -> float:
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if self.is_open else 0.0

    def seek(self, seconds: float) -> None:
        if self.is_open and self.file_path:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, min(seconds, self.duration)) * 1000)

    def open_file(self, path: str) -> bool:
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        self.file_path = path
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self, index: int) -> bool:
        self.close()
        cap = cv2.VideoCapture(index, _CAPTURE_BACKEND)
        if not cap.isOpened():
            return False
        self._cap = cap
        self._index = index
        return True

    def close(self) -> None:
        self.file_path = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._index = None

    def read_frame(self) -> np.ndarray | None:
        if not self.is_open:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    @property
    def next_frame_index(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.is_open else 0

    def read_frame_at(self, seconds: float) -> np.ndarray | None:
        """Read the frame due on the playback clock, discarding late frames."""
        if not self.is_open or not self.file_path:
            return self.read_frame()
        target = max(0, int(seconds * self.fps))
        behind = target - self.next_frame_index
        if behind < 0 or behind > self.fps / 2:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        else:
            for _ in range(behind):
                if not self._cap.grab():
                    return None
        return self.read_frame()
