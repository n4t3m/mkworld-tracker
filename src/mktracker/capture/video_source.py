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
