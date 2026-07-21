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

    def get_supported_resolutions(self) -> list[(int, int)]:
        common_resolutions = [
            (640, 480),    # VGA
            (1280, 720),   # HD
            (1920, 1080),  # Full HD
            (2560, 1440),  # QHD
            (3840, 2160)   # 4K UHD
        ]

        supported_resolutions = []
        for width, height in common_resolutions:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                supported_resolutions.append((width, height))

        return supported_resolutions

    def read_frame(self) -> np.ndarray | None:
        if not self.is_open:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame
