from __future__ import annotations

import difflib
import logging
import time

import cv2
import numpy as np
import pytesseract

from mktracker.detection.tracks import TRACK_NAMES

logger = logging.getLogger(__name__)

# The left ~42% of the screen is the dark player-list panel.
_LEFT_PANEL_RATIO = 0.42
# Maximum average brightness (0-255) for the left panel to count as "dark".
_DARK_THRESHOLD = 55
# Minimum brightness difference between right and left panels.
_BRIGHTNESS_DIFF_MIN = 18

# Normalised bounding box for the track-name text banner (x1, y1, x2, y2).
_TRACK_NAME_ROI = (0.52, 0.33, 0.85, 0.37)

# Scale factor applied to the ROI before OCR (small text needs upscaling).
_OCR_SCALE = 3

# Minimum characters for a valid track name.
_MIN_NAME_LENGTH = 3

# Minimum similarity (0-1) for fuzzy matching against the track list.
_MATCH_CUTOFF = 0.6

# Seconds to skip detection after a confirmed match.
_COOLDOWN_SECONDS = 15.0


class TrackSelectDetector:
    """Detects the Mario Kart track-selection screen and reads the track name.

    Designed to be called per-frame from a game-state machine.
    * ``is_active(frame)`` — fast brightness check (~<1 ms).
    * ``read_track_name(frame)`` — slower OCR step; call only when active.
    * ``detect(frame)`` — full pipeline with fuzzy matching and cooldown.
    """

    def __init__(self) -> None:
        self._last_match_time: float = 0.0

    def is_active(self, frame: np.ndarray) -> bool:
        """Return True if *frame* looks like the track-selection screen."""
        h, w = frame.shape[:2]
        split = int(w * _LEFT_PANEL_RATIO)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_mean = float(np.mean(gray[:, :split]))
        right_mean = float(np.mean(gray[:, split:]))

        return (
            left_mean < _DARK_THRESHOLD
            and (right_mean - left_mean) > _BRIGHTNESS_DIFF_MIN
        )

    def read_track_name(self, frame: np.ndarray) -> str | None:
        """OCR the track-name banner.  Returns the cleaned name or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _TRACK_NAME_ROI
        roi = frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]

        # Upscale for better OCR accuracy on small text.
        roi_up = cv2.resize(
            roi, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)

        # The banner is white text on a dark translucent background.
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        text: str = pytesseract.image_to_string(
            thresh, config="--psm 7"
        ).strip()

        if len(text) < _MIN_NAME_LENGTH:
            return None
        return text

    @staticmethod
    def match_track_name(ocr_text: str) -> str | None:
        """Return the closest canonical track name, or None if no good match."""
        matches = difflib.get_close_matches(
            ocr_text, TRACK_NAMES, n=1, cutoff=_MATCH_CUTOFF
        )
        return matches[0] if matches else None

    def detect(self, frame: np.ndarray) -> dict | None:
        """Full pipeline: screen check, OCR, fuzzy match, and cooldown.

        Returns ``{"track_name": str}`` or ``None``.
        The cooldown is managed externally by the state machine via
        ``_last_match_time``.
        """
        if time.monotonic() - self._last_match_time < _COOLDOWN_SECONDS:
            return None
        if not self.is_active(frame):
            return None
        raw = self.read_track_name(frame)
        if raw is None:
            return None
        name = self.match_track_name(raw)
        if name is None:
            logger.debug("OCR text '%s' did not match any known track", raw)
            return None
        return {"track_name": name}
