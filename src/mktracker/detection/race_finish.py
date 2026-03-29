"""Detect the FINISH! screen that appears when a race ends."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tight ROI around where the FINISH! text sits (x1, y1, x2, y2 normalised).
_CENTER_ROI = (0.15, 0.33, 0.82, 0.58)

# HSV range for the orange-yellow FINISH text.
_HUE_LOW, _HUE_HIGH = 15, 35
_SAT_MIN = 150
_VAL_MIN = 180

# Orange pixel ratio bounds — the full text sits in a narrow band;
# partial animations (overlapping letters) push the ratio above the max.
_PIXEL_RATIO_MIN = 0.20
_PIXEL_RATIO_MAX = 0.35

# The ROI is split into 5 vertical strips; every strip must exceed this
# threshold so that scattered orange (GO!, environment) is rejected.
_STRIP_COUNT = 5
_STRIP_MIN = 0.08


class RaceFinishDetector:
    """Detects the FINISH! banner displayed at the end of a race."""

    def is_active(self, frame: np.ndarray) -> bool:
        """Return True if *frame* shows the fully-displayed FINISH! text."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (
            int(w * _CENTER_ROI[0]),
            int(h * _CENTER_ROI[1]),
            int(w * _CENTER_ROI[2]),
            int(h * _CENTER_ROI[3]),
        )
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([_HUE_LOW, _SAT_MIN, _VAL_MIN]),
            np.array([_HUE_HIGH, 255, 255]),
        )

        ratio = float(np.count_nonzero(mask)) / mask.size
        if not (_PIXEL_RATIO_MIN <= ratio <= _PIXEL_RATIO_MAX):
            return False

        # Verify the orange is spread across the full width of the text.
        strip_w = mask.shape[1] // _STRIP_COUNT
        for i in range(_STRIP_COUNT):
            strip = mask[:, i * strip_w : (i + 1) * strip_w]
            if float(np.count_nonzero(strip)) / strip.size < _STRIP_MIN:
                return False

        return True
