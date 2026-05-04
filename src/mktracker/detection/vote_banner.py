"""Detect the "The course has been selected!" yellow banner.

Shown briefly during the voting roulette right after the final track is
chosen, before the track-name banner appears on the map screen.  We use it
to recover a clean "race vote" frame from the rolling buffer the state
machine keeps during ``WAITING_FOR_TRACK_PICK``.

Detection is HSV-based: the banner is a horizontally-centred rectangle of
saturated bright yellow that spans a contiguous vertical band of roughly
~15-30 rows in the upper-mid portion of the frame.  We tighten the hue
range to specifically the yellow tone of this banner so desert sand,
gold scenery, and orange UI elements don't trigger false positives.
"""

from __future__ import annotations

import cv2
import numpy as np

# ROI: horizontally-centred band where the banner sits (the banner never
# reaches the side thumbnails, so we exclude the outer 20% on each side).
_ROI_X1, _ROI_X2 = 0.20, 0.80
_ROI_Y1, _ROI_Y2 = 0.13, 0.30

# Tight yellow hue range — excludes the orange tones used by sand,
# desert ground, and the FINISH! banner.
_HUE_LOW, _HUE_HIGH = 22, 32
_SAT_MIN = 140
_VAL_MIN = 180

# A row counts as "yellow" when at least this fraction of its pixels match
# the yellow mask.
_ROW_YELLOW_RATIO_MIN = 0.35

# Require this many contiguous high-density rows for a positive detection.
# Real banners occupy ~30 rows in 1080p frames; gameplay scenery yellow is
# scattered and never produces this kind of solid horizontal band.
_MIN_CONTIGUOUS_ROWS = 15


class VoteBannerDetector:
    """Detects the yellow "The course has been selected!" banner."""

    def is_active(self, frame: np.ndarray) -> bool:
        """Return ``True`` if *frame* shows the vote-confirmation banner."""
        h, w = frame.shape[:2]
        x1, y1 = int(w * _ROI_X1), int(h * _ROI_Y1)
        x2, y2 = int(w * _ROI_X2), int(h * _ROI_Y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([_HUE_LOW, _SAT_MIN, _VAL_MIN]),
            np.array([_HUE_HIGH, 255, 255]),
        )

        row_density = np.count_nonzero(mask, axis=1) / mask.shape[1]
        high = row_density >= _ROW_YELLOW_RATIO_MIN

        best_run = 0
        cur = 0
        for v in high:
            if v:
                cur += 1
                if cur > best_run:
                    best_run = cur
            else:
                cur = 0

        return best_run >= _MIN_CONTIGUOUS_ROWS
