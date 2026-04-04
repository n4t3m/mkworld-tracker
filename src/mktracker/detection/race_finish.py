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
_PIXEL_RATIO_MAX = 0.42

# The ROI is split into 5 vertical strips; at least _STRIP_REQUIRED of them
# must exceed the threshold so that scattered orange (GO!, environment) is
# rejected while allowing slight horizontal shifts of the text.
_STRIP_COUNT = 5
_STRIP_REQUIRED = 3
_STRIP_MIN = 0.08

# The FINISH text has a red outline (H 0-10).  Require a minimum red ratio
# in the ROI to reject desert sand / boost effects that produce diffuse
# orange without the characteristic red border.
_RED_HUE_LOW, _RED_HUE_HIGH = 0, 10
_RED_SAT_MIN = 100
_RED_VAL_MIN = 100
_RED_RATIO_MIN = 0.08

# Horizontal spread check: FINISH! spans ~7 characters while GO! spans ~3.
# Split the ROI into finer strips and require a minimum number of contiguous
# strips with significant orange to reject narrow text like GO!.
_FINE_STRIP_COUNT = 10
_FINE_STRIP_MIN = 0.15
_FINE_CONTIGUOUS_REQUIRED = 5


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

        # Verify the orange is spread across most of the text width.
        strip_w = mask.shape[1] // _STRIP_COUNT
        passing_strips = 0
        for i in range(_STRIP_COUNT):
            strip = mask[:, i * strip_w : (i + 1) * strip_w]
            if float(np.count_nonzero(strip)) / strip.size >= _STRIP_MIN:
                passing_strips += 1

        if passing_strips < _STRIP_REQUIRED:
            return False

        # Reject narrow text (GO!) and one-sided environment orange by
        # requiring the orange to span enough contiguous fine strips AND
        # start in the left half of the ROI (FINISH! is always centred).
        fine_w = mask.shape[1] // _FINE_STRIP_COUNT
        best_run = 0
        best_start = 0
        current_run = 0
        run_start = 0
        for i in range(_FINE_STRIP_COUNT):
            fine_strip = mask[:, i * fine_w : (i + 1) * fine_w]
            if float(np.count_nonzero(fine_strip)) / fine_strip.size >= _FINE_STRIP_MIN:
                if current_run == 0:
                    run_start = i
                current_run += 1
                if current_run > best_run:
                    best_run = current_run
                    best_start = run_start
            else:
                current_run = 0
        if best_run < _FINE_CONTIGUOUS_REQUIRED or best_start > _FINE_STRIP_COUNT // 2 - 1:
            return False

        # The FINISH text has a red outline that gameplay orange lacks.
        red_mask = cv2.inRange(
            hsv,
            np.array([_RED_HUE_LOW, _RED_SAT_MIN, _RED_VAL_MIN]),
            np.array([_RED_HUE_HIGH, 255, 255]),
        )
        red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size
        return red_ratio >= _RED_RATIO_MIN
