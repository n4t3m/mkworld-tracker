from __future__ import annotations

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Panel occupies the right portion of the screen.
_PANEL_X1 = 0.35
_PANEL_X2 = 0.95
_PANEL_Y1 = 0.04
_PANEL_Y2 = 0.96

# Detection: minimum percentage of rows with strong horizontal edges.
_EDGE_ROW_THRESHOLD = 40
_EDGE_PERCENT_MIN = 10.0

# Target panel width for OCR (in pixels). Panels smaller than this get
# upscaled; larger ones are passed as-is.
_OCR_TARGET_WIDTH = 1800


class RaceResultDetector:
    """Detects the post-race results screen and reads placements."""

    def is_active(self, frame: np.ndarray) -> bool:
        """Fast check: does the right panel have the structured row pattern?"""
        h, w = frame.shape[:2]
        panel = cv2.cvtColor(
            frame[int(h * 0.05) : int(h * 0.95), int(w * 0.40) :],
            cv2.COLOR_BGR2GRAY,
        )
        sobel_y = cv2.Sobel(panel, cv2.CV_64F, 0, 1, ksize=3)
        row_edge = np.mean(np.abs(sobel_y), axis=1)
        strong_pct = np.sum(row_edge > _EDGE_ROW_THRESHOLD) / len(row_edge) * 100
        return strong_pct > _EDGE_PERCENT_MIN

    def is_race_result(self, text: str) -> bool:
        """Check if OCR text contains '+N' deltas (race result, not standings)."""
        return bool(re.search(r"\+\d{1,2}\s+\d", text))

    def read_results(self, frame: np.ndarray) -> dict[int, str]:
        """OCR the results panel. Returns {placement: name}, possibly empty."""
        h, w = frame.shape[:2]
        panel = frame[
            int(h * _PANEL_Y1) : int(h * _PANEL_Y2),
            int(w * _PANEL_X1) : int(w * _PANEL_X2),
        ]
        # Scale to a consistent size so OCR works at any capture resolution.
        pw = panel.shape[1]
        if pw < _OCR_TARGET_WIDTH:
            scale = _OCR_TARGET_WIDTH / pw
            panel = cv2.resize(
                panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        text: str = pytesseract.image_to_string(panel, config="--psm 6")
        return _parse_results(text)


def _parse_results(text: str) -> dict[int, str]:
    """Extract {placement: name} pairs from raw OCR text."""
    results: dict[int, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strategy: find a 1-2 digit placement number, then the player name.
        # The name is the longest capitalized-word sequence after the number.
        m = re.search(
            r"\b(\d{1,2})\b"           # placement number
            r"[^A-Za-z]*"              # avatar/junk between number and name
            r"([A-Z][A-Za-z_.  ]+)",   # name starting with uppercase
            line,
        )
        if not m:
            continue

        num = int(m.group(1))
        name = m.group(2).strip()

        # Clean leading avatar junk (single uppercase + space).
        name = re.sub(r"^[A-Z]{1,2}\s+(?=[A-Z])", "", name)
        # Clean trailing junk: symbols, then short lowercase fragments.
        name = re.sub(r"\s+[^A-Za-z\s].*$", "", name)
        name = re.sub(r"(\s+[a-z]{1,3})+\s*$", "", name)
        name = re.sub(r"\s+\S$", "", name)
        name = name.rstrip(".")
        if len(name) < 2 or num < 1 or num > 24:
            continue
        if num not in results:
            results[num] = name

    return results
