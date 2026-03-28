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

# Detection: minimum strong horizontal edges in right panel.
_EDGE_THRESHOLD = 150

_OCR_SCALE = 3


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
        return int(np.sum(row_edge > 50)) > _EDGE_THRESHOLD

    def is_race_result(self, text: str) -> bool:
        """Check if OCR text contains '+N' deltas (race result, not standings)."""
        return bool(re.search(r"\+\d{1,2}\s+\d", text))

    def read_results(self, frame: np.ndarray) -> dict[int, str] | None:
        """OCR the results panel. Returns {placement: name} or None.

        Returns None if the screen is detected as overall standings
        (no '+' deltas) rather than race results.
        """
        h, w = frame.shape[:2]
        panel = frame[
            int(h * _PANEL_Y1) : int(h * _PANEL_Y2),
            int(w * _PANEL_X1) : int(w * _PANEL_X2),
        ]
        panel_up = cv2.resize(
            panel, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC
        )
        text: str = pytesseract.image_to_string(panel_up, config="--psm 6")

        if not self.is_race_result(text):
            return None

        return _parse_results(text)


def _parse_results(text: str) -> dict[int, str]:
    """Extract {placement: name} pairs from raw OCR text."""
    results: dict[int, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Look for a 1-2 digit number near the start (the placement),
        # followed eventually by a recognizable player name.
        m = re.search(
            r"(?:^|(?<=\s))(\d{1,2})\s+\S*\s*"  # placement + avatar junk
            r"([A-Za-z][\w._\- ]*?)"              # name (letters, digits, _.-  )
            r"\s+[+\d]",                           # followed by delta or score
            line,
        )
        if not m:
            continue

        num = int(m.group(1))
        name = m.group(2).strip()

        # Drop short noise and trailing single-char artefacts.
        name = re.sub(r"\s+\S$", "", name).strip()
        if len(name) < 2 or num < 1 or num > 24:
            continue
        if num not in results:
            results[num] = name

    return results
