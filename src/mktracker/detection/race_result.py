from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Panel occupies the right portion of the screen.
# Tuned for 1920x1080 capture output.
_PANEL_X1 = 0.47
_PANEL_X2 = 0.82
_PANEL_Y1 = 0.04
_PANEL_Y2 = 0.96

# Detection: minimum percentage of rows with strong horizontal edges.
_EDGE_ROW_THRESHOLD = 40
_EDGE_PERCENT_MIN = 10.0

# Target panel width for OCR (in pixels). Panels smaller than this get
# upscaled; larger ones are passed as-is.
_OCR_TARGET_WIDTH = 1200

# Minimum times a placement must be seen to be accepted.
_MIN_CONSENSUS = 1


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

    def read_results(self, frame: np.ndarray) -> dict[int, str]:
        """OCR the results panel. Returns {placement: name}, possibly empty."""
        h, w = frame.shape[:2]
        panel = frame[
            int(h * _PANEL_Y1) : int(h * _PANEL_Y2),
            int(w * _PANEL_X1) : int(w * _PANEL_X2),
        ]
        pw = panel.shape[1]
        if pw < _OCR_TARGET_WIDTH:
            scale = _OCR_TARGET_WIDTH / pw
            panel = cv2.resize(
                panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        text: str = pytesseract.image_to_string(panel, config="--psm 6")
        return _parse_results(text)


class RaceResultAccumulator:
    """Accumulates placement readings across multiple frames and resolves
    the best name per placement via consensus voting."""

    def __init__(self) -> None:
        self._readings: defaultdict[int, list[str]] = defaultdict(list)

    def add_frame(self, results: dict[int, str]) -> None:
        for num, name in results.items():
            self._readings[num].append(name)

    @property
    def placement_count(self) -> int:
        """Number of unique placements seen (before consensus filtering)."""
        return len(self._readings)

    def finalize(self) -> dict[int, str]:
        """Return the consensus results, filtered and deduplicated."""
        # Pick the most-common reading per placement, requiring min count.
        candidates: dict[int, tuple[str, int]] = {}  # num -> (name, count)
        for num in sorted(self._readings):
            counts = Counter(self._readings[num])
            best_name, best_count = counts.most_common(1)[0]
            if best_count >= _MIN_CONSENSUS:
                candidates[num] = (best_name, best_count)

        # Deduplicate: if the same name appears at multiple placements,
        # keep the one with the highest consensus count for that name.
        name_to_nums: defaultdict[str, list[int]] = defaultdict(list)
        for num, (name, _count) in candidates.items():
            name_to_nums[name].append(num)
        for name, nums in name_to_nums.items():
            if len(nums) > 1:
                best_num = max(nums, key=lambda n: candidates[n][1])
                for n in nums:
                    if n != best_num:
                        del candidates[n]

        return {num: name for num, (name, _) in candidates.items()}

    def clear(self) -> None:
        self._readings.clear()


def _parse_results(text: str) -> dict[int, str]:
    """Extract {placement: name} pairs from raw OCR text."""
    results: dict[int, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m = re.search(
            r"(\d{1,2})(?!\d)"         # placement number (not followed by digit)
            r".{0,20}?"                # avatar/junk (up to 20 chars, non-greedy)
            r"([A-Z][A-Za-z_.  ]+)",   # name starting with uppercase
            line,
        )
        # Fallback: try lowercase-start names (e.g. "luigi")
        if not m:
            m = re.search(
                r"(\d{1,2})(?!\d)"
                r".{0,20}?"
                r"([a-z][a-z]{3,})",   # lowercase name, min 4 chars
                line,
            )
        if not m:
            continue

        num = int(m.group(1))
        name = m.group(2).strip()
        # Capitalize lowercase fallback names.
        if name[0].islower():
            name = name.capitalize()

        # Clean leading avatar junk (single uppercase + space).
        name = re.sub(r"^[A-Z]{1,2}\s+(?=[A-Z])", "", name)
        # Clean trailing junk: symbols, then short lowercase fragments.
        name = re.sub(r"\s+[^A-Za-z\s].*$", "", name)
        name = re.sub(r"(\s+[a-z]{1,3})+\s*$", "", name)
        name = re.sub(r"\s+\S$", "", name)
        name = name.rstrip(".")

        if len(name) < 4 or num < 1 or num > 24:
            continue
        if num not in results:
            results[num] = name

    return results
