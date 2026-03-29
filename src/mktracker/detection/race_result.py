from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Panel occupies the right portion of the screen (1920x1080 capture).
_PANEL_X1 = 0.48
_PANEL_X2 = 0.88

# Row geometry.
_ROW_H_RATIO = 0.0713  # ~77px at 1080p
_MAX_VISIBLE_ROWS = 14

# Detection: minimum percentage of rows with strong horizontal edges.
_EDGE_ROW_THRESHOLD = 40
_EDGE_PERCENT_MIN = 10.0

# Minimum placements in a single frame to confirm results screen.
MIN_PLACEMENTS_TO_CONFIRM = 3


class RaceResultDetector:
    """Detects the post-race results screen and reads placements via per-row OCR."""

    def is_active(self, frame: np.ndarray) -> bool:
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
        """Per-row OCR with dynamic offset detection. Returns {placement: name}."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_h = int(h * _ROW_H_RATIO)
        offset = _find_row_offset(gray, h, w, row_h)
        x1, x2 = int(w * _PANEL_X1), int(w * _PANEL_X2)

        results: dict[int, str] = {}
        for i in range(_MAX_VISIBLE_ROWS):
            y1 = offset + i * row_h
            y2 = y1 + row_h
            if y2 > h:
                break

            row = frame[y1:y2, x1:x2]
            row_gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
            if np.max(row_gray) < 140:
                continue

            row_up = cv2.resize(
                row, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            g = cv2.cvtColor(row_up, cv2.COLOR_BGR2GRAY)

            # Try raw image first, then threshold 180 as fallback.
            for img in [row_up, cv2.threshold(g, 180, 255, cv2.THRESH_BINARY)[1]]:
                text = pytesseract.image_to_string(img, config="--psm 7").strip()
                num, name = _parse_line(text)
                if num is not None and num not in results:
                    results[num] = name
                    break

        return results


class RaceResultAccumulator:
    """Accumulates placement readings across frames and resolves via consensus."""

    def __init__(self) -> None:
        self._readings: defaultdict[int, list[str]] = defaultdict(list)

    def add_frame(self, results: dict[int, str]) -> None:
        for num, name in results.items():
            self._readings[num].append(name)

    @property
    def placement_count(self) -> int:
        return len(self._readings)

    def finalize(self) -> dict[int, str]:
        candidates: dict[int, tuple[str, int]] = {}
        for num in sorted(self._readings):
            counts = Counter(self._readings[num])
            best_name, best_count = counts.most_common(1)[0]
            candidates[num] = (best_name, best_count)

        # Deduplicate: same name at multiple placements → keep highest count.
        name_to_nums: defaultdict[str, list[int]] = defaultdict(list)
        for num, (name, _) in candidates.items():
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


def _find_row_offset(gray: np.ndarray, h: int, w: int, row_h: int) -> int:
    """Find the y-offset where row separators (brightness dips) align best."""
    panel = gray[:, int(w * _PANEL_X1) : int(w * _PANEL_X2)]
    profile = np.mean(panel, axis=1)

    best_off, best_score = 0, float("inf")
    for off in range(row_h):
        vals = [
            np.mean(profile[max(0, off + i * row_h - 2) : min(h, off + i * row_h + 3)])
            for i in range(_MAX_VISIBLE_ROWS)
            if off + i * row_h < h
        ]
        if vals:
            score = np.mean(vals)
            if score < best_score:
                best_score = score
                best_off = off
    return best_off


def _parse_line(text: str) -> tuple[int | None, str | None]:
    """Extract (placement, name) from a single OCR line."""
    m = re.search(
        r"(\d{1,2})(?!\d)"        # placement number
        r".{0,20}?"               # avatar junk
        r"([A-Z][A-Za-z_.  ]+)",  # name
        text,
    )
    if not m:
        m = re.search(r"(\d{1,2})(?!\d).{0,20}?([a-z][a-z]{3,})", text)
    if not m:
        return None, None

    num = int(m.group(1))
    name = m.group(2).strip()
    if name[0].islower():
        name = name.capitalize()

    # Clean avatar prefix and trailing junk.
    name = re.sub(r"^[A-Z]{1,2}\s+(?=[A-Z])", "", name)
    name = re.sub(r"\s+[+#].*$", "", name)
    name = re.sub(r"(\s+[a-z]{1,3})+\s*$", "", name)
    name = re.sub(r"\s+\S$", "", name)
    name = name.rstrip(".")

    if len(name) < 3 or num < 1 or num > 24:
        return None, None
    return num, name
