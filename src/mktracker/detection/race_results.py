"""Detect and read race result placements from the post-race results screen.

After a race ends, the game displays each player's placement with a ``+N``
point differential.  When the ``+`` disappears the view transitions to
overall standings.  This module detects which phase is shown and reads
placement numbers and player names from the race-result rows.

Two preprocessing paths are supported:

* **No-teams mode** — grey semi-transparent bars with white text.  A hybrid
  threshold ``(grayscale > 170) & (blue > 100)`` isolates the text while
  filtering the gold first-place bar.

* **Teams mode** — red / blue opaque bars where the text takes on the bar
  colour.  A Gaussian-blur + Otsu on the HSV *V* channel is used for names,
  and an adaptive-threshold cluster check on a narrow strip detects ``+``.
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# OCR region — right side of screen where result bars appear.
_ROI_X1 = 0.56
_ROI_X2 = 0.98

# X-position zones (normalised to full frame width).
_PLACEMENT_X_MAX = 0.62  # placement numbers live left of here
_ICON_X_MIN = 0.62       # character icon area (skip)
_ICON_X_MAX = 0.66
_NAME_X_MAX = 0.82       # player names live between ICON_X_MAX and here
_SCORE_X_MIN = 0.82      # +N differentials and totals are right of here

# Minimum detected name-rows to consider the frame as showing overall results.
_MIN_ROWS_FOR_OVERALL = 5

# Plus-column strip for team-mode cluster detection.
_PLUS_STRIP_X1 = 0.84
_PLUS_STRIP_X2 = 0.90
_PLUS_MIN_CLUSTERS = 3


class RaceResultDetector:
    """Detects race-result rows and reads placement / name data."""

    def detect(
        self,
        frame: np.ndarray,
        *,
        teams: bool = False,
    ) -> dict | None:
        """Analyse *frame* for race results.

        Parameters
        ----------
        frame:
            BGR video frame.
        teams:
            When ``True`` use the team-colour preprocessing path
            (red/blue bars instead of grey).

        Returns
        -------
        dict or None
            ``{"type": "race", "results": [(placement, name), ...]}``
            when race results with ``+`` differentials are visible.

            ``{"type": "overall"}`` when overall standings are shown
            (result bars without any ``+`` sign).

            ``None`` when no result rows are detected.
        """
        h, w = frame.shape[:2]
        x1 = int(w * _ROI_X1)
        x2 = int(w * _ROI_X2)
        roi = frame[:, x1:x2]

        if teams:
            binary = self._binarise_teams(roi)
            has_plus = self._has_plus_clusters(frame)
        else:
            binary = self._binarise_default(roi)
            has_plus = None  # determined from OCR below

        ocr_input = cv2.resize(binary, None, fx=2, fy=2,
                               interpolation=cv2.INTER_CUBIC)

        data = pytesseract.image_to_data(
            ocr_input, config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )

        words = self._extract_words(data, x1, w)
        rows = self._group_into_rows(words)

        parsed_rows: list[tuple[int, int | None, str]] = []

        for row_words in rows:
            y_center, placement, name, row_has_plus = self._parse_row(
                row_words,
            )
            # In default mode, derive has_plus from the OCR output.
            if has_plus is None and row_has_plus:
                has_plus = True
            if name:
                parsed_rows.append((y_center, placement, name))

        if has_plus is None:
            has_plus = False

        if not parsed_rows:
            return None

        results = self._fix_placements(parsed_rows)

        if has_plus and len(results) >= 3:
            return {"type": "race", "results": results}

        if len(results) >= _MIN_ROWS_FOR_OVERALL:
            return {"type": "overall"}

        return None

    # ------------------------------------------------------------------
    # Binarisation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _binarise_default(roi: np.ndarray) -> np.ndarray:
        """Hybrid threshold for grey / gold bars (no-teams mode)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blue = roi[:, :, 0]
        return ((gray > 170) & (blue > 100)).astype(np.uint8) * 255

    @staticmethod
    def _binarise_teams(roi: np.ndarray) -> np.ndarray:
        """Blur + Otsu on V channel for coloured team bars.

        The gold first-place bar is the same regardless of team mode, so
        the top portion also gets the default hybrid threshold OR'd in.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_blur = cv2.GaussianBlur(v, (5, 5), 0)
        _, binary = cv2.threshold(
            v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # The gold first-place bar sits in roughly the top 10% of the ROI.
        # The V-channel Otsu turns the entire gold bar white (V ≈ 240 for
        # both text and background), making text unreadable.  Replace the
        # top portion with the hybrid threshold which cleanly isolates
        # the white text from the gold background.
        gold_h = int(roi.shape[0] * 0.10)
        top = roi[:gold_h]
        gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        blue_top = top[:, :, 0]
        binary[:gold_h] = ((gray_top > 170) & (blue_top > 100)).astype(
            np.uint8,
        ) * 255

        return binary

    @staticmethod
    def _has_plus_clusters(frame: np.ndarray) -> bool:
        """Detect ``+N`` text in the plus column via adaptive-V clusters.

        Works for both team and non-team frames.  The ``+`` column is a
        narrow vertical strip; each ``+N`` creates a bright cluster in
        the adaptive-threshold output.  Three or more clusters indicate
        race results.
        """
        h, w = frame.shape[:2]
        strip = frame[:, int(w * _PLUS_STRIP_X1):int(w * _PLUS_STRIP_X2)]
        v = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)[:, :, 2]
        binary = cv2.adaptiveThreshold(
            v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -15,
        )
        bright_per_row = binary.mean(axis=1)

        clusters = 0
        in_cluster = False
        cluster_start = 0
        for y in range(h):
            if bright_per_row[y] > 30 and not in_cluster:
                cluster_start = y
                in_cluster = True
            elif bright_per_row[y] <= 30 and in_cluster:
                if y - cluster_start >= 8:
                    clusters += 1
                in_cluster = False
        if in_cluster and h - cluster_start >= 8:
            clusters += 1

        return clusters >= _PLUS_MIN_CLUSTERS

    # ------------------------------------------------------------------
    # Word extraction and row parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_words(
        data: dict,
        x_offset: int,
        frame_w: int,
    ) -> list[dict]:
        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 20:
                continue
            ox = data["left"][i] // 2 + x_offset
            oy = data["top"][i] // 2
            oh = data["height"][i] // 2
            words.append({
                "text": text,
                "x": ox,
                "nx": ox / frame_w,
                "y": oy + oh // 2,
                "conf": conf,
            })
        return words

    @staticmethod
    def _group_into_rows(words: list[dict]) -> list[list[dict]]:
        if not words:
            return []
        words.sort(key=lambda w: w["y"])
        rows: list[list[dict]] = [[words[0]]]
        for w in words[1:]:
            if abs(w["y"] - rows[-1][-1]["y"]) > 25:
                rows.append([w])
            else:
                rows[-1].append(w)
        return rows

    @staticmethod
    def _parse_row(
        row_words: list[dict],
    ) -> tuple[int, int | None, str | None, bool]:
        """Return ``(y_center, placement, name, has_plus)``."""
        row_words.sort(key=lambda w: w["x"])
        placement: int | None = None
        name_parts: list[str] = []
        has_plus = False
        y_center = sum(w["y"] for w in row_words) // len(row_words)

        for w in row_words:
            text = w["text"]
            nx = w["nx"]

            # Score / plus area
            if nx >= _SCORE_X_MIN:
                if re.match(r"^[+#]\d+$", text):
                    has_plus = True
                continue

            # Icon area — skip
            if _ICON_X_MIN <= nx < _ICON_X_MAX:
                continue

            # Placement number area
            if nx < _PLACEMENT_X_MAX:
                clean = re.sub(r"[^0-9]", "", text)
                if clean and placement is None:
                    try:
                        num = int(clean)
                        if 1 <= num <= 24:
                            placement = num
                    except ValueError:
                        pass
                continue

            # Name area (ICON_X_MAX .. NAME_X_MAX)
            clean = re.sub(r"[^A-Za-z0-9\s'\-.]", "", text).strip()
            # Skip all-digit artefacts and very short noise
            if not clean or len(clean) < 2 or clean.isdigit():
                continue
            name_parts.append(clean)

        name = " ".join(name_parts) if name_parts else None
        return y_center, placement, name, has_plus

    # ------------------------------------------------------------------
    # Placement correction
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_placements(
        parsed_rows: list[tuple[int, int | None, str]],
    ) -> list[tuple[int, str]]:
        """Infer missing placements using sequential numbering.

        Result rows are always consecutive (e.g. 1-13, 8-20), but OCR may
        miss some rows entirely, leaving gaps in the detected list.  This
        method uses the y-position spacing to detect those gaps and assign
        correct placement numbers even when rows are missing.
        """
        parsed_rows.sort(key=lambda r: r[0])
        n = len(parsed_rows)
        if n < 2:
            return [(p if p else 0, name) for _, p, name in parsed_rows]

        # ---- estimate row spacing from the y-positions ------------------
        gaps = [
            parsed_rows[i + 1][0] - parsed_rows[i][0] for i in range(n - 1)
        ]
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        if median_gap <= 0:
            median_gap = 1

        # ---- convert y-positions to row indices -------------------------
        y0 = parsed_rows[0][0]
        row_indices = [round((r[0] - y0) / median_gap) for r in parsed_rows]

        # ---- find the best anchor placement -----------------------------
        best_first: int | None = None
        best_score = -1

        for i, (_, p, _) in enumerate(parsed_rows):
            if p is None:
                continue
            candidate = p - row_indices[i]
            if candidate < 1:
                continue
            score = sum(
                1
                for j, (_, pp, _) in enumerate(parsed_rows)
                if pp is not None and pp == candidate + row_indices[j]
            )
            if score > best_score:
                best_score = score
                best_first = candidate

        if best_first is not None and best_score >= 2:
            return [
                (best_first + row_indices[i], name)
                for i, (_, _, name) in enumerate(parsed_rows)
                if 1 <= best_first + row_indices[i] <= 24
            ]

        # Fallback — keep whatever placements were detected.
        return [(p if p else 0, name) for _, p, name in parsed_rows]
