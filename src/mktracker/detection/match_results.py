"""Detect and read final match results from the CONGRATULATIONS! screen.

After all races are completed, the game displays a CONGRATULATIONS! screen
with final standings showing each player's total points.

Supported layouts:

* **Two Teams, ≤12 players** — two side-by-side columns (red/blue) with
  up to 6 players each.

Readiness is checked via hue-based bar counting (coloured bars must be
fully visible).  Names and scores are then OCR'd from each column using
team-colour-specific binarisation:

* **Red column** — R-channel Otsu, with a hybrid ``(gray>170 & blue>100)``
  override for the gold first-place bar at the top.
* **Blue column** — unsharp-mask on the V channel followed by Otsu.
  Blue-bar text has only ~10 levels of V contrast with the background, so
  name accuracy is limited (scores are more reliable).
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Bar region — vertical extent where player bars appear (below team scores).
_BARS_Y1 = 0.57
_BARS_Y2 = 0.99

# Two-team column boundaries (normalised x).
_LEFT_COL_X1 = 0.005
_LEFT_COL_X2 = 0.485
_RIGHT_COL_X1 = 0.50
_RIGHT_COL_X2 = 0.985

# Within each column, skip the left portion (rank emblem + character icon).
_TEXT_SKIP_X = 0.28

# Score zone — words with normalised x (within text region) above this
# are classified as point totals rather than name fragments.
_SCORE_NX_MIN = 0.60

# Hue ranges for bar detection (OpenCV H is 0-179).
_RED_HUE = (0, 35)   # includes gold first-place bar
_BLUE_HUE = (90, 130)

# Bar detection thresholds.
_BAR_HUE_SAT_MIN = 80       # minimum saturation to count as coloured
_BAR_HUE_COVERAGE_MIN = 0.3  # fraction of row that must match hue
_BAR_MIN_HEIGHT = 40         # pixels — filters team-score box edge

# Gold bar — the top portion of the winning team's column.
_GOLD_TOP_FRACTION = 0.20

# OCR upscale factor.
_UPSCALE = 3


class MatchResultDetector:
    """Detect and read final match results from the CONGRATULATIONS! screen."""

    def detect(
        self,
        frame: np.ndarray,
        *,
        teams: str = "No Teams",
        player_count: int = 12,
    ) -> dict | None:
        """Attempt to read match results from *frame*.

        Returns ``{"results": [(name, points), ...]}`` when the screen
        is fully loaded and all players are readable.  Returns ``None``
        when the screen is not ready (still loading, wrong screen, etc.).
        """
        if teams != "No Teams" and player_count <= 12:
            return self._detect_two_teams(frame, player_count)
        # TODO: <=12 no-teams, >12 no-teams layouts.
        return None

    # ------------------------------------------------------------------
    # Two-team layout (≤12 players)
    # ------------------------------------------------------------------

    def _detect_two_teams(
        self, frame: np.ndarray, player_count: int,
    ) -> dict | None:
        h, w = frame.shape[:2]
        per_team = max(player_count // 2, 1)

        y1, y2 = int(h * _BARS_Y1), int(h * _BARS_Y2)

        left_col = frame[y1:y2, int(w * _LEFT_COL_X1):int(w * _LEFT_COL_X2)]
        right_col = frame[y1:y2, int(w * _RIGHT_COL_X1):int(w * _RIGHT_COL_X2)]

        # Readiness check — require expected number of coloured bars.
        left_bars = self._count_bars(left_col, _RED_HUE)
        right_bars = self._count_bars(right_col, _BLUE_HUE)

        if left_bars < per_team or right_bars < per_team:
            logger.debug(
                "Match results not ready: bars %d/%d (need %d per team)",
                left_bars, right_bars, per_team,
            )
            return None

        # OCR each column with team-specific binarisation.
        left_results = self._read_column(left_col, team="red")
        right_results = self._read_column(right_col, team="blue")

        all_results = left_results[:per_team] + right_results[:per_team]
        return {"results": all_results}

    # ------------------------------------------------------------------
    # Bar detection via hue
    # ------------------------------------------------------------------

    @staticmethod
    def _count_bars(col_img: np.ndarray, hue_range: tuple[int, int]) -> int:
        """Count coloured bars by scanning row-wise hue coverage."""
        hsv = cv2.cvtColor(col_img, cv2.COLOR_BGR2HSV)
        h_lo, h_hi = hue_range
        hue_mask = (
            (hsv[:, :, 0] >= h_lo)
            & (hsv[:, :, 0] <= h_hi)
            & (hsv[:, :, 1] > _BAR_HUE_SAT_MIN)
        )
        row_coverage = hue_mask.mean(axis=1)

        count = 0
        in_bar = False
        start = 0
        for y, cov in enumerate(row_coverage):
            if cov > _BAR_HUE_COVERAGE_MIN and not in_bar:
                start = y
                in_bar = True
            elif cov <= _BAR_HUE_COVERAGE_MIN and in_bar:
                if y - start > _BAR_MIN_HEIGHT:
                    count += 1
                in_bar = False
        if in_bar and len(row_coverage) - start > _BAR_MIN_HEIGHT:
            count += 1
        return count

    # ------------------------------------------------------------------
    # Column OCR
    # ------------------------------------------------------------------

    def _read_column(
        self, col_img: np.ndarray, *, team: str,
    ) -> list[tuple[str, int]]:
        """Read player names and scores from a single column of bars."""
        ch, cw = col_img.shape[:2]

        # Skip icon area on the left.
        text_x1 = int(cw * _TEXT_SKIP_X)
        text_region = col_img[:, text_x1:]
        tw = text_region.shape[1]

        # Team-specific binarisation.
        if team == "red":
            binary = self._binarise_red(text_region)
        else:
            binary = self._binarise_blue(text_region)

        # Upscale for better OCR.
        up = cv2.resize(
            binary, None, fx=_UPSCALE, fy=_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        data = pytesseract.image_to_data(
            up, config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )

        words = self._extract_words(data, tw)
        rows = self._group_into_rows(words)

        results: list[tuple[str, int]] = []
        for row in rows:
            name, score = self._parse_row(row)
            if name and score is not None:
                results.append((name, score))

        return results

    # ------------------------------------------------------------------
    # Binarisation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _binarise_red(text_region: np.ndarray) -> np.ndarray:
        """R-channel Otsu for red bars, hybrid threshold for gold bar."""
        r_ch = text_region[:, :, 2]  # Red channel in BGR
        r_blur = cv2.GaussianBlur(r_ch, (3, 3), 0)
        _, binary = cv2.threshold(
            r_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # The gold first-place bar sits in the top portion.  R-channel
        # Otsu turns the entire gold bar white; replace with the hybrid
        # threshold that cleanly isolates white text from the gold bg.
        gold_h = int(text_region.shape[0] * _GOLD_TOP_FRACTION)
        top = text_region[:gold_h]
        gray_top = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        blue_top = top[:, :, 0]
        binary[:gold_h] = (
            (gray_top > 170) & (blue_top > 100)
        ).astype(np.uint8) * 255

        return binary

    @staticmethod
    def _binarise_blue(text_region: np.ndarray) -> np.ndarray:
        """Unsharp-mask on V channel for blue bars.

        Blue-bar text has only ~10 levels of V-channel contrast with the
        background.  An unsharp mask amplifies local brightness differences
        before Otsu thresholding.
        """
        v = cv2.cvtColor(text_region, cv2.COLOR_BGR2HSV)[:, :, 2]
        blurred = cv2.GaussianBlur(v, (21, 21), 0)
        sharp = cv2.addWeighted(
            v.astype(np.float32), 3.0,
            blurred.astype(np.float32), -2.0, 0,
        )
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(
            sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary

    # ------------------------------------------------------------------
    # Word extraction and row parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_words(data: dict, text_width: int) -> list[dict]:
        """Extract OCR words with normalised positions."""
        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 15:
                continue
            ox = data["left"][i] / _UPSCALE
            ow = data["width"][i] / _UPSCALE
            oy = data["top"][i] / _UPSCALE
            oh = data["height"][i] / _UPSCALE
            cx = (ox + ow / 2) / text_width
            words.append({
                "text": text,
                "nx": cx,
                "y": oy + oh / 2,
                "conf": conf,
            })
        return words

    @staticmethod
    def _group_into_rows(words: list[dict]) -> list[list[dict]]:
        """Group words into rows by y-proximity."""
        if not words:
            return []
        words.sort(key=lambda w: w["y"])
        rows: list[list[dict]] = [[words[0]]]
        for w in words[1:]:
            if abs(w["y"] - rows[-1][-1]["y"]) > 20:
                rows.append([w])
            else:
                rows[-1].append(w)
        return rows

    @staticmethod
    def _parse_row(
        row_words: list[dict],
    ) -> tuple[str | None, int | None]:
        """Return ``(name, score)`` from a row of OCR words."""
        row_words.sort(key=lambda w: w["nx"])
        name_parts: list[str] = []
        score: int | None = None

        for w in row_words:
            text = w["text"]
            nx = w["nx"]

            # Score zone (right side of bar).
            if nx > _SCORE_NX_MIN:
                clean = re.sub(r"[^0-9]", "", text)
                if clean:
                    try:
                        score = int(clean)
                    except ValueError:
                        pass
                continue

            # Name zone.
            clean = re.sub(r"[^A-Za-z0-9\s'\-=._]", "", text).strip()
            if clean and not (len(clean) == 1 and clean.isdigit()):
                name_parts.append(clean)

        name = " ".join(name_parts) if name_parts else None
        return name, score
