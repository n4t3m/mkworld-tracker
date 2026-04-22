"""Detect and read final match results from the CONGRATULATIONS! screen.

After all races are completed, the game displays a CONGRATULATIONS! screen
with final standings showing each player's total points.

Supported layouts:

* **Two Teams, ≤12 players** — two side-by-side columns (red/blue) with
  up to 6 players each.

* **No Teams, ≤12 players** — single left-side column with up to 12 players.
  Bars are grey (low saturation, mid brightness); the gold first-place bar
  is at the top.

Readiness is checked via hue-based bar counting (coloured bars must be
fully visible).  Names and scores are then OCR'd from each column using
team-colour-specific binarisation:

* **Red column** — R-channel Otsu, with a hybrid ``(gray>170 & blue>100)``
  override for the gold first-place bar at the top.
* **Blue column** — unsharp-mask on the V channel followed by Otsu.
  Blue-bar text has only ~10 levels of V contrast with the background, so
  name accuracy is limited (scores are more reliable).
* **No-teams grey bars** — hybrid ``(gray>170 & blue>100)`` threshold,
  which cleanly isolates white text on both grey and gold bars.
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

# No-teams ≤12 layout — single column occupies the left portion of the screen.
_NT_COL_X1 = 0.00
_NT_COL_X2 = 0.56
_NT_BARS_Y1 = 0.18   # below CONGRATULATIONS!/NICE TRY! banner
_NT_BARS_Y2 = 0.97

# No-teams >12 layout — two columns side by side.
# Left column: rank+icon+name+score for players 1–12.
# Right column: rank+icon+name+score (or just score) for players 13–24.
_NT24_LEFT_X1 = 0.00
_NT24_LEFT_X2 = 0.48
_NT24_RIGHT_X1 = 0.50
_NT24_RIGHT_X2 = 0.97
_NT24_BARS_Y1 = 0.18
_NT24_BARS_Y2 = 0.97
# Icon skip for >12 columns — the rank badge + avatar icon occupies a larger
# fraction of the narrower column than in the ≤12 single-column layout.
_NT24_TEXT_SKIP_X = 0.45

# Banner readiness check — CONGRATULATIONS! / NICE TRY! banner at the top.
# Real result screens have BOTH yellow text AND a red/orange diagonal stripe.
# Gameplay frames may have incidental yellow (golden tracks, UI badges) but
# never the red stripe.  Both conditions must be met.
_NT_BANNER_Y2 = 0.18        # banner occupies top 18% of frame
_NT_BANNER_X2 = 0.50        # yellow text is always on the left half
_NT_BANNER_H_LO = 20        # yellow hue range (OpenCV H 0-179)
_NT_BANNER_H_HI = 35
_NT_BANNER_SAT_MIN = 150    # yellow must be saturated
_NT_BANNER_VAL_MIN = 150    # yellow must be bright
_NT_BANNER_RATIO_MIN = 0.09  # minimum yellow pixel fraction
# Red/orange diagonal stripe behind the banner text.
_NT_BANNER_RED_SAT_MIN = 120
_NT_BANNER_RED_VAL_MIN = 80
_NT_BANNER_RED_RATIO_MIN = 0.30  # real screens: ~0.40; gameplay: <0.22

# Two-team banner readiness check.  In team mode the banner background takes
# the *winning* team's colour (red or blue) — and is dark grey for DRAW! —
# so the no-teams "yellow text + red stripe" signature does not apply.
# Instead we look at the team score boxes that sit just below the banner:
# a saturated red panel on the left half and a saturated blue panel on the
# right half.  These appear together with the banner regardless of which
# team won (or even on a draw), which makes them a stable readiness signal.
_TT_SCORES_Y1 = 0.15
_TT_SCORES_Y2 = 0.50
_TT_SCORES_SAT_MIN = 120
_TT_SCORES_VAL_MIN = 80
_TT_RED_LEFT_RATIO_MIN = 0.10   # real screens: ~0.17–0.20; gameplay: <0.10
_TT_BLUE_RIGHT_RATIO_MIN = 0.05  # real screens: ~0.10–0.13; gameplay: <0.02

# Two-team banner stripe — the top ~10% of the screen is a solid coloured
# banner on real result screens (red, blue, or dark grey for DRAW).  On
# mid-race overall-standings screens, the red/blue team-score panels are
# also visible but the top strip shows gameplay scenery (sky, track), so
# no single banner colour dominates.
_TT_BANNER_Y2 = 0.10
_TT_BANNER_SAT_MIN = 120
_TT_BANNER_VAL_MIN = 80
_TT_BANNER_GRAY_SAT_MAX = 60
_TT_BANNER_GRAY_VAL_LO = 30
_TT_BANNER_GRAY_VAL_HI = 100
_TT_BANNER_COVERAGE_MIN = 0.40   # real screens: >=0.63; mid-race: <0.25


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
        if teams == "No Teams" and player_count <= 12:
            return self._detect_no_teams(frame, player_count)
        if teams == "No Teams" and player_count <= 24:
            return self._detect_no_teams_24(frame, player_count)
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
    # No-teams layout (≤12 players, single left column)
    # ------------------------------------------------------------------

    def _detect_no_teams(
        self, frame: np.ndarray, player_count: int,
    ) -> dict | None:
        h, w = frame.shape[:2]

        # Readiness check — yellow CONGRATULATIONS!/NICE TRY! banner at top.
        if not self._has_result_banner(frame):
            logger.debug("No-teams match results not ready: banner not detected")
            return None

        y1 = int(h * _NT_BARS_Y1)
        y2 = int(h * _NT_BARS_Y2)
        x1 = int(w * _NT_COL_X1)
        x2 = int(w * _NT_COL_X2)

        col = frame[y1:y2, x1:x2]
        results = self._read_no_teams_column(col)
        return {"results": results}

    # ------------------------------------------------------------------
    # No-teams layout (>12 players, two columns)
    # ------------------------------------------------------------------

    def _detect_no_teams_24(
        self, frame: np.ndarray, player_count: int,
    ) -> dict | None:
        h, w = frame.shape[:2]

        if not self._has_result_banner(frame):
            logger.debug("No-teams 24p match results not ready: banner not detected")
            return None

        y1 = int(h * _NT24_BARS_Y1)
        y2 = int(h * _NT24_BARS_Y2)

        left_col = frame[y1:y2, int(w * _NT24_LEFT_X1):int(w * _NT24_LEFT_X2)]
        right_col = frame[y1:y2, int(w * _NT24_RIGHT_X1):int(w * _NT24_RIGHT_X2)]

        left_results = self._read_no_teams_column(left_col, text_skip=_NT24_TEXT_SKIP_X)
        right_results = self._read_no_teams_column(right_col, text_skip=_NT24_TEXT_SKIP_X)

        # Left column holds players 1–(player_count//2), right holds the rest.
        per_col = player_count // 2
        all_results = left_results[:per_col] + right_results[:per_col]
        return {"results": all_results}

    def _read_no_teams_column(
        self, col_img: np.ndarray, *, text_skip: float = _TEXT_SKIP_X,
    ) -> list[tuple[str, int]]:
        """Read player names and scores from the no-teams single column."""
        ch, cw = col_img.shape[:2]

        # Skip rank emblem + avatar icon on the left.
        text_x1 = int(cw * text_skip)
        text_region = col_img[:, text_x1:]
        tw = text_region.shape[1]

        binary = self._binarise_no_teams(text_region)

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

    @staticmethod
    def _has_result_banner(
        frame: np.ndarray, *, teams: str = "No Teams",
    ) -> bool:
        """Return True if the CONGRATULATIONS!/NICE TRY!/DRAW! banner is
        present.

        In ``No Teams`` mode the banner has yellow text on a red/orange
        diagonal stripe.  In ``Two Teams`` mode the banner background
        instead takes the *winning* team's colour (red or blue), or is
        dark grey on a draw, with white text — so the no-teams signature
        does not match.  For team mode we delegate to the team-specific
        check that looks at the red/blue team-score panels below the banner.
        """
        if teams == "Two Teams":
            return MatchResultDetector._has_two_team_result_banner(frame)
        h, w = frame.shape[:2]
        banner = frame[:int(h * _NT_BANNER_Y2), :int(w * _NT_BANNER_X2)]
        hsv = cv2.cvtColor(banner, cv2.COLOR_BGR2HSV)
        yellow_mask = (
            (hsv[:, :, 0] >= _NT_BANNER_H_LO)
            & (hsv[:, :, 0] <= _NT_BANNER_H_HI)
            & (hsv[:, :, 1] > _NT_BANNER_SAT_MIN)
            & (hsv[:, :, 2] > _NT_BANNER_VAL_MIN)
        )
        if float(yellow_mask.mean()) < _NT_BANNER_RATIO_MIN:
            return False
        red_mask = (
            ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 160))
            & (hsv[:, :, 1] > _NT_BANNER_RED_SAT_MIN)
            & (hsv[:, :, 2] > _NT_BANNER_RED_VAL_MIN)
        )
        return float(red_mask.mean()) >= _NT_BANNER_RED_RATIO_MIN

    @staticmethod
    def _has_two_team_result_banner(frame: np.ndarray) -> bool:
        """Return True if the two-team CONGRATULATIONS!/NICE TRY!/DRAW!
        screen is showing.

        The banner colour itself varies (red/blue for the winning team,
        dark grey for a draw), so we detect it in two parts:

        1. The giant team-score panels just below the banner — a
           saturated red panel on the left and blue panel on the right.
        2. A solid banner stripe occupying the top ~10% of the screen,
           which must be dominated by a single colour (red, blue, or
           dark grey).  This rejects mid-race overall-standings screens,
           which still show the red/blue score panels but leave gameplay
           scenery visible at the top.
        """
        h, w = frame.shape[:2]
        region = frame[int(h * _TT_SCORES_Y1):int(h * _TT_SCORES_Y2), :]
        rh, rw = region.shape[:2]
        left = region[:, : rw // 2]
        right = region[:, rw // 2:]
        hsv_l = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        hsv_r = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
        red_mask = (
            ((hsv_l[:, :, 0] < 15) | (hsv_l[:, :, 0] > 160))
            & (hsv_l[:, :, 1] > _TT_SCORES_SAT_MIN)
            & (hsv_l[:, :, 2] > _TT_SCORES_VAL_MIN)
        )
        blue_mask = (
            (hsv_r[:, :, 0] >= 90) & (hsv_r[:, :, 0] <= 130)
            & (hsv_r[:, :, 1] > _TT_SCORES_SAT_MIN)
            & (hsv_r[:, :, 2] > _TT_SCORES_VAL_MIN)
        )
        if (
            float(red_mask.mean()) < _TT_RED_LEFT_RATIO_MIN
            or float(blue_mask.mean()) < _TT_BLUE_RIGHT_RATIO_MIN
        ):
            return False

        top = frame[:int(h * _TT_BANNER_Y2), :]
        hsv_t = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        gray_t = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        top_red = (
            ((hsv_t[:, :, 0] < 15) | (hsv_t[:, :, 0] > 160))
            & (hsv_t[:, :, 1] > _TT_BANNER_SAT_MIN)
            & (hsv_t[:, :, 2] > _TT_BANNER_VAL_MIN)
        )
        top_blue = (
            (hsv_t[:, :, 0] >= 90) & (hsv_t[:, :, 0] <= 130)
            & (hsv_t[:, :, 1] > _TT_BANNER_SAT_MIN)
            & (hsv_t[:, :, 2] > _TT_BANNER_VAL_MIN)
        )
        top_gray = (
            (hsv_t[:, :, 1] < _TT_BANNER_GRAY_SAT_MAX)
            & (gray_t > _TT_BANNER_GRAY_VAL_LO)
            & (gray_t < _TT_BANNER_GRAY_VAL_HI)
        )
        dominant = max(
            float(top_red.mean()),
            float(top_blue.mean()),
            float(top_gray.mean()),
        )
        return dominant >= _TT_BANNER_COVERAGE_MIN

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

    @staticmethod
    def _binarise_no_teams(text_region: np.ndarray) -> np.ndarray:
        """Hybrid threshold for white text on grey and gold no-teams bars.

        ``(gray > 170) & (blue > 100)`` isolates white text from both the
        grey player bars and the gold first-place bar without needing a
        separate gold-override pass.
        """
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        blue = text_region[:, :, 0]
        binary = ((gray > 170) & (blue > 100)).astype(np.uint8) * 255
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
