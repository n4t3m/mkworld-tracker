from __future__ import annotations

import dataclasses
import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Grid geometry (normalised to frame size).
_LEFT_PANEL_RIGHT = 0.47  # right edge of left panel
_ROW_DETECT_LEFT = 0.02   # left edge for row brightness scan
_ROW_BRIGHTNESS_MIN = 40  # mean brightness for an occupied row
_MAX_ROWS = 12            # maximum player rows

# Column boundaries (absolute pixel positions are detected dynamically,
# but these ratios define the two-column split).
_GAP_SEARCH_LEFT = 0.20
_GAP_SEARCH_RIGHT = 0.35

# Cell crop ratios (relative to individual cell width).
_AVATAR_SKIP = 0.17       # skip avatar on left
_SCORE_CROP = 0.75        # keep left 75% of name area to exclude score

_OCR_SCALE = 3
_CELL_PAD_Y = 12          # pixels to skip at top/bottom of cell


@dataclasses.dataclass(frozen=True)
class PlayerInfo:
    name: str


class PlayerReader:
    """Reads player names from the track-selection screen."""

    def read_players(self, frame: np.ndarray) -> list[PlayerInfo]:
        """Return all visible players from the left panel."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rows = self._find_rows(gray, h, w)
        if not rows:
            return []

        left_col, right_col = self._find_columns(gray, rows, h, w)
        if left_col is None:
            return []

        players: list[PlayerInfo] = []
        for ry1, ry2 in rows:
            for cx1, cx2 in [left_col, right_col]:
                if cx1 is None:
                    continue
                info = self._read_cell(frame, gray, ry1, ry2, cx1, cx2)
                if info is not None:
                    players.append(info)

        return players

    # ------------------------------------------------------------------
    # Grid detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_rows(
        gray: np.ndarray, h: int, w: int
    ) -> list[tuple[int, int]]:
        """Find occupied player-box rows via horizontal brightness bands."""
        panel = gray[:, int(w * _ROW_DETECT_LEFT) : int(w * _LEFT_PANEL_RIGHT)]
        row_means = np.mean(panel, axis=1)

        # Collect raw bands.
        raw: list[tuple[int, int]] = []
        in_band = False
        start = 0
        for y in range(int(h * 0.90)):
            if not in_band and row_means[y] > _ROW_BRIGHTNESS_MIN:
                start = y
                in_band = True
            elif in_band and row_means[y] < _ROW_BRIGHTNESS_MIN:
                raw.append((start, y))
                in_band = False

        # Merge bands separated by tiny gaps (< 6px, caused by brightness
        # fluctuations within a single row) then drop fragments shorter than
        # 40px (not real player rows).
        merged: list[tuple[int, int]] = []
        for y1, y2 in raw:
            if merged and y1 - merged[-1][1] < 6:
                merged[-1] = (merged[-1][0], y2)
            else:
                merged.append((y1, y2))

        return [(y1, y2) for y1, y2 in merged if y2 - y1 >= 40][:_MAX_ROWS]

    @staticmethod
    def _find_columns(
        gray: np.ndarray,
        rows: list[tuple[int, int]],
        h: int,
        w: int,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """Return (left_col, right_col) as (x1, x2) tuples."""
        # Average brightness across all rows for a clean profile.
        sample_ys = []
        for y1, y2 in rows:
            margin = (y2 - y1) // 4
            sample_ys.extend(range(y1 + margin, y2 - margin))
        if not sample_ys:
            return None, None

        profile = np.mean(gray[sample_ys, :], axis=0)

        gap_lo = int(w * _GAP_SEARCH_LEFT)
        gap_hi = int(w * _GAP_SEARCH_RIGHT)
        gap_center = int(gap_lo + np.argmin(profile[gap_lo:gap_hi]))

        # Left column edges
        left_start = None
        for x in range(int(w * 0.01), gap_center):
            if profile[x] > _ROW_BRIGHTNESS_MIN:
                left_start = x
                break
        left_end = None
        for x in range(gap_center, int(w * 0.01), -1):
            if profile[x] > _ROW_BRIGHTNESS_MIN:
                left_end = x
                break
        left_col = (left_start, left_end) if left_start and left_end else None

        # Right column: starts after gap, scan right for edge but cap at
        # the left panel boundary.
        right_start = None
        right_limit = int(w * _LEFT_PANEL_RIGHT)
        for x in range(gap_center, right_limit):
            if profile[x] > _ROW_BRIGHTNESS_MIN:
                right_start = x
                break
        right_end = None
        for x in range(right_limit, gap_center, -1):
            if profile[x] > _ROW_BRIGHTNESS_MIN:
                right_end = x
                break
        right_col = (right_start, right_end) if right_start and right_end else None

        return left_col, right_col

    # ------------------------------------------------------------------
    # Per-cell OCR
    # ------------------------------------------------------------------

    def _read_cell(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        ry1: int,
        ry2: int,
        cx1: int,
        cx2: int,
    ) -> PlayerInfo | None:
        box_w = cx2 - cx1
        if box_w < 20:
            return None

        # ---- check occupancy ----
        inner = gray[ry1 + _CELL_PAD_Y : ry2 - _CELL_PAD_Y, cx1 : cx2]
        if inner.size == 0 or float(np.max(inner)) < 80:
            return None  # empty cell

        # ---- name ----
        name_x1 = cx1 + int(box_w * _AVATAR_SKIP)
        # Crop right side to exclude score area
        name_x2 = name_x1 + int((cx2 - name_x1) * _SCORE_CROP)
        name = self._ocr_name(frame, ry1, ry2, name_x1, name_x2)
        name = _clean_name(name)
        if len(name) < 1:
            return None

        return PlayerInfo(name=name)

    @staticmethod
    def _ocr_name(
        frame: np.ndarray, ry1: int, ry2: int, x1: int, x2: int
    ) -> str:
        """OCR a player name cell using V-channel threshold with OTSU fallback."""
        cell = frame[ry1 + _CELL_PAD_Y : ry2 - _CELL_PAD_Y, x1:x2]
        if cell.size == 0:
            return ""

        # Prepare V-channel thresholded image (handles colored backgrounds)
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, v_mask = cv2.threshold(v_channel, 160, 255, cv2.THRESH_BINARY)
        v_up = cv2.resize(
            v_mask, None, fx=_OCR_SCALE, fy=_OCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        # Prepare OTSU thresholded image (handles bright/highlighted cells)
        cell_up = cv2.resize(
            cell, None, fx=_OCR_SCALE, fy=_OCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(cell_up, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Try V-channel then OTSU, PSM 7 (line) then PSM 8 (word)
        for img, psm in [(v_up, 7), (otsu, 7), (v_up, 8), (otsu, 8)]:
            result = pytesseract.image_to_string(
                img, config=f"--psm {psm}"
            ).strip()
            cleaned = _clean_name(result)
            if len(cleaned) >= 2:
                return result
        return ""


def _clean_name(text: str) -> str:
    """Remove avatar-icon noise and box-edge artefacts from OCR text."""
    # --- leading noise (avatar/sticker icon leaking into crop) ---
    # Sticker pattern: optional chars + ">" followed by optional space
    # e.g. "2€> Kod49", "E> choko", ">€> name", "o. RK AciiD", "oo. RK AciiD"
    text = re.sub(r"^.*?>\s*", "", text)
    # Leading "o." / "oo." pattern from sticker icons
    text = re.sub(r"^o{1,2}\.\s*", "", text)
    # One or more non-alnum chars (possibly with commas/spaces) at the start
    text = re.sub(r"^[^a-zA-Z0-9]+[\s,]*", "", text)
    # Single lowercase letter or digit + space before an uppercase letter
    text = re.sub(r"^[a-z0-9]\s+(?=[A-Z])", "", text)
    # Two lowercase letters + comma/space (e.g. "ma Kod49", "e, Kaleb")
    text = re.sub(r"^[a-z]{1,2}[,\s]+", "", text)

    # --- trailing noise (score digits / box border artefacts) ---
    # Trailing " <letters>)" pattern from box edge
    text = re.sub(r"\s+\w{0,2}[)}\]]+$", "", text)
    # Trailing bare digits with optional punctuation (leftover scores: "50.", "60")
    text = re.sub(r"\s+\d+[.\s]*$", "", text)
    # Trailing pipe/bracket
    text = re.sub(r"\s*[|}\]]+$", "", text)
    # Trailing degree symbols, dashes, or OCR artefacts (e.g. "59°", "77—")
    text = re.sub(r"\s+\d*[°—–\-\u00b0\ufffd]+$", "", text)
    # Trailing single non-alnum char (e.g. stray period, tilde)
    text = re.sub(r"\s+[^a-zA-Z0-9]$", "", text)

    return text.strip()
