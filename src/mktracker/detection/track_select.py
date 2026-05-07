from __future__ import annotations

import difflib
import logging
import time

import cv2
import numpy as np
import pytesseract

from mktracker.detection.tracks import TRACK_NAMES

logger = logging.getLogger(__name__)

# OCR the right ~60% of the frame at scale. The track-name banner moves with
# the selected island/cursor, so we can't pin it to a fixed sub-rectangle —
# instead we sparse-OCR a wide region and fuzzy-match every line.
_RIGHT_REGION_X = 0.40

# Scale factor applied before OCR (small text needs upscaling).
_OCR_SCALE = 2

# Threshold for white banner text. The banner is white on a dark translucent
# rounded rectangle; values >=200 pull just the glyphs.
_TEXT_THRESHOLD = 200

# PSM modes tried in order. 11 = sparse text (good for finding a stray banner
# in scenery), 6 = uniform block, 7 = single line.
_PSM_MODES = (11, 6, 7)

# Minimum similarity (0-1) for fuzzy matching against the track list. Set
# tight: real banner OCR consistently returns ratio >=0.95 across all
# fixtures, while incidental text on results screens (character names, etc.)
# tops out around 0.8.
_MATCH_CUTOFF = 0.85

# Seconds to skip detection after a confirmed match.
_COOLDOWN_SECONDS = 15.0

# Player-list-panel detector: we sample a vertical strip well inside the left
# panel, mean-collapse to a 1D row-brightness profile, detrend with a gaussian
# blur, and look at the autocorrelation peak in the lag range covering one
# pill row (~50-180 px on a 1080p frame). The proper track-select screen has
# a clearly periodic pill pattern; voting screens (full-screen world map) and
# in-race overlays (3D scenery) do not. Real fixtures land at 0.43-0.92 across
# all layouts (12p/24p, no-team, team-mode); voting and gameplay frames stay
# under 0.20. 0.30 sits comfortably in the gap.
_PANEL_STRIP_X1 = 0.05
_PANEL_STRIP_X2 = 0.30
_PANEL_DETREND_KSIZE = 51
_PANEL_PERIOD_MIN_LAG = 40
_PANEL_PERIOD_MAX_LAG = 180
_PANEL_AUTOCORR_MIN = 0.30


class TrackSelectDetector:
    """Detects the Mario Kart track-selection screen and reads the track name.

    The track-name banner can appear anywhere on the right-side map (it
    follows the selected island), so OCR sparse-scans the right ~60% and
    fuzzy-matches every line against the canonical track list with a tight
    cutoff. To reject look-alike screens whose text *also* fuzzy-matches a
    canonical name (the voting screen has 3-5 candidate banners; the in-race
    track-name overlay has one), ``has_player_panel`` requires a periodic
    pill-row pattern in the left panel area as a screen pre-check.

    The state machine guards calls to ``detect()`` by game state and applies
    the cooldown after a successful match.
    """

    def __init__(self) -> None:
        self._last_match_time: float = 0.0

    @staticmethod
    def has_player_panel(frame: np.ndarray) -> bool:
        """Return True iff the left side has the periodic pill-row pattern.

        Discriminates the proper track-select screen (which always has the
        player-list panel on the left) from the voting screen (full-screen
        world map, no panel) and from in-race gameplay frames where the
        track-name overlay briefly appears (3D scenery, no panel).
        """
        h, w = frame.shape[:2]
        strip = frame[:, int(w * _PANEL_STRIP_X1) : int(w * _PANEL_STRIP_X2)]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        profile = gray.mean(axis=1)

        # Detrend so we measure the high-freq pill rhythm, not slow lighting
        # gradients across the panel.
        smoothed = cv2.GaussianBlur(
            profile.reshape(-1, 1), (1, _PANEL_DETREND_KSIZE), 0
        ).flatten()
        detrended = profile - smoothed
        detrended -= detrended.mean()

        # Autocorrelation via FFT, normalised to [0, 1] at lag 0.
        n = len(detrended)
        if n < _PANEL_PERIOD_MAX_LAG + 1:
            return False
        spec = np.fft.fft(detrended, n=2 * n)
        acorr = np.fft.ifft(spec * np.conj(spec)).real[:n]
        if acorr[0] <= 0:
            return False
        acorr /= acorr[0]

        peak = float(
            acorr[_PANEL_PERIOD_MIN_LAG : _PANEL_PERIOD_MAX_LAG + 1].max()
        )
        return peak >= _PANEL_AUTOCORR_MIN

    def read_track_name(self, frame: np.ndarray) -> tuple[str, float] | None:
        """OCR the track-name banner anywhere in the right region.

        Returns ``(canonical_name, similarity)`` or ``None``. Tries multiple
        PSM modes and joins adjacent fragments because Tesseract sometimes
        splits the banner across two lines.
        """
        h, w = frame.shape[:2]
        roi = frame[:, int(w * _RIGHT_REGION_X) :]
        roi_up = cv2.resize(
            roi, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, _TEXT_THRESHOLD, 255, cv2.THRESH_BINARY)

        candidates: list[str] = []
        for psm in _PSM_MODES:
            text = pytesseract.image_to_string(thresh, config=f"--psm {psm}")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    candidates.append(line)

        # Banner sometimes splits across two lines (e.g. "Wario S" + "hipyard");
        # try adjacent pairs joined both with and without a space.
        base = list(candidates)
        for i in range(len(base) - 1):
            candidates.append((base[i] + " " + base[i + 1]).strip())
            candidates.append((base[i] + base[i + 1]).strip())

        best_name: str | None = None
        best_ratio = 0.0
        for cand in candidates:
            matches = difflib.get_close_matches(
                cand, TRACK_NAMES, n=1, cutoff=_MATCH_CUTOFF
            )
            if not matches:
                continue
            ratio = difflib.SequenceMatcher(
                None, cand.lower(), matches[0].lower()
            ).ratio()
            if ratio > best_ratio:
                best_name, best_ratio = matches[0], ratio

        if best_name is None:
            return None
        return best_name, best_ratio

    def detect(self, frame: np.ndarray) -> dict | None:
        """Full pipeline: cooldown gate, panel pre-check, OCR, fuzzy match.

        Returns ``{"track_name": str}`` or ``None``.
        """
        if time.monotonic() - self._last_match_time < _COOLDOWN_SECONDS:
            return None
        if not self.has_player_panel(frame):
            return None
        result = self.read_track_name(frame)
        if result is None:
            return None
        name, _ratio = result
        return {"track_name": name}
