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


class TrackSelectDetector:
    """Detects the Mario Kart track-selection screen and reads the track name.

    The track-name banner can appear anywhere on the right-side map (it
    follows the selected island), and the player-list panel on the left is
    not always dark — team-mode matches fill it with bright colored bars,
    and 24-player matches widen the panel past the old 42% boundary. So we
    skip the brightness pre-check and rely on the canonical-track-list fuzzy
    match (with a tight cutoff) as the screen discriminator.

    The state machine guards calls to ``detect()`` by game state and applies
    the cooldown after a successful match.
    """

    def __init__(self) -> None:
        self._last_match_time: float = 0.0

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
        """Full pipeline: cooldown gate, OCR, fuzzy match.

        Returns ``{"track_name": str}`` or ``None``.
        """
        if time.monotonic() - self._last_match_time < _COOLDOWN_SECONDS:
            return None
        result = self.read_track_name(frame)
        if result is None:
            return None
        name, _ratio = result
        return {"track_name": name}
