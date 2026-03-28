from __future__ import annotations

import dataclasses
import difflib
import logging
import re

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

# Centre of the screen must be bright (the white settings card).
_CENTER_ROI = (0.20, 0.25, 0.80, 0.85)
_CENTER_BRIGHTNESS_MIN = 150

# "The rules have been decided!" banner at screen bottom.
_BANNER_ROI = (0.20, 0.94, 0.80, 0.98)

# Full settings area inside the card (labels + values).
_SETTINGS_ROI = (0.14, 0.34, 0.82, 0.86)

_OCR_SCALE = 2
_MATCH_CUTOFF = 0.5

# Valid values for each setting.
_VALID_CLASSES = ("50cc", "100cc", "150cc")
_VALID_TEAMS = ("No Teams", "Two Teams", "Three Teams", "Four Teams")
_VALID_ITEMS = ("Normal", "Frantic", "Custom Items", "Mushrooms Only")
_VALID_COM = ("No COM", "Easy", "Normal", "Hard")
_VALID_INTERMISSION = ("10 seconds", "One minute")

# Labels to look for in OCR text, mapped to the dataclass field name.
_SETTING_LABELS: tuple[tuple[str, str], ...] = (
    ("Class", "cc_class"),
    ("Teams", "teams"),
    ("tems", "items"),  # Tesseract often reads "Items" as "ltems"
    ("COM", "com_difficulty"),
    ("Race Count", "race_count"),
    ("Intermission", "intermission"),
)


@dataclasses.dataclass(frozen=True)
class MatchSettings:
    cc_class: str
    teams: str
    items: str
    com_difficulty: str
    race_count: int
    intermission: str


def _fuzzy(text: str, options: tuple[str, ...]) -> str | None:
    matches = difflib.get_close_matches(text, options, n=1, cutoff=_MATCH_CUTOFF)
    return matches[0] if matches else None


class MatchSettingsDetector:
    """Detects the 'rules decided' screen and extracts match settings."""

    def is_active(self, frame: np.ndarray) -> bool:
        """Fast brightness check — is the white settings card visible?"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _CENTER_ROI
        center = frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) > _CENTER_BRIGHTNESS_MIN

    def _check_banner(self, frame: np.ndarray) -> bool:
        """OCR the bottom banner and look for 'rules'/'decided'."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _BANNER_ROI
        roi = frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]
        roi_up = cv2.resize(
            roi, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 7").strip().lower()
        return "rules" in text or "decided" in text

    def _read_settings(self, frame: np.ndarray) -> MatchSettings | None:
        """OCR the settings card and parse the six setting values."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _SETTINGS_ROI
        roi = frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]
        roi_up = cv2.resize(
            roi, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        text: str = pytesseract.image_to_string(thresh, config="--psm 6").strip()

        return self._parse_settings_text(text)

    @staticmethod
    def _parse_settings_text(text: str) -> MatchSettings | None:
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        raw: dict[str, str] = {}
        for line in lines:
            lower = line.lower()
            for label, field in _SETTING_LABELS:
                if label.lower() in lower:
                    idx = lower.index(label.lower())
                    value = line[idx + len(label) :].strip()
                    raw[field] = value
                    break

        if len(raw) < 6:
            logger.debug("Only parsed %d/6 settings from OCR text", len(raw))
            return None

        cc_class = _fuzzy(raw.get("cc_class", ""), _VALID_CLASSES)
        teams = _fuzzy(raw.get("teams", ""), _VALID_TEAMS)
        items = _fuzzy(raw.get("items", ""), _VALID_ITEMS)
        com_difficulty = _fuzzy(raw.get("com_difficulty", ""), _VALID_COM)
        intermission = _fuzzy(raw.get("intermission", ""), _VALID_INTERMISSION)

        rc_match = re.search(r"\d+", raw.get("race_count", ""))
        race_count = int(rc_match.group()) if rc_match else None

        if not all([cc_class, teams, items, com_difficulty, race_count, intermission]):
            logger.debug(
                "Some settings failed validation: class=%s teams=%s items=%s "
                "com=%s races=%s intermission=%s",
                cc_class, teams, items, com_difficulty, race_count, intermission,
            )
            return None

        return MatchSettings(
            cc_class=cc_class,
            teams=teams,
            items=items,
            com_difficulty=com_difficulty,
            race_count=race_count,
            intermission=intermission,
        )

    def detect(self, frame: np.ndarray) -> MatchSettings | None:
        """Full pipeline: brightness check → banner OCR → settings OCR."""
        if not self.is_active(frame):
            return None
        if not self._check_banner(frame):
            return None
        return self._read_settings(frame)
