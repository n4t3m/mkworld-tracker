"""Locate the map's course banner and read it with a trained recognizer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from mktracker.detection.tracks import match_track_name

if TYPE_CHECKING:
    from mktracker.detection.name_model import NameModelReader


def course_banner_crops(frame: np.ndarray) -> list[np.ndarray]:
    """Find white course lettering on its dark map label.

    The selected island moves the banner, so detect the lettering rather than
    assuming one fixed location. The caller must separately confirm the
    player-list panel to distinguish course selection from gameplay overlays.
    """
    h, w = frame.shape[:2]
    scaled = cv2.resize(frame, (1920, 1080)) if (w, h) != (1920, 1080) else frame
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    white[:120] = 0
    white[970:] = 0
    white[:, :1100] = 0
    merged = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not (100 < width < 450 and 15 < height < 70):
            continue
        crop = scaled[max(0, y - 8):min(1080, y + height + 8),
                      max(0, x - 18):min(1920, x + width + 18)]
        if (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 80).mean() < .25:
            continue
        candidates.append((y, x, crop))
    return [crop for _, _, crop in sorted(candidates)]


def read_course_banner_model(frame: np.ndarray, reader: NameModelReader) -> tuple[str, float] | None:
    crops = course_banner_crops(frame)
    if not crops:
        return None
    predictions = reader.predict_crops(crops)
    matches: dict[str, float] = {}
    for text, confidence in predictions:
        if confidence < .65:
            continue
        match = match_track_name(text, cutoff=.85)
        if match is None:
            # The tiny retro console badge sometimes loses a letter while
            # the rest of the title stays exact.
            _, space, rest = text.partition(" ")
            if space:
                match = match_track_name("SNES " + rest, cutoff=1)
                if match is not None and not match[0].startswith("SNES "):
                    match = None
        if match is not None:
            name, ratio = match
            matches[name] = max(matches.get(name, 0.0), ratio * confidence)
    if len(matches) != 1:
        return None
    name, score = next(iter(matches.items()))
    return name, score


def race_start_card_crop(frame: np.ndarray) -> np.ndarray | None:
    """Locate the white course card shown briefly during the race countdown."""
    h, w = frame.shape[:2]
    scaled = cv2.resize(frame, (1920, 1080)) if (w, h) != (1920, 1080) else frame
    hsv = cv2.cvtColor(scaled[:365, :510], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 195)
    caption = float(white[295:355, 85:485].mean())
    left_edge = float(white[100:285, 65:95].mean())
    right_edge = float(white[100:285, 465:500].mean())
    if (caption < .55 or not .12 <= left_edge <= .55 or
            not .12 <= right_edge <= .55):
        return None
    return scaled[292:365, 70:500]


def read_race_start_card_model(frame: np.ndarray, reader: NameModelReader
                               ) -> tuple[str, float] | None:
    """Read a canonical course from the countdown card with the trained model."""
    crop = race_start_card_crop(frame)
    if crop is None:
        return None
    predictions = reader.predict_crops([crop])
    if len(predictions) != 1 or predictions[0][1] < .7:
        return None
    match = match_track_name(predictions[0][0], cutoff=.85)
    if match is None:
        return None
    return match[0], match[1] * predictions[0][1]
