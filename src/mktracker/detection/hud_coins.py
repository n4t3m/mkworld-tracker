"""Local coin-counter reading for the single-player custom-room HUD."""
from pathlib import Path

import cv2
import numpy as np


def coin_regions(frame: np.ndarray) -> list[np.ndarray]:
    """Locate compact gold coin icons within the lower-left HUD search area.

    Coordinates are relative to each detected icon, not the recording border.
    The icon's diameter supplies scale, accommodating padding and capture size.
    """
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    if h < 40 or w < 40:
        return []
    # Work at a bounded resolution; do not assume black bars are present.
    area = frame[round(h * .7):, :round(w * .35)]
    area = cv2.resize(area, (672, 324), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (16, 130, 140), (39, 255, 255))
    n, _, stats, _ = cv2.connectedComponentsWithStats(gold)
    regions = []
    for x, y, width, height, ink in stats[1:]:
        if not (20 <= height <= 65 and .75 <= width / height <= 1.2
                and .55 <= ink / (width * height) <= .95):
            continue
        scale = height / 40
        left = x + width / 2 - 41 * scale
        top = y + height / 2 - 34 * scale
        if left < 0 or top < 0 or left + 150 * scale > 672 or top + 70 * scale > 324:
            continue
        transform = np.float32([[scale, 0, left], [0, scale, top]])
        regions.append(cv2.warpAffine(area, transform, (150, 70),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    return regions


def _region_digits(roi: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # The maximum count uses yellow numerals, including on neutral HUD panels.
    white = (((sat < 85) | ((hue > 15) & (hue < 40) & (sat > 130)))
             & (val > 190)).astype(np.uint8) * 255
    digits = []
    for left, right in [(68, 106), (106, 145)]:
        mask = white[:65, left:right]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = [i for i in range(1, n)
                      if stats[i, 3] >= 25 and stats[i, 4] >= 180]
        if len(candidates) != 1:
            return []
        x, y, width, height, area = stats[candidates[0]]
        if not (10 <= width <= 35 and 28 <= height <= 55):
            return []
        cut = (labels[y:y+height, x:x+width] == candidates[0]).astype(np.uint8) * 255
        digits.append(cv2.resize(cut, (28, 40), interpolation=cv2.INTER_NEAREST))
    return digits


def coin_digits(frame: np.ndarray) -> list[np.ndarray]:
    """Extract a single unambiguous icon-anchored digit pair."""
    pairs = [digits for roi in coin_regions(frame) if (digits := _region_digits(roi))]
    return pairs[0] if len(pairs) == 1 else []


class HudCoinReader:
    """Read visible 00–20 counters; return None for missing or ambiguous HUDs.

    Reference footage covers colored/neutral single-player custom-room HUDs.
    Each frame is independent, so seeking never retains a stale count.
    """

    def __init__(self) -> None:
        self.templates = [(int(p.stem), cv2.imread(str(p), 0) > 127)
                          for p in sorted((Path(__file__).parent / "coin_templates").glob("*.png"))]

    def read(self, frame: np.ndarray) -> int | None:
        masks = coin_digits(frame)
        if len(masks) != 2:
            return None
        values = []
        for mask in masks:
            ink = mask > 127
            scores = sorted([(np.count_nonzero(ink & template) /
                              max(1, np.count_nonzero(ink | template)), value)
                             for value, template in self.templates], reverse=True)
            if len(scores) < 2 or scores[0][0] < .80 or scores[0][0] - scores[1][0] < .08:
                return None
            values.append(scores[0][1])
        count = values[0] * 10 + values[1]
        return count if count <= 20 else None
