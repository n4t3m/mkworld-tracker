"""Local, conservative shape matching for the single-player race-position HUD.

Templates include the ordinal suffix, so ordinary numbers in menus/results
are not accepted. No network calls or writes are made while reading frames.
"""
from pathlib import Path

import cv2
import numpy as np


def position_masks(frame: np.ndarray, *, faded: bool = False) -> list[np.ndarray]:
    """Extract colored HUD silhouettes at a resolution independent scale."""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    roi = frame[int(h * .75):h, int(w * .82):w]
    roi = cv2.resize(roi, (346, 270), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = (hue >= 80) & (hue <= 115)
    warm = (hue <= 30) | (hue >= 170)
    if faded:
        # Alpha blending both desaturates the fill and changes its hue. On
        # bright backgrounds the translucent outline can join numeral and
        # suffix; brighter thresholds separate the fill from that outline.
        colors = [((hue >= 80) & (hue <= 150), 10, 40),
                  ((hue >= 70) & (hue <= 150), 15, 40),
                  ((hue >= 70) & (hue <= 105), 40, 40)]
        colors += [(color, 40, brightness)
                   for color in (blue, warm) for brightness in (150, 180, 200)]
    else:
        colors = [((hue >= 80) & (hue <= 103) & (sat > 150), 90, 100),
                  (blue, 90, 100),
                  ((hue <= 18) | (hue >= 170), 90, 100), (hue <= 30, 90, 100)]
    results = []
    for color, saturation, brightness in colors:
        mask = (color & (sat > saturation) & (val > brightness)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        digits = [i for i in range(1, n)
                  if 90 <= stats[i, 3] <= 220 and 18 <= stats[i, 2] <= 260
                  and stats[i, 4] >= 500 and stats[i, 1] >= 15
                  and 195 <= stats[i, 1] + stats[i, 3] <= 240
                  and stats[i, 0] > 0 and stats[i, 0] + stats[i, 2] < 345]
        if not digits:
            continue
        right = max(stats[i, 0] + stats[i, 2] for i in digits)
        bottom = max(stats[i, 1] + stats[i, 3] for i in digits)
        suffix = [i for i in range(1, n) if i not in digits
                  and 23 <= stats[i, 3] <= 85 and stats[i, 4] >= 400
                  and stats[i, 2] <= 100 and stats[i, 0] + stats[i, 2] < 345
                  and abs(stats[i, 1] + stats[i, 3] - bottom) <= 15
                  and right - 70 <= stats[i, 0] <= right + 55
                  and bottom - 90 <= stats[i, 1] <= bottom + 10]
        if not suffix:
            continue
        selected = digits + suffix
        silhouette = np.isin(labels, selected).astype(np.uint8) * 255
        ys, xs = np.where(silhouette)
        # Tiny compression bridges can attach a background speck to a digit.
        # Ink quantiles prevent that speck from changing the normalized scale.
        top, bottom = np.quantile(ys, [.015, .995]).astype(int)
        left, right = np.quantile(xs, [.005, .995]).astype(int)
        cut = silhouette[top:bottom+1, left:right+1]
        # Preserve aspect ratio: stretching would confuse 1st and 11th.
        scale = min(120 / cut.shape[1], 100 / cut.shape[0])
        cut = cv2.resize(cut, (max(1, round(cut.shape[1]*scale)),
                               max(1, round(cut.shape[0]*scale))))
        canvas = np.zeros((108, 128), np.uint8)
        canvas[4:4+cut.shape[0], 4:4+cut.shape[1]] = cut
        results.append(canvas)
    return results


class HudPositionReader:
    """Read 1st–12th in a single-player HUD; reject missing/ambiguous shapes.

    These templates cover the 12-player custom-room recordings. Other layouts
    and larger rooms need additional reference footage before being supported.
    """

    def __init__(self) -> None:
        self.templates = []
        for path in sorted((Path(__file__).parent / "position_templates").glob("*.png")):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.templates.append((int(path.stem.split("-")[0]), mask > 127))

    def read(self, frame: np.ndarray) -> int | None:
        rank = self._match(position_masks(frame), minimum=.76, margin=.065)
        if rank is not None:
            return rank
        # This is new evidence from the current frame, never a held-over rank.
        # Broader color masks require stronger shape agreement and separation.
        return self._match(position_masks(frame, faded=True), minimum=.78, margin=.10)

    def _match(self, masks: list[np.ndarray], *, minimum: float, margin: float) -> int | None:
        scores = {}
        for mask in masks:
            mask = mask > 127
            for rank, template in self.templates:
                union = np.count_nonzero(mask | template)
                score = np.count_nonzero(mask & template) / max(1, union)
                scores[rank] = max(score, scores.get(rank, 0))
        ranked = sorted(scores, key=scores.get, reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if scores[best] < minimum or (len(ranked) > 1 and scores[best] - scores[ranked[1]] < margin):
            return None
        return best
