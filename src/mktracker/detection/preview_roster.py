"""Read name cells on the course-preview roster for later cross-checking."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
import cv2
import numpy as np

from mktracker.detection.team_results import _binary, _read_sheet
from mktracker.detection.name_consensus import name_key, similar_names

if TYPE_CHECKING:
    from mktracker.detection.name_model import NameModelReader


def read_preview_names(frame: np.ndarray, reader: NameModelReader | None = None,
                       *, return_slots: bool = False):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    colored = (((hue < 12) | (hue > 170) | ((hue > 95) & (hue < 125))) & (sat > 120) & (val > 95)).astype(np.uint8) * 255
    boxes = []
    for left, right in [(.025, .308), (.308, .59)]:
        x0, x2 = int(w * left), int(w * right)
        mask = cv2.morphologyEx(colored[:, x0:x2], cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if w * .15 < bw < w * .31 and h * .035 < bh < h * .09 and y + bh < h * .93:
                boxes.append((y, x + x0, bw, bh))
    if len(boxes) < 3:
        return ([], 0) if return_slots else []
    cells = []
    model_crops = []
    for y, x, bw, bh in sorted(boxes):
        pad = max(2, int(bh * .12))
        crop = frame[y + pad:y + bh - pad, x + int(bw * .20):x + int(bw * .85)]
        color = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Voice-chat overlays can cover a cell with white text on black.
        # Do not turn the speaker label into a supposed roster member.
        if ((color[:, :, 2] < 45) & (color[:, :, 1] < 90)).mean() > .08:
            continue
        # A voice-chat overlay can leave most of the colored bar visible but
        # cover the name with a dark rectangle. The model otherwise reads the
        # overlay label as a player name.
        if reader is not None and (color[:, :, 2] < 65).mean() > .10:
            continue
        model_crops.append(crop)
        if reader is not None:
            continue
        cell = _binary(crop)
        scale = min(800 / cell.shape[1], 120 / cell.shape[0])
        cell = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        top = (160 - cell.shape[0]) // 2
        cells.append(cv2.copyMakeBorder(cell, top, 160 - cell.shape[0] - top,
                                       10, 810 - cell.shape[1], cv2.BORDER_CONSTANT, value=255))
    if not model_crops:
        return ([], len(boxes)) if return_slots else []
    if reader is not None:
        reads = reader.predict_crops(model_crops)
        names = [(name.strip().strip("<>"), confidence * 100)
                 for name, confidence in reads
                 if len(name_key(name)) >= 3 and confidence >= .6]
    else:
        names = [(name.strip(), confidence)
                 for name, confidence in _read_sheet(np.vstack(cells), len(cells), deskew=False)
                 if len(name_key(name)) >= 3 and confidence >= 40]
    return (names, len(boxes)) if return_slots else names


class PreviewRoster:
    def __init__(self, reader: NameModelReader | None = None) -> None:
        self._reader = reader
        self._reads = defaultdict(dict)
        self._seen = set()
        self._cooccurring = set()
        self._slot_counts: list[int] = []

    def add(self, frame: np.ndarray) -> None:
        key = hash(frame.tobytes())
        if key in self._seen:
            return
        self._seen.add(key)
        reads, slots = read_preview_names(frame, self._reader, return_slots=True)
        self._slot_counts.append(slots)
        keys = [name_key(n) for n, _ in reads]
        for i, name in enumerate(keys):
            for other in keys[i + 1:]:
                if name != other:
                    self._cooccurring.add(frozenset((name, other)))
        for name, confidence in reads:
            self._reads[name_key(name)][key] = (name, confidence)

    def complete(self) -> bool:
        """Stop model reads once two 12-slot pages establish the full roster."""
        return (self._reader is not None and not self._reader.disabled
                and len(self._slot_counts) >= 2
                and self._slot_counts[-2:] == [12, 12]
                and len(self.names()) >= 12)

    def names(self) -> list[str]:
        remaining = dict(self._reads)
        names = []
        while remaining:
            seed = max(remaining, key=lambda k: sum(c for n, c in remaining[k].values()))
            group = {k: reads for k, reads in remaining.items()
                     if similar_names(seed, k) and frozenset((seed, k)) not in self._cooccurring}
            frames = set().union(*(set(reads) for reads in group.values()))
            best = max(group, key=lambda k: sum(c for n, c in group[k].values()))
            if len(frames) >= 2:
                names.append(max(group[best].values(), key=lambda r: r[1])[0])
            for key in group:
                del remaining[key]
        return names
