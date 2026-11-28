"""Read a live finish rank with the trained text model and HUD shape evidence."""
from __future__ import annotations

import re

import numpy as np

from mktracker.detection.hud_position import HudPositionReader
from mktracker.detection.race_rank import _find_rank_region


_RANK_TEXT = re.compile(r"^([1-9]|1[0-9]|2[0-4])(?:[a-z]{0,3})?$", re.I)


def read_finish_rank(frame: np.ndarray, reader, verifier: HudPositionReader) -> int | None:
    """Accept the model when it agrees with a visible rank-shaped HUD.

    The existing shape reader handles a missing or confused model digit. If
    the HUD itself is not visible, leave the rank for the existing API path.
    """
    visible_rank = verifier.read(frame)
    if visible_rank is None:
        return None
    if reader is None or reader.disabled:
        return visible_rank
    region = _find_rank_region(frame)
    if region is None:
        return visible_rank
    y1, y2, x1, x2 = region
    reads = reader.predict_crops([frame[y1:y2, x1:x2]])
    if reads:
        text, confidence = reads[0]
        match = _RANK_TEXT.fullmatch(text.strip())
        if match and confidence >= .75 and int(match.group(1)) == visible_rank:
            return int(match.group(1))
    return visible_rank
