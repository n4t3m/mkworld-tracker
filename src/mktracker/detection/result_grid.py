"""Rank positions from independently read result-row digits."""
from __future__ import annotations

from collections import Counter


def rank_origin(readings: list[tuple[int, int | None]], *, min_support: int = 1
                ) -> tuple[int, int] | None:
    """Return the first rank and anchor count for a sequential row grid."""
    anchors = Counter(rank - index for index, rank in readings if rank is not None)
    if not anchors:
        return None
    first, support = anchors.most_common(1)[0]
    return (first, support) if support >= min_support else None
