"""Backfill ``vote.png`` for legacy matches by scanning ``debug_votes/``.

For every ``matches/<id>/race_NN/`` folder that has a ``debug_votes/``
directory but no ``vote.png``, this picks the oldest ``vote_NN.png`` whose
frame contains the yellow "The course has been selected!" banner and
copies it to ``vote.png`` next to ``track.png``.  Skips folders that
already have a ``vote.png`` unless ``--force`` is passed.

Usage:
    python -m scripts.backfill_vote_frames [--matches-dir matches]
                                           [--force]
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

import cv2

from mktracker.detection.vote_banner import VoteBannerDetector
from mktracker.match_record import DEFAULT_MATCHES_DIR

logger = logging.getLogger(__name__)

_VOTE_FRAME_RE = re.compile(r"^vote_(\d+)\.png$")


def _sorted_vote_frames(votes_dir: Path) -> list[Path]:
    frames: list[tuple[int, Path]] = []
    for p in votes_dir.iterdir():
        m = _VOTE_FRAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]


def _backfill_race(race_dir: Path, detector: VoteBannerDetector, *, force: bool) -> str:
    """Return one of: ``saved``, ``skipped-exists``, ``skipped-no-debug``,
    ``skipped-no-banner``, or ``error``."""
    target = race_dir / "vote.png"
    if target.exists() and not force:
        return "skipped-exists"
    votes_dir = race_dir / "debug_votes"
    if not votes_dir.is_dir():
        return "skipped-no-debug"

    for path in _sorted_vote_frames(votes_dir):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if detector.is_active(frame):
            try:
                shutil.copyfile(path, target)
            except OSError:
                logger.exception("Failed to copy %s -> %s", path, target)
                return "error"
            logger.info("[saved] %s -> vote.png (from %s)", race_dir, path.name)
            return "saved"
    return "skipped-no-banner"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-dir", type=Path, default=DEFAULT_MATCHES_DIR)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite vote.png even if it already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.matches_dir.exists():
        logger.error("Directory not found: %s", args.matches_dir)
        return

    detector = VoteBannerDetector()
    counts: dict[str, int] = {}

    for match_dir in sorted(d for d in args.matches_dir.iterdir() if d.is_dir()):
        for race_dir in sorted(d for d in match_dir.iterdir() if d.is_dir()):
            if not re.match(r"^race_\d+$", race_dir.name):
                continue
            outcome = _backfill_race(race_dir, detector, force=args.force)
            counts[outcome] = counts.get(outcome, 0) + 1

    logger.info("Done. %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
