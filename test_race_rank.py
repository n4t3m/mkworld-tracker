"""Test RaceRankDetector against all images in race_rank_examples/.

Each filename encodes the expected result:
  - Starts with a number (e.g. "12.png", "7_ex2.png")
    -> expected: rank IS visible (crop returned)
  - Contains "no_placement_displayed" or "none_displayed"
    -> expected: no rank visible (None returned)
  - Contains "partial" -> skipped
  - Anything else -> skipped

Run:
    python test_race_rank.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mktracker.detection import *  # noqa: F401,F403 — triggers Tesseract setup
from mktracker.detection.race_rank import RaceRankDetector

EXAMPLES_DIR = Path(__file__).parent / "race_rank_examples"

_RANK_RE = re.compile(r"^(\d{1,2})")
_NO_RANK_RE = re.compile(r"no_placement_displayed|none_displayed", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"partial", re.IGNORECASE)


def expected_for(path: Path) -> str:
    """Return 'rank', 'none', or 'skip'."""
    name = path.stem
    if _PARTIAL_RE.search(name):
        return "skip"
    if _NO_RANK_RE.search(name):
        return "none"
    m = _RANK_RE.match(name)
    if m:
        return "rank"
    return "skip"


def main() -> None:
    detector = RaceRankDetector()

    images = sorted(EXAMPLES_DIR.glob("*.png"))

    passed = 0
    failed = 0
    skipped = 0

    for img_path in images:
        exp = expected_for(img_path)
        if exp == "skip":
            print(f"  SKIP  {img_path.name}")
            skipped += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ERROR cannot read {img_path.name}")
            failed += 1
            continue

        crop = detector.detect(frame)
        got_rank = crop is not None

        if exp == "rank":
            ok = got_rank
            exp_str = "visible"
            got_str = f"visible ({crop.shape[1]}x{crop.shape[0]})" if got_rank else "NOT visible"
        else:  # exp == "none"
            ok = not got_rank
            exp_str = "NOT visible"
            got_str = f"visible ({crop.shape[1]}x{crop.shape[0]})" if got_rank else "NOT visible"

        status = "  PASS" if ok else "  FAIL"
        print(f"{status}  {img_path.name:<45}  expected={exp_str:<14}  got={got_str}")

        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} passed  ({skipped} skipped)")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
