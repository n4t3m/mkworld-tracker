"""One-off test: run gemini_match_results on a single captured PNG.

Runs the request N times (default 5) and reports per-attempt success/failure
so we can verify JSON-repair robustness.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

import cv2

from mktracker.gemini_match_results import request_match_results


def run_once(frame, attempt: int) -> tuple[bool, dict | None, list[tuple[str, int]]]:
    done = threading.Event()
    result_box: dict = {}

    def _cb(parsed, results):
        result_box["parsed"] = parsed
        result_box["results"] = results
        done.set()

    request_match_results(frame, _cb, log_dir=None)

    if not done.wait(timeout=240):
        print(f"[attempt {attempt}] TIMEOUT")
        return False, None, []

    parsed = result_box.get("parsed")
    results = result_box.get("results", [])
    ok = parsed is not None
    print(f"[attempt {attempt}] {'OK' if ok else 'FAIL'}  "
          f"({len(results)} players)" if ok else f"[attempt {attempt}] FAIL")
    return ok, parsed, results


def main(path: str, runs: int = 5) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    frame = cv2.imread(path)
    if frame is None:
        print(f"Failed to read image: {path}")
        return 1

    successes = 0
    last_parsed = None
    last_results: list[tuple[str, int]] = []
    for i in range(1, runs + 1):
        ok, parsed, results = run_once(frame, i)
        if ok:
            successes += 1
            last_parsed = parsed
            last_results = results

    print()
    print(f"=== {successes}/{runs} runs succeeded ===")

    if last_parsed is not None:
        print()
        print("=== Last successful parsed JSON ===")
        print(json.dumps(last_parsed, indent=2, ensure_ascii=False))
        print()
        print("=== Last successful flattened results ===")
        for name, score in last_results:
            print(f"  {name}: {score}")

    return 0 if successes == runs else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_match_results_image <image_path> [runs]")
        sys.exit(1)
    path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sys.exit(main(path, runs))
