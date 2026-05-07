"""Run TrackSelectDetector.detect() on every captured frame in matches/.

Reports per-bucket detection counts and flags suspicious cases:
  * track.png / players.png that fail to detect (potential false negative).
  * Any other captured frame that *does* detect (potential false positive).
  * track.png / players.png whose detected name disagrees with match.json.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import cv2

from mktracker.detection.track_select import TrackSelectDetector


def _bucket_for(path: str, match_dir: str) -> tuple[str, int | None]:
    rel = os.path.relpath(path, match_dir).replace("\\", "/")
    parts = rel.split("/")
    if len(parts) == 1:
        return parts[0], None
    race_num = (
        int(parts[0].split("_", 1)[1]) if parts[0].startswith("race_") else None
    )
    if len(parts) == 2:
        leaf = parts[1]
        if leaf.startswith("placement_"):
            return "placement_NN.png", race_num
        return leaf, race_num
    # debug subfolders.
    sub, leaf = parts[1], parts[2]
    if sub == "debug_placements":
        prefix = leaf.split("_", 1)[0]  # pre / post
        return f"debug_placements/{prefix}_NN.png", race_num
    if sub == "debug_votes":
        return "debug_votes/vote_NN.png", race_num
    return f"{sub}/{leaf}", race_num


def main() -> None:
    matches_root = "matches"
    buckets: dict[str, list[tuple[str, str | None, str | None]]] = defaultdict(list)

    for match_id in sorted(os.listdir(matches_root)):
        match_dir = os.path.join(matches_root, match_id)
        if not os.path.isdir(match_dir):
            continue
        expected_by_race: dict[int, str | None] = {}
        json_path = os.path.join(match_dir, "match.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                for r in data.get("races", []):
                    expected_by_race[r["race_number"]] = r.get("track_name")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: bad json in {match_id}: {e}")

        for png in glob.glob(os.path.join(match_dir, "**", "*.png"), recursive=True):
            bucket, race_num = _bucket_for(png, match_dir)
            frame = cv2.imread(png)
            if frame is None:
                continue
            # Fresh detector each time to bypass the 15s cooldown.
            result = TrackSelectDetector().detect(frame)
            detected = result["track_name"] if result else None
            expected = expected_by_race.get(race_num) if race_num is not None else None
            buckets[bucket].append((png, detected, expected))

    total = sum(len(v) for v in buckets.values())
    print(f"\n=== {total} frames analyzed ===\n")

    expected_positive_buckets = {"track.png", "players.png"}
    suspect_positives: list[tuple[str, str]] = []
    suspect_negatives: list[tuple[str, str]] = []
    track_mismatches: list[tuple[str, str | None, str | None]] = []

    for bucket in sorted(buckets):
        rows = buckets[bucket]
        det = sum(1 for _, d, _ in rows if d is not None)
        n = len(rows)
        print(f"  {bucket:38s} {det:5d}/{n:5d} detected")

        for path, detected, expected in rows:
            if bucket in expected_positive_buckets:
                if detected is None:
                    suspect_negatives.append((path, expected or "?"))
                elif expected and detected != expected:
                    track_mismatches.append((path, detected, expected))
            else:
                if detected is not None:
                    # debug_votes are mostly voting-screen frames but may
                    # include the final track-select frame at the end of the
                    # buffer — it's expected to *sometimes* detect there.
                    suspect_positives.append((path, detected))

    print(
        f"\nSuspect positives (non-track non-players frames that detected): "
        f"{len(suspect_positives)}"
    )
    for path, detected in suspect_positives[:50]:
        print(f"  {path}  ->  {detected}")
    if len(suspect_positives) > 50:
        print(f"  ... and {len(suspect_positives) - 50} more")

    print(
        f"\nSuspect negatives (track.png/players.png that did NOT detect): "
        f"{len(suspect_negatives)}"
    )
    for path, expected in suspect_negatives[:50]:
        print(f"  {path}  (expected {expected!r})")
    if len(suspect_negatives) > 50:
        print(f"  ... and {len(suspect_negatives) - 50} more")

    print(
        f"\nTrack mismatches (detected != match.json track_name): "
        f"{len(track_mismatches)}"
    )
    for path, detected, expected in track_mismatches[:50]:
        print(f"  {path}  detected={detected!r}  expected={expected!r}")
    if len(track_mismatches) > 50:
        print(f"  ... and {len(track_mismatches) - 50} more")


if __name__ == "__main__":
    main()
