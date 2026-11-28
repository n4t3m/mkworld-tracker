"""Reread saved Match History race frames with verified local OCR."""
from __future__ import annotations

from collections import Counter
import unicodedata

import numpy as np

from mktracker.detection.match_results_model import reconcile_final_rows
from mktracker.detection.name_consensus import PlacementConsensus
from mktracker.detection.name_model import NameModelReader
from mktracker.detection.race_results import RaceResultDetector
from mktracker.team_scoring import TEAM_COLOR_ORDER, local_race_teams


def saved_race_model_available(teams: str, roster: list[str]) -> bool:
    """Only layouts verified on complete local race boards use this route."""
    if len(roster) != len(set(roster)):
        return False
    if teams == "Two Teams" and len(roster) == 12:
        return NameModelReader.color_available()
    if teams in TEAM_COLOR_ORDER and len(roster) == 24:
        return NameModelReader.color_available()
    if teams == "No Teams" and len(roster) == 24:
        return NameModelReader.available()
    return False


def _verified_placements(votes: PlacementConsensus, roster: list[str]):
    count = len(roster)
    placements = [(int(rank), name) for rank, name in votes.placements()]
    if [rank for rank, _ in placements] != list(range(1, count + 1)):
        return None
    evidence = votes.evidence()
    if any(evidence.get(str(rank), {}).get("frames", 0) < 2
           for rank in range(1, count + 1)):
        return None
    rows = [(0, rank, name, 0) for rank, name in placements]
    reconciled = reconcile_final_rows(rows, roster, min_similarity=.75)

    def digits(value: str) -> str:
        return "".join(char for char in unicodedata.normalize("NFKC", value)
                       if char.isdecimal())

    if (any(digits(old[2]) and digits(new[2]) and
            digits(old[2]) != digits(new[2])
            for old, new in zip(rows, reconciled)) or
            {name for _, _, name, _ in reconciled} != set(roster)):
        return None
    return [(rank, name) for _, rank, name, _ in reconciled]


def read_saved_race_results(frames: list[np.ndarray], *, teams: str,
                            roster: list[str]) -> dict | None:
    """Return Gemini-shaped local results only for a fully verified board."""
    if not frames or not saved_race_model_available(teams, roster):
        return None
    reader = NameModelReader(color=teams != "No Teams")
    detector = RaceResultDetector()
    votes = PlacementConsensus(roster)
    try:
        if teams == "No Teams":
            candidates = []
            for frame_id, frame in enumerate(frames):
                grid = detector.detect(frame, teams=True, model_only=True,
                                       dark_model=reader, rank_only=True)
                if grid and grid.get("type") == "race":
                    candidates.append((frame_id, frame, grid))
            covered = set().union(*(set(grid["row_centers"])
                                    for _, _, grid in candidates))
            if not set(range(1, 25)) <= covered:
                return None
            remaining = set(range(1, 25))
            pending = list(candidates)
            selected = []
            while remaining:
                best = max(pending, key=lambda item:
                           len(set(item[2]["row_centers"]) & remaining))
                selected.append(best)
                pending.remove(best)
                remaining -= set(best[2]["row_centers"])
            for frame_id, frame, grid in selected + pending:
                votes.add(reader.read(frame, grid), frame_id=frame_id)
                placements = _verified_placements(votes, roster)
                if placements is not None:
                    return {
                        "mode": "no_teams",
                        "teams": [{
                            "name": None, "tag": None,
                            "race_points": None, "race_winner": None,
                            "players": [{"place": rank, "name": name}
                                        for rank, name in placements],
                        }],
                    }
        else:
            color_votes: dict[int, Counter[str]] = {}
            if len(roster) == 24:
                from mktracker.detection.team_results import _bands

                candidates = []
                for frame_id, frame in enumerate(frames):
                    if len(_bands(frame)) < 3:
                        continue
                    grid = detector.detect(frame, teams=True, model_only=True,
                                           color_model=reader, rank_only=True)
                    if grid and grid.get("type") == "race":
                        candidates.append((frame_id, frame, grid))
                remaining = {rank: 2 for rank in range(1, 25)}
                selected = []
                pending = list(candidates)
                while pending and any(remaining.values()):
                    best = max(pending, key=lambda item:
                               sum(remaining.get(rank, 0) > 0
                                   for rank in item[2]["row_centers"]))
                    if not any(remaining.get(rank, 0) > 0
                               for rank in best[2]["row_centers"]):
                        break
                    selected.append(best)
                    pending.remove(best)
                    for rank in best[2]["row_centers"]:
                        if rank in remaining:
                            remaining[rank] = max(0, remaining[rank] - 1)
                if any(remaining.values()):
                    return None
                ordered = selected + pending
            else:
                ordered = [(frame_id, frame, None)
                           for frame_id, frame in enumerate(frames)]
            for frame_id, frame, grid in ordered:
                if grid is None:
                    grid = detector.detect(frame, teams=True, model_only=True,
                                           color_model=reader)
                    if not grid or grid.get("type") != "race":
                        continue
                result = reader.read(frame, grid)
                votes.add(result, frame_id=frame_id)
                for rank, color in result.get("team_colors", {}).items():
                    if color in TEAM_COLOR_ORDER[teams]:
                        color_votes.setdefault(int(rank), Counter())[color] += 1
                placements = _verified_placements(votes, roster)
                if placements is None:
                    continue
                colors = {}
                for rank, counts in color_votes.items():
                    ranked = counts.most_common()
                    if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                        colors[rank] = ranked[0][0]
                structured = local_race_teams(placements, colors, teams)
                if structured is not None:
                    return structured
        return None
    finally:
        reader.close()
