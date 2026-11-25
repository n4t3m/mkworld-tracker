"""Compare independent result frames and expose unresolved disagreements."""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import unicodedata
from mktracker.detection.font_symbols import canonical_font_text


def name_key(name: str) -> str:
    return "".join(c for c in canonical_font_text(name).casefold()
                   if not c.isspace() and unicodedata.category(c) != "Cc")


def similar_names(a: str, b: str) -> bool:
    if a == b:
        return True
    # Numeric IDs are identities, not OCR decoration (Player1 != Player2).
    if "".join(c for c in a if c.isdecimal()) != "".join(c for c in b if c.isdecimal()):
        return False
    return min(len(a), len(b)) >= 6 and SequenceMatcher(None, a, b).ratio() >= .82


@dataclass
class _Candidate:
    text: str
    weight: float = 0
    frames: set = field(default_factory=set)
    best_confidence: float = -1
    uncertain_frames: set = field(default_factory=set)


class PlacementConsensus:
    def __init__(self, roster: list[str] | None = None) -> None:
        self._votes: dict[int, dict[str, _Candidate]] = {}
        self._seen_frames: set = set()
        self._sequence = 0
        self._cooccurring: set[frozenset[str]] = set()
        self._roster = list(roster or [])

    def _preview_match(self, key: str) -> str | None:
        numeric = "".join(c for c in key if c.isdecimal())
        scores = sorted([(SequenceMatcher(None, key, name_key(n)).ratio(), n) for n in self._roster
                         if "".join(c for c in name_key(n) if c.isdecimal()) == numeric], reverse=True)
        if not scores or scores[0][0] < .82:
            return None
        if len(scores) > 1 and scores[0][0] - scores[1][0] < .1:
            return None
        return scores[0][1]

    def _can_alias(self, a: str, b: str) -> bool:
        return similar_names(a, b) and frozenset((a, b)) not in self._cooccurring

    def add(self, result: dict, *, frame_id=None) -> None:
        if result.get("type", "race") != "race":
            return
        if frame_id is None:
            self._sequence += 1
            frame_id = ("observation", self._sequence)
        if frame_id in self._seen_frames:
            return
        self._seen_frames.add(frame_id)
        rows = result.get("results", [])
        ranks = [r for r, _ in rows]
        keys = [name_key(n) for _, n in rows]
        for i, key in enumerate(keys):
            for other in keys[i + 1:]:
                if key and other and key != other:
                    self._cooccurring.add(frozenset((key, other)))
        completeness = min(1.0, len(rows) / 12)
        for rank, name in rows:
            key = name_key(name)
            if not 1 <= rank <= 24 or len(key) < 2 or ranks.count(rank) != 1 or keys.count(key) != 1:
                continue
            confidence = max(0, min(100, result.get("confidences", {}).get(rank, 50)))
            if confidence < 25:
                continue
            weight = (confidence / 100) ** 2 * (.5 + .5 * completeness)
            # Preview OCR is supporting evidence, not ground truth. Give an
            # exact roster spelling a modest boost; never invent its text in
            # the result table or force an ambiguous/unseen player to match.
            match = self._preview_match(key)
            if match and name_key(match) == key:
                weight *= 1.2
            candidate = self._votes.setdefault(rank, {}).setdefault(key, _Candidate(name))
            candidate.weight += weight
            candidate.frames.add(frame_id)
            if result.get("glyph_reviews", {}).get(rank, {}).get("needs_review"):
                candidate.uncertain_frames.add(frame_id)
            if confidence > candidate.best_confidence:
                candidate.text, candidate.best_confidence = name, confidence

    def _ranked(self):
        ranked = {}
        for rank, candidates in self._votes.items():
            # Compare close spellings to the seed, avoiding fuzzy-name chains.
            remaining = dict(candidates)
            groups = []
            while remaining:
                seed = max(remaining, key=lambda k: remaining[k].weight)
                group = {k: v for k, v in remaining.items() if self._can_alias(seed, k)}
                for key in group:
                    del remaining[key]
                winner = max(group, key=lambda k: (group[k].weight, len(group[k].frames), group[k].best_confidence))
                frames = set().union(*(c.frames for c in group.values()))
                groups.append((sum(c.weight for c in group.values()), winner, group[winner], frames))
            groups.sort(key=lambda g: g[0], reverse=True)
            ranked[rank] = (groups, sum(c.weight for c in candidates.values()))
        return ranked

    def _resolve(self):
        selected = {}
        used_names = []
        ranked = self._ranked()
        # Resolve strong rows first. Never assign one identity to two places.
        edges = sorted(((g[0], rank, g) for rank, (groups, _) in ranked.items() for g in groups),
                       reverse=True, key=lambda e: e[0])
        for score, rank, group in edges:
            if rank in selected or any(self._can_alias(group[1], key) for key in used_names):
                continue
            selected[rank] = group
            used_names.append(group[1])
        return ranked, selected

    def placements(self) -> list[list]:
        _, selected = self._resolve()
        return [[rank, group[2].text] for rank, group in sorted(selected.items())]

    def evidence(self) -> dict[str, dict]:
        ranked, selected = self._resolve()
        evidence = {}
        for rank, (groups, total) in ranked.items():
            group = selected.get(rank)
            if group is None:
                evidence[str(rank)] = {"status": "conflict", "frames": 0, "agreement": 0}
                continue
            score, key, candidate, frames = group
            agreement = score / total if total else 0
            spelling_agreement = candidate.weight / score if score else 0
            confirmed = len(frames) >= 2 and agreement >= .65 and spelling_agreement >= .6 and group is groups[0]
            preview_match = self._preview_match(key)
            if self._roster and (not preview_match or name_key(preview_match) != key):
                confirmed = False
            glyph_unresolved = bool(candidate.uncertain_frames) and len(candidate.frames-candidate.uncertain_frames) < 2
            if glyph_unresolved:
                confirmed = False
            evidence[str(rank)] = {
                "status": "confirmed" if confirmed else "review",
                "frames": len(frames), "agreement": round(agreement, 3),
                "spelling_agreement": round(spelling_agreement, 3),
                "alternatives": [g[2].text for g in groups if g is not group],
                "preview_checked": bool(self._roster), "preview_match": preview_match,
                "glyph_review": glyph_unresolved,
            }
        return evidence
