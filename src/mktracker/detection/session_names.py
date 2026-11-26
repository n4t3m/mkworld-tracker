"""Use repeated course previews to resolve noisy result-name spellings."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import unicodedata


def _key(text: str) -> str:
    # These are common OCR renderings of the same small clan-tag glyphs.
    text = text.casefold().translate(str.maketrans({"γ": "y", "ł": "l", "ı": "i", "²": ""}))
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _roster(races) -> list[tuple[str, str, int]]:
    counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    seen: dict[str, set[int]] = {}
    for race_index, race in enumerate(races):
        for name in set(race.preview_players):
            counts[name] += 1
            seen.setdefault(name, set()).add(race_index)
        result_counts.update(set(name for _, name in race.placements))
    groups: list[tuple[str, list[tuple[str, int]]]] = []
    for name, count in counts.most_common():
        if count < 2 or len(_key(name)) < 5:
            continue
        key = _key(name)
        matches = [i for i, (seed, _) in enumerate(groups) if _similar(key, seed) >= .82]
        if len(matches) == 1:
            groups[matches[0]][1].append((name, count))
        elif not matches:
            groups.append((key, [(name, count)]))
    roster = []
    for key, variants in groups:
        support = len(set().union(*(seen[name] for name, _ in variants)))
        if support < 4:
            continue
        # Result reads may provide a clearer rendering of the *same* letters.
        # They cannot establish membership or increase preview support.
        for name, count in result_counts.items():
            if count >= 2 and _key(name) == key and name not in counts:
                variants.append((name, count))
        # First settle the letters from independent preview observations.
        # A readable but less-supported misspelling (Parrett/Barrett) must
        # not win merely because its clan tag happens to be ASCII.
        spellings = Counter()
        for name, count in variants:
            if name in counts:
                spellings[_key(name)] += count
        best_spelling = max(spellings, key=lambda spelling: (spellings[spelling], len(spelling)))
        variants = [(name, count) for name, count in variants if _key(name) == best_spelling]
        # Prefer a recurring readable spelling over a more frequent OCR
        # confusable, while retaining the original reading as evidence.
        readable = [(name, count) for name, count in variants
                    if all(ord(c) < 128 for c in name if c.isalpha()) and count >= 2]
        options = readable or variants
        name = max(options, key=lambda item: (item[1], len(_key(item[0]))))[0]
        roster.append((best_spelling, name, support))
    return roster


def reconcile_names(races) -> int:
    """Correct uniquely matched rows; leave uncertain or colliding rows alone."""
    roster = _roster(races)
    changes = 0
    for race in races:
        proposed = {}
        for rank, displayed in race.placements:
            evidence = race.placement_evidence.setdefault(str(rank), {})
            original = evidence.get("raw_name", displayed)
            key = _key(original)
            if not key or not roster:
                continue
            ranked = sorted(((_similar(key, candidate_key), name, support)
                             for candidate_key, name, support in roster), reverse=True)
            best = ranked[0]
            runner = ranked[1][0] if len(ranked) > 1 else 0
            if best[0] < .72:
                # The per-race consensus sometimes chose a short bad read
                # while retaining a clearer spelling among its alternatives.
                for alternative in evidence.get("alternatives", []):
                    alternative_key = _key(alternative)
                    if _similar(alternative_key, _key(best[1])) >= .85 and best[0] >= .60:
                        best = (_similar(alternative_key, _key(best[1])), best[1], best[2])
                        break
            if best[0] >= .72 and best[0] - runner >= .10:
                proposed[rank] = (best[1], best[0], best[2], original)
        # A roster name can identify only one place in a race.
        collisions = Counter(_key(name) for name, _, _, _ in proposed.values())
        for row in race.placements:
            rank, displayed = row
            if rank not in proposed:
                continue
            name, score, support, original = proposed[rank]
            if collisions[_key(name)] != 1:
                continue
            evidence = race.placement_evidence.setdefault(str(rank), {})
            if displayed != name:
                row[1] = name
                evidence["raw_name"] = original
                evidence["session_match"] = {"name": name, "similarity": round(score, 3),
                                             "preview_races": support}
                evidence["status"] = "review"
                changes += 1
    return changes
