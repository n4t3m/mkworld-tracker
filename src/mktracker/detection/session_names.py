"""Use repeated course previews to resolve noisy result-name spellings."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from itertools import permutations
import unicodedata


def _key(text: str) -> str:
    # These are common OCR renderings of the same small clan-tag glyphs.
    text = text.casefold().translate(str.maketrans({"γ": "y", "ł": "l", "ı": "i", "²": ""}))
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _recover_supported_alternatives(races) -> int:
    """Use a same-row alternative after other races establish its spelling."""
    support = Counter(name for race in races for name in
                      {name for _, name in race.placements})
    changes = 0
    for race in races:
        present = {name for _, name in race.placements}
        for rank, displayed in race.placements:
            evidence = race.placement_evidence.get(str(rank), {})
            if evidence.get("status") != "review" or support[displayed] > 3:
                continue
            options = [name for name in set(evidence.get("alternatives", []))
                       if support[name] >= 4 and name not in present
                       and _similar(_key(displayed), _key(name)) >= .4]
            if len(options) != 1:
                continue
            candidate = options[0]
            for row in race.placements:
                if row[0] == rank:
                    row[1] = candidate
                    break
            evidence.setdefault("raw_name", displayed)
            evidence["session_match"] = {"name": candidate,
                                         "result_races": support[candidate]}
            present.add(candidate)
            changes += 1
    return changes


def _reconcile_initial_i_l(races) -> int:
    """Resolve a short I/l visual alias from repeated fixed-room results."""
    if sum(len(race.placements) >= 20 for race in races) < 4:
        return 0
    support = Counter(name for race in races for name in
                      {name for _, name in race.placements})
    changes = 0
    for source, count in support.items():
        if len(source) < 3 or source[0] not in "Il" or count > 2:
            continue
        candidate = ("l" if source[0] == "I" else "I") + source[1:]
        if support[candidate] < max(2, 2 * count):
            continue
        if any({source, candidate} <= {name for _, name in race.placements}
               for race in races):
            continue
        for race in races:
            present = {name for _, name in race.placements}
            if candidate in present:
                continue
            for rank, displayed in race.placements:
                if displayed != source:
                    continue
                for row in race.placements:
                    if row[0] == rank:
                        row[1] = candidate
                        break
                evidence = race.placement_evidence.setdefault(str(rank), {})
                evidence.setdefault("raw_name", displayed)
                evidence["session_match"] = {"name": candidate,
                                             "result_races": support[candidate],
                                             "font_alias": "I/l"}
                evidence["status"] = "review"
                changes += 1
    return changes


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
        # Preview OCR often renders a tiny raised team ² as a prime or drops
        # it. Restore it only when independent result screens repeatedly read
        # the same identity with that marker. Keep the preview's clearer
        # spelling of the actual player name after the tag.
        raised = [(candidate, count) for candidate, count in result_counts.items()
                  if (_key(candidate) == best_spelling and len(candidate) >= 4
                      and candidate[:2].casefold() == name[:2].casefold()
                      and candidate[2] == "²")]
        raised_support = sum(count for _, count in raised)
        if raised_support >= max(4, (support + 1) // 2) and len(name) >= 4:
            tail = name[2:].lstrip(" ²'’′")
            if tail:
                separated = sum(count for candidate, count in raised
                                if candidate[3] == " ")
                name = name[:2] + "²" + (" " if separated * 2 >= raised_support else "") + tail
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
    # A long 12-player room supplies its own roster evidence. Repair one
    # visibly corrupted spelling only when eleven other identities still
    # match and that player's name recurs across nearly every race. A genuine
    # substitute with a different name stays as read.
    complete_12p = [race for race in races if len(race.placements) == 12]
    if len(complete_12p) >= 8 and len(complete_12p) == len(races):
        counts = Counter(_key(name) for race in races for _, name in race.placements)
        variants: dict[str, Counter[str]] = {}
        for race in races:
            for _, name in race.placements:
                variants.setdefault(_key(name), Counter())[name] += 1
        stable = {key: variants[key].most_common(1)[0][0]
                  for key, count in counts.items()
                  if count >= max(7, len(races) - 2) and len(key) >= 4}
        if len(stable) == 12:
            for race in races:
                present = {_key(name) for _, name in race.placements}
                missing = set(stable) - present
                extra = [(rank, name) for rank, name in race.placements
                         if _key(name) not in stable]
                if len(missing) != 1 or len(extra) != 1:
                    continue
                target = missing.pop()
                rank, displayed = extra[0]
                key = _key(displayed)
                if ("".join(c for c in key if c.isdecimal()) !=
                        "".join(c for c in target if c.isdecimal())):
                    continue
                score = _similar(key, target)
                # The other stable identities already occupy the other eleven
                # rows, so similarity to one of them cannot identify this row.
                if score < .6:
                    continue
                candidate = stable[target]
                for row in race.placements:
                    if row[0] == rank:
                        row[1] = candidate
                        break
                evidence = race.placement_evidence.setdefault(str(rank), {})
                evidence.setdefault("raw_name", displayed)
                evidence["session_match"] = {"name": candidate,
                                             "similarity": round(score, 3),
                                             "result_races": counts[target]}
                evidence["status"] = "review"
                changes += 1
    if sum(len(race.placements) >= 20 for race in races) < 4:
        return changes
    # The 24-player layout often has no readable course-preview roster.
    # A spelling independently selected in several races can still settle a
    # noisy row when that exact spelling was also read as an alternative at
    # the same rank. Keep these corrections marked for review.
    changes += _recover_supported_alternatives(races)
    support = Counter(name for race in races for name in
                      {name for _, name in race.placements})
    stable: dict[str, tuple[str, int]] = {}
    for name, count in support.items():
        key = _key(name)
        if count >= 4 and len(key) >= 4:
            if key not in stable or count > stable[key][1]:
                stable[key] = (name, count)
    for race in races:
        present = Counter(_key(name) for _, name in race.placements)
        for rank, displayed in race.placements:
            key = _key(displayed)
            if support[displayed] > 3 or len(key) < 4:
                continue
            digits = "".join(c for c in key if c.isdecimal())
            scores = sorted(((_similar(key, candidate_key), name, count)
                             for candidate_key, (name, count) in stable.items()
                             if (not present[candidate_key] or
                                 (candidate_key == key and present[candidate_key] == 1))
                             and "".join(c for c in candidate_key if c.isdecimal()) == digits),
                            reverse=True)
            if not scores:
                continue
            threshold = .88 if len(_key(scores[0][1])) <= 4 else .82
            if scores[0][0] < threshold or (len(scores) > 1 and
                                                     scores[0][0] - scores[1][0] < .15):
                continue
            score, candidate, count = scores[0]
            for row in race.placements:
                if row[0] == rank:
                    row[1] = candidate
                    break
            evidence = race.placement_evidence.setdefault(str(rank), {})
            evidence.setdefault("raw_name", displayed)
            evidence["session_match"] = {"name": candidate,
                                         "similarity": round(score, 3),
                                         "result_races": count}
            evidence["status"] = "review"
            present[key] -= 1
            present[_key(candidate)] += 1
            changes += 1
    # A complete room normally repeats its roster. When exactly one common
    # identity is absent from a race, a same-rank OCR alternative can recover
    # it even if the winning per-frame spelling was unrelated (a scrolling
    # row can briefly contaminate a crop). Require a clear margin over every
    # other unresolved row, and keep the assignment for review.
    identity_support: dict[str, tuple[str, int]] = {}
    for name, count in support.items():
        key = _key(name)
        if count >= 4 and (key not in identity_support or
                           count > identity_support[key][1]):
            identity_support[key] = (name, count)
    stable_keys = set(identity_support)
    if len(stable_keys) >= 20:
        for race in races:
            if len(race.placements) != 24:
                continue
            missing = stable_keys - {_key(name) for _, name in race.placements}
            if len(missing) != 1:
                continue
            target = missing.pop()
            target_digits = "".join(c for c in target if c.isdecimal())
            ranked = []
            for rank, displayed in race.placements:
                if _key(displayed) in stable_keys:
                    continue
                evidence = race.placement_evidence.get(str(rank), {})
                alternatives = evidence.get("alternatives", [])
                score = max((_similar(_key(name), target) for name in alternatives
                             if "".join(c for c in _key(name) if c.isdecimal()) == target_digits),
                            default=0)
                ranked.append((score, rank, displayed))
            ranked.sort(reverse=True)
            if not ranked or ranked[0][0] < .75 or (len(ranked) > 1 and
                                                    ranked[0][0] - ranked[1][0] < .15):
                continue
            score, rank, displayed = ranked[0]
            candidate, count = identity_support[target]
            for row in race.placements:
                if row[0] == rank:
                    row[1] = candidate
                    break
            evidence = race.placement_evidence.setdefault(str(rank), {})
            evidence.setdefault("raw_name", displayed)
            evidence["session_match"] = {"name": candidate,
                                         "similarity": round(score, 3),
                                         "result_races": count}
            evidence["status"] = "review"
            changes += 1
    highlighted = [(race, race.highlighted_rank) for race in races
                   if race.highlighted_rank is not None
                   and race.highlighted_rank in dict(race.placements)]
    if len(highlighted) >= 4:
        digit_reads: dict[str, set[int]] = {}
        for index, (race, rank) in enumerate(highlighted):
            selected = dict(race.placements)[rank]
            evidence = race.placement_evidence.get(str(rank), {})
            for name in {selected, *evidence.get("alternatives", [])}:
                if (len(name) >= 4 and name[0].isalpha()
                        and all(c.isascii() and c.isalnum() for c in name)
                        and any(c.isdecimal() for c in name)):
                    digit_reads.setdefault(name, set()).add(index)
        ranked_digits = sorted(((len(seen), name) for name, seen in digit_reads.items()),
                               reverse=True)
        if (ranked_digits and ranked_digits[0][0] >= 2
                and (len(ranked_digits) == 1 or ranked_digits[0][0] > ranked_digits[1][0])):
            support_count, candidate = ranked_digits[0]
            candidate_key = _key(candidate)
            for race, rank in highlighted:
                displayed = dict(race.placements)[rank]
                if displayed == candidate or candidate_key in {
                        _key(name) for other_rank, name in race.placements if other_rank != rank}:
                    continue
                evidence = race.placement_evidence.setdefault(str(rank), {})
                if (_similar(_key(displayed), candidate_key) < .25
                        and candidate not in evidence.get("alternatives", [])):
                    continue
                for row in race.placements:
                    if row[0] == rank:
                        row[1] = candidate
                        break
                evidence.setdefault("raw_name", displayed)
                evidence["session_match"] = {"name": candidate,
                                             "highlighted_races": support_count}
                evidence["status"] = "review"
                changes += 1
    complete_support = Counter(name for race in races for name in
                               {name for _, name in race.placements})
    complete_roster: dict[str, tuple[str, int]] = {}
    for name, count in complete_support.items():
        key = _key(name)
        if count >= 4 and (key not in complete_roster or
                           count > complete_roster[key][1]):
            complete_roster[key] = (name, count)
    if len(complete_roster) == 24:
        for race in races:
            if len(race.placements) != 24:
                continue
            present = {_key(name) for _, name in race.placements}
            missing = sorted(set(complete_roster) - present)
            outliers = [(rank, name) for rank, name in race.placements
                        if _key(name) not in complete_roster]
            if not 1 <= len(missing) <= 3 or len(outliers) != len(missing):
                continue
            proposals = []
            for assignment in permutations(missing):
                scores = []
                for (rank, displayed), target in zip(outliers, assignment):
                    alternatives = race.placement_evidence.get(str(rank), {}).get(
                        "alternatives", [])
                    scores.append(max(_similar(_key(name), target)
                                      for name in [displayed, *alternatives]))
                proposals.append((sum(scores), min(scores), assignment, scores))
            proposals.sort(reverse=True)
            total, weakest, assignment, scores = proposals[0]
            if weakest < .4 or (len(proposals) > 1 and total - proposals[1][0] < .15):
                continue
            for (rank, displayed), target, score in zip(outliers, assignment, scores):
                candidate, count = complete_roster[target]
                for row in race.placements:
                    if row[0] == rank:
                        row[1] = candidate
                        break
                evidence = race.placement_evidence.setdefault(str(rank), {})
                evidence.setdefault("raw_name", displayed)
                evidence["session_match"] = {"name": candidate,
                                             "similarity": round(score, 3),
                                             "result_races": count}
                evidence["status"] = "review"
                changes += 1
    # Short names can have a convincing letter-only OCR consensus even when
    # the same digit was independently read in several races (Li4z/Lidz).
    # Use those repeated alternatives only for the matching roster identity.
    digit_evidence: dict[str, dict[str, set[int]]] = {}
    for race_index, race in enumerate(races):
        for rank, displayed in race.placements:
            source = _key(displayed)
            if len(source) != 4 or any(c.isdecimal() for c in source):
                continue
            evidence = race.placement_evidence.get(str(rank), {})
            for alternative in evidence.get("alternatives", []):
                candidate = _key(alternative)
                if (len(candidate) == 4 and candidate.isascii()
                        and candidate[:2] == source[:2]
                        and candidate[-1] == source[-1]
                        and any(c.isdecimal() for c in candidate)):
                    digit_evidence.setdefault(source, {}).setdefault(
                        candidate, set()).add(race_index)
    digit_corrections = {}
    for source, candidates in digit_evidence.items():
        supported = [candidate for candidate, seen in candidates.items()
                     if len(seen) >= 3]
        if len(supported) == 1:
            digit_corrections[source] = supported[0]
    digit_spellings = {}
    for candidate in digit_corrections.values():
        observed = Counter(name for race in races
                           for rank, _ in race.placements
                           for name in race.placement_evidence.get(
                               str(rank), {}).get("alternatives", [])
                           if _key(name) == candidate and name.isascii()
                           and name.isalnum())
        digit_spellings[candidate] = max(
            observed, key=lambda name: (observed[name], name[0].isupper()),
            default=candidate)
    for race in races:
        present = {_key(name) for _, name in race.placements}
        for rank, displayed in race.placements:
            candidate = digit_corrections.get(_key(displayed))
            if candidate is None or candidate in present:
                continue
            evidence = race.placement_evidence.setdefault(str(rank), {})
            spelling = digit_spellings[candidate]
            for row in race.placements:
                if row[0] == rank:
                    row[1] = spelling
                    break
            evidence.setdefault("raw_name", displayed)
            evidence["session_match"] = {"name": spelling,
                                         "digit_races": len(digit_evidence[_key(displayed)][candidate])}
            evidence["status"] = "review"
            present.add(candidate)
            changes += 1
    changes += _reconcile_initial_i_l(races)
    final_support = Counter(name for race in races for name in
                            {name for _, name in race.placements})
    for race in races:
        for rank, name in race.placements:
            if final_support[name] < 4:
                race.placement_evidence.setdefault(str(rank), {})["status"] = "review"
    # Earlier passes can raise a spelling's support from three to four.
    # A partial result page cannot use the complete-roster recovery above,
    # so revisit its directly observed alternatives once with final support.
    changes += _recover_supported_alternatives(races)
    return changes
