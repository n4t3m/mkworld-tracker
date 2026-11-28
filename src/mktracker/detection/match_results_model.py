"""Crop final-standings names and scores for the trained text recognizer."""
from __future__ import annotations

from difflib import SequenceMatcher
import unicodedata

import cv2
import numpy as np

from mktracker.match_record import FinalStandings


# Coordinates in the game's 1920 × 1080 full-screen layout. Each entry is
# (name left, name right, score left, score right). The first row center and
# pitch come from the game's result-card layout, not the source recording.
_LAYOUTS = {
    "no_teams_small": (214, 69, ((280, 790, 800, 875),)),
    "no_teams_large": (240, 64, ((180, 460, 480, 535),
                                   (705, 980, 995, 1060))),
    "two_teams_small": (695, 65, ((400, 810, 850, 940),
                                    (1060, 1490, 1520, 1605))),
    "two_teams_large": (695, 65, ((85, 395, 410, 480),
                                    (565, 865, 880, 955),
                                    (1045, 1340, 1360, 1430),
                                    (1520, 1820, 1840, 1915))),
    "three_teams": (570, 65, ((300, 610, 630, 700),
                               (805, 1110, 1130, 1200),
                               (1305, 1620, 1635, 1700))),
    "four_teams": (695, 65, ((90, 400, 410, 480),
                              (565, 870, 880, 955),
                              (1040, 1340, 1350, 1430),
                              (1510, 1820, 1840, 1915))),
}

# Large team totals are independent evidence for the row-score OCR. These
# coordinates enclose only the numerals, below the team crests.
_TEAM_TOTAL_CROPS = {
    "two_teams_small": (465, 610, ((535, 815), (1180, 1510))),
    "two_teams_large": (465, 610, ((375, 685), (1330, 1660))),
    "three_teams": (375, 500, ((390, 620), (885, 1140), (1380, 1660))),
    "four_teams": (490, 615, ((250, 430), (720, 900),
                              (1110, 1360), (1570, 1840))),
}


def _name_key(name: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", name).casefold()
                   if char.isalnum())


def reconcile_final_rows(
    rows: list[tuple[int, int, str, int]], roster: list[str],
    *, min_similarity: float = .78,
) -> list[tuple[int, int, str, int]]:
    """Corroborate final-card names against players read in prior races.

    Only unambiguous one-to-one matches are replaced. A truncated team tag
    can be completed by exclusion when every other row has a unique match.
    Scores and card positions are never changed.
    """
    candidates = list(dict.fromkeys(name for name in roster if _name_key(name)))
    if not rows or len(candidates) != len(rows):
        return rows
    matched: dict[int, int] = {}
    available = set(range(len(candidates)))
    # Process the strongest reads first; this leaves short, ambiguous tags
    # until the full names have claimed their respective roster entries.
    options = []
    for index, (_, _, name, _) in enumerate(rows):
        key = _name_key(name)
        ranked = sorted(((SequenceMatcher(None, key, _name_key(candidate)).ratio(),
                          candidate_index)
                         for candidate_index, candidate in enumerate(candidates)),
                        reverse=True)
        if ranked:
            options.append((ranked[0][0], index, ranked))
    for _, index, ranked in sorted(options, reverse=True):
        remaining = [(score, candidate_index) for score, candidate_index in ranked
                     if candidate_index in available]
        if not remaining:
            continue
        best, candidate_index = remaining[0]
        next_score = remaining[1][0] if len(remaining) > 1 else 0.0
        if best >= min_similarity and best - next_score >= .08:
            matched[index] = candidate_index
            available.remove(candidate_index)
    unmatched_rows = [index for index in range(len(rows)) if index not in matched]
    if len(unmatched_rows) == len(available) == 1:
        index = unmatched_rows[0]
        candidate_index = next(iter(available))
        observed = _name_key(rows[index][2])
        expected = _name_key(candidates[candidate_index])
        if len(observed) >= 2 and expected.startswith(observed):
            matched[index] = candidate_index
    return [(column, row, candidates[matched[index]] if index in matched else name,
             score)
            for index, (column, row, name, score) in enumerate(rows)]


def _small_team_x_shift(image: np.ndarray) -> int:
    """Align the two-team bars when gameplay has horizontal capture padding."""
    hsv = cv2.cvtColor(image[670:720], cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 20) & (hsv[:, :, 1] > 90) &
           (hsv[:, :, 2] > 70))
    columns = (red.mean(axis=0) > .15).astype(np.uint8)[None, :]
    columns = cv2.morphologyEx(
        columns, cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8))[0]
    edges = np.diff(np.pad(columns.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    spans = [(int(start), int(end)) for start, end in zip(starts, ends)
             if 500 <= end - start <= 700]
    if not spans:
        return 0
    start, _ = max(spans, key=lambda span: span[1] - span[0])
    shift = start - 320
    return shift if abs(shift) <= 100 else 0


def read_final_standings(
    frame: np.ndarray, *, teams: str, player_count: int, reader,
) -> list[tuple[str, int]]:
    """Read final cards in one name batch and one narrow score batch.

    Screen readiness is checked by ``MatchResultDetector`` before this call.
    Blank rows and score-only cells are omitted.
    """
    return [(name, score) for _, _, name, score in read_final_standings_rows(
        frame, teams=teams, player_count=player_count, reader=reader)]


def read_final_standings_rows(
    frame: np.ndarray, *, teams: str, player_count: int, reader,
) -> list[tuple[int, int, str, int]]:
    """Return ``(column, row, name, score)`` for each readable card."""
    if teams == "No Teams":
        layout = "no_teams_small" if player_count <= 12 else "no_teams_large"
    elif teams == "Two Teams":
        layout = "two_teams_small" if player_count <= 12 else "two_teams_large"
    elif teams == "Three Teams":
        layout = "three_teams"
    elif teams == "Four Teams":
        layout = "four_teams"
    else:
        return []
    first, pitch, columns = _LAYOUTS[layout]
    image = (frame if frame.shape[:2] == (1080, 1920) else
             cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA))
    x_shift = _small_team_x_shift(image) if layout == "two_teams_small" else 0
    per_column = (player_count + len(columns) - 1) // len(columns)
    name_crops = []
    score_crops = []
    retry_crops = []
    short_name_crops = []
    score_retry_crops = []
    for x1, x2, sx1, sx2 in columns:
        x1, x2, sx1, sx2 = (x1 + x_shift, x2 + x_shift,
                            sx1 + x_shift, sx2 + x_shift)
        for index in range(per_column):
            center = first + index * pitch
            if center + 28 > 1080:
                break
            y1, y2 = center - 26, center + 28
            name_crops.append(image[y1:y2, x1:x2])
            score_crops.append(image[y1:y2, sx1:sx2])
            retry_crops.append(cv2.convertScaleAbs(
                image[center-19:center+21, x1:x2], alpha=1.3, beta=10))
            short_name_crops.append(image[center-20:center+20,
                                          x1+5:min(x1+115, x2)])
            score_retry_crops.append(image[center-19:center+21, sx1:sx2])
    if not name_crops:
        return []
    score_predictions = reader.predict_crops(score_crops, width=64)
    name_predictions = reader.predict_crops(name_crops)
    if len(score_predictions) != len(name_crops) or len(name_predictions) != len(name_crops):
        return []
    retry_indices = [index for index, ((name, confidence), (score, score_conf))
                     in enumerate(zip(name_predictions, score_predictions))
                     if confidence < .55 and score.isdigit() and score_conf >= .6]
    if retry_indices:
        retries = reader.predict_crops([retry_crops[index] for index in retry_indices])
        if len(retries) == len(retry_indices):
            for index, (name, confidence) in zip(retry_indices, retries):
                if (len(name.strip()) >= 2 and confidence >= .7 and
                        confidence - name_predictions[index][1] >= .15):
                    name_predictions[index] = name, confidence
    short_retry_indices = [index for index, ((name, confidence), (score, score_conf))
                           in enumerate(zip(name_predictions, score_predictions))
                           if layout == "no_teams_large" and
                           len(name.strip()) == 2 and confidence < .8 and
                           score.isdigit() and score_conf >= .6]
    if short_retry_indices:
        retries = reader.predict_crops(
            [short_name_crops[index] for index in short_retry_indices])
        if len(retries) == len(short_retry_indices):
            for index, (name, confidence) in zip(short_retry_indices, retries):
                old_name, old_confidence = name_predictions[index]
                if (len(name.strip()) == 2 and
                        name[0].casefold() == old_name[0].casefold() and
                        confidence >= .85 and confidence - old_confidence >= .15):
                    name_predictions[index] = name.strip(), confidence
    score_retry_indices = [index for index, ((name, name_conf), (score, _))
                           in enumerate(zip(name_predictions, score_predictions))
                           if name.strip() and name_conf >= .55 and not score.strip().isdigit()]
    if score_retry_indices:
        retries = reader.predict_crops(
            [score_retry_crops[index] for index in score_retry_indices], width=64)
        if len(retries) == len(score_retry_indices):
            for index, (score, confidence) in zip(score_retry_indices, retries):
                if (score.strip().isdigit() and confidence >= .8 and
                        confidence - score_predictions[index][1] >= .05):
                    score_predictions[index] = score, confidence
    # Scores are displayed in descending order within each column. A clipped
    # leading digit can still produce a plausible, confident number (11 -> 1).
    # Retry just the cells that break this ordering with a tighter crop.
    order_retry_indices = []
    for column in range(len(columns)):
        start = column * per_column
        for row in range(per_column - 1):
            index = start + row
            upper, lower = score_predictions[index][0].strip(), score_predictions[index + 1][0].strip()
            if upper.isdigit() and lower.isdigit() and int(upper) < int(lower):
                order_retry_indices.append(index)
    if order_retry_indices:
        retries = reader.predict_crops(
            [score_retry_crops[index] for index in order_retry_indices], width=64)
        if len(retries) == len(order_retry_indices):
            for index, (score, confidence) in zip(order_retry_indices, retries):
                score = score.strip()
                previous = score_predictions[index - 1][0].strip() if index % per_column else ""
                following = score_predictions[index + 1][0].strip()
                if (score.isdigit() and confidence >= .8 and
                        confidence > score_predictions[index][1] and
                        (not previous.isdigit() or int(score) <= int(previous)) and
                        (not following.isdigit() or int(score) >= int(following))):
                    score_predictions[index] = score, confidence
    # An unresolved ordering contradiction makes the whole model card unsafe.
    # The caller can then use the established OCR/API fallback.
    for column in range(len(columns)):
        start = column * per_column
        values = [score_predictions[start + row][0].strip()
                  for row in range(per_column)]
        if any(upper.isdigit() and lower.isdigit() and int(upper) < int(lower)
               for upper, lower in zip(values, values[1:])):
            return []
    results = []
    for index, ((name, name_conf), (score, score_conf)) in enumerate(
        zip(name_predictions, score_predictions)
    ):
        name = name.strip()
        score = score.strip()
        if (name and name_conf >= .55 and score.isdigit() and
                score_conf >= .6 and 0 <= int(score) <= 999):
            results.append((index // per_column, index % per_column,
                            name, int(score)))
    if layout in _TEAM_TOTAL_CROPS and len(results) == player_count:
        top, bottom, horizontal = _TEAM_TOTAL_CROPS[layout]
        header_shift = x_shift if layout == "two_teams_small" else 0
        totals = reader.predict_crops(
            [image[top:bottom, left + header_shift:right + header_shift]
             for left, right in horizontal])
        if len(totals) != len(horizontal):
            return []
        for column, (total, confidence) in enumerate(totals):
            column_span = (2 * column, 2 * column + 1) if layout == "two_teams_large" else (column,)
            if (not total.strip().isdigit() or confidence < .9 or
                    int(total) != sum(score for cell, _, _, score in results
                                      if cell in column_span)):
                return []
    return results


def read_saved_final_standings(frame: np.ndarray, *, teams: str,
                               roster: list[str]) -> FinalStandings | None:
    """Use a complete, roster-checked model card for Match History rereads."""
    if not roster or len(set(roster)) != len(roster):
        return None
    from mktracker.detection.match_results import MatchResultDetector
    from mktracker.detection.name_model import NameModelReader
    from mktracker.team_scoring import local_final_standings

    color = teams != "No Teams"
    if not (NameModelReader.color_available() if color else NameModelReader.available()):
        return None
    reader = NameModelReader(color=color)
    try:
        detector = MatchResultDetector(
            color_model=reader if color else None,
            dark_model=None if color else reader,
        )
        result = detector.detect(frame, teams=teams,
                                 player_count=len(roster), model_only=True)
        if result is None or reader.disabled:
            return None
        rows = reconcile_final_rows(result.get("model_rows", []), roster)
        if {name for _, _, name, _ in rows} != set(roster):
            return None
        return local_final_standings(rows, teams, len(roster))
    finally:
        reader.close()
