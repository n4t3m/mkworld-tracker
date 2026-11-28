"""Use colored result-row pixels to restore team tags and separators.

The recognizer often reads the main letters but loses a raised tag symbol or
space. A correction is applied only when several rows on the same board share
the prefix and their pixels show the expected tag boundary. The board-level
gate avoids treating an ordinary player name as a two-letter team tag.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np

from mktracker.detection.font_decoder import FontDecoder
from mktracker.detection.team_results import _binary


def trim_yellow_green_pattern(image: np.ndarray) -> np.ndarray:
    """Blank faint chevrons after yellow/green text before model inference.

    Dark text has substantially lower HSV value than these bright bars;
    highlighted white text is selected by low saturation instead. Red/blue
    crops are returned untouched because this preprocessing hurts them.
    """
    if image.ndim != 3 or not 45 <= image.shape[0] <= 75 or image.shape[1] < 300:
        return image
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    sample = np.s_[max(1, round(height * .05)):max(2, round(height * .17)),
                   round(width * .7):round(width * .8)]
    bar_hue = float(np.median(hue[sample]))
    bar_saturation = float(np.median(saturation[sample]))
    if bar_saturation < 80 or not (14 <= bar_hue <= 36 or 40 <= bar_hue <= 90):
        return image
    background_value = float(np.median(value[sample]))
    search_right = min(round(width * .82), width - 20)
    selected_pixels = (saturation < 70) & (value > 210)
    selected = selected_pixels[round(height * .25):round(height * .75), :search_right].mean() > .02
    ink = selected_pixels if selected else ((value < background_value - 35) & (saturation > 70))
    counts = ink[round(height * .25):round(height * .83), :search_right].sum(axis=0)
    columns = np.flatnonzero(counts >= 2)
    if not len(columns) or columns[-1] < 25:
        return image
    right = min(search_right, int(columns[-1]) + 6)
    fill = np.median(image[sample].reshape(-1, 3), axis=0).astype(np.uint8)
    cleaned = image.copy()
    cleaned[:, right:] = fill
    return cleaned


def _prefix_evidence(image: np.ndarray, decoder: FontDecoder | None) -> tuple[str, bool] | None:
    if image.ndim != 3 or not 45 <= image.shape[0] <= 75 or image.shape[1] < 150:
        return None
    binary = _binary(image)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary < 127).astype(np.uint8), 8)
    components = sorted(
        [(i, int(x), int(y), int(w), int(h), int(area))
         for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
         if 10 <= x <= 125 and area >= 8],
        key=lambda row: (row[1], row[2]),
    )
    full = [c for c in components if c[4] >= 16 and c[5] >= 45]
    if len(full) < 3:
        return None
    first, second = full[:2]
    if not 14 <= first[4] <= 35 or not 14 <= second[4] <= 35:
        return None
    if second[1] - first[1] - first[3] > 5:
        # A short lowercase tag glyph may fall out of ``full``. In that
        # case the next name glyph is not the second tag letter.
        return None
    end = second[1] + second[3]
    marker = next((c for c in components if end <= c[1] <= end + 9
                   and c[4] <= 13 and c[3] <= 12 and c[2] <= first[2] + 8), None)
    if marker is not None and decoder is not None:
        index, x, y, width, height, _ = marker
        read = decoder.classify(labels[y:y+height, x:x+width] == index)
        if read.score >= .8 and read.char not in ("²", "′", "'", "’", "2"):
            marker = None
    if marker is not None:
        # A raised 2 is close to a prime at video resolution. Use the visual
        # marker here; disagreement can still be flagged by name consensus.
        end = marker[1] + marker[3]
        following = next((c for c in full if c[1] >= end - 1), None)
        return ("²", following[1] - end >= 5) if following is not None else None
    third = full[2]
    if third[1] - end <= 3 and decoder is not None:
        x, y, width, height = third[1:5]
        read = decoder.classify(labels[y:y+height, x:x+width] == third[0])
        if read.char == "Σ" and read.score >= .83:
            end = x + width
            following = next((c for c in full[3:] if c[1] >= end - 2), None)
            return "Σ", following is not None and following[1] - end >= 5
    return "", third[1] - end >= 5


def _correct_one(prediction: str, evidence: tuple[str, bool] | None) -> str:
    if evidence is None or len(prediction) < 3 or not prediction[:2].isalpha():
        return prediction
    suffix, spaced = evidence
    remainder = prediction[2:]
    if not suffix and 3 <= prediction.find(" ") <= 4:
        return prediction
    # These glyphs can be part of a longer team tag. A two-letter gap check
    # must not move the separator in front of the tag's own punctuation.
    if not suffix and remainder[0] in "²Σ★!_-.":
        return prediction
    if suffix == "²" and remainder.startswith(("²", "'", "’", "′", '"')):
        remainder = remainder[1:]
    elif suffix == "Σ" and remainder.startswith(("Σ", "E", "Z")):
        remainder = remainder[1:]
    remainder = remainder.lstrip(" ")
    if not remainder:
        return prediction
    return prediction[:2] + suffix + (" " if spaced else "") + remainder


def _two_letter_end(image: np.ndarray) -> int | None:
    """Locate the end of two separate full-height tag letters."""
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    letters = sorted((int(x), int(w), int(h), int(area))
                     for x, _, w, h, area in stats[1:]
                     if 10 <= x <= 125 and 14 <= h <= 35 and area >= 45)
    if len(letters) < 3 or letters[0][0] > 30:
        return None
    first, second = letters[:2]
    if not 14 <= first[2] <= 35 or not 14 <= second[2] <= 35:
        return None
    if second[0] - first[0] - first[1] > 5:
        return None
    return second[0] + second[1]


def _merged_two_letter_gap(image: np.ndarray, expected_end: int) -> bool:
    """Recover a visible gap when the two tag letters touch in the mask."""
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    letters = sorted((int(x), int(w)) for x, _, w, h, area in stats[1:]
                     if 10 <= x <= 125 and 14 <= h <= 35 and area >= 45)
    if len(letters) < 2:
        return False
    first, following = letters[:2]
    end = first[0] + first[1]
    return (first[0] <= 30 and first[1] >= 30 and
            abs(end - expected_end) <= 5 and following[0] - end >= 5)


def _three_letter_boundary(image: np.ndarray) -> int | None:
    """Return the pixel end of a three-letter prefix with a visible gap."""
    if image.ndim != 3 or not 45 <= image.shape[0] <= 75 or image.shape[1] < 150:
        return None
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    letters = sorted((int(x), int(w), int(h), int(area))
                     for x, _, w, h, area in stats[1:]
                     if 10 <= x <= 150 and 14 <= h <= 35 and area >= 45)
    if len(letters) < 4:
        return None
    first = letters[:4]
    gaps = [first[i + 1][0] - first[i][0] - first[i][1]
            for i in range(3)]
    return first[2][0] + first[2][1] if max(gaps[:2]) <= 4 and gaps[2] >= 7 else None


def _three_letter_gap(image: np.ndarray) -> bool:
    return _three_letter_boundary(image) is not None


def _merged_three_letter_gap(image: np.ndarray, expected_end: int) -> bool:
    """Check the same boundary when adjacent tag letters form one component."""
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    letters = sorted((int(x), int(w)) for x, _, w, h, area in stats[1:]
                     if 10 <= x <= 150 and 14 <= h <= 35 and area >= 45)
    if len(letters) < 2 or letters[0][0] > 40:
        return False
    for index in range(min(3, len(letters) - 1)):
        end = letters[index][0] + letters[index][1]
        if abs(end - expected_end) <= 5 and letters[index + 1][0] - end >= 7:
            return True
    return False


def _four_letter_gap(image: np.ndarray) -> bool:
    """Find the visible separator after four adjacent full-size tag glyphs."""
    if image.ndim != 3 or not 45 <= image.shape[0] <= 75 or image.shape[1] < 150:
        return False
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    letters = sorted((int(x), int(w)) for x, _, w, h, area in stats[1:]
                     if 12 <= x <= 150 and 14 <= h <= 35 and area >= 45)
    if len(letters) < 5:
        return False
    gaps = [letters[i + 1][0] - letters[i][0] - letters[i][1]
            for i in range(4)]
    return max(gaps[:3]) <= 4 and gaps[3] >= 7


def _fourth_tag_symbol(image: np.ndarray, decoder: FontDecoder) -> tuple[str, bool] | None:
    """Read a strong fourth tag glyph and the visible gap after it."""
    if image.ndim != 3 or not 45 <= image.shape[0] <= 75 or image.shape[1] < 150:
        return None
    reads = decoder.segment(_binary(image[:, 12:130]))
    if len(reads) < 4:
        return None
    fourth = reads[3]
    if (fourth.char not in ("Σ", "_") or fourth.score < .86 or
            fourth.margin < .025 or fourth.left > 105):
        return None
    gap = (reads[4].left - fourth.left - fourth.width >= 6
           if len(reads) > 4 else False)
    return fourth.char, gap


def _leading_symbol_tag(
    image: np.ndarray, prediction: str, decoder: FontDecoder,
) -> str:
    """Restore a visible star/diamond after a repeated one-letter team tag."""
    if (len(prediction) < 2 or not prediction[0].isascii() or
            not prediction[0].isalpha() or image.shape[1] < 150 or
            not all(char in decoder.chars for char in ("★", "◆"))):
        return prediction
    binary = _binary(image[:, 12:130])
    reads = decoder.segment(binary)
    if len(reads) < 3:
        return prediction
    initial = reads[0]
    initial_match = initial.char.casefold() == prediction[0].casefold()
    if not initial_match:
        mask = binary[initial.top:initial.top+initial.height,
                      initial.left:initial.left+initial.width] < 127
        scores = decoder._shape_scores(mask)
        initial_match = any(glyph == prediction[0] and float(scores[i]) >= .84
                            for i, glyph in enumerate(decoder.chars))
    if not initial_match or initial.score < .84:
        return prediction
    symbol = reads[1]
    mask = binary[symbol.top:symbol.top+symbol.height,
                  symbol.left:symbol.left+symbol.width] < 127
    scores = decoder._shape_scores(mask)
    candidates = {char: max(float(scores[i]) for i, glyph in enumerate(decoder.chars)
                            if glyph == char) for char in ("★", "◆")}
    best = max(candidates, key=candidates.get)
    other = "◆" if best == "★" else "★"
    existing = prediction[1] if prediction[1] in "★◆" else None
    if existing is None and (candidates[best] < .82 or
                             candidates[best] - candidates[other] < .02):
        return prediction
    if existing is not None:
        if candidates[existing] < .80 or candidates[best] - candidates[existing] > .04:
            return prediction
        best = existing
    remainder = prediction[2:] if existing is not None else prediction[1:]
    if not remainder.strip():
        return prediction
    gap = reads[2].left - symbol.left - symbol.width
    return prediction[0] + best + (" " if gap >= 6 else "") + remainder.lstrip()


def _restore_music_note(
    image: np.ndarray, prediction: str, decoder: FontDecoder,
    *, repeated_prefix: bool,
) -> str:
    """Use a local game-font shape to disambiguate a tag or trailing note."""
    if ("♪" not in decoder.chars or len(prediction) < 5 or
            image.ndim != 3 or image.shape[1] < 150):
        return prediction
    reads = [read for read in decoder.segment(_binary(image[:, 10:230]))
             if read.score >= .5]
    if (repeated_prefix and len(reads) >= 3 and prediction[2] in "Jj" and
            reads[2].char == "♪" and reads[2].score >= .68 and
            reads[2].margin >= .03):
        return prediction[:2] + "♪" + prediction[3:]
    if (prediction[-1] in "sJj" and reads and reads[-1].char == "♪" and
            reads[-1].score >= .72 and reads[-1].margin >= .08):
        return prediction[:-1] + "♪"
    return prediction


def _restore_emoticon_colon(image: np.ndarray, prediction: str) -> str:
    """Keep a visibly dotted colon before a final 3 in a colored name."""
    if not prediction.endswith(" 3") or image.ndim != 3:
        return prediction
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    full = sorted((int(x), int(w)) for x, _, w, h, area in stats[1:]
                  if 10 <= x <= 220 and 14 <= h <= 35 and area >= 80)
    if not full:
        return prediction
    digit_x, _ = full[-1]
    dots = [(int(x + w / 2), int(y + h / 2))
            for x, y, w, h, area in stats[1:]
            if digit_x - 16 <= x < digit_x and 10 <= area <= 50 and
            w <= 8 and h <= 8 and 18 <= y <= 40]
    if any(abs(x1 - x2) <= 5 and 8 <= abs(y1 - y2) <= 16
           for x1, y1 in dots for x2, y2 in dots if y1 < y2):
        return prediction[:-1] + ":3"
    return prediction


def _restore_visible_slash(image: np.ndarray, prediction: str) -> str:
    """Distinguish a steep slash from an italic capital I in a player name."""
    candidates = [index for index in range(1, len(prediction) - 1)
                  if prediction[index] == "I" and prediction[index - 1].islower()
                  and prediction[index + 1].isupper()]
    if not candidates or image.ndim != 3:
        return prediction
    binary = _binary(image)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary < 127).astype(np.uint8), 8)
    glyphs = [(index, int(x), int(y), int(w), int(h), int(area))
              for index, (x, y, w, h, area) in enumerate(stats[1:], 1)
              if 10 <= x <= 250 and 14 <= h <= 35 and 80 <= area <= 800 and w <= 40]
    if len(glyphs) < 4:
        return prediction
    left = min(glyph[1] for glyph in glyphs)
    right = max(glyph[1] + glyph[3] for glyph in glyphs)
    if right - left < 70:
        return prediction
    for component, x, y, width, height, _ in glyphs:
        if not (18 <= height <= 35 and 10 <= width <= 25 and
                .45 * height <= width <= .95 * height):
            continue
        mask = labels[y:y+height, x:x+width] == component
        quarter = max(3, height // 4)
        top = [float(np.mean(np.flatnonzero(mask[row])))
               for row in range(quarter) if mask[row].any()]
        bottom = [float(np.mean(np.flatnonzero(mask[row])))
                  for row in range(height-quarter, height) if mask[row].any()]
        if len(top) < 2 or len(bottom) < 2 or np.mean(top) - np.mean(bottom) < 8:
            continue
        position = (x + width / 2 - left) / (right - left)
        for index in candidates:
            if abs(position - (index + .5) / len(prediction)) <= .17:
                return prediction[:index] + "/" + prediction[index + 1:]
    return prediction


def correct_colored_board(
    rows: list[tuple[np.ndarray, str]], decoder: FontDecoder | None,
) -> list[str]:
    """Correct a batch of model names from one colored result screen."""
    evidence = [_prefix_evidence(image, decoder) for image, _ in rows]
    three_groups: dict[str, list[int]] = defaultdict(list)
    four_groups: dict[str, list[int]] = defaultdict(list)
    for index, (_, prediction) in enumerate(rows):
        if len(prediction) >= 4 and prediction[:3].isascii() and prediction[:3].isalpha():
            three_groups[prediction[:3].casefold()].append(index)
        if len(prediction) >= 5 and prediction[:4].isascii() and prediction[:4].isalpha():
            four_groups[prediction[:4].casefold()].append(index)
    three_tag_indices = {index for indices in three_groups.values()
                         if len(indices) >= 3 and not any(
                             evidence[i] is not None and evidence[i][0]
                             for i in indices)
                         for index in indices}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (_, prediction) in enumerate(rows):
        if index not in three_tag_indices and len(prediction) >= 2 and prediction[:2].isalpha():
            groups[prediction[:2].casefold()].append(index)
    # A tiny raised marker can be classified as an unrelated glyph at video
    # resolution. When it appears at the same two-letter boundary throughout
    # a team, use the repeated visible markers despite those individual font
    # classifications. This also covers frames where the model drops ².
    for indices in groups.values():
        if len(indices) < 3:
            continue
        visual = {index: _prefix_evidence(rows[index][0], None) for index in indices}
        raised = [index for index in indices
                  if visual[index] is not None and visual[index][0] == "²"]
        if len(raised) < 3 or len(raised) * 2 < len(indices):
            continue
        for index in raised:
            if evidence[index] is None or not evidence[index][0]:
                evidence[index] = visual[index]
    corrected = [prediction for _, prediction in rows]
    for indices in groups.values():
        supports = [index for index in indices
                    if evidence[index] is not None and
                    (evidence[index][0] or evidence[index][1])]
        if len(indices) < 3 or len(supports) < 2:
            continue
        ends = [_two_letter_end(rows[index][0]) for index in supports]
        expected_end = round(float(np.median([end for end in ends if end is not None]))) if any(
            end is not None for end in ends) else None
        variants = Counter(rows[index][1][:2] for index in indices)
        preferred, votes = variants.most_common(1)[0]
        for index in indices:
            current_evidence = evidence[index]
            if (current_evidence is None and expected_end is not None and
                    _merged_two_letter_gap(rows[index][0], expected_end)):
                current_evidence = ("", True)
            value = _correct_one(rows[index][1], current_evidence)
            if (value != rows[index][1] and votes >= 2 and len(value) >= 2 and
                    value[:2].casefold() == preferred.casefold()):
                value = preferred + value[2:]
            corrected[index] = value
    for indices in three_groups.values():
        boundaries = {i: _three_letter_boundary(rows[i][0]) for i in indices}
        supported = [end for end in boundaries.values() if end is not None]
        if (len(indices) < 3 or not set(indices) <= three_tag_indices or
                len(supported) < 2):
            continue
        expected_end = round(float(np.median(supported)))
        variants = Counter(rows[i][1][:3] for i in indices)
        preferred, votes = variants.most_common(1)[0]
        for index in indices:
            if (boundaries[index] is None and
                    not _merged_three_letter_gap(rows[index][0], expected_end)):
                continue
            value = corrected[index]
            if len(value) < 4 or not value[:3].isascii() or not value[:3].isalpha():
                continue
            if value[3] == " ":
                continue
            if value[3] in "²Σ★!_-.":
                continue
            corrected[index] = (preferred if votes >= 2 else value[:3]) + " " + value[3:].lstrip()
    for indices in four_groups.values():
        if len(indices) < 3 or sum(_four_letter_gap(rows[i][0]) for i in indices) < 2:
            continue
        variants = Counter(rows[i][1][:4] for i in indices)
        preferred, votes = variants.most_common(1)[0]
        for index in indices:
            value = corrected[index]
            if len(value) < 5 or value[:4].casefold() != preferred.casefold():
                continue
            if votes >= 2:
                value = preferred + value[4:]
            if (_four_letter_gap(rows[index][0]) and value[4] != " " and
                    value[4] not in "²Σ★◆!_-."):
                value = value[:4] + " " + value[4:]
            corrected[index] = value
    if decoder is not None and hasattr(decoder, "segment"):
        for indices in three_groups.values():
            if len(indices) < 3:
                continue
            symbols = {index: _fourth_tag_symbol(rows[index][0], decoder)
                       for index in indices}
            votes = Counter(value[0] for value in symbols.values() if value is not None)
            if not votes:
                continue
            symbol, support = votes.most_common(1)[0]
            if support < 2 or any(other != symbol for other in votes):
                continue
            for index in indices:
                evidence = symbols[index]
                if evidence is None or evidence[0] != symbol:
                    continue
                value = corrected[index]
                if len(value) < 4 or value[:3].casefold() != rows[index][1][:3].casefold():
                    continue
                remainder = value[3:]
                if remainder.startswith(symbol):
                    remainder = remainder[1:]
                elif (symbol == "Σ" and remainder.startswith(("E", "Z"))) or (symbol == "_" and remainder.startswith("-")):
                    remainder = remainder[1:]
                elif remainder[0] in "²Σ★!_-.":
                    continue
                corrected[index] = value[:3] + symbol + (" " if evidence[1] else "") + remainder.lstrip()
    if (decoder is not None and hasattr(decoder, "segment") and
            hasattr(decoder, "_shape_scores") and hasattr(decoder, "chars")):
        supported_initials = Counter(
            value[0].casefold() for value in corrected
            if len(value) >= 3 and value[0].isascii() and
            value[0].isalpha() and value[1] in "★◆")
        for index, ((image, _), value) in enumerate(zip(rows, corrected)):
            if value and supported_initials[value[0].casefold()] >= 2:
                corrected[index] = _leading_symbol_tag(image, value, decoder)
        prefix_counts = Counter(value[:2].casefold() for value in corrected
                                if len(value) >= 4 and value[:2].isalpha())
        corrected = [_restore_music_note(
            image, value, decoder,
            repeated_prefix=prefix_counts[value[:2].casefold()] >= 3,
        ) for (image, _), value in zip(rows, corrected)]
    corrected = [_restore_emoticon_colon(image, value)
                 for (image, _), value in zip(rows, corrected)]
    corrected = [_restore_visible_slash(image, value)
                 for (image, _), value in zip(rows, corrected)]
    return corrected
