"""Read colored result rows independently, excluding avatars and scores."""
from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract
from mktracker.detection.font_symbols import correct_symbol_words
from mktracker.detection.result_grid import rank_origin


def _team_color(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    team_hue = ((hue < 12) | (hue > 170) | ((hue > 95) & (hue < 125)) |
                ((hue >= 14) & (hue <= 36)) | ((hue >= 40) & (hue <= 90)))
    return team_hue & (sat > 140) & (val > 100)


def _bands(frame: np.ndarray) -> list[tuple[int, int]]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame[:, int(w * .59):int(w * .99)], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = _team_color(hue, sat, val).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, max(3, int(w * .034))), np.uint8))
    rows = mask.mean(axis=1) > 180
    diff = np.diff(np.r_[False, rows, False].astype(int))
    runs = [(int(a), int(b)) for a, b in zip(np.where(diff == 1)[0], np.where(diff == -1)[0])
            if h * .045 < b - a < h * .18 and a >= h * .01 and b <= h * .93]
    normal = [(a, b) for a, b in runs if b - a < h * .09]
    bands = list(normal)
    if len(normal) >= 3:
        height = float(np.median([b - a for a, b in normal]))
        for a, b in runs:
            # Adjacent opaque result bars can remain connected in the broad
            # HSV strip even across their thin border. Restore two row bands
            # only when their combined height matches the same page's rows.
            if 1.8 * height <= b - a <= 2.2 * height:
                middle = round((a + b) / 2)
                bands.extend(((a, middle), (middle, b)))
    bands.sort()
    if not bands:
        return []
    # Scrolling can omit a row, but distant scenery is not part of the table.
    # Only infer ranks within the largest connected group of row bands.
    height = float(np.median([b-a for a, b in bands]))
    groups = [[]]
    for band in bands:
        if groups[-1] and band[0]-groups[-1][-1][0] > height*2.5:
            groups.append([])
        groups[-1].append(band)
    return max(groups, key=len)


def _layout_shift(frame: np.ndarray, bands: list[tuple[int, int]]) -> float:
    """Locate the rounded bar edge; 24-player tables shift left ~50px."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    colored = _team_color(hue, sat, val)
    centers = [round((a + b) / 2) for a, b in bands]
    left, right = round(w * .50), round(w * .70)
    coverage = colored[centers, left:right].mean(axis=0)
    columns = np.flatnonzero(coverage >= .70)
    if not len(columns):
        return 0.0
    edge = (left + int(columns[0])) / w
    shift = edge - 1100 / 1920
    if abs(shift) < .003 or abs(shift) > .045:
        return 0.0
    return shift


def _plus_rows(frame: np.ndarray, bands: list[tuple[int, int]], shift: float) -> int:
    h, w = frame.shape[:2]
    count = 0
    for a, b in bands:
        zone = cv2.cvtColor(frame[a:b, int(w * (.89 + shift)):int(w * (.945 + shift))],
                            cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(zone, np.array([18, 100, 140]), np.array([42, 255, 255]))
        pixels = np.count_nonzero(yellow)
        # Yellow team's bar fills the whole score zone. Count only compact
        # +point lettering on non-yellow bars.
        if h * w * .000015 < pixels < zone.shape[0] * zone.shape[1] * .15:
            count += 1
    return count


def _band_team_color(frame: np.ndarray, band: tuple[int, int], shift: float) -> str | None:
    """Read the solid team colour behind a result row without text OCR."""
    h, w = frame.shape[:2]
    a, b = band
    margin = max(2, round((b - a) * .2))
    cell = frame[a + margin:b - margin,
                 round(w * (.80 + shift)):round(w * (.85 + shift))]
    if cell.size == 0:
        return None
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    visible = (sat > 85) & (val > 75)
    if visible.mean() < .4:
        return None
    masks = {
        "red": (hue < 15) | (hue >= 170),
        "yellow": (hue >= 15) & (hue < 40),
        "green": (hue >= 40) & (hue < 90),
        "blue": (hue >= 90) & (hue < 145),
    }
    scores = {name: float((visible & mask).mean()) for name, mask in masks.items()}
    color, coverage = max(scores.items(), key=lambda item: item[1])
    return color if coverage >= .35 else None


def _binary(cell: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 210)
    if white.mean() > .035:
        return (~white).astype(np.uint8) * 255
    return cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _sheet(frame: np.ndarray, bands: list[tuple[int, int]], x1: float, x2: float, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cells = []
    for a, b in bands:
        pad = max(2, round((b - a) * .07))
        cell = _binary(frame[a + pad:b - pad, int(w * x1):int(w * x2)])
        scale = min(width / cell.shape[1], 120 / cell.shape[0])
        cell = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        top = (160 - cell.shape[0]) // 2
        cells.append(cv2.copyMakeBorder(cell, top, 160 - cell.shape[0] - top,
                                       10, width + 10 - cell.shape[1], cv2.BORDER_CONSTANT, value=255))
    return np.vstack(cells)


def _read_sheet(sheet: np.ndarray, count: int, *, digits: bool = False, deskew: bool = True,
                details: list | None = None) -> list[tuple[str, float]]:
    if not digits and deskew:
        sheet = sheet.copy()
        for row in range(count):
            a, b = row * 160, (row + 1) * 160
            sheet[a:b] = _deslant(sheet[a:b])
    config = "--psm 6" + (" -c tessedit_char_whitelist=0123456789" if digits else "")
    data = pytesseract.image_to_data(sheet, config=config, output_type=pytesseract.Output.DICT)
    words: list[list[dict]] = [[] for _ in range(count)]
    for i, text in enumerate(data["text"]):
        text = text.strip()
        confidence = float(data["conf"][i])
        if not text or confidence < 0:
            continue
        row = int((data["top"][i] + data["height"][i] / 2) // 160)
        if 0 <= row < count:
            words[row].append({"left": data["left"][i], "width": data["width"][i], "text": text, "conf": confidence})
    readings = []
    for i, row in enumerate(words):
        if not digits:
            row = correct_symbol_words(sheet[i*160:(i+1)*160], row, deskewed=deskew)
        row.sort(key=lambda word: word["left"])
        readings.append((" ".join(word["text"] for word in row), float(np.mean([word["conf"] for word in row])) if row else 0))
        if details is not None:
            details.append({"needs_review": any(word.get("font_uncertain", False) for word in row)})
    return readings


def _deslant(image: np.ndarray) -> np.ndarray:
    # The game's italic lettering turns B into P (and t into e) in OCR.
    # Undo its lean before recognition, preserving each row's own baseline.
    h, w = image.shape[:2]
    matrix = np.float32([[1, .18, -.09 * h], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h), borderValue=255)


def read_team_results(frame: np.ndarray) -> dict | None:
    bands = _bands(frame)
    if len(bands) < 3:
        return None
    h, w = frame.shape[:2]
    shift = _layout_shift(frame, bands)
    name_details = []
    names = _read_sheet(_sheet(frame, bands, .689 + shift, .89 + shift, 800), len(bands), details=name_details)
    for i, (name, confidence) in enumerate(names):
        if confidence >= 65:
            continue
        a, b = bands[i]
        pad = max(2, round((b - a) * .1))
        cell = _binary(frame[a + pad:b - pad, int(w * (.689 + shift)):int(w * (.89 + shift))])
        cell = cv2.resize(cell, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cell = cv2.copyMakeBorder(cell, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        cell = _deslant(cell)
        data = pytesseract.image_to_data(cell, config="--psm 7", output_type=pytesseract.Output.DICT)
        words = [{"text": t.strip(), "conf": float(data["conf"][i]), "left": data["left"][i], "width": data["width"][i]}
                 for i, t in enumerate(data["text"]) if t.strip() and float(data["conf"][i]) >= 0]
        if words:
            words = correct_symbol_words(cell, words)
            fallback_confidence = float(np.mean([word["conf"] for word in words]))
            if fallback_confidence > confidence + 5:
                names[i] = (" ".join(word["text"] for word in words), fallback_confidence)
                name_details[i] = {"needs_review": any(word.get("font_uncertain", False) for word in words)}
    ranks = _read_sheet(_sheet(frame, bands, .606 + shift, .637 + shift, 100), len(bands), digits=True)
    parsed = [int(text) if text.isdigit() and 1 <= int(text) <= 24 else None for text, conf in ranks]
    # Only infer unreadable ranks when independent digit reads agree on the
    # same row grid. A conflicting reading is discarded, never propagated.
    step = float(np.median([b - a for a, b in bands])) + h * .003
    indices = [round((a - bands[0][0]) / step) for a, b in bands]
    origin = rank_origin(list(zip(indices, parsed)))
    if origin is None:
        return None
    first, support = origin
    results = []
    confidences = {}
    glyph_reviews = {}
    row_centers = {}
    for i, (name, confidence) in enumerate(names):
        rank = first + indices[i] if support >= 2 else parsed[i]
        if rank is not None and 1 <= rank <= 24:
            row_centers[rank] = (bands[i][0] + bands[i][1]) // 2
        if rank is None or not 1 <= rank <= 24 or len(name) < 2 or confidence < 25:
            continue
        # Preserve Unicode and legitimate punctuation; strip only border debris.
        name = re.sub(r"\s+", " ", name).strip()
        results.append((rank, name))
        confidences[rank] = confidence
        glyph_reviews[rank] = name_details[i]
    if len(results) < 3 or len({r for r, _ in results}) != len(results):
        return None
    # +points are yellow in the colored table; cumulative standings have no
    # yellow glyphs in this column. Inspect only inside detected result rows.
    plus_rows = _plus_rows(frame, bands, shift)
    return {"type": "race" if plus_rows >= 2 else "overall", "results": results,
            "confidences": confidences, "glyph_reviews": glyph_reviews,
            "row_centers": row_centers}


def read_team_grid_for_model(frame: np.ndarray, bands: list[tuple[int, int]] | None = None,
                             rank_reader=None, *, rank_only: bool = False) -> dict | None:
    """Locate colored result rows without reading names that the model replaces."""
    bands = _bands(frame) if bands is None else bands
    if len(bands) < 3:
        return None
    h, w = frame.shape[:2]
    shift = _layout_shift(frame, bands)
    plus_rows = _plus_rows(frame, bands, shift)
    if plus_rows < 2:
        return {"type": "overall"} if len(bands) >= 5 else None

    model_names = None
    if rank_reader is None:
        ranks = _read_sheet(_sheet(frame, bands, .606 + shift, .637 + shift, 100), len(bands), digits=True)
        parsed = [int(text) if text.isdigit() and 1 <= int(text) <= 24 else None
                  for text, _ in ranks]
        strong = [None] * len(parsed)
    else:
        from mktracker.detection.color_name_postprocess import trim_yellow_green_pattern
        rank_crops = [frame[a:b, int(w * (.606 + shift)):int(w * (.637 + shift))]
                      for a, b in bands]
        name_crops = []
        if not rank_only:
            for a, b in bands:
                center = (a + b) // 2
                y1 = max(0, round(center - h * 24 / 1080))
                y2 = min(h, round(center + h * 36 / 1080))
                crop = frame[y1:y2, round(w * (.68 + shift)):round(w * (.89 + shift))]
                name_crops.append(trim_yellow_green_pattern(crop))
        if getattr(rank_reader, "supports_narrow_ranks", False):
            ranks = rank_reader.predict_crops(rank_crops, width=64)
            names = rank_reader.predict_crops(name_crops) if name_crops else []
            predictions = ranks + names
        else:
            predictions = rank_reader.predict_crops(rank_crops + name_crops)
        if len(predictions) != len(bands) + len(name_crops):
            return None
        ranks = predictions[:len(bands)]
        model_names = predictions[len(bands):] if name_crops else None
        parsed = [int(text) if text.isdigit() and 1 <= int(text) <= 24 and conf >= .5 else None
                  for text, conf in ranks]
        strong = [int(text) if text.isdigit() and 1 <= int(text) <= 24 and conf >= .85 else None
                  for text, conf in ranks]
    step = float(np.median([b - a for a, b in bands])) + h * .003
    indices = [round((a - bands[0][0]) / step) for a, _ in bands]
    origin = rank_origin(list(zip(indices, parsed)))
    if origin is None:
        return None
    first, support = origin
    centers = {}
    precomputed = {}
    team_colors = {}
    # During the 24-player scroll, adjacent visible rows can move farther
    # apart for a frame. Geometry may then invent an empty slot even though
    # the rank model reads two consecutive numbers on those same rows. Keep
    # such independently corroborated digits; ignore isolated conflicts.
    direct = {
        i for i, rank in enumerate(strong) if rank is not None and (
            (i > 0 and strong[i - 1] == rank - 1) or
            (i + 1 < len(strong) and strong[i + 1] == rank + 1)
        )
    }
    assigned = [
        parsed[i] if i in direct or support < 2 else first + indices[i]
        for i in range(len(bands))
    ]
    duplicates = {rank for rank in assigned if rank is not None and
                  assigned.count(rank) > 1}
    for i, (a, b) in enumerate(bands):
        rank = assigned[i]
        if rank is not None and rank not in duplicates and 1 <= rank <= 24:
            centers[rank] = (a + b) // 2
            color = _band_team_color(frame, (a, b), shift)
            if color is not None:
                team_colors[rank] = color
            if model_names is not None:
                precomputed[rank] = model_names[i]
    if len(centers) < 3:
        return None
    result = {"type": "race", "results": [], "row_centers": centers,
            # Include the left edge of the clan tag. At .689 the italic L/V
            # starts outside the crop on native 1080p colored result rows.
            "name_crop_x": (.68 + shift, .89 + shift)}
    if model_names is not None:
        result["_model_name_predictions"] = precomputed
    if team_colors:
        result["team_colors"] = team_colors
    return result
