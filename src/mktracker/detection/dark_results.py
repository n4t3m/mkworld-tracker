"""Read dark translucent result tables using their row geometry."""
from __future__ import annotations

from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from mktracker.detection.font_symbols import correct_symbol_words
from mktracker.detection.result_grid import rank_origin


def has_dark_bar_contrast(frame: np.ndarray) -> bool:
    """Find the dark result overlay's left edge against the gameplay scene."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outside = np.median(gray[:, int(w*.50):int(w*.535)], axis=1)
    inside = np.median(gray[:, int(w*.557):int(w*.60)], axis=1)
    return bool(np.mean(outside - inside > 20) >= .30)


def _ocr(cell: np.ndarray, *, digits: bool = False, otsu: bool = False,
         bright: bool = False) -> tuple[str, float]:
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    if otsu:
        binary = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif bright:
        binary = ((hsv[:, :, 1] < 150) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    elif digits:
        binary = (cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) > 160).astype(np.uint8) * 255
    else:
        # White lettering stays legible on both dark bars and the yellow
        # highlighted player bar; a brightness-only mask loses that name.
        binary = ((hsv[:, :, 1] < 95) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
    binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    config = "--psm 7" + (" -c tessedit_char_whitelist=0123456789" if digits else "")
    data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
    words = [{"text": word.strip(), "conf": float(data["conf"][i]),
              "left": data["left"][i], "width": data["width"][i]}
             for i, word in enumerate(data["text"])
             if word.strip() and float(data["conf"][i]) >= 0]
    if not words:
        return "", 0.0
    original = " ".join(word["text"] for word in words)
    confidence = float(np.mean([word["conf"] for word in words]))
    if not digits and (confidence < 55 or len(original) <= 4):
        corrected = correct_symbol_words(255 - binary, words, deskewed=False)
        if corrected:
            name = corrected[0]["text"]
            if (corrected[0].get("font_uncertain") and len(original) <= 4
                    and any(not c.isalnum() and c not in ". _-" for c in original)):
                # A single compression edge can turn a short glyph into
                # punctuation. Require two independent mild smoothings to
                # agree before replacing that uncertain OCR character.
                alternatives = []
                for smoothed in (cv2.GaussianBlur(cell, (3, 3), 0),
                                 cv2.imdecode(cv2.imencode(
                                     ".jpg", cell, [cv2.IMWRITE_JPEG_QUALITY, 90])[1],
                                     cv2.IMREAD_COLOR)):
                    smooth_hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
                    smooth_binary = ((smooth_hsv[:, :, 1] < 95) &
                                     (smooth_hsv[:, :, 2] > 180)).astype(np.uint8) * 255
                    smooth_binary = cv2.resize(smooth_binary, None, fx=3, fy=3,
                                               interpolation=cv2.INTER_CUBIC)
                    read = correct_symbol_words(255 - smooth_binary, words,
                                                deskewed=False)
                    if read and not read[0].get("font_uncertain"):
                        alternatives.append(read[0]["text"])
                if len(alternatives) == 2 and alternatives[0] == alternatives[1]:
                    name = alternatives[0]
                    corrected[0]["font_uncertain"] = False
            supported = (not corrected[0].get("font_uncertain", False) or
                         (confidence < 45 and SequenceMatcher(None, original, name).ratio() >= .65))
            if name != original and supported:
                floor = 70.0 if corrected[0].get("font_verified", False) else 40.0
                return name.strip(" |;"), max(confidence, floor)
    return original.strip(" |;"), confidence


def _highlighted_band(frame: np.ndarray) -> tuple[int, int] | None:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 100, 140]), np.array([45, 255, 255]))
    coverage = (yellow[:, int(w*.56):int(w*.94)] > 0).mean(axis=1) > .5
    transitions = np.diff(np.r_[False, coverage, False].astype(int))
    bands = [(a, b) for a, b in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0])
             if h*.045 <= b-a <= h*.085]
    if len(bands) != 1:
        return None
    return bands[0]


def read_highlighted_row(frame: np.ndarray) -> tuple[int, str, float, float] | None:
    """Read the yellow player row, which can stay pinned during scrolling."""
    band = _highlighted_band(frame)
    if band is None:
        return None
    h, w = frame.shape[:2]
    a, b = band
    rank_cell = cv2.cvtColor(frame[a+4:b-6, int(w*.573):int(w*.612)], cv2.COLOR_BGR2GRAY)
    rank_binary = cv2.threshold(rank_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    rank_binary = cv2.resize(rank_binary, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    rank_data = pytesseract.image_to_data(
        rank_binary, config="--psm 10 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT)
    rank_reads = [(word.strip(), float(rank_data["conf"][i]))
                  for i, word in enumerate(rank_data["text"]) if word.strip()]
    if len(rank_reads) != 1:
        return None
    rank_text, rank_confidence = rank_reads[0]
    if not rank_text.isdigit() or not 1 <= int(rank_text) <= 24:
        return None
    name, confidence = _ocr(frame[a:b, int(w*.66):int(w*.83)])
    if len(name) < 2 or confidence < 20:
        return None
    return int(rank_text), name, confidence, rank_confidence


def add_highlighted_row(result: dict, frame: np.ndarray) -> dict:
    if result.get("type") != "race":
        return result
    highlighted = read_highlighted_row(frame)
    if highlighted is None:
        return result
    rank, name, confidence, rank_confidence = highlighted
    centers = result.get("row_centers", {})
    band = _highlighted_band(frame)
    source = "ocr_confident" if rank_confidence >= 65 else "ocr"
    if centers and band:
        middle = (band[0] + band[1]) / 2
        nearest = min(centers, key=lambda candidate: abs(centers[candidate] - middle))
        if (abs(centers[nearest] - middle) <= frame.shape[0] * .035
                and (nearest == rank or rank_confidence < 65)):
            rank = nearest
            source = "grid"
    result["highlighted_rank"] = rank
    result["highlighted_rank_source"] = source
    if rank not in {placement for placement, _ in result["results"]}:
        result["results"].append((rank, name))
        result["results"].sort()
        result.setdefault("confidences", {})[rank] = confidence
    return result


def read_dark_results(frame: np.ndarray, starts: list[int]) -> dict | None:
    """Infer the visible rank grid from repeated +points, then OCR each row."""
    h, w = frame.shape[:2]
    step = round(h * 77 / 1080)
    tolerance = max(8, round(step * .17))
    if len(starts) < 4:
        return None
    phases = [y % step for y in starts]
    phase = max(phases, key=lambda p: sum(min(abs(q-p), step-abs(q-p)) <= tolerance for q in phases))
    matching = [y for y in starts if min(abs(y % step-phase), step-abs(y % step-phase)) <= tolerance]
    if len(matching) < 4:
        return None
    # Refine the phase with the matching observations, including those that
    # wrap around zero. The row crop is wide enough for small animation drift.
    offsets = [((y - phase + step // 2) % step) - step // 2 for y in matching]
    phase = round((phase + np.median(offsets)) % step)
    rows = []
    for index, y in enumerate(range(phase, h, step)):
        top, bottom = max(0, y - round(step * .42)), min(h, y + round(step * .56))
        if bottom - top < step * .6:
            continue
        rank_text, rank_conf = _ocr(frame[top:bottom, int(w*.575):int(w*.615)], digits=True)
        name_cell = frame[top:bottom, int(w*.66):int(w*.83)]
        name, confidence = _ocr(name_cell)
        if confidence < 25 and any(abs(y - observed) <= tolerance for observed in matching):
            for alternate_mode in ({"bright": True}, {"otsu": True}):
                alternate, alternate_confidence = _ocr(name_cell, **alternate_mode)
                if alternate_confidence >= 25 and alternate_confidence > confidence:
                    name, confidence = alternate, alternate_confidence
        rank = int(rank_text) if rank_text.isdigit() and 1 <= int(rank_text) <= 24 else None
        rows.append((index, rank, rank_conf, name.strip(), confidence))
    origin = rank_origin([(index, rank) for index, rank, conf, _, _ in rows
                          if conf >= 25], min_support=2)
    if origin is None:
        return None
    first, support = origin
    results = []
    confidences = {}
    for index, _, _, name, confidence in rows:
        rank = first + index
        expected_y = phase + index * step
        has_plus = any(abs(expected_y - observed) <= tolerance for observed in matching)
        min_confidence = 25 if has_plus else 60
        if 1 <= rank <= 24 and len(name) >= 2 and confidence >= min_confidence:
            results.append((rank, name))
            confidences[rank] = confidence
    if len(results) < 5:
        return None
    centers = {first + index: phase + index * step for index, *_ in rows
               if 1 <= first + index <= 24}
    return {"type": "race", "results": results, "confidences": confidences,
            "row_centers": centers}
