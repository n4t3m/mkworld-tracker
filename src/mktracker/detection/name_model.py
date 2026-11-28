"""Optional local model reader for Video Review result-board names."""
from __future__ import annotations

import base64
from difflib import SequenceMatcher
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from dotenv import dotenv_values

from mktracker.detection.font_decoder import FontDecoder
from mktracker.detection.font_symbols import configured_font_path
from mktracker.detection.color_name_postprocess import (
    correct_colored_board, trim_yellow_green_pattern,
)
from mktracker.detection.team_results import _binary

logger = logging.getLogger(__name__)
_ROOT = Path.cwd()


def _setting(key: str) -> str:
    return os.environ.get(key) or dotenv_values(_ROOT / ".env").get(key) or ""


def _font_binary(image: np.ndarray) -> np.ndarray:
    """Return dark glyphs on white for both bright and faded result text."""
    binary = _binary(image)
    border = np.concatenate((binary[:2, :].ravel(), binary[-2:, :].ravel(),
                             binary[:, :2].ravel(), binary[:, -2:].ravel()))
    return 255 - binary if np.mean(border < 127) > .6 else binary


def _correct_zero(image: np.ndarray, prediction: str, decoder: FontDecoder | None) -> str:
    if decoder is None or "o" not in prediction or not prediction.isascii():
        return prediction
    binary = _font_binary(image)
    reads = decoder.segment(binary)
    if len(reads) != len(prediction):
        return prediction
    corrected = list(prediction)
    for index, char in enumerate(prediction):
        if char != "o" or reads[index].char not in ("0", "０", "O", "Ｏ"):
            continue
        read = reads[index]
        mask = binary[read.top:read.top + read.height,
                      read.left:read.left + read.width] < 127
        scores = decoder._shape_scores(mask)
        def best(chars: str) -> float:
            return max(float(scores[i]) for i, glyph in enumerate(decoder.chars)
                       if glyph in chars)
        zero, lower, upper = best("0０"), best("oｏ"), best("OＯ")
        if zero >= .8 and zero - lower >= .035:
            corrected[index] = "0" if zero - upper >= .003 else "O"
    return "".join(corrected)


def _correct_short_one(image: np.ndarray, prediction: str,
                       decoder: FontDecoder | None) -> str:
    """Resolve a short digit/letter run only when the I glyph wins clearly."""
    if decoder is None or not re.fullmatch(r"[A-Z0-9]{3,4}", prediction):
        return prediction
    targets = [index for index in range(1, len(prediction))
               if prediction[index] == "1" and prediction[index - 1].isdigit()]
    if not targets:
        return prediction
    for left in range(0, min(25, image.shape[1] // 3), 2):
        binary = _font_binary(image[:, left:])
        reads = decoder.segment(binary)
        if len(reads) != len(prediction):
            continue
        corrected = list(prediction)
        for index in targets:
            read = reads[index]
            if read.width < 5 or read.height < 15:
                continue
            mask = binary[read.top:read.top + read.height,
                          read.left:read.left + read.width] < 127
            scores = decoder._shape_scores(mask, aspect_tolerance=.7,
                                           force_chars="1I")
            one = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                      if char == "1")
            letter = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                         if char == "I")
            if letter >= .65 and letter - one >= .12:
                corrected[index] = "I"
        return "".join(corrected)
    return prediction


def _recover_short_missing_i(image: np.ndarray, prediction: str,
                             decoder: FontDecoder | None) -> str:
    """Insert a visible capital I between two recognized digits."""
    if decoder is None or not re.fullmatch(r"[0-9]{2}", prediction):
        return prediction
    for left in range(0, min(25, image.shape[1] // 3), 2):
        binary = _font_binary(image[:, left:])
        reads = decoder.segment(binary)
        if len(reads) != 3:
            continue
        middle = reads[1]
        if min(read.height for read in reads) < 15 or middle.width < 5:
            continue
        mask = binary[middle.top:middle.top + middle.height,
                      middle.left:middle.left + middle.width] < 127
        scores = decoder._shape_scores(mask, aspect_tolerance=.7,
                                       force_chars="1I")
        one = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                  if char == "1")
        letter = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                     if char == "I")
        if letter >= .68 and letter - one >= .12:
            return prediction[0] + "I" + prediction[1]
        break
    return prediction


def _correct_dotted_initial_zero(image: np.ndarray, prediction: str,
                                 decoder: FontDecoder | None) -> str:
    """Read a near-tied round glyph as O in a dotted initialism."""
    if decoder is None or not re.fullmatch(r"(?:[A-Z]\.0\.|[A-Z]0\.|\.0\.|0\.)",
                                            prediction):
        return prediction
    for left in range(0, min(45, image.shape[1] // 3), 2):
        binary = _font_binary(image[:, left:])
        reads = decoder.segment(binary)
        if len(reads) != 4 or not re.fullmatch(r"[A-Z]", reads[0].char):
            continue
        if prediction[0].isalpha() and not decoder.equivalent(reads[0].char,
                                                               prediction[0]):
            continue
        read = reads[2]
        if (reads[0].score < .65 or reads[1].height > read.height * .35
                or reads[3].height > read.height * .35):
            continue
        mask = binary[read.top:read.top + read.height,
                      read.left:read.left + read.width] < 127
        scores = decoder._shape_scores(mask)
        zero = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                   if char in "0０")
        letter = max(float(scores[i]) for i, char in enumerate(decoder.chars)
                     if char in "OＯ")
        if letter >= .82 and zero - letter <= .02:
            return reads[0].char + ".O."
        break
    return prediction


def _recover_short_tagged_tail(image: np.ndarray, prediction: str) -> str:
    """Use strong row OCR to complete a model output ending inside a name.

    The board must already establish its two-letter tag before this is called.
    Keep the model's tag spelling; Tesseract is only allowed to supply a longer
    final word that agrees with the model's visible suffix.
    """
    if len(prediction) < 2 or not prediction[:2].isalpha():
        return prediction
    tail = prediction[2:].lstrip(" ²Σ'’′")
    if len(tail) > 2:
        return prediction
    binary = cv2.resize(_binary(image), None, fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(
            binary, config="--psm 7", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        return prediction
    words = sorted(((str(word).strip(), float(data["conf"][i]), int(data["left"][i]))
                    for i, word in enumerate(data["text"])
                    if any(char.isalpha() for char in str(word))),
                   key=lambda item: item[2])
    if len(words) < 2 or not words[0][0][:2].casefold() == prediction[:2].casefold():
        return prediction
    candidate, confidence, _ = words[-1]
    threshold = 75 if tail else 80
    if confidence < threshold or len(candidate) < 4 or not candidate.isalpha():
        return prediction
    expected = tail.casefold()
    observed = candidate.casefold()
    characters = iter(observed)
    if expected and (len(observed) <= len(expected) or
                     not all(any(actual == character for actual in characters)
                             for character in expected)):
        return prediction
    return prediction[:2] + " " + candidate


def _restore_visible_word_gap(image: np.ndarray, prediction: str) -> str:
    """Insert a space when both pixels and row OCR show a word boundary.

    This keeps the model's letter choices. The row OCR only proposes where to
    split, after a large gap among real letter components in the image.
    """
    if not prediction.isascii() or not prediction.isalpha() or len(prediction) < 5:
        return prediction
    binary = _binary(image)
    _, _, stats, _ = cv2.connectedComponentsWithStats((binary < 127).astype(np.uint8), 8)
    components = sorted((int(x), int(width)) for x, _, width, height, area in stats[1:]
                        if 10 <= x <= image.shape[1] * .8 and height >= 12 and area >= 25)
    if len(components) < 4 or max((components[i+1][0] - components[i][0] - components[i][1]
                                   for i in range(len(components)-1)), default=0) < 7:
        return prediction
    enlarged = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        data = pytesseract.image_to_data(
            enlarged, config="--psm 7", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        return prediction
    words = sorted(((str(word).strip(), float(data["conf"][i]), int(data["left"][i]))
                    for i, word in enumerate(data["text"])
                    if str(word).strip().isalpha()), key=lambda item: item[2])
    if len(words) != 2 or min(word[1] for word in words) < 70:
        return prediction
    first, second = words[0][0], words[1][0]
    if (prediction[:len(first)].casefold() != first.casefold() or
            len(prediction) - len(first) != len(second) or
            SequenceMatcher(None, prediction[len(first):].casefold(),
                            second.casefold()).ratio() < .6):
        return prediction
    return prediction[:len(first)] + " " + prediction[len(first):]


class NameModelReader:
    """Keep one external Paddle worker loaded across result frames."""

    supports_narrow_ranks = True

    def __init__(self, *, color: bool = False) -> None:
        self.process: subprocess.Popen | None = None
        self.disabled = False
        self.color = color
        self.decoder: FontDecoder | None = None
        font = configured_font_path()
        if font and Path(font).is_file():
            try:
                self.decoder = FontDecoder(font)
            except (OSError, ValueError):
                logger.warning("Could not load game font for name model correction")

    @staticmethod
    def available() -> bool:
        paths = [_setting(key) for key in ("NAME_OCR_PYTHON", "NAME_OCR_CONFIG",
                                            "NAME_OCR_CHECKPOINT", "NAME_OCR_PADDLEOCR")]
        return all(path and Path(path).exists() for path in paths)

    @staticmethod
    def color_available() -> bool:
        return NameModelReader.available() and all(
            _setting(key) and Path(_setting(key)).exists()
            for key in ("NAME_OCR_COLOR_CONFIG", "NAME_OCR_COLOR_CHECKPOINT")
        )

    def _start(self) -> bool:
        if self.disabled or not (self.color_available() if self.color else self.available()):
            return False
        if self.process is not None:
            return True
        try:
            config_key = "NAME_OCR_COLOR_CONFIG" if self.color else "NAME_OCR_CONFIG"
            checkpoint_key = "NAME_OCR_COLOR_CHECKPOINT" if self.color else "NAME_OCR_CHECKPOINT"
            self.process = subprocess.Popen(
                [_setting("NAME_OCR_PYTHON"), str(Path(__file__).with_name("name_ocr_worker.py")),
                 "--config", _setting(config_key),
                 "--checkpoint", _setting(checkpoint_key),
                 "--paddleocr", _setting("NAME_OCR_PADDLEOCR")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=_ROOT,
            )
            self._reply()  # Wait for the checkpoint to load.
            return True
        except (OSError, RuntimeError, ValueError) as error:
            logger.warning("Name OCR model unavailable: %s", error)
            self.close()
            self.disabled = True
            return False

    def _reply(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("MKW_OCR:"):
                response = json.loads(line[len("MKW_OCR:"):])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response
        raise RuntimeError("Name OCR worker exited")

    def predict_crops(self, crops: list[np.ndarray], *, width: int = 320) -> list[tuple[str, float]]:
        """Recognize a batch of raw name cells with the selected checkpoint."""
        if not crops or not self._start():
            return []
        try:
            payload = {"width": width,
                       "crops": [base64.b64encode(cv2.imencode(".png", crop)[1]).decode("ascii")
                                 for crop in crops]}
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            predictions = self._reply()["results"]
            if len(predictions) != len(crops):
                raise RuntimeError("Name OCR worker returned the wrong number of rows")
            return [(str(name), float(confidence)) for name, confidence in predictions]
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            logger.warning("Name OCR model read failed: %s", error)
            self.close()
            self.disabled = True
            return []

    def read(self, frame: np.ndarray, result: dict) -> dict:
        """Replace legacy names using every located row, retaining rank evidence."""
        if result.get("type") != "race" or not result.get("row_centers") or not self._start():
            return result
        h, w = frame.shape[:2]
        rows = []
        for rank, center in sorted(result["row_centers"].items()):
            if not 1 <= rank <= 24 or not 0 <= center < h:
                continue
            y1, y2 = max(0, round(center - h * 24 / 1080)), min(h, round(center + h * 36 / 1080))
            crop_x = result.get("name_crop_x")
            if crop_x is None:
                x1, x2 = round(w * 1267 / 1920), round(w * 1667 / 1920)
            else:
                x1, x2 = round(w * crop_x[0]), round(w * crop_x[1])
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                rows.append((rank, crop))
        if not rows:
            return result
        colored = result.get("name_crop_x") is not None
        cached = result.get("_model_name_predictions")
        predictions = ([cached[rank] for rank, _ in rows]
                       if cached is not None and all(rank in cached for rank, _ in rows)
                       else self.predict_crops([
                           trim_yellow_green_pattern(crop) if colored else crop
                           for _, crop in rows
                       ], width=320))
        if len(predictions) != len(rows):
            return result
        updated = dict(result)
        updated.pop("_model_name_predictions", None)
        names = dict(result["results"])
        confidences = dict(result.get("confidences", {}))
        adjusted = (correct_colored_board(
            [(crop, name.strip()) for (_, crop), (name, _) in zip(rows, predictions)], self.decoder)
            if colored else [name.strip() for name, _ in predictions])
        if colored:
            tagged = {name[:2].casefold() for name in adjusted
                      if len(name) >= 3 and name[:2].isalpha() and name[2] in "²Σ"}
            recovered = [(_recover_short_tagged_tail(crop, name)
                          if name[:2].casefold() in tagged else name)
                         for (_, crop), name in zip(rows, adjusted)]
            if recovered != adjusted:
                adjusted = correct_colored_board(
                    [(crop, name) for (_, crop), name in zip(rows, recovered)], self.decoder)
            adjusted = [_restore_visible_word_gap(crop, name)
                        for (_, crop), name in zip(rows, adjusted)]
        glyph_reviews = dict(result.get("glyph_reviews", {}))
        for (rank, crop), (raw_name, confidence), name in zip(rows, predictions, adjusted):
            name = _correct_zero(crop, name.strip(), self.decoder)
            font_checked = name
            name = _correct_short_one(crop, name, self.decoder)
            name = _recover_short_missing_i(crop, name, self.decoder)
            name = _correct_dotted_initial_zero(crop, name, self.decoder)
            if (len(name) >= 2 and confidence >= .3) or (len(name) == 1 and confidence >= .8):
                names[rank] = name
                confidences[rank] = confidence * 100
                if name != font_checked:
                    glyph_reviews[rank] = {"needs_review": True}
                if colored and any(symbol in name and symbol not in raw_name
                                   for symbol in ("²", "Σ")):
                    # A restored symbol needs corroboration. Raised 2 and
                    # prime in particular can look alike at video resolution.
                    glyph_reviews[rank] = {"needs_review": True}
        updated["results"] = sorted(names.items())
        updated["confidences"] = confidences
        updated["glyph_reviews"] = glyph_reviews
        return updated

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None and self.process.stdin is not None:
                try:
                    self.process.stdin.write('{"shutdown": true}\n')
                    self.process.stdin.flush()
                    self._reply()
                except (OSError, RuntimeError, ValueError):
                    pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
