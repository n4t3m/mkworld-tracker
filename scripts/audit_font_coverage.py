"""Exhaustively exercise the configured font, including unseen tag glyphs."""
import argparse
import json
from pathlib import Path
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mktracker.detection.font_decoder import FontDecoder
from mktracker.detection.font_symbols import configured_font_path

parser = argparse.ArgumentParser()
parser.add_argument("--font", default=None)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sizes", default="64")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
path = args.font or configured_font_path()
decoder = FontDecoder(path)
args.output.mkdir(parents=True, exist_ok=True)
report = {"font": decoder.catalog.summary(), "sizes": {}, "cases": []}
started = time.perf_counter()
characters = list(decoder.catalog.glyphs)[:args.limit]
for size in map(int, args.sizes.split(",")):
    font = ImageFont.truetype(path, size)
    correct = 0
    confident_errors = 0
    recovered_without_ocr = 0
    for i, char in enumerate(characters):
        text = "A"+char+"A"
        x0, y0, x1, y1 = font.getbbox(text)
        image = Image.new("L", (x1-x0+80, y1-y0+80), 255)
        ImageDraw.Draw(image).text((40-x0, 40-y0), text, font=font, fill=0)
        reads = decoder.segment(np.array(image))
        expected = decoder.canonical(text)
        actual = "".join(r.char for r in reads)
        passed = expected == actual
        correct += passed
        verified = all(r.verified for r in reads)
        confident_errors += not passed and verified
        recovered = decoder.correct_words(np.array(image), [])
        recovered_without_ocr += bool(recovered) and decoder.canonical(recovered[0]["text"]) == expected
        report["cases"].append({"codepoint": f"U+{ord(char):04X}", "char": char, "name": unicodedata.name(char, "private-use"),
                                "size": size, "expected": expected, "actual": actual, "passed": passed, "verified": verified,
                                "scores": [round(r.score, 3) for r in reads], "margins": [round(r.margin, 3) for r in reads]})
        if not passed:
            image.save(args.output / f"failure-{size}-{ord(char):04x}.png")
        if (i+1) % 100 == 0:
            print(f"{size}px: {i+1}/{len(characters)} tested, {correct} matched", flush=True)
            (args.output / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["sizes"][str(size)] = {"correct": correct, "total": len(characters), "confident_errors": confident_errors,
                                 "recovered_without_ocr": recovered_without_ocr}
    print(f"{size}px: {correct}/{len(characters)} matched", flush=True)
report["seconds"] = round(time.perf_counter()-started, 2)
(args.output / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Audit saved to {args.output}; {report['seconds']} seconds", flush=True)
