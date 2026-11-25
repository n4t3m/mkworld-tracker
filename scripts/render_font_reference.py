"""Render local game-font references without bundling the font in the repo."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("font", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype(str(args.font), 48)
samples = ["Vy²Barrett", "Vy²Parrett", "ArΣ★Rascal", "ArZ Rascal", "Ly Bert / Ly Bere", "B P  Σ Z  ★  ²  TC!"]
sheet = Image.new("RGB", (1100, len(samples) * 85 + 20), "white")
draw = ImageDraw.Draw(sheet)
for i, sample in enumerate(samples):
    bounds = font.getbbox(sample)
    draw.text((20 - bounds[0], 10 + i * 85 - bounds[1]), sample, font=font, fill="black")
sheet.save(args.output / "game-font-reference.png")
metadata = {"source": str(args.font.resolve()), "family": font.getname()[0], "style": font.getname()[1],
            "samples": samples,
            "glyphs": {c: bytes(font.getmask(c)) != bytes(font.getmask(chr(0xffff))) for c in "BPΣZ★²"}}
(args.output / "font-reference.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(args.output / "game-font-reference.png")
