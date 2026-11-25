"""Inspect result-row glyphs using the application's complete font recognizer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from mktracker.detection.font_decoder import FontDecoder
from mktracker.detection.font_symbols import configured_font_path
from mktracker.detection.team_results import _bands, _deslant, _sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='Full result-screen screenshot')
    parser.add_argument('--output', type=Path, required=True, help='Directory for row images and glyphs.json')
    parser.add_argument('--font', type=Path, help='Game TTF/OTF; defaults to GAME_FONT_PATH or Settings')
    parser.add_argument('--rows', type=int, nargs='+', help='One-based visible row indices; default: all rows')
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f'Cannot read image: {args.image}')
    font = args.font or configured_font_path()
    if not font or not Path(font).is_file():
        parser.error('Select a local game font with --font or Settings > Game font.')
    try:
        decoder = FontDecoder(str(font))
    except (OSError, ValueError) as error:
        parser.error(f'Cannot load font: {error}')
    frame = cv2.resize(frame, (1920, 1080))
    bands = _bands(frame)
    if not bands:
        parser.error('No custom-room team result rows detected in the image.')
    rows = args.rows if args.rows is not None else list(range(1, len(bands)+1))
    if any(row < 1 or row > len(bands) for row in rows):
        parser.error(f'--rows must be between 1 and {len(bands)} (visible row indices, not racer ranks).')
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = _sheet(frame, bands, .689, .89, 800)
    report = []
    for row in dict.fromkeys(rows):
        image = _deslant(sheet[(row-1)*160:row*160])
        reads = decoder.segment(image)
        entry = {'row': row, 'band': bands[row-1],
                 'glyph_text': ''.join(read.char for read in reads),
                 'font_only_read': decoder.correct_words(image, []),
                 'glyphs': [dict(asdict(read), verified=read.verified) for read in reads]}
        report.append(entry)
        if not cv2.imwrite(str(args.output / f'row-{row:02d}.png'), image):
            raise OSError(f'Could not save row {row}')
        print(f'Row {row}: {ascii(entry["glyph_text"])}; {sum(r.verified for r in reads)}/{len(reads)} verified')
    (args.output / 'glyphs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
