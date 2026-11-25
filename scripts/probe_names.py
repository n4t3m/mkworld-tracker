"""Inspect custom-room result OCR at selected timestamps in a recording."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from mktracker.detection.team_results import _bands, _deslant, _sheet, read_team_results


def timestamp(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('Timestamps must be seconds, e.g. 243 or 243.5.') from error
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError('Timestamps must be finite and nonnegative.')
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='Gameplay recording')
    parser.add_argument('--times', type=timestamp, nargs='+', required=True, help='One or more timestamps in seconds')
    parser.add_argument('--output', type=Path, required=True, help='Directory for screenshots, row sheets and readings.json')
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            parser.error(f'Cannot open video: {args.video}')
        args.output.mkdir(parents=True, exist_ok=True)
        report = []
        for index, seconds in enumerate(args.times, 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds*1000)
            ok, frame = cap.read()
            if not ok:
                parser.error(f'Cannot read a frame at {seconds:g} seconds; check recording duration.')
            frame = cv2.resize(frame, (1920, 1080))
            stem = f'{index:03d}-{seconds:.6f}s'
            if not cv2.imwrite(str(args.output / f'{stem}.jpg'), frame):
                raise OSError(f'Could not save frame at {seconds:g}s')
            bands = _bands(frame)
            if bands:
                sheet = _sheet(frame, bands, .689, .89, 800)
                for row in range(len(bands)):
                    sheet[row*160:(row+1)*160] = _deslant(sheet[row*160:(row+1)*160])
                if not cv2.imwrite(str(args.output / f'{stem}-names.png'), sheet):
                    raise OSError(f'Could not save row sheet at {seconds:g}s')
            result = read_team_results(frame)
            report.append({'seconds': seconds, 'bands': bands, 'result': result})
            (args.output / 'readings.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{seconds:g}s: {len(bands)} row bands; {ascii(result)}', flush=True)
    finally:
        cap.release()


if __name__ == '__main__':
    main()
