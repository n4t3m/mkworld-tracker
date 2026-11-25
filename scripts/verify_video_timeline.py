"""Verify cached boundaries against the original video and render review sheets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mktracker.capture.video_source import VideoCapture
from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.video_analysis import has_result_table, read_course
from mktracker.video_timeline import VideoTimeline, timestamp

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("--expected-races", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--check-ui", action="store_true")
args = parser.parse_args()
capture = VideoCapture()
assert capture.open_file(args.source)
timeline = VideoTimeline.load(args.source, capture.duration)
assert timeline.complete, "Analysis is incomplete"
assert len(timeline.races) == args.expected_races
args.output.mkdir(parents=True, exist_ok=True)
finish_detector = RaceFinishDetector()
rows = []
report = []
previous_finish = -1
for number, race in enumerate(timeline.races, 1):
    assert race.finish is not None and previous_finish < race.start < race.finish
    previous_finish = race.finish
    tiles = []
    checks = {"race": number, "course": race.track}
    for kind, seconds in [("preview", race.start), ("finish", race.finish), ("results", race.results)]:
        assert seconds is not None, f"Race {number}: missing {kind}"
        capture.seek(seconds)
        frame = capture.read_frame()
        assert frame is not None
        normalized = cv2.resize(frame, (1920, 1080))
        if kind == "preview":
            observed = read_course(normalized)
            assert observed == race.track, (number, observed, race.track)
        elif kind == "finish":
            assert finish_detector.is_active(normalized), f"Race {number}: finish not visible"
        else:
            assert has_result_table(normalized), f"Race {number}: results table not visible"
        checks[kind] = seconds
        tile = cv2.resize(frame, (640, 360))
        cv2.rectangle(tile, (0, 328), (640, 360), (0, 0, 0), -1)
        cv2.putText(tile, f"{number}: {kind} {timestamp(seconds)}", (8, 350), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        tiles.append(tile)
    rows.append(np.hstack(tiles))
    report.append(checks)
    print(checks, flush=True)
capture.close()
for start in range(0, len(rows), 4):
    cv2.imwrite(str(args.output / f"races-{start+1:02d}-{min(start+4,len(rows)):02d}.jpg"), np.vstack(rows[start:start+4]))
(args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Verified {len(report)} races against the original video.")

if args.check_ui:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from mktracker.ui.main_window import MainWindow
    app = QApplication([])
    if Path("C:/Windows/Fonts/segoeui.ttf").exists():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 10))
    window = MainWindow(initial_video=args.source)
    window.show()
    app.processEvents()
    try:
        assert window._mode_combo.currentText() == "Custom Room"
        assert window._timeline.complete
        assert window._timeline_list.count() == len(report) * 3
        for i in range(window._timeline_list.count()):
            item = window._timeline_list.item(i)
            window._timeline_list.scrollToItem(item)
            app.processEvents()
            QTest.mouseClick(window._timeline_list.viewport(), Qt.MouseButton.LeftButton,
                             pos=window._timeline_list.visualItemRect(item).center())
            target = item.data(Qt.ItemDataRole.UserRole)
            assert abs(window._capture.position - target) <= .1
            assert window._paused
            assert len(window._timeline.races) == args.expected_races
        window._seek_video(timeline.races[0].finish)
        window._timeline_list.scrollToTop()
        app.processEvents()
        window.grab().save(str(args.output / "review-window.png"))
        print(f"Clicked and verified all {window._timeline_list.count()} GUI timeline entries.")
    finally:
        window.close()
