"""List visual boundary candidates in a sampled video (timestamps are source seconds)."""
import json
import sys
from pathlib import Path

import cv2

from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.track_select import TrackSelectDetector

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
track = TrackSelectDetector()
finish = RaceFinishDetector()
results = RaceResultDetector()
events = []
i = 0
last = {"panel": -100, "finish": -100, "results": -100}
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    i += 1
    if i % max(1, round(fps / 2)):
        continue
    full = cv2.resize(frame, (1920, 1080))
    for kind, active in [("panel", track.has_player_panel(full)),
                         ("finish", finish.is_active(full)),
                         ("results", results.has_race_results(full))]:
        if active and t - last[kind] > 15:
            event = {"time": t, "kind": kind}
            if kind == "panel":
                event["track"] = track.read_track_name(full)
            events.append(event)
            print(event, flush=True)
            cv2.imwrite(str(out / f"{t:07.1f}-{kind}.jpg"), frame)
        if active:
            last[kind] = t
    if i % 240 == 0:
        (out / "events.json").write_text(json.dumps(events, indent=2))
cap.release()
(out / "events.json").write_text(json.dumps(events, indent=2))
