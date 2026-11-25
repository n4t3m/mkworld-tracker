"""Local, cancellable analysis of recorded custom-room sessions."""
from __future__ import annotations

import re
from typing import Callable

import cv2
import numpy as np
import pytesseract

from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.name_consensus import PlacementConsensus
from mktracker.detection.preview_roster import PreviewRoster
from mktracker.detection.track_select import TrackSelectDetector
from mktracker.detection.tracks import course_text_candidates, match_track_name
from mktracker.video_timeline import VideoRace, VideoTimeline


def read_course(frame: np.ndarray) -> str | None:
    """Read the course banner on the map; reject multiple voting choices."""
    h, w = frame.shape[:2]
    # In free roam the *previous* course name is displayed at the very
    # top-right. It must never become the next race's course preview.
    roi = frame[int(h * .12):int(h * .90), int(w * .60):]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    data = pytesseract.image_to_data(binary, config="--psm 11", output_type=pytesseract.Output.DICT)
    grouped: dict[tuple, list[tuple[str, float]]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append((word.strip(), float(data["conf"][i])))
    lines = [" ".join(word for word, conf in words) for words in grouped.values()]
    candidates = course_text_candidates(lines)
    names = set()
    for candidate in candidates:
        match = match_track_name(candidate, cutoff=.86)
        if match:
            names.add(match[0])
    if names:
        return names.pop() if len(names) == 1 else None
    # Preserve readable retro labels absent from the canonical course list.
    labels = [line for line, words in zip(lines, grouped.values())
              if re.match(r"^(?:SNES|N64|GBA|GCN|DS|Wii|3DS)\s", line, re.IGNORECASE)
              and len(words) >= 3 and min(conf for _, conf in words[1:]) >= 75
              and re.fullmatch(r"(?:[A-Za-z0-9][\w?'’.-]*\s+){1,6}[A-Za-z0-9][\w?'’.-]*", line)]
    if len(labels) == 1:
        return re.sub(r"^snes\b", "SNES", labels[0], flags=re.IGNORECASE)
    return None


def has_result_table(frame: np.ndarray) -> bool:
    """Recognize several broad red/blue result rows after a finish."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame[:, int(w * .59):int(w * .99)], cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = (((hue < 12) | (hue > 170) | ((hue > 95) & (hue < 125))) & (sat > 140) & (val > 100))
    rows = mask.mean(axis=1) > .65
    transitions = np.diff(np.r_[False, rows, False].astype(int))
    starts, ends = np.where(transitions == 1)[0], np.where(transitions == -1)[0]
    return int(((ends - starts >= h * .012) & (ends - starts <= h * .10)).sum()) >= 4


def analyze_video(
    source: str,
    *,
    progress: Callable[[VideoTimeline], None] = lambda timeline: None,
    cancelled: Callable[[], bool] = lambda: False,
    analysis_source: str | None = None,
) -> VideoTimeline:
    cap = cv2.VideoCapture(analysis_source or source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    timeline = VideoTimeline.create(source, count / fps)
    finish_detector = RaceFinishDetector()
    result_detector = RaceResultDetector()
    sample_step = max(1, round(fps / 2))
    last_course_check = -100.0
    last_result_check = -100.0
    last_progress = -100.0
    current: VideoRace | None = None
    name_votes = PlacementConsensus()
    results_done = False
    overall_streak = 0
    preview_roster = PreviewRoster()
    last_preview_check = -100.0
    frame_index = -1
    try:
        while not cancelled():
            if not cap.grab():
                if count > 0 and frame_index + 2 < count:
                    raise RuntimeError(f"Video ended unexpectedly at {frame_index / fps:.1f}s")
                timeline.complete = True
                timeline.analyzed_until = timeline.duration
                break
            frame_index += 1
            in_results = current is not None and current.finish is not None and not results_done and frame_index / fps - current.finish <= 35
            step = max(1, round(fps / 4)) if in_results else sample_step
            if frame_index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Cannot decode video at {frame_index / fps:.1f}s")
            t = frame_index / fps
            frame = cv2.resize(frame, (1920, 1080))
            timeline.analyzed_until = t
            if t - last_course_check >= 3 and (current is None or current.finish is not None):
                if TrackSelectDetector.has_player_panel(frame):
                    last_course_check = t
                    course = read_course(frame)
                    if course:
                        current = VideoRace(start=t, track=course,
                                            review_note="Start marks the course preview; boundaries sampled every 0.5 seconds.")
                        timeline.races.append(current)
                        preview_roster = PreviewRoster()
                        last_preview_check = -100.0
                        timeline.save()
            if current and current.finish is None and t - current.start <= 3 and t - last_preview_check >= .5:
                last_preview_check = t
                if TrackSelectDetector.has_player_panel(frame):
                    preview_roster.add(frame)
                    current.preview_players = preview_roster.names()
            if finish_detector.is_active(frame):
                # A sustained banner is one event. Require a plausible gap to
                # avoid creating races from the same finish animation.
                previous_finish = next((r.finish for r in reversed(timeline.races) if r.finish is not None), -100)
                if t - previous_finish > 45:
                    if current is None or current.finish is not None:
                        current = VideoRace(start=0 if not timeline.races else previous_finish,
                                            review_note="Course preview was not detected; start needs review.")
                        timeline.races.append(current)
                    current.finish = t
                    name_votes = PlacementConsensus(current.preview_players)
                    results_done = False
                    overall_streak = 0
                    timeline.save()
            if current and current.finish is not None and not results_done and 2 <= t - current.finish <= 35:
                if t - last_result_check >= .2 and has_result_table(frame):
                    last_result_check = t
                    if current.results is None:
                        current.results = t
                    result = result_detector.detect(frame, teams=True)
                    if result and result.get("type") == "race":
                        if current.results is None:
                            current.results = t
                        overall_streak = 0
                        name_votes.add(result, frame_id=hash(frame.tobytes()))
                        current.placements = name_votes.placements()
                        current.placement_evidence = name_votes.evidence()
                    elif result and result.get("type") == "overall" and current.placements:
                        overall_streak += 1
                        results_done = overall_streak >= 3
            if t - last_progress >= 5:
                last_progress = t
                timeline.save()
                progress(timeline)
        timeline.save()
        progress(timeline)
        return timeline
    finally:
        cap.release()


def refine_result_screens(timeline: VideoTimeline) -> None:
    """Recheck a cached scan's result targets against stable visible tables."""
    cap = cv2.VideoCapture(timeline.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {timeline.source}")
    try:
        for race in timeline.races:
            if race.finish is None:
                continue
            race.results = None
            for t in np.arange(race.finish + 2, min(race.finish + 35, timeline.duration), .5):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
                ok, frame = cap.read()
                if ok and has_result_table(cv2.resize(frame, (1920, 1080))):
                    race.results = float(t)
                    break
        timeline.save()
    finally:
        cap.release()


def _sample_window(cap, start: float, end: float, *, cancelled=lambda: False):
    """Decode a result window sequentially instead of seeking every 250ms."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    next_frame = round(start * fps)
    for t in np.arange(start, end, .25):
        if cancelled():
            return
        target = round(float(t) * fps)
        while next_frame < target:
            if cancelled() or not cap.grab():
                return
            next_frame += 1
        ok, frame = cap.read()
        if not ok:
            return
        next_frame += 1
        yield float(t), frame


def refresh_names(timeline: VideoTimeline, *, progress=lambda timeline: None, cancelled=lambda: False) -> None:
    """Re-read saved race results without decoding an entire recording again."""
    cap = cv2.VideoCapture(timeline.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {timeline.source}")
    detector = RaceResultDetector()
    try:
        for race in timeline.races:
            if cancelled():
                break
            if race.finish is None:
                continue
            preview_roster = PreviewRoster()
            for offset in (.5, 1, 1.5, 2):
                if cancelled():
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, (race.start + offset) * 1000)
                ok, preview_frame = cap.read()
                if ok:
                    preview_frame = cv2.resize(preview_frame, (1920, 1080))
                    if TrackSelectDetector.has_player_panel(preview_frame):
                        preview_roster.add(preview_frame)
            race.preview_players = preview_roster.names()
            votes = PlacementConsensus(race.preview_players)
            seen_race = False
            overall_streak = 0
            for t, frame in _sample_window(cap, race.finish + 5, min(race.finish + 23, timeline.duration), cancelled=cancelled):
                if cancelled():
                    break
                frame = cv2.resize(frame, (1920, 1080))
                if not has_result_table(frame):
                    continue
                result = detector.detect(frame, teams=True)
                if not result:
                    continue
                if result["type"] == "race":
                    seen_race = True
                    overall_streak = 0
                    votes.add(result, frame_id=hash(frame.tobytes()))
                elif seen_race and result["type"] == "overall":
                    overall_streak += 1
                    if overall_streak >= 3:
                        break
            if cancelled():
                break
            race.placements = votes.placements()
            race.placement_evidence = votes.evidence()
            timeline.save()
            progress(timeline)
    finally:
        cap.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--analysis-source")
    parser.add_argument("--refine-results", action="store_true", help="Recheck saved result-screen timestamps without scanning all races again")
    parser.add_argument("--refresh-names", action="store_true")
    args = parser.parse_args()
    if args.refine_results or args.refresh_names:
        cap = cv2.VideoCapture(args.source)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        timeline = VideoTimeline.load(args.source, duration)
        if args.refresh_names:
            refresh_names(timeline, progress=lambda t: print([len(r.placements) for r in t.races], flush=True))
        else:
            refine_result_screens(timeline)
        print(timeline.cache_path)
        raise SystemExit(0)
    result = analyze_video(args.source, analysis_source=args.analysis_source,
                           progress=lambda t: print(f"{t.analyzed_until:.0f}/{t.duration:.0f}s: {len(t.races)} races", flush=True))
    print(result.cache_path)
