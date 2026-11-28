"""Local, cancellable analysis of recorded custom-room sessions."""
from __future__ import annotations

import re
from collections import Counter, deque
from difflib import SequenceMatcher
from typing import Callable

import cv2
import numpy as np
import pytesseract

from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.name_model import NameModelReader
from mktracker.detection.course_model import (
    read_course_banner_model, read_race_start_card_model,
)
from mktracker.detection.dark_results import has_dark_bar_contrast
from mktracker.detection.name_consensus import PlacementConsensus
from mktracker.detection.session_names import reconcile_names
from mktracker.detection.preview_roster import PreviewRoster
from mktracker.detection.track_select import TrackSelectDetector
from mktracker.detection.tracks import course_text_candidates, match_track_name
from mktracker.video_timeline import VideoRace, VideoTimeline


def _at_analysis_size(frame: np.ndarray) -> np.ndarray:
    """Return a 1080p frame without copying sources already at that size."""
    return frame if frame.shape[:2] == (1080, 1920) else cv2.resize(frame, (1920, 1080))


def read_course(frame: np.ndarray, reader: NameModelReader | None = None) -> str | None:
    """Read the course banner on the map; reject multiple voting choices."""
    if reader is not None and not reader.disabled:
        read = read_course_banner_model(frame, reader)
        return read[0] if read is not None else None
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
        else:
            # Newer Tesseract can read the small SNES badge as "sues".
            # Recover it only when the rest is an exact, numbered SNES title.
            prefix, separator, rest = candidate.partition(" ")
            if (separator and len(prefix) == 4
                    and SequenceMatcher(None, prefix.casefold(), "snes").ratio() >= .75):
                retro = match_track_name("SNES " + rest, cutoff=1)
                if retro and retro[0].startswith("SNES "):
                    names.add(retro[0])
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
    """Recognize colored tables and dark translucent result tables."""
    h, w = frame.shape[:2]
    from mktracker.detection.team_results import _bands
    if len(_bands(frame)) >= 4:
        return True
    # Dark translucent overlays lack team-colored bars. Require a fuller
    # grid of +point labels so the sparse
    # finish animation (which can pass has_race_results alone) stays excluded.
    detector = RaceResultDetector()
    plus_starts = detector._plus_cluster_starts(frame)
    grid_rows = detector._plus_clusters_on_grid(plus_starts, h)
    if grid_rows < 4 and has_dark_bar_contrast(frame):
        wide_starts = detector._plus_cluster_starts(frame, x1=.90, x2=.96)
        wide_grid = detector._plus_clusters_on_grid(wide_starts, h, spacing=81 / 1080)
        if wide_grid >= 9 and wide_grid >= len(wide_starts) * .7:
            result = detector.detect(frame, teams=True, model_only=True)
            return bool(result and result.get("type") == "race"
                        and len(result.get("row_centers", {})) >= 9)
    if grid_rows >= 4:
        if not has_dark_bar_contrast(frame):
            return False
        # During the scroll only a few lower ranks are visible. Confirm those
        # with parsed placements; the finish animation has spurious + clusters
        # but no coherent result rows.
        result = detector.detect(frame, teams=True)
        if not result or result["type"] != "race":
            return False
        ranks = [rank for rank, _ in result["results"] if 1 <= rank <= 24]
        return (len(set(ranks)) >= 5 or
                (len(set(ranks)) >= 3 and min(ranks) >= 8
                 and max(ranks) - min(ranks) >= 2))
    return False


def has_race_point_differentials(frame: np.ndarray) -> bool:
    """Distinguish a result clip's yellow +points from white total standings.

    The narrow score-column strip avoids the rank arrows; the upper area
    bound rejects broad scenery and the highlighted player's yellow bar.
    """
    h, w = frame.shape[:2]
    strip = frame[:, round(w * 1670 / 1920):round(w * 1735 / 1920)]
    if strip.size == 0:
        return False
    yellow = cv2.inRange(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV),
                         np.array([15, 100, 140]), np.array([45, 255, 255]))
    scale = h * w / (1080 * 1920)
    top, step, height = (round(h * value / 1080) for value in (45, 77, 55))
    counts = [cv2.countNonZero(yellow[y:min(h, y + height)])
              for y in (top + index * step for index in range(13)) if y + height <= h]
    if sum(80 * scale < count < 650 * scale for count in counts) >= 5:
        return True
    # A three-team 24-player result uses an 81px grid with +points at the
    # far right. Confirm the full dark result grid before accepting a short
    # result-only clip; a single yellow HUD label is insufficient evidence.
    detector = RaceResultDetector()
    if not has_dark_bar_contrast(frame):
        return False
    starts = detector._plus_cluster_starts(frame, x1=.90, x2=.96)
    if (len(starts) < 9 or
            detector._plus_clusters_on_grid(starts, h, spacing=81 / 1080) < 9):
        return False
    result = detector.detect(frame, teams=True, model_only=True)
    return bool(result and result.get("type") == "race"
                and len(result.get("row_centers", {})) >= 9)


def _fill_colored_scroll_gaps(
    votes: PlacementConsensus,
    quarter_frames: deque[tuple[int, np.ndarray]],
    detector: RaceResultDetector,
    reader: NameModelReader,
) -> bool:
    """Read a cached quarter frame only for an interior 24p rank gap."""
    ranks = {rank for rank, _ in votes.placements()}
    if not ranks or max(ranks) <= 12 or len(ranks) < 12:
        return False
    missing = sorted(set(range(min(ranks), max(ranks) + 1)) - ranks)
    if not missing or len(missing) > 2:
        return False
    added = False
    for rank in missing:
        if rank in {candidate for candidate, _ in votes.placements()}:
            continue
        for frame_index, frame in reversed(quarter_frames):
            rank_reader = reader if hasattr(reader, "predict_crops") else None
            grid = detector.detect(frame, teams=True, model_only=True,
                                   color_model=rank_reader, rank_only=True)
            if (not grid or grid.get("type") != "race" or
                    grid.get("name_crop_x") is None or rank not in grid.get("row_centers", {})):
                continue
            read = reader.read(frame, grid)
            if rank not in {candidate for candidate, _ in read.get("results", [])}:
                continue
            votes.add(read, frame_id=hash(frame.tobytes()))
            added = True
            break
    return added


def _fill_dark_scroll_gaps(
    votes: PlacementConsensus,
    quarter_frames: deque[tuple[int, np.ndarray]],
    examined: set[int],
    detector: RaceResultDetector,
    reader: NameModelReader,
) -> bool:
    """Recover dark-board ranks shown only between the half-second samples.

    A recording can cut away during the scroll. Inspect buffered quarter
    frames only while ranks are missing, and read at most one extra board per
    regular sample so recovery does not dominate analysis time.
    """
    ranks = {rank for rank, _ in votes.placements()}
    if len(ranks) < 8 or len(ranks) >= 24:
        return False
    missing = set(range(1, 25)) - ranks
    for frame_index, frame in reversed(quarter_frames):
        if frame_index in examined:
            continue
        examined.add(frame_index)
        rank_reader = reader if hasattr(reader, "predict_crops") else None
        grid = detector.detect(frame, teams=True, model_only=True,
                               dark_model=rank_reader, rank_only=True)
        if (not grid or grid.get("type") != "race" or
                grid.get("name_crop_x") is not None or
                not (missing & grid.get("row_centers", {}).keys())):
            continue
        read = reader.read(frame, grid)
        new = missing & {rank for rank, _ in read.get("results", [])}
        if new:
            votes.add(read, frame_id=hash(frame.tobytes()))
        return bool(new)
    return False


def _colored_room_size(result: dict) -> int:
    """Distinguish the shifted 24p colored grid from the 12p team table."""
    if (max(result.get("row_centers", {}), default=0) > 12 or
            result.get("name_crop_x", (.68,))[0] < .67):
        return 24
    return 12


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
    name_model = NameModelReader() if NameModelReader.available() else None
    color_model = NameModelReader(color=True) if NameModelReader.color_available() else None
    sample_step = max(1, round(fps / 2))
    last_course_check = -100.0
    last_result_check = -100.0
    last_progress = -100.0
    current: VideoRace | None = None
    name_votes = PlacementConsensus()
    highlighted_votes: Counter[int] = Counter()
    results_done = False
    overall_streak = 0
    preview_roster = PreviewRoster(color_model)
    quarter_result_frames: deque[tuple[int, np.ndarray]] = deque(maxlen=8)
    examined_quarter_frames: set[int] = set()
    last_preview_check = -100.0
    frame_index = -1
    try:
        while not cancelled():
            if not cap.grab():
                # Some downloaded clips advertise a short tail that has no
                # decodable frames. Keep a genuine larger truncation visible.
                missing_seconds = max(0, count - frame_index - 1) / fps
                tail_tolerance = 5 if timeline.duration <= 30 else 2
                if count > 0 and missing_seconds > tail_tolerance:
                    raise RuntimeError(f"Video ended unexpectedly at {frame_index / fps:.1f}s")
                timeline.duration = min(timeline.duration, (frame_index + 1) / fps)
                timeline.complete = True
                timeline.analyzed_until = timeline.duration
                break
            frame_index += 1
            in_results = (current is not None and not results_done
                          and ((current.finish is not None and frame_index / fps - current.finish <= 35)
                               or (current.results is not None and timeline.duration <= 30)))
            # Sample result pages twice a second. Their rank grid and names
            # can change quickly while the board scrolls, but reading every
            # displayed video frame repeats the same expensive OCR work.
            step = sample_step
            quarter_step = max(1, round(fps / 4)) if in_results else sample_step
            if frame_index % step and frame_index % quarter_step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Cannot decode video at {frame_index / fps:.1f}s")
            if frame_index % step:
                if in_results and ((color_model is not None and not color_model.disabled)
                                   or (name_model is not None and not name_model.disabled)):
                    quarter_result_frames.append((frame_index, _at_analysis_size(frame)))
                continue
            t = frame_index / fps
            frame = _at_analysis_size(frame)
            timeline.analyzed_until = t
            if t - last_course_check >= 3 and (current is None or current.finish is not None):
                if TrackSelectDetector.has_player_panel(frame):
                    last_course_check = t
                    course = read_course(frame, name_model)
                    if course:
                        current = VideoRace(start=t, track=course,
                                            review_note="Start marks the course preview; boundaries sampled every 0.5 seconds.")
                        timeline.races.append(current)
                        preview_roster = PreviewRoster(color_model)
                        last_preview_check = -100.0
                        timeline.save()
            if (name_model is not None and not name_model.disabled and
                    (current is None or current.finish is not None)):
                card = read_race_start_card_model(frame, name_model)
                if card is not None:
                    current = VideoRace(
                        start=t, track=card[0],
                        review_note="Start marks the race countdown course card; the map preview was not visible.",
                    )
                    timeline.races.append(current)
                    preview_roster = PreviewRoster(color_model)
                    last_preview_check = -100.0
                    last_course_check = t
                    timeline.save()
            if (current and current.finish is None and t - current.start <= 3
                    and t - last_preview_check >= .5 and not preview_roster.complete()):
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
                    highlighted_votes = Counter()
                    results_done = False
                    overall_streak = 0
                    quarter_result_frames.clear()
                    examined_quarter_frames.clear()
                    timeline.save()
            after_finish = (current is not None and current.finish is not None
                            and 2 <= t - current.finish <= 35)
            short_result_clip = timeline.duration <= 30 and (current is None or current.finish is None)
            if not results_done and (after_finish or short_result_clip):
                if t - last_result_check >= .2:
                    last_result_check = t
                    if (short_result_clip and not after_finish
                            and (current is None or current.results is None)
                            and not has_race_point_differentials(frame)):
                        continue
                    model_ready = ((name_model is not None and not name_model.disabled)
                                   or (color_model is not None and not color_model.disabled))
                    colored_result = False
                    if model_ready:
                        result = result_detector.detect(frame, teams=True, model_only=True,
                                                        dark_model=name_model, color_model=color_model)
                        if result and result.get("type") == "race":
                            colored_result = result.get("name_crop_x") is not None
                            if colored_result:
                                if color_model is not None and not color_model.disabled:
                                    result = color_model.read(frame, result)
                                else:
                                    result = result_detector.detect(frame, teams=True)
                            else:
                                if name_model is not None and not name_model.disabled:
                                    result = name_model.read(frame, result)
                                else:
                                    result = result_detector.detect(frame, teams=True)
                    else:
                        if not has_result_table(frame):
                            continue
                        result = result_detector.detect(frame, teams=True)
                    if result and result.get("type") == "race" and result.get("results"):
                        if current is None:
                            current = VideoRace(
                                start=0, track="Result clip", results=t,
                                review_note="The clip starts after the race; no finish or course preview is visible.",
                            )
                            timeline.races.append(current)
                            name_votes = PlacementConsensus()
                            highlighted_votes = Counter()
                        elif current.finish is None and current.results is None:
                            current.review_note += " Result table found; finish banner is outside this clip."
                        if current.results is None:
                            current.results = t
                        overall_streak = 0
                        previous_placements = len(current.placements)
                        name_votes.add(result, frame_id=hash(frame.tobytes()))
                        if (colored_result and color_model is not None
                                and not color_model.disabled and
                                _fill_colored_scroll_gaps(
                                    name_votes, quarter_result_frames,
                                    result_detector, color_model)):
                            quarter_result_frames.clear()
                        if (not colored_result and name_model is not None
                                and not name_model.disabled):
                            _fill_dark_scroll_gaps(
                                name_votes, quarter_result_frames,
                                examined_quarter_frames, result_detector, name_model)
                        if result.get("highlighted_rank_source") in ("grid", "ocr_confident"):
                            highlighted_votes[result["highlighted_rank"]] += 1
                            best_rank, best_count = highlighted_votes.most_common(1)[0]
                            if best_count >= 2 and best_count / highlighted_votes.total() >= .6:
                                current.highlighted_rank = best_rank
                        current.placements = name_votes.placements()
                        current.placement_evidence = name_votes.evidence()
                        if len(current.placements) > previous_placements:
                            timeline.save()
                            progress(timeline)
                        expected = _colored_room_size(result) if colored_result else 24
                        needed_frames = 3 if colored_result and expected == 12 else 2
                        if (len(current.placements) == expected
                                and all(item.get("frames", 0) >= needed_frames
                                        for item in current.placement_evidence.values())):
                            results_done = True
                    elif result and result.get("type") == "overall" and current is not None and current.placements:
                        overall_streak += 1
                        results_done = overall_streak >= 3
            if t - last_progress >= 5:
                last_progress = t
                timeline.save()
                progress(timeline)
        if timeline.complete:
            reconcile_names(timeline.races)
        timeline.save()
        progress(timeline)
        return timeline
    finally:
        cap.release()
        if name_model is not None:
            name_model.close()
        if color_model is not None:
            color_model.close()


def refine_result_screens(timeline: VideoTimeline) -> None:
    """Recheck a cached scan's result targets against stable visible tables."""
    cap = cv2.VideoCapture(timeline.source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {timeline.source}")
    try:
        for race in timeline.races:
            if race.finish is None and race.results is None:
                continue
            if race.finish is None:
                window_start = max(0, race.results - .5)
                window_end = min(race.results + 18, timeline.duration)
            else:
                window_start = race.finish + 2
                window_end = min(race.finish + 35, timeline.duration)
            race.results = None
            for t in np.arange(window_start, window_end, .5):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
                ok, frame = cap.read()
                if ok and has_result_table(_at_analysis_size(frame)):
                    race.results = float(t)
                    break
        timeline.save()
    finally:
        cap.release()


def _sample_window(cap, start: float, end: float, *, cancelled=lambda: False,
                   interval: float = .25):
    """Decode a result window sequentially instead of seeking every 250ms."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    next_frame = round(start * fps)
    for t in np.arange(start, end, interval):
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
    name_model = NameModelReader() if NameModelReader.available() else None
    color_model = NameModelReader(color=True) if NameModelReader.color_available() else None
    try:
        for race in timeline.races:
            if cancelled():
                break
            if race.finish is None and race.results is None:
                continue
            if race.finish is None:
                window_start = max(0, race.results - .5)
                window_end = min(race.results + 18, timeline.duration)
            else:
                window_start = race.finish + 5
                window_end = min(race.finish + 23, timeline.duration)
            preview_roster = PreviewRoster(color_model)
            for offset in (.5, 1, 1.5, 2):
                if cancelled():
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, (race.start + offset) * 1000)
                ok, preview_frame = cap.read()
                if ok:
                    preview_frame = _at_analysis_size(preview_frame)
                    if TrackSelectDetector.has_player_panel(preview_frame):
                        preview_roster.add(preview_frame)
                        if preview_roster.complete():
                            break
            race.preview_players = preview_roster.names()
            votes = PlacementConsensus(race.preview_players)
            highlighted_votes: Counter[int] = Counter()
            seen_race = False
            saw_colored_24p = False
            overall_streak = 0
            race.results = None
            race.highlighted_rank = None
            samples = (_sample_window(cap, window_start, window_end,
                                      cancelled=cancelled, interval=.5)
                       if name_model is not None or color_model is not None else
                       _sample_window(cap, window_start, window_end, cancelled=cancelled))
            for t, frame in samples:
                if cancelled():
                    break
                frame = _at_analysis_size(frame)
                model_ready = ((name_model is not None and not name_model.disabled)
                               or (color_model is not None and not color_model.disabled))
                colored_result = False
                if model_ready:
                    result = detector.detect(frame, teams=True, model_only=True,
                                             dark_model=name_model, color_model=color_model)
                    if result and result.get("type") == "race":
                        colored_result = result.get("name_crop_x") is not None
                        if colored_result:
                            if color_model is not None and not color_model.disabled:
                                result = color_model.read(frame, result)
                            else:
                                result = detector.detect(frame, teams=True)
                        else:
                            if name_model is not None and not name_model.disabled:
                                result = name_model.read(frame, result)
                            else:
                                result = detector.detect(frame, teams=True)
                else:
                    if not has_result_table(frame):
                        continue
                    result = detector.detect(frame, teams=True)
                if not result:
                    continue
                if result["type"] == "race" and result.get("results"):
                    if race.results is None:
                        race.results = t
                    seen_race = True
                    overall_streak = 0
                    if colored_result and _colored_room_size(result) == 24:
                        saw_colored_24p = True
                    votes.add(result, frame_id=hash(frame.tobytes()))
                    if result.get("highlighted_rank_source") in ("grid", "ocr_confident"):
                        highlighted_votes[result["highlighted_rank"]] += 1
                    if (colored_result and _colored_room_size(result) == 12
                            and len(votes.placements()) == 12
                            and all(item.get("frames", 0) >= 3
                                    for item in votes.evidence().values())):
                        break
                elif seen_race and result["type"] == "overall":
                    overall_streak += 1
                    if overall_streak >= 3:
                        break
            if cancelled():
                break
            if (saw_colored_24p and color_model is not None and
                    not color_model.disabled and race.results is not None):
                # The ordinary model pass samples every 0.5s. Only when that
                # leaves an interior rank gap, inspect interleaved quarter
                # frames around the result scroll; stop after the gap closes.
                start = race.results + .25
                end = min(race.results + 5, window_end)
                for _, extra in _sample_window(cap, start, end, cancelled=cancelled,
                                               interval=.5):
                    ranks = {rank for rank, _ in votes.placements()}
                    if (cancelled() or not ranks or
                            not (set(range(min(ranks), max(ranks) + 1)) - ranks)):
                        break
                    extra = _at_analysis_size(extra)
                    _fill_colored_scroll_gaps(
                        votes, deque([(0, extra)], maxlen=1), detector, color_model)
            race.placements = votes.placements()
            race.placement_evidence = votes.evidence()
            if highlighted_votes:
                rank, count = highlighted_votes.most_common(1)[0]
                if count >= 2 and count / highlighted_votes.total() >= .6:
                    race.highlighted_rank = rank
            timeline.save()
            progress(timeline)
        if not cancelled():
            reconcile_names(timeline.races)
            timeline.save()
            progress(timeline)
    finally:
        cap.release()
        if name_model is not None:
            name_model.close()
        if color_model is not None:
            color_model.close()


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
