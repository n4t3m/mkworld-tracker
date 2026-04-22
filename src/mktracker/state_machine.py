from __future__ import annotations

import collections
import dataclasses
import logging
import re
import threading
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from mktracker.detection.match_results import MatchResultDetector
from mktracker.detection.match_settings import MatchSettings, MatchSettingsDetector
from mktracker.detection.player_reader import PlayerReader
from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_rank import RaceRankDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.track_select import TrackSelectDetector
from mktracker.debug_config import load_debug_mode
from mktracker.gemini_client import load_api_key
from mktracker.gemini_match_results import request_match_results
from mktracker.gemini_rank import request_race_rank
from mktracker.gemini_results import request_race_results
from mktracker.match_record import (
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    RaceRecord,
    TeamGroup,
)
from mktracker.table_generator import generate_table

TABLE_FILE = "table.png"

logger = logging.getLogger(__name__)

# How long to wait after match settings are detected before looking for tracks.
_MATCH_START_DELAY = 5.0

# Directory where match records and per-match debug frames are saved.
_MATCHES_DIR = Path("matches")
_DEBUG_FINISH_DIR = Path("debug_finish")
_DEBUG_RANK_DIR = Path("debug_rank")

# Number of frames before/after the placement burst to save when debug mode
# is enabled. Saved into <race_dir>/debug_placements/.
_DEBUG_PLACEMENT_CONTEXT = 5


class GameState(Enum):
    WAITING_FOR_MATCH = auto()
    MATCH_STARTED = auto()
    WAITING_FOR_TRACK_PICK = auto()
    READING_PLAYERS_IN_RACE = auto()
    WAITING_FOR_RACE_END = auto()
    DETECTING_RACE_RANK = auto()
    READING_RACE_RESULTS = auto()
    FINALIZING_MATCH = auto()


@dataclasses.dataclass(frozen=True)
class RaceInfo:
    track_name: str
    players: tuple[str, ...]
    placements: tuple[tuple[int, str], ...] = ()
    race_rank: int | None = None
    gemini_results: dict | None = None


class GameStateMachine:
    """Orchestrates detection across game states.

    Call ``update(frame)`` on each detection cycle.
    """

    def __init__(self) -> None:
        self._state = GameState.WAITING_FOR_MATCH
        self._state_entered_at = time.monotonic()

        self._match_settings: MatchSettings | None = None
        self._races: list[RaceInfo] = []
        self._current_race: int = 0
        self._match_dir: Path | None = None
        self._match_started_at: datetime | None = None
        self._match_completed_at: datetime | None = None
        # Monotonic counter that uniquely identifies the current match.
        # Bumped on every new match start AND on reset(), so any in-flight
        # Gemini callback can detect that it has been orphaned and re-route
        # itself to the original match's on-disk record.
        self._match_seq: int = 0
        self._pending_track: str | None = None

        self._match_detector = MatchSettingsDetector()
        self._track_detector = TrackSelectDetector()
        self._player_reader = PlayerReader()
        self._finish_detector = RaceFinishDetector()
        self._rank_detector = RaceRankDetector()
        self._result_detector = RaceResultDetector()
        self._match_result_detector = MatchResultDetector()

        # Lock protecting _races from concurrent writes by Gemini callbacks.
        self._races_lock = threading.Lock()

        self._race_placements: dict[int, str] = {}
        self._race_placement_quality: dict[int, int] = {}
        self._seen_race_results = False
        self._result_frames: list[np.ndarray] = []
        # Debug-mode rolling buffer of frames seen *before* the first
        # placement frame, plus counters for the post-burst capture window.
        self._pre_results_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=_DEBUG_PLACEMENT_CONTEXT,
        )
        self._pre_results_dumped: bool = False
        self._post_results_count: int = 0
        self._match_final_results: list[tuple[str, int]] | None = None
        self._gemini_match_results: dict | None = None
        self._match_banner_seen_at: float | None = None

        self.debug_mode: bool = load_debug_mode()

        logger.info(
            "State: WAITING_FOR_MATCH (debug_mode=%s)", self.debug_mode,
        )

    # -- public properties -------------------------------------------------

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def match_settings(self) -> MatchSettings | None:
        return self._match_settings

    @match_settings.setter
    def match_settings(self, value: MatchSettings | None) -> None:
        self._match_settings = value

    @property
    def current_race(self) -> int:
        """Current race number (1-based), or 0 if no match is active."""
        return self._current_race

    @property
    def current_match_id(self) -> str | None:
        """ID of the currently-tracked match (its on-disk folder name), or
        ``None`` if no match folder has been created yet."""
        return self._match_dir.name if self._match_dir is not None else None

    @property
    def is_match_active(self) -> bool:
        """``True`` while a match is in progress (between leaving
        ``WAITING_FOR_MATCH`` and returning to it)."""
        return (
            self._state is not GameState.WAITING_FOR_MATCH
            and self._match_started_at is not None
        )

    @property
    def player_count(self) -> int | None:
        """Number of players in the most recent race, or ``None`` if unknown."""
        if self._races:
            return len(self._races[-1].players)
        return None

    @property
    def match_final_results(self) -> list[tuple[str, int]] | None:
        """Final match results as ``[(name, points), ...]``, or ``None``."""
        return self._match_final_results

    @property
    def gemini_match_results(self) -> dict | None:
        """Full structured Gemini match results, or ``None``."""
        return self._gemini_match_results

    @property
    def races(self) -> list[RaceInfo]:
        return list(self._races)

    def reset(self) -> None:
        """Reset to WAITING_FOR_MATCH, clearing all match data."""
        self._match_settings = None
        self._races = []
        self._current_race = 0
        self._match_started_at = None
        self._match_completed_at = None
        # Drop the match folder pointer so the next real or manual match
        # gets a fresh timestamped directory rather than reusing the old one.
        self._match_dir = None
        # Invalidate any in-flight Gemini callbacks for the prior match —
        # they will route their result to the on-disk match record instead
        # of writing into the (now-cleared) live state.
        self._match_seq += 1
        self._pending_track = None
        self._race_placements = {}
        self._race_placement_quality = {}
        self._seen_race_results = False
        self._result_frames = []
        self._pre_results_buffer.clear()
        self._pre_results_dumped = False
        self._post_results_count = 0
        self._match_final_results = None
        self._gemini_match_results = None
        self._match_banner_seen_at = None
        self._track_detector._last_match_time = 0.0
        self._transition(GameState.WAITING_FOR_MATCH)

    def start_manual_match(self) -> bool:
        """Mark the current state machine as the start of a manual match
        session, so subsequent meaningful state changes get persisted to
        ``matches/<timestamp>/match.json``.

        This is the explicit opt-in for users who advance past
        ``WAITING_FOR_MATCH`` manually and bypass real settings detection.
        Without this call, ``_save_match_record`` refuses to write because
        ``_match_started_at`` stays ``None`` — preventing accidental
        "ghost match" pollution of the history store.

        Returns ``True`` if a manual match was actually started, ``False``
        if it was a no-op (already in progress, or no settings configured).

        No-op when a match is already in progress; call :meth:`reset` first
        to start over.
        """
        if self._match_started_at is not None:
            logger.info(
                "start_manual_match: ignored — match already in progress",
            )
            return False
        if self._match_settings is None:
            logger.warning(
                "start_manual_match: ignored — no match settings configured",
            )
            return False
        self._match_seq += 1
        now = datetime.now()
        self._match_started_at = now
        self._match_dir = _MATCHES_DIR / now.strftime("%Y%m%d_%H%M%S")
        self._match_dir.mkdir(parents=True, exist_ok=True)
        s = self._match_settings
        logger.info(
            "Manual match started — %s, %s, %s, %s, %d races, %s",
            s.cc_class, s.teams, s.items, s.com_difficulty,
            s.race_count, s.intermission,
        )
        self._save_match_record()
        return True

    _ADVANCE_ORDER = {
        GameState.WAITING_FOR_MATCH: GameState.MATCH_STARTED,
        GameState.MATCH_STARTED: GameState.WAITING_FOR_TRACK_PICK,
        GameState.WAITING_FOR_TRACK_PICK: GameState.READING_PLAYERS_IN_RACE,
        GameState.READING_PLAYERS_IN_RACE: GameState.WAITING_FOR_RACE_END,
        GameState.WAITING_FOR_RACE_END: GameState.DETECTING_RACE_RANK,
        GameState.DETECTING_RACE_RANK: GameState.READING_RACE_RESULTS,
        GameState.READING_RACE_RESULTS: GameState.WAITING_FOR_TRACK_PICK,
        GameState.FINALIZING_MATCH: GameState.WAITING_FOR_MATCH,
    }

    def advance(self) -> None:
        """Manually advance to the next logical state."""
        next_state = self._ADVANCE_ORDER.get(self._state)
        if next_state is None:
            return
        self._transition(next_state)

    # -- main update -------------------------------------------------------

    def update(self, frame: np.ndarray) -> None:
        if self._state is GameState.WAITING_FOR_MATCH:
            self._handle_waiting(frame)
        elif self._state is GameState.MATCH_STARTED:
            self._handle_match_started()
        elif self._state is GameState.WAITING_FOR_TRACK_PICK:
            self._handle_racing(frame)
        elif self._state is GameState.READING_PLAYERS_IN_RACE:
            self._handle_reading_players(frame)
        elif self._state is GameState.WAITING_FOR_RACE_END:
            self._handle_race_ending(frame)
        elif self._state is GameState.DETECTING_RACE_RANK:
            self._handle_detecting_rank(frame)
        elif self._state is GameState.READING_RACE_RESULTS:
            self._handle_reading_results(frame)
        elif self._state is GameState.FINALIZING_MATCH:
            self._handle_finalizing_match(frame)

    # -- state handlers ----------------------------------------------------

    def _handle_waiting(self, frame: np.ndarray) -> None:
        settings = self._match_detector.detect(frame)
        if settings is None:
            return
        self._match_settings = settings
        self._races = []
        self._match_completed_at = None
        # Invalidate any in-flight Gemini callbacks for the prior match.
        self._match_seq += 1
        now = datetime.now()
        self._match_started_at = now
        self._match_dir = _MATCHES_DIR / now.strftime("%Y%m%d_%H%M%S")
        self._match_dir.mkdir(parents=True, exist_ok=True)
        self._save_frame(frame, "match_settings")
        logger.info(
            "Match detected — %s, %s, %s, %s, %d races, %s",
            settings.cc_class,
            settings.teams,
            settings.items,
            settings.com_difficulty,
            settings.race_count,
            settings.intermission,
        )
        self._save_match_record()
        self._transition(GameState.MATCH_STARTED)

    def _handle_match_started(self) -> None:
        elapsed = time.monotonic() - self._state_entered_at
        if elapsed >= _MATCH_START_DELAY:
            self._transition(GameState.WAITING_FOR_TRACK_PICK)

    def _handle_racing(self, frame: np.ndarray) -> None:
        result = self._track_detector.detect(frame)
        if result is None:
            return
        self._pending_track = result["track_name"]
        self._save_race_frame(frame, len(self._races) + 1, "track")
        logger.info("Track detected: %s — reading players on next frame", self._pending_track)
        self._transition(GameState.READING_PLAYERS_IN_RACE)

    def _handle_reading_players(self, frame: np.ndarray) -> None:
        use_teams = (
            self._match_settings is not None
            and self._match_settings.teams != "No Teams"
        )
        player_names = tuple(
            p.name for p in self._player_reader.read_players(
                frame, teams=use_teams)
        )

        race = RaceInfo(track_name=self._pending_track, players=player_names)
        self._races.append(race)
        self._save_race_frame(frame, len(self._races), "players")

        total = self._match_settings.race_count if self._match_settings else "?"
        logger.info("Race %d/%s: %s", len(self._races), total, self._pending_track)
        for name in player_names:
            logger.info("  - %s", name)

        self._pending_track = None
        self._save_match_record()
        self._transition(GameState.WAITING_FOR_RACE_END)

    def _handle_race_ending(self, frame: np.ndarray) -> None:
        if not self._finish_detector.is_active(frame):
            return

        # Temporary debugging: log the frame that triggered FINISH detection.
        _DEBUG_FINISH_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(_DEBUG_FINISH_DIR / f"{ts}.png"), frame)
        self._save_race_frame(frame, len(self._races), "finish")
        logger.info("Race finished — FINISH! detected")
        self._track_detector._last_match_time = time.monotonic()
        self._race_placements = {}
        self._race_placement_quality = {}
        self._seen_race_results = False
        self._result_frames = []
        self._pre_results_buffer.clear()
        self._pre_results_dumped = False
        self._post_results_count = 0
        self._transition(GameState.DETECTING_RACE_RANK)

    def _handle_detecting_rank(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self._state_entered_at

        if elapsed < 2.0:
            return

        # Save the full frame to debug_rank/ for LLM experimentation.
        _DEBUG_RANK_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(_DEBUG_RANK_DIR / f"{ts}.png"), frame)
        self._save_race_frame(frame, len(self._races), "rank")
        logger.info("Race rank frame captured (1s after FINISH)")

        # Fire async Gemini call to determine the user's race placement.
        # We capture (race_index, match_seq, match_dir) so the callback can
        # tell whether the state machine has since moved on to a new match
        # and, if so, route its result to the original match's on-disk record.
        race_index = len(self._races) - 1
        race_num = race_index + 1
        race_log_dir = self._race_dir(race_num)
        request_race_rank(
            frame.copy(), race_num,
            self._make_rank_callback(race_index, self._match_seq, self._match_dir),
            log_dir=race_log_dir,
        )

        self._transition(GameState.READING_RACE_RESULTS)

    def _make_rank_callback(
        self,
        race_index: int,
        match_seq: int,
        match_dir: Path | None,
    ):
        """Return a callback that stores a Gemini rank result into the
        correct ``RaceInfo``, or routes it to disk if the match is stale.

        Runs on a background thread."""
        def _on_rank(rank: int | None) -> None:
            with self._races_lock:
                if match_seq != self._match_seq:
                    self._apply_stale_rank(match_dir, race_index, rank)
                    return
                if race_index < 0 or race_index >= len(self._races):
                    logger.warning(
                        "Gemini rank callback: race index %d out of range "
                        "(have %d races)", race_index, len(self._races),
                    )
                    return
                old = self._races[race_index]
                self._races[race_index] = dataclasses.replace(old, race_rank=rank)
                logger.info(
                    "Race %d rank set to %s (track: %s)",
                    race_index + 1, rank, old.track_name,
                )
                self._save_match_record()
        return _on_rank

    def _handle_reading_results(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self._state_entered_at

        if elapsed > 30.0:
            logger.warning(
                "Race results reading timed out after %.0fs", elapsed,
            )
            self._finalise_race_placements()
            self._transition(self._state_after_race_results())
            return

        if load_api_key():
            self._handle_reading_results_gemini(frame)
        else:
            self._handle_reading_results_ocr(frame)

    def _handle_reading_results_gemini(self, frame: np.ndarray) -> None:
        """Collect frames for Gemini; use lightweight bar + plus-cluster
        check for state transitions instead of OCR."""
        has_results = self._result_detector.has_race_results(frame)

        if has_results:
            if self.debug_mode and not self._pre_results_dumped:
                self._dump_pre_placement_buffer()
                self._pre_results_dumped = True
            self._seen_race_results = True
            self._result_frames.append(frame.copy())
            frame_num = len(self._result_frames)
            self._save_race_frame(frame, len(self._races), f"placement_{frame_num:02d}")
            logger.debug(
                "Collected race result frame %d for Gemini", frame_num,
            )
        elif self._seen_race_results:
            if self.debug_mode and self._post_results_count < _DEBUG_PLACEMENT_CONTEXT:
                self._post_results_count += 1
                self._save_post_placement_frame(frame, self._post_results_count)
                if self._post_results_count < _DEBUG_PLACEMENT_CONTEXT:
                    return  # keep sampling for the rest of the post-burst window
            logger.info("Overall standings detected — sending %d frames to Gemini",
                        len(self._result_frames))
            self._finalise_race_placements()
            self._transition(self._state_after_race_results())
        elif self.debug_mode:
            # Pre-placement window: keep a rolling buffer of the most recent
            # frames so we can dump them once placements first appear.
            self._pre_results_buffer.append(frame.copy())

    def _handle_reading_results_ocr(self, frame: np.ndarray) -> None:
        """Original OCR-based race results reading."""
        use_teams = (
            self._match_settings is not None
            and self._match_settings.teams != "No Teams"
        )
        result = self._result_detector.detect(frame, teams=use_teams)
        if result is None:
            return

        if result["type"] == "race":
            results = result["results"]
            self._seen_race_results = True
            frame_quality = len(results)
            new_count = 0
            for placement, name in results:
                if not placement:
                    continue
                prev_quality = self._race_placement_quality.get(placement, 0)
                if placement not in self._race_placements:
                    # New placement — always store.
                    self._race_placements[placement] = name
                    self._race_placement_quality[placement] = frame_quality
                    new_count += 1
                elif frame_quality > prev_quality:
                    # This frame detected more rows overall, so its OCR is
                    # likely more reliable.  Overwrite the earlier value.
                    self._race_placements[placement] = name
                    self._race_placement_quality[placement] = frame_quality
                    new_count += 1
            if new_count:
                self._save_race_frame(frame, len(self._races), f"placement_{len(self._race_placements):02d}")
            logger.debug(
                "Race results so far: %d placements", len(self._race_placements),
            )
        elif result["type"] == "overall" and self._seen_race_results:
            logger.info("Overall standings detected — race results complete")
            self._finalise_race_placements()
            self._transition(self._state_after_race_results())

    def _handle_finalizing_match(self, frame: np.ndarray) -> None:
        if self._match_final_results is not None:
            return  # Already captured.

        if load_api_key():
            self._handle_finalizing_match_gemini(frame)
        else:
            self._handle_finalizing_match_ocr(frame)

    def _handle_finalizing_match_gemini(self, frame: np.ndarray) -> None:
        """Two-phase detection: spot the banner first, then wait 1s for
        results to load before capturing the frame for Gemini."""
        teams = self._match_settings.teams if self._match_settings else "No Teams"
        if not self._match_result_detector._has_result_banner(frame, teams=teams):
            return

        if self._match_banner_seen_at is None:
            # First frame with banner — save it and start the delay.
            self._match_banner_seen_at = time.monotonic()
            self._save_frame(frame, "match_banner")
            logger.info("Match results banner detected — waiting for results to load")
            return

        if time.monotonic() - self._match_banner_seen_at < 1.0:
            return

        # 1s has passed since banner appeared — capture and send.
        self._save_frame(frame, "match_results")
        log_dir = self._match_dir
        request_match_results(
            frame.copy(),
            self._make_match_results_callback(self._match_seq, self._match_dir),
            log_dir=log_dir,
        )
        # Mark as captured so we don't fire again, and transition.
        self._match_final_results = []
        logger.info("Match results frame sent to Gemini")
        self._transition(GameState.WAITING_FOR_MATCH)

    def _handle_finalizing_match_ocr(self, frame: np.ndarray) -> None:
        """Original OCR-based match results reading."""
        teams = self._match_settings.teams if self._match_settings else "No Teams"
        player_count = self.player_count or 12

        result = self._match_result_detector.detect(
            frame, teams=teams, player_count=player_count,
        )
        if result is None:
            return

        self._match_final_results = result["results"]
        self._match_completed_at = datetime.now()
        self._save_frame(frame, "match_results")
        logger.info("Match final results:")
        for name, points in result["results"]:
            logger.info("  %s: %d pts", name, points)
        self._dump_match_summary()
        self._save_match_record()
        self._transition(GameState.WAITING_FOR_MATCH)

    def _make_match_results_callback(
        self,
        match_seq: int,
        match_dir: Path | None,
    ):
        """Return a callback that stores Gemini match results, or routes
        them to disk if the match is stale.

        Runs on a background thread."""
        def _on_results(parsed: dict | None, results: list[tuple[str, int]]) -> None:
            with self._races_lock:
                if match_seq != self._match_seq:
                    self._apply_stale_match_results(match_dir, parsed, results)
                    return
                self._gemini_match_results = parsed
                if results:
                    self._match_final_results = results
                    self._match_completed_at = datetime.now()
                    logger.info("Match final results from Gemini (%d players):", len(results))
                    for name, points in results:
                        logger.info("  %s: %d pts", name, points)
                else:
                    logger.warning("Gemini returned no match results")
                self._dump_match_summary()
                self._save_match_record()
        return _on_results

    def _dump_match_summary(self) -> None:
        """Print a full match summary to the terminal."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("MATCH SUMMARY")
        print(sep)

        if self._match_settings:
            s = self._match_settings
            print(f"Settings: {s.cc_class}, {s.teams}, {s.items}, "
                  f"{s.com_difficulty}, {s.race_count} races, {s.intermission}")
        print()

        for i, race in enumerate(self._races, 1):
            rank_str = f"  (Your rank: {race.race_rank})" if race.race_rank else ""
            print(f"Race {i}: {race.track_name}{rank_str}")
            if race.placements:
                for place, name in race.placements:
                    print(f"  {place:>2}. {name}")
            else:
                print("  (no placements)")
            print()

        if self._match_final_results:
            print("Final Standings:")
            for name, score in self._match_final_results:
                print(f"  {name}: {score} pts")
        else:
            print("Final Standings: (not available)")

        print(sep)

    def _state_after_race_results(self) -> GameState:
        """Return the next state after race results are finalised."""
        if (
            self._match_settings is not None
            and self._current_race >= self._match_settings.race_count
        ):
            logger.info("Final race completed (%d/%d)", self._current_race, self._match_settings.race_count)
            return GameState.FINALIZING_MATCH
        return GameState.WAITING_FOR_TRACK_PICK

    def _finalise_race_placements(self) -> None:
        """Log and store accumulated race placements.

        When Gemini frames have been collected, fires an async API call
        instead of using the OCR-accumulated placements.
        """
        if self._result_frames:
            # Gemini path — fire async call with collected frames.
            race_index = len(self._races) - 1
            race_num = race_index + 1
            race_log_dir = self._race_dir(race_num)
            request_race_results(
                list(self._result_frames), race_num,
                self._make_results_callback(
                    race_index, self._match_seq, self._match_dir,
                ),
                log_dir=race_log_dir,
            )
            self._result_frames = []
            return

        # OCR path — use accumulated placements.
        if not self._race_placements:
            logger.info("No race placements captured")
            return

        logger.info("Race placements collected (%d):", len(self._race_placements))
        for p in sorted(self._race_placements):
            logger.info("  %2d. %s", p, self._race_placements[p])

        # Update the most recent RaceInfo with placements.
        if self._races:
            with self._races_lock:
                old = self._races[-1]
                self._races[-1] = dataclasses.replace(
                    old,
                    placements=tuple(sorted(self._race_placements.items())),
                )
                self._save_match_record()

    def _make_results_callback(
        self,
        race_index: int,
        match_seq: int,
        match_dir: Path | None,
    ):
        """Return a callback that stores Gemini race results into the
        correct ``RaceInfo``, or routes them to disk if the match is stale.

        Runs on a background thread."""
        def _on_results(parsed: dict | None, placements: list[tuple[int, str]]) -> None:
            with self._races_lock:
                if match_seq != self._match_seq:
                    self._apply_stale_results(
                        match_dir, race_index, parsed, placements,
                    )
                    return
                if race_index < 0 or race_index >= len(self._races):
                    logger.warning(
                        "Gemini results callback: race index %d out of range "
                        "(have %d races)", race_index, len(self._races),
                    )
                    return
                old = self._races[race_index]
                self._races[race_index] = dataclasses.replace(
                    old,
                    placements=tuple(placements),
                    gemini_results=parsed,
                )
                if placements:
                    logger.info("Race %d results set (%d placements, track: %s)",
                                race_index + 1, len(placements), old.track_name)
                    for p, name in placements:
                        logger.info("  %2d. %s", p, name)
                else:
                    logger.warning("Race %d: Gemini returned no placements", race_index + 1)
                self._save_match_record()
        return _on_results

    # -- match record persistence -----------------------------------------

    def _save_match_record(self) -> None:
        """Snapshot current match state and write ``match.json`` to the
        match folder.  Cheap and idempotent — call freely whenever match
        data changes.

        **Strict opt-in rule**: persistence is gated on
        ``_match_started_at`` being set.  That field is only populated by
        either real settings detection in :meth:`_handle_waiting` or an
        explicit :meth:`start_manual_match` call.  This prevents "ghost
        matches" — folders created lazily by frame-saving handlers after
        a manual ``advance()`` — from polluting the history store with
        empty or partial records.

        Callers may optionally hold ``_races_lock``; this method does not
        acquire it.  Reads of ``self._races`` are safe under the GIL because
        the only mutation by background threads is single-index assignment
        via :func:`dataclasses.replace`.
        """
        if self._match_dir is None or self._match_settings is None:
            return
        if self._match_started_at is None:
            return  # strict rule: never persist a match that wasn't started
        try:
            record = self._build_match_record()
        except Exception:
            logger.exception("Failed to build match record")
            return
        if record.final_standings is not None:
            self._save_match_table(self._match_dir, record)
        try:
            record.save(self._match_dir)
        except Exception:
            logger.exception("Failed to save match record")

    @staticmethod
    def _save_match_table(match_dir: Path, record: MatchRecord) -> None:
        """Render the Lorenzi-style results table and save it to
        ``match_dir/table.png``. Safe to call whenever final standings are
        present; logs and swallows any rendering error."""
        try:
            png = generate_table(record)
            (match_dir / TABLE_FILE).write_bytes(png)
        except Exception:
            logger.exception("Failed to generate match table for %s", match_dir.name)

    def _build_match_record(self) -> MatchRecord:
        """Build a :class:`MatchRecord` snapshot of the current match."""
        assert self._match_settings is not None
        settings = MatchSettingsRecord(
            cc_class=self._match_settings.cc_class,
            teams=self._match_settings.teams,
            items=self._match_settings.items,
            com_difficulty=self._match_settings.com_difficulty,
            race_count=self._match_settings.race_count,
            intermission=self._match_settings.intermission,
        )
        races = [
            self._build_race_record(i + 1, race)
            for i, race in enumerate(self._races)
        ]
        return MatchRecord(
            match_id=self._match_dir.name if self._match_dir else "",
            started_at=(self._match_started_at or datetime.now()).isoformat(),
            completed_at=(
                self._match_completed_at.isoformat()
                if self._match_completed_at else None
            ),
            settings=settings,
            races=races,
            final_standings=self._build_final_standings(),
        )

    @staticmethod
    def _race_fields_from_gemini(
        gemini: dict,
    ) -> tuple[str | None, list[TeamGroup], list[PlayerPlacement]]:
        """Extract ``(mode, teams, placements)`` from a Gemini *race results*
        response dict.  Used by both the live and stale write paths."""
        mode = gemini.get("mode")
        teams_list = gemini.get("teams") or []
        teams: list[TeamGroup] = []
        all_placements: list[PlayerPlacement] = []
        for team in teams_list:
            tg_players: list[PlayerPlacement] = []
            for p in team.get("players", []):
                if p.get("place") is None:
                    continue
                tg_players.append(PlayerPlacement(
                    place=int(p["place"]),
                    name=str(p.get("name", "")),
                ))
            teams.append(TeamGroup(
                name=team.get("name"),
                tag=team.get("tag"),
                points=team.get("race_points"),
                winner=team.get("race_winner"),
                players=tg_players,
            ))
            all_placements.extend(tg_players)
        all_placements.sort(key=lambda pl: pl.place)
        return mode, teams, all_placements

    @staticmethod
    def _final_standings_from_gemini(gemini: dict) -> FinalStandings:
        """Build :class:`FinalStandings` from a Gemini *match results*
        response dict.  Used by both the live and stale write paths."""
        mode = gemini.get("mode")
        teams_list = gemini.get("teams") or []
        teams: list[TeamGroup] = []
        all_players: list[PlayerPlacement] = []
        for team in teams_list:
            tg_players: list[PlayerPlacement] = []
            for p in team.get("players", []):
                if p.get("place") is None:
                    continue
                score = p.get("score")
                tg_players.append(PlayerPlacement(
                    place=int(p["place"]),
                    name=str(p.get("name", "")),
                    score=int(score) if score is not None else None,
                ))
            teams.append(TeamGroup(
                name=team.get("name"),
                tag=team.get("tag"),
                points=team.get("points"),
                winner=team.get("winner"),
                players=tg_players,
            ))
            all_players.extend(tg_players)
        all_players.sort(key=lambda pl: pl.place)
        return FinalStandings(mode=mode, players=all_players, teams=teams)

    @classmethod
    def _build_race_record(cls, race_number: int, race: RaceInfo) -> RaceRecord:
        """Convert a :class:`RaceInfo` into a serialisable :class:`RaceRecord`.

        When Gemini structured data is available it is used as the source
        of truth (richer team info); otherwise the flat OCR placements are
        used and team fields are left ``None``.
        """
        if race.gemini_results:
            mode, teams, placements = cls._race_fields_from_gemini(race.gemini_results)
            teams_field: list[TeamGroup] | None = teams
        else:
            mode = None
            teams_field = None
            placements = [
                PlayerPlacement(place=int(p), name=str(n))
                for p, n in race.placements
            ]

        return RaceRecord(
            race_number=race_number,
            track_name=race.track_name,
            players=list(race.players),
            user_rank=race.race_rank,
            mode=mode,
            placements=placements,
            teams=teams_field,
        )

    def _build_final_standings(self) -> FinalStandings | None:
        """Build :class:`FinalStandings` from Gemini structured results when
        available, otherwise from the flat OCR ``(name, score)`` list."""
        gemini = self._gemini_match_results
        if gemini:
            return self._final_standings_from_gemini(gemini)

        flat = self._match_final_results
        if flat:
            # OCR path: ordered list of (name, score) — derive place from index.
            return FinalStandings(
                mode=None,
                players=[
                    PlayerPlacement(place=i + 1, name=name, score=score)
                    for i, (name, score) in enumerate(flat)
                ],
                teams=None,
            )

        return None

    # -- stale-callback writes --------------------------------------------

    def _load_stale_record(self, match_dir: Path | None) -> MatchRecord | None:
        """Load the on-disk ``match.json`` for an orphaned in-flight callback.

        Returns ``None`` (and logs a warning) if the file is missing or
        unreadable — the callback then silently drops its result, which is
        the right thing to do for an orphaned write.
        """
        if match_dir is None:
            return None
        try:
            return MatchRecord.load(match_dir)
        except Exception:
            logger.warning(
                "Stale callback: failed to load match %s",
                match_dir.name, exc_info=True,
            )
            return None

    def _save_stale_record(self, match_dir: Path, record: MatchRecord) -> None:
        try:
            record.save(match_dir)
        except OSError:
            logger.exception("Failed to save stale match %s", match_dir.name)

    def _apply_stale_rank(
        self, match_dir: Path | None, race_index: int, rank: int | None,
    ) -> None:
        """Persist a Gemini rank result whose match has already been
        replaced in the live state machine."""
        record = self._load_stale_record(match_dir)
        if record is None:
            return
        if not (0 <= race_index < len(record.races)):
            logger.warning(
                "Stale match %s: race index %d out of range (have %d races)",
                match_dir.name if match_dir else "?", race_index, len(record.races),
            )
            return
        record.races[race_index].user_rank = rank
        assert match_dir is not None  # _load_stale_record returned non-None
        self._save_stale_record(match_dir, record)
        logger.info(
            "Stale match %s: race %d rank set to %s",
            match_dir.name, race_index + 1, rank,
        )

    def _apply_stale_results(
        self,
        match_dir: Path | None,
        race_index: int,
        parsed: dict | None,
        placements: list[tuple[int, str]],
    ) -> None:
        """Persist Gemini race results whose match has already been replaced."""
        record = self._load_stale_record(match_dir)
        if record is None:
            return
        if not (0 <= race_index < len(record.races)):
            logger.warning(
                "Stale match %s: race index %d out of range (have %d races)",
                match_dir.name if match_dir else "?", race_index, len(record.races),
            )
            return
        target = record.races[race_index]
        if parsed:
            mode, teams, places = self._race_fields_from_gemini(parsed)
            target.mode = mode
            target.teams = teams
            target.placements = places
        elif placements:
            target.placements = [
                PlayerPlacement(place=int(p), name=str(n)) for p, n in placements
            ]
        else:
            return  # nothing to write
        assert match_dir is not None
        self._save_stale_record(match_dir, record)
        logger.info(
            "Stale match %s: race %d results set (%d placements)",
            match_dir.name, race_index + 1, len(target.placements),
        )

    def _apply_stale_match_results(
        self,
        match_dir: Path | None,
        parsed: dict | None,
        results: list[tuple[str, int]],
    ) -> None:
        """Persist Gemini final-standings whose match has already been replaced."""
        record = self._load_stale_record(match_dir)
        if record is None:
            return
        if parsed:
            record.final_standings = self._final_standings_from_gemini(parsed)
        elif results:
            record.final_standings = FinalStandings(
                mode=None,
                players=[
                    PlayerPlacement(place=i + 1, name=name, score=score)
                    for i, (name, score) in enumerate(results)
                ],
                teams=None,
            )
        else:
            return  # nothing to write
        if record.completed_at is None:
            record.completed_at = datetime.now().isoformat()
        assert match_dir is not None
        if record.final_standings is not None:
            self._save_match_table(match_dir, record)
        self._save_stale_record(match_dir, record)
        player_count = (
            len(record.final_standings.players)
            if record.final_standings else 0
        )
        logger.info(
            "Stale match %s: final standings set (%d players)",
            match_dir.name, player_count,
        )

    # -- debug helpers -----------------------------------------------------

    def _race_dir(self, race_num: int) -> Path:
        """Return the debug directory for a given race, creating it if needed."""
        if self._match_dir is None:
            self._match_dir = _MATCHES_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            self._match_dir.mkdir(parents=True, exist_ok=True)
        race_dir = self._match_dir / f"race_{race_num:02d}"
        race_dir.mkdir(exist_ok=True)
        return race_dir

    def _debug_placements_dir(self, race_num: int) -> Path:
        d = self._race_dir(race_num) / "debug_placements"
        d.mkdir(exist_ok=True)
        return d

    def _dump_pre_placement_buffer(self) -> None:
        """Write the rolling pre-placement frame buffer to
        ``<race_dir>/debug_placements/pre_NN.png``. Called when the first
        placement frame arrives in debug mode."""
        if not self._pre_results_buffer:
            return
        race_num = len(self._races)
        out_dir = self._debug_placements_dir(race_num)
        for i, buf in enumerate(self._pre_results_buffer, 1):
            cv2.imwrite(str(out_dir / f"pre_{i:02d}.png"), buf)
        logger.info(
            "Debug: saved %d pre-placement frames for race %d",
            len(self._pre_results_buffer), race_num,
        )
        self._pre_results_buffer.clear()

    def _save_post_placement_frame(self, frame: np.ndarray, n: int) -> None:
        race_num = len(self._races)
        out_dir = self._debug_placements_dir(race_num)
        cv2.imwrite(str(out_dir / f"post_{n:02d}.png"), frame)
        logger.debug(
            "Debug: saved post-placement frame %d for race %d", n, race_num,
        )

    def _save_race_frame(self, frame: np.ndarray, race_num: int, label: str) -> None:
        race_dir = self._race_dir(race_num)
        safe = re.sub(r"[^\w\-]", "_", label)
        path = race_dir / f"{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame: %s", path)

    def _save_frame(self, frame: np.ndarray, label: str) -> None:
        if self._match_dir is None:
            self._match_dir = _MATCHES_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            self._match_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-]", "_", label)
        path = self._match_dir / f"{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame: %s", path)

    # -- transitions -------------------------------------------------------

    def _transition(self, new_state: GameState) -> None:
        logger.info("State: %s -> %s", self._state.name, new_state.name)
        if new_state is GameState.WAITING_FOR_TRACK_PICK:
            self._current_race += 1
        elif new_state is GameState.WAITING_FOR_MATCH:
            self._current_race = 0
        self._state = new_state
        self._state_entered_at = time.monotonic()
