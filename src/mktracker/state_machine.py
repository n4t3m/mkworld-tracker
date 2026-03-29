from __future__ import annotations

import dataclasses
import logging
import re
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from mktracker.detection.match_settings import MatchSettings, MatchSettingsDetector
from mktracker.detection.player_reader import PlayerReader
from mktracker.detection.race_result import RaceResultAccumulator, RaceResultDetector
from mktracker.detection.track_select import TrackSelectDetector

logger = logging.getLogger(__name__)

# How long to wait after match settings are detected before looking for tracks.
_MATCH_START_DELAY = 5.0

# Directory where debug frames are saved.
_DEBUG_DIR = Path("debug_frames")


class GameState(Enum):
    WAITING_FOR_MATCH = auto()
    MATCH_STARTED = auto()
    RACING = auto()
    READING_PLAYERS = auto()
    RACE_ACTIVE = auto()
    READING_RESULTS = auto()


@dataclasses.dataclass(frozen=True)
class RaceInfo:
    track_name: str
    players: tuple[str, ...]
    placements: dict[int, str] = dataclasses.field(default_factory=dict)


class GameStateMachine:
    """Orchestrates detection across game states.

    Call ``update(frame)`` on each detection cycle.
    """

    def __init__(self) -> None:
        self._state = GameState.WAITING_FOR_MATCH
        self._state_entered_at = time.monotonic()

        self._match_settings: MatchSettings | None = None
        self._races: list[RaceInfo] = []
        self._match_dir: Path | None = None
        self._pending_track: str | None = None
        self._result_accumulator = RaceResultAccumulator()
        self._stale_result_frames: int = 0
        self._result_frame_buffer: list[np.ndarray] = []

        self._match_detector = MatchSettingsDetector()
        self._track_detector = TrackSelectDetector()
        self._player_reader = PlayerReader()
        self._result_detector = RaceResultDetector()

        logger.info("State: WAITING_FOR_MATCH")

    # -- public properties -------------------------------------------------

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def match_settings(self) -> MatchSettings | None:
        return self._match_settings

    @property
    def races(self) -> list[RaceInfo]:
        return list(self._races)

    def reset(self) -> None:
        """Reset to WAITING_FOR_MATCH, clearing all match data."""
        self._match_settings = None
        self._races = []
        self._pending_track = None
        self._result_accumulator.clear()
        self._result_frame_buffer.clear()
        self._stale_result_frames = 0
        self._track_detector._last_match_time = 0.0
        self._transition(GameState.WAITING_FOR_MATCH)

    _ADVANCE_ORDER = {
        GameState.WAITING_FOR_MATCH: GameState.MATCH_STARTED,
        GameState.MATCH_STARTED: GameState.RACING,
        GameState.RACING: GameState.RACE_ACTIVE,
        GameState.READING_PLAYERS: GameState.RACE_ACTIVE,
        GameState.RACE_ACTIVE: GameState.RACING,
        GameState.READING_RESULTS: GameState.RACING,
    }

    def advance(self) -> None:
        """Manually advance to the next logical state."""
        next_state = self._ADVANCE_ORDER.get(self._state)
        if next_state is None:
            return
        if self._state is GameState.READING_RESULTS:
            self._finalize_results()
        else:
            self._transition(next_state)

    # -- main update -------------------------------------------------------

    def update(self, frame: np.ndarray) -> None:
        if self._state is GameState.WAITING_FOR_MATCH:
            self._handle_waiting(frame)
        elif self._state is GameState.MATCH_STARTED:
            self._handle_match_started()
        elif self._state is GameState.RACING:
            self._handle_racing(frame)
        elif self._state is GameState.READING_PLAYERS:
            self._handle_reading_players(frame)
        elif self._state is GameState.RACE_ACTIVE:
            self._handle_race_active(frame)
        elif self._state is GameState.READING_RESULTS:
            self._handle_reading_results(frame)

    # -- state handlers ----------------------------------------------------

    def _handle_waiting(self, frame: np.ndarray) -> None:
        settings = self._match_detector.detect(frame)
        if settings is None:
            return
        self._match_settings = settings
        self._races = []
        self._match_dir = _DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        self._transition(GameState.MATCH_STARTED)

    def _handle_match_started(self) -> None:
        elapsed = time.monotonic() - self._state_entered_at
        if elapsed >= _MATCH_START_DELAY:
            self._transition(GameState.RACING)

    def _handle_racing(self, frame: np.ndarray) -> None:
        result = self._track_detector.detect(frame)
        if result is None:
            return
        self._pending_track = result["track_name"]
        self._save_frame(frame, f"race_{len(self._races) + 1:02d}_{self._pending_track}_track")
        logger.info("Track detected: %s — reading players on next frame", self._pending_track)
        self._transition(GameState.READING_PLAYERS)

    def _handle_reading_players(self, frame: np.ndarray) -> None:
        player_names = tuple(
            p.name for p in self._player_reader.read_players(frame)
        )

        race = RaceInfo(track_name=self._pending_track, players=player_names)
        self._races.append(race)
        self._save_frame(frame, f"race_{len(self._races):02d}_{self._pending_track}_players")

        total = self._match_settings.race_count if self._match_settings else "?"
        logger.info("Race %d/%s: %s", len(self._races), total, self._pending_track)
        for name in player_names:
            logger.info("  - %s", name)

        self._transition(GameState.RACE_ACTIVE)

    # Minimum placements in a single frame to confirm it's a real results screen.
    _MIN_PLACEMENTS_TO_CONFIRM = 3

    def _handle_race_active(self, frame: np.ndarray) -> None:
        if not self._result_detector.is_active(frame):
            return
        results = self._result_detector.read_results(frame)
        if len(results) < self._MIN_PLACEMENTS_TO_CONFIRM:
            self._save_result_frame(frame, f"below_threshold_{len(results)}")
            logger.debug(
                "is_active with %d placements (need %d), ignoring",
                len(results),
                self._MIN_PLACEMENTS_TO_CONFIRM,
            )
            return
        self._result_accumulator.add_frame(results)
        self._stale_result_frames = 0
        self._save_frame(frame, f"race_{len(self._races):02d}_result")
        self._save_result_frame(frame, f"result_{len(results)}_placements")
        logger.info(
            "Race results detected (%d placements so far)",
            self._result_accumulator.placement_count,
        )
        self._transition(GameState.READING_RESULTS)

    def _handle_reading_results(self, frame: np.ndarray) -> None:
        # Just buffer frames cheaply — OCR happens in _finalize_results.
        if self._result_detector.is_active(frame):
            self._result_frame_buffer.append(frame)
            self._stale_result_frames = 0
        else:
            self._stale_result_frames += 1
            if self._stale_result_frames >= 3:
                logger.info(
                    "Results screen ended, processing %d buffered frames",
                    len(self._result_frame_buffer),
                )
                self._finalize_results()

    def _finalize_results(self) -> None:
        # Batch-process all buffered frames now.
        for i, buffered in enumerate(self._result_frame_buffer):
            results = self._result_detector.read_results(buffered)
            self._result_accumulator.add_frame(results)
            self._save_result_frame(buffered, f"batch_{i:03d}_{len(results)}")
        self._result_frame_buffer.clear()

        placements = self._result_accumulator.finalize()
        if placements:
            if self._races:
                last = self._races[-1]
                self._races[-1] = RaceInfo(
                    track_name=last.track_name,
                    players=last.players,
                    placements=placements,
                )
            race_label = self._races[-1].track_name if self._races else "unknown"
            logger.info(
                "Final race results for %s (%d placements):",
                race_label,
                len(placements),
            )
            for num in sorted(placements):
                logger.info("  %2d: %s", num, placements[num])
        self._result_accumulator.clear()
        self._stale_result_frames = 0
        self._pending_track = None
        self._track_detector._last_match_time = time.monotonic()
        self._transition(GameState.RACING)

    # -- debug helpers -----------------------------------------------------

    _result_frame_counter: int = 0

    def _save_frame(self, frame: np.ndarray, label: str) -> None:
        if self._match_dir is None:
            return
        safe = re.sub(r"[^\w\-]", "_", label)
        path = self._match_dir / f"{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame: %s", path)

    def _save_result_frame(self, frame: np.ndarray, label: str) -> None:
        """Save a result frame to debug_frames/results/ (always available)."""
        result_dir = _DEBUG_DIR / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        GameStateMachine._result_frame_counter += 1
        safe = re.sub(r"[^\w\-]", "_", label)
        path = result_dir / f"{GameStateMachine._result_frame_counter:04d}_{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved result debug frame: %s", path)

    # -- transitions -------------------------------------------------------

    def _transition(self, new_state: GameState) -> None:
        logger.info("State: %s -> %s", self._state.name, new_state.name)
        self._state = new_state
        self._state_entered_at = time.monotonic()
