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
from mktracker.detection.race_result import RaceResultDetector
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
        self._accumulated_placements: dict[int, str] = {}

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
        self._accumulated_placements = {}
        self._track_detector._last_match_time = 0.0
        self._transition(GameState.WAITING_FOR_MATCH)

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

    def _handle_race_active(self, frame: np.ndarray) -> None:
        if not self._result_detector.is_active(frame):
            return
        results = self._result_detector.read_results(frame)
        if results is None:
            # Detected a results panel but no '+' deltas → overall standings.
            # Race results were already collected (or skipped). Move on.
            logger.info("Overall standings detected, skipping")
            self._finalize_results()
            return
        self._accumulated_placements.update(results)
        self._save_frame(frame, f"race_{len(self._races):02d}_result")
        logger.info(
            "Race results detected (%d placements so far)",
            len(self._accumulated_placements),
        )
        self._transition(GameState.READING_RESULTS)

    def _handle_reading_results(self, frame: np.ndarray) -> None:
        if not self._result_detector.is_active(frame):
            # Results screen gone — finalize what we have.
            self._finalize_results()
            return
        results = self._result_detector.read_results(frame)
        if results is None:
            # Switched to overall standings — finalize.
            self._finalize_results()
            return
        new_count = 0
        for num, name in results.items():
            if num not in self._accumulated_placements:
                self._accumulated_placements[num] = name
                new_count += 1
        if new_count > 0:
            self._save_frame(frame, f"race_{len(self._races):02d}_result_more")
            logger.info(
                "Read %d new placements (%d total)",
                new_count,
                len(self._accumulated_placements),
            )

    def _finalize_results(self) -> None:
        if self._accumulated_placements and self._races:
            # Attach placements to the most recent race.
            last = self._races[-1]
            self._races[-1] = RaceInfo(
                track_name=last.track_name,
                players=last.players,
                placements=dict(self._accumulated_placements),
            )
            logger.info(
                "Final race results for %s (%d placements):",
                last.track_name,
                len(self._accumulated_placements),
            )
            for num in sorted(self._accumulated_placements):
                logger.info(
                    "  %2d: %s", num, self._accumulated_placements[num]
                )
        self._accumulated_placements = {}
        self._pending_track = None
        self._track_detector._last_match_time = time.monotonic()
        self._transition(GameState.RACING)

    # -- debug helpers -----------------------------------------------------

    def _save_frame(self, frame: np.ndarray, label: str) -> None:
        if self._match_dir is None:
            return
        safe = re.sub(r"[^\w\-]", "_", label)
        path = self._match_dir / f"{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame: %s", path)

    # -- transitions -------------------------------------------------------

    def _transition(self, new_state: GameState) -> None:
        logger.info("State: %s -> %s", self._state.name, new_state.name)
        self._state = new_state
        self._state_entered_at = time.monotonic()
