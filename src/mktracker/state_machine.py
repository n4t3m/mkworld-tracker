from __future__ import annotations

import dataclasses
import logging
import time
from enum import Enum, auto

import numpy as np

from mktracker.detection.match_settings import MatchSettings, MatchSettingsDetector
from mktracker.detection.player_reader import PlayerReader
from mktracker.detection.track_select import TrackSelectDetector

logger = logging.getLogger(__name__)

# How long to wait after match settings are detected before looking for tracks.
_MATCH_START_DELAY = 5.0


class GameState(Enum):
    WAITING_FOR_MATCH = auto()
    MATCH_STARTED = auto()
    RACING = auto()


@dataclasses.dataclass(frozen=True)
class RaceInfo:
    track_name: str
    players: tuple[str, ...]


class GameStateMachine:
    """Orchestrates detection across game states.

    Call ``update(frame)`` on each detection cycle.
    """

    def __init__(self) -> None:
        self._state = GameState.WAITING_FOR_MATCH
        self._state_entered_at = time.monotonic()

        self._match_settings: MatchSettings | None = None
        self._races: list[RaceInfo] = []

        self._match_detector = MatchSettingsDetector()
        self._track_detector = TrackSelectDetector()
        self._player_reader = PlayerReader()

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

    # -- state handlers ----------------------------------------------------

    def _handle_waiting(self, frame: np.ndarray) -> None:
        settings = self._match_detector.detect(frame)
        if settings is None:
            return
        self._match_settings = settings
        self._races = []
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
        track_name = result["track_name"]
        player_names = tuple(
            p.name for p in self._player_reader.read_players(frame)
        )

        race = RaceInfo(track_name=track_name, players=player_names)
        self._races.append(race)

        total = self._match_settings.race_count if self._match_settings else "?"
        logger.info("Race %d/%s: %s", len(self._races), total, track_name)
        for name in player_names:
            logger.info("  - %s", name)

    # -- transitions -------------------------------------------------------

    def _transition(self, new_state: GameState) -> None:
        logger.info("State: %s -> %s", self._state.name, new_state.name)
        self._state = new_state
        self._state_entered_at = time.monotonic()
