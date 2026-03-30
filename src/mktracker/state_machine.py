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
from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.track_select import TrackSelectDetector

logger = logging.getLogger(__name__)

# How long to wait after match settings are detected before looking for tracks.
_MATCH_START_DELAY = 5.0

# Directory where debug frames are saved.
_DEBUG_DIR = Path("debug_frames")


class GameState(Enum):
    WAITING_FOR_MATCH = auto()
    MATCH_STARTED = auto()
    WAITING_FOR_TRACK_PICK = auto()
    READING_PLAYERS_IN_RACE = auto()
    WAITING_FOR_RACE_END = auto()
    READING_RACE_RESULTS = auto()


@dataclasses.dataclass(frozen=True)
class RaceInfo:
    track_name: str
    players: tuple[str, ...]
    placements: tuple[tuple[int, str], ...] = ()


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

        self._match_detector = MatchSettingsDetector()
        self._track_detector = TrackSelectDetector()
        self._player_reader = PlayerReader()
        self._finish_detector = RaceFinishDetector()
        self._result_detector = RaceResultDetector()

        self._race_placements: dict[int, str] = {}
        self._race_placement_quality: dict[int, int] = {}
        self._seen_race_results = False

        logger.info("State: WAITING_FOR_MATCH")

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
    def races(self) -> list[RaceInfo]:
        return list(self._races)

    def reset(self) -> None:
        """Reset to WAITING_FOR_MATCH, clearing all match data."""
        self._match_settings = None
        self._races = []
        self._pending_track = None
        self._race_placements = {}
        self._race_placement_quality = {}
        self._seen_race_results = False
        self._track_detector._last_match_time = 0.0
        self._transition(GameState.WAITING_FOR_MATCH)

    _ADVANCE_ORDER = {
        GameState.WAITING_FOR_MATCH: GameState.MATCH_STARTED,
        GameState.MATCH_STARTED: GameState.WAITING_FOR_TRACK_PICK,
        GameState.WAITING_FOR_TRACK_PICK: GameState.READING_PLAYERS_IN_RACE,
        GameState.READING_PLAYERS_IN_RACE: GameState.WAITING_FOR_RACE_END,
        GameState.WAITING_FOR_RACE_END: GameState.READING_RACE_RESULTS,
        GameState.READING_RACE_RESULTS: GameState.WAITING_FOR_TRACK_PICK,
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
        elif self._state is GameState.READING_RACE_RESULTS:
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
            self._transition(GameState.WAITING_FOR_TRACK_PICK)

    def _handle_racing(self, frame: np.ndarray) -> None:
        result = self._track_detector.detect(frame)
        if result is None:
            return
        self._pending_track = result["track_name"]
        self._save_frame(frame, f"race_{len(self._races) + 1:02d}_{self._pending_track}_track")
        logger.info("Track detected: %s — reading players on next frame", self._pending_track)
        self._transition(GameState.READING_PLAYERS_IN_RACE)

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

        self._pending_track = None
        self._transition(GameState.WAITING_FOR_RACE_END)

    def _handle_race_ending(self, frame: np.ndarray) -> None:
        if not self._finish_detector.is_active(frame):
            return
        logger.info("Race finished — FINISH! detected")
        self._track_detector._last_match_time = time.monotonic()
        self._race_placements = {}
        self._race_placement_quality = {}
        self._seen_race_results = False
        self._transition(GameState.READING_RACE_RESULTS)

    def _handle_reading_results(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self._state_entered_at

        if elapsed > 30.0:
            logger.warning(
                "Race results reading timed out after %.0fs", elapsed,
            )
            self._finalise_race_placements()
            self._transition(GameState.WAITING_FOR_TRACK_PICK)
            return

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
                _debug_placement_dir = Path("debug_placements")
                _debug_placement_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = _debug_placement_dir / f"{ts}_placements_{len(self._race_placements):02d}.png"
                cv2.imwrite(str(path), frame)
                logger.info("Saved placement frame: %s", path)
            logger.debug(
                "Race results so far: %d placements", len(self._race_placements),
            )
        elif result["type"] == "overall" and self._seen_race_results:
            logger.info("Overall standings detected — race results complete")
            self._finalise_race_placements()
            self._transition(GameState.WAITING_FOR_TRACK_PICK)

    def _finalise_race_placements(self) -> None:
        """Log and store accumulated race placements."""
        if not self._race_placements:
            logger.info("No race placements captured")
            return

        logger.info("Race placements collected (%d):", len(self._race_placements))
        for p in sorted(self._race_placements):
            logger.info("  %2d. %s", p, self._race_placements[p])

        # Update the most recent RaceInfo with placements.
        if self._races:
            old = self._races[-1]
            self._races[-1] = RaceInfo(
                track_name=old.track_name,
                players=old.players,
                placements=tuple(sorted(self._race_placements.items())),
            )

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
