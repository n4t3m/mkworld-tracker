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

from mktracker.detection.match_results import MatchResultDetector
from mktracker.detection.match_settings import MatchSettings, MatchSettingsDetector
from mktracker.detection.player_reader import PlayerReader
from mktracker.detection.race_finish import RaceFinishDetector
from mktracker.detection.race_rank import RaceRankDetector
from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.track_select import TrackSelectDetector

logger = logging.getLogger(__name__)

# How long to wait after match settings are detected before looking for tracks.
_MATCH_START_DELAY = 5.0

# Directory where debug frames are saved.
_DEBUG_DIR = Path("debug_frames")
_DEBUG_FINISH_DIR = Path("debug_finish")
_DEBUG_RANK_DIR = Path("debug_rank")


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
        self._pending_track: str | None = None

        self._match_detector = MatchSettingsDetector()
        self._track_detector = TrackSelectDetector()
        self._player_reader = PlayerReader()
        self._finish_detector = RaceFinishDetector()
        self._rank_detector = RaceRankDetector()
        self._result_detector = RaceResultDetector()
        self._match_result_detector = MatchResultDetector()

        self._race_placements: dict[int, str] = {}
        self._race_placement_quality: dict[int, int] = {}
        self._seen_race_results = False
        self._match_final_results: list[tuple[str, int]] | None = None

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
    def current_race(self) -> int:
        """Current race number (1-based), or 0 if no match is active."""
        return self._current_race

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
    def races(self) -> list[RaceInfo]:
        return list(self._races)

    def reset(self) -> None:
        """Reset to WAITING_FOR_MATCH, clearing all match data."""
        self._match_settings = None
        self._races = []
        self._current_race = 0
        self._pending_track = None
        self._race_placements = {}
        self._race_placement_quality = {}
        self._seen_race_results = False
        self._match_final_results = None
        self._track_detector._last_match_time = 0.0
        self._transition(GameState.WAITING_FOR_MATCH)

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
        self._transition(GameState.DETECTING_RACE_RANK)

    def _handle_detecting_rank(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self._state_entered_at

        if elapsed < 1.0:
            return

        # Save the full frame to debug_rank/ for LLM experimentation.
        _DEBUG_RANK_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(_DEBUG_RANK_DIR / f"{ts}.png"), frame)
        self._save_race_frame(frame, len(self._races), "rank")
        logger.info("Race rank frame captured (1s after FINISH)")

        # # Smart detection logic (commented out for now):
        # crop = self._rank_detector.detect(frame)
        # if crop is None:
        #     if elapsed > 10.0:
        #         logger.warning("Race rank detection timed out")
        #     else:
        #         return
        # else:
        #     self._save_race_frame(crop, len(self._races), "rank")
        #     logger.info("Race rank detected — crop saved")

        self._transition(GameState.READING_RACE_RESULTS)

    def _handle_reading_results(self, frame: np.ndarray) -> None:
        elapsed = time.monotonic() - self._state_entered_at

        if elapsed > 30.0:
            logger.warning(
                "Race results reading timed out after %.0fs", elapsed,
            )
            self._finalise_race_placements()
            self._transition(self._state_after_race_results())
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

        teams = self._match_settings.teams if self._match_settings else "No Teams"
        player_count = self.player_count or 12

        result = self._match_result_detector.detect(
            frame, teams=teams, player_count=player_count,
        )
        if result is None:
            return

        self._match_final_results = result["results"]
        self._save_frame(frame, "match_results")
        logger.info("Match final results:")
        for name, points in result["results"]:
            logger.info("  %s: %d pts", name, points)

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

    def _save_race_frame(self, frame: np.ndarray, race_num: int, label: str) -> None:
        if self._match_dir is None:
            self._match_dir = _DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
            self._match_dir.mkdir(parents=True, exist_ok=True)
        race_dir = self._match_dir / f"race_{race_num:02d}"
        race_dir.mkdir(exist_ok=True)
        safe = re.sub(r"[^\w\-]", "_", label)
        path = race_dir / f"{safe}.png"
        cv2.imwrite(str(path), frame)
        logger.debug("Saved debug frame: %s", path)

    def _save_frame(self, frame: np.ndarray, label: str) -> None:
        if self._match_dir is None:
            self._match_dir = _DEBUG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
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
