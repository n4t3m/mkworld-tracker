"""Standardised, on-disk representation of a Mario Kart match.

A :class:`MatchRecord` captures everything we know about a single match: the
settings, every race that was played, and the final standings.  Records are
persisted as ``match.json`` next to the per-match debug frames in
``matches/<timestamp>/``, so the existing folder structure doubles as
the match history store.

The schema is intentionally a superset of both the OCR and Gemini detection
paths, with optional fields (``mode``, ``teams``) for the richer per-team
data that only Gemini provides.  :func:`list_matches` scans
``matches/`` and returns every persisted record, newest first — this
is the entry point for the future "match history" UI.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MATCH_FILE = "match.json"
DEFAULT_MATCHES_DIR = Path("matches")


@dataclasses.dataclass
class PlayerPlacement:
    place: int
    name: str
    score: int | None = None  # Cumulative score; only set in final standings.

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"place": self.place, "name": self.name}
        if self.score is not None:
            d["score"] = self.score
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerPlacement":
        score = data.get("score")
        return cls(
            place=int(data["place"]),
            name=str(data["name"]),
            score=int(score) if score is not None else None,
        )


@dataclasses.dataclass
class TeamGroup:
    name: str | None
    points: int | None
    winner: bool | None
    players: list[PlayerPlacement]
    tag: str | None = None  # Clan tag prefix stripped from player names by Gemini

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tag": self.tag,
            "points": self.points,
            "winner": self.winner,
            "players": [p.to_dict() for p in self.players],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamGroup":
        return cls(
            name=data.get("name"),
            tag=data.get("tag"),
            points=data.get("points"),
            winner=data.get("winner"),
            players=[PlayerPlacement.from_dict(p) for p in data.get("players", [])],
        )


@dataclasses.dataclass
class RaceRecord:
    race_number: int
    track_name: str | None
    players: list[str]
    user_rank: int | None = None
    # "no_teams" | "two_teams" | "three_teams" | "four_teams" | None
    mode: str | None = None
    placements: list[PlayerPlacement] = dataclasses.field(default_factory=list)
    teams: list[TeamGroup] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_number": self.race_number,
            "track_name": self.track_name,
            "players": list(self.players),
            "user_rank": self.user_rank,
            "mode": self.mode,
            "placements": [p.to_dict() for p in self.placements],
            "teams": (
                [t.to_dict() for t in self.teams] if self.teams is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceRecord":
        teams_data = data.get("teams")
        return cls(
            race_number=int(data["race_number"]),
            track_name=data.get("track_name"),
            players=list(data.get("players", [])),
            user_rank=data.get("user_rank"),
            mode=data.get("mode"),
            placements=[
                PlayerPlacement.from_dict(p) for p in data.get("placements", [])
            ],
            teams=(
                [TeamGroup.from_dict(t) for t in teams_data]
                if teams_data is not None
                else None
            ),
        )


@dataclasses.dataclass
class FinalStandings:
    mode: str | None = None
    players: list[PlayerPlacement] = dataclasses.field(default_factory=list)
    teams: list[TeamGroup] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "players": [p.to_dict() for p in self.players],
            "teams": (
                [t.to_dict() for t in self.teams] if self.teams is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalStandings":
        teams_data = data.get("teams")
        return cls(
            mode=data.get("mode"),
            players=[
                PlayerPlacement.from_dict(p) for p in data.get("players", [])
            ],
            teams=(
                [TeamGroup.from_dict(t) for t in teams_data]
                if teams_data is not None
                else None
            ),
        )


def race_fields_from_gemini(
    gemini: dict[str, Any],
) -> tuple[str | None, list[TeamGroup], list[PlayerPlacement]]:
    """Extract ``(mode, teams, placements)`` from a Gemini *race results* response.

    When the total visible player count matches a known MK scoring table
    (12 or 24), each team's ``points`` is *derived* locally from its
    placements rather than trusting ``race_points`` from the model —
    Gemini reads placements reliably but its arithmetic on the +score
    column is unreliable.  For any other player count Gemini's reported
    ``race_points`` is used as-is.
    """
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

    total_players = sum(len(t.players) for t in teams)
    # Lazy import: team_scoring imports from this module, so a top-level
    # import would form a cycle.  By the time this function is invoked
    # both modules are fully loaded.
    from mktracker.team_scoring import DERIVED_PLAYER_COUNTS, points_for_place
    if total_players in DERIVED_PLAYER_COUNTS:
        for team in teams:
            team.points = sum(
                points_for_place(p.place, total_players) for p in team.players
            )

    return mode, teams, all_placements


def final_standings_from_gemini(gemini: dict[str, Any]) -> FinalStandings:
    """Build :class:`FinalStandings` from a Gemini *match results* response."""
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


@dataclasses.dataclass
class MatchSettingsRecord:
    cc_class: str
    teams: str
    items: str
    com_difficulty: str
    race_count: int
    intermission: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchSettingsRecord":
        return cls(
            cc_class=str(data["cc_class"]),
            teams=str(data["teams"]),
            items=str(data["items"]),
            com_difficulty=str(data["com_difficulty"]),
            race_count=int(data["race_count"]),
            intermission=str(data["intermission"]),
        )


@dataclasses.dataclass
class MatchRecord:
    match_id: str
    started_at: str  # ISO 8601 timestamp
    completed_at: str | None  # ISO 8601, or None if match never finalised
    settings: MatchSettingsRecord
    races: list[RaceRecord] = dataclasses.field(default_factory=list)
    final_standings: FinalStandings | None = None
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "match_id": self.match_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "settings": self.settings.to_dict(),
            "races": [r.to_dict() for r in self.races],
            "final_standings": (
                self.final_standings.to_dict() if self.final_standings else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchRecord":
        return cls(
            version=int(data.get("version", SCHEMA_VERSION)),
            match_id=str(data["match_id"]),
            started_at=str(data["started_at"]),
            completed_at=data.get("completed_at"),
            settings=MatchSettingsRecord.from_dict(data["settings"]),
            races=[RaceRecord.from_dict(r) for r in data.get("races", [])],
            final_standings=(
                FinalStandings.from_dict(data["final_standings"])
                if data.get("final_standings")
                else None
            ),
        )

    def save(self, match_dir: Path) -> Path:
        """Atomically write this record to ``<match_dir>/match.json``."""
        match_dir.mkdir(parents=True, exist_ok=True)
        path = match_dir / MATCH_FILE
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, match_dir: Path) -> "MatchRecord":
        path = match_dir / MATCH_FILE
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def list_matches(matches_dir: Path = DEFAULT_MATCHES_DIR) -> list[MatchRecord]:
    """Return every persisted match in *matches_dir*, newest first.

    Folders without a ``match.json`` are skipped silently — they are
    legacy debug-only matches predating the standardised format.
    """
    if not matches_dir.exists():
        return []
    records: list[MatchRecord] = []
    for sub in sorted(matches_dir.iterdir(), reverse=True):
        if not sub.is_dir() or not (sub / MATCH_FILE).exists():
            continue
        try:
            records.append(MatchRecord.load(sub))
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            logger.warning(
                "Failed to load match record from %s", sub, exc_info=True,
            )
    return records
