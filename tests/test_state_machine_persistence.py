"""Tests for the state machine's match-record persistence and stale-callback
routing.

These tests poke directly at ``GameStateMachine`` internals (``_races``,
``_match_dir``, etc.) instead of driving real frames through ``update()`` —
that lets us cover both the live and stale code paths in isolation, without
having to fake every detector.

The two things under test:

1. ``_build_match_record`` correctly converts state machine state into a
   :class:`MatchRecord` for both OCR and Gemini detection paths.
2. The Gemini callback factories correctly route to the live in-memory
   state when the match is current, and to the on-disk ``match.json`` when
   the state machine has moved on (the race condition fix).

We use ``tmp_path`` for ``_match_dir`` so each test is self-contained and
doesn't touch the real ``matches/`` directory.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mktracker.detection.match_settings import MatchSettings
from mktracker.match_record import MatchRecord
from mktracker.state_machine import (
    GameState,
    GameStateMachine,
    RaceInfo,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _settings(race_count: int = 4, teams: str = "No Teams") -> MatchSettings:
    return MatchSettings(
        cc_class="150cc",
        teams=teams,
        items="Normal",
        com_difficulty="No COM",
        race_count=race_count,
        intermission="10 seconds",
    )


def _start_match(sm: GameStateMachine, match_dir: Path, settings: MatchSettings | None = None) -> None:
    """Bring a fresh state machine into a 'match in progress' state with
    ``match_dir`` as its target folder.  Mirrors what ``_handle_waiting``
    does on a real settings detection."""
    sm._match_settings = settings or _settings()
    sm._races = []
    sm._match_completed_at = None
    sm._match_seq += 1
    sm._match_started_at = datetime.now()
    sm._match_dir = match_dir
    match_dir.mkdir(parents=True, exist_ok=True)


def _gemini_race_results(*placements: tuple[int, str]) -> dict:
    return {
        "mode": "no_teams",
        "teams": [{
            "name": None,
            "race_points": None,
            "race_winner": None,
            "players": [{"place": p, "name": n} for p, n in placements],
        }],
    }


def _gemini_match_results(*players: tuple[int, str, int]) -> dict:
    return {
        "mode": "no_teams",
        "teams": [{
            "name": None,
            "points": None,
            "winner": None,
            "players": [
                {"place": p, "name": n, "score": s} for p, n, s in players
            ],
        }],
    }


def _two_team_results(red: list[tuple[int, str]], blue: list[tuple[int, str]]) -> dict:
    return {
        "mode": "two_teams",
        "teams": [
            {
                "name": "Red Team",
                "race_points": sum(13 - p for p, _ in red),
                "race_winner": True,
                "players": [{"place": p, "name": n} for p, n in red],
            },
            {
                "name": "Blue Team",
                "race_points": sum(13 - p for p, _ in blue),
                "race_winner": False,
                "players": [{"place": p, "name": n} for p, n in blue],
            },
        ],
    }


# ---------------------------------------------------------------------------
# __init__ / reset
# ---------------------------------------------------------------------------


class TestInitAndReset:
    def test_initial_state_is_waiting(self):
        sm = GameStateMachine()
        assert sm.state is GameState.WAITING_FOR_MATCH
        assert sm._match_seq == 0
        assert sm._match_started_at is None
        assert sm._match_completed_at is None
        assert sm._match_dir is None

    def test_reset_clears_match_data(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [RaceInfo(track_name="X", players=("Y",))]
        sm._match_completed_at = datetime.now()

        sm.reset()
        assert sm._match_settings is None
        assert sm._races == []
        assert sm._match_started_at is None
        assert sm._match_completed_at is None

    def test_reset_bumps_match_seq(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        seq_before = sm._match_seq
        sm.reset()
        assert sm._match_seq == seq_before + 1


# ---------------------------------------------------------------------------
# _build_match_record
# ---------------------------------------------------------------------------


class TestBuildMatchRecord:
    def test_returns_settings_record(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "20260406_120000",
                     settings=_settings(race_count=8, teams="Two Teams"))
        rec = sm._build_match_record()
        assert rec.settings.cc_class == "150cc"
        assert rec.settings.race_count == 8
        assert rec.settings.teams == "Two Teams"

    def test_match_id_is_folder_name(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "20260406_120000")
        rec = sm._build_match_record()
        assert rec.match_id == "20260406_120000"

    def test_started_at_is_iso_string(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        rec = sm._build_match_record()
        # Should round-trip via fromisoformat without raising.
        datetime.fromisoformat(rec.started_at)

    def test_no_settings_returns_none_safely(self):
        sm = GameStateMachine()
        sm._match_settings = None
        # _save_match_record should be a no-op rather than crashing.
        sm._save_match_record()  # no exception
        # _build_match_record itself asserts settings are present, so we
        # don't call it directly here — the public guarantee is via _save_*.

    def test_races_from_ocr_path(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [
            RaceInfo(
                track_name="Mario Bros. Circuit",
                players=("Alice", "Bob"),
                placements=((1, "Alice"), (2, "Bob")),
                race_rank=1,
            )
        ]
        rec = sm._build_match_record()
        assert len(rec.races) == 1
        race = rec.races[0]
        assert race.race_number == 1
        assert race.track_name == "Mario Bros. Circuit"
        assert race.players == ["Alice", "Bob"]
        assert race.user_rank == 1
        assert race.mode is None
        assert race.teams is None
        assert [p.name for p in race.placements] == ["Alice", "Bob"]
        assert [p.place for p in race.placements] == [1, 2]

    def test_races_prefer_gemini_results_when_present(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [
            RaceInfo(
                track_name="Rainbow Road",
                players=("Alice", "Bob"),
                placements=((1, "Alice"), (2, "Bob")),  # OCR data
                race_rank=2,
                gemini_results=_gemini_race_results((1, "Bob"), (2, "Alice")),
            )
        ]
        rec = sm._build_match_record()
        race = rec.races[0]
        # Gemini ordering wins over the OCR placements field.
        assert [p.name for p in race.placements] == ["Bob", "Alice"]
        assert race.mode == "no_teams"
        assert race.teams is not None
        assert len(race.teams) == 1

    def test_races_two_team_mode(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match", settings=_settings(teams="Two Teams"))
        sm._races = [
            RaceInfo(
                track_name="Rainbow Road",
                players=("R1", "R2", "B1", "B2"),
                gemini_results=_two_team_results(
                    red=[(1, "R1"), (3, "R2")],
                    blue=[(2, "B1"), (4, "B2")],
                ),
            )
        ]
        rec = sm._build_match_record()
        race = rec.races[0]
        assert race.mode == "two_teams"
        assert race.teams is not None
        assert len(race.teams) == 2
        assert race.teams[0].name == "Red Team"
        assert race.teams[0].winner is True
        assert race.teams[1].name == "Blue Team"
        assert race.teams[1].winner is False
        # Combined placements should be sorted across teams.
        assert [p.place for p in race.placements] == [1, 2, 3, 4]
        assert [p.name for p in race.placements] == ["R1", "B1", "R2", "B2"]

    def test_race_numbering_is_one_based(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [
            RaceInfo(track_name="A", players=("X",)),
            RaceInfo(track_name="B", players=("Y",)),
            RaceInfo(track_name="C", players=("Z",)),
        ]
        rec = sm._build_match_record()
        assert [r.race_number for r in rec.races] == [1, 2, 3]

    def test_final_standings_from_ocr_flat_list(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._match_final_results = [("Alice", 180), ("Bob", 170), ("Carol", 160)]
        rec = sm._build_match_record()
        assert rec.final_standings is not None
        assert rec.final_standings.mode is None
        assert rec.final_standings.teams is None
        # Place is derived from list order.
        assert [(p.place, p.name, p.score) for p in rec.final_standings.players] == [
            (1, "Alice", 180),
            (2, "Bob", 170),
            (3, "Carol", 160),
        ]

    def test_final_standings_prefer_gemini_when_present(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._match_final_results = [("X", 0)]  # stale OCR data — should be ignored
        sm._gemini_match_results = _gemini_match_results(
            (1, "Alice", 180), (2, "Bob", 165),
        )
        rec = sm._build_match_record()
        assert rec.final_standings is not None
        assert rec.final_standings.mode == "no_teams"
        assert [p.name for p in rec.final_standings.players] == ["Alice", "Bob"]
        assert [p.score for p in rec.final_standings.players] == [180, 165]

    def test_final_standings_none_when_no_data(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        rec = sm._build_match_record()
        assert rec.final_standings is None


# ---------------------------------------------------------------------------
# _save_match_record
# ---------------------------------------------------------------------------


class TestSaveMatchRecord:
    def test_save_writes_match_json(self, tmp_path: Path):
        sm = GameStateMachine()
        match_dir = tmp_path / "20260406_120000"
        _start_match(sm, match_dir)
        sm._save_match_record()
        assert (match_dir / "match.json").exists()
        rec = MatchRecord.load(match_dir)
        assert rec.match_id == "20260406_120000"

    def test_save_is_idempotent(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._save_match_record()
        sm._save_match_record()
        sm._save_match_record()
        # Three saves, one file.
        assert (tmp_path / "match" / "match.json").exists()
        assert list((tmp_path / "match").glob("*.tmp")) == []

    def test_save_noop_without_match_dir(self):
        sm = GameStateMachine()
        sm._match_settings = _settings()
        sm._match_dir = None
        # Should not raise.
        sm._save_match_record()

    def test_save_noop_without_settings(self, tmp_path: Path):
        sm = GameStateMachine()
        sm._match_dir = tmp_path
        sm._match_settings = None
        sm._save_match_record()
        assert not (tmp_path / "match.json").exists()

    def test_save_after_race_added(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races.append(
            RaceInfo(
                track_name="Mario Bros. Circuit",
                players=("Alice", "Bob"),
                placements=((1, "Alice"), (2, "Bob")),
            )
        )
        sm._save_match_record()
        rec = MatchRecord.load(tmp_path / "match")
        assert len(rec.races) == 1
        assert rec.races[0].track_name == "Mario Bros. Circuit"


# ---------------------------------------------------------------------------
# Live callback path
# ---------------------------------------------------------------------------


class TestLiveCallbacks:
    def test_rank_callback_writes_to_live_state(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [RaceInfo(track_name="A", players=("Alice",))]
        sm._save_match_record()

        cb = sm._make_rank_callback(0, sm._match_seq, sm._match_dir)
        cb(3)
        assert sm._races[0].race_rank == 3
        # Disk record was also updated by _save_match_record.
        rec = MatchRecord.load(tmp_path / "match")
        assert rec.races[0].user_rank == 3

    def test_rank_callback_out_of_range_does_not_crash(self, tmp_path: Path, caplog):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = []
        sm._save_match_record()

        cb = sm._make_rank_callback(5, sm._match_seq, sm._match_dir)
        with caplog.at_level("WARNING"):
            cb(2)
        assert any("out of range" in r.message for r in caplog.records)

    def test_results_callback_writes_to_live_state(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [RaceInfo(track_name="A", players=("Alice", "Bob"))]
        sm._save_match_record()

        parsed = _gemini_race_results((1, "Alice"), (2, "Bob"))
        cb = sm._make_results_callback(0, sm._match_seq, sm._match_dir)
        cb(parsed, [(1, "Alice"), (2, "Bob")])

        assert sm._races[0].placements == ((1, "Alice"), (2, "Bob"))
        assert sm._races[0].gemini_results == parsed
        rec = MatchRecord.load(tmp_path / "match")
        assert rec.races[0].mode == "no_teams"
        assert [p.name for p in rec.races[0].placements] == ["Alice", "Bob"]

    def test_match_results_callback_writes_to_live_state(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [RaceInfo(track_name="A", players=("Alice",))]
        sm._save_match_record()

        parsed = _gemini_match_results((1, "Alice", 100), (2, "Bob", 80))
        cb = sm._make_match_results_callback(sm._match_seq, sm._match_dir)
        cb(parsed, [("Alice", 100), ("Bob", 80)])

        assert sm._gemini_match_results == parsed
        assert sm._match_final_results == [("Alice", 100), ("Bob", 80)]
        assert sm._match_completed_at is not None
        rec = MatchRecord.load(tmp_path / "match")
        assert rec.completed_at is not None
        assert rec.final_standings is not None
        assert [p.score for p in rec.final_standings.players] == [100, 80]

    def test_match_results_callback_with_empty_results(self, tmp_path: Path):
        """Gemini sometimes returns no results — should not set completed_at."""
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [RaceInfo(track_name="A", players=("Alice",))]
        sm._save_match_record()

        cb = sm._make_match_results_callback(sm._match_seq, sm._match_dir)
        cb(None, [])
        assert sm._match_completed_at is None
        assert sm._match_final_results is None


# ---------------------------------------------------------------------------
# Stale callback path (the race condition fix)
# ---------------------------------------------------------------------------


class TestStaleCallbacks:
    """Verifies the race-condition fix.

    Each test follows the same pattern:
      1. Start match A, save initial record, build callback closures
         capturing match_seq and match_dir.
      2. Bump _match_seq (simulating reset() or a new _handle_waiting).
      3. Optionally start match B with different in-memory state.
      4. Fire the stale callback.
      5. Assert match A's on-disk record was updated.
      6. Assert match B's in-memory state and on-disk record are untouched.
    """

    def test_stale_rank_writes_to_disk_not_live(self, tmp_path: Path):
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice",))]
        sm._save_match_record()

        a_seq = sm._match_seq
        cb = sm._make_rank_callback(0, a_seq, match_a)

        # Move on to match B.
        match_b = tmp_path / "match_b"
        _start_match(sm, match_b)
        sm._races = [RaceInfo(track_name="B1", players=("Bob",))]
        sm._save_match_record()

        cb(7)

        # A's on-disk record now has the rank.
        rec_a = MatchRecord.load(match_a)
        assert rec_a.races[0].user_rank == 7

        # B's live state is untouched.
        assert sm._races[0].race_rank is None
        # B's on-disk record is also untouched.
        rec_b = MatchRecord.load(match_b)
        assert rec_b.races[0].user_rank is None

    def test_stale_results_writes_to_disk_not_live(self, tmp_path: Path):
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice", "Bob"))]
        sm._save_match_record()

        a_seq = sm._match_seq
        cb = sm._make_results_callback(0, a_seq, match_a)

        match_b = tmp_path / "match_b"
        _start_match(sm, match_b)
        sm._races = [RaceInfo(track_name="B1", players=("Carol", "Dave"))]
        sm._save_match_record()

        parsed = _gemini_race_results((1, "Alice"), (2, "Bob"))
        cb(parsed, [(1, "Alice"), (2, "Bob")])

        rec_a = MatchRecord.load(match_a)
        assert rec_a.races[0].mode == "no_teams"
        assert [p.name for p in rec_a.races[0].placements] == ["Alice", "Bob"]
        assert rec_a.races[0].teams is not None

        # B's live state is untouched.
        assert sm._races[0].placements == ()
        assert sm._races[0].gemini_results is None
        # B's on-disk record is also untouched.
        rec_b = MatchRecord.load(match_b)
        assert rec_b.races[0].placements == []

    def test_stale_match_results_writes_to_disk_not_live(self, tmp_path: Path):
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice",))]
        sm._save_match_record()

        a_seq = sm._match_seq
        cb = sm._make_match_results_callback(a_seq, match_a)

        match_b = tmp_path / "match_b"
        _start_match(sm, match_b)
        sm._races = [RaceInfo(track_name="B1", players=("Carol",))]
        sm._save_match_record()

        parsed = _gemini_match_results((1, "Alice", 100), (2, "Bob", 80))
        cb(parsed, [("Alice", 100), ("Bob", 80)])

        rec_a = MatchRecord.load(match_a)
        assert rec_a.final_standings is not None
        assert [p.name for p in rec_a.final_standings.players] == ["Alice", "Bob"]
        assert [p.score for p in rec_a.final_standings.players] == [100, 80]
        assert rec_a.completed_at is not None

        # B's live state is untouched.
        assert sm._gemini_match_results is None
        assert sm._match_final_results is None
        assert sm._match_completed_at is None
        # B's on-disk record has no final standings.
        rec_b = MatchRecord.load(match_b)
        assert rec_b.final_standings is None
        assert rec_b.completed_at is None

    def test_reset_invalidates_in_flight_callbacks(self, tmp_path: Path):
        """reset() bumps _match_seq, so any callback fired after a reset
        should take the stale path."""
        sm = GameStateMachine()
        match_dir = tmp_path / "match"
        _start_match(sm, match_dir)
        sm._races = [RaceInfo(track_name="X", players=("Y",))]
        sm._save_match_record()

        cb = sm._make_rank_callback(0, sm._match_seq, match_dir)

        sm.reset()
        assert sm._races == []  # live state is gone

        cb(4)

        # Disk still has the rank, written via the stale path.
        rec = MatchRecord.load(match_dir)
        assert rec.races[0].user_rank == 4

    def test_stale_rank_with_missing_match_json_logs_warning(
        self, tmp_path: Path, caplog,
    ):
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice",))]
        sm._save_match_record()

        a_seq = sm._match_seq
        cb = sm._make_rank_callback(0, a_seq, match_a)

        sm._match_seq += 1
        # Delete the on-disk record before the callback fires.
        (match_a / "match.json").unlink()

        with caplog.at_level("WARNING"):
            cb(3)
        assert any(
            "failed to load match" in r.message.lower() for r in caplog.records
        )

    def test_stale_rank_with_none_match_dir_drops_silently(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        cb = sm._make_rank_callback(0, 999, None)
        # Should not raise; should drop the result.
        cb(5)

    def test_stale_results_with_out_of_range_index(
        self, tmp_path: Path, caplog,
    ):
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice",))]
        sm._save_match_record()

        cb = sm._make_results_callback(99, sm._match_seq, match_a)
        sm._match_seq += 1

        with caplog.at_level("WARNING"):
            cb(_gemini_race_results((1, "Alice")), [(1, "Alice")])
        assert any("out of range" in r.message.lower() for r in caplog.records)
        # Disk record is unchanged.
        rec = MatchRecord.load(match_a)
        assert rec.races[0].placements == []

    def test_stale_match_results_does_not_overwrite_existing_completed_at(
        self, tmp_path: Path,
    ):
        """If the on-disk record already has completed_at (which it normally
        wouldn't, but defensively...), the stale write should leave it alone."""
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice",))]
        sm._match_completed_at = datetime(2026, 1, 1, 12, 0, 0)
        sm._save_match_record()
        original_ts = MatchRecord.load(match_a).completed_at

        cb = sm._make_match_results_callback(sm._match_seq, match_a)
        sm._match_seq += 1
        cb(_gemini_match_results((1, "Alice", 50)), [("Alice", 50)])

        rec = MatchRecord.load(match_a)
        assert rec.completed_at == original_ts  # unchanged
        assert rec.final_standings is not None  # but standings were written

    def test_concurrent_stale_callbacks_are_serialised(self, tmp_path: Path):
        """Two stale callbacks for the same match should both succeed and
        the latter shouldn't lose the former's update.  The lock makes
        this trivially true since they serialise on _races_lock."""
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice", "Bob"))]
        sm._save_match_record()

        a_seq = sm._match_seq
        rank_cb = sm._make_rank_callback(0, a_seq, match_a)
        results_cb = sm._make_results_callback(0, a_seq, match_a)

        sm._match_seq += 1

        rank_cb(2)
        results_cb(_gemini_race_results((1, "Alice"), (2, "Bob")),
                   [(1, "Alice"), (2, "Bob")])

        rec = MatchRecord.load(match_a)
        # Both updates landed.
        assert rec.races[0].user_rank == 2
        assert [p.name for p in rec.races[0].placements] == ["Alice", "Bob"]
        assert rec.races[0].mode == "no_teams"

    def test_full_stale_cycle_all_three_callbacks(self, tmp_path: Path):
        """End-to-end: rank + results + match results all fire stale and
        the resulting on-disk record is fully populated."""
        sm = GameStateMachine()
        match_a = tmp_path / "match_a"
        _start_match(sm, match_a)
        sm._races = [RaceInfo(track_name="A1", players=("Alice", "Bob"))]
        sm._save_match_record()

        a_seq = sm._match_seq
        rank_cb = sm._make_rank_callback(0, a_seq, match_a)
        results_cb = sm._make_results_callback(0, a_seq, match_a)
        match_results_cb = sm._make_match_results_callback(a_seq, match_a)

        # Move on.
        match_b = tmp_path / "match_b"
        _start_match(sm, match_b)

        rank_cb(1)
        results_cb(_gemini_race_results((1, "Alice"), (2, "Bob")),
                   [(1, "Alice"), (2, "Bob")])
        match_results_cb(
            _gemini_match_results((1, "Alice", 15), (2, "Bob", 12)),
            [("Alice", 15), ("Bob", 12)],
        )

        rec = MatchRecord.load(match_a)
        assert rec.races[0].user_rank == 1
        assert [p.name for p in rec.races[0].placements] == ["Alice", "Bob"]
        assert rec.races[0].mode == "no_teams"
        assert rec.final_standings is not None
        assert rec.final_standings.mode == "no_teams"
        assert [p.score for p in rec.final_standings.players] == [15, 12]
        assert rec.completed_at is not None


# ---------------------------------------------------------------------------
# Unicode safety
# ---------------------------------------------------------------------------


class TestStrictPersistenceAndManualStart:
    """Tests for the 'strict + Start Manual Match' rule.

    The rule: ``_save_match_record`` only persists when
    ``_match_started_at`` is set, which happens *only* via real settings
    detection in ``_handle_waiting`` or via an explicit
    ``start_manual_match()`` call.  Manual ``advance()`` paths never
    populate that field, so ghost matches never produce a ``match.json``.
    """

    # -- strict rule blocks ghost saves -----------------------------------

    def test_save_blocked_when_started_at_is_none(self, tmp_path: Path):
        """Even with _match_dir, _match_settings, and races populated, a
        save with _match_started_at=None must produce no file."""
        sm = GameStateMachine()
        sm._match_settings = _settings()
        sm._match_dir = tmp_path / "ghost"
        sm._match_dir.mkdir()
        sm._races = [
            RaceInfo(
                track_name="Mario Bros. Circuit",
                players=("Alice", "Bob"),
                placements=((1, "Alice"), (2, "Bob")),
                race_rank=1,
            )
        ]
        # Belt-and-braces — explicitly None.
        sm._match_started_at = None

        sm._save_match_record()
        assert not (sm._match_dir / "match.json").exists()

    def test_save_proceeds_when_started_at_is_set(self, tmp_path: Path):
        """Sanity: setting _match_started_at unblocks the same save."""
        sm = GameStateMachine()
        sm._match_settings = _settings()
        sm._match_dir = tmp_path / "real"
        sm._match_dir.mkdir()
        sm._races = [RaceInfo(track_name="X", players=("Y",))]
        sm._match_started_at = datetime.now()

        sm._save_match_record()
        assert (sm._match_dir / "match.json").exists()

    def test_handle_waiting_still_persists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The real detection path in _handle_waiting must continue to
        work — it sets _match_started_at + _match_dir before the save
        call so the strict rule lets the write through."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()

        class _StubDetector:
            def detect(self, _frame):
                return _settings()

        sm._match_detector = _StubDetector()

        import numpy as np
        sm._handle_waiting(np.zeros((1080, 1920, 3), dtype=np.uint8))

        assert sm._match_started_at is not None
        assert sm._match_dir is not None
        assert (sm._match_dir / "match.json").exists()

    # -- start_manual_match() basic behaviour -----------------------------

    def test_start_manual_match_returns_true_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()  # mirrors what the UI does on startup

        assert sm.start_manual_match() is True
        assert sm._match_started_at is not None
        assert sm._match_dir is not None
        assert sm._match_dir.parent == tmp_path
        assert (sm._match_dir / "match.json").exists()

    def test_start_manual_match_writes_initial_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The initial record after a manual start should have settings,
        no races, no final standings."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings(race_count=8, teams="Two Teams")
        sm.start_manual_match()

        rec = MatchRecord.load(sm._match_dir)
        assert rec.settings.race_count == 8
        assert rec.settings.teams == "Two Teams"
        assert rec.races == []
        assert rec.final_standings is None
        assert rec.completed_at is None

    def test_start_manual_match_returns_false_when_already_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()
        assert sm.start_manual_match() is True

        first_dir = sm._match_dir
        first_started_at = sm._match_started_at

        # Second call must be a no-op.
        assert sm.start_manual_match() is False
        assert sm._match_dir == first_dir
        assert sm._match_started_at == first_started_at

    def test_start_manual_match_returns_false_without_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog,
    ):
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm._match_settings = None  # paranoid: ensure UI hasn't pushed any

        with caplog.at_level("WARNING"):
            assert sm.start_manual_match() is False
        assert sm._match_started_at is None
        assert sm._match_dir is None
        assert any("no match settings" in r.message.lower() for r in caplog.records)

    def test_start_manual_match_bumps_match_seq(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Bumping _match_seq invalidates any in-flight callbacks from a
        prior aborted session, just like a real match start does."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()
        before = sm._match_seq
        sm.start_manual_match()
        assert sm._match_seq == before + 1

    def test_reset_then_start_manual_match_starts_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """The documented 'use Reset first to start over' workflow."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()
        sm.start_manual_match()
        first_dir = sm._match_dir
        first_id = first_dir.name

        # Reset, re-push settings (mirrors what the UI does), restart.
        sm.reset()
        assert sm._match_dir is None
        assert sm._match_started_at is None

        sm.match_settings = _settings()
        sm.start_manual_match()

        second_dir = sm._match_dir
        assert second_dir is not None
        # Two distinct match folders on disk.
        assert (first_dir / "match.json").exists()
        assert (second_dir / "match.json").exists()
        # match_id should differ in most cases (timestamps to the second).
        # Don't assert inequality — the test could in theory run within the
        # same second.  Just verify both are real records.
        assert MatchRecord.load(first_dir).match_id == first_id

    # -- end-to-end ghost vs manual scenarios -----------------------------

    def test_pure_ghost_advance_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Scenario A: user clicks Advance with no real frames. No JSON
        should ever be written, regardless of how many states they
        traverse or how many _save_match_record calls happen."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()  # UI default

        for _ in range(20):
            sm.advance()
        sm._save_match_record()  # explicit save attempt

        json_files = list(tmp_path.rglob("match.json"))
        assert json_files == []

    def test_ghost_then_manual_start_persists_only_after_button(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Scenario B refined: user advances around a bit, then clicks
        Start Manual Match. Nothing is persisted before the click; saves
        proceed normally after."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()

        # User pokes around with advance.
        sm.advance()
        sm.advance()
        sm._save_match_record()
        assert list(tmp_path.rglob("match.json")) == []

        # User clicks Start Manual Match.
        assert sm.start_manual_match() is True
        json_files = list(tmp_path.rglob("match.json"))
        assert len(json_files) == 1

        # Subsequent updates persist normally.
        sm._races = [RaceInfo(track_name="Bowser Castle", players=("Alice",))]
        sm._save_match_record()
        rec = MatchRecord.load(sm._match_dir)
        assert len(rec.races) == 1
        assert rec.races[0].track_name == "Bowser Castle"

    def test_ghost_callback_routes_to_disk_via_stale_path_only_if_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A Gemini callback fired against a manually-started match dir
        that was then reset should write to disk via the stale path
        (because the JSON file actually exists)."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()
        sm.start_manual_match()
        manual_dir = sm._match_dir
        # Add a race so the rank callback has something to write to.
        sm._races = [RaceInfo(track_name="X", players=("Y",))]
        sm._save_match_record()

        cb = sm._make_rank_callback(0, sm._match_seq, manual_dir)

        sm.reset()  # bumps _match_seq, clears in-memory state

        cb(5)  # stale callback fires

        rec = MatchRecord.load(manual_dir)
        assert rec.races[0].user_rank == 5

    def test_callback_against_unstarted_match_drops_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog,
    ):
        """If somehow a callback is fired with a match_dir that has no
        match.json on disk (because the strict rule blocked the save),
        the stale path should drop the result with a warning rather
        than crashing."""
        import mktracker.state_machine as sm_mod
        monkeypatch.setattr(sm_mod, "_MATCHES_DIR", tmp_path)

        sm = GameStateMachine()
        sm.match_settings = _settings()
        # Simulate a ghost folder that exists on disk but has no JSON.
        ghost_dir = tmp_path / "ghost"
        ghost_dir.mkdir()
        cb = sm._make_rank_callback(0, sm._match_seq, ghost_dir)

        sm._match_seq += 1  # force stale path
        with caplog.at_level("WARNING"):
            cb(3)
        assert any(
            "failed to load match" in r.message.lower() for r in caplog.records
        )

    # -- reset side effects -----------------------------------------------

    def test_reset_clears_match_dir(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        assert sm._match_dir is not None
        sm.reset()
        assert sm._match_dir is None


class TestUnicode:
    def test_special_characters_survive_save_load(self, tmp_path: Path):
        sm = GameStateMachine()
        _start_match(sm, tmp_path / "match")
        sm._races = [
            RaceInfo(
                track_name="Rainbow Road",
                players=("TA☆Collins", "★PlayerB", "π♪user"),
                gemini_results=_gemini_race_results(
                    (1, "TA☆Collins"),
                    (2, "★PlayerB"),
                    (3, "π♪user"),
                ),
            )
        ]
        sm._save_match_record()
        rec = MatchRecord.load(tmp_path / "match")
        names = [p.name for p in rec.races[0].placements]
        assert names == ["TA☆Collins", "★PlayerB", "π♪user"]
