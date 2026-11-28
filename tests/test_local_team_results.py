"""Local structured team results from trained names and coloured bars."""
from __future__ import annotations

import numpy as np
import time
import pytest

import mktracker.state_machine as state_machine_module
from mktracker.detection.match_settings import MatchSettings
from mktracker.state_machine import GameState, GameStateMachine, RaceInfo
from mktracker.team_scoring import local_final_standings, local_race_teams


_PLACEMENTS = [
    (1, "Ly Lara"), (2, "Vy² Axis"), (3, "Vy² Will"),
    (4, "Vy²Barrett"), (5, "Ly Raised"), (6, "Ly Lipix"),
    (7, "Vy²TCRapha"), (8, "Ly Hansry"), (9, "Vy² Fishii"),
    (10, "Ly Bert"), (11, "Ly Kuri"), (12, "Vy² Azusa"),
]
_COLORS = {place: "blue" if name.startswith("Vy") else "red"
           for place, name in _PLACEMENTS}


def test_local_teams_keep_names_and_derive_game_points():
    structured = local_race_teams(_PLACEMENTS, _COLORS, "Two Teams")
    assert structured is not None
    assert structured["mode"] == "two_teams"
    assert [(team["name"], team["race_points"], team["race_winner"])
            for team in structured["teams"]] == [
                ("Ly", 40, False), ("Vy²", 42, True),
            ]
    race = RaceInfo("SNES Choco Island 1", (), placements=tuple(_PLACEMENTS),
                    local_results=structured)
    record = GameStateMachine._build_race_record(1, race)
    assert record.mode == "two_teams"
    assert len(record.placements) == 12
    assert [(team.name, team.points) for team in record.teams] == [
        ("Ly", 40), ("Vy²", 42),
    ]


def test_local_teams_refuse_missing_color_evidence():
    assert local_race_teams(_PLACEMENTS, {1: "red"}, "Two Teams") is None
    assert local_race_teams(_PLACEMENTS[:6], _COLORS, "Two Teams") is None


def test_live_local_result_path_persists_structured_teams(monkeypatch):
    class Reader:
        disabled = False

        @staticmethod
        def read(_frame, result):
            return result

    class Detector:
        @staticmethod
        def detect(_frame, **_kwargs):
            return {"type": "race", "results": _PLACEMENTS,
                    "team_colors": _COLORS}

    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Two Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("SNES Choco Island 1", ())]
    machine._color_name_model = Reader()
    machine._result_detector = Detector()
    monkeypatch.setattr(machine, "_save_race_frame", lambda *_args: None)
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)

    machine._handle_reading_results_ocr(np.zeros((4, 4, 3), dtype=np.uint8))
    machine._finalise_race_placements()

    assert machine._races[0].local_results is not None
    record = machine._build_race_record(1, machine._races[0])
    assert [(team.name, team.points) for team in record.teams] == [
        ("Ly", 40), ("Vy²", 42),
    ]


def test_local_final_cards_keep_team_columns_and_total_scores():
    red = [96, 86, 76, 73, 57, 17]
    blue = [110, 98, 88, 83, 71, 68]
    rows = [(column, row, f"Player {column}-{row}", score)
            for column, scores in enumerate((red, blue))
            for row, score in enumerate(scores)]
    standings = local_final_standings(rows, "Two Teams", 12)
    assert standings is not None
    assert standings.mode == "two_teams"
    assert [(team.name, team.points, team.winner)
            for team in standings.teams] == [
                ("Red", 405, False), ("Blue", 518, True),
            ]
    assert [(player.place, player.score) for player in standings.players[:3]] == [
        (1, 110), (2, 98), (3, 96),
    ]
    assert local_final_standings(rows[:-1], "Two Teams", 12) is None
    tied = [(*entry[:3], 96) if entry[:2] == (1, 2) else entry
            for entry in rows]
    tied_standings = local_final_standings(tied, "Two Teams", 12)
    assert tied_standings is not None
    assert [player.place for player in tied_standings.players if player.score == 96] == [
        3, 3,
    ]


def test_no_team_final_card_uses_card_order_with_score_ties():
    scores = [91, 91, 90, 85, 81, 77, 77, 76, 75, 74, 74, 72,
              71, 70, 69, 68, 67, 65, 60, 59, 56, 53, 16, 4]
    rows = [(index // 12, index % 12, f"Player {index + 1}", score)
            for index, score in enumerate(scores)]
    standings = local_final_standings(rows, "No Teams", 24)
    assert standings is not None
    assert standings.mode == "no_teams" and standings.teams is None
    assert [(player.place, player.name, player.score)
            for player in standings.players] == [
                (next(position for position, value in enumerate(scores, 1)
                      if value == score), f"Player {index + 1}", score)
                for index, score in enumerate(scores)
            ]
    assert local_final_standings(rows[:-1], "No Teams", 24) is None
    assert local_final_standings(rows[:-1] + [rows[0]], "No Teams", 24) is None
    assert local_final_standings(rows[:-1] + [(1, 11, "Player 1", 4)],
                                 "No Teams", 24) is None


def test_local_final_two_team_24p_groups_two_columns_per_team():
    rows = [(column, row, f"Player {column}-{row}", 45 - column * 6 - row)
            for column in range(4) for row in range(6)]
    standings = local_final_standings(rows, "Two Teams", 24)
    assert standings is not None
    assert [team.points for team in standings.teams] == [474, 330]
    assert [[player.name for player in team.players[:2]]
            for team in standings.teams] == [
        ["Player 0-0", "Player 0-1"],
        ["Player 2-0", "Player 2-1"],
    ]
    assert local_final_standings(rows[:-1], "Two Teams", 24) is None


def test_keyed_no_team_final_uses_complete_model_card(monkeypatch):
    scores = list(range(80, 68, -1))
    rows = [(0, index, f"Player {index + 1}", score)
            for index, score in enumerate(scores)]

    class Detector:
        @staticmethod
        def detect(_frame, **kwargs):
            assert kwargs["model_only"] is True
            return {"results": [(name, score) for _, _, name, score in rows],
                    "model_rows": rows}

    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "No Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("SNES Choco Island 1", (),
                               placements=tuple((index, f"Player {index}")
                                                for index in range(1, 13)))]
    machine._match_result_detector = Detector()
    machine._name_model = type("Reader", (), {"disabled": False})()
    monkeypatch.setattr(state_machine_module, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(machine, "_handle_finalizing_match_gemini",
                        lambda _frame: (_ for _ in ()).throw(AssertionError("Gemini called")))
    for method in ("_save_frame", "_dump_match_summary", "_save_match_record",
                   "_notify_match_ended", "_transition"):
        monkeypatch.setattr(machine, method, lambda *_args: None)
    machine._handle_finalizing_match(np.zeros((4, 4, 3), dtype=np.uint8))
    standings = machine._build_final_standings()
    assert standings is not None
    assert standings.mode == "no_teams"
    assert standings.teams is None
    assert len(standings.players) == 12
    assert standings.players[0].score == 80


def test_keyed_no_team_final_falls_back_when_name_differs_from_roster(monkeypatch):
    rows = [(0, index, f"Player {index + 1}", 80 - index)
            for index in range(11)] + [(0, 11, "Unrelated", 69)]
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "No Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("Track", (),
                               placements=tuple((index, f"Player {index}")
                                                for index in range(1, 13)))]
    machine._name_model = type("Reader", (), {"disabled": False})()
    machine._match_result_detector = type("Detector", (), {
        "detect": lambda _self, _frame, **_kwargs: {
            "results": [(name, score) for _, _, name, score in rows],
            "model_rows": rows,
        },
    })()
    monkeypatch.setattr(state_machine_module, "load_api_key", lambda: "test-key")
    calls = []
    monkeypatch.setattr(machine, "_handle_finalizing_match_gemini",
                        lambda _frame: calls.append(True))
    machine._handle_finalizing_match(np.zeros((4, 4, 3), dtype=np.uint8))
    assert calls == [True]


def test_keyed_live_final_uses_complete_model_card(monkeypatch):
    scores = ((96, 86, 76, 73, 57, 17),
              (110, 98, 88, 83, 71, 68))
    rows = [(column, row, f"Player {column}-{row}", score)
            for column, team_scores in enumerate(scores)
            for row, score in enumerate(team_scores)]
    model_rows = [(*entry[:2], "Player", entry[3]) if entry[:2] == (1, 5)
                  else entry for entry in rows]

    class Detector:
        @staticmethod
        def detect(_frame, **_kwargs):
            assert _kwargs["model_only"] is True
            return {"results": [(name, score) for _, _, name, score in model_rows],
                    "model_rows": model_rows}

    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Two Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("SNES Choco Island 1", (),
                               placements=tuple((index, name)
                                                for index, (_, _, name, _) in enumerate(rows, 1)))]
    machine._match_result_detector = Detector()
    machine._color_name_model = type("Reader", (), {"disabled": False})()
    monkeypatch.setattr(state_machine_module, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(machine, "_handle_finalizing_match_gemini",
                        lambda _frame: (_ for _ in ()).throw(AssertionError("Gemini called")))
    for method in ("_save_frame", "_dump_match_summary", "_save_match_record",
                   "_notify_match_ended", "_transition"):
        monkeypatch.setattr(machine, method, lambda *_args: None)
    machine._handle_finalizing_match(np.zeros((4, 4, 3), dtype=np.uint8))
    standings = machine._build_final_standings()
    assert standings is not None
    assert [(team.points, team.winner) for team in standings.teams] == [
        (405, False), (518, True),
    ]
    assert machine._match_final_results[-1] == ("Player 1-5", 68)


def test_keyed_live_final_keeps_gemini_fallback_for_partial_model(monkeypatch):
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Two Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("SNES Choco Island 1", (),
                               placements=tuple((index, f"Player {index}")
                                                for index in range(1, 13)))]
    machine._color_name_model = type("Reader", (), {"disabled": False})()
    machine._match_result_detector = type("Detector", (), {
        "detect": lambda _self, _frame, **_kwargs: {
            "results": [("Player 1", 100)],
            "model_rows": [(0, 0, "Player 1", 100)],
        },
    })()
    monkeypatch.setattr(state_machine_module, "load_api_key", lambda: "test-key")
    called = []
    monkeypatch.setattr(machine, "_handle_finalizing_match_gemini",
                        lambda _frame: called.append(True))
    machine._handle_finalizing_match(np.zeros((4, 4, 3), dtype=np.uint8))
    assert called == [True]


@pytest.mark.parametrize("teams,column_count,rows_per_column", [
    ("No Teams", 2, 12),
    ("Three Teams", 3, 8),
])
def test_keyed_24p_final_runs_model_off_capture_thread(
    monkeypatch, teams, column_count, rows_per_column,
):
    rows = [(column, row, f"Player {column}-{row}",
             30 - row - (column * rows_per_column if teams == "No Teams" else 0))
            for column in range(column_count) for row in range(rows_per_column)]
    dispatched = []

    class DeferredThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target, self.args = target, args

        def start(self):
            dispatched.append(self)

    class BannerDetector:
        @staticmethod
        def _has_result_banner(_frame, **_kwargs):
            return True

    class ModelDetector:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def detect(_frame, **kwargs):
            assert kwargs == {"teams": teams, "player_count": 24,
                              "model_only": True}
            return {"results": [(name, score) for _, _, name, score in rows],
                    "model_rows": rows}

    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", teams, "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("Track", (), placements=tuple(
        (index, name) for index, (_, _, name, _) in enumerate(rows, 1)))]
    machine._match_result_detector = BannerDetector()
    if teams == "No Teams":
        machine._dark_race_result_model = type("Reader", (), {"disabled": False})()
    else:
        machine._race_result_model = type("Reader", (), {"disabled": False})()
    machine._match_banner_seen_at = time.monotonic() - 2
    monkeypatch.setattr(state_machine_module, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(state_machine_module, "MatchResultDetector", ModelDetector)
    monkeypatch.setattr(state_machine_module.threading, "Thread", DeferredThread)
    for method in ("_save_frame", "_dump_match_summary", "_save_match_record",
                   "_notify_match_ended"):
        monkeypatch.setattr(machine, method, lambda *_args: None)
    monkeypatch.setattr(machine, "_handle_finalizing_match_gemini",
                        lambda _frame: (_ for _ in ()).throw(AssertionError("Gemini called")))

    machine._handle_finalizing_match(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(dispatched) == 1
    assert machine.state is GameState.WAITING_FOR_MATCH
    assert machine._local_final_standings is None
    dispatched[0].target(*dispatched[0].args)
    assert machine._match_final_results == [(name, score) for _, _, name, score in rows]
    assert machine._local_final_standings is not None
    if teams == "No Teams":
        assert machine._local_final_standings.teams is None
        assert len(machine._local_final_standings.players) == 24
    else:
        assert [team.points for team in machine._local_final_standings.teams] == [212] * 3


def test_keyed_24p_final_falls_back_when_model_is_partial(monkeypatch):
    class ModelDetector:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def detect(_frame, **_kwargs):
            return {"model_rows": [(0, 0, "Player 0", 30)]}

    machine = GameStateMachine()
    machine._race_result_model = type("Reader", (), {"disabled": False})()
    monkeypatch.setattr(state_machine_module, "MatchResultDetector", ModelDetector)
    calls = []
    monkeypatch.setattr(state_machine_module, "request_match_results",
                        lambda frame, callback, **kwargs: calls.append((frame, callback, kwargs)))
    machine._read_keyed_final_card_model(
        np.zeros((4, 4, 3), dtype=np.uint8), "Three Teams",
        [f"Player {index}" for index in range(24)], machine._match_seq, None)
    assert len(calls) == 1
