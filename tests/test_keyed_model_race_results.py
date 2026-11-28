"""Background trained race reads preserve the keyed Gemini fallback."""
from __future__ import annotations

import numpy as np

import mktracker.state_machine as state_machine_module
import mktracker.detection.team_results as team_results_module
from mktracker.detection.match_settings import MatchSettings
from mktracker.state_machine import GameStateMachine, RaceInfo


PLACEMENTS = [(place, f"R Alpha{place}" if place <= 6 else f"B Beta{place}")
              for place in range(1, 13)]
COLORS = {place: "red" if place <= 6 else "blue" for place, _ in PLACEMENTS}
ROSTER = [name for _, name in PLACEMENTS]


class Reader:
    disabled = False

    @staticmethod
    def read(_frame, result):
        return result


class Detector:
    def __init__(self, placements=PLACEMENTS):
        self.placements = placements

    def detect(self, _frame, **kwargs):
        assert kwargs["model_only"] is True
        return {"type": "race", "results": self.placements,
                "confidences": {place: 90 for place, _ in self.placements},
                "team_colors": {place: COLORS[place] for place, _ in self.placements}}


def _machine():
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Two Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("SNES Choco Island 1", tuple(ROSTER))]
    machine._race_result_model = Reader()
    return machine


def test_complete_model_read_stores_structured_race_without_gemini(monkeypatch):
    machine = _machine()
    misread = PLACEMENTS[:-1] + [(12, "B Betz12")]
    machine._result_detector = Detector(misread)
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    monkeypatch.setattr(machine, "_notify_race_results", lambda _index: None)
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, ROSTER)
    race = machine._races[0]
    assert race.gemini_results is None
    assert race.local_results is not None
    assert race.placements == tuple(PLACEMENTS)
    assert sum(team["race_points"] for team in race.local_results["teams"]) == 82


def test_incomplete_model_read_sends_original_frames_to_gemini(monkeypatch):
    machine = _machine()
    machine._result_detector = Detector(PLACEMENTS[:6])
    calls = []
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, ROSTER)
    assert len(calls) == 1
    assert calls[0][0][0] is frames
    assert calls[0][1]["teams_setting"] == "Two Teams"
    assert machine._races[0].local_results is None


def test_unmatched_complete_model_names_keep_gemini_fallback(monkeypatch):
    machine = _machine()
    misread = PLACEMENTS[:-1] + [(12, "unrelated")]
    machine._result_detector = Detector(misread)
    calls = []
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, ROSTER)
    assert len(calls) == 1
    assert machine._races[0].local_results is None


def test_keyed_finalise_dispatches_model_work_off_live_thread(monkeypatch, tmp_path):
    machine = _machine()
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    machine._result_frames = frames
    machine._match_dir = tmp_path
    monkeypatch.setattr(machine._preview_roster, "names", lambda: ROSTER)
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called synchronously")))
    jobs = []

    class Thread:
        def __init__(self, **kwargs):
            jobs.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(state_machine_module.threading, "Thread", Thread)
    machine._finalise_race_placements()
    assert len(jobs) == 1
    assert jobs[0]["target"] == machine._read_keyed_race_results_model
    assert jobs[0]["args"][0] is frames
    assert machine._result_frames == []


def test_late_local_result_cannot_overwrite_new_match(monkeypatch):
    machine = _machine()
    old_seq = machine._match_seq
    machine._match_seq += 1
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    monkeypatch.setattr(machine, "_notify_race_results", lambda _index: None)
    machine._store_local_keyed_race_results(
        0, old_seq, None, {"mode": "two_teams", "teams": []}, PLACEMENTS)
    assert machine._races[0].placements == ()
    assert machine._races[0].local_results is None


def test_dark_24p_model_combines_pre_signal_tail_and_result_head(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._dark_race_result_model = Reader()

    class DarkDetector:
        @staticmethod
        def detect(frame, **kwargs):
            assert kwargs["model_only"] is True
            ranks = (range(15, 25) if frame[0, 0, 0] == 1
                     else range(1, 15))
            return {"type": "race",
                    "results": [(rank, roster[rank - 1]) for rank in ranks],
                    "row_centers": {rank: rank for rank in ranks},
                    "confidences": {rank: 90 for rank in ranks}}

    machine._result_detector = DarkDetector()
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    monkeypatch.setattr(machine, "_notify_race_results", lambda _index: None)
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    pre = [np.ones((4, 4, 3), dtype=np.uint8)]
    frames = [np.full((4, 4, 3), 2, dtype=np.uint8)]
    machine._read_keyed_dark_race_results_model(
        frames, pre, 0, machine._match_seq, None, None, roster)
    race = machine._races[0]
    assert len(race.placements) == 24
    assert race.local_results["mode"] == "no_teams"
    assert len(race.local_results["teams"][0]["players"]) == 24


def test_dark_24p_model_keeps_gemini_fallback_when_tail_missing(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._dark_race_result_model = Reader()
    machine._result_detector = type("DarkDetector", (), {
        "detect": lambda _self, _frame, **_kwargs: {
            "type": "race", "results": [(rank, roster[rank - 1])
                                        for rank in range(1, 15)],
            "row_centers": {rank: rank for rank in range(1, 15)},
            "confidences": {rank: 90 for rank in range(1, 15)},
        },
    })()
    calls = []
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    frames = [np.full((4, 4, 3), 2, dtype=np.uint8)]
    machine._read_keyed_dark_race_results_model(
        frames, [], 0, machine._match_seq, None, None, roster)
    assert len(calls) == 1
    assert calls[0][0][0] is frames
    assert calls[0][1]["teams_setting"] == "No Teams"


def test_dark_24p_keeps_pre_signal_frame_for_model_only(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine.debug_mode = False
    machine._match_settings = MatchSettings(
        "150cc", "No Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._dark_race_result_model = Reader()
    answers = iter((False, True))
    machine._result_detector.has_race_results = lambda _frame: next(answers)
    monkeypatch.setattr(machine, "_save_race_frame", lambda *_args: None)
    before = np.full((4, 4, 3), 1, dtype=np.uint8)
    first_result = np.full((4, 4, 3), 2, dtype=np.uint8)
    machine._handle_reading_results_gemini(before)
    machine._handle_reading_results_gemini(first_result)
    assert len(machine._model_pre_result_frames) == 1
    assert np.array_equal(machine._model_pre_result_frames[0], before)
    assert len(machine._result_frames) == 1
    assert np.array_equal(machine._result_frames[0], first_result)


def test_dark_24p_finalise_dispatches_prebuffer_without_changing_gemini_frames(
    monkeypatch, tmp_path,
):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "No Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._dark_race_result_model = Reader()
    machine._match_dir = tmp_path
    pre = np.ones((4, 4, 3), dtype=np.uint8)
    result = np.full((4, 4, 3), 2, dtype=np.uint8)
    machine._model_pre_result_frames.append(pre)
    machine._result_frames = [result]
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called synchronously")))
    jobs = []

    class Thread:
        def __init__(self, **kwargs):
            jobs.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(state_machine_module.threading, "Thread", Thread)
    machine._finalise_race_placements()
    assert len(jobs) == 1
    assert jobs[0]["target"] == machine._read_keyed_dark_race_results_model
    assert jobs[0]["args"][0] == [result]
    assert np.array_equal(jobs[0]["args"][1][0], pre)
    assert machine._result_frames == []
    assert list(machine._model_pre_result_frames) == []


def test_four_team_24p_model_combines_scrolling_halves(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    colors = ("red", "blue", "yellow", "green")
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]

    class TeamReader:
        disabled = False

        @staticmethod
        def read(frame, grid):
            return {**grid,
                    "results": [(rank, roster[rank - 1])
                                for rank in grid["row_centers"]],
                    "confidences": {rank: 95 for rank in grid["row_centers"]}}

    class TeamDetector:
        @staticmethod
        def detect(frame, **kwargs):
            assert kwargs["model_only"] is True
            assert kwargs["rank_only"] is True
            ranks = (range(1, 13) if frame[0, 0, 0] in (1, 3)
                     else range(13, 25))
            return {"type": "race", "results": [],
                    "row_centers": {rank: rank for rank in ranks},
                    "team_colors": {rank: colors[(rank - 1) % 4]
                                    for rank in ranks}}

    machine._race_result_model = TeamReader()
    machine._result_detector = TeamDetector()
    monkeypatch.setattr(team_results_module, "_bands", lambda _frame: [(0, 1)] * 3)
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    monkeypatch.setattr(machine, "_notify_race_results", lambda _index: None)
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    frames = [np.full((4, 4, 3), value, dtype=np.uint8)
              for value in (1, 2, 3, 4)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, roster, "Four Teams")
    race = machine._races[0]
    assert race.placements == tuple(enumerate(roster, 1))
    assert race.local_results["mode"] == "four_teams"
    assert [len(team["players"]) for team in race.local_results["teams"]] == [6] * 4


def test_four_team_24p_retries_missing_name_on_spare_frame(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    colors = ("red", "blue", "yellow", "green")
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]

    class TeamReader:
        disabled = False

        @staticmethod
        def read(frame, grid):
            ranks = list(grid["row_centers"])
            if frame[0, 0, 0] == 2:
                ranks.remove(23)
            return {**grid,
                    "results": [(rank, roster[rank - 1]) for rank in ranks],
                    "confidences": {rank: 95 for rank in ranks}}

    class TeamDetector:
        @staticmethod
        def detect(frame, **kwargs):
            assert kwargs["model_only"] and kwargs["rank_only"]
            ranks = (range(1, 13) if frame[0, 0, 0] in (1, 3)
                     else range(13, 25))
            return {"type": "race", "results": [],
                    "row_centers": {rank: rank for rank in ranks},
                    "team_colors": {rank: colors[(rank - 1) % 4]
                                    for rank in ranks}}

    machine._race_result_model = TeamReader()
    machine._result_detector = TeamDetector()
    monkeypatch.setattr(team_results_module, "_bands", lambda _frame: [(0, 1)] * 3)
    monkeypatch.setattr(machine, "_save_match_record", lambda: None)
    monkeypatch.setattr(machine, "_notify_race_results", lambda _index: None)
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    frames = [np.full((4, 4, 3), value, dtype=np.uint8)
              for value in (1, 2, 3, 4, 5)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, roster, "Four Teams")
    assert machine._races[0].placements == tuple(enumerate(roster, 1))


def test_four_team_24p_model_falls_back_for_missing_rank(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._race_result_model = Reader()
    machine._result_detector = type("Detector", (), {
        "detect": lambda _self, _frame, **_kwargs: {
            "type": "race", "results": [],
            "row_centers": {rank: rank for rank in range(1, 24)},
        },
    })()
    monkeypatch.setattr(team_results_module, "_bands", lambda _frame: [(0, 1)] * 3)
    calls = []
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    frames = [np.ones((4, 4, 3), dtype=np.uint8)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, roster, "Four Teams")
    assert len(calls) == 1
    assert calls[0][0][0] is frames
    assert calls[0][1]["teams_setting"] == "Four Teams"


def test_four_team_dark_board_skips_colored_model(monkeypatch):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._race_result_model = Reader()
    machine._result_detector = type("Detector", (), {
        "detect": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("colored model called for a dark board")),
    })()
    monkeypatch.setattr(team_results_module, "_bands", lambda _frame: [])
    calls = []
    monkeypatch.setattr(state_machine_module, "request_race_results",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    machine._read_keyed_race_results_model(
        frames, 0, machine._match_seq, None, None, roster, "Four Teams")
    assert len(calls) == 1
    assert calls[0][0][0] is frames
    assert calls[0][1]["teams_setting"] == "Four Teams"


def test_four_team_24p_finalise_dispatches_background_model(monkeypatch, tmp_path):
    roster = [f"Player {index}" for index in range(1, 25)]
    machine = GameStateMachine()
    machine._match_settings = MatchSettings(
        "150cc", "Four Teams", "Normal", "No COM", 12, "10 seconds")
    machine._races = [RaceInfo("Track", tuple(roster))]
    machine._race_result_model = Reader()
    machine._match_dir = tmp_path
    machine._result_frames = [np.ones((4, 4, 3), dtype=np.uint8)]
    monkeypatch.setattr(machine._preview_roster, "names", lambda: roster)
    jobs = []

    class Thread:
        def __init__(self, **kwargs):
            jobs.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(state_machine_module.threading, "Thread", Thread)
    machine._finalise_race_placements()
    assert len(jobs) == 1
    assert jobs[0]["target"] == machine._read_keyed_race_results_model
    assert jobs[0]["args"][-1] == "Four Teams"
