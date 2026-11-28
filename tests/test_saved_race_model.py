"""Saved Match History race frames use local OCR only with full evidence."""
from pathlib import Path
import os

import cv2
import numpy as np
import pytest

from mktracker.detection.name_model import NameModelReader
from mktracker.detection.saved_race_model import read_saved_race_results


FRAME = (Path(__file__).parent / "fixtures" / "video_analysis" /
         "race-results-colored-12p-vy-tags.jpg")
ROSTER = [
    "Ly Lara", "Vy² Axis", "Vy² Will", "Vy²Barrett", "Ly Raised",
    "Ly Lipix", "Vy²TCRapha", "Ly Hansry", "Vy² Fishii", "Ly Bert",
    "Ly Kuri", "Vy² Azusa",
]


def test_saved_twelve_player_team_board_uses_model_and_roster():
    if not NameModelReader.color_available():
        pytest.skip("No local colored checkpoint configured")
    frame = cv2.imread(str(FRAME))
    assert frame is not None
    structured = read_saved_race_results(
        [frame, frame.copy()], teams="Two Teams", roster=ROSTER)
    assert structured is not None
    assert structured["mode"] == "two_teams"
    assert sum(len(team["players"]) for team in structured["teams"]) == 12
    assert sum(team["race_points"] for team in structured["teams"]) == 82
    assert any(player["name"] == "Vy² Will"
               for team in structured["teams"] for player in team["players"])


def test_saved_race_requires_repeated_frames():
    if not NameModelReader.color_available():
        pytest.skip("No local colored checkpoint configured")
    frame = cv2.imread(str(FRAME))
    assert read_saved_race_results(
        [frame], teams="Two Teams", roster=ROSTER) is None


def test_saved_race_rejects_unverified_roster():
    assert read_saved_race_results([], teams="Two Teams", roster=ROSTER) is None
    assert read_saved_race_results([], teams="Four Teams", roster=ROSTER) is None


def test_saved_four_team_scrolling_board_uses_model(monkeypatch):
    import mktracker.detection.saved_race_model as module
    import mktracker.detection.team_results as team_results

    roster = [f"Player {rank}" for rank in range(1, 25)]
    colors = ("red", "blue", "yellow", "green")

    class Reader:
        @staticmethod
        def color_available():
            return True

        def __init__(self, *, color=False):
            assert color

        @staticmethod
        def read(_frame, grid):
            ranks = grid["row_centers"]
            return {**grid, "results": [(rank, roster[rank - 1]) for rank in ranks],
                    "confidences": {rank: 95 for rank in ranks}}

        def close(self):
            pass

    class Detector:
        @staticmethod
        def detect(frame, **kwargs):
            assert kwargs["model_only"] and kwargs["rank_only"]
            ranks = (range(1, 13) if frame[0, 0, 0] in (1, 3)
                     else range(13, 25))
            return {"type": "race", "row_centers": {rank: rank for rank in ranks},
                    "team_colors": {rank: colors[(rank - 1) % 4] for rank in ranks}}

    monkeypatch.setattr(module, "NameModelReader", Reader)
    monkeypatch.setattr(module, "RaceResultDetector", Detector)
    monkeypatch.setattr(team_results, "_bands", lambda _frame: [(0, 1)] * 3)
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in (1, 2, 3, 4)]
    result = read_saved_race_results(frames, teams="Four Teams", roster=roster)
    assert result is not None and result["mode"] == "four_teams"
    assert [len(team["players"]) for team in result["teams"]] == [6] * 4


def test_saved_four_team_requires_complete_scroll(monkeypatch):
    import mktracker.detection.saved_race_model as module
    import mktracker.detection.team_results as team_results

    roster = [f"Player {rank}" for rank in range(1, 25)]
    monkeypatch.setattr(module.NameModelReader, "color_available", lambda: True)
    monkeypatch.setattr(team_results, "_bands", lambda _frame: [(0, 1)] * 3)
    monkeypatch.setattr(module, "RaceResultDetector", lambda: type("Detector", (), {
        "detect": lambda _self, _frame, **_kwargs: {
            "type": "race", "row_centers": {rank: rank for rank in range(1, 24)},
        },
    })())
    frames = [np.ones((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    assert read_saved_race_results(frames, teams="Four Teams", roster=roster) is None


def test_history_race_refetch_prefers_model_without_api(monkeypatch, tmp_path):
    from mktracker.ui import match_history

    expected = {"mode": "no_teams", "teams": [{
        "name": None, "tag": None, "race_points": None,
        "race_winner": None,
        "players": [{"place": 1, "name": "Player"}],
    }]}
    monkeypatch.setattr(match_history, "read_saved_race_results",
                        lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(match_history, "request_race_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    thread = match_history._RefetchPlacementsThread(
        [np.zeros((4, 4, 3), dtype=np.uint8)], 1, tmp_path,
        teams_setting="No Teams", roster=["Player"], use_gemini=False,
    )
    returned = []
    thread.finished.connect(returned.append)
    thread.run()
    assert returned == [expected]


def test_history_race_refetch_persists_model_placements(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from mktracker.match_record import (
        MatchRecord, MatchSettingsRecord, RaceRecord,
    )
    from mktracker.ui import match_history

    app = QApplication.instance() or QApplication([])
    record = MatchRecord(
        match_id="saved-model-race", started_at="2026-11-28T00:00:00",
        completed_at=None,
        settings=MatchSettingsRecord(
            cc_class="150cc", teams="No Teams", items="Normal",
            com_difficulty="No COM", race_count=1, intermission="10 seconds",
        ),
        races=[RaceRecord(race_number=1, track_name="Track", players=["Player"])],
    )
    match_dir = tmp_path / record.match_id
    record.save(match_dir)
    parsed = {"mode": "no_teams", "teams": [{
        "name": None, "tag": None, "race_points": None,
        "race_winner": None,
        "players": [{"place": 1, "name": "Player"}],
    }]}
    view = match_history.MatchDetailView(matches_dir=tmp_path)
    view._on_refetch_placements_finished(record.match_id, 1, parsed)
    saved = MatchRecord.load(match_dir)
    assert saved.races[0].mode == "no_teams"
    assert [(player.place, player.name) for player in saved.races[0].placements] == [
        (1, "Player")]
    view.close()
    del app


def test_history_offers_model_regeneration_without_api(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from mktracker.match_record import (
        MatchRecord, MatchSettingsRecord, RaceRecord,
    )
    from mktracker.ui import match_history

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(match_history, "load_api_key", lambda: None)
    monkeypatch.setattr(match_history, "saved_race_model_available",
                        lambda teams, roster: teams == "Two Teams" and len(roster) == 12)
    record = MatchRecord(
        match_id="model-only-race", started_at="2026-11-28T00:00:00",
        completed_at=None,
        settings=MatchSettingsRecord(
            cc_class="150cc", teams="Two Teams", items="Normal",
            com_difficulty="No COM", race_count=1, intermission="10 seconds",
        ),
        races=[RaceRecord(race_number=1, track_name="Track",
                          players=list(ROSTER))],
    )
    race_dir = tmp_path / record.match_id / "race_01"
    race_dir.mkdir(parents=True)
    cv2.imwrite(str(race_dir / "placement_001.png"),
                np.zeros((4, 4, 3), dtype=np.uint8))
    view = match_history.MatchDetailView(matches_dir=tmp_path)
    view.set_record(record)
    assert view._timeline._race_cards[1]._regenerate_btn is not None
    view.close()
    del app
