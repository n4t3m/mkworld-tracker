"""Trained final-standings OCR on saved game screenshots."""
from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np
import pytest

from mktracker.detection.match_results import MatchResultDetector
from mktracker.detection.name_model import NameModelReader
from mktracker.detection.match_results_model import (
    read_saved_final_standings, reconcile_final_rows,
)
from mktracker.team_scoring import local_final_standings


_FIXTURES = Path(__file__).parent / "fixtures" / "match_results"


@pytest.fixture(scope="module")
def detector():
    if not (NameModelReader.available() and NameModelReader.color_available()):
        pytest.skip("No local trained checkpoints configured")
    dark = NameModelReader()
    color = NameModelReader(color=True)
    try:
        yield MatchResultDetector(dark, color)
    finally:
        dark.close()
        color.close()


@pytest.mark.parametrize("filename,teams,count,minimum,first_name,first_score", [
    ("match-standings-no-teams-8p-silver.png", "No Teams", 8, 8, "Cecil", 26),
    ("20260329_162559.png", "No Teams", 24, 22, "Baby Daisy", 45),
    ("20260403_172613.png", "No Teams", 23, 22, "nineballo", 82),
    ("24p-vod-final.jpg", "No Teams", 24, 22, "Carbuncle2", 91),
    ("twoteams_result.png", "Two Teams", 12, 12, "Zog Zhit A", 103),
    ("12p-vod-final.jpg", "Two Teams", 12, 12, "Ly Raised", 96),
    ("20260329_174714.png", "Two Teams", 24, 24, "Baby Daisy", 45),
    ("threeteams_banner_yellow_win.png", "Three Teams", 24, 22, "Peepa", 28),
    ("fourteams_banner_green_win.png", "Four Teams", 24, 22, "Yoshi", 30),
])
def test_model_final_standings(detector, filename, teams, count, minimum,
                               first_name, first_score):
    frame = cv2.imread(str(_FIXTURES / filename))
    assert frame is not None
    result = detector.detect(frame, teams=teams, player_count=count)
    assert result is not None
    assert len(result["results"]) >= minimum
    assert result["results"][0] == (first_name, first_score)


def test_model_rejects_loading_screen(detector):
    frame = cv2.imread(str(_FIXTURES / "fourteams_banner_green_win_loading.png"))
    assert frame is not None
    assert detector.detect(frame, teams="Four Teams", player_count=24) is None


@pytest.mark.parametrize("filename,teams,name,score,totals", [
    ("threeteams_banner_yellow_win.png", "Three Teams", "Swoop", 19,
     [134, 138, 160]),
    ("fourteams_banner_green_win.png", "Four Teams", "Cow", 17,
     [96, 99, 106, 131]),
])
def test_model_recovers_low_contrast_final_names(detector, filename, teams,
                                                 name, score, totals):
    frame = cv2.imread(str(_FIXTURES / filename))
    result = detector.detect(frame, teams=teams, player_count=24, model_only=True)
    assert result is not None
    assert len(result["model_rows"]) == 24
    assert (name, score) in result["results"]
    if teams == "Three Teams":
        assert (1, 5, "Toad", 11) in result["model_rows"]
    standings = local_final_standings(result["model_rows"], teams, 24)
    assert standings is not None
    assert [team.points for team in standings.teams] == totals


def test_model_rejects_team_score_that_disagrees_with_header(detector):
    from mktracker.detection.match_results_model import read_final_standings_rows

    class WrongHeaderReader:
        def predict_crops(self, crops, *, width=None):
            options = {"width": width} if width is not None else {}
            predictions = detector.color_model.predict_crops(crops, **options)
            if len(crops) == 3 and crops[0].shape[0] == 125:
                predictions[1] = ("137", .99)
            return predictions

    frame = cv2.imread(str(_FIXTURES / "threeteams_banner_yellow_win.png"))
    assert read_final_standings_rows(
        frame, teams="Three Teams", player_count=24,
        reader=WrongHeaderReader(),
    ) == []


def test_model_recovers_punctuated_score_from_tighter_crop(detector):
    frame = cv2.imread(str(_FIXTURES / "20260329_162559.png"))
    result = detector.detect(frame, teams="No Teams", player_count=24,
                             model_only=True)
    assert result is not None
    assert len(result["model_rows"]) == 24
    assert (0, 3, "Baby Peach", 27) in result["model_rows"]


def test_model_reads_vod_final_scores(detector):
    frame = cv2.imread(str(_FIXTURES / "12p-vod-final.jpg"))
    assert frame is not None
    result = detector.detect(frame, teams="Two Teams", player_count=12)
    assert result is not None
    assert [score for _, score in result["results"]] == [
        96, 86, 76, 73, 57, 17, 110, 98, 88, 83, 71, 68,
    ]
    standings = local_final_standings(result["model_rows"], "Two Teams", 12)
    assert standings is not None
    assert [(team.points, team.winner) for team in standings.teams] == [
        (405, False), (518, True),
    ]
    roster = [
        "Ly Hansry", "Vy²TCRapha", "Vy² Azusa", "Ly Kuri",
        "Ly Lara", "Vy² Axis", "Vy² Will", "Ly Bert", "Ly Lipix",
        "Ly Raised", "Vy² Fishii", "Vy²Barrett",
    ]
    reconciled = reconcile_final_rows(result["model_rows"], roster)
    assert [name for _, _, name, _ in reconciled] == [
        "Ly Raised", "Ly Hansry", "Ly Lipix", "Ly Kuri", "Ly Bert",
        "Ly Lara", "Vy² Axis", "Vy²Barrett", "Vy² Azusa",
        "Vy²TCRapha", "Vy² Will", "Vy² Fishii",
    ]
    tagged = local_final_standings(reconciled, "Two Teams", 12)
    assert tagged is not None
    assert [(team.name, team.tag) for team in tagged.teams] == [
        ("Ly", "Ly"), ("Vy²", "Vy²"),
    ]
    frame_24p = cv2.imread(str(_FIXTURES / "24p-vod-final.jpg"))
    assert frame_24p is not None
    result_24p = detector.detect(frame_24p, teams="No Teams", player_count=24)
    assert result_24p is not None
    assert len(result_24p["model_rows"]) == 24
    assert result_24p["results"][:2] == [("Carbuncle2", 91), ("Akagane", 91)]
    assert (1, 2, "rq", 69) in result_24p["model_rows"]


def test_model_reads_complete_two_team_24p_card(detector):
    frame = cv2.imread(str(_FIXTURES / "20260329_174714.png"))
    result = detector.detect(frame, teams="Two Teams", player_count=24,
                             model_only=True)
    assert result is not None
    assert len(result["model_rows"]) == 24
    standings = local_final_standings(result["model_rows"], "Two Teams", 24)
    assert standings is not None
    assert [team.points for team in standings.teams] == [226, 206]
    roster = [name.replace("BabyMario", "Baby Mario")
              for _, _, name, _ in result["model_rows"]]
    saved = read_saved_final_standings(frame, teams="Two Teams", roster=roster)
    assert saved is not None
    assert [team.points for team in saved.teams] == [226, 206]
    assert any(player.name == "Baby Mario" for player in saved.players)


def test_model_keeps_legacy_on_mismatched_room_size(detector):
    """The old fixture labels a 24-player card as 12 players."""
    frame = cv2.imread(str(_FIXTURES / "20260329_174714.png"))
    assert frame is not None
    result = detector.detect(frame, teams="Two Teams", player_count=12)
    assert result is not None
    assert len(result["results"]) >= 9


def test_saved_final_card_uses_model_when_roster_verifies_every_name():
    if not NameModelReader.color_available():
        pytest.skip("No local colored checkpoint configured")
    frame = cv2.imread(str(_FIXTURES / "12p-vod-final.jpg"))
    roster = [
        "Ly Hansry", "Vy²TCRapha", "Vy² Azusa", "Ly Kuri",
        "Ly Lara", "Vy² Axis", "Vy² Will", "Ly Bert", "Ly Lipix",
        "Ly Raised", "Vy² Fishii", "Vy²Barrett",
    ]
    standings = read_saved_final_standings(
        frame, teams="Two Teams", roster=roster)
    assert standings is not None
    assert len(standings.players) == 12
    assert [(team.name, team.points) for team in standings.teams] == [
        ("Ly", 405), ("Vy²", 518),
    ]


def test_saved_no_team_card_rejects_names_covered_by_chat_overlay():
    if not NameModelReader.available():
        pytest.skip("No local dark checkpoint configured")
    frame = cv2.imread(str(_FIXTURES / "24p-vod-final.jpg"))
    roster = [
        "Mhlia Blan", "rq", "Sato Masao", "Loberia", "Masato Hay",
        "nerune", "Li4z", "Kanata", "Crude play", "SKaR",
        "Star Anise", "Carbuncle2", "Mda", "u1st", "Bakuten",
        "Mus", "ain", "luce", "Hakua", "Kusaan", "Akagane",
        "Magia", "L1am", "atena",
    ]
    assert read_saved_final_standings(
        frame, teams="No Teams", roster=roster) is None


def test_history_refetch_prefers_model_without_api(monkeypatch, tmp_path):
    from mktracker.match_record import FinalStandings, PlayerPlacement
    from mktracker.ui import match_history

    expected = FinalStandings(
        mode="no_teams",
        players=[PlayerPlacement(place=1, name="Player", score=50)],
        teams=None,
    )
    monkeypatch.setattr(match_history, "read_saved_final_standings",
                        lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(match_history, "request_match_results",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("Gemini called")))
    thread = match_history._RefetchTableThread(
        np.zeros((4, 4, 3), dtype=np.uint8), tmp_path,
        teams="No Teams", roster=["Player"], use_gemini=False,
    )
    returned = []
    thread.finished.connect(returned.append)
    thread.run()
    assert returned == [expected]


def test_history_refetch_saves_model_standings(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from mktracker.match_record import (
        FinalStandings, MatchRecord, MatchSettingsRecord, PlayerPlacement,
    )
    from mktracker.ui import match_history

    app = QApplication.instance() or QApplication([])
    record = MatchRecord(
        match_id="saved-model-card", started_at="2026-11-28T00:00:00",
        completed_at=None,
        settings=MatchSettingsRecord(
            cc_class="150cc", teams="No Teams", items="Normal",
            com_difficulty="No COM", race_count=1, intermission="10 seconds",
        ),
    )
    match_dir = tmp_path / record.match_id
    record.save(match_dir)
    expected = FinalStandings(
        mode="no_teams",
        players=[PlayerPlacement(place=1, name="Player", score=50)],
        teams=None,
    )
    monkeypatch.setattr(match_history, "generate_table", lambda _record: b"table")
    view = match_history.MatchDetailView(matches_dir=tmp_path)
    view._current_record = record
    view._on_refetch_finished(record.match_id, expected)
    saved = MatchRecord.load(match_dir)
    assert saved.final_standings == expected
    assert saved.completed_at is not None
    assert (match_dir / "table.png").read_bytes() == b"table"
    view.close()
    del app


def test_history_offers_model_table_regeneration_without_api(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from mktracker.match_record import MatchRecord, MatchSettingsRecord
    from mktracker.ui import match_history

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(match_history, "load_api_key", lambda: None)
    monkeypatch.setattr(match_history, "_saved_final_model_available",
                        lambda _record: True)
    record = MatchRecord(
        match_id="model-only-final", started_at="2026-11-28T00:00:00",
        completed_at=None,
        settings=MatchSettingsRecord(
            cc_class="150cc", teams="Two Teams", items="Normal",
            com_difficulty="No COM", race_count=1, intermission="10 seconds",
        ),
    )
    match_dir = tmp_path / record.match_id
    match_dir.mkdir(parents=True)
    cv2.imwrite(str(match_dir / "match_results.png"),
                np.zeros((4, 4, 3), dtype=np.uint8))
    view = match_history.MatchDetailView(matches_dir=tmp_path)
    view.set_record(record)
    assert view._timeline._table_card._refetch_btn is not None
    view.close()
    del app
