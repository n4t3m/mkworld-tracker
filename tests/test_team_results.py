"""Regression screens from two real custom-room recordings."""
from pathlib import Path
import re

import cv2
import pytest

from mktracker.detection.race_results import RaceResultDetector
from mktracker.detection.name_consensus import PlacementConsensus


def read(name):
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "team_results" / name))
    assert frame is not None
    return RaceResultDetector().detect(frame, teams=True)


def plain(text):
    # These fixtures originally used an ASCII transcription of the clan tag.
    # Font recognition can recover its Greek gamma; exact glyphs have their
    # own integration tests, while this helper checks racer identity/rank.
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("γ", "y"))


@pytest.mark.parametrize("filename,expected", [
    ("race-results-colored-12p-vy-tags.jpg",
     ["red", "blue", "blue", "blue", "red", "red", "blue", "red", "blue", "red", "red", "blue"]),
    ("race-results-colored-4team-24p.jpg",
     ["red", "blue", "red", "yellow", "green", "green", "green", "red", "blue", "yellow", "green", "yellow"]),
])
def test_colored_result_bars_identify_teams(filename, expected):
    from mktracker.detection.team_results import _bands, _layout_shift, _band_team_color

    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" / filename))
    assert frame is not None
    bands = _bands(frame)
    shift = _layout_shift(frame, bands)
    assert [_band_team_color(frame, band, shift) for band in bands] == expected


def test_colored_12p_race_board_has_fast_results_signal():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-colored-12p-vy-tags.jpg"))
    assert frame is not None
    assert RaceResultDetector().has_race_results(frame)


def test_full_table_keeps_names_and_explicit_ranks():
    result = read("race-1.jpg")
    assert result["type"] == "race"
    rows = dict(result["results"])
    assert sorted(rows) == list(range(1, 13))
    expected = {1: "Ly Hansry", 2: "Vy Azusa", 3: "Vy Galaxy", 4: "Ly Maximus",
                5: "Ly Fewings", 6: "Vy TC!", 7: "Vy zen", 8: "Ly Lipix", 9: "Vy Barrett",
                10: "Ly Viny", 11: "Vy Acing", 12: "Ly TBChuck"}
    for rank, name in expected.items():
        assert plain(rows[rank]) == plain(name)


def test_scrolled_results_do_not_restart_ranks_at_one():
    result = read("scrolling-2.jpg")
    assert result["type"] == "race"
    assert [rank for rank, name in result["results"]] == [8, 9, 10, 11, 12]


def test_distant_background_stripe_cannot_become_a_result_row():
    result = read("distant-background.jpg")
    assert result["type"] == "race"
    assert [rank for rank, name in result["results"]] == [8, 9, 10, 11, 12]


def test_overall_standings_are_not_race_placements():
    result = read("overall-1.jpg")
    assert result["type"] == "overall"


def test_second_recording_handles_new_names_and_digits():
    result = read("october-race.jpg")
    rows = dict(result["results"])
    assert sorted(rows) == list(range(1, 13))
    assert plain(rows[1]).endswith("ag64")
    assert plain(rows[2]).endswith("spider")
    assert plain(rows[9]).endswith("rascal")
    assert plain(rows[10]).endswith("tcrapha")


def test_consensus_does_not_overwrite_good_reads_with_one_bad_frame():
    consensus = PlacementConsensus()
    consensus.add({"results": [(1, "Player1")], "confidences": {1: 90}})
    consensus.add({"results": [(1, "Player1")], "confidences": {1: 90}})
    consensus.add({"results": [(1, "Player2"), (0, "noise")], "confidences": {1: 99}})
    assert consensus.placements() == [[1, "Player1"]]


def test_punctuation_variants_combine_but_numeric_names_stay_distinct():
    consensus = PlacementConsensus()
    consensus.add({"results": [(1, "Vy² Azusa")], "confidences": {1: 90}})
    consensus.add({"results": [(1, "Vy' Azusa")], "confidences": {1: 80}})
    consensus.add({"results": [(1, "Vy Wrong")], "confidences": {1: 99}})
    assert consensus.placements() == [[1, "Vy² Azusa"]]


def test_clear_frame_outweighs_repeated_low_confidence_noise():
    consensus = PlacementConsensus()
    for i in range(3):
        consensus.add({"results": [(1, "Wrong Driver")], "confidences": {1: 25}}, frame_id=i)
    consensus.add({"results": [(1, "Correct Driver")], "confidences": {1: 95}}, frame_id=4)
    assert consensus.placements() == [[1, "Correct Driver"]]
    assert consensus.evidence()["1"]["status"] == "review"  # needs corroboration


def test_same_frame_cannot_confirm_itself():
    consensus = PlacementConsensus()
    result = {"results": [(1, "Player1")], "confidences": {1: 90}}
    for _ in range(10):
        consensus.add(result, frame_id="same-image")
    assert consensus.evidence()["1"]["frames"] == 1
    assert consensus.evidence()["1"]["status"] == "review"
    consensus.add(result, frame_id="different-image")
    assert consensus.evidence()["1"]["status"] == "confirmed"


def test_close_spelling_variants_share_evidence_but_flag_spelling():
    consensus = PlacementConsensus()
    consensus.add({"results": [(1, "Vy Barrett")], "confidences": {1: 90}}, frame_id=1)
    consensus.add({"results": [(1, "Vy Parrett")], "confidences": {1: 89}}, frame_id=2)
    evidence = consensus.evidence()["1"]
    assert evidence["frames"] == 2
    assert evidence["agreement"] == 1
    assert evidence["status"] == "review"
    assert evidence["spelling_agreement"] < .6


def test_similar_players_seen_together_are_not_merged():
    consensus = PlacementConsensus()
    frame = {"results": [(1, "Ly Bert"), (2, "Ly Bere")], "confidences": {1: 90, 2: 90}}
    consensus.add(frame, frame_id=1)
    consensus.add(frame, frame_id=2)
    assert consensus.placements() == [[1, "Ly Bert"], [2, "Ly Bere"]]


def test_conflicting_place_cannot_assign_same_player_twice():
    consensus = PlacementConsensus()
    for i in range(3):
        consensus.add({"results": [(1, "Player1")], "confidences": {1: 95}}, frame_id=i)
    consensus.add({"results": [(2, "Player1")], "confidences": {2: 70}}, frame_id=4)
    assert consensus.placements() == [[1, "Player1"]]
    assert consensus.evidence()["2"]["status"] == "conflict"


def test_standings_and_duplicate_rows_do_not_add_votes():
    consensus = PlacementConsensus()
    consensus.add({"type": "overall", "results": [(1, "Player1")]})
    consensus.add({"results": [(1, "Player1"), (1, "Player2")]})
    assert consensus.placements() == []


def test_preview_roster_cross_checks_without_inventing_a_name():
    consensus = PlacementConsensus(roster=["Vy Barrett", "Ly Bert"])
    frame = {"results": [(1, "Vy Parrett"), (2, "Someone New")], "confidences": {1: 90, 2: 90}}
    consensus.add(frame, frame_id=1)
    consensus.add(frame, frame_id=2)
    assert consensus.placements() == [[1, "Vy Parrett"], [2, "Someone New"]]
    assert consensus.evidence()["1"]["preview_match"] == "Vy Barrett"
    assert consensus.evidence()["1"]["status"] == "review"
    assert consensus.evidence()["2"]["preview_match"] is None


def test_clear_preview_support_breaks_a_close_spelling_tie():
    consensus = PlacementConsensus(roster=["Vy Barrett", "Ly Bert"])
    consensus.add({"results": [(1, "Vy Parrett")], "confidences": {1: 90}}, frame_id=1)
    consensus.add({"results": [(1, "Vy Barrett")], "confidences": {1: 90}}, frame_id=2)
    assert consensus.placements() == [[1, "Vy Barrett"]]
    assert consensus.evidence()["1"]["preview_checked"]


def test_background_below_table_does_not_create_thirteenth_place():
    result = read("background-row.jpg")
    assert result is not None
    assert all(rank <= 12 for rank, name in result["results"])


def test_dark_24_player_result_grid_reads_scrolled_lower_half():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-bottom.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert result["type"] == "race"
    assert [rank for rank, _ in result["results"]] == list(range(13, 25))
    names = dict(result["results"])
    assert names[19] == "BippaXD"
    assert names[24] == "BIKZO"


def test_dark_24_player_grid_keeps_rows_with_missed_plus_labels():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-highlight.png"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert result["type"] == "race"
    assert 23 in dict(result["results"])


def test_dark_24_player_highlighted_rank_is_read():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-highlighted-rank.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert result["type"] == "race"
    assert result["highlighted_rank"] == 3
    assert len(dict(result["results"])[3]) >= 2


def test_dark_grid_corrects_misread_highlighted_digit():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-highlighted-rank2.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert result["highlighted_rank"] == 2
    assert result["highlighted_rank_source"] == "grid"


def test_pinned_highlight_keeps_confident_rank_when_grid_scrolls():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-pinned-highlight.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert result["highlighted_rank"] == 14
    assert result["highlighted_rank_source"] == "ocr_confident"


def test_dark_partial_table_keeps_legacy_tail_read():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-partial.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert dict(result["results"])[24] == "nerune"


def test_dark_icon_overlap_keeps_short_name():
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-icon.jpg"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert 19 in dict(result["results"])


def test_dark_short_name_recovers_from_compression_edge():
    from mktracker.detection.font_symbols import configured_font_path
    if not Path(configured_font_path()).is_file():
        pytest.skip("Optional game font is not configured")
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "race-results-dark-24p-short-symbol.png"))
    result = RaceResultDetector().detect(frame, teams=True)
    assert dict(result["results"])[3] == "K4I"


def test_preview_reader_ignores_obscuring_voice_chat_labels():
    from mktracker.detection.preview_roster import read_preview_names
    from mktracker.detection.name_model import NameModelReader
    if not NameModelReader.color_available():
        pytest.skip("Optional colored name model is not configured")
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" / "preset-course.png"))
    reader = NameModelReader(color=True)
    try:
        names = read_preview_names(frame, reader)
    finally:
        reader.close()
    assert any("azusa" in name.casefold() for name, confidence in names)
    assert any("barrett" in name.casefold() for name, confidence in names)
    assert not any("impact player" in name.casefold() for name, confidence in names)


def test_preview_reader_uses_model_and_excludes_voice_chat_overlay():
    from mktracker.detection.preview_roster import read_preview_names

    class Reader:
        def __init__(self):
            self.count = 0

        def predict_crops(self, crops):
            self.count = len(crops)
            return [("Player", .9) for _ in crops]

    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" / "preset-course.png"))
    reader = Reader()
    names = read_preview_names(frame, reader)
    assert reader.count == 11  # The obscured twelfth bar is not a player name.
    assert len(names) == 11
    assert all(name == "Player" and confidence == 90 for name, confidence in names)


def test_preview_roster_stops_after_two_complete_twelve_slot_frames():
    from mktracker.detection.preview_roster import PreviewRoster

    class Reader:
        disabled = False

        def predict_crops(self, crops):
            return [(f"Player{i:02d}", .9) for i in range(len(crops))]

    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "video_analysis" /
                           "course-preview.jpg"))
    assert frame is not None
    roster = PreviewRoster(Reader())
    roster.add(frame)
    assert not roster.complete()
    second = frame.copy()
    second[0, 0, 0] ^= 1
    roster.add(second)
    assert len(roster.names()) == 12
    assert roster.complete()
