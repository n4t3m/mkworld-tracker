"""Real-screen checks for model reads on colored team rows."""
import json
from pathlib import Path
from io import StringIO
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mktracker.detection.color_name_postprocess import (
    _correct_one, _prefix_evidence, _restore_visible_slash, correct_colored_board,
    trim_yellow_green_pattern,
)
from mktracker.detection.font_decoder import FontDecoder, GlyphRead
from mktracker.detection.font_symbols import configured_font_path
from mktracker.detection.name_model import (
    NameModelReader, _recover_short_tagged_tail, _restore_visible_word_gap,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _october_crop(rank: int) -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / "team_results" / "october-race.jpg"))
    centers = {2: 154, 4: 308, 5: 385, 7: 539, 10: 770, 11: 847, 12: 924}
    center = centers[rank]
    return frame[center-24:center+36, round(frame.shape[1] * .68):round(frame.shape[1] * .89)]


def _tagged_color_crops(fixture_name: str, color: str) -> list[np.ndarray]:
    directory = FIXTURES / "video_analysis"
    fixture = json.loads((directory / fixture_name).read_text())
    frame = cv2.imread(str(directory / fixture["image"]))
    x1, x2 = fixture["columns"][color]["x_pixels"]
    return [frame[y-24:y+36, x1:x2] for y in fixture["row_centers"]]


def test_real_green_kt_tag_gap_survives_merged_letters():
    crops = _tagged_color_crops("rayzopp-4team-color-tags.json", "green")
    raw = ["KTViti", "KTDani★", "KT★Grego", "KT Couk", "KTkenjii", "KT broski"]
    assert correct_colored_board(list(zip(crops, raw)), None) == [
        "KT Viti", "KT Dani★", "KT★Grego", "KT Couk", "KT kenjii", "KT broski",
    ]


def test_scrolling_four_team_grid_keeps_correlated_direct_rank_reads():
    from mktracker.detection.race_results import RaceResultDetector

    if not NameModelReader.color_available():
        pytest.skip("No local colored checkpoint configured")
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-4team-scroll-rank21.jpg"))
    assert frame is not None
    reader = NameModelReader(color=True)
    try:
        grid = RaceResultDetector().detect(
            frame, teams=True, model_only=True,
            color_model=reader, rank_only=True,
        )
        assert grid is not None
        assert sorted(grid["row_centers"]) == list(range(11, 22))
        names = dict(reader.read(frame, grid)["results"])
        assert names[21] == "A★ROOSTR"
        assert 22 not in names
    finally:
        reader.close()


def test_real_green_bdd_tag_gap_ignores_bottom_border():
    crops = _tagged_color_crops("mankalor-4team-color-tags.json", "green")
    raw = ["BDD obiyo", "BDDIONA", "BDD Sales", "BDD Mitch", "BDD Julian", "BDDMouse"]
    assert correct_colored_board(list(zip(crops, raw)), None) == [
        "BDD obiyo", "BDD IONA", "BDD Sales", "BDD Mitch", "BDD Julian", "BDDMouse",
    ]


def test_repeated_raised_vy_tag_survives_tiny_glyph_font_disagreement():
    from mktracker.detection.race_results import RaceResultDetector

    # Full local 12p result page at 2:56.5. The same raised ² is visible on
    # six blue rows, including ones the font matcher calls another tiny glyph.
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-12p-vy-tags.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and list(grid["row_centers"]) == list(range(1, 13))
    x1, x2 = (round(frame.shape[1] * value) for value in grid["name_crop_x"])
    raw = ["Ly Lara", "Vy Axis", "Vy Will", "Vy Barrett", "Ly Raised",
           "Ly Lipix", "Vy TCRapha", "Ly Hansry", "Vy Fishii", "Ly Bert",
           "Ly Kuri", "Vy Azusa"]
    rows = [(frame[center-24:center+36, x1:x2], raw[rank-1])
            for rank, center in grid["row_centers"].items()]
    font = configured_font_path()
    decoder = FontDecoder(font) if font and Path(font).is_file() else None
    assert correct_colored_board(rows, decoder) == [
        "Ly Lara", "Vy² Axis", "Vy² Will", "Vy²Barrett", "Ly Raised",
        "Ly Lipix", "Vy²TCRapha", "Ly Hansry", "Vy² Fishii", "Ly Bert",
        "Ly Kuri", "Vy² Azusa",
    ]


def test_real_yellow_emoticon_keeps_visible_colon():
    crops = _tagged_color_crops("rayzopp-4team-color-tags.json", "yellow")
    raw = ["stJMikuel", "st Zay", "st Wood", "st Bella", "st awo 3", "st Plunky"]
    assert correct_colored_board(list(zip(crops, raw)), None)[4] == "st awo :3"


def test_visible_slash_is_not_a_genuine_capital_i():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-streamers-4team-24p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None
    y = grid["row_centers"][7]
    x1, x2 = (round(frame.shape[1] * value) for value in grid["name_crop_x"])
    assert _restore_visible_slash(frame[y-24:y+36, x1:x2], "GregIMG4R") == "Greg/MG4R"

    for image, label in (
        ("synthetic-genuine-i-uINectar.png", "uINectar"),
        ("synthetic-genuine-i-rIBv.png", "E★ rIBv"),
    ):
        crop = cv2.imread(str(FIXTURES / "team_results" / image))
        assert crop is not None
        assert _restore_visible_slash(crop, label) == label


@pytest.mark.skipif(not configured_font_path(), reason="game font unavailable")
def test_game_font_restores_visible_music_notes_on_yellow_rows():
    decoder = FontDecoder(configured_font_path())
    rayzopp = _tagged_color_crops("rayzopp-4team-color-tags.json", "yellow")
    names = ["stJMikuel", "st Zay", "st Wood", "st Bella", "st awo 3", "st Plunky"]
    assert correct_colored_board(list(zip(rayzopp, names)), decoder)[0] == "st♪Mikuel"
    mankalor = _tagged_color_crops("mankalor-4team-color-tags.json", "yellow")
    names = ["Infi.rar", "Kuppa.rar", "Kanade.rar", "Nayeon.rar", "Woomy.rars", "Clover.rar"]
    assert correct_colored_board(list(zip(mankalor, names)), decoder)[4] == "Woomy.rar♪"


def test_restores_raised_tag_and_visible_separator():
    rows = [(_october_crop(rank), raw) for rank, raw in
            [(5, "VyAxis"), (10, "VyTCRapha"), (11, "VyBarrett"), (12, "VyAzusa")]]
    assert correct_colored_board(rows, None) == [
        "Vy² Axis", "Vy²TCRapha", "Vy²Barrett", "Vy² Azusa",
    ]


def test_restores_sigma_from_an_independent_glyph_read():
    class SigmaDecoder:
        def classify(self, mask):
            assert mask.any()
            return GlyphRead("Σ", ("Σ",), .9, .1, "Σ")

    rows = [(_october_crop(rank), raw) for rank, raw in
            [(2, "ArE Spider"), (4, "ArE NoDice"), (7, "ArE Kod")]]
    assert correct_colored_board(rows, SigmaDecoder()) == [
        "ArΣ Spider", "ArΣ NoDice", "ArΣ Kod",
    ]


def test_ordinary_yellow_and_green_match_names_are_unchanged():
    frame = cv2.imread(str(FIXTURES / "match_results" / "fourteams_banner_green_win.png"))
    names = ["Baby Daisy", "Wario", "Koopa Troopa", "Coin Coffer", "Baby Rosalina", "Peach"]
    rows = []
    for index, name in enumerate(names):
        crop = frame[665+65*index:725+65*index, 1515:1845]
        crop = np.concatenate([crop, np.repeat(crop[:, -1:], 73, axis=1)], axis=1)
        rows.append((crop, name))
    assert correct_colored_board(rows, None) == names


def test_name_model_reader_applies_colored_board_correction(monkeypatch):
    frame = cv2.imread(str(FIXTURES / "team_results" / "october-race.jpg"))
    reader = NameModelReader(color=True)
    reader.decoder = None
    reader.process = SimpleNamespace(stdin=StringIO())
    monkeypatch.setattr(reader, "_start", lambda: True)
    monkeypatch.setattr(reader, "_reply", lambda: {"results": [
        ["VyAxis", .9], ["VyTCRapha", .9], ["VyBarrett", .9], ["VyAzusa", .9],
    ]})
    result = {"type": "race", "results": [], "row_centers":
              {5: 385, 10: 770, 11: 847, 12: 924}, "name_crop_x": (.68, .89)}
    read = reader.read(frame, result)
    assert read["results"] == [
        (5, "Vy² Axis"), (10, "Vy²TCRapha"),
        (11, "Vy²Barrett"), (12, "Vy² Azusa"),
    ]
    assert all(read["glyph_reviews"][rank]["needs_review"] for rank in (5, 10, 11, 12))


def test_yellow_green_pattern_trim_preserves_text_and_leaves_red_alone():
    frame = cv2.imread(str(FIXTURES / "match_results" / "fourteams_banner_green_win.png"))
    for crop in (frame[665:725, 1035:1438], frame[665:725, 1515:1918]):
        cleaned = trim_yellow_green_pattern(crop)
        assert np.array_equal(cleaned[:, :80], crop[:, :80])
        assert not np.array_equal(cleaned[:, 250:], crop[:, 250:])
        assert np.all(cleaned[:, 300:] == cleaned[0, 300])
    red = _october_crop(2)
    assert np.array_equal(trim_yellow_green_pattern(red), red)


def test_short_tagged_tail_recovery_uses_visible_october_words():
    frame = cv2.imread(str(FIXTURES / "team_results" / "october-race.jpg"))
    width = frame.shape[1]
    def crop(center):
        return frame[center-24:center+36, round(width*.68):round(width*.89)]

    assert _recover_short_tagged_tail(crop(231), "ArΣ pp") == "Ar peep"
    assert _recover_short_tagged_tail(crop(231), "ArΣ xy") == "ArΣ xy"
    assert _recover_short_tagged_tail(crop(461), "Vy") == "Vy Will"
    assert _recover_short_tagged_tail(crop(461), "Vy² Will") == "Vy² Will"
    # Local 12p race 11 at 2526.35s: compression lowers OCR confidence but
    # the model's `wi` and the visible `Will` agree on the same suffix.
    local = cv2.imread(str(FIXTURES / "team_results" / "local-race11-rank06.png"))
    assert _recover_short_tagged_tail(local, "Vy² wi") == "Vy Will"


def test_human_four_team_crop_restores_only_the_visible_space():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" / "race-results-colored-human-4team-24p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    width = frame.shape[1]
    left, right = (round(width*x) for x in grid["name_crop_x"])
    def crop(rank):
        center = grid["row_centers"][rank]
        return frame[center-24:center+36, left:right]

    assert _restore_visible_word_gap(crop(1), "MMGMike") == "MMG Mike"
    assert _restore_visible_word_gap(crop(6), "OneOneTwo") == "OneOneTwo"
    assert _restore_visible_word_gap(crop(12), "Yoshiller") == "Yoshiller"


def test_repeated_three_letter_team_tag_uses_visible_gap():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" / "race-results-colored-human-4team-24p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None
    width = frame.shape[1]
    left, right = (round(width * x) for x in grid["name_crop_x"])
    def crop(rank):
        center = grid["row_centers"][rank]
        return frame[center-24:center+36, left:right]

    tagged = crop(1)
    ordinary = crop(6)
    rows = [(tagged, "MMGMike")] * 3 + [(ordinary, "OneOneTwo")] * 3
    assert correct_colored_board(rows, None) == ["MMG Mike"] * 3 + ["OneOneTwo"] * 3


def test_repeated_lowercase_tag_keeps_space_on_real_war_board():
    # Danny's 6v6 war, 3sC0CWFOfI4 at 15:02. The first lowercase `s`
    # starts at x=11 in this crop; the older x>=12 component filter missed it.
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "overall-standings-colored-human-tags-6v6.jpg"))
    ranks = (1, 2, 4, 5, 9, 10)
    predictions = ("svc Feysal", "svcFox", "svc xandzr", "svc Danny",
                   "svc barney", "svc Tyler")
    rows = [(frame[c-24:c+36, 1265:1650], name)
            for c, name in zip((77+77*(rank-1) for rank in ranks), predictions)]
    assert correct_colored_board(rows, None) == [
        "svc Feysal", "svc Fox", "svc xandzr", "svc Danny",
        "svc barney", "svc Tyler",
    ]


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_real_war_team_tag_spaces():
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "overall-standings-colored-human-tags-6v6.jpg"))
    result = {"type": "race", "results": [],
              "row_centers": {rank: 77+77*(rank-1) for rank in range(1, 13)},
              "name_crop_x": (1265/1920, 1650/1920)}
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, result)["results"])
    finally:
        reader.close()
    assert {rank: names[rank] for rank in (1, 2, 4, 5, 9, 10)} == {
        1: "svc Feysal", 2: "svc Fox", 4: "svc xandzr",
        5: "svc Danny", 9: "svc barney", 10: "svc Tyler",
    }
    # The last red row uses a filled diamond, unlike its star-tagged teammates.
    assert names[8] == "Z★ Daxx"
    assert names[12] == "Z◆ Toto"


def test_real_war_board_keeps_star_and_diamond_tags_distinct():
    font = configured_font_path()
    if not font or not Path(font).is_file():
        pytest.skip("Optional game font is unavailable")
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "overall-standings-colored-human-tags-6v6.jpg"))
    ranks = (3, 6, 7, 8, 11, 12)
    predictions = ("Z★iEL", "Z★Kiver", "Z★etoiles", "Z★Daxx",
                   "Z★ Demon", "ZToto")
    rows = [(frame[c-24:c+36, 1265:1650], name)
            for c, name in zip((77+77*(rank-1) for rank in ranks), predictions)]
    assert correct_colored_board(rows, FontDecoder(font)) == [
        "Z★iEL", "Z★Kiver", "Z★etoiles", "Z★ Daxx",
        "Z★ Demon", "Z◆ Toto",
    ]


def test_real_war_race_page_locates_all_twelve_colored_rows():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-tags-6v6.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    assert list(grid["row_centers"]) == list(range(1, 13))


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_real_war_race_page():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-tags-6v6.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    # Rank 2's player name is still misread; its Z★ tag is preserved.
    assert names[2].startswith("Z★")
    assert {rank: name for rank, name in names.items() if rank != 2} == {
        1: "svc Fox", 3: "Z★Kiver", 4: "svc Danny", 5: "Z★etoiles",
        6: "Z★ Daxx", 7: "svc xandzr", 8: "svc Feysal",
        9: "Z◆ Toto", 10: "svc Tyler", 11: "Z★ Demon", 12: "svc barney",
    }


def test_repeated_four_letter_tag_uses_real_race_row_gaps():
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-tags-gyro-war-12p.jpg"))
    ranks = (1, 2, 4, 9, 10, 12)
    predictions = ("COoK Jsaav", "COOK★TACOS", "COOK HARU",
                   "CooKyezy", "Cook July", "cooKonata")
    rows = [(frame[c-24:c+36, 1259:1663], name)
            for c, name in zip((77+77*(rank-1) for rank in ranks), predictions)]
    assert correct_colored_board(rows, None) == [
        "COOK Jsaav", "COOK★TACOS", "COOK HARU",
        "COOK yezy", "COOK July", "COOKonata",
    ]


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_independent_gyro_war_tags_and_spaces():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-tags-gyro-war-12p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    assert list(grid["row_centers"]) == list(range(1, 13))
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    assert {rank: name for rank, name in names.items() if rank != 11} == {
        1: "COOK Jsaav", 2: "COOK★TACOS", 3: "RK Milotic",
        4: "COOK HARU", 5: "RK Sanvi", 6: "RK Aleek",
        7: "RK Santi", 8: "RK Splendy", 9: "COOK yezy",
        10: "COOK July", 12: "COOKonata",
    }
    assert names[11] in ("RK Cesar", "RK César")


def test_team_tag_punctuation_stays_with_tag():
    assert _correct_one("Jt_Mint", ("", True)) == "Jt_Mint"
    assert _correct_one("tm!Pepper", ("", True)) == "tm!Pepper"
    assert _correct_one("Xl★ Honey", ("", True)) == "Xl★ Honey"


def test_short_lowercase_tag_glyph_is_not_mistaken_for_a_raised_marker():
    # Game-font seed-94 crop: the short z drops out of the full-height
    # component list; the R in gzRiver must not become the second tag glyph.
    image = cv2.imread(str(FIXTURES / "team_results" /
                           "synthetic-gz-river-no-space.png"))
    assert image is not None
    assert _prefix_evidence(image, None) is None


def test_game_font_recovers_fourth_team_tag_symbol_on_blue_rows():
    font = configured_font_path()
    if not font or not Path(font).is_file():
        pytest.skip("Optional game font is unavailable")
    decoder = FontDecoder(font)
    cases = (
        ((25, "rBUE Pepper>>)", "rBUΣ Pepper>>)"),
         (29, "rBUEPioneer>", "rBUΣPioneer>"),
         (33, "rBUEPaddy", "rBUΣPaddy")),
        ((121, "NAR-Rocket))", "NAR_ Rocket))"),
         (125, "NAR-Honey)", "NAR_ Honey)"),
         (129, "NAR-Twilight>", "NAR_ Twilight>")),
    )
    for group in cases:
        rows = []
        for number, raw, _ in group:
            image = cv2.imread(str(FIXTURES / "team_results" /
                                   f"synthetic-blue-tag-{number:05d}.png"))
            assert image is not None
            rows.append((image, raw))
        assert correct_colored_board(rows, decoder) == [expected for _, _, expected in group]


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_four_team_race_names_and_spaces():
    frame = cv2.imread(str(FIXTURES / "video_analysis" / "race-results-colored-4team-24p.jpg"))
    from mktracker.detection.race_results import RaceResultDetector

    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    # Manually read from the original t2MFiGRmP-g race frame at 2:41.
    assert names == dict(enumerate((
        "Mario", "Yoshi", "Luigi", "Baby Peach", "Toad", "Toadette",
        "Bowser Jr.", "Lakitu", "Waluigi", "Peach", "Rocky Wrench",
        "Baby Daisy",
    ), 1))


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_independent_october_tags_and_spaces():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "team_results" / "october-race.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    assert names == dict(enumerate((
        "Vy² AG64", "ArΣ Spider", "ArΣ peep", "ArΣ NoDice",
        "Vy² Axis", "Vy² Will", "ArΣ Kod", "ArΣ tsu",
        "ArΣ★Rascal", "Vy²TCRapha", "Vy²Barrett", "Vy² Azusa",
    ), 1))


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_restores_human_four_team_name_space():
    from mktracker.detection.race_results import RaceResultDetector

    frame = cv2.imread(str(FIXTURES / "video_analysis" / "race-results-colored-human-4team-24p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    assert names[1] == "MMG Mike"
    assert names[2] == "Stasi"
    assert names[6] == "OneOneTwo"
    assert names[7] == "Maple"


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_independent_streamer_four_team_rows():
    from mktracker.detection.race_results import RaceResultDetector

    # Bryce's separate POV of the June 2025 four-team streamer room,
    # oe4aaGV7i50 at 3:02. The webcam obscures ranks 1-4; ranks 5-12 show
    # red, yellow, green and blue race-result bars with a new human roster.
    frame = cv2.imread(str(FIXTURES / "video_analysis" /
                           "race-results-colored-human-streamers-4team-24p.jpg"))
    grid = RaceResultDetector().detect(frame, teams=True, model_only=True)
    assert grid is not None and grid["type"] == "race"
    assert list(grid["row_centers"]) == list(range(5, 13))
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    assert names == {
        5: "Ze", 6: "HCJustin", 7: "Greg/MG4R", 8: "Elaina",
        9: "speczii", 10: "Penelope", 11: "SideArms", 12: "Rainhoe",
    }


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_reads_independent_yellow_green_tags():
    # Direct crop diagnostic on a held-out four-team overall page. The app's
    # model path currently reads per-race pages, not this overall page.
    directory = FIXTURES / "video_analysis"
    fixture = json.loads((directory / "rayzopp-4team-color-tags.json").read_text())
    frame = cv2.imread(str(directory / fixture["image"]))
    reader = NameModelReader(color=True)
    try:
        for column in fixture["columns"].values():
            grid = {
                "type": "race",
                "results": [(rank, "?") for rank in range(1, 7)],
                "row_centers": dict(enumerate(fixture["row_centers"], 1)),
                "name_crop_x": tuple(x / 1920 for x in column["x_pixels"]),
            }
            names = dict(reader.read(frame, grid)["results"])
            assert [names[rank] for rank in range(1, 7)] == column["names"]
    finally:
        reader.close()


@pytest.mark.skipif(not NameModelReader.color_available(), reason="local color checkpoint unavailable")
def test_local_color_checkpoint_keeps_independent_green_bdd_tag_gaps():
    directory = FIXTURES / "video_analysis"
    fixture = json.loads((directory / "mankalor-4team-color-tags.json").read_text())
    frame = cv2.imread(str(directory / fixture["image"]))
    column = fixture["columns"]["green"]
    grid = {
        "type": "race",
        "results": [(rank, "?") for rank in range(1, 7)],
        "row_centers": dict(enumerate(fixture["row_centers"], 1)),
        "name_crop_x": tuple(x / 1920 for x in column["x_pixels"]),
    }
    reader = NameModelReader(color=True)
    try:
        names = dict(reader.read(frame, grid)["results"])
    finally:
        reader.close()
    assert names[1].startswith("BDD ")
    assert [names[rank] for rank in range(2, 7)] == column["names"][1:]
