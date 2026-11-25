"""Verify glyph recovery with the optional, externally supplied game font."""
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest
from dotenv import dotenv_values

from mktracker.detection.font_symbols import FontSymbolMatcher, correct_symbol_words, save_font_path, configured_font_path, _load_matcher
from mktracker.detection.name_consensus import name_key, PlacementConsensus
from mktracker.detection.team_results import read_team_results


@pytest.fixture
def game_font(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = os.environ.get("GAME_FONT_PATH") or dotenv_values(root / ".env").get("GAME_FONT_PATH", "")
    if not path or not Path(path).is_file():
        pytest.skip("Set GAME_FONT_PATH to the local game font for glyph integration tests")
    monkeypatch.setenv("GAME_FONT_PATH", path)
    return path


def word_image(font_path, text):
    font = ImageFont.truetype(font_path, 64)
    x, y, right, bottom = font.getbbox(text)
    image = Image.new("L", (right-x+80, bottom-y+80), 255)
    ImageDraw.Draw(image).text((40-x, 40-y), text, font=font, fill=0)
    return np.array(image), {"left": 40, "width": right-x, "conf": 75}


@pytest.mark.parametrize("visible,ocr,expected", [
    ("ArΣ", "ArZ", "ArΣ"),
    ("ArΣ★Rascal", "ArZ®Rascal", "ArΣ★Rascal"),
    ("ArΣ★Rascal", "ArZRascal", "ArΣ★Rascal"),
    ("Vy²", "Vy’", "Vy²"),
])
def test_font_restores_only_visually_verified_symbols(game_font, visible, ocr, expected):
    image, word = word_image(game_font, visible)
    word["text"] = ocr
    corrected = FontSymbolMatcher(game_font).correct_words(image, [word])
    assert corrected[0]["text"] == expected


@pytest.mark.parametrize("visible,ocr", [("ArZ", "ArZ"), ("Parrett", "Parrett"), ("Vy'", "Vy'")])
def test_font_preserves_correct_ascii_and_does_not_force_specials(game_font, visible, ocr):
    image, word = word_image(game_font, visible)
    word["text"] = ocr
    assert FontSymbolMatcher(game_font).correct_words(image, [word])[0]["text"] == ocr


def test_full_font_can_correct_a_visually_proven_ascii_error(game_font):
    image, word = word_image(game_font, "Barrett")
    word["text"] = "Parrett"
    assert FontSymbolMatcher(game_font).correct_words(image, [word])[0]["text"] == "Barrett"


@pytest.mark.parametrize("text", ["AéA", "AγA", "AЖA", "AあA", "AカA", "A★A", "A→A", "A\ue068A"])
def test_unseen_tags_and_icons_without_tesseract(game_font, text):
    matcher = _load_matcher(game_font, Path(game_font).stat().st_mtime_ns)
    image, _ = word_image(game_font, text)
    corrected = matcher.correct_words(image, [])
    assert corrected and matcher.canonical(corrected[0]["text"]) == matcher.canonical(text)


def test_visual_families_preserve_spelling_and_distinct_symbols(game_font):
    matcher = _load_matcher(game_font, Path(game_font).stat().st_mtime_ns)
    assert name_key("Alpha") == name_key("Αlpha") == name_key("Аlpha")
    assert name_key("Tag★") != name_key("Tag→")
    assert name_key("Tagé") != name_key("Tage")
    assert name_key("Tagγ") != name_key("Tagy")
    image, word = word_image(game_font, "AΑA")
    word["text"] = "AAA"
    assert matcher.correct_words(image, [word])[0]["text"] == "AAA"


@pytest.mark.parametrize("char", ["Ì", "“"])
def test_small_ambiguous_glyph_preserves_ocr_and_requests_review(game_font, char):
    matcher = _load_matcher(game_font, Path(game_font).stat().st_mtime_ns)
    font = ImageFont.truetype(game_font, 32)
    text = "A"+char+"A"
    x, y, right, bottom = font.getbbox(text)
    image = Image.new("L", (right-x+80, bottom-y+80), 255)
    ImageDraw.Draw(image).text((40-x, 40-y), text, font=font, fill=0)
    corrected = matcher.correct_words(np.array(image), [{"text": text, "left":40, "conf":90}])
    assert corrected[0]["text"] == text
    assert corrected[0]["font_uncertain"]


def test_unresolved_glyph_is_not_confirmed_by_repeating_frames():
    consensus = PlacementConsensus()
    result = {"results": [(1, "Player")], "confidences": {1:95},
              "glyph_reviews": {1: {"needs_review":True}}}
    consensus.add(result, frame_id=1)
    consensus.add(result, frame_id=2)
    assert consensus.evidence()["1"]["glyph_review"]
    assert consensus.evidence()["1"]["status"] == "review"


def test_real_october_results_recover_sigma_and_star(game_font):
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "team_results" / "october-race.jpg"))
    result = read_team_results(frame)
    assert result["type"] == "race"
    names = dict(result["results"])
    assert sorted(names) == list(range(1, 13))
    for rank in (2, 3, 4, 7, 8, 9):
        assert names[rank].startswith("ArΣ"), (rank, names[rank])
    assert names[9] == "ArΣ★Rascal"


def test_missing_font_keeps_normal_ocr(monkeypatch, tmp_path):
    monkeypatch.setenv("GAME_FONT_PATH", str(tmp_path / "missing.ttf"))
    words = [{"text": "ArZ", "left": 0, "width": 50, "conf": 80}]
    assert correct_symbol_words(np.full((60, 100), 255, np.uint8), words) == words


def test_font_setting_is_validated_and_can_be_cleared(game_font, tmp_path, monkeypatch):
    monkeypatch.delenv("GAME_FONT_PATH")
    monkeypatch.chdir(tmp_path)
    save_font_path(game_font)
    assert configured_font_path() == str(Path(game_font).resolve())
    bad = tmp_path / "bad.ttf"
    bad.write_text("not a font")
    with pytest.raises(OSError):
        save_font_path(str(bad))
    assert configured_font_path() == str(Path(game_font).resolve())
    save_font_path("")
    assert configured_font_path() == ""
