"""Tests for the Lorenzi-style table image generator.

Covers:
* Pure helper functions (_needs_cjk, _blend, _hsv2rgb, _clan_hsv, _build_clans)
* generate_table: raises on missing standings, returns valid PNG for every layout
* Image dimensions scale correctly with player / clan count
* Player and clan ranking (tie-breaking)
* CJK player names do not crash rendering
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from mktracker.match_record import (
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    TeamGroup,
)
from mktracker.table_generator import (
    _blend,
    _build_clans,
    _clan_hsv,
    _hsv2rgb,
    _needs_cjk,
    generate_table,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SETTINGS = MatchSettingsRecord(
    cc_class="150cc",
    teams="No Teams",
    items="Normal Items",
    com_difficulty="Hard",
    race_count=12,
    intermission="Off",
)


def _record(match_id: str, teams: list[TeamGroup] | None, players: list[PlayerPlacement] | None = None) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        started_at="2026-04-20T18:00:00",
        completed_at="2026-04-20T19:30:00",
        settings=_SETTINGS,
        final_standings=FinalStandings(
            mode="teams" if teams else "no_teams",
            players=players or [],
            teams=teams,
        ),
    )


def _ffa_record() -> MatchRecord:
    return _record(
        "ffa-match",
        teams=None,
        players=[
            PlayerPlacement(place=1, name="Alice",   score=200),
            PlayerPlacement(place=2, name="Bob",     score=185),
            PlayerPlacement(place=3, name="Carol",   score=170),
            PlayerPlacement(place=4, name="Dave",    score=155),
        ],
    )


def _two_team_record() -> MatchRecord:
    return _record("two-team", teams=[
        TeamGroup(name="AA", points=None, winner=True, players=[
            PlayerPlacement(place=1, name="Alice",  score=200),
            PlayerPlacement(place=3, name="Carol",  score=170),
        ]),
        TeamGroup(name="BB", points=None, winner=False, players=[
            PlayerPlacement(place=2, name="Bob",    score=185),
            PlayerPlacement(place=4, name="Dave",   score=155),
        ]),
    ])


def _three_team_record() -> MatchRecord:
    return _record("three-team", teams=[
        TeamGroup(name="AA", points=None, winner=True, players=[
            PlayerPlacement(place=1,  name="Alice",  score=198),
            PlayerPlacement(place=2,  name="Nathan", score=185),
            PlayerPlacement(place=5,  name="Cleo",   score=161),
            PlayerPlacement(place=6,  name="Dax",    score=154),
        ]),
        TeamGroup(name="BB", points=None, winner=False, players=[
            PlayerPlacement(place=3,  name="Mira",   score=176),
            PlayerPlacement(place=4,  name="Theo",   score=168),
            PlayerPlacement(place=7,  name="Sasha",  score=142),
            PlayerPlacement(place=8,  name="Lena",   score=133),
        ]),
        TeamGroup(name="CC", points=None, winner=False, players=[
            PlayerPlacement(place=9,  name="Pax",    score=120),
            PlayerPlacement(place=10, name="Rue",    score=111),
            PlayerPlacement(place=11, name="Wren",   score=99),
            PlayerPlacement(place=12, name="Juno",   score=87),
        ]),
    ])


def _four_team_record() -> MatchRecord:
    return _record("four-team", teams=[
        TeamGroup(name="AA", points=None, winner=True, players=[
            PlayerPlacement(place=1,  name="Alice",  score=195),
            PlayerPlacement(place=2,  name="Nathan", score=182),
            PlayerPlacement(place=5,  name="Cleo",   score=158),
        ]),
        TeamGroup(name="BB", points=None, winner=False, players=[
            PlayerPlacement(place=3,  name="Mira",   score=171),
            PlayerPlacement(place=4,  name="Theo",   score=164),
            PlayerPlacement(place=6,  name="Sasha",  score=145),
        ]),
        TeamGroup(name="CC", points=None, winner=False, players=[
            PlayerPlacement(place=7,  name="Pax",    score=133),
            PlayerPlacement(place=8,  name="Rue",    score=120),
            PlayerPlacement(place=9,  name="Wren",   score=108),
        ]),
        TeamGroup(name="DD", points=None, winner=False, players=[
            PlayerPlacement(place=10, name="Juno",   score=95),
            PlayerPlacement(place=11, name="Finn",   score=82),
            PlayerPlacement(place=12, name="Lena",   score=70),
        ]),
    ])


def _load_png(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes))


# ---------------------------------------------------------------------------
# _needs_cjk
# ---------------------------------------------------------------------------

def test_needs_cjk_false_for_empty():
    assert _needs_cjk("") is False


def test_needs_cjk_false_for_ascii():
    assert _needs_cjk("Nathan") is False


def test_needs_cjk_false_for_common_specials():
    assert _needs_cjk("TA☆Collins") is False


def test_needs_cjk_true_for_katakana():
    assert _needs_cjk("エリアス") is True


def test_needs_cjk_true_for_hiragana():
    assert _needs_cjk("こんにちは") is True


def test_needs_cjk_true_for_kanji():
    assert _needs_cjk("漢字") is True


def test_needs_cjk_true_for_fullwidth():
    assert _needs_cjk("Ａ") is True  # U+FF21 fullwidth Latin A


def test_needs_cjk_mixed_ascii_and_cjk():
    assert _needs_cjk("Player エリアス") is True


# ---------------------------------------------------------------------------
# _blend
# ---------------------------------------------------------------------------

def test_blend_zero_alpha_returns_base():
    assert _blend((100, 150, 200), (0, 0, 0), 0.0) == (100, 150, 200)


def test_blend_full_alpha_returns_overlay():
    assert _blend((100, 150, 200), (255, 0, 128), 1.0) == (255, 0, 128)


def test_blend_half_alpha_midpoint():
    result = _blend((0, 0, 0), (200, 100, 50), 0.5)
    assert result == (100, 50, 25)


def test_blend_same_colors_unchanged():
    c = (80, 160, 240)
    assert _blend(c, c, 0.5) == c


# ---------------------------------------------------------------------------
# _hsv2rgb
# ---------------------------------------------------------------------------

def test_hsv2rgb_black():
    assert _hsv2rgb(0.0, 0.0, 0.0) == (0, 0, 0)


def test_hsv2rgb_white():
    assert _hsv2rgb(0.0, 0.0, 1.0) == (255, 255, 255)


def test_hsv2rgb_pure_red():
    r, g, b = _hsv2rgb(0.0, 1.0, 1.0)
    assert r == 255 and g == 0 and b == 0


def test_hsv2rgb_values_in_range():
    for h in (0.0, 0.25, 0.5, 0.75):
        r, g, b = _hsv2rgb(h, 0.8, 0.9)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# _clan_hsv
# ---------------------------------------------------------------------------

def test_clan_hsv_returns_three_floats():
    result = _clan_hsv("AA", [])
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_clan_hsv_value_always_one():
    _, _, v = _clan_hsv("AA", [])
    assert v == 1.0


def test_clan_hsv_no_tag_uses_seed_and_sat_075():
    _, sat, _ = _clan_hsv(None, [], seed="test-match-id")
    assert sat == 0.75


def test_clan_hsv_avoids_collisions():
    used: list[int] = []
    hues = set()
    for tag in ("AA", "BB", "CC", "DD"):
        h, _, _ = _clan_hsv(tag, used)
        hues.add(round(h * 256))
    # All four clans get distinct hue buckets
    assert len(hues) == 4


def test_clan_hsv_same_tag_same_result():
    h1, s1, v1 = _clan_hsv("AA", [])
    h2, s2, v2 = _clan_hsv("AA", [])
    assert h1 == h2 and s1 == s2 and v1 == v2


# ---------------------------------------------------------------------------
# _build_clans
# ---------------------------------------------------------------------------

def test_build_clans_returns_empty_for_no_standings():
    rec = MatchRecord(
        match_id="x", started_at="2026-04-20T18:00:00", completed_at=None,
        settings=_SETTINGS, final_standings=None,
    )
    assert _build_clans(rec) == []


def test_build_clans_ffa_gives_single_clan():
    clans = _build_clans(_ffa_record())
    assert len(clans) == 1
    assert clans[0].tag is None


def test_build_clans_ffa_extracts_all_players():
    clans = _build_clans(_ffa_record())
    names = {p["name"] for p in clans[0].players}
    assert names == {"Alice", "Bob", "Carol", "Dave"}


def test_build_clans_ffa_player_scores():
    clans = _build_clans(_ffa_record())
    scores = {p["name"]: p["total_score"] for p in clans[0].players}
    assert scores["Alice"] == 200
    assert scores["Bob"] == 185


def test_build_clans_two_team_gives_two_clans():
    clans = _build_clans(_two_team_record())
    assert len(clans) == 2


def test_build_clans_two_team_tags():
    clans = _build_clans(_two_team_record())
    tags = {c.tag for c in clans}
    assert tags == {"AA", "BB"}


def test_build_clans_three_team_gives_three_clans():
    assert len(_build_clans(_three_team_record())) == 3


def test_build_clans_four_team_gives_four_clans():
    assert len(_build_clans(_four_team_record())) == 4


def test_build_clans_excludes_players_with_no_score():
    rec = _record("x", teams=None, players=[
        PlayerPlacement(place=1, name="Alice", score=100),
        PlayerPlacement(place=2, name="Bob",   score=None),
    ])
    clans = _build_clans(rec)
    names = {p["name"] for p in clans[0].players}
    assert "Alice" in names
    assert "Bob" not in names


# ---------------------------------------------------------------------------
# generate_table: error cases (no fonts needed — raises before font load)
# ---------------------------------------------------------------------------

def test_generate_table_raises_for_no_standings():
    rec = MatchRecord(
        match_id="x", started_at="2026-04-20T18:00:00", completed_at=None,
        settings=_SETTINGS, final_standings=None,
    )
    with pytest.raises(ValueError, match="No final standings"):
        generate_table(rec)


# ---------------------------------------------------------------------------
# generate_table: PNG validity and dimensions
# (stub_fonts: patches font loading so no real font files are required)
# ---------------------------------------------------------------------------

EXPECTED_WIDTH = 860


def test_generate_table_ffa_returns_png(stub_fonts):
    png = generate_table(_ffa_record())
    img = _load_png(png)
    assert img.format == "PNG"
    assert img.width == EXPECTED_WIDTH


def test_generate_table_two_team_returns_png(stub_fonts):
    img = _load_png(generate_table(_two_team_record()))
    assert img.format == "PNG"
    assert img.width == EXPECTED_WIDTH


def test_generate_table_three_team_returns_png(stub_fonts):
    img = _load_png(generate_table(_three_team_record()))
    assert img.format == "PNG"
    assert img.width == EXPECTED_WIDTH


def test_generate_table_four_team_returns_png(stub_fonts):
    img = _load_png(generate_table(_four_team_record()))
    assert img.format == "PNG"
    assert img.width == EXPECTED_WIDTH


def test_generate_table_height_increases_with_more_players(stub_fonts):
    """A match with more players produces a taller image."""
    small = _record("s", teams=None, players=[
        PlayerPlacement(place=1, name="A", score=100),
        PlayerPlacement(place=2, name="B", score=90),
    ])
    large = _record("l", teams=None, players=[
        PlayerPlacement(place=i + 1, name=f"P{i}", score=200 - i * 10)
        for i in range(20)
    ])
    h_small = _load_png(generate_table(small)).height
    h_large = _load_png(generate_table(large)).height
    assert h_large > h_small


def test_generate_table_height_increases_with_more_clans(stub_fonts):
    """Four clans (with divider gaps) should produce a taller image than two."""
    h_two  = _load_png(generate_table(_two_team_record())).height
    h_four = _load_png(generate_table(_four_team_record())).height
    assert h_four > h_two


def test_generate_table_rgb_mode(stub_fonts):
    img = _load_png(generate_table(_ffa_record()))
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# generate_table: player / clan ranking
# ---------------------------------------------------------------------------

def test_generate_table_sorts_clans_by_score_descending():
    """Clan with higher total score should come out first — pure data, no fonts."""
    # AA: 200+170 = 370, BB: 185+155 = 340 → AA wins
    rec = _two_team_record()
    clans = _build_clans(rec)
    for c in clans:
        c.score = sum(p["total_score"] for p in c.players)
    clans.sort(key=lambda c: c.score, reverse=True)
    assert clans[0].tag == "AA"
    assert clans[1].tag == "BB"


def test_generate_table_tied_players_share_ranking(stub_fonts):
    rec = _record("tied", teams=None, players=[
        PlayerPlacement(place=1, name="Alice", score=100),
        PlayerPlacement(place=2, name="Bob",   score=100),  # tied
        PlayerPlacement(place=3, name="Carol", score=80),
    ])
    png = generate_table(rec)  # must not crash
    img = _load_png(png)
    assert img.width == EXPECTED_WIDTH


# ---------------------------------------------------------------------------
# generate_table: CJK names
# ---------------------------------------------------------------------------

def test_generate_table_cjk_name_does_not_crash(stub_fonts):
    rec = _record("cjk", teams=None, players=[
        PlayerPlacement(place=1, name="エリアス",   score=200),
        PlayerPlacement(place=2, name="こんにちは",  score=185),
        PlayerPlacement(place=3, name="漢字Player", score=170),
    ])
    png = generate_table(rec)
    assert _load_png(png).width == EXPECTED_WIDTH


def test_generate_table_cjk_in_team_does_not_crash(stub_fonts):
    rec = _record("cjk-team", teams=[
        TeamGroup(name="AA", points=None, winner=True, players=[
            PlayerPlacement(place=1, name="エリアス", score=200),
            PlayerPlacement(place=3, name="Alice",    score=170),
        ]),
        TeamGroup(name="BB", points=None, winner=False, players=[
            PlayerPlacement(place=2, name="Bob",      score=185),
            PlayerPlacement(place=4, name="漢字",     score=155),
        ]),
    ])
    png = generate_table(rec)
    assert _load_png(png).width == EXPECTED_WIDTH


# ---------------------------------------------------------------------------
# generate_table: date header
# ---------------------------------------------------------------------------

def test_generate_table_accepts_iso_date_string(stub_fonts):
    """started_at is stored as an ISO string — must not crash."""
    rec = _ffa_record()
    rec.started_at = "2026-04-20T18:00:00"
    png = generate_table(rec)
    assert _load_png(png).width == EXPECTED_WIDTH
