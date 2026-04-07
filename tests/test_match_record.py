"""Tests for the standardised on-disk match record schema.

Covers:
* dataclass round-tripping (``to_dict`` / ``from_dict``)
* atomic ``save`` / ``load`` against a temp directory
* ``list_matches`` ordering and skipping of legacy folders
* schema versioning constant
* Unicode preservation in player names (☆ etc.)
* missing-file / corrupt-file handling
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mktracker.match_record import (
    MATCH_FILE,
    SCHEMA_VERSION,
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    PlayerPlacement,
    RaceRecord,
    TeamGroup,
    list_matches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _settings(**overrides) -> MatchSettingsRecord:
    base = dict(
        cc_class="150cc",
        teams="No Teams",
        items="Normal",
        com_difficulty="No COM",
        race_count=12,
        intermission="10 seconds",
    )
    base.update(overrides)
    return MatchSettingsRecord(**base)


def _ocr_race(race_number: int = 1) -> RaceRecord:
    return RaceRecord(
        race_number=race_number,
        track_name="Mario Bros. Circuit",
        players=["Alice", "Bob", "Carol"],
        user_rank=2,
        mode=None,
        placements=[
            PlayerPlacement(place=1, name="Alice"),
            PlayerPlacement(place=2, name="Bob"),
            PlayerPlacement(place=3, name="Carol"),
        ],
        teams=None,
    )


def _gemini_race(race_number: int = 2) -> RaceRecord:
    return RaceRecord(
        race_number=race_number,
        track_name="Rainbow Road",
        players=["Alice", "Bob", "Carol"],
        user_rank=1,
        mode="no_teams",
        placements=[
            PlayerPlacement(place=1, name="Carol"),
            PlayerPlacement(place=2, name="Alice"),
            PlayerPlacement(place=3, name="Bob"),
        ],
        teams=[
            TeamGroup(
                name=None,
                points=None,
                winner=None,
                players=[
                    PlayerPlacement(place=1, name="Carol"),
                    PlayerPlacement(place=2, name="Alice"),
                    PlayerPlacement(place=3, name="Bob"),
                ],
            )
        ],
    )


def _final_standings() -> FinalStandings:
    return FinalStandings(
        mode="no_teams",
        players=[
            PlayerPlacement(place=1, name="Alice", score=180),
            PlayerPlacement(place=2, name="Carol", score=165),
            PlayerPlacement(place=3, name="Bob", score=150),
        ],
        teams=None,
    )


def _full_record() -> MatchRecord:
    return MatchRecord(
        match_id="20260406_120000",
        started_at="2026-04-06T12:00:00",
        completed_at="2026-04-06T12:45:00",
        settings=_settings(),
        races=[_ocr_race(1), _gemini_race(2)],
        final_standings=_final_standings(),
    )


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


def test_schema_version_is_one():
    """Bump the version field whenever the schema changes — this test is
    a deliberate alarm to remind whoever bumps it to write a migration."""
    assert SCHEMA_VERSION == 1


def test_match_file_constant():
    assert MATCH_FILE == "match.json"


# ---------------------------------------------------------------------------
# PlayerPlacement
# ---------------------------------------------------------------------------


def test_player_placement_omits_score_when_none():
    p = PlayerPlacement(place=1, name="Alice")
    assert p.to_dict() == {"place": 1, "name": "Alice"}


def test_player_placement_includes_score_when_set():
    p = PlayerPlacement(place=1, name="Alice", score=15)
    assert p.to_dict() == {"place": 1, "name": "Alice", "score": 15}


def test_player_placement_round_trip_with_score():
    p = PlayerPlacement(place=3, name="Carol", score=42)
    assert PlayerPlacement.from_dict(p.to_dict()) == p


def test_player_placement_round_trip_without_score():
    p = PlayerPlacement(place=3, name="Carol")
    assert PlayerPlacement.from_dict(p.to_dict()) == p


def test_player_placement_from_dict_coerces_strings():
    p = PlayerPlacement.from_dict({"place": "2", "name": "Bob", "score": "10"})
    assert p.place == 2
    assert p.score == 10


def test_player_placement_unicode_preserved():
    p = PlayerPlacement(place=1, name="TA☆Collins")
    back = PlayerPlacement.from_dict(p.to_dict())
    assert back.name == "TA☆Collins"


# ---------------------------------------------------------------------------
# TeamGroup
# ---------------------------------------------------------------------------


def test_team_group_round_trip():
    t = TeamGroup(
        name="Red Team",
        points=42,
        winner=True,
        players=[PlayerPlacement(place=1, name="A"), PlayerPlacement(place=3, name="B")],
    )
    assert TeamGroup.from_dict(t.to_dict()) == t


def test_team_group_round_trip_with_nulls():
    t = TeamGroup(name=None, points=None, winner=None, players=[])
    assert TeamGroup.from_dict(t.to_dict()) == t


# ---------------------------------------------------------------------------
# RaceRecord
# ---------------------------------------------------------------------------


def test_race_record_ocr_round_trip():
    r = _ocr_race()
    assert RaceRecord.from_dict(r.to_dict()) == r


def test_race_record_gemini_round_trip():
    r = _gemini_race()
    assert RaceRecord.from_dict(r.to_dict()) == r


def test_race_record_teams_none_serialised_as_null():
    r = _ocr_race()
    assert r.to_dict()["teams"] is None


def test_race_record_is_mutable_for_stale_writes():
    """Stale-callback paths mutate fields in place, so RaceRecord must
    not be a frozen dataclass."""
    r = _ocr_race()
    r.user_rank = 7
    r.mode = "no_teams"
    r.placements = [PlayerPlacement(place=1, name="Updated")]
    assert r.user_rank == 7
    assert r.mode == "no_teams"
    assert r.placements[0].name == "Updated"


# ---------------------------------------------------------------------------
# FinalStandings
# ---------------------------------------------------------------------------


def test_final_standings_round_trip():
    f = _final_standings()
    assert FinalStandings.from_dict(f.to_dict()) == f


def test_final_standings_empty_round_trip():
    f = FinalStandings()
    assert FinalStandings.from_dict(f.to_dict()) == f


# ---------------------------------------------------------------------------
# MatchSettingsRecord
# ---------------------------------------------------------------------------


def test_match_settings_round_trip():
    s = _settings()
    assert MatchSettingsRecord.from_dict(s.to_dict()) == s


def test_match_settings_from_dict_coerces_race_count():
    s = MatchSettingsRecord.from_dict({
        "cc_class": "150cc",
        "teams": "No Teams",
        "items": "Normal",
        "com_difficulty": "No COM",
        "race_count": "8",  # string!
        "intermission": "10 seconds",
    })
    assert s.race_count == 8
    assert isinstance(s.race_count, int)


# ---------------------------------------------------------------------------
# MatchRecord
# ---------------------------------------------------------------------------


def test_match_record_full_round_trip():
    rec = _full_record()
    back = MatchRecord.from_dict(rec.to_dict())
    assert back == rec


def test_match_record_to_dict_includes_version():
    rec = _full_record()
    d = rec.to_dict()
    assert d["version"] == SCHEMA_VERSION


def test_match_record_with_no_final_standings():
    rec = MatchRecord(
        match_id="20260406_120000",
        started_at="2026-04-06T12:00:00",
        completed_at=None,
        settings=_settings(),
        races=[_ocr_race()],
        final_standings=None,
    )
    d = rec.to_dict()
    assert d["final_standings"] is None
    assert d["completed_at"] is None
    back = MatchRecord.from_dict(d)
    assert back.final_standings is None
    assert back.completed_at is None


def test_match_record_with_no_races():
    rec = MatchRecord(
        match_id="20260406_120000",
        started_at="2026-04-06T12:00:00",
        completed_at=None,
        settings=_settings(),
        races=[],
        final_standings=None,
    )
    back = MatchRecord.from_dict(rec.to_dict())
    assert back.races == []


def test_match_record_from_dict_defaults_version_when_missing():
    """Old records without an explicit version field still load."""
    d = _full_record().to_dict()
    del d["version"]
    rec = MatchRecord.from_dict(d)
    assert rec.version == SCHEMA_VERSION


def test_match_record_from_dict_missing_settings_raises():
    d = _full_record().to_dict()
    del d["settings"]
    with pytest.raises(KeyError):
        MatchRecord.from_dict(d)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_creates_match_json(tmp_path: Path):
    rec = _full_record()
    match_dir = tmp_path / "20260406_120000"
    written = rec.save(match_dir)
    assert written == match_dir / MATCH_FILE
    assert written.exists()
    # File is valid JSON.
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["match_id"] == "20260406_120000"


def test_save_creates_parent_directory(tmp_path: Path):
    rec = _full_record()
    match_dir = tmp_path / "nested" / "deeper" / "20260406_120000"
    rec.save(match_dir)
    assert (match_dir / MATCH_FILE).exists()


def test_save_overwrites_existing(tmp_path: Path):
    """Repeated saves are atomic and idempotent."""
    rec1 = _full_record()
    rec1.save(tmp_path)
    rec2 = _full_record()
    rec2.completed_at = "2026-04-06T13:00:00"
    rec2.save(tmp_path)
    loaded = MatchRecord.load(tmp_path)
    assert loaded.completed_at == "2026-04-06T13:00:00"


def test_save_atomic_no_tmp_left_behind(tmp_path: Path):
    """The atomic-rename pattern should leave no .json.tmp turd behind."""
    _full_record().save(tmp_path)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_save_load_round_trip(tmp_path: Path):
    rec = _full_record()
    rec.save(tmp_path)
    loaded = MatchRecord.load(tmp_path)
    assert loaded == rec


def test_save_load_preserves_unicode(tmp_path: Path):
    """Special characters (☆, ★, π, ♪) common in MK player names must
    survive a JSON round-trip — this is a known weak spot for Tesseract,
    so we explicitly verify the storage layer doesn't mangle them."""
    rec = _full_record()
    rec.races[0].placements[0].name = "TA☆Collins★π♪"
    rec.save(tmp_path)
    loaded = MatchRecord.load(tmp_path)
    assert loaded.races[0].placements[0].name == "TA☆Collins★π♪"


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MatchRecord.load(tmp_path / "doesnotexist")


def test_load_corrupt_file_raises(tmp_path: Path):
    (tmp_path / MATCH_FILE).write_text("not valid json {", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        MatchRecord.load(tmp_path)


# ---------------------------------------------------------------------------
# list_matches
# ---------------------------------------------------------------------------


def test_list_matches_empty_dir(tmp_path: Path):
    assert list_matches(tmp_path) == []


def test_list_matches_missing_dir(tmp_path: Path):
    assert list_matches(tmp_path / "doesnotexist") == []


def test_list_matches_skips_folders_without_match_json(tmp_path: Path):
    """Legacy debug-only folders predating the schema must not blow up
    list_matches."""
    legacy = tmp_path / "20260101_120000"
    legacy.mkdir()
    (legacy / "match_settings.png").write_bytes(b"fake png")

    new = tmp_path / "20260201_120000"
    new.mkdir()
    rec = _full_record()
    rec.match_id = "20260201_120000"
    rec.save(new)

    matches = list_matches(tmp_path)
    assert len(matches) == 1
    assert matches[0].match_id == "20260201_120000"


def test_list_matches_skips_files(tmp_path: Path):
    """A loose file at the top level shouldn't break iteration."""
    (tmp_path / "stray.txt").write_text("ignore me")
    rec = _full_record()
    rec.save(tmp_path / "20260406_120000")
    matches = list_matches(tmp_path)
    assert len(matches) == 1


def test_list_matches_orders_newest_first(tmp_path: Path):
    """Folder names are timestamps, so reverse-sorted iteration is the
    correct chronological ordering."""
    for stamp in ("20260101_120000", "20260301_120000", "20260201_120000"):
        rec = _full_record()
        rec.match_id = stamp
        rec.save(tmp_path / stamp)
    matches = list_matches(tmp_path)
    assert [m.match_id for m in matches] == [
        "20260301_120000",
        "20260201_120000",
        "20260101_120000",
    ]


def test_list_matches_skips_corrupt_records(tmp_path: Path, caplog):
    """A bad match.json shouldn't take down the whole listing."""
    good = tmp_path / "20260301_120000"
    good.mkdir()
    rec = _full_record()
    rec.match_id = "20260301_120000"
    rec.save(good)

    bad = tmp_path / "20260201_120000"
    bad.mkdir()
    (bad / MATCH_FILE).write_text("garbage", encoding="utf-8")

    with caplog.at_level("WARNING"):
        matches = list_matches(tmp_path)
    assert [m.match_id for m in matches] == ["20260301_120000"]
    assert any("Failed to load match record" in r.message for r in caplog.records)
