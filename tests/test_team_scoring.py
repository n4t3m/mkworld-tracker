"""Tests for the MK race-points scoring tables and team-score derivation.

Pinned invariants:

* 12-player race totals sum to 82.
* 24-player race totals sum to 144.
* Per-race team scores must be derived locally for 12/24-player races
  (Gemini's +points arithmetic is unreliable) and may trust Gemini's
  reported ``team.points`` for any other count.
"""
from __future__ import annotations

import pytest

from mktracker.match_record import (
    MatchSettingsRecord,
    PlayerPlacement,
    RaceRecord,
    TeamGroup,
    race_fields_from_gemini,
)
from mktracker.team_scoring import (
    DERIVED_PLAYER_COUNTS,
    MK_POINTS_12P,
    MK_POINTS_24P,
    points_for_place,
    race_player_count,
    race_team_scores,
)


# ---------------------------------------------------------------------------
# Scoring-table invariants
# ---------------------------------------------------------------------------


def test_mk_points_12p_sum():
    assert sum(MK_POINTS_12P) == 82
    assert len(MK_POINTS_12P) == 12


def test_mk_points_24p_sum():
    """24-player race totals to 144 (1+2+2+3+3+3+4+4+4+5+5+5+6+6+6+7+7+8+8+9+9+10+12+15)."""
    assert sum(MK_POINTS_24P) == 144
    assert len(MK_POINTS_24P) == 24


def test_derived_player_counts_only_12_and_24():
    """We only locally derive points where MK has a fixed scoring table."""
    assert DERIVED_PLAYER_COUNTS == (12, 24)


# ---------------------------------------------------------------------------
# points_for_place
# ---------------------------------------------------------------------------


def test_points_for_place_default_is_12p():
    assert points_for_place(1) == 15
    assert points_for_place(12) == 1


def test_points_for_place_24p_lookup():
    assert points_for_place(1, 24) == 15
    assert points_for_place(5, 24) == 9
    assert points_for_place(12, 24) == 6
    assert points_for_place(21, 24) == 3
    assert points_for_place(24, 24) == 1


def test_points_for_place_unknown_count_falls_back_to_12p():
    """Old callers that pass arbitrary counts (or none) must still work."""
    assert points_for_place(3, 9) == 10
    assert points_for_place(13, 9) == 0


def test_points_for_place_out_of_range():
    assert points_for_place(0) == 0
    assert points_for_place(13, 12) == 0
    assert points_for_place(25, 24) == 0


# ---------------------------------------------------------------------------
# race_fields_from_gemini — points override
# ---------------------------------------------------------------------------


def _gemini_response_12p_two_teams(
    *, red_points: int, blue_points: int,
) -> dict:
    """Build a fake Gemini race-results dict for a 12-player two-team race.

    Red holds places 1, 3, 5, 7, 9, 11 (15+10+8+6+4+2 = 45);
    Blue holds places 2, 4, 6, 8, 10, 12 (12+9+7+5+3+1 = 37).
    Sum = 82.
    """
    return {
        "mode": "two_teams",
        "teams": [
            {
                "name": "Red",
                "race_points": red_points,
                "race_winner": True,
                "players": [
                    {"place": p, "name": f"R{p}"}
                    for p in (1, 3, 5, 7, 9, 11)
                ],
            },
            {
                "name": "Blue",
                "race_points": blue_points,
                "race_winner": False,
                "players": [
                    {"place": p, "name": f"B{p}"}
                    for p in (2, 4, 6, 8, 10, 12)
                ],
            },
        ],
    }


def test_race_fields_overrides_gemini_points_for_12p():
    """If Gemini reports nonsense point totals for a 12p race, we
    overwrite them with locally-derived sums."""
    gem = _gemini_response_12p_two_teams(red_points=999, blue_points=42)
    _, teams, _ = race_fields_from_gemini(gem)

    points_by_name = {t.name: t.points for t in teams}
    assert points_by_name == {"Red": 45, "Blue": 37}
    assert points_by_name["Red"] + points_by_name["Blue"] == 82


def test_race_fields_overrides_gemini_points_for_24p():
    """24-player two-team split: red holds odd places, blue holds even."""
    odd = list(range(1, 25, 2))
    even = list(range(2, 25, 2))
    expected_red = sum(MK_POINTS_24P[p - 1] for p in odd)
    expected_blue = sum(MK_POINTS_24P[p - 1] for p in even)
    assert expected_red + expected_blue == 144

    gem = {
        "mode": "two_teams",
        "teams": [
            {
                "name": "Red",
                "race_points": 0,  # nonsense
                "race_winner": True,
                "players": [{"place": p, "name": f"R{p}"} for p in odd],
            },
            {
                "name": "Blue",
                "race_points": 0,  # nonsense
                "race_winner": False,
                "players": [{"place": p, "name": f"B{p}"} for p in even],
            },
        ],
    }
    _, teams, _ = race_fields_from_gemini(gem)

    points_by_name = {t.name: t.points for t in teams}
    assert points_by_name == {"Red": expected_red, "Blue": expected_blue}


def test_race_fields_trusts_gemini_points_for_non_12_non_24():
    """If a race is configured with an unusual player count (e.g. 9 in
    a small CPU lobby), we have no fixed scoring table and must trust
    whatever Gemini reported."""
    gem = {
        "mode": "two_teams",
        "teams": [
            {
                "name": "Red",
                "race_points": 28,
                "race_winner": True,
                "players": [
                    {"place": p, "name": f"R{p}"} for p in (1, 3, 5, 7, 9)
                ],
            },
            {
                "name": "Blue",
                "race_points": 22,
                "race_winner": False,
                "players": [
                    {"place": p, "name": f"B{p}"} for p in (2, 4, 6, 8)
                ],
            },
        ],
    }
    _, teams, _ = race_fields_from_gemini(gem)

    points_by_name = {t.name: t.points for t in teams}
    assert points_by_name == {"Red": 28, "Blue": 22}


def test_race_fields_no_teams_12p_leaves_single_team_with_derived_points():
    """In FFA the single 'team' contains every player — derived points
    is meaningful (= total race points = 82)."""
    gem = {
        "mode": "no_teams",
        "teams": [
            {
                "name": None,
                "race_points": 0,
                "race_winner": None,
                "players": [
                    {"place": p, "name": f"P{p}"} for p in range(1, 13)
                ],
            },
        ],
    }
    _, teams, placements = race_fields_from_gemini(gem)

    assert len(teams) == 1
    assert teams[0].points == 82
    assert len(placements) == 12


def test_race_fields_placements_sorted_by_place():
    """Sanity: placements still come out sorted regardless of override."""
    gem = _gemini_response_12p_two_teams(red_points=999, blue_points=42)
    _, _, placements = race_fields_from_gemini(gem)
    places = [p.place for p in placements]
    assert places == sorted(places)


# ---------------------------------------------------------------------------
# race_team_scores — runtime derivation for legacy on-disk records
# ---------------------------------------------------------------------------


def _settings(teams: str = "Two Teams") -> MatchSettingsRecord:
    return MatchSettingsRecord(
        cc_class="150cc",
        teams=teams,
        items="Normal",
        com_difficulty="No COM",
        race_count=12,
        intermission="10 seconds",
    )


def _two_team_race_12p(*, red_points: int, blue_points: int) -> RaceRecord:
    """Same partition as `_gemini_response_12p_two_teams` but as a
    persisted RaceRecord (e.g. loaded from match.json)."""
    return RaceRecord(
        race_number=1,
        track_name="Mario Bros. Circuit",
        players=[],
        mode="two_teams",
        placements=[
            PlayerPlacement(place=p, name=f"R{p}" if p % 2 else f"B{p}")
            for p in range(1, 13)
        ],
        teams=[
            TeamGroup(
                name="Red",
                points=red_points,
                winner=True,
                players=[
                    PlayerPlacement(place=p, name=f"R{p}")
                    for p in (1, 3, 5, 7, 9, 11)
                ],
            ),
            TeamGroup(
                name="Blue",
                points=blue_points,
                winner=False,
                players=[
                    PlayerPlacement(place=p, name=f"B{p}")
                    for p in (2, 4, 6, 8, 10, 12)
                ],
            ),
        ],
    )


def test_race_team_scores_derives_for_12p_even_when_team_points_set():
    """Even if an old match.json has bad Gemini-reported team.points,
    race_team_scores must re-derive them for 12-player races."""
    race = _two_team_race_12p(red_points=999, blue_points=999)
    scores = race_team_scores(race, _settings())

    assert dict(scores) == {"Red": 45, "Blue": 37}


def test_race_team_scores_trusts_team_points_for_unusual_count():
    """A 9-player race has no fixed table, so we trust team.points."""
    race = RaceRecord(
        race_number=1,
        track_name="Mario Bros. Circuit",
        players=[],
        mode="two_teams",
        placements=[
            PlayerPlacement(place=p, name=f"P{p}") for p in range(1, 10)
        ],
        teams=[
            TeamGroup(
                name="Red",
                points=28,
                winner=True,
                players=[
                    PlayerPlacement(place=p, name=f"R{p}")
                    for p in (1, 3, 5, 7, 9)
                ],
            ),
            TeamGroup(
                name="Blue",
                points=22,
                winner=False,
                players=[
                    PlayerPlacement(place=p, name=f"B{p}")
                    for p in (2, 4, 6, 8)
                ],
            ),
        ],
    )
    scores = race_team_scores(race, _settings())
    assert dict(scores) == {"Red": 28, "Blue": 22}


def test_race_team_scores_returns_none_for_ffa_match():
    """FFA matches don't have team_count, so race_team_scores bails."""
    race = _two_team_race_12p(red_points=45, blue_points=37)
    assert race_team_scores(race, _settings(teams="No Teams")) is None


def test_race_player_count_prefers_team_partition():
    """Falls back to placements when teams aren't populated."""
    race_with_teams = _two_team_race_12p(red_points=1, blue_points=1)
    assert race_player_count(race_with_teams) == 12

    race_flat = RaceRecord(
        race_number=1,
        track_name="X",
        players=[],
        placements=[PlayerPlacement(place=p, name=f"P{p}") for p in range(1, 13)],
    )
    assert race_player_count(race_flat) == 12
