"""Per-race team-score calculation for Mario Kart team modes.

Hoisted out of ``ui/match_history.py`` so non-UI code (notably the Discord
webhook notifier in the state machine) can share the exact same winner
logic that the history view already displays.
"""

from __future__ import annotations

from mktracker.match_record import (
    FinalStandings, MatchSettingsRecord, PlayerPlacement, RaceRecord, TeamGroup,
)

# Mario Kart scoring tables: points awarded for each finishing position
# in a 12-player race (sum=82) and a 24-player race (sum=144).
MK_POINTS_12P: tuple[int, ...] = (15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
MK_POINTS_24P: tuple[int, ...] = (
    15, 12, 10, 9, 9,
    8, 8, 7, 7, 6, 6, 6, 5, 5, 5,
    4, 4, 4, 3, 3, 3, 2, 2, 1,
)

# Player counts whose race-point totals are fully prescribed by the
# scoring tables above.  For these we always derive points locally from
# Gemini's placements rather than trusting Gemini's own point arithmetic.
DERIVED_PLAYER_COUNTS: tuple[int, ...] = (12, 24)

_POINTS_TABLES: dict[int, tuple[int, ...]] = {
    12: MK_POINTS_12P,
    24: MK_POINTS_24P,
}

TEAM_COUNTS: dict[str, int] = {
    "Two Teams": 2,
    "Three Teams": 3,
    "Four Teams": 4,
}

TEAM_COLOR_ORDER: dict[str, tuple[str, ...]] = {
    "Two Teams": ("red", "blue"),
    "Three Teams": ("red", "blue", "yellow"),
    "Four Teams": ("red", "blue", "yellow", "green"),
}


def _team_label(names: list[str], color: str) -> tuple[str, str | None]:
    """Use a shared short clan tag only when most players support it."""
    candidates = {name.split(" ", 1)[0] for name in names if " " in name}
    matches = [(sum(name.startswith(tag) for name in names), tag)
               for tag in candidates if 1 <= len(tag) <= 8]
    support, tag = max(matches, default=(0, ""))
    tag = tag if support >= 3 and support / len(names) >= .8 else None
    return tag or color.title(), tag


def local_final_standings(
    rows: list[tuple[int, int, str, int]], teams_setting: str,
    player_count: int,
) -> FinalStandings | None:
    """Keep the model's final-card columns as team groups.

    Each column is one team; scores establish global place across columns.
    Incomplete cards are left for the existing fallback instead of inventing
    missing players or team totals. Equal scores receive the same competition
    rank, as on the no-team game card.
    """
    if teams_setting == "No Teams" and 1 <= player_count <= 24:
        columns = 1 if player_count <= 12 else 2
        column_sizes = ([player_count] if columns == 1 else
                        [(player_count + 1) // 2, player_count // 2])
        by_cell = {(column, row): (name, score)
                   for column, row, name, score in rows}
        if (len(rows) != player_count or len(by_cell) != player_count or
                any((column, row) not in by_cell
                    for column, size in enumerate(column_sizes)
                    for row in range(size))):
            return None
        ordered = [by_cell[column, row]
                   for column, size in enumerate(column_sizes)
                   for row in range(size)]
        if (len({name for name, _ in ordered}) != player_count or
                any(ordered[index][1] < ordered[index + 1][1]
                    for index in range(player_count - 1))):
            return None
        ranks = {}
        for index, (_, score) in enumerate(ordered, 1):
            ranks.setdefault(score, index)
        return FinalStandings(
            mode="no_teams",
            players=[PlayerPlacement(place=ranks[score], name=name, score=score)
                     for name, score in ordered],
            teams=None,
        )

    colors = TEAM_COLOR_ORDER.get(teams_setting)
    if (colors is None or len(rows) != player_count or
            player_count not in (12, 24)):
        return None
    columns_per_team = 2 if teams_setting == "Two Teams" and player_count == 24 else 1
    column_count = len(colors) * columns_per_team
    if player_count % column_count:
        return None
    per_column = player_count // column_count
    by_cell = {(column, row): (name, score)
               for column, row, name, score in rows}
    if len(by_cell) != player_count or any(
        (column, row) not in by_cell
        for column in range(column_count) for row in range(per_column)
    ):
        return None
    ordered = sorted(rows, key=lambda item: (-item[3], item[0], item[1]))
    ranks = {}
    for index, (_, _, _, score) in enumerate(ordered, 1):
        ranks.setdefault(score, index)
    placements = {
        (column, row): PlayerPlacement(place=ranks[score], name=name, score=score)
        for column, row, name, score in ordered
    }
    team_cells = [[(column, row)
                   for column in range(team_index * columns_per_team,
                                       (team_index + 1) * columns_per_team)
                   for row in range(per_column)]
                  for team_index in range(len(colors))]
    totals = [sum(by_cell[cell][1] for cell in cells) for cells in team_cells]
    high = max(totals)
    labels = [_team_label([placements[cell].name for cell in cells], color)
              for cells, color in zip(team_cells, colors)]
    groups = []
    for team_index, (cells, color) in enumerate(zip(team_cells, colors)):
        players = [placements[cell] for cell in cells]
        label, tag = labels[team_index]
        if tag and sum(other_tag == tag for _, other_tag in labels) > 1:
            label, tag = color.title(), None
        groups.append(TeamGroup(
            name=label, tag=tag, points=totals[team_index],
            winner=totals[team_index] == high, players=players,
        ))
    return FinalStandings(
        mode=teams_setting.lower().replace(" ", "_"),
        players=sorted(placements.values(), key=lambda player: player.place),
        teams=groups,
    )


def local_race_teams(
    placements: list[tuple[int, str]], colors_by_place: dict[int, str],
    teams_setting: str,
) -> dict | None:
    """Build structured teams from model names and the game-coloured bars.

    Return ``None`` when a placement lacks an unambiguous configured colour.
    Race points come from the fixed game scoring table, so no score OCR or
    model arithmetic is needed.
    """
    colors = TEAM_COLOR_ORDER.get(teams_setting)
    if colors is None or len(placements) not in DERIVED_PLAYER_COUNTS:
        return None
    if any(colors_by_place.get(place) not in colors for place, _ in placements):
        return None
    groups = {color: [] for color in colors}
    for place, name in placements:
        groups[colors_by_place[place]].append((place, name))
    if any(not group for group in groups.values()):
        return None
    player_count = len(placements)
    points = {
        color: sum(points_for_place(place, player_count) for place, _ in group)
        for color, group in groups.items()
    }
    highest = max(points.values())
    labels = {color: _team_label([name for _, name in groups[color]], color)
              for color in colors}
    structured = []
    for color in colors:
        group = groups[color]
        label, tag = labels[color]
        if tag and sum(other_tag == tag for _, other_tag in labels.values()) > 1:
            label, tag = color.title(), None
        structured.append({
            "name": label, "tag": tag,
            "race_points": points[color], "race_winner": points[color] == highest,
            "players": [{"place": place, "name": name} for place, name in group],
        })
    return {"mode": teams_setting.lower().replace(" ", "_"), "teams": structured}


def points_for_place(place: int, player_count: int = 12) -> int:
    """Return MK race points for *place* under a *player_count*-player scoring table.

    Falls back to the 12-player table for any *player_count* not in
    :data:`DERIVED_PLAYER_COUNTS` so callers that don't know the count
    (or are operating on partial data) keep their previous behaviour.
    """
    table = _POINTS_TABLES.get(player_count, MK_POINTS_12P)
    idx = place - 1
    if 0 <= idx < len(table):
        return table[idx]
    return 0


def race_player_count(race: RaceRecord) -> int:
    """Total visible players in *race*.

    Prefers the structured-team partition (Gemini's authoritative split)
    and falls back to the flat placement list when team data is absent.
    """
    if race.teams:
        return sum(len(t.players) for t in race.teams)
    return len(race.placements)


def assign_tag(name: str, known_tags: set[str]) -> str | None:
    """Return the team tag from *known_tags* that prefixes *name*, or None.

    Prefers an exact space-delimited match, then the longest prefix match
    (handles cases like ``EMA☆Flav`` — no space but still tagged ``EMA``).
    """
    space = name.find(" ")
    if space > 0 and name[:space] in known_tags:
        return name[:space]
    matches = [t for t in known_tags if name.startswith(t)]
    if matches:
        return max(matches, key=len)
    return None


def race_team_scores(
    race: RaceRecord, settings: MatchSettingsRecord,
) -> list[tuple[str, int]] | None:
    """Return ``[(team_name, points), ...]`` for *race*, or ``None`` if we
    can't determine team scores.

    Two code paths:

    1. Gemini-labelled teams (``race.teams`` has ≥2 groups): when the
       race has a known scoring table (12 or 24 players) we always
       *derive* the per-team total from the placements — Gemini reads
       placements reliably but its point arithmetic on the +score column
       is unreliable. For any other player count we trust the value
       Gemini reported in ``team.points`` when present, falling back to a
       derived sum.
    2. Fallback — infer team tags from space-delimited prefixes in the
       placement names (e.g. ``EMA``, ``>€>``) and bucket per tag. Bails
       out if a scoring placement can't be assigned to any detected tag.
    """
    team_count = TEAM_COUNTS.get(settings.teams)
    if team_count is None:
        return None

    player_count = race_player_count(race)
    derive_locally = player_count in DERIVED_PLAYER_COUNTS

    if race.teams and len(race.teams) >= 2:
        scores: list[tuple[str, int]] = []
        for i, team in enumerate(race.teams):
            if derive_locally:
                pts = sum(
                    points_for_place(p.place, player_count) for p in team.players
                )
            elif team.points is not None:
                pts = team.points
            else:
                pts = sum(
                    points_for_place(p.place, player_count) for p in team.players
                )
            scores.append((team.name or f"Team {i + 1}", pts))
        return scores

    if not race.placements:
        return None

    spaced_tags: set[str] = set()
    for p in race.placements:
        space = p.name.find(" ")
        if space > 0:
            spaced_tags.add(p.name[:space])
    if len(spaced_tags) < team_count:
        return None

    buckets: dict[str, int] = {tag: 0 for tag in spaced_tags}
    for p in race.placements:
        tag = assign_tag(p.name, spaced_tags)
        points = points_for_place(p.place, player_count)
        if tag is None:
            if points == 0:
                continue
            return None
        buckets[tag] += points

    nonzero = {t: pts for t, pts in buckets.items() if pts > 0}
    if len(nonzero) < 2:
        return None
    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return top[:team_count]
