"""Per-race team-score calculation for Mario Kart team modes.

Hoisted out of ``ui/match_history.py`` so non-UI code (notably the Discord
webhook notifier in the state machine) can share the exact same winner
logic that the history view already displays.
"""

from __future__ import annotations

from mktracker.match_record import MatchSettingsRecord, RaceRecord

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
