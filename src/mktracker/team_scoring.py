"""Per-race team-score calculation for Mario Kart team modes.

Hoisted out of ``ui/match_history.py`` so non-UI code (notably the Discord
webhook notifier in the state machine) can share the exact same winner
logic that the history view already displays.
"""

from __future__ import annotations

from mktracker.match_record import MatchSettingsRecord, RaceRecord

# Mario Kart 12-player scoring: points awarded for each finishing position.
MK_POINTS_12P: tuple[int, ...] = (15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

TEAM_COUNTS: dict[str, int] = {
    "Two Teams": 2,
    "Three Teams": 3,
    "Four Teams": 4,
}


def points_for_place(place: int) -> int:
    idx = place - 1
    if 0 <= idx < len(MK_POINTS_12P):
        return MK_POINTS_12P[idx]
    return 0


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

    1. Gemini-labelled teams (``race.teams`` has ≥2 groups): sum the MK
       scoring table per group, using ``team.points`` when it's already
       provided.
    2. Fallback — infer team tags from space-delimited prefixes in the
       placement names (e.g. ``EMA``, ``>€>``) and bucket per tag. Bails
       out if a scoring placement can't be assigned to any detected tag.
    """
    team_count = TEAM_COUNTS.get(settings.teams)
    if team_count is None:
        return None

    if race.teams and len(race.teams) >= 2:
        scores: list[tuple[str, int]] = []
        for i, team in enumerate(race.teams):
            if team.points is not None:
                pts = team.points
            else:
                pts = sum(points_for_place(p.place) for p in team.players)
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
        points = points_for_place(p.place)
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
