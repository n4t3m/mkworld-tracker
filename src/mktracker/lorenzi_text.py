"""Lorenzi-style table-editor text format.

Round-trips between :class:`FinalStandings` and the line-oriented text
format used by gb2.hlorenzi.com/table, so users can edit the final
results by hand in a textbox.

Format:

    <tag>
    <player name> <score expression>
    <player name> <score expression>

    <tag>
    ...

Score expressions support addition/subtraction of integers, e.g.
``70+20+8`` → 98 or ``80-5`` → 75.  For FFA matches (single team / no
team tag), the tag line is omitted and the block is just player rows.
"""

from __future__ import annotations

import re

from mktracker.match_record import FinalStandings, PlayerPlacement, TeamGroup

_MODE_FROM_COUNT = {
    1: "no_teams",
    2: "two_teams",
    3: "three_teams",
    4: "four_teams",
}

_SCORE_RE = re.compile(r"^[\d+\-\s]+$")
_LINE_RE = re.compile(r"^(.+?)\s+([\d+\-\s]+)$")
_TOKEN_RE = re.compile(r"[+-]?\s*\d+")


def standings_to_text(standings: FinalStandings | None) -> str:
    """Render final standings as Lorenzi-style editable text."""
    if standings is None:
        return ""
    teams = standings.teams
    if not teams:
        lines = [
            f"{p.name} {p.score if p.score is not None else 0}"
            for p in sorted(standings.players, key=lambda x: x.place)
        ]
        return "\n".join(lines)

    blocks: list[str] = []
    for team in teams:
        lines: list[str] = []
        tag = team.tag or team.name
        if tag:
            lines.append(tag)
        for p in sorted(team.players, key=lambda x: x.place):
            lines.append(f"{p.name} {p.score if p.score is not None else 0}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_score(expr: str) -> int | None:
    expr = expr.strip()
    if not expr or not _SCORE_RE.match(expr):
        return None
    tokens = _TOKEN_RE.findall(expr)
    if not tokens:
        return None
    return sum(int(t.replace(" ", "")) for t in tokens)


def _parse_player_line(line: str) -> tuple[str, int] | None:
    line = line.strip()
    if not line:
        return None
    m = _LINE_RE.match(line)
    if not m:
        return None
    name = m.group(1).strip()
    score = _parse_score(m.group(2))
    if score is None or not name:
        return None
    return name, score


def text_to_standings(text: str) -> FinalStandings:
    """Parse Lorenzi-style text back into :class:`FinalStandings`."""
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    teams: list[TeamGroup] = []

    for raw in raw_blocks:
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            continue

        if _parse_player_line(lines[0]) is None:
            tag: str | None = lines[0].strip()
            player_lines = lines[1:]
        else:
            tag = None
            player_lines = lines

        parsed_players: list[tuple[str, int]] = []
        for ln in player_lines:
            parsed = _parse_player_line(ln)
            if parsed is not None:
                parsed_players.append(parsed)

        if not parsed_players and tag is None:
            continue

        team_players = [
            PlayerPlacement(place=0, name=n, score=s) for n, s in parsed_players
        ]
        points = sum(s for _, s in parsed_players)
        teams.append(TeamGroup(
            name=tag,
            tag=tag,
            points=points,
            winner=False,
            players=team_players,
        ))

    all_players: list[PlayerPlacement] = []
    for team in teams:
        all_players.extend(team.players)
    ranked = sorted(
        enumerate(all_players),
        key=lambda kv: (-(kv[1].score or 0), kv[0]),
    )
    for rank, (_, p) in enumerate(ranked, 1):
        p.place = rank

    if teams:
        max_pts = max((t.points or 0) for t in teams)
        for t in teams:
            t.winner = (t.points or 0) == max_pts

    has_tags = any(t.tag for t in teams)
    if has_tags:
        mode = _MODE_FROM_COUNT.get(len(teams))
        teams_field: list[TeamGroup] | None = teams
    else:
        mode = "no_teams"
        teams_field = None

    flat = sorted(all_players, key=lambda p: p.place)
    return FinalStandings(mode=mode, players=flat, teams=teams_field)
