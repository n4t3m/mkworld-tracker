"""Match history UI.

Shows a list of previously played matches (loaded from ``debug_frames/``) on
the left and a scrollable per-race timeline on the right.  The detail widget
accepts any :class:`MatchRecord`, so it is also used to display the
currently-running match — the state machine writes ``match.json`` after every
update, so the on-disk record is always live.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mktracker.detection.tracks import TRACK_ICONS_DIR, TRACK_IMAGES
from mktracker.match_record import (
    DEFAULT_DEBUG_DIR,
    MATCH_FILE,
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    RaceRecord,
    TeamGroup,
    list_matches,
)

if TYPE_CHECKING:
    from mktracker.state_machine import GameStateMachine

logger = logging.getLogger(__name__)

_ICON_SIZE = 96
_ICON_CACHE: dict[str, QPixmap] = {}

# Human-readable description of what the state machine is currently doing.
# Used to render the live status banner in the detail view.  Keyed by
# ``GameState.name`` (string) so this module doesn't import the enum.
_LIVE_STATUS_TEXT: dict[str, str] = {
    "MATCH_STARTED": "Match starting…",
    "WAITING_FOR_TRACK_PICK": "Waiting for track selection",
    "READING_PLAYERS_IN_RACE": "Reading player names",
    "WAITING_FOR_RACE_END": "Race in progress — waiting for FINISH",
    "DETECTING_RACE_RANK": "Detecting your placement",
    "READING_RACE_RESULTS": "Reading race results",
    "FINALIZING_MATCH": "Reading final standings",
}


def _load_track_icon(track_name: str | None) -> QPixmap | None:
    """Return a cached ``QPixmap`` for *track_name*, or ``None`` if unknown."""
    if not track_name:
        return None
    cached = _ICON_CACHE.get(track_name)
    if cached is not None:
        return cached
    filename = TRACK_IMAGES.get(track_name)
    if not filename:
        return None
    path = TRACK_ICONS_DIR / filename
    if not path.exists():
        return None
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None
    _ICON_CACHE[track_name] = pixmap
    return pixmap


# Mario Kart 12-player scoring: points awarded for each finishing position.
_MK_POINTS_12P: tuple[int, ...] = (15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

_TEAM_COUNTS: dict[str, int] = {
    "Two Teams": 2,
    "Three Teams": 3,
    "Four Teams": 4,
}


def _points_for_place(place: int) -> int:
    idx = place - 1
    if 0 <= idx < len(_MK_POINTS_12P):
        return _MK_POINTS_12P[idx]
    return 0


def _assign_tag(name: str, known_tags: set[str]) -> str | None:
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


def _race_team_scores(
    race: RaceRecord, settings: MatchSettingsRecord,
) -> list[tuple[str, int]] | None:
    """Return ``[(team_name, points), ...]`` for *race*, or ``None`` if we
    can't determine team scores.

    Two code paths:

    1. Gemini-labelled teams (``race.teams`` has ≥2 groups): sum the MK
       scoring table per group, using ``team.points`` when it's already
       provided.
    2. Fallback — infer team tags from space-delimited prefixes in the
       placement names (e.g. ``EMA``, ``>€>``) and bucket per tag.  Bails
       out if a scoring placement can't be assigned to any detected tag.
    """
    team_count = _TEAM_COUNTS.get(settings.teams)
    if team_count is None:
        return None

    if race.teams and len(race.teams) >= 2:
        scores: list[tuple[str, int]] = []
        for i, team in enumerate(race.teams):
            if team.points is not None:
                pts = team.points
            else:
                pts = sum(_points_for_place(p.place) for p in team.players)
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
        tag = _assign_tag(p.name, spaced_tags)
        points = _points_for_place(p.place)
        if tag is None:
            # Unassignable but off-table — safe to drop (no score change).
            if points == 0:
                continue
            return None
        buckets[tag] += points

    nonzero = {t: pts for t, pts in buckets.items() if pts > 0}
    if len(nonzero) < 2:
        return None
    top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    return top[:team_count]


def _format_timestamp(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _summary_line(record: MatchRecord, *, live: bool = False) -> str:
    s = record.settings
    if live:
        progress = f"{len(record.races)}/{s.race_count} races"
    else:
        race_count = len(record.races) or s.race_count
        progress = f"{race_count} races"
    parts = [s.cc_class, progress, s.teams]
    if not live and record.final_standings is None and record.completed_at is None:
        parts.append("partial")
    return " · ".join(parts)


class _RaceCard(QFrame):
    """One race in the timeline: icon + track name + placements."""

    clicked = Signal(int)  # race_number

    def __init__(
        self,
        race: RaceRecord,
        settings: MatchSettingsRecord,
        *,
        live: bool = False,
    ) -> None:
        super().__init__()
        self._live = live
        self._settings = settings
        self._race_number = race.race_number
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(
            "_RaceCard { background-color: #1a1a1a; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
            " _RaceCard:hover { background-color: #22262b; border: 1px solid #3a4a5a; }"
            " QLabel { color: #ddd; }"
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        # --- Track icon ---
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = _load_track_icon(race.track_name)
        if pixmap is not None:
            scaled = pixmap.scaledToHeight(
                _ICON_SIZE, Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(scaled)
            icon_label.setFixedSize(scaled.size())
        else:
            icon_label.setFixedSize(_ICON_SIZE, _ICON_SIZE)
            icon_label.setStyleSheet(
                "QLabel { background-color: #0e0e0e; border-radius: 4px; color: #555; }"
            )
            icon_label.setText("?")
        outer.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignTop)

        # --- Middle column: title + placements ---
        mid = QVBoxLayout()
        mid.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        number = QLabel(f"Race {race.race_number}")
        number.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 14px; color: #8ab; }"
        )
        title_row.addWidget(number)

        track = QLabel(race.track_name or "Unknown track")
        track.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #fff; }")
        title_row.addWidget(track)
        title_row.addStretch()

        if race.user_rank is not None:
            badge = QLabel(f"Rank {race.user_rank}")
            badge.setStyleSheet(
                "QLabel { background-color: #2d4; color: #042; font-weight: bold;"
                " padding: 2px 8px; border-radius: 10px; }"
            )
            title_row.addWidget(badge)
        elif self._live:
            badge = QLabel("Rank …")
            badge.setStyleSheet(
                "QLabel { background-color: #333; color: #aaa; font-style: italic;"
                " padding: 2px 8px; border-radius: 10px; }"
            )
            badge.setToolTip("Awaiting Gemini rank result")
            title_row.addWidget(badge)

        mid.addLayout(title_row)

        team_score_widget = self._build_team_score(race)
        if team_score_widget is not None:
            mid.addWidget(team_score_widget)

        placements_widget = self._build_placements(race)
        if placements_widget is not None:
            mid.addWidget(placements_widget)

        outer.addLayout(mid, stretch=1)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._race_number)
        super().mousePressEvent(event)

    def _build_team_score(self, race: RaceRecord) -> QWidget | None:
        scores = _race_team_scores(race, self._settings)
        if scores is None or len(scores) < 2:
            return None
        ranked = sorted(scores, key=lambda kv: kv[1], reverse=True)
        parts = [f"{name} {pts}" for name, pts in ranked]
        summary = "Team score: " + "  ·  ".join(parts)
        delta = ranked[0][1] - ranked[1][1]
        if delta > 0:
            summary += f"   (+{delta} {ranked[0][0]})"
        else:
            summary += "   (tied)"
        label = QLabel(summary)
        label.setStyleSheet(
            "QLabel { color: #fc8; font-weight: bold; font-size: 12px; }"
        )
        return label

    def _build_placements(self, race: RaceRecord) -> QWidget | None:
        if race.teams and len(race.teams) >= 2:
            return self._build_team_placements(race.teams)
        if race.placements:
            return self._build_solo_placements(race)
        if self._live:
            note = QLabel("Awaiting placements…")
            note.setStyleSheet("QLabel { color: #c93; font-style: italic; }")
        else:
            note = QLabel("No placements recorded.")
            note.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        return note

    def _build_solo_placements(self, race: RaceRecord) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        for p in sorted(race.placements, key=lambda x: x.place):
            row = QLabel(f"{p.place:>2}.  {p.name}")
            row.setStyleSheet("QLabel { color: #ccc; font-family: Consolas, monospace; }")
            layout.addWidget(row)
        return container

    def _build_team_placements(self, teams: list[TeamGroup]) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for team in teams:
            header_parts = []
            if team.name:
                header_parts.append(team.name)
            if team.points is not None:
                header_parts.append(f"{team.points} pts")
            header_text = " — ".join(header_parts) or "Team"
            header = QLabel(header_text)
            style = "QLabel { font-weight: bold; color: #fc8; }"
            if team.winner:
                style = "QLabel { font-weight: bold; color: #fd4; }"
            header.setStyleSheet(style)
            layout.addWidget(header)
            for p in sorted(team.players, key=lambda x: x.place):
                row = QLabel(f"    {p.place:>2}.  {p.name}")
                row.setStyleSheet("QLabel { color: #ccc; font-family: Consolas, monospace; }")
                layout.addWidget(row)
        return container


class _LiveStatusBanner(QFrame):
    """Prominent banner shown above the live match telling the user what
    the state machine is currently doing."""

    def __init__(self, status_text: str, race_progress: str | None) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_LiveStatusBanner { background-color: #3a1414;"
            " border: 2px solid #c33; border-radius: 6px; }"
            " QLabel { color: #fee; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        dot = QLabel("●")
        dot.setStyleSheet(
            "QLabel { color: #f55; font-size: 20px; font-weight: bold; }"
        )
        layout.addWidget(dot)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        title = QLabel("LIVE MATCH IN PROGRESS")
        title.setStyleSheet(
            "QLabel { color: #fff; font-size: 13px; font-weight: bold;"
            " letter-spacing: 1px; }"
        )
        text_col.addWidget(title)

        sub_parts = [status_text]
        if race_progress:
            sub_parts.insert(0, race_progress)
        sub = QLabel("  ·  ".join(sub_parts))
        sub.setStyleSheet("QLabel { color: #fcc; font-size: 13px; }")
        text_col.addWidget(sub)
        layout.addLayout(text_col, stretch=1)


class _PendingRaceCard(QFrame):
    """Placeholder card for a race that hasn't been played yet."""

    def __init__(self, race_number: int) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_PendingRaceCard { background-color: #15171a;"
            " border: 1px dashed #2d2f33; border-radius: 6px; }"
            " QLabel { color: #777; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        icon = QLabel("?")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(_ICON_SIZE, _ICON_SIZE)
        icon.setStyleSheet(
            "QLabel { background-color: #0e0e0e; border-radius: 4px;"
            " color: #444; font-size: 32px; }"
        )
        layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        title = QLabel(f"Race {race_number}")
        title.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 14px; color: #667; }"
        )
        text_col.addWidget(title)
        sub = QLabel("Pending — race not yet played")
        sub.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        text_col.addWidget(sub)
        text_col.addStretch()
        layout.addLayout(text_col, stretch=1)


class _PendingFinalStandingsCard(QFrame):
    """Placeholder shown while the live match has no final standings yet."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_PendingFinalStandingsCard { background-color: #1a1a1f;"
            " border: 1px dashed #3a3a4a; border-radius: 6px; }"
            " QLabel { color: #aab; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        title = QLabel("Final Standings")
        title.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; color: #889; }"
        )
        layout.addWidget(title)
        sub = QLabel("Awaiting final results — match still in progress.")
        sub.setStyleSheet("QLabel { color: #778; font-style: italic; }")
        layout.addWidget(sub)


class _FinalStandingsCard(QFrame):
    """Summary of final standings shown at the bottom of the timeline."""

    def __init__(self, standings: FinalStandings) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_FinalStandingsCard { background-color: #1b241b; border: 1px solid #3a5a3a;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        title = QLabel("Final Standings")
        title.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; color: #fff; }")
        layout.addWidget(title)

        if standings.teams:
            for team in standings.teams:
                header_parts = []
                if team.name:
                    header_parts.append(team.name)
                if team.points is not None:
                    header_parts.append(f"{team.points} pts")
                header_text = " — ".join(header_parts) or "Team"
                header = QLabel(header_text)
                style = "QLabel { font-weight: bold; color: #fc8; }"
                if team.winner:
                    style = "QLabel { font-weight: bold; color: #fd4; }"
                header.setStyleSheet(style)
                layout.addWidget(header)
                for p in sorted(team.players, key=lambda x: x.place):
                    score = f"  ({p.score} pts)" if p.score is not None else ""
                    row = QLabel(f"    {p.place:>2}.  {p.name}{score}")
                    row.setStyleSheet(
                        "QLabel { color: #ccc; font-family: Consolas, monospace; }"
                    )
                    layout.addWidget(row)
        else:
            for p in sorted(standings.players, key=lambda x: x.place):
                score = f"  ({p.score} pts)" if p.score is not None else ""
                row = QLabel(f"{p.place:>2}.  {p.name}{score}")
                row.setStyleSheet(
                    "QLabel { color: #ccc; font-family: Consolas, monospace; }"
                )
                layout.addWidget(row)


class _MatchTimelinePane(QScrollArea):
    """Scrollable timeline of races for a single match.

    Emits :attr:`raceSelected` when the user clicks a race card.
    """

    raceSelected = Signal(int)  # race_number

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)
        self._layout.addStretch()
        self.setWidget(self._inner)
        self._show_empty()

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _show_empty(self) -> None:
        self._clear()
        msg = QLabel("Select a match to view its races.")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 40px; }")
        self._layout.addWidget(msg)
        self._layout.addStretch()

    def clear(self) -> None:
        self._show_empty()

    def set_record(
        self,
        record: MatchRecord,
        *,
        live: bool = False,
        live_status: str | None = None,
    ) -> None:
        self._clear()

        if live:
            races_played = len(record.races)
            total = max(record.settings.race_count, races_played)
            progress = f"Race {min(races_played + 1, total)} of {total}"
            status_text = live_status or "Live"
            self._layout.addWidget(_LiveStatusBanner(status_text, progress))

        header = self._build_header(record, live=live)
        self._layout.addWidget(header)

        races = sorted(record.races, key=lambda r: r.race_number)
        if not races and not live:
            note = QLabel("No races recorded yet.")
            note.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 10px; }")
            self._layout.addWidget(note)
        else:
            for race in races:
                card = _RaceCard(race, record.settings, live=live)
                card.clicked.connect(self.raceSelected.emit)
                self._layout.addWidget(card)

        if live:
            played = max((r.race_number for r in races), default=0)
            total = max(record.settings.race_count, played)
            for n in range(played + 1, total + 1):
                self._layout.addWidget(_PendingRaceCard(n))

        if record.final_standings is not None:
            self._layout.addWidget(_FinalStandingsCard(record.final_standings))
        elif live:
            self._layout.addWidget(_PendingFinalStandingsCard())

        self._layout.addStretch()

    def _build_header(self, record: MatchRecord, *, live: bool = False) -> QWidget:
        frame = QFrame()
        if live:
            frame.setStyleSheet(
                "QFrame { background-color: #1f1418; border: 2px solid #c33;"
                " border-radius: 6px; }"
                " QLabel { color: #ddd; }"
            )
        else:
            frame.setStyleSheet(
                "QFrame { background-color: #14181f; border: 1px solid #232a33;"
                " border-radius: 6px; }"
                " QLabel { color: #ddd; }"
            )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        if live:
            badge = QLabel("● LIVE")
            badge.setStyleSheet(
                "QLabel { background-color: #c33; color: #fff; font-weight: bold;"
                " padding: 4px 12px; border-radius: 12px; font-size: 14px;"
                " letter-spacing: 1px; }"
            )
            badge.setToolTip("This match is currently in progress")
            title_row.addWidget(badge)
        title = QLabel(f"Match {record.match_id}")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #fff; }")
        title_row.addWidget(title)
        title_row.addStretch()
        layout.addLayout(title_row)

        started = _format_timestamp(record.started_at)
        if live:
            when = QLabel(f"Started: {started}    Status: in progress")
            when.setStyleSheet("QLabel { color: #fcc; }")
        else:
            completed = _format_timestamp(record.completed_at)
            when = QLabel(f"Started: {started}    Completed: {completed}")
            when.setStyleSheet("QLabel { color: #889; }")
        layout.addWidget(when)

        s = record.settings
        settings_text = (
            f"{s.cc_class} · {s.teams} · Items: {s.items} · COM: {s.com_difficulty}"
            f" · {s.race_count} races · {s.intermission}"
        )
        settings_label = QLabel(settings_text)
        settings_label.setWordWrap(True)
        settings_label.setStyleSheet("QLabel { color: #aab; }")
        layout.addWidget(settings_label)

        return frame


class _ImageCarousel(QWidget):
    """Shows one image at a time from a list, with prev/next navigation."""

    def __init__(self, paths: list[Path], *, max_width: int = 720) -> None:
        super().__init__()
        self._paths = paths
        self._max_width = max_width
        self._index = 0
        self._pixmaps: dict[int, QPixmap] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Nav row
        nav = QHBoxLayout()
        nav.setSpacing(8)
        self._prev_btn = QPushButton("◀  Prev")
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.clicked.connect(self._go_prev)
        nav.addWidget(self._prev_btn)

        self._counter = QLabel()
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._counter.setStyleSheet(
            "QLabel { color: #bbb; font-weight: bold; font-size: 12px; }"
        )
        nav.addWidget(self._counter, stretch=1)

        self._next_btn = QPushButton("Next  ▶")
        self._next_btn.setFixedWidth(90)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Image display
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background-color: #0a0a0a; border: 1px solid #2a2a2a;"
            " border-radius: 4px; }"
        )
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(self._image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._render()

    def _load_pixmap(self, index: int) -> QPixmap | None:
        if index in self._pixmaps:
            return self._pixmaps[index]
        path = self._paths[index]
        pm = QPixmap(str(path))
        if pm.isNull():
            return None
        self._pixmaps[index] = pm
        return pm

    def _render(self) -> None:
        n = len(self._paths)
        if n == 0:
            self._counter.setText("No images")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._image_label.setText("No placement frames saved for this race.")
            self._image_label.setStyleSheet(
                "QLabel { color: #666; font-style: italic;"
                " background-color: #0a0a0a; border: 1px solid #2a2a2a;"
                " border-radius: 4px; padding: 40px; }"
            )
            self._image_label.setFixedHeight(120)
            return
        self._prev_btn.setEnabled(n > 1)
        self._next_btn.setEnabled(n > 1)
        self._counter.setText(
            f"{self._paths[self._index].name}   —   Frame {self._index + 1} of {n}"
        )
        pm = self._load_pixmap(self._index)
        if pm is None:
            self._image_label.setText(f"Could not load {self._paths[self._index].name}")
            self._image_label.setFixedHeight(80)
            return
        scaled = pm.scaledToWidth(
            self._max_width, Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        # Lock the label to the pixmap's exact size so surrounding layout can't
        # squeeze it shorter and clip the top with the center-aligned pixmap.
        self._image_label.setFixedSize(scaled.size())

    def _go_prev(self) -> None:
        if not self._paths:
            return
        self._index = (self._index - 1) % len(self._paths)
        self._render()

    def _go_next(self) -> None:
        if not self._paths:
            return
        self._index = (self._index + 1) % len(self._paths)
        self._render()


def _fit_image_to_width(path: Path, max_width: int = 720) -> QPixmap | None:
    if not path.exists():
        return None
    pm = QPixmap(str(path))
    if pm.isNull():
        return None
    return pm.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)


class _RaceDetailView(QWidget):
    """Single-race detail screen with a sticky header and a scrollable body
    (Gemini summary + debug images)."""

    backRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Sticky header — lives outside the scroll area.
        self._header_host = QWidget()
        self._header_layout = QVBoxLayout(self._header_host)
        self._header_layout.setContentsMargins(12, 12, 12, 0)
        self._header_layout.setSpacing(0)
        outer.addWidget(self._header_host)

        # Scrollable body below it.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(12)
        self._layout.addStretch()
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll, stretch=1)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def set_race(
        self,
        race: RaceRecord,
        settings: MatchSettingsRecord,
        match_dir: Path,
    ) -> None:
        self._clear_layout(self._header_layout)
        self._clear_layout(self._layout)

        # Sticky header stays visible while scrolling.
        self._header_layout.addWidget(self._build_header(race))

        self._layout.addWidget(self._build_gemini_summary(race, settings))

        race_dir = match_dir / f"race_{race.race_number:02d}"
        self._layout.addWidget(self._build_single_image_section(
            "Track Selection", race_dir / "track.png",
        ))
        self._layout.addWidget(self._build_single_image_section(
            "Finish", race_dir / "finish.png",
        ))
        self._layout.addWidget(self._build_placements_section(race_dir))

        self._layout.addStretch()
        self._scroll.verticalScrollBar().setValue(0)

    # --- header ---

    def _build_header(self, race: RaceRecord) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #14181f; border: 1px solid #232a33;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        outer = QHBoxLayout(frame)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(12)

        back = QPushButton("◀  Back to match")
        back.setFixedHeight(32)
        back.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        back.setStyleSheet(
            "QPushButton { background-color: #2a3b4a; color: #fff; border: none;"
            " border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            " QPushButton:hover { background-color: #35506b; }"
        )
        back.clicked.connect(self.backRequested.emit)
        outer.addWidget(back)

        icon = _load_track_icon(race.track_name)
        if icon is not None:
            icon_label = QLabel()
            scaled = icon.scaledToHeight(
                48, Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(scaled)
            outer.addWidget(icon_label)

        title = QLabel(
            f"Race {race.race_number}  —  {race.track_name or 'Unknown track'}",
        )
        title.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #fff; }"
        )
        outer.addWidget(title, stretch=1)

        if race.user_rank is not None:
            badge = QLabel(f"Your rank: {race.user_rank}")
            badge.setStyleSheet(
                "QLabel { background-color: #2d4; color: #042; font-weight: bold;"
                " padding: 4px 12px; border-radius: 12px; font-size: 13px; }"
            )
            outer.addWidget(badge)

        top_btn = QPushButton("▲  Top")
        top_btn.setFixedHeight(32)
        top_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        top_btn.setToolTip("Scroll to top")
        top_btn.setStyleSheet(
            "QPushButton { background-color: #2a2a2a; color: #ddd; border: none;"
            " border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            " QPushButton:hover { background-color: #3a3a3a; color: #fff; }"
        )
        top_btn.clicked.connect(
            lambda: self._scroll.verticalScrollBar().setValue(0),
        )
        outer.addWidget(top_btn)

        return frame

    # --- Gemini summary section ---

    def _build_gemini_summary(
        self, race: RaceRecord, settings: MatchSettingsRecord,
    ) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #16201f; border: 1px solid #2a4a44;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        heading = QLabel("Race Results")
        heading.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; color: #8fd; }"
        )
        layout.addWidget(heading)

        # Mode line
        meta_parts: list[str] = []
        if race.mode:
            meta_parts.append(f"Mode: {race.mode}")
        meta_parts.append(f"Settings mode: {settings.teams}")
        if race.user_rank is not None:
            meta_parts.append(f"Your rank: {race.user_rank}")
        meta = QLabel("  ·  ".join(meta_parts))
        meta.setStyleSheet("QLabel { color: #9ab; font-size: 12px; }")
        layout.addWidget(meta)

        # Team score summary if applicable
        scores = _race_team_scores(race, settings)
        if scores is not None and len(scores) >= 2:
            ranked = sorted(scores, key=lambda kv: kv[1], reverse=True)
            winner_name, winner_pts = ranked[0]
            runner_pts = ranked[1][1]
            delta = winner_pts - runner_pts
            if delta > 0:
                winner_line = QLabel(
                    f"🏆 Winner: {winner_name}  ({winner_pts} pts,  +{delta})",
                )
            else:
                winner_line = QLabel(f"🤝 Tied at {winner_pts} pts")
            winner_line.setStyleSheet(
                "QLabel { color: #fd4; font-weight: bold; font-size: 14px; }"
            )
            layout.addWidget(winner_line)

            score_line = QLabel(
                "Team score:  "
                + "    ·    ".join(f"{name} {pts}" for name, pts in ranked),
            )
            score_line.setStyleSheet("QLabel { color: #fc8; font-size: 13px; }")
            layout.addWidget(score_line)

        # Placements
        if race.teams and len(race.teams) >= 2:
            for team in race.teams:
                header_parts: list[str] = []
                if team.name:
                    header_parts.append(team.name)
                if team.points is not None:
                    header_parts.append(f"{team.points} pts")
                header_text = " — ".join(header_parts) or "Team"
                th = QLabel(header_text)
                style = "QLabel { font-weight: bold; color: #fc8; margin-top: 4px; }"
                if team.winner:
                    style = (
                        "QLabel { font-weight: bold; color: #fd4; margin-top: 4px; }"
                    )
                th.setStyleSheet(style)
                layout.addWidget(th)
                for p in sorted(team.players, key=lambda x: x.place):
                    row = QLabel(f"    {p.place:>2}.  {p.name}")
                    row.setStyleSheet(
                        "QLabel { color: #ccc; font-family: Consolas, monospace; }"
                    )
                    layout.addWidget(row)
        elif race.placements:
            for p in sorted(race.placements, key=lambda x: x.place):
                row = QLabel(f"{p.place:>2}.  {p.name}")
                row.setStyleSheet(
                    "QLabel { color: #ccc; font-family: Consolas, monospace; }"
                )
                layout.addWidget(row)
        else:
            note = QLabel("No placements recorded for this race.")
            note.setStyleSheet("QLabel { color: #888; font-style: italic; }")
            layout.addWidget(note)

        return frame

    # --- image sections ---

    def _build_single_image_section(self, title: str, path: Path) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #141414; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        heading = QLabel(title)
        heading.setStyleSheet(
            "QLabel { font-size: 14px; font-weight: bold; color: #ace; }"
        )
        layout.addWidget(heading)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet(
            "QLabel { background-color: #0a0a0a; border: 1px solid #222;"
            " border-radius: 4px; }"
        )
        pm = _fit_image_to_width(path, max_width=720)
        if pm is None:
            image_label.setText(f"{path.name} not available")
            image_label.setStyleSheet(
                "QLabel { color: #666; font-style: italic;"
                " background-color: #0a0a0a; border: 1px solid #222;"
                " border-radius: 4px; padding: 30px; }"
            )
            image_label.setFixedHeight(80)
        else:
            image_label.setPixmap(pm)
            image_label.setFixedSize(pm.size())
        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        return frame

    def _build_placements_section(self, race_dir: Path) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #141414; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        paths: list[Path] = []
        if race_dir.exists():
            paths = sorted(race_dir.glob("placement_*.png"))

        heading = QLabel(
            f"Placements screens ({len(paths)} frame"
            f"{'s' if len(paths) != 1 else ''})",
        )
        heading.setStyleSheet(
            "QLabel { font-size: 14px; font-weight: bold; color: #ace; }"
        )
        layout.addWidget(heading)

        carousel = _ImageCarousel(paths, max_width=720)
        layout.addWidget(carousel)
        return frame


class MatchDetailView(QWidget):
    """Right-hand detail pane: stacked match-timeline + race-detail views."""

    def __init__(self, debug_dir: Path = DEFAULT_DEBUG_DIR) -> None:
        super().__init__()
        self._debug_dir = debug_dir
        self._current_record: MatchRecord | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._timeline = _MatchTimelinePane()
        self._timeline.raceSelected.connect(self._on_race_selected)
        self._race_detail = _RaceDetailView()
        self._race_detail.backRequested.connect(self._show_timeline)

        self._stack.addWidget(self._timeline)    # index 0
        self._stack.addWidget(self._race_detail)  # index 1
        layout.addWidget(self._stack)

    @property
    def is_showing_race_detail(self) -> bool:
        return self._stack.currentIndex() == 1

    def set_debug_dir(self, debug_dir: Path) -> None:
        self._debug_dir = debug_dir

    def set_record(
        self,
        record: MatchRecord,
        *,
        live: bool = False,
        live_status: str | None = None,
    ) -> None:
        prev_id = self._current_record.match_id if self._current_record else None
        self._current_record = record
        self._timeline.set_record(record, live=live, live_status=live_status)
        # Snap back to the timeline when switching to a different match so the
        # user doesn't see a stale race-detail page from a prior selection.
        # tick() already skips updates while race-detail is showing, so same-
        # match live refreshes won't reach this branch.
        if prev_id != record.match_id:
            self._stack.setCurrentIndex(0)

    def clear(self) -> None:
        self._current_record = None
        self._timeline.clear()
        self._stack.setCurrentIndex(0)

    def _show_timeline(self) -> None:
        self._stack.setCurrentIndex(0)

    def _on_race_selected(self, race_number: int) -> None:
        record = self._current_record
        if record is None:
            return
        race = next(
            (r for r in record.races if r.race_number == race_number),
            None,
        )
        if race is None:
            return
        match_dir = self._debug_dir / record.match_id
        self._race_detail.set_race(race, record.settings, match_dir)
        self._stack.setCurrentIndex(1)


class MatchHistoryView(QWidget):
    """Full history screen: match list on the left, detail pane on the right."""

    backRequested = Signal()

    def __init__(
        self,
        debug_dir: Path = DEFAULT_DEBUG_DIR,
        state_machine: "GameStateMachine | None" = None,
    ) -> None:
        super().__init__()
        self._debug_dir = debug_dir
        self._state_machine = state_machine
        self._records: list[MatchRecord] = []
        # Tracks the live match id we showed on the most recent tick(); used
        # to detect match-start / match-end transitions and rebuild the list.
        self._last_live_match_id: str | None = None
        # mtime of the live match's match.json the last time we re-rendered
        # the detail pane.  Used to skip redundant rebuilds (tick fires at
        # ~30fps; the file only changes when state machine writes).
        self._last_live_mtime: float = 0.0
        # The live status string we last rendered.  Lets the status banner
        # update on state changes that don't trigger a match.json write
        # (e.g. transitioning into WAITING_FOR_RACE_END).
        self._last_live_status: str | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # --- top bar: title + refresh ---
        top = QHBoxLayout()
        top.setContentsMargins(4, 0, 4, 0)
        title = QLabel("Match History")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        top.addWidget(title)
        top.addStretch()
        self._count_label = QLabel()
        self._count_label.setStyleSheet("QLabel { color: #888; }")
        top.addWidget(self._count_label)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        top.addWidget(refresh_btn)
        root.addLayout(top)

        # --- split: list + detail ---
        split = QHBoxLayout()
        split.setSpacing(8)

        self._list = QListWidget()
        self._list.setFixedWidth(280)
        self._list.setStyleSheet(
            "QListWidget { background-color: #151515; border: 1px solid #2a2a2a; }"
            " QListWidget::item { padding: 6px; border-bottom: 1px solid #222; }"
            " QListWidget::item:selected { background-color: #2a3b4a; color: #fff; }"
        )
        self._list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._list.currentRowChanged.connect(self._on_row_changed)
        split.addWidget(self._list)

        self._detail = MatchDetailView(debug_dir=self._debug_dir)
        split.addWidget(self._detail, stretch=1)

        root.addLayout(split, stretch=1)

    def _live_match_id(self) -> str | None:
        sm = self._state_machine
        if sm is None or not sm.is_match_active:
            return None
        return sm.current_match_id

    def _live_status_text(self) -> str | None:
        """Human-readable description of what the state machine is doing,
        or ``None`` if no match is active."""
        sm = self._state_machine
        if sm is None or not sm.is_match_active:
            return None
        return _LIVE_STATUS_TEXT.get(sm.state.name, sm.state.name)

    def refresh(self) -> None:
        """Re-scan ``debug_dir`` and rebuild the match list."""
        previous_id = None
        current_row = self._list.currentRow()
        if 0 <= current_row < len(self._records):
            previous_id = self._records[current_row].match_id

        self._records = list_matches(self._debug_dir)
        live_id = self._live_match_id()
        self._last_live_match_id = live_id
        self._last_live_mtime = 0.0
        self._last_live_status = None

        self._list.clear()
        for rec in self._records:
            started = _format_timestamp(rec.started_at)
            is_live = rec.match_id == live_id
            summary = _summary_line(rec, live=is_live)
            prefix = "●  LIVE\n" if is_live else ""
            item = QListWidgetItem(f"{prefix}{started}\n{summary}")
            if is_live:
                item.setBackground(QColor("#3a1414"))
                item.setForeground(QColor("#fff"))
                item.setToolTip("This match is currently in progress")
            self._list.addItem(item)

        self._count_label.setText(
            f"{len(self._records)} match{'es' if len(self._records) != 1 else ''}"
        )

        if not self._records:
            self._detail.clear()
            return

        # Restore previous selection if still present, otherwise pick the top.
        target_row = 0
        if previous_id:
            for idx, rec in enumerate(self._records):
                if rec.match_id == previous_id:
                    target_row = idx
                    break
        self._list.setCurrentRow(target_row)

    def tick(self) -> None:
        """Re-render the live match if it's currently selected.

        Cheap when nothing has changed.  Called from the main window's frame
        loop so the live timeline updates without waiting for a tab switch.
        """
        if not self.isVisible():
            return
        # Don't clobber the race-detail page if the user is viewing one.
        if self._detail.is_showing_race_detail:
            return
        live_id = self._live_match_id()

        # Match started or ended since the last tick — rebuild the list so
        # the LIVE badge appears/disappears in the right place.
        if live_id != self._last_live_match_id:
            logger.info(
                "Live match changed: %s -> %s (rebuilding list)",
                self._last_live_match_id, live_id,
            )
            self.refresh()
            return

        # Same live match still in progress.  If it's the selected row,
        # re-load it from disk (the state machine writes match.json on every
        # update) and re-render the detail pane — but only when match.json
        # has actually changed OR when the state machine has stepped into a
        # new state (so the status banner updates between disk writes).
        if live_id is None:
            return
        current_row = self._list.currentRow()
        if not (0 <= current_row < len(self._records)):
            return
        if self._records[current_row].match_id != live_id:
            return
        match_path = self._debug_dir / live_id / MATCH_FILE
        try:
            mtime = match_path.stat().st_mtime
        except OSError:
            return
        live_status = self._live_status_text()
        if mtime == self._last_live_mtime and live_status == self._last_live_status:
            return
        try:
            record = MatchRecord.load(self._debug_dir / live_id)
        except (OSError, ValueError, KeyError):
            return
        self._last_live_mtime = mtime
        self._last_live_status = live_status
        self._records[current_row] = record
        self._detail.set_record(record, live=True, live_status=live_status)
        logger.debug(
            "Live match %s re-rendered (status: %s, races: %d)",
            live_id, live_status, len(record.races),
        )

    def _on_row_changed(self, row: int) -> None:
        self._last_live_mtime = 0.0
        self._last_live_status = None
        if 0 <= row < len(self._records):
            rec = self._records[row]
            is_live = rec.match_id == self._live_match_id()
            live_status = self._live_status_text() if is_live else None
            self._detail.set_record(rec, live=is_live, live_status=live_status)
        else:
            self._detail.clear()
