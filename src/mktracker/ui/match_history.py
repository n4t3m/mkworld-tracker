"""Match history UI.

Shows a list of previously played matches (loaded from ``matches/``) on
the left and a scrollable per-race timeline on the right.  The detail widget
accepts any :class:`MatchRecord`, so it is also used to display the
currently-running match — the state machine writes ``match.json`` after every
update, so the on-disk record is always live.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mktracker.detection.tracks import TRACK_ICONS_DIR, TRACK_IMAGES
from mktracker.gemini_client import load_api_key
from mktracker.gemini_match_results import request_match_results
from mktracker.gemini_results import request_race_results
from mktracker.lorenzi_text import standings_to_text, text_to_standings
from mktracker.match_record import (
    DEFAULT_MATCHES_DIR,
    MATCH_FILE,
    FinalStandings,
    MatchRecord,
    MatchSettingsRecord,
    RaceRecord,
    TeamGroup,
    final_standings_from_gemini,
    list_matches,
    race_fields_from_gemini,
)
from mktracker.table_generator import generate_table

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


from mktracker.team_scoring import race_team_scores as _race_team_scores


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
    refetchPlacementsRequested = Signal(int)  # race_number

    def __init__(
        self,
        race: RaceRecord,
        settings: MatchSettingsRecord,
        *,
        live: bool = False,
        match_dir: Path | None = None,
        api_key_available: bool = False,
    ) -> None:
        super().__init__()
        self._live = live
        self._settings = settings
        self._race_number = race.race_number
        self._match_dir = match_dir
        self._api_key_available = api_key_available
        self._regenerate_btn: QPushButton | None = None
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
            inner = self._build_team_placements(race.teams)
            return self._wrap_with_regenerate(race, inner)
        if race.placements:
            inner = self._build_solo_placements(race)
            return self._wrap_with_regenerate(race, inner)
        if self._live:
            note = QLabel("Awaiting placements…")
            note.setStyleSheet("QLabel { color: #c93; font-style: italic; }")
            return note

        # Non-live, no placements: show the message and a regenerate button
        # when the user has Gemini configured and we have placement frames on
        # disk to feed back in.
        note = QLabel("No placements recorded.")
        note.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        return self._wrap_with_regenerate(race, note, inline=True)

    def _wrap_with_regenerate(
        self, race: RaceRecord, inner: QWidget, *, inline: bool = False,
    ) -> QWidget:
        """Attach a Regenerate button to *inner* when applicable.

        When *inline* is True the button sits on the same row as *inner*
        (used for the "No placements recorded." message); otherwise a
        small button row is rendered above the placement list.
        """
        if not self._can_regenerate(race):
            return inner

        btn = QPushButton("↻  Regenerate")
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip("Re-run Gemini against the saved placement_*.png frames")
        self._apply_regenerate_button_style(btn, in_progress=False)
        btn.clicked.connect(
            lambda: self.refetchPlacementsRequested.emit(self._race_number),
        )
        self._regenerate_btn = btn

        if inline:
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(inner)
            row.addStretch()
            row.addWidget(btn)
            return container

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addStretch()
        header.addWidget(btn)
        layout.addLayout(header)
        layout.addWidget(inner)
        return container

    def _can_regenerate(self, race: RaceRecord) -> bool:
        if not self._api_key_available:
            return False
        if self._match_dir is None:
            return False
        race_dir = self._match_dir / f"race_{race.race_number:02d}"
        if not race_dir.exists():
            return False
        return any(race_dir.glob("placement_*.png"))

    @staticmethod
    def _apply_regenerate_button_style(
        btn: QPushButton, *, in_progress: bool,
    ) -> None:
        if in_progress:
            btn.setText("Regenerating…")
            btn.setEnabled(False)
            btn.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
            btn.setStyleSheet(
                "QPushButton { background-color: #2a2a2a; color: #888;"
                " border: 1px solid #333; border-radius: 4px;"
                " padding: 4px 10px; font-size: 12px; font-weight: bold; }"
            )
        else:
            btn.setText("↻  Regenerate")
            btn.setEnabled(True)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                "QPushButton { background-color: #2a3b4a; color: #fff;"
                " border: none; border-radius: 4px; padding: 4px 10px;"
                " font-size: 12px; font-weight: bold; }"
                " QPushButton:hover { background-color: #35506b; }"
            )

    def set_refetch_in_progress(self, in_progress: bool) -> None:
        if self._regenerate_btn is None:
            return
        self._apply_regenerate_button_style(
            self._regenerate_btn, in_progress=in_progress,
        )

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


_PIP_WIDTH = 74
_PIP_HEIGHT = 76
_PIP_ICON_HEIGHT = 40
_PIP_BADGE_HEIGHT = 20
_PIP_SPACING = 3


def _rank_badge_style(rank: int | None) -> tuple[str, str]:
    """Return ``(background, foreground)`` colours for a rank badge."""
    if rank is None:
        return ("#333", "#aaa")
    if rank == 1:
        return ("#fd4", "#5a3800")
    if rank <= 4:
        return ("#2d4", "#042")
    if rank <= 8:
        return ("#c93", "#3a2000")
    return ("#c55", "#400")


class _RacePip(QFrame):
    """Compact overview tile for a single race: track icon + rank badge."""

    clicked = Signal(int)  # race_number

    def __init__(self, race: RaceRecord) -> None:
        super().__init__()
        self._race_number = race.race_number
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(_PIP_WIDTH, _PIP_HEIGHT)
        tip_parts = [f"Race {race.race_number}", race.track_name or "Unknown track"]
        if race.user_rank is not None:
            tip_parts.append(f"Rank {race.user_rank}")
        self.setToolTip("  ·  ".join(tip_parts))
        self.setStyleSheet(
            "_RacePip { background-color: #1a1a1a; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
            " _RacePip:hover { border: 1px solid #5a7a9a;"
            " background-color: #22262b; }"
            " QLabel { background: transparent; border: none; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFixedHeight(_PIP_ICON_HEIGHT)
        pixmap = _load_track_icon(race.track_name)
        if pixmap is not None:
            # Track icons are 16:9 — use aspect-preserving scale so they fit
            # inside the pip width without being clipped by the label bounds.
            scaled = pixmap.scaled(
                _PIP_WIDTH - 8,
                _PIP_ICON_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(scaled)
        else:
            icon_label.setText("?")
            icon_label.setStyleSheet("QLabel { color: #555; font-size: 22px; }")
        layout.addWidget(icon_label)

        badge_text = str(race.user_rank) if race.user_rank is not None else "…"
        bg, fg = _rank_badge_style(race.user_rank)
        badge = QLabel(badge_text)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(_PIP_BADGE_HEIGHT)
        badge.setStyleSheet(
            f"QLabel {{ background-color: {bg}; color: {fg}; font-weight: bold;"
            " font-size: 13px; border-radius: 9px; }"
        )
        layout.addWidget(badge)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._race_number)
        super().mousePressEvent(event)


class _PendingRacePip(QFrame):
    """Dashed placeholder pip for a race that hasn't been played yet."""

    def __init__(self, race_number: int) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(_PIP_WIDTH, _PIP_HEIGHT)
        self.setToolTip(f"Race {race_number}: not yet played")
        self.setStyleSheet(
            "_PendingRacePip { background-color: #15171a;"
            " border: 1px dashed #2d2f33; border-radius: 6px; }"
            " QLabel { background: transparent; border: none; color: #444; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        icon = QLabel(str(race_number))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedHeight(_PIP_ICON_HEIGHT)
        icon.setStyleSheet("QLabel { color: #3a3a3a; font-size: 22px; font-weight: bold; }")
        layout.addWidget(icon)

        dash = QLabel("—")
        dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash.setFixedHeight(_PIP_BADGE_HEIGHT)
        dash.setStyleSheet("QLabel { color: #444; font-size: 13px; }")
        layout.addWidget(dash)


_RAIL_COLOR_PLAYED = QColor("#6a9fd4")
_RAIL_COLOR_PENDING = QColor("#32353c")
_RAIL_WIDTH = 3
_PULSE_COLOR = QColor(255, 90, 90)


def _apply_pulse(widget: QWidget) -> None:
    """Attach an animated red drop-shadow glow to *widget*, looping forever."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setColor(_PULSE_COLOR)
    effect.setOffset(0, 0)
    effect.setBlurRadius(10)
    widget.setGraphicsEffect(effect)

    anim = QPropertyAnimation(effect, b"blurRadius", widget)
    anim.setDuration(1400)
    anim.setKeyValueAt(0.0, 6.0)
    anim.setKeyValueAt(0.5, 28.0)
    anim.setKeyValueAt(1.0, 6.0)
    anim.setLoopCount(-1)
    anim.setEasingCurve(QEasingCurve.Type.InOutSine)
    anim.start()


class _PipTimeline(QWidget):
    """Row of race pips with a subway-style rail painted behind them.

    The rail is solid/blue through played pips and dashed/grey through
    pending ones.  Because pip backgrounds are opaque, the rail is visible
    in the gaps between pips — reads as "connected stops on a route".
    """

    def __init__(self) -> None:
        super().__init__()
        # Leave breathing room top/bottom so the pulse glow isn't clipped by
        # the widget's own bounds.
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 14, 0, 14)
        self._row.setSpacing(_PIP_SPACING)
        self._row.addStretch()
        self._row.addStretch()
        self._pips: list[tuple[QWidget, bool]] = []

    def add_pip(
        self,
        widget: QWidget,
        *,
        played: bool,
        pulse: bool = False,
    ) -> None:
        insert_at = self._row.count() - 1  # before trailing stretch
        self._row.insertWidget(insert_at, widget)
        self._pips.append((widget, played))
        if pulse:
            _apply_pulse(widget)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._pips:
            return
        first_w = self._pips[0][0]
        last_w = self._pips[-1][0]
        # Skip until layout has assigned real geometry.
        if first_w.width() == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        y = first_w.y() + first_w.height() // 2
        x_start = first_w.x() + first_w.width() // 2
        x_end = last_w.x() + last_w.width() // 2

        last_played_x: int | None = None
        for widget, played in self._pips:
            if played:
                last_played_x = widget.x() + widget.width() // 2

        if last_played_x is not None and last_played_x > x_start:
            pen = QPen(_RAIL_COLOR_PLAYED, _RAIL_WIDTH, Qt.PenStyle.SolidLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(x_start, y, last_played_x, y)

        dash_start = last_played_x if last_played_x is not None else x_start
        if dash_start < x_end:
            pen = QPen(_RAIL_COLOR_PENDING, _RAIL_WIDTH, Qt.PenStyle.DashLine)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawLine(dash_start, y, x_end, y)


class _RacePipStrip(QFrame):
    """Horizontal at-a-glance strip: one pip per race, connected by a rail.

    For live matches, the "current" pip (the race in progress or the next
    one up) pulses to draw the eye to ongoing progress.
    """

    raceSelected = Signal(int)  # race_number

    def __init__(
        self,
        races: list[RaceRecord],
        total_races: int,
        *,
        live: bool = False,
    ) -> None:
        super().__init__()
        self.setStyleSheet(
            "_RacePipStrip { background-color: #14181f;"
            " border: 1px solid #232a33; border-radius: 6px; }"
            " QLabel#pip_caption { color: #889; font-weight: bold;"
            " font-size: 11px; letter-spacing: 1px; background: transparent;"
            " border: none; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(2)

        played = len(races)
        if live:
            caption_text = f"PROGRESS  ·  {played}/{total_races} RACES PLAYED"
        else:
            caption_text = (
                f"OVERVIEW  ·  {played} RACE{'S' if played != 1 else ''}"
            )
        caption = QLabel(caption_text)
        caption.setObjectName("pip_caption")
        outer.addWidget(caption)

        by_number = {r.race_number: r for r in races}
        max_played = max(by_number.keys(), default=0)
        total = max(total_races, max_played)

        # For live matches, figure out which pip is "current" so it can pulse.
        # The race in progress is either the most-recent played pip (if its
        # user_rank hasn't come back yet) or the first pending pip.
        current_n: int | None = None
        if live:
            last_played = by_number.get(max_played) if max_played else None
            if last_played is not None and last_played.user_rank is None:
                current_n = max_played
            elif played < total:
                current_n = played + 1

        timeline = _PipTimeline()
        for n in range(1, total + 1):
            race = by_number.get(n)
            is_current = live and n == current_n
            if race is not None:
                pip = _RacePip(race)
                pip.clicked.connect(self.raceSelected.emit)
                timeline.add_pip(pip, played=True, pulse=is_current)
            else:
                timeline.add_pip(
                    _PendingRacePip(n), played=False, pulse=is_current,
                )
        outer.addWidget(timeline)


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


class _TableEditDialog(QDialog):
    """Lorenzi-style textbox editor for the results table.

    Prefilled from ``record.final_standings``.  On save, parses the text
    back into a :class:`FinalStandings`, writes it to ``match.json``, and
    regenerates ``table.png`` in the match folder.
    """

    def __init__(
        self,
        record: MatchRecord,
        matches_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._record = record
        self._matches_dir = matches_dir
        self._match_dir = matches_dir / record.match_id
        self.setWindowTitle(f"Edit table — {record.match_id}")
        self.resize(560, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        intro = QLabel(
            "One team per block, blank line between teams.\n"
            "First line of a block is the team tag. Player lines are"
            " '<name> <score>' — scores may be expressions like '70+20+8'.\n"
            "Omit tag lines for FFA matches."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("QLabel { color: #aab; }")
        layout.addWidget(intro)

        self._editor = QPlainTextEdit()
        self._editor.setPlainText(standings_to_text(record.final_standings))
        self._editor.setStyleSheet(
            "QPlainTextEdit { background-color: #101418; color: #eee;"
            " font-family: Consolas, monospace; font-size: 13px;"
            " border: 1px solid #2a2a2a; border-radius: 4px; padding: 6px; }"
        )
        layout.addWidget(self._editor, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        try:
            standings = text_to_standings(self._editor.toPlainText())
        except Exception as exc:
            QMessageBox.warning(
                self, "Invalid table text",
                f"Could not parse table text:\n{exc}",
            )
            return
        if not standings.players:
            QMessageBox.warning(
                self, "Empty table",
                "No player rows found — add at least one '<name> <score>' line.",
            )
            return

        self._record.final_standings = standings
        try:
            png = generate_table(self._record)
            (self._match_dir / "table.png").write_bytes(png)
            self._record.save(self._match_dir)
        except Exception as exc:
            logger.exception("Failed to save edited table")
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save edited table:\n{exc}",
            )
            return
        self.accept()


class _RefetchConfirmDialog(QDialog):
    """Preview ``match_results.png`` and confirm before re-running Gemini.

    Re-fetching only succeeds when the saved frame actually shows the final
    CONGRATULATIONS!/NICE TRY!/DRAW! standings screen.  Surfacing the image
    up front lets the user verify that before spending a Gemini call.
    """

    def __init__(self, results_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Regenerate table?")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        warning = QLabel(
            "This will re-run Gemini against the captured match-results frame"
            " below and overwrite the current table.\n\n"
            "The frame must show the final results screen"
            " (CONGRATULATIONS!/NICE TRY!/DRAW!). If it shows anything else"
            " — a mid-race screen, the player list, gameplay — Gemini will"
            " return garbage or fail."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("QLabel { color: #f0c674; }")
        layout.addWidget(warning)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setStyleSheet(
            "QLabel { background-color: #101418; border: 1px solid #2a2a2a;"
            " border-radius: 4px; }"
        )
        pixmap = QPixmap(str(results_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                720, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            preview.setPixmap(scaled)
        else:
            preview.setText(f"Could not load {results_path.name}")
            preview.setStyleSheet("QLabel { color: #f88; padding: 24px; }")
        layout.addWidget(preview, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel,
        )
        regenerate_btn = buttons.addButton(
            "Regenerate", QDialogButtonBox.ButtonRole.AcceptRole
        )
        regenerate_btn.setDefault(False)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _RefetchTableThread(QThread):
    """Re-runs the Gemini match-results call against ``match_results.png``.

    Synchronously bridges the existing fire-and-forget
    :func:`request_match_results` API into a QThread so the caller gets a
    single ``finished(parsed)`` signal on the Qt event loop.
    """

    finished = Signal(object)  # parsed dict | None

    def __init__(self, frame, log_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._log_dir = log_dir

    def run(self) -> None:
        import threading
        done = threading.Event()
        holder: dict[str, object | None] = {"parsed": None}

        def cb(parsed, results) -> None:  # noqa: ARG001 — results unused
            del results
            holder["parsed"] = parsed
            done.set()

        request_match_results(self._frame, cb, log_dir=self._log_dir)
        done.wait()
        self.finished.emit(holder["parsed"])


class _RefetchPlacementsThread(QThread):
    """Re-runs the Gemini race-results call against the saved ``placement_*.png``
    frames for one race.

    Bridges :func:`request_race_results` into a single
    ``finished(parsed)`` Qt signal.
    """

    finished = Signal(object)  # parsed dict | None

    def __init__(
        self,
        frames: list,
        race_number: int,
        log_dir: Path,
        parent=None,
        *,
        teams_setting: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._frames = frames
        self._race_number = race_number
        self._log_dir = log_dir
        self._teams_setting = teams_setting

    def run(self) -> None:
        import threading
        done = threading.Event()
        holder: dict[str, object | None] = {"parsed": None}

        def cb(parsed, placements) -> None:  # noqa: ARG001 — placements unused
            del placements
            holder["parsed"] = parsed
            done.set()

        request_race_results(
            self._frames, self._race_number, cb,
            log_dir=self._log_dir,
            teams_setting=self._teams_setting,
        )
        done.wait()
        self.finished.emit(holder["parsed"])


class _RefetchPlacementsConfirmDialog(QDialog):
    """Preview the saved placement frames and confirm before re-running Gemini.

    The Gemini race-results call only succeeds when the supplied frames show
    the post-race placement screen — surfacing the carousel up front lets the
    user verify that.
    """

    def __init__(
        self,
        frame_paths: list[Path],
        race_number: int,
        track_name: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Regenerate placements — race {race_number}")
        self.resize(780, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_text = f"Race {race_number}"
        if track_name:
            title_text += f"  —  {track_name}"
        title = QLabel(title_text)
        title.setStyleSheet(
            "QLabel { color: #fff; font-size: 14px; font-weight: bold; }"
        )
        layout.addWidget(title)

        warning = QLabel(
            "This will run Gemini against the placement frames captured for"
            " this race and overwrite any existing placements.\n\n"
            "The frames must show the post-race results screen (the scrolling"
            " placement bars). If they show anything else — gameplay, the"
            " track-select map, the player list — Gemini will return"
            " garbage or fail."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("QLabel { color: #f0c674; }")
        layout.addWidget(warning)

        carousel = _ImageCarousel(frame_paths, max_width=720)
        layout.addWidget(carousel, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel,
        )
        regenerate_btn = buttons.addButton(
            "Regenerate", QDialogButtonBox.ButtonRole.AcceptRole
        )
        regenerate_btn.setDefault(False)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _TableImageCard(QFrame):
    """Embeds the generated Lorenzi-style results table image."""

    editRequested = Signal()
    refetchRequested = Signal()

    def __init__(
        self,
        path: Path,
        *,
        show_edit_button: bool = True,
        show_refetch_button: bool = False,
    ) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_TableImageCard { background-color: #141414; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        heading = QLabel("Results Table")
        heading.setStyleSheet(
            "QLabel { font-size: 15px; font-weight: bold; color: #ace; }"
        )
        header_row.addWidget(heading)
        header_row.addStretch()

        self._full_pixmap: QPixmap | None = None
        pm = QPixmap(str(path))
        if not pm.isNull():
            self._full_pixmap = pm

        btn_style = (
            "QPushButton { background-color: #2a3b4a; color: #fff; border: none;"
            " border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            " QPushButton:hover { background-color: #35506b; }"
        )
        if self._full_pixmap is not None:
            self._copy_btn = QPushButton("Copy Table")
            self._copy_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self._copy_btn.setStyleSheet(btn_style)
            self._copy_btn.setToolTip("Copy the table image to the clipboard")
            self._copy_btn.clicked.connect(self._copy_to_clipboard)
            header_row.addWidget(self._copy_btn)
        if show_edit_button:
            edit_btn = QPushButton("Edit Table")
            edit_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            edit_btn.setStyleSheet(btn_style)
            edit_btn.clicked.connect(self.editRequested.emit)
            header_row.addWidget(edit_btn)

            if show_refetch_button:
                self._refetch_btn = QPushButton("Regenerate Table")
                self._refetch_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                self._refetch_btn.setStyleSheet(btn_style)
                self._refetch_btn.setToolTip(
                    "Re-run Gemini against match_results.png and rebuild "
                    "the final standings",
                )
                self._refetch_btn.clicked.connect(self.refetchRequested.emit)
                header_row.addWidget(self._refetch_btn)
        layout.addLayout(header_row)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self._full_pixmap is None:
            image_label.setText(f"{path.name} could not be loaded")
            image_label.setStyleSheet(
                "QLabel { color: #666; font-style: italic; padding: 20px; }"
            )
        else:
            scaled = self._full_pixmap.scaledToWidth(
                720, Qt.TransformationMode.SmoothTransformation,
            )
            image_label.setPixmap(scaled)
            image_label.setFixedSize(scaled.size())
        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)

    def set_refetch_in_progress(self, in_progress: bool) -> None:
        """Disable the regenerate button and grey it out while the Gemini
        refetch is running."""
        btn = getattr(self, "_refetch_btn", None)
        if btn is None:
            return
        if in_progress:
            btn.setEnabled(False)
            btn.setText("Regenerating…")
            btn.setStyleSheet(
                "QPushButton { background-color: #1a1a1a; color: #666; border: none;"
                " border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            )
            btn.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
        else:
            btn.setEnabled(True)
            btn.setText("Regenerate Table")
            btn.setStyleSheet(
                "QPushButton { background-color: #2a3b4a; color: #fff; border: none;"
                " border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
                " QPushButton:hover { background-color: #35506b; }"
            )
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def _copy_to_clipboard(self) -> None:
        if self._full_pixmap is None:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setPixmap(self._full_pixmap)
        original = self._copy_btn.text()
        self._copy_btn.setText("Copied!")
        self._copy_btn.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, lambda: (
            self._copy_btn.setText(original),
            self._copy_btn.setEnabled(True),
        ))


class _MatchTimelinePane(QScrollArea):
    """Scrollable timeline of races for a single match.

    Emits :attr:`raceSelected` when the user clicks a race card.
    """

    raceSelected = Signal(int)  # race_number
    editTableRequested = Signal()
    refetchTableRequested = Signal()
    refetchPlacementsRequested = Signal(int)  # race_number

    def __init__(self, matches_dir: Path = DEFAULT_MATCHES_DIR) -> None:
        super().__init__()
        self._matches_dir = matches_dir
        self._refetch_in_progress = False
        self._table_card: _TableImageCard | None = None
        self._refetch_placements_race: int | None = None
        self._race_cards: dict[int, _RaceCard] = {}
        self.setWidgetResizable(True)
        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(10)
        self._layout.addStretch()
        self.setWidget(self._inner)

        # Floating "jump to top" button — parented to the viewport so it stays
        # pinned to the visible area.  Hidden until the user has scrolled down
        # far enough that getting back to the top matters.
        self._jump_top_btn = QPushButton("▲  Top", self.viewport())
        self._jump_top_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._jump_top_btn.setToolTip("Scroll to top")
        self._jump_top_btn.setFixedHeight(34)
        self._jump_top_btn.setStyleSheet(
            "QPushButton { background-color: rgba(42, 59, 74, 230);"
            " color: #fff; border: 1px solid #4a6b8a; border-radius: 17px;"
            " padding: 4px 16px; font-weight: bold; }"
            " QPushButton:hover { background-color: #35506b; }"
        )
        self._jump_top_btn.clicked.connect(
            lambda: self.verticalScrollBar().setValue(0),
        )
        self._jump_top_btn.hide()
        self.verticalScrollBar().valueChanged.connect(
            self._update_jump_top_visibility,
        )

        self._show_empty()

    def _update_jump_top_visibility(self, value: int) -> None:
        visible = value > 300
        if visible != self._jump_top_btn.isVisible():
            self._jump_top_btn.setVisible(visible)
            if visible:
                self._position_jump_top()
                self._jump_top_btn.raise_()

    def _position_jump_top(self) -> None:
        btn = self._jump_top_btn
        btn.adjustSize()
        vp = self.viewport()
        margin = 14
        btn.move(
            vp.width() - btn.width() - margin,
            vp.height() - btn.height() - margin,
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_jump_top()

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
        self._table_card = None
        self._show_empty()

    def set_refetch_in_progress(self, in_progress: bool) -> None:
        self._refetch_in_progress = in_progress
        if self._table_card is not None:
            self._table_card.set_refetch_in_progress(in_progress)

    def set_refetch_placements_in_progress(
        self, race_number: int | None,
    ) -> None:
        prev = self._refetch_placements_race
        self._refetch_placements_race = race_number
        if prev is not None and prev in self._race_cards and prev != race_number:
            self._race_cards[prev].set_refetch_in_progress(False)
        if race_number is not None and race_number in self._race_cards:
            self._race_cards[race_number].set_refetch_in_progress(True)

    def set_record(
        self,
        record: MatchRecord,
        *,
        live: bool = False,
        live_status: str | None = None,
    ) -> None:
        self._clear()
        self._race_cards = {}
        match_dir = self._matches_dir / record.match_id
        api_key_available = bool(load_api_key())

        if live:
            races_played = len(record.races)
            total = max(record.settings.race_count, races_played)
            progress = f"Race {min(races_played + 1, total)} of {total}"
            status_text = live_status or "Live"
            self._layout.addWidget(_LiveStatusBanner(status_text, progress))

        header = self._build_header(record, live=live)
        self._layout.addWidget(header)

        table_path = self._matches_dir / record.match_id / "table.png"
        results_path = self._matches_dir / record.match_id / "match_results.png"
        if record.final_standings is not None and table_path.exists():
            card = _TableImageCard(
                table_path,
                show_edit_button=not live,
                show_refetch_button=not live and results_path.exists(),
            )
            card.editRequested.connect(self.editTableRequested.emit)
            card.refetchRequested.connect(self.refetchTableRequested.emit)
            self._table_card = card
            if self._refetch_in_progress:
                card.set_refetch_in_progress(True)
            self._layout.addWidget(card)
        else:
            self._table_card = None

        races = sorted(record.races, key=lambda r: r.race_number)
        if races or live:
            strip = _RacePipStrip(races, record.settings.race_count, live=live)
            strip.raceSelected.connect(self.raceSelected.emit)
            self._layout.addWidget(strip)

        if not races and not live:
            note = QLabel("No races recorded yet.")
            note.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 10px; }")
            self._layout.addWidget(note)
        else:
            for race in races:
                card = _RaceCard(
                    race, record.settings,
                    live=live,
                    match_dir=match_dir,
                    api_key_available=api_key_available,
                )
                card.clicked.connect(self.raceSelected.emit)
                card.refetchPlacementsRequested.connect(
                    self.refetchPlacementsRequested.emit,
                )
                self._race_cards[race.race_number] = card
                if (
                    self._refetch_placements_race is not None
                    and self._refetch_placements_race == race.race_number
                ):
                    card.set_refetch_in_progress(True)
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
            "Race Vote", race_dir / "vote.png",
        ))
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

    recordEdited = Signal(str)  # match_id

    def __init__(self, matches_dir: Path = DEFAULT_MATCHES_DIR) -> None:
        super().__init__()
        self._matches_dir = matches_dir
        self._current_record: MatchRecord | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._refetch_thread: _RefetchTableThread | None = None
        self._refetch_placements_thread: _RefetchPlacementsThread | None = None

        self._stack = QStackedWidget()
        self._timeline = _MatchTimelinePane(matches_dir=self._matches_dir)
        self._timeline.raceSelected.connect(self._on_race_selected)
        self._timeline.editTableRequested.connect(self._on_edit_table)
        self._timeline.refetchTableRequested.connect(self._on_refetch_table)
        self._timeline.refetchPlacementsRequested.connect(
            self._on_refetch_placements,
        )
        self._race_detail = _RaceDetailView()
        self._race_detail.backRequested.connect(self._show_timeline)

        self._stack.addWidget(self._timeline)    # index 0
        self._stack.addWidget(self._race_detail)  # index 1
        layout.addWidget(self._stack)

    @property
    def is_showing_race_detail(self) -> bool:
        return self._stack.currentIndex() == 1

    def set_matches_dir(self, matches_dir: Path) -> None:
        self._matches_dir = matches_dir
        self._timeline._matches_dir = matches_dir

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

    def _on_edit_table(self) -> None:
        record = self._current_record
        if record is None:
            return
        dialog = _TableEditDialog(record, self._matches_dir, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                updated = MatchRecord.load(self._matches_dir / record.match_id)
            except (OSError, ValueError, KeyError):
                updated = record
            self.set_record(updated)
            self.recordEdited.emit(record.match_id)

    def _on_refetch_table(self) -> None:
        record = self._current_record
        if record is None:
            return
        if self._refetch_thread is not None and self._refetch_thread.isRunning():
            return
        match_dir = self._matches_dir / record.match_id
        results_path = match_dir / "match_results.png"
        if not results_path.exists():
            QMessageBox.warning(
                self, "Refetch failed",
                f"No match_results.png in {match_dir.name}.",
            )
            return
        if not load_api_key():
            QMessageBox.warning(
                self, "Refetch failed",
                "No Gemini API key configured. Set one in the Settings tab.",
            )
            return

        confirm = _RefetchConfirmDialog(results_path, parent=self)
        if confirm.exec() != QDialog.DialogCode.Accepted:
            return

        import cv2
        frame = cv2.imread(str(results_path))
        if frame is None:
            QMessageBox.critical(
                self, "Refetch failed",
                f"Could not read {results_path.name}.",
            )
            return

        match_id = record.match_id
        thread = _RefetchTableThread(frame, match_dir, parent=self)
        thread.finished.connect(
            lambda parsed: self._on_refetch_finished(match_id, parsed)
        )
        self._refetch_thread = thread
        self._timeline.set_refetch_in_progress(True)
        thread.start()

    def _on_refetch_finished(self, match_id: str, parsed: object) -> None:
        self._timeline.set_refetch_in_progress(False)
        if not isinstance(parsed, dict):
            QMessageBox.warning(
                self, "Refetch failed",
                "Gemini did not return a usable response. See "
                "gemini_match_results.txt in the match folder for details.",
            )
            return
        match_dir = self._matches_dir / match_id
        try:
            record = MatchRecord.load(match_dir)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(
                self, "Refetch failed",
                f"Could not reload match record:\n{exc}",
            )
            return
        record.final_standings = final_standings_from_gemini(parsed)
        if record.completed_at is None:
            record.completed_at = datetime.now().isoformat()
        try:
            png = generate_table(record)
            (match_dir / "table.png").write_bytes(png)
        except Exception:
            logger.exception("Failed to regenerate table after refetch")
        try:
            record.save(match_dir)
        except Exception as exc:
            QMessageBox.critical(
                self, "Refetch failed",
                f"Could not save match record:\n{exc}",
            )
            return
        self.set_record(record)
        self.recordEdited.emit(match_id)

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
        match_dir = self._matches_dir / record.match_id
        self._race_detail.set_race(race, record.settings, match_dir)
        self._stack.setCurrentIndex(1)

    def _on_refetch_placements(self, race_number: int) -> None:
        record = self._current_record
        if record is None:
            return
        thread = self._refetch_placements_thread
        if thread is not None and thread.isRunning():
            return
        race = next(
            (r for r in record.races if r.race_number == race_number),
            None,
        )
        if race is None:
            return
        match_dir = self._matches_dir / record.match_id
        race_dir = match_dir / f"race_{race_number:02d}"
        frame_paths = sorted(race_dir.glob("placement_*.png"))
        if not frame_paths:
            QMessageBox.warning(
                self, "Regenerate failed",
                f"No placement_*.png frames in {race_dir.name}.",
            )
            return
        if not load_api_key():
            QMessageBox.warning(
                self, "Regenerate failed",
                "No Gemini API key configured. Set one in the Settings tab.",
            )
            return

        confirm = _RefetchPlacementsConfirmDialog(
            frame_paths, race_number, race.track_name, parent=self,
        )
        if confirm.exec() != QDialog.DialogCode.Accepted:
            return

        import cv2
        frames: list = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        if not frames:
            QMessageBox.critical(
                self, "Regenerate failed",
                f"Could not read any placement frames in {race_dir.name}.",
            )
            return

        match_id = record.match_id
        teams_setting = (
            record.settings.teams if record.settings is not None else None
        )
        thread = _RefetchPlacementsThread(
            frames, race_number, race_dir, parent=self,
            teams_setting=teams_setting,
        )
        thread.finished.connect(
            lambda parsed: self._on_refetch_placements_finished(
                match_id, race_number, parsed,
            )
        )
        self._refetch_placements_thread = thread
        self._timeline.set_refetch_placements_in_progress(race_number)
        thread.start()

    def _on_refetch_placements_finished(
        self, match_id: str, race_number: int, parsed: object,
    ) -> None:
        self._timeline.set_refetch_placements_in_progress(None)
        if not isinstance(parsed, dict):
            QMessageBox.warning(
                self, "Regenerate failed",
                "Gemini did not return a usable response. See "
                "gemini_results.txt in the race folder for details.",
            )
            return
        match_dir = self._matches_dir / match_id
        try:
            record = MatchRecord.load(match_dir)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(
                self, "Regenerate failed",
                f"Could not reload match record:\n{exc}",
            )
            return
        idx = next(
            (i for i, r in enumerate(record.races) if r.race_number == race_number),
            None,
        )
        if idx is None:
            QMessageBox.warning(
                self, "Regenerate failed",
                f"Race {race_number} no longer exists in this match.",
            )
            return
        try:
            mode, teams, placements = race_fields_from_gemini(parsed)
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(
                self, "Regenerate failed",
                f"Could not interpret Gemini response:\n{exc}",
            )
            return
        existing = record.races[idx]
        record.races[idx] = dataclasses.replace(
            existing,
            mode=mode or existing.mode,
            placements=placements,
            teams=teams if teams else None,
        )
        try:
            record.save(match_dir)
        except Exception as exc:
            QMessageBox.critical(
                self, "Regenerate failed",
                f"Could not save match record:\n{exc}",
            )
            return
        self.set_record(record)
        self.recordEdited.emit(match_id)


class MatchHistoryView(QWidget):
    """Full history screen: match list on the left, detail pane on the right."""

    backRequested = Signal()

    def __init__(
        self,
        matches_dir: Path = DEFAULT_MATCHES_DIR,
        state_machine: "GameStateMachine | None" = None,
    ) -> None:
        super().__init__()
        self._matches_dir = matches_dir
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
        self._list.setFixedWidth(230)
        self._list.setStyleSheet(
            "QListWidget { background-color: #151515; border: 1px solid #2a2a2a; }"
            " QListWidget::item { padding: 6px; border-bottom: 1px solid #222; }"
            " QListWidget::item:selected { background-color: #2a3b4a; color: #fff; }"
        )
        self._list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._list.currentRowChanged.connect(self._on_row_changed)
        split.addWidget(self._list)

        self._detail = MatchDetailView(matches_dir=self._matches_dir)
        self._detail.recordEdited.connect(lambda _mid: self.refresh())
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
        """Re-scan ``matches_dir`` and rebuild the match list."""
        previous_id = None
        current_row = self._list.currentRow()
        if 0 <= current_row < len(self._records):
            previous_id = self._records[current_row].match_id

        self._records = list_matches(self._matches_dir)
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
        match_path = self._matches_dir / live_id / MATCH_FILE
        try:
            mtime = match_path.stat().st_mtime
        except OSError:
            return
        live_status = self._live_status_text()
        if mtime == self._last_live_mtime and live_status == self._last_live_status:
            return
        try:
            record = MatchRecord.load(self._matches_dir / live_id)
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
