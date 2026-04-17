"""Match history UI.

Shows a list of previously played matches (loaded from ``debug_frames/``) on
the left and a scrollable per-race timeline on the right.  The detail widget
accepts any :class:`MatchRecord`, so it can also be reused in the future to
display the currently-running match.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mktracker.detection.tracks import TRACK_ICONS_DIR, TRACK_IMAGES
from mktracker.match_record import (
    DEFAULT_DEBUG_DIR,
    FinalStandings,
    MatchRecord,
    RaceRecord,
    TeamGroup,
    list_matches,
)

logger = logging.getLogger(__name__)

_ICON_SIZE = 96
_ICON_CACHE: dict[str, QPixmap] = {}


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


def _format_timestamp(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _summary_line(record: MatchRecord) -> str:
    s = record.settings
    race_count = len(record.races) or s.race_count
    parts = [s.cc_class, f"{race_count} races", s.teams]
    if record.final_standings is None and record.completed_at is None:
        parts.append("partial")
    return " · ".join(parts)


class _RaceCard(QFrame):
    """One race in the timeline: icon + track name + placements."""

    def __init__(self, race: RaceRecord) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_RaceCard { background-color: #1a1a1a; border: 1px solid #2a2a2a;"
            " border-radius: 6px; }"
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

        mid.addLayout(title_row)

        placements_widget = self._build_placements(race)
        if placements_widget is not None:
            mid.addWidget(placements_widget)

        outer.addLayout(mid, stretch=1)

    def _build_placements(self, race: RaceRecord) -> QWidget | None:
        if race.teams:
            return self._build_team_placements(race.teams)
        if race.placements:
            return self._build_solo_placements(race)
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


class MatchDetailView(QScrollArea):
    """Scrollable timeline of races for a single match.

    Exposes a single :meth:`set_record` entry point so the same widget can
    later be reused for the live, in-progress match.
    """

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

    def set_record(self, record: MatchRecord) -> None:
        self._clear()

        header = self._build_header(record)
        self._layout.addWidget(header)

        if not record.races:
            note = QLabel("No races recorded yet.")
            note.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 10px; }")
            self._layout.addWidget(note)
        else:
            for race in sorted(record.races, key=lambda r: r.race_number):
                self._layout.addWidget(_RaceCard(race))

        if record.final_standings is not None:
            self._layout.addWidget(_FinalStandingsCard(record.final_standings))

        self._layout.addStretch()

    def _build_header(self, record: MatchRecord) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #14181f; border: 1px solid #232a33;"
            " border-radius: 6px; }"
            " QLabel { color: #ddd; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        started = _format_timestamp(record.started_at)
        completed = _format_timestamp(record.completed_at)
        title = QLabel(f"Match {record.match_id}")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #fff; }")
        layout.addWidget(title)

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


class MatchHistoryView(QWidget):
    """Full history screen: match list on the left, detail pane on the right."""

    backRequested = Signal()

    def __init__(self, debug_dir: Path = DEFAULT_DEBUG_DIR) -> None:
        super().__init__()
        self._debug_dir = debug_dir
        self._records: list[MatchRecord] = []

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

        self._detail = MatchDetailView()
        split.addWidget(self._detail, stretch=1)

        root.addLayout(split, stretch=1)

    def refresh(self) -> None:
        """Re-scan ``debug_dir`` and rebuild the match list."""
        previous_id = None
        current_row = self._list.currentRow()
        if 0 <= current_row < len(self._records):
            previous_id = self._records[current_row].match_id

        self._records = list_matches(self._debug_dir)
        self._list.clear()
        for rec in self._records:
            started = _format_timestamp(rec.started_at)
            summary = _summary_line(rec)
            item = QListWidgetItem(f"{started}\n{summary}")
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

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._records):
            self._detail.set_record(self._records[row])
        else:
            self._detail.clear()
