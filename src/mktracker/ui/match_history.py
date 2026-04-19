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
from PySide6.QtGui import QColor, QPixmap
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
    MATCH_FILE,
    FinalStandings,
    MatchRecord,
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

    def __init__(self, race: RaceRecord, *, live: bool = False) -> None:
        super().__init__()
        self._live = live
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
        elif self._live:
            badge = QLabel("Rank …")
            badge.setStyleSheet(
                "QLabel { background-color: #333; color: #aaa; font-style: italic;"
                " padding: 2px 8px; border-radius: 10px; }"
            )
            badge.setToolTip("Awaiting Gemini rank result")
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
                self._layout.addWidget(_RaceCard(race, live=live))

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

        self._detail = MatchDetailView()
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
