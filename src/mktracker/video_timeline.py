"""Persistent video race timeline, independent of playback and match setup."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from pathlib import Path


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


@dataclass
class VideoRace:
    start: float
    track: str = "Unknown course"
    finish: float | None = None
    results: float | None = None
    placements: list[list] = field(default_factory=list)
    review_note: str = ""
    placement_evidence: dict[str, dict] = field(default_factory=dict)
    preview_players: list[str] = field(default_factory=list)
    highlighted_rank: int | None = None


@dataclass
class VideoTimeline:
    source: str
    size: int
    modified_ns: int
    duration: float
    races: list[VideoRace] = field(default_factory=list)
    analyzed_until: float = 0.0
    complete: bool = False

    @classmethod
    def create(cls, source: str, duration: float) -> VideoTimeline:
        path = Path(source).resolve()
        stat = path.stat()
        return cls(str(path), stat.st_size, stat.st_mtime_ns, duration)

    @property
    def cache_path(self) -> Path:
        key = hashlib.sha256(f"{self.source}|{self.size}|{self.modified_ns}".encode()).hexdigest()[:20]
        return Path("video_timelines") / f"{key}.json"

    def save(self) -> None:
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        # Sync/indexing software can briefly hold an existing file on Windows.
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.05 * (attempt + 1))

    @classmethod
    def load(cls, source: str, duration: float) -> VideoTimeline:
        fresh = cls.create(source, duration)
        try:
            data = json.loads(fresh.cache_path.read_text(encoding="utf-8"))
            data["races"] = [VideoRace(**race) for race in data["races"]]
            return cls(**data)
        except (OSError, ValueError, TypeError, KeyError):
            return fresh
