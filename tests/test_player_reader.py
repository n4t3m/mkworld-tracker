"""Tests for PlayerReader against debug frame screenshots."""
from __future__ import annotations

import glob
import os

import cv2
import pytest

from mktracker.detection.player_reader import PlayerReader

DEBUG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "matches",
    "20260329_110511",
)

# The 12 players present in every screenshot (races 1-10).
# Races 11-12 have roster changes (hawkey leaves, Axelfy joins).
EXPECTED_PLAYERS = [
    "nthn",
    "Spritz",
    "Kod49",
    "Azusa",
    "hawkey",
    "choko",
    "RK sombluz",
    "RK AciiD",
    "RK Pero",
    "RK Mimosa",
    "RKCASTILLA",
    "RK Jamal",
]

PLAYER_FILES = sorted(glob.glob(os.path.join(DEBUG_DIR, "race_*_players.png")))


def _name_matches(detected: str, expected: str) -> bool:
    """Fuzzy match allowing substring containment or up to 2 character differences.

    Accounts for common OCR errors like 'o' -> '6', 'o' -> 'e', etc.
    """
    d = detected.lower().replace(" ", "")
    e = expected.lower().replace(" ", "")
    # Exact or substring
    if d in e or e in d or d == e:
        return True
    # Allow up to 2 character edits (handles 'choko'->'choke', 'pero'->'per6')
    if abs(len(d) - len(e)) > 2:
        return False
    diffs = sum(a != b for a, b in zip(d, e))
    diffs += abs(len(d) - len(e))
    return diffs <= 2


@pytest.fixture(scope="module")
def reader() -> PlayerReader:
    return PlayerReader()


@pytest.mark.skipif(
    not PLAYER_FILES,
    reason="Debug frames not found; run a live session first",
)
class TestPlayerReader:
    """Verify that all 12 player names are detected from each screenshot."""

    @pytest.mark.parametrize(
        "image_path",
        PLAYER_FILES,
        ids=[os.path.basename(f).replace("_players.png", "") for f in PLAYER_FILES],
    )
    def test_detects_12_players(self, reader: PlayerReader, image_path: str) -> None:
        frame = cv2.imread(image_path)
        assert frame is not None, f"Could not load {image_path}"

        players = reader.read_players(frame, teams=True)
        names = [p.name for p in players]

        assert len(names) == 12, (
            f"Expected 12 players, got {len(names)}: {names}"
        )

    @pytest.mark.parametrize(
        "image_path",
        PLAYER_FILES,
        ids=[os.path.basename(f).replace("_players.png", "") for f in PLAYER_FILES],
    )
    def test_each_expected_player_found(
        self, reader: PlayerReader, image_path: str
    ) -> None:
        frame = cv2.imread(image_path)
        players = reader.read_players(frame, teams=True)
        names = [p.name for p in players]

        # Races 11-12 have a roster change: hawkey -> Axelfy
        basename = os.path.basename(image_path)
        race_num = int(basename.split("_")[1])
        expected = list(EXPECTED_PLAYERS)
        if race_num >= 11:
            expected.remove("hawkey")

        missing = []
        for exp in expected:
            if not any(_name_matches(det, exp) for det in names):
                missing.append(exp)

        assert not missing, (
            f"Players not found: {missing}\nDetected: {names}"
        )

    def test_no_empty_names(self, reader: PlayerReader) -> None:
        """No player should have a blank or whitespace-only name."""
        for image_path in PLAYER_FILES:
            frame = cv2.imread(image_path)
            players = reader.read_players(frame, teams=True)
            for p in players:
                assert p.name.strip(), (
                    f"Empty name in {os.path.basename(image_path)}"
                )
