"""Canonical list of Mario Kart track names."""

from pathlib import Path
import difflib
import re

TRACK_ICONS_DIR = Path(__file__).resolve().parents[3] / "assets" / "track_icons"

TRACK_NAMES: tuple[str, ...] = (
    "Acorn Heights",
    "Airship Fortress",
    "Boo Cinema",
    "Bowser's Castle",
    "Cheep Cheep Falls",
    "Choco Mountain",
    "Crown City",
    "Dandelion Depths",
    "Desert Hills",
    "Dino Dino Jungle",
    "DK Pass",
    "DK Spaceport",
    "Dry Bones Burnout",
    "Faraway Oasis",
    "Great ? Block Ruins",
    "Koopa Troopa Beach",
    "Mario Bros. Circuit",
    "Mario Circuit",
    "Moo Moo Meadows",
    "Peach Beach",
    "Peach Stadium",
    "Rainbow Road",
    "Salty Salty Speedway",
    "Shy Guy Bazaar",
    "Sky-High Sundae",
    "Starview Peak",
    "Toad's Factory",
    "Wario Shipyard",
    "Wario Stadium",
    "Whistlestop Summit",
    # Ver. 1.8.0 (September 2026).
    "SNES Mario Circuit 1",
    "SNES Mario Circuit 2",
    "SNES Mario Circuit 3",
    "SNES Ghost Valley 1",
    "SNES Ghost Valley 2",
    "SNES Ghost Valley 3",
    "SNES Choco Island 1",
    "SNES Choco Island 2",
    "SNES Vanilla Lake 1",
    "SNES Koopa Beach 1",
    # Ver. 1.9.0 (November 2026).
    "Airship Fortress 2",
    "Desert Hills 2",
    "Salty Salty Speedway 2",
    "DK Pass 2",
    "Koopa Troopa Beach 2",
    "Shy Guy Bazaar 2",
    "Mario Bros. Circuit 2",
    "Wario Stadium 2",
)


def match_track_name(candidate: str, cutoff: float = .85) -> tuple[str, float] | None:
    """Match OCR text without changing a course's number or SNES identity."""
    candidate = " ".join(candidate.split()).casefold()
    numbers = re.findall(r"\d+", candidate)
    names = [name for name in TRACK_NAMES
             if re.findall(r"\d+", name) == numbers
             and name.startswith("SNES ") == candidate.startswith("snes ")]
    matches = difflib.get_close_matches(candidate, [n.casefold() for n in names],
                                        n=1, cutoff=cutoff)
    if not matches:
        return None
    name = next(n for n in names if n.casefold() == matches[0])
    return name, difflib.SequenceMatcher(None, candidate, matches[0]).ratio()


def course_text_candidates(lines: list[str]) -> list[str]:
    """Rejoin wrapped course banners before matching individual fragments."""
    merged: list[str] = []
    for line in lines:
        if merged:
            joined = merged[-1] + " " + line
            match = match_track_name(joined)
            if match and (match[1] == 1 or line.isdigit()):
                merged[-1] = joined
                continue
        merged.append(line)
    candidates = list(merged)
    for a, b in zip(merged, merged[1:]):
        candidates.extend((a + " " + b, a + b))
    return candidates

TRACK_IMAGES: dict[str, str] = {
    "Acorn Heights": "MKWorld_Acorn_Heights_Icon.png",
    "Airship Fortress": "MKWd_Airship_Fortress_Icon.png",
    "Boo Cinema": "MKWorld_Boo_Cinema_icon.png",
    "Bowser's Castle": "MKWorld_Bowsers_Castle_icon.png",
    "Cheep Cheep Falls": "MKWorld_Cheep_Cheep_Falls_icon.png",
    "Choco Mountain": "MKWorld_Choco_Mountain_icon.png",
    "Crown City": "MKWd_Crown_City_Icon.png",
    "Dandelion Depths": "MKWorld_Dandelion_Depths_icon.png",
    "Desert Hills": "MKWd_Desert_Hills_Icon.png",
    "Dino Dino Jungle": "Dino_Dino_Jungle_Mario_Kart_World.png",
    "DK Pass": "MKWd_DK_Pass_Icon.png",
    "DK Spaceport": "MKWd_DK_Spaceport_Icon.png",
    "Dry Bones Burnout": "MKWorld_Dry_Bones_Burnout_icon.png",
    "Faraway Oasis": "MKWd_Faraway_Oasis_Icon.png",
    "Great ? Block Ruins": "MKWorld_Question_Ruins_icon.png",
    "Koopa Troopa Beach": "MKWd_Koopa_Troopa_Beach_Icon.png",
    "Mario Bros. Circuit": "MKWd_Mario_Bros_Circuit_Icon.png",
    "Mario Circuit": "MKWorld_Mario_Circuit_icon.png",
    "Moo Moo Meadows": "MKWorld_Moo_Moo_Meadows_icon.png",
    "Peach Beach": "Peach-Beach-MarioKartWorld.jpg",
    "Peach Stadium": "MKWorld_Peach_Stadium_Icon.png",
    "Rainbow Road": "MKWorld_Rainbow_Road_icon.png",
    "Salty Salty Speedway": "Salty_Salty_Speedway_Mario_Kart_World.jpg",
    "Shy Guy Bazaar": "MKWd_Shy_Guy_Bazaar_Icon.png",
    "Sky-High Sundae": "MKWd_Sky-High_Sundae_Icon.png",
    "Starview Peak": "MKWd_Starview_Peak_Icon.png",
    "Toad's Factory": "MKWorld_Toads_Factory_icon.png",
    "Wario Shipyard": "MKWd_Wario_Shipyard_Icon.png",
    "Wario Stadium": "MKWd_Wario_Stadium_Icon.png",
    "Whistlestop Summit": "MKWd_Whistlestop_Summit_Icon.png",
}
