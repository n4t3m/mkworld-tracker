"""Lorenzi-style MK match result table image generator using Pillow.

Replicates the visual style of https://gb2.hlorenzi.com/table
(the ``drawTableDefault`` renderer in table.js) without requiring a browser.

Fonts (Roboto variable + RubikMonoOne) are downloaded from the google/fonts
GitHub repo on first use and cached in ``src/mktracker/assets/fonts/``.
"""
from __future__ import annotations

import colorsys
import io
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .match_record import MatchRecord

# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------

_FONTS_DIR = Path(__file__).parent / "assets" / "fonts"

_FONT_SOURCES: dict[str, str] = {
    "Roboto-variable.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/roboto/"
        "Roboto%5Bwdth%2Cwght%5D.ttf"
    ),
    "RubikMonoOne-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/rubikmonoone/"
        "RubikMonoOne-Regular.ttf"
    ),
    "NotoSansJP-variable.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/notosansjp/"
        "NotoSansJP%5Bwght%5D.ttf"
    ),
}


def _ensure_font(filename: str) -> Path:
    dest = _FONTS_DIR / filename
    if dest.exists():
        return dest
    _FONTS_DIR.mkdir(parents=True, exist_ok=True)
    url = _FONT_SOURCES[filename]
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download font '{filename}': {exc}.\n"
            f"Place it manually in {_FONTS_DIR}"
        ) from exc
    return dest


def _bold(roboto: Path, size: float) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(roboto), size=max(1, round(size)))
    try:
        font.set_variation_by_axes([900, 100])  # wght=900 (black), wdth=100 (normal)
    except Exception:
        pass
    return font


def _mono(rubik: Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(rubik), size=max(1, round(size)))


def _needs_cjk(text: str) -> bool:
    """Return True if text contains characters outside Roboto's coverage."""
    for c in text:
        cp = ord(c)
        if (0x3000 <= cp <= 0x9FFF     # CJK, Hiragana, Katakana
                or 0xF900 <= cp <= 0xFAFF   # CJK compatibility
                or 0xFF00 <= cp <= 0xFFEF): # Halfwidth/fullwidth forms
            return True
    return False


def _name_font(
    roboto: Path, noto_jp: Path, text: str, size: float
) -> ImageFont.FreeTypeFont:
    if _needs_cjk(text):
        font = ImageFont.truetype(str(noto_jp), size=max(1, round(size)))
        try:
            font.set_variation_by_axes([700])  # wght=700 bold
        except Exception:
            pass
        return font
    return _bold(roboto, size)


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _Clan:
    tag: str | None
    players: list[dict]  # {name: str, total_score: int, ranking: int}
    penalty: int = 0
    hue: float = 0.0
    saturation: float = 0.0
    y: float = 0.0
    h: float = 0.0
    score: int = 0
    ranking: int = 0


# ---------------------------------------------------------------------------
# Color helpers (ported from table.js, extended for no-tag clans)
# ---------------------------------------------------------------------------

def _clan_hsv(
    tag: str | None,
    used_hashes: list[int],
    seed: str = "",
) -> tuple[float, float, float]:
    """Return (hue, saturation, value) for a clan.

    When *tag* is None (no-teams match), *seed* (e.g. the match_id) is used
    so each match still gets a distinct, consistent, vibrant colour.
    """
    color_key = tag if tag is not None else seed
    h = 122
    for c in color_key.lower():
        if c not in (" ", "_"):
            h += ord(c) * 12
    while any(((abs(h - u) + 256) % 256) < 10 for u in used_hashes):
        h += 10
    h %= 256
    used_hashes.append(h)
    hue = h / 256
    if tag is None:
        sat = 0.75          # always vibrant for untagged FFA
    elif 165 <= h <= 200:
        sat = 0.5
    elif 150 <= h <= 215:
        sat = 0.7
    else:
        sat = 0.9
    return hue, sat, 1.0


def _hsv2rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return round(r * 255), round(g * 255), round(b * 255)


def _blend(
    base: tuple[int, int, int],
    overlay: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Alpha-composite *overlay* over *base* at *alpha* opacity."""
    return (
        round(base[0] * (1 - alpha) + overlay[0] * alpha),
        round(base[1] * (1 - alpha) + overlay[1] * alpha),
        round(base[2] * (1 - alpha) + overlay[2] * alpha),
    )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _build_clans(record: MatchRecord) -> list[_Clan]:
    fs = record.final_standings
    if fs is None:
        return []

    if fs.teams and len(fs.teams) >= 2:
        return [
            _Clan(
                tag=t.name or "?",
                players=[
                    {"name": p.name, "total_score": p.score or 0}
                    for p in t.players
                    if p.score is not None
                ],
            )
            for t in fs.teams
        ]

    return [
        _Clan(
            tag=None,
            players=[
                {"name": p.name, "total_score": p.score or 0}
                for p in fs.players
                if p.score is not None
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Decorative dot pattern (Lorenzi-style halftone on clan backgrounds)
# ---------------------------------------------------------------------------

def _draw_dots(
    img: Image.Image,
    clan_y: float,
    clan_h: float,
    total_w: int,
    dot_rgb: tuple[int, int, int],
) -> None:
    """Overlay the subtle polka-dot texture used in drawTableDefault."""
    center_y = clan_y + clan_h / 2
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw  = ImageDraw.Draw(overlay)

    alternate = True
    dy = 0.0
    while dy < clan_h / 2 + 30:
        alternate = not alternate
        x_off = 7.5 if alternate else 0.0
        # alpha follows a half-sine: strong at centre, zero at edges
        t     = math.sin(math.pi * dy / clan_h) if clan_h > 0 else 0.0
        alpha = round(0.15 * t * 255)
        if alpha >= 3:
            fill = (*dot_rgb, alpha)
            cx = x_off
            while cx < total_w + 14:
                r = 4
                candidates = [center_y + dy] if dy == 0 else [center_y + dy, center_y - dy]
                for cy_dot in candidates:
                    # Clip dots to the clan background rectangle
                    if clan_y <= cy_dot <= clan_y + clan_h:
                        ov_draw.ellipse(
                            [(cx - r, cy_dot - r), (cx + r, cy_dot + r)],
                            fill=fill,
                        )
                cx += 14
        dy += 10

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"),
              (0, 0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_table(record: MatchRecord) -> bytes:
    """Render a Lorenzi-style match result table and return PNG bytes."""
    clans = _build_clans(record)
    if not clans:
        raise ValueError("No final standings in record — cannot generate table.")

    for clan in clans:
        clan.players.sort(key=lambda p: p["total_score"], reverse=True)

    for clan in clans:
        clan.score = sum(p["total_score"] for p in clan.players) + clan.penalty
    clans.sort(key=lambda c: c.score, reverse=True)

    # Colours — pass match_id as seed so no-tag clans still get vibrant hues
    used_hashes: list[int] = []
    for clan in clans:
        h, s, _ = _clan_hsv(clan.tag, used_hashes, seed=record.match_id)
        clan.hue, clan.saturation = h, s

    # Global player rankings
    all_players = [p for c in clans for p in c.players]
    all_players.sort(key=lambda p: p["total_score"], reverse=True)
    for i, p in enumerate(all_players):
        if i > 0 and p["total_score"] == all_players[i - 1]["total_score"]:
            p["ranking"] = all_players[i - 1]["ranking"]
        else:
            p["ranking"] = i + 1

    for i, clan in enumerate(clans):
        if i > 0 and clan.score == clans[i - 1].score:
            clan.ranking = clans[i - 1].ranking
        else:
            clan.ranking = i + 1

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    STANDARD_H    = 520.0
    HEADER_H      = STANDARD_H / 13     # ≈ 40
    CLAN_MARGIN_H = HEADER_H / 2        # ≈ 20
    PLAYER_H      = STANDARD_H / 14     # ≈ 37

    row_num      = sum(len(c.players) for c in clans)
    penalty_rows = sum(1 for c in clans if c.penalty != 0)

    TOTAL_W    = 860
    DIVIDER_H  = int(PLAYER_H * 0.5) if len(clans) > 1 else 0  # gap between teams
    TOTAL_H    = int(STANDARD_H + max(0, row_num + penalty_rows - 12) * PLAYER_H
                     + DIVIDER_H * max(0, len(clans) - 1))

    single_clan      = len(clans) == 1
    show_clan_ranks  = not single_clan

    if single_clan:
        # Full-width: player rows span almost the entire image.
        # A compact score panel sits on the right.
        SCORE_PANEL_W = 160.0
        LEFT_PAD      = 24.0
        PLAYER_X      = LEFT_PAD
        PLAYER_W      = TOTAL_W - LEFT_PAD - SCORE_PANEL_W
        CLAN_SCORE_X  = TOTAL_W - SCORE_PANEL_W / 2
        CLAN_NAME_X   = 0.0       # unused
        CLAN_RANK_X   = 0.0       # unused
    else:
        # Original Lorenzi column layout for multi-clan (team vs team).
        COL           = TOTAL_W / 60.0
        CLAN_RANK_W   = COL * 4
        CLAN_NAME_W   = COL * 16
        PLAYER_X      = CLAN_RANK_W + CLAN_NAME_W
        PLAYER_W      = COL * 20
        CLAN_SCORE_W  = COL * 20
        CLAN_SCORE_X  = PLAYER_X + PLAYER_W + CLAN_SCORE_W / 2
        CLAN_NAME_X   = CLAN_RANK_W + CLAN_NAME_W / 2
        CLAN_RANK_X   = CLAN_RANK_W / 2

    # Sub-columns inside the player area (same proportions as table.js)
    PCOL     = PLAYER_W / 20.0

    px = 0.0
    PNAME_W  = PCOL * 10
    PNAME_X  = PNAME_W / 2
    px      += PNAME_W
    px      += PCOL * 3            # flag column skipped (no flag data)
    PSCORE_W = PCOL * 4
    PSCORE_X = px + PSCORE_W / 2
    px      += PSCORE_W
    PRANK_W  = PCOL * 3
    PRANK_X  = px + PRANK_W / 2

    # Absorb the flag column into the name column (matches JS when no flags)
    PNAME_X += PCOL * 1.5
    PNAME_W += PCOL * 3

    # Clan vertical layout
    for clan in clans:
        clan.h = max(1, len(clan.players) + (1 if clan.penalty else 0)) * PLAYER_H

    total_ph = sum(c.h for c in clans)
    gap_total = DIVIDER_H * max(0, len(clans) - 1)
    extra    = (TOTAL_H - HEADER_H - total_ph - gap_total) / len(clans)
    for clan in clans:
        clan.h += extra

    cy = HEADER_H
    for i, clan in enumerate(clans):
        if i > 0:
            cy += DIVIDER_H
        clan.y = cy + CLAN_MARGIN_H
        cy    += clan.h
        clan.h -= CLAN_MARGIN_H * 2

    # ------------------------------------------------------------------
    # Fonts
    # ------------------------------------------------------------------
    roboto   = _ensure_font("Roboto-variable.ttf")
    rubik    = _ensure_font("RubikMonoOne-Regular.ttf")
    noto_jp  = _ensure_font("NotoSansJP-variable.ttf")

    # ------------------------------------------------------------------
    # Draw (pure RGB — all colours pre-computed, no alpha compositing)
    # ------------------------------------------------------------------
    img  = Image.new("RGB", (TOTAL_W, TOTAL_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Header — use match start time if available, otherwise today
    _raw_date = record.started_at
    if isinstance(_raw_date, str):
        _raw_date = datetime.fromisoformat(_raw_date)
    date = _raw_date if _raw_date else datetime.now()
    months   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    date_str = f"{date.day} {months[date.month - 1]} {date.year}"
    draw.text(
        (TOTAL_W / 2, HEADER_H / 2), date_str,
        fill=(255, 255, 255),
        font=_bold(roboto, HEADER_H * 0.65),
        anchor="mm",
    )

    for clan in clans:
        bg_rgb  = _hsv2rgb(clan.hue, clan.saturation, 0.72)
        dot_rgb = _hsv2rgb(clan.hue, max(0.0, clan.saturation - 0.1), 0.35)
        row_bg  = _blend(bg_rgb, (255, 255, 255), 0.58)  # light rows against darker bg

        cy0 = int(clan.y - CLAN_MARGIN_H)
        cy1 = int(clan.y + clan.h + CLAN_MARGIN_H)
        draw.rectangle([(0, cy0), (TOTAL_W, cy1)], fill=bg_rgb)

        # Dot texture
        _draw_dots(img, clan.y - CLAN_MARGIN_H, clan.h + CLAN_MARGIN_H * 2,
                   TOTAL_W, dot_rgb)
        draw = ImageDraw.Draw(img)   # re-acquire after paste

        mid_y = clan.y + clan.h / 2

        # Clan tag + total score
        if not single_clan and len(clan.players) > 1:
            score_sz = min(clan.h * 1.5, 100.0)
            draw.text(
                (CLAN_SCORE_X, mid_y), str(clan.score),
                fill=(0, 0, 0),
                font=_mono(rubik, score_sz),
                anchor="mm",
            )

            if not single_clan:
                if clan.tag:
                    draw.text(
                        (CLAN_NAME_X, mid_y), clan.tag,
                        fill=(0, 0, 0),
                        font=_bold(roboto, score_sz),
                        anchor="mm",
                    )
                if show_clan_ranks:
                    r  = clan.ranking
                    rs = "1st" if r==1 else "2nd" if r==2 else "3rd" if r==3 else f"{r}th"
                    draw.text(
                        (CLAN_RANK_X, mid_y), rs,
                        fill=(0, 0, 0),
                        font=_bold(roboto, PLAYER_H * 0.95 * 0.6),
                        anchor="mm",
                    )

        # Player rows
        n = len(clan.players)
        for p_idx, player in enumerate(clan.players):
            offset  = (-n / 2 + p_idx - (0.5 if clan.penalty else 0)) * PLAYER_H
            row_y   = clan.y + clan.h / 2 + offset
            row_mid = row_y + PLAYER_H / 2

            draw.rounded_rectangle(
                [(int(PLAYER_X), int(row_y + 2)),
                 (int(PLAYER_X + PLAYER_W), int(row_y + PLAYER_H - 4))],
                radius=5,
                fill=row_bg,
            )

            draw.text(
                (PLAYER_X + PNAME_X, row_mid), player["name"],
                fill=(0, 0, 0),
                font=_name_font(roboto, noto_jp, player["name"], PLAYER_H * 0.65),
                anchor="mm",
            )
            draw.text(
                (PLAYER_X + PSCORE_X, row_mid), str(player["total_score"]),
                fill=(0, 0, 0),
                font=_mono(rubik, PLAYER_H * 0.65),
                anchor="mm",
            )

            r  = player["ranking"]
            rs = "1st" if r==1 else "2nd" if r==2 else "3rd" if r==3 else f"{r}th"
            draw.text(
                (PLAYER_X + PRANK_X, row_mid), rs,
                fill=(0, 0, 0),
                font=_bold(roboto, PLAYER_H * 0.65 * 0.6),
                anchor="mm",
            )

        # Penalty row
        if clan.penalty != 0:
            offset  = (n / 2 - 0.5) * PLAYER_H
            row_y   = clan.y + clan.h / 2 + offset
            row_mid = row_y + PLAYER_H / 2
            draw.rounded_rectangle(
                [(int(PLAYER_X + 30), int(row_y + 7)),
                 (int(PLAYER_X + PLAYER_W - 30), int(row_y + PLAYER_H - 5))],
                radius=5, fill=row_bg,
            )
            pf = _bold(roboto, PLAYER_H * 0.45)
            draw.text((PLAYER_X + PNAME_X,  row_mid), "Penalty", fill=(0,0,0), font=pf, anchor="mm")
            draw.text((PLAYER_X + PSCORE_X, row_mid), str(clan.penalty), fill=(0,0,0), font=pf, anchor="mm")

    # Clan dividers + score-differential labels (multi-clan only)
    for i in range(len(clans) - 1):
        c      = clans[i]
        gap_y  = c.y + c.h + CLAN_MARGIN_H          # top of the gap
        mid_y  = int(gap_y + DIVIDER_H / 2)          # centre of the gap
        draw.rectangle([(0, mid_y - 1), (TOTAL_W, mid_y + 1)],
                       fill=_blend((0, 0, 0), (255, 255, 255), 0.25))
        diff = c.score - clans[i + 1].score
        draw.text(
            (CLAN_SCORE_X, mid_y), f"+{diff}",
            fill=(255, 255, 255),
            font=_bold(roboto, PLAYER_H * 0.65),
            anchor="mm",
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
