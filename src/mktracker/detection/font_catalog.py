"""Enumerate the configured font's complete Unicode cmap and visible glyphs."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


@dataclass
class FontGlyph:
    char: str
    glyph_name: str
    mask: np.ndarray
    ink_top: float
    advance: float
    aliases: tuple[str, ...] = ()
    ink_left: float = 0

    @property
    def aspect(self) -> float:
        return self.mask.shape[1] / self.mask.shape[0]


class FontCatalog:
    def __init__(self, path: str, size: int = 64, *, phase: float = 0, build_groups: bool = True) -> None:
        self.path, self.size = path, size
        self.font = ImageFont.truetype(path, size)
        with TTFont(path, lazy=True) as tt:
            self.cmap = dict(tt.getBestCmap() or {})
        self.glyphs: dict[str, FontGlyph] = {}
        self.layout_characters: dict[str, str] = {}
        aliases = defaultdict(list)
        for codepoint, glyph_name in sorted(self.cmap.items()):
            char = chr(codepoint)
            if unicodedata.category(char) == "Cc":
                self.layout_characters[char] = "control"
                continue
            x0, y0, x1, y1 = self.font.getbbox(char)
            image = Image.new("L", (max(1, x1-x0)+40, max(1, y1-y0)+40), 0)
            ImageDraw.Draw(image).text((20-x0+phase, 20-y0), char, font=self.font, fill=255)
            mask = np.array(image)
            points = cv2.findNonZero(mask)
            if points is None:
                self.layout_characters[char] = "whitespace" if char.isspace() else "empty"
                continue
            x, y, w, h = cv2.boundingRect(points)
            tight = mask[y:y+h, x:x+w]
            glyph = FontGlyph(char, glyph_name, tight, float(y-20+y0), self.font.getlength(char))
            glyph.ink_left = float(x-20+x0)
            self.glyphs[char] = glyph
            key = (tight.shape, glyph.ink_top, hashlib.sha256(tight.tobytes()).digest())
            aliases[key].append(char)
        for characters in aliases.values():
            for char in characters:
                self.glyphs[char].aliases = tuple(characters)
        self.groups = self._visual_groups() if build_groups else [(c,) for c in self.glyphs]
        self.group_for = {char: index for index, group in enumerate(self.groups) for char in group}

    def _near_equal(self, a: str, b: str, threshold: float = .95) -> bool:
        first, second = self.glyphs[a], self.glyphs[b]
        if abs(np.log(first.mask.shape[0] / second.mask.shape[0])) > .12:
            return False
        if abs(np.log(first.aspect / second.aspect)) > .14 or abs(first.ink_top-second.ink_top) > 5:
            return False
        x = cv2.resize(first.mask, (48, 64), interpolation=cv2.INTER_AREA) > 127
        y = cv2.resize(second.mask, (48, 64), interpolation=cv2.INTER_AREA) > 127
        return 2*np.logical_and(x, y).sum()/max(1, x.sum()+y.sum()) >= threshold

    def _visual_groups(self) -> list[tuple[str, ...]]:
        groups = [{char} for char in self.glyphs]
        def merge(characters, threshold=.95):
            characters = set(characters) & self.glyphs.keys()
            touched = [group for group in groups if group & characters]
            if len(touched) < 2:
                return
            combined = set().union(*touched)
            # No transitive "almost the same" chains joining unlike letters.
            if not all(self._near_equal(a, b, threshold) for a in combined for b in combined):
                return
            for group in touched:
                groups.remove(group)
            groups.append(combined)
        for glyph in self.glyphs.values():
            merge(glyph.aliases)
        for char in self.glyphs:
            compatible = unicodedata.normalize("NFKC", char)
            if len(compatible) == 1:
                merge((char, compatible))
        # Group only genuine homoglyph families, never accent differences.
        # The bitmap check is mandatory even for these familiar confusables.
        for family in ("AΑА", "BΒВ", "CС", "EΕЕ", "HΗН", "IΙl", "KΚК", "MΜМ", "NΝ",
                       "OΟО", "PΡР", "TΤТ", "XΧХ", "YΥ", "aа", "cс", "eе", "oοо",
                       "pρр", "xх", "yу", "·•·"):
            merge(family)
        # Extend established homoglyph families to the same accent, while
        # keeping acute/grave/diaeresis/etc. distinct from one another.
        for _ in range(2):
            canonical = {char: min(group, key=lambda c: (not c.isascii(), ord(c))) for group in groups for char in group}
            accented = defaultdict(list)
            for char in self.glyphs:
                decomposed = unicodedata.normalize("NFD", char)
                if len(decomposed) > 1 and all(unicodedata.combining(c) for c in decomposed[1:]):
                    accented[canonical.get(decomposed[0], decomposed[0]), decomposed[1:]].append(char)
            for characters in accented.values():
                merge(characters, threshold=.93)
        def priority(char):
            return (not char.isascii(), ord(char))
        return sorted((tuple(sorted(group, key=priority)) for group in groups), key=lambda group: priority(group[0]))

    def summary(self) -> dict:
        groups = {g.aliases for g in self.glyphs.values() if len(g.aliases) > 1}
        return {"mapped_characters": len(self.cmap), "visible_characters": len(self.glyphs),
                "layout_characters": {f"U+{ord(c):04X}": reason for c, reason in self.layout_characters.items()},
                "identical_shape_groups": [list(group) for group in sorted(groups)],
                "visual_groups": len(self.groups),
                "homoglyph_groups": [list(group) for group in self.groups if len(group) > 1],
                "categories": {category: sum(unicodedata.category(c) == category for c in self.glyphs)
                               for category in sorted({unicodedata.category(c) for c in self.glyphs})}}
