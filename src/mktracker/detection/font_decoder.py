"""Shape recognition and multi-component segmentation for an entire game font."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import cv2
import numpy as np

from mktracker.detection.font_catalog import FontCatalog


@dataclass
class GlyphRead:
    char: str
    alternatives: tuple[str, ...]
    score: float
    margin: float
    reference: str
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    reference_height: float = 1
    reference_top: float = 0

    @property
    def verified(self) -> bool:
        return (self.score >= .97 and self.margin >= .006) or (self.score >= .86 and self.margin >= .035)


class FontDecoder:
    def __init__(self, path: str) -> None:
        self.catalog = FontCatalog(path)
        self.chars = []
        masks, heights, tops, aspects = [], [], [], []
        for size, phase in [(64, 0)]+[(size, phase) for size in (44, 32) for phase in (0, .25, .5, .75)]:
            catalog = self.catalog if size == 64 else FontCatalog(path, size, phase=phase, build_groups=False)
            for char, glyph in catalog.glyphs.items():
                binary = (glyph.mask > 127).astype(np.uint8)
                points = cv2.findNonZero(binary)
                if points is None:
                    continue
                x, y, w, h = cv2.boundingRect(points)
                self.chars.append(char)
                masks.append(cv2.resize(binary[y:y+h, x:x+w]*255, (32, 48), interpolation=cv2.INTER_AREA) > 127)
                heights.append(h*64/size)
                tops.append((glyph.ink_top+y)*64/size)
                aspects.append(w/h)
        self.masks = np.stack(masks)
        self.counts = self.masks.sum(axis=(1, 2))
        self.coarse = np.stack([cv2.resize(mask.astype(np.uint8)*255, (12, 18), interpolation=cv2.INTER_AREA) > 90
                                for mask in self.masks])
        self.coarse_counts = self.coarse.sum(axis=(1, 2))
        self.bands = self.masks.reshape(len(self.chars), 4, -1)
        self.band_counts = self.bands.sum(axis=2)
        self.heights = np.array(heights)
        self.tops = np.array(tops)
        self.aspects = np.array(aspects)
        self.families = np.array([self.catalog.group_for[c] for c in self.chars])
        self.indices = {c: i for i, c in enumerate(self.chars)}
        self._shape_cache = {}

    def _shape_scores(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        normalized = cv2.resize(mask.astype(np.uint8)*255, (32, 48), interpolation=cv2.INTER_AREA) > 127
        key = (h, w, np.packbits(normalized).tobytes())
        if key in self._shape_cache:
            return self._shape_cache[key]
        candidates = np.flatnonzero(np.abs(np.log((w/h)/self.aspects)) <= .35)
        scores = np.full(len(self.chars), -2.0)
        if not len(candidates):
            return scores
        # Shortlist at low resolution, then compare fine details/diacritics.
        # Multiple raster phases of an alias count separately here, so keep a
        # broad shortlist rather than trusting just the best coarse match.
        if len(candidates) > 128:
            small = cv2.resize(mask.astype(np.uint8)*255, (12, 18), interpolation=cv2.INTER_AREA) > 90
            overlap = np.logical_and(self.coarse[candidates], small).sum(axis=(1, 2))
            coarse_score = 2*overlap/np.maximum(1, self.coarse_counts[candidates]+small.sum())
            coarse_score -= .08*np.abs(np.log((w/h)/self.aspects[candidates]))
            candidates = candidates[np.argpartition(coarse_score, -128)[-128:]]
        intersection = np.logical_and(self.masks[candidates], normalized).sum(axis=(1, 2))
        dice = 2*intersection/np.maximum(1, self.counts[candidates]+normalized.sum())
        bands = normalized.reshape(4, -1)
        band_intersection = np.logical_and(self.bands[candidates], bands).sum(axis=2)
        denominator = self.band_counts[candidates]+bands.sum(axis=1)
        band_dice = 2*band_intersection/np.maximum(1, denominator)
        band_score = (band_dice*(denominator > 0)).sum(axis=1)/np.maximum(1, (denominator > 0).sum(axis=1))
        # Give small diacritics their own weight instead of letting the base
        # letter swamp the distinction between, for example, é and è.
        scores[candidates] = .65*dice+.35*band_score-.08*np.abs(np.log((w/h)/self.aspects[candidates]))
        if len(self._shape_cache) > 512:
            self._shape_cache.clear()
        self._shape_cache[key] = scores
        return scores

    def classify(self, mask: np.ndarray, *, top: float | None = None,
                 scale: float | None = None, origin: float | None = None) -> GlyphRead:
        binary = np.asarray(mask) > 127 if np.asarray(mask).dtype != bool else np.asarray(mask)
        points = cv2.findNonZero(binary.astype(np.uint8))
        if points is None:
            return GlyphRead("", (), 0, 0, "")
        x, y, w, h = cv2.boundingRect(points)
        scores = self._shape_scores(binary[y:y+h, x:x+w]).copy()
        if scale is not None:
            scores -= .12*np.minimum(2, np.abs(np.log(h/(self.heights*scale))))
        if origin is not None and scale is not None and top is not None:
            scores -= .15*np.minimum(1, np.abs(top+y-(origin+self.tops*scale))/max(1, 64*scale))
        # The runner-up must be a different visual family, not another code
        # point for an indistinguishable A/O/etc.
        best = int(np.argmax(scores))
        family = self.families[best]
        runner = int(np.argmax(np.where(self.families == family, -np.inf, scores)))
        alternatives = self.catalog.groups[family]
        return GlyphRead(alternatives[0], alternatives, float(scores[best]),
                         float(scores[best]-scores[runner]), self.chars[best], x, y, w, h,
                         float(self.heights[best]), float(self.tops[best]))

    def equivalent(self, first: str, second: str) -> bool:
        return first == second or (first in self.catalog.group_for and second in self.catalog.group_for
                                  and self.catalog.group_for[first] == self.catalog.group_for[second])

    def canonical(self, text: str) -> str:
        return "".join(self.catalog.groups[self.catalog.group_for[c]][0]
                       if c in self.catalog.group_for else c for c in text)

    def _text_tokens(self, reads: list[GlyphRead]) -> list[tuple[str, bool]]:
        if not reads:
            return []
        reliable = [r for r in reads if r.score >= .8] or reads
        scale = float(np.median([r.height/r.reference_height for r in reliable]))
        tokens = []
        for i, read in enumerate(reads):
            if i:
                previous = reads[i-1]
                first = self.catalog.glyphs[previous.reference]
                second = self.catalog.glyphs[read.reference]
                natural_gap = first.advance-first.ink_left-first.mask.shape[1]+second.ink_left
                kerning = (self.catalog.font.getlength(previous.reference+read.reference)
                           -first.advance-second.advance)
                actual_gap = read.left-previous.left-previous.width
                space_width = self.catalog.font.getlength(" ")*scale
                if actual_gap-(natural_gap+kerning)*scale > space_width*.6:
                    tokens.append((" ", previous.verified and read.verified))
            tokens.append((read.char, read.verified))
        return tokens

    def correct_words(self, image: np.ndarray, words: list[dict]) -> list[dict]:
        reads = self.segment(image)
        if not reads:
            return words
        tokens = self._text_tokens(reads)
        observed = "".join(c for c, _ in tokens)
        text = " ".join(w["text"] for w in sorted(words, key=lambda w: w["left"]))
        all_verified = all(r.verified for r in reads)
        if not text and not all_verified:
            return words
        changes = []
        unresolved = False
        for op, a, b, c, d in SequenceMatcher(None, self.canonical(text), self.canonical(observed), autojunk=False).get_opcodes():
            if op == "equal":
                # Keep the user's/OCR's code point for indistinguishable glyphs.
                changes.append(text[a:b])
            elif c < d and all(verified for _, verified in tokens[c:d]):
                changes.append(observed[c:d])
            elif c < d and b-a == d-c:
                changes.append("".join(tokens[c+i][0] if tokens[c+i][1] else text[a+i] for i in range(b-a)))
                unresolved |= any(not verified for _, verified in tokens[c:d])
            elif c == d and all_verified:
                continue
            else:
                changes.append(text[a:b])
                unresolved = True
        corrected = "".join(changes).strip()
        if not corrected:
            return words
        confidence = float(np.mean([w["conf"] for w in words])) if words else 0
        if all_verified:
            confidence = max(confidence, min(95, min(r.score for r in reads)*100))
        return [{"text": corrected, "left": reads[0].left,
                 "width": reads[-1].left+reads[-1].width-reads[0].left, "conf": confidence,
                 "font_verified": all_verified,
                 "font_uncertain": unresolved,
                 "font_groups": [list(r.alternatives) for r in reads if len(r.alternatives) > 1]}]

    def segment(self, image: np.ndarray) -> list[GlyphRead]:
        """Decode groups of up to six components, including accents/icons."""
        foreground = (image < 127).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        components = sorted([(i, tuple(map(int, s))) for i, s in enumerate(stats[1:], 1) if s[4] >= 1],
                            key=lambda pair: (pair[1][0]+pair[1][2]/2, pair[1][1]))
        if not components:
            return []
        median_area = float(np.median([s[4] for _, s in components]))
        spans = {}
        for i in range(len(components)):
            for end in range(i+1, min(i+6, len(components))+1):
                selected = components[i:end]
                x = min(s[0] for _, s in selected)
                y = min(s[1] for _, s in selected)
                right = max(s[0]+s[2] for _, s in selected)
                bottom = max(s[1]+s[3] for _, s in selected)
                mask = np.isin(labels[y:bottom, x:right], [label for label, _ in selected])
                area = int(mask.sum())
                spans[i, end] = (mask, x, y, right-x, bottom-y, area)

        def decode(scale=None, origin=None):
            best = [float("-inf")]*(len(components)+1)
            best[0] = 0
            previous = {}
            for i in range(len(components)):
                for end in range(i+1, min(i+6, len(components))+1):
                    mask, x, y, w, h, area = spans[i, end]
                    read = self.classify(mask, top=y, scale=scale, origin=origin)
                    read.left, read.top, read.width, read.height = x, y, w, h
                    score = best[i]+read.score*area-.045*median_area
                    if score > best[end]:
                        best[end] = score
                        previous[end] = (i, read)
            reads, pos = [], len(components)
            while pos:
                pos, read = previous[pos]
                reads.append(read)
            return list(reversed(reads))

        reads = decode()
        for _ in range(2):
            reliable = [r for r in reads if r.score >= .86 and r.reference]
            if not reliable:
                break
            scale = float(np.median([r.height/r.reference_height for r in reliable]))
            origin = float(np.median([r.top-r.reference_top*scale for r in reliable]))
            updated = decode(scale, origin)
            if [(r.char, r.left, r.width) for r in updated] == [(r.char, r.left, r.width) for r in reads]:
                reads = updated
                break
            reads = updated
        return reads
