"""Detect the race placement rank indicator in the bottom-right HUD.

The rank indicator (e.g. "1st", "12th") appears in the bottom-right corner
of the gameplay screen as large styled text with a 3-D shadow/outline.

This module detects WHETHER a rank indicator is currently visible and returns
a cropped colour image of just the indicator region.  The actual rank number
is intended to be read by an LLM in a separate step.

Detection pipeline
------------------
1. Crop the bottom-right ROI of the frame.
2. Build colour-targeted binary masks that specifically capture the rank
   text's saturated fill colours (orange, yellow, red, blue).
3. Find the best text-like connected component from each mask.
4. Require at least two colour-targeted masks to agree on the location
   (consensus filter).
5. If a valid blob is found, map its bounding box back to the original
   frame and return the colour crop.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEBUG_RANK_DIR = Path("debug_rank")

# ---------------------------------------------------------------------------
# ROI — bottom-right corner where the rank indicator lives.
# ---------------------------------------------------------------------------
_ROI_X1 = 0.78
_ROI_Y1 = 0.75

# ---------------------------------------------------------------------------
# Blob extraction
# ---------------------------------------------------------------------------
_SCALE = 3

# Connected-component area bounds (fraction of scaled ROI area).
_CC_AREA_MIN = 0.005
_CC_AREA_MAX = 0.35

# Focus region — text is always in the lower-right portion of the ROI.
_FOCUS_Y = 0.35
_FOCUS_X = 0.25

# Merge margin for nearby components (ordinal suffix, shadow fragments).
_MERGE_MARGIN = 60

# ---------------------------------------------------------------------------
# Shape heuristics
# ---------------------------------------------------------------------------
_AR_MIN = 0.7
_AR_MAX = 4.5
_DENSITY_MIN = 0.10
_DENSITY_MAX = 0.75
_MIN_HEIGHT_FRAC = 0.15

# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------
_MIN_CONSENSUS = 2
_CLUSTER_DIST = 80


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RaceRankDetector:
    """Detects whether a race-placement rank is visible and crops it."""

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a colour crop of the rank indicator, or ``None``.

        Parameters
        ----------
        frame:
            BGR frame from the capture card (any resolution).

        Returns
        -------
        np.ndarray or None
            A cropped BGR image tightly framing the rank text, suitable
            for sending to an LLM for digit recognition.  ``None`` if no
            rank indicator is visible.
        """
        result = _find_rank_region(frame)
        if result is None:
            logger.debug("RaceRankDetector: no rank indicator found")
            return None

        y1, y2, x1, x2 = result
        crop = frame[y1:y2, x1:x2]
        logger.debug(
            "RaceRankDetector: rank region [%d:%d, %d:%d] (%dx%d)",
            y1, y2, x1, x2, x2 - x1, y2 - y1,
        )

        # Temporary debugging: save every detected rank crop.
        _save_debug_crop(crop)

        return crop


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_debug_crop(crop: np.ndarray) -> None:
    """Save *crop* to debug_rank/ with a timestamp filename."""
    _DEBUG_RANK_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = _DEBUG_RANK_DIR / f"{ts}.png"
    cv2.imwrite(str(path), crop)
    logger.debug("Saved rank crop: %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_rank_region(
    frame: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Return (y1, y2, x1, x2) in original-frame coordinates, or None."""
    h, w = frame.shape[:2]
    roi_y1 = int(h * _ROI_Y1)
    roi_x1 = int(w * _ROI_X1)
    roi = frame[roi_y1:h, roi_x1:w]
    rh, rw = roi.shape[:2]

    big = cv2.resize(
        roi, (rw * _SCALE, rh * _SCALE), interpolation=cv2.INTER_CUBIC,
    )
    bh, bw = big.shape[:2]
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)

    # --- Colour-targeted binary masks ---
    # Only paths that specifically capture the rank text's fill colour.
    # This avoids the false positives from untargeted thresholds (greyscale
    # Otsu, etc.) that pick up random game objects.
    binaries: list[np.ndarray] = []

    # 1. Saturation + brightness — captures all coloured text fills.
    #    (yellow, orange, red, blue — everything except pure white)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    bright_coloured = ((sat > 100) & (val > 150)).astype(np.uint8) * 255
    binaries.append(bright_coloured)

    # 2. Orange-difference (R - B) > 40 — targets the dominant orange fill
    #    used by ranks 4-24 (and partially 1st/3rd).
    r_arr = big[:, :, 2].astype(np.int16)
    b_arr = big[:, :, 0].astype(np.int16)
    orange = ((r_arr - b_arr) > 40).astype(np.uint8) * 255
    binaries.append(orange)

    # 3. Warm hue mask — orange/yellow/red text (H 0-35 with high S+V).
    hue = hsv[:, :, 0]
    warm = (
        ((hue < 35) | (hue > 170))  # red wraps around 180
        & (hsv[:, :, 1] > 120)
        & (hsv[:, :, 2] > 150)
    ).astype(np.uint8) * 255
    binaries.append(warm)

    # 4. Blue-channel dominance — targets 2nd-place blue-white text.
    blue_dom = ((b_arr - r_arr) > 20).astype(np.uint8) * 255
    binaries.append(blue_dom)

    # 5. High-value yellow/green — targets 1st-place yellow/lime text.
    g_arr = big[:, :, 1].astype(np.int16)
    yellow = (
        (g_arr > 130) & (r_arr > 130) & (b_arr < 140)
    ).astype(np.uint8) * 255
    binaries.append(yellow)

    # 6. More permissive saturation mask (catches lower-saturation fills).
    bright_coloured_lax = (
        (sat > 60) & (val > 130)
    ).astype(np.uint8) * 255
    binaries.append(bright_coloured_lax)

    # 7. R-channel Otsu + inverse — reliable for orange text.
    r_ch = big[:, :, 2]
    _, r_thresh = cv2.threshold(
        r_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binaries += [r_thresh, cv2.bitwise_not(r_thresh)]

    # 8. Warm with relaxed saturation — catches lower-saturation red.
    warm_lax = (
        ((hue < 35) | (hue > 170))
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 130)
    ).astype(np.uint8) * 255
    binaries.append(warm_lax)

    # 9. Greyscale Otsu + inverse — fallback for text on same-colour
    #    backgrounds where no colour mask can separate text from BG.
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, g_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binaries += [g_thresh, cv2.bitwise_not(g_thresh)]

    # 10. Adaptive threshold — uses LOCAL contrast so the dark text
    #     outline is detected even on same-colour backgrounds.
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -10,
    )
    binaries += [adapt, cv2.bitwise_not(adapt)]

    # 11. Canny edges + morphological closing — outlines only.
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    binaries.append(closed)

    # --- Collect detections, cluster, require consensus ---
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for binary in binaries:
        bbox, score = _best_text_component(binary, bh, bw)
        if bbox is not None:
            detections.append((bbox, score))

    if len(detections) < _MIN_CONSENSUS:
        return None

    # Cluster by bbox centre proximity.
    groups: list[list[tuple[tuple[int, int, int, int], float]]] = []
    for bbox, score in detections:
        cy = (bbox[0] + bbox[1]) / 2
        cx = (bbox[2] + bbox[3]) / 2
        placed = False
        for group in groups:
            ref = group[0][0]
            ref_cy = (ref[0] + ref[1]) / 2
            ref_cx = (ref[2] + ref[3]) / 2
            if (abs(cy - ref_cy) < _CLUSTER_DIST
                    and abs(cx - ref_cx) < _CLUSTER_DIST):
                group.append((bbox, score))
                placed = True
                break
        if not placed:
            groups.append([(bbox, score)])

    # Pick the largest cluster that meets consensus.
    groups.sort(key=len, reverse=True)
    best_bbox: tuple[int, int, int, int] | None = None
    for group in groups:
        if len(group) < _MIN_CONSENSUS:
            break
        best_bbox = max(group, key=lambda x: x[1])[0]
        break

    if best_bbox is None:
        return None

    # Map from upscaled ROI coords back to original frame coords.
    br1, br2, bc1, bc2 = best_bbox
    pad = 10
    br1 = max(0, br1 - pad)
    br2 = min(bh, br2 + pad)
    bc1 = max(0, bc1 - pad)
    bc2 = min(bw, bc2 + pad)

    fy1 = roi_y1 + br1 // _SCALE
    fy2 = roi_y1 + br2 // _SCALE
    fx1 = roi_x1 + bc1 // _SCALE
    fx2 = roi_x1 + bc2 // _SCALE

    fy1 = max(0, fy1)
    fy2 = min(h, fy2)
    fx1 = max(0, fx1)
    fx2 = min(w, fx2)

    if fy2 - fy1 < 10 or fx2 - fx1 < 10:
        return None

    return fy1, fy2, fx1, fx2


def _best_text_component(
    binary: np.ndarray, bh: int, bw: int,
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Find the best text-like connected component in *binary*.

    Returns ``(bbox, score)`` where bbox is ``(r1, r2, c1, c2)`` in the
    upscaled-ROI coordinate space, or ``(None, 0)``.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    img_area = bh * bw
    focus_y = int(bh * _FOCUS_Y)
    focus_x = int(bw * _FOCUS_X)

    candidates: list[tuple[float, int, np.ndarray]] = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (img_area * _CC_AREA_MIN < area < img_area * _CC_AREA_MAX):
            continue
        cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
        if cy < focus_y or cx < focus_x:
            continue
        score = area * (cx / bw) * (cy / bh)
        candidates.append((score, i, stats[i]))

    if not candidates:
        return None, 0.0

    candidates.sort(reverse=True)
    _, primary_lbl, primary_s = candidates[0]

    px1 = primary_s[cv2.CC_STAT_LEFT]
    py1 = primary_s[cv2.CC_STAT_TOP]
    px2 = px1 + primary_s[cv2.CC_STAT_WIDTH]
    py2 = py1 + primary_s[cv2.CC_STAT_HEIGHT]

    # Merge nearby components (ordinal suffix, outline fragments).
    merged_mask = (labels == primary_lbl).astype(np.uint8) * 255
    for _, lbl, s in candidates[1:]:
        lx1 = s[cv2.CC_STAT_LEFT]
        ly1 = s[cv2.CC_STAT_TOP]
        lx2 = lx1 + s[cv2.CC_STAT_WIDTH]
        ly2 = ly1 + s[cv2.CC_STAT_HEIGHT]
        if (
            lx1 < px2 + _MERGE_MARGIN
            and lx2 > px1 - _MERGE_MARGIN
            and ly1 < py2 + _MERGE_MARGIN
            and ly2 > py1 - _MERGE_MARGIN
        ):
            merged_mask |= (labels == lbl).astype(np.uint8) * 255
            px1 = min(px1, lx1)
            py1 = min(py1, ly1)
            px2 = max(px2, lx2)
            py2 = max(py2, ly2)

    blob_w = px2 - px1
    blob_h = py2 - py1

    if blob_h < 20 or blob_w < 20:
        return None, 0.0

    # --- Shape heuristics ---
    ar = blob_w / blob_h
    if not (_AR_MIN <= ar <= _AR_MAX):
        return None, 0.0

    if blob_h < bh * _MIN_HEIGHT_FRAC:
        return None, 0.0

    # Reject blobs that span most of the ROI — these are background.
    if blob_h > bh * 0.65:
        return None, 0.0

    # Pixel density inside the bounding box.
    roi_mask = merged_mask[py1:py2, px1:px2]
    density = float(np.count_nonzero(roi_mask)) / (blob_w * blob_h)
    if not (_DENSITY_MIN <= density <= _DENSITY_MAX):
        return None, 0.0

    quality = (blob_w * blob_h) * (px2 / bw) * (py2 / bh)
    return (py1, py2, px1, px2), quality
