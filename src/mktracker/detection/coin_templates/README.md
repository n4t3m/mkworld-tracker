# Coin digit references

Binary silhouettes extracted from `2026-10-24 16-11-42.mp4`, normalized by
`coin_digits` to 28×40 pixels. Sources (seconds, digit slot left/right):
0: 80 left; 1: 160 left; 2: 88 right; 3: 100 right; 4: 190 right;
5: 110 right; 6: 345 right; 7: 400 right; 8: 555.4 right; 9: 130 right.

Validation crops in `tests/fixtures/hud_coins` use other timestamps, plus the
red HUD from `2026-09-21 18-09-47.mp4`. Labels were assigned visually from
the recordings. Crops retain the lower-left 480×140 region of 1920×1080
normalized frames. Reconstruct it at x=0, y=940 for tests. No full recordings
or local paths are needed to run the regression suite.

The reader locates the gold coin icon and derives the digit crop from its
position and diameter, rather than assuming fixed recording borders. Tests
also reuse full original capture-card frames in `race_finish` and
`race_results_negative`, including neutral panels and yellow maximum-count
digits. Synthetic padding tests cover side bars, letterboxing and asymmetric
borders at multiple resolutions; menu/results fixtures check rejection.

The reader requires one unambiguous icon-anchored digit pair and a
result in 0–20. Shape ambiguity and missing evidence return unknown. These
assets must be included in installed packages alongside the reader.
