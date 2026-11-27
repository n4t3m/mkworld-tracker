These binary silhouettes were extracted from the position HUD in
`2026-10-24 16-11-42.mp4`, using `position_masks(frame)` and the broad cyan mask.
They include the numeral and ordinal suffix and contain no player names.

| Position | Source seconds |
| --- | --- |
| 1 | 1450 |
| 2 | 760 |
| 3 | 180 |
| 4 | 1240 |
| 5 | 315 |
| 6 | 330 |
| 7 | 150 |
| 8 | 640 |
| 9 | 120 |
| 10 | 105 |
| 11 | 580 |
| 12 | 90 |

Regression fixtures use different timestamps from both the September and
October recordings. Their source times and manually assigned positions are in
`tests/fixtures/hud_position/labels.json`. Only the bottom-right HUD region is
retained; tests restore it to its original full-frame location.

`tests/fixtures/hud_position/fades.json` labels four quarter-second sequences
around real blue/red position changes, including translucent fills, glow,
partial item-overlay occlusion, and fully hidden HUDs. These frames are test
inputs, not additional templates. A video regression also recompresses fade
frames and seeks directly to them, so passing does not depend on remembering
the preceding solid position.

Capture-layout regressions also reuse full frames from `race_finish`, covering
unbarred capture-card footage, gold/pink 1st-place fills, red and cyan shapes,
hidden HUDs and 14th/21st indicators. Added side bars, letterboxing,
asymmetric padding and resizing exercise translation/scale/aspect changes.
The reader searches the lower-right region for numeral/suffix geometry rather
than requiring a fixed baseline from the source recording. Limited aspect
variants compensate for padding followed by capture resizing. Uncertain gradient
or heavily faded shapes remain unknown.

The `13-neutral.png` through `24-neutral.png` masks come from the 1080p60
24-player neutral HUD in YouTube `GxDrSh-Ee20`. Source times (seconds) are:
13: 120, 14: 140, 15: 135, 16: 891, 17: 210, 18: 896,
19: 180, 20: 215, 21: 160, 22: 145, 23: 900, 24: 930.
`tests/fixtures/hud_position/labels_24p.json` records separately sampled
validation frames, including the 19th-place frame previously read as 10th.
The full recording is an external asset; only HUD crops and masks are bundled.

The `neutral_edges/` templates are Canny outlines of pastel 2nd and 3rd HUDs
from the same source (2nd at 550 seconds; 3rd at 300 and 540 seconds). Color
thresholding merges these gradient fills with the background. Edge matching
is a conservative fallback after ordinary and faded masks fail. Held-out
2nd/3rd crops come from other timestamps and races.
