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
