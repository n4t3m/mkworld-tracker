# Result-screen fixtures

These images cover two different screens. **Race results** appear after each
race and show placement changes with `+points`. **Match standings** appear at
the end of a match under a `CONGRATULATIONS!`, `NICE TRY!`, or similar banner.
Do not use a match-standings image to evaluate the race-result reader.

| Fixture location | Screen | Room size | What it covers |
| --- | --- | --- | --- |
| `race_results/`, `race_results_dandelion/`, `race_results_ktb/`, `race_results_ddj/` | Race results and overall standings | 12 players | Four older no-teams race sequences, including scrolling and noisy names. Each sequence labels its frames in `tests/test_race_results.py`. |
| `team_results/` | Colored race results and overall standings | 12 players | Red/blue row detection, scrolling, background noise, and OCR. |
| `video_analysis/race-results-dark-24p*` | Dark race results | 24 players | Upper and lower halves, scrolling tail, highlighted row, overlapping icons, and short names. |
| `video_analysis/overall-standings-colored-12p.jpg` | Overall standings | 12 players | Colored table without `+points`; used for table presence, not race placements. |
| `match_results/match-standings-no-teams-8p-*`, `match_results/20260404_151819.png` | Final match standings | 8 players | Silver-cup and `NICE TRY!` variants. |
| `match_results/twoteams_result.png`, `match_results/20260329_163426.png`, `match_results/20260329_174714.png`, `match_results/20260329_174922.png` | Final match standings | 12 players | Two-team banner and scores. |
| `match_results/20260329_162559.png`, `match_results/20260329_162602.png`, `match_results/20260330_190417.png` | Final match standings | 24 players | Two-column no-teams standings. These are **not** per-race results. |
| `match_results/20260403_172613.png`, `match_results/20260403_175939.png` | Final match standings | 23 and 22 players | `NICE TRY!` layouts with a score-only right column. |
| `match_results/threeteams_banner_yellow_win.png`, `match_results/fourteams_banner_green_win.png` | Final match standings | 24 players | Three-team yellow and four-team green winning banners, team scores, and player rows. The match reader OCRs these rows. |
| `match_results/fourteams_banner_green_win_loading.png` | Final match standings loading frame | 24-player layout | Banner visible before player rows; full match OCR should return no result yet. |

The remaining `match_results/` images test gameplay, player-list, and banner
false positives. `race_results_negative/` and
`video_analysis/ghost-valley-false-results.jpg` are gameplay negatives.

The `24p` suffix records the player count **in the captured fixture**. Reader
routing is based on the screen's appearance and row geometry, not player count.
