# Full local Video Review audit (2026-11-27)

Fresh scans used the app's `analyze_video` path, the configured local name models,
and isolated timeline caches. These timings are from this Apple M3 Pro, including
video seeking, frame decoding, result detection, OCR, and timeline updates. They
do not measure GUI playback speed. The source recordings and full JSON/JSONL
reports are local, ignored assets under `analysis/video-review-audit/` and
`analysis/video-review-audit-12p/`.

| Recording | Source length | Full scan | Races with finish/results | Placements |
| --- | ---: | ---: | ---: | ---: |
| Local 24-player custom room | 33:32 | 4:39 | 12/12 | 279/288 |
| Local 12-player war | 46:15 | 4:03 | 12/12 | 144/144 |

The time from the first detected result frame to all *available* names was
0–6.74 seconds in the 24-player recording and 0–3.96 seconds in the 12-player
recording. A zero means the first progress update already contained the complete
board; it still includes OCR work before that update. Result OCR model calls
totaled roughly 1–12 seconds per race, spread over multiple frames. Neither
recording showed the earlier minutes-per-result behavior.

| Race | 24p names | 24p seconds to available names | 12p names | 12p seconds to 12 names |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 4.18 | 12 | 1.98 |
| 2 | 24 | 6.74 | 12 | 0.00 |
| 3 | 24 | 4.05 | 12 | 2.16 |
| 4 | 24 | 4.23 | 12 | 1.97 |
| 5 | 24 | 4.26 | 12 | 2.02 |
| 6 | 24 | 4.24 | 12 | 1.87 |
| 7 | 15 | 0.00 | 12 | 1.89 |
| 8 | 24 | 2.36 | 12 | 1.55 |
| 9 | 24 | 6.01 | 12 | 1.90 |
| 10 | 24 | 4.27 | 12 | 2.06 |
| 11 | 24 | 6.27 | 12 | 3.96 |
| 12 | 24 | 6.48 | 12 | 3.69 |

The 24-player scan originally found 269 names. Sampling one cached quarter
frame when scroll ranks were missing recovered ten names for only 6.56 more
seconds of total scan time. In race 7, rank 16 is clipped at the bottom and
ranks 17–24 are absent before an edit cuts back to gameplay. Those nine names
cannot be read from the available result footage.

Names were compared with result screenshots from each race. Clear 24-player
spelling issues remain: `u1st` should have a space (`u 1st`) in seven races,
one `Llam` should be `L1am`, and one `ra` should be `rq`. A later
session-reconciliation fix now resolves the partial-page `Llam` to `L1am`
from a same-row alternative and the repeated session spelling. The 12-player scan
has one clear name error: race 4 rank 9 reads `Toadetre` where the footage
shows `Toadette`. The repeated raised `²` in the `Vy²` tag is now retained
across the 12-player session. These names remain OCR drafts for review.

After the 12-player tag change, direct real-screen checks still read all six
yellow and all six green names on the Rayzopp four-team screen. On the
Mankalor four-team screen, yellow was 6/6 and green 5/6; the remaining green
error is capitalization of `BDD Obiyo` (`BDD obiyo`).

The broad test run produced 175 passing tests and two repeatable failures in
legacy no-team Tesseract expectations (`KT`→`Rte`, `KT viti`→`viti`) in
`tests/test_race_results.py`. The focused video-review, colored-name,
session-name, and state-persistence suite passed 119/119 tests. The GUI was covered by offscreen Video Review
tests, but the full recordings were scanned through the analyzer rather than
played end to end in a visible window.

## Priority-glyph follow-up

After adding font-backed checks for `0/O` and `1/I`, both full recordings were
scanned again with the same selected checkpoints. The 24-player run took
278.408 seconds, found the same 12 races and 279 placements, and changed only
race 7 rank 7 from `Llam` to `L1am` through session reconciliation. The
12-player run took 242.654 seconds, found the same 12 races and 144
placements, and changed no displayed names. These runs confirm the added
checks did not cause a minutes-per-result regression on either local video.
Focused real-crop and color diagnostics are in the separate private
`n4t3m/mkworld-tracker-ocr` research repository.

## Countdown course-card follow-up (2026-11-28)

The trained dark recognizer now reads a course on the race countdown card when
the map preview was cut from the recording. Fresh scans of the same two VODs
and an additional 46:09 local 24-player VOD used the unchanged selected
checkpoints and isolated caches:

| Recording | Scan time | Races with results | Placement rows | Longest wait from first result update to available rows |
| --- | ---: | ---: | ---: | ---: |
| Original 33:32 24-player room | 7:03 | 12/12 | 279/288 | 8.21 s |
| Original 46:15 12-player room | 5:24 | 12/12 | 144/144 | 4.48 s |
| Additional 46:09 24-player room | 5:46 | 12/12 | 288/288 | 12.48 s |

The original 24-player recording now identifies race 4 as **SNES Ghost
Valley 2** and race 8 as **SNES Koopa Beach 1**. The other four course gaps
remain unknown. Race 7 still has only 15 available names because its result
video cuts away. The two repeated scans produced exactly the same placement
names and ranks as their prior audits. The additional VOD provides row coverage,
not independently verified spelling accuracy.

Total scan times were longer than the earlier 4:38 and 4:03 on the two repeat
VODs. This was not a minutes-per-result recurrence: each available board
completed within seconds, and timed result-model calls were shorter in these
runs. The countdown-card shape check takes about 0.2 ms per 1080p frame in a
local microbenchmark; the cause of the total wall-time difference has not been
isolated. Treat full-scan throughput as machine-dependent until a controlled
profile identifies the cost.

On the same first five minutes of the 12-player VOD, a back-to-back analyzer
run took 43.827 seconds with the countdown-card check disabled and 42.336
seconds with it enabled; both produced two race entries and 12 names at the
cutoff. The card check did not explain the longer full-scan times in this
bounded comparison.

The full repository suite on this machine reported 1,215 passes and ten
failures. Two failures came from the generic coin-fixture test assuming the
new countdown screenshots had no HUD; visual inspection showed `00` and
`01`, and the corrected expectations passed with all 219 coin/course tests.
The other eight failures were the two previously recorded legacy Tesseract
name assertions and six unavailable `testdata/trackselected/` screenshots.
