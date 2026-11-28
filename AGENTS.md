# Continuing development

Read this file and the relevant sections of README.md before changing the app.
CLAUDE.md contains additional architecture notes, some about older detector
paths; current source and this handoff take precedence over those descriptions.

## Current product scope

- This is a modified MKWorld Tracker, developed in this same repository.
- Live View and Video Review use independent sources, modes and state. Recorded
  playback/analysis must not reset live tracking or send Discord notifications.
- Custom Room is implemented. Worldwides and Time Trial selectors are placeholders.
- Race position is implemented **only in Video Review**, for single-player,
  full-screen 1st–24th HUDs. Do not connect it to live tracking as part of unrelated
  work. Coin count is also implemented only in Video Review for blue/red
  single-player colored/neutral HUDs (0–20); lap/checkpoint and item reading are
  not implemented.
- The position is read from each displayed frame, including direct seeks while
  paused. It is not saved as a time series. Missing/ambiguous evidence shows a
  dash; do not silently retain the last position across hidden HUDs or seeks.

## Code map

- `src/mktracker/detection/hud_coins.py` and `coin_templates/`: local two-digit
  coin matching anchored to a detected gold coin icon in the lower-left region.
  Icon diameter supplies crop scale; do not restore recording-specific offsets.
  Supports neutral panels and yellow maximum-count numerals. Each frame is
  independent; hidden/ambiguous counts show a dash. `tests/test_hud_coins.py`
  uses tracked held-out crops from two recordings, including digit animations,
  resizing, recompressed video, direct seeks and mode changes, plus original
  full capture-card fixtures and asymmetric side/top/bottom padding. Run it alongside
  position and video-review regressions when changing the shared UI.

- `src/mktracker/detection/hud_position.py`: local position shape matching. The
  search covers the lower-right 35% in each dimension; numeral/suffix alignment
  replaces recording-specific baseline checks. Narrow aspect variations handle
  capture padding/resizing. Tests reuse full direct-capture frames and transformed
  versions, including 14th/21st and hidden HUDs. Some gradient fills
  remain unknown; do not turn these into guessed positions.
  Normal masks run first; fade masks compensate for low saturation, shifted hue
  and translucent outlines, with stronger shape/margin requirements.
- `src/mktracker/detection/position_templates/`: bundled 1st–24th silhouettes
  and neutral 2nd/3rd edge templates, with their provenance. These must remain
  available in installed packages.
- `src/mktracker/ui/main_window.py`: Video Review position label, playback,
  seeking, independent live capture and background video-analysis thread.
- `src/mktracker/video_analysis.py`: recorded race boundaries and result-name
  analysis. `video_timeline.py`: cached race metadata, independent of playback.
- `team_results.py`, `name_consensus.py`, `preview_roster.py`,
  `font_catalog.py`, `font_decoder.py`, `font_symbols.py` under `src/mktracker/detection/`:
  custom-room name reading, repeated-frame/preview corroboration and optional
  game-font matching. A name's `[review]` flag describes conflicting or limited
  evidence, not a calibrated probability of error.
- `name_model.py` and `name_ocr_worker.py`: optional persistent PaddleOCR readers.
  The dark checkpoint reads result names and ranks; a separate colored checkpoint
  reads colored rows and preview rosters. `course_model.py`, `hud_rank_model.py`,
  `match_results_model.py`, and `saved_race_model.py` apply those readers to their
  respective screens. `state_machine.py` and `video_analysis.py` select local
  model reads where supported, then retain Tesseract and Gemini fallbacks.
  Inspect each caller's completeness and roster checks before broadening model use.
- `src/mktracker/detection/race_rank.py` and `gemini_rank.py`: legacy live/API
  rank paths; `hud_rank_model.py` adds local model reading to Live View.
- `src/mktracker/detection/tracks.py`: shared 48-course catalog through 1.9.0.

## New machine setup

Run from the repository root with Python >=3.10, uv and the Tesseract binary
installed (see README.md for platform details):

```sh
uv sync --locked
uv run mktracker
uv run mktracker "/path/to/gameplay.mp4"
```

On Windows, `Start-Tracker.ps1` selects `.venv-app` if present, otherwise `.venv`,
and sets the working directory. `.venv-app` is an existing local environment,
not a requirement; `uv sync --locked` creates `.venv` on a clean machine. Never
copy virtual environments between computers. Windows has been exercised;
macOS/Linux capture and OCR setup still need platform verification.

For trained OCR, install the dark and color checkpoint release assets from the
private `n4t3m/mkworld-tracker-ocr` repository. Follow that repository's README
to generate local YAML files and `characters.txt`. Set `NAME_OCR_PYTHON` to a
separate Python with PaddlePaddle, `NAME_OCR_PADDLEOCR` to a compatible
PaddleOCR source checkout, and the dark/color `NAME_OCR_*_CONFIG` and
`NAME_OCR_*_CHECKPOINT` variables in the ignored `.env`. The runtime configs
contain an absolute `character_dict_path`; regenerate them after moving the
files. This app can still run without checkpoints using its fallback readers.

## What Git does and does not carry

- Git carries application code, uv.lock, position templates, and regression
  crops/labels in `tests/fixtures/`. Position tests need no recordings, game
  font, Tesseract process or API key; video-review integration tests use OCR.
- The private `n4t3m/mkworld-tracker-ocr` repository carries the labeled
  name-crop corpus, training/evaluation scripts, experiment log, and portable
  config templates. Selected `.pdparams` weights are private GitHub release
  assets (`ocr-v1`), outside Git history. Neither repository carries the game
  font. Keep model weights, generated runtime configs, and the font out of
  application commits.
- Full recordings and the optional game font are external assets. Copy them
  separately if needed; select the font again in Settings on the new machine.
  `.env.example` documents the settings, including `GAME_FONT_PATH`.
- `.env` is ignored and may contain API credentials and a Discord webhook.
  Reconfigure via Settings; never commit it or paste its values into handoffs.
  Local Video Review and position detection do not require a Gemini key.
- `video_timelines/`, `matches/`, captures and debug/analysis output are ignored.
  Back up match history separately if it matters. Run the app from the repo
  root to keep relative data/settings paths consistent.
- Timeline cache identity includes **absolute video path, byte size and mtime**.
  Copying a recording/cache to a different location usually invalidates reuse.
  Reanalyze the recording; portable cache import/remapping is not implemented.

## Validation

Install the test runner explicitly; pytest is not an application dependency:

```sh
uv run --with pytest python -m pytest tests/test_hud_position.py tests/test_video_review.py -q
```

For position changes, also run the relevant regressions used for this feature:

```sh
uv run --with pytest python -m pytest tests/test_hud_position.py tests/test_video_review.py tests/test_race_results.py tests/test_state_machine_persistence.py -q
```

The feature baseline was 229 passing tests. Counts are historical evidence,
not a substitute for running current tests. Qt tests select offscreen mode.
The broader suite includes optional local-font tests and older tests that need
ignored `testdata/`; inspect skips/failures before claiming full-suite coverage.
For OCR changes, run focused model and video-review tests and consult the
private repository's model log for recording-separated evaluations. The
November 2026 app OCR run had 193 passes and two legacy Tesseract name misses
(`KT`/`Rte` and `KT viti`/`viti`); the same misses reproduce with the
pre-OCR detector. Do not attribute those two assertions to the repo split.

Position fixtures include separate template-source and validation timestamps;
`labels.json`, `labels_24p.json` and `fades.json` record manually assigned values. Tests cover
real blue/red and 24-player neutral HUDs, fades, no-HUD frames, resizing, recompressed video, direct
seeks, playback and live/review isolation. Keep unknown-frame rejection when
improving sensitivity. Add real evidence for larger rooms or other HUD layouts
before claiming support.

## Optional full-recording spot checks

### Latest validation and next work

- YouTube `GxDrSh-Ee20`, source 1:00–33:14, was downloaded for validation
  (1080p60, 32:14) and later deleted locally at the user's request.
  The recording is not carried by Git; tracked HUD crops preserve test evidence.
- This footage uses a 24-player neutral HUD. The earlier visible 19th at clip
  2:00 (source 3:00) was read as 10th; the 13th–24th templates now read it as
  19th. Held-out crops cover all twelve high ranks and pastel 2nd/3rd HUDs.
  Other gradient fills and extreme fades can still be unknown.
- The full scan produced 12 race entries, seven unknown courses, and only one
  results timestamp with 11 draft names. Boundary verification fails on race 1's
  missing results. The timeline is not a verified complete race/results record.
- Five-second sampling produced 387 frames, with 155 position and 339 coin
  readings. These measure availability, not accuracy. Coin samples looked
  plausible but have not been exhaustively hand-labeled.
- Earlier ignored evidence and scripts were under `analysis/youtube-validation/`
  (report, timeline, HUD samples, contact sheets, GUI and boundary checks); that
  directory is not carried by Git. New HUD crops and labels are tracked under
  `tests/fixtures/hud_position/`. Cached opening and eight earlier direct seeks
  passed. Earlier full regression log: `analysis/youtube-tests-final.log`:
  **887 passed, six failed**, all six due to missing legacy screenshots under
  `testdata/trackselected/`; four pytest fixture deprecation warnings.

These recordings are external; the filenames identify source footage, not
paths expected to exist on every machine:

- `2026-10-24 16-11-42.mp4`: 1:10 faded 4th; 1:11.25 fading 2nd; 2:20 solid 7th;
  13:40 faint 9th; 4:00 menu should show a dash.
- `2026-09-21 18-09-47.mp4`: 5:59 faded 9th; 5:59.5 fading 10th.

Check displayed position against the HUD and normal-speed playback. A direct
seek into a fade must work without first showing a solid position. Extreme
fades, occlusion and unsupported layouts may still be unknown.
