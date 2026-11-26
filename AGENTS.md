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
  full-screen 1st–12th HUDs. Do not connect it to live tracking as part of unrelated
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
  versions, including unsupported ranks and hidden HUDs. Some gradient fills
  remain unknown; do not turn these into guessed positions.
  Normal masks run first; fade masks compensate for low saturation, shifted hue
  and translucent outlines, with stronger shape/margin requirements.
- `src/mktracker/detection/position_templates/`: bundled 1st–12th silhouettes
  and their provenance. These must remain available in installed packages.
- `src/mktracker/ui/main_window.py`: Video Review position label, playback,
  seeking, independent live capture and background video-analysis thread.
- `src/mktracker/video_analysis.py`: recorded race boundaries and result-name
  analysis. `video_timeline.py`: cached race metadata, independent of playback.
- `team_results.py`, `name_consensus.py`, `preview_roster.py`,
  `font_catalog.py`, `font_decoder.py`, `font_symbols.py` under `src/mktracker/detection/`:
  custom-room name reading, repeated-frame/preview corroboration and optional
  game-font matching. A name's `[review]` flag describes conflicting or limited
  evidence, not a calibrated probability of error.
- `src/mktracker/detection/race_rank.py` and `gemini_rank.py`: older live/API
  rank path; separate from the new local Video Review reader.
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

## What Git does and does not carry

- Git carries application code, uv.lock, position templates, and regression
  crops/labels in `tests/fixtures/`. Position tests need no recordings, game
  font, Tesseract process or API key; video-review integration tests use OCR.
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

Position fixtures include separate template-source and validation timestamps;
`labels.json` and `fades.json` record manually assigned values. Tests cover
real blue/red HUDs, fades, no-HUD frames, resizing, recompressed video, direct
seeks, playback and live/review isolation. Keep unknown-frame rejection when
improving sensitivity. Add real evidence for larger rooms or other HUD layouts
before claiming support.

## Optional full-recording spot checks

### Latest validation and next work

- YouTube `GxDrSh-Ee20`, source 1:00–33:14, was downloaded locally to
  `../recordings/GxDrSh-Ee20_00-01-00_to_00-33-14.mp4` (1080p60, 32:14).
  Recording and evidence are ignored local assets, not carried by Git.
- This footage uses a 24-player neutral HUD. At clip 2:00 (source 3:00),
  visible **19th is incorrectly read as 10th**. Fix unsupported-rank rejection
  before claiming support; several valid ranks also remain unknown.
- The full scan produced 12 race entries, seven unknown courses, and only one
  results timestamp with 11 draft names. Boundary verification fails on race 1's
  missing results. The timeline is not a verified complete race/results record.
- Five-second sampling produced 387 frames, with 155 position and 339 coin
  readings. These measure availability, not accuracy. Coin samples looked
  plausible but have not been exhaustively hand-labeled.
- Local evidence and scripts: `analysis/youtube-validation/` (report, timeline,
  HUD samples, contact sheets, GUI and boundary checks). Cached opening and eight
  direct seeks passed. Full regression log: `analysis/youtube-tests-final.log`:
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
