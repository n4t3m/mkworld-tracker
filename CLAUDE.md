# MKWorld Tracker — CLAUDE.md

## Project Overview
A Python app that watches a capture card's video feed of Mario Kart and automatically detects game state, match settings, track names, player names, and race result placements using computer vision, OCR, and the Gemini LLM API.

## Tech Stack
- **PySide6** — Qt6 GUI (video display, source selector dropdown, state label, advance/reset buttons, match settings editor, settings tab)
- **OpenCV** (`opencv-python`) — video capture and frame processing
- **pytesseract** — OCR via Tesseract (binary must be installed; auto-detected at `C:\Program Files\Tesseract-OCR\`). Used as fallback when no Gemini API key is configured.
- **Pillow** — image generation for the Lorenzi-style match result table (`table_generator.py`)
- **python-dotenv** — loads/saves `.env` for Gemini API key and model persistence
- **Python >=3.10**, packaged with `pyproject.toml` + hatchling

## Project Structure
```
src/mktracker/
├── main.py                      # Entry point, logging setup (INFO level), QApplication
├── state_machine.py             # GameStateMachine with 8 states, debug frame saving
├── debug_config.py              # Debug-mode flag persistence (.env)
├── gemini_client.py             # Gemini API key/model persistence (.env) and health check
├── gemini_rank.py               # Async Gemini call: race placement rank from gameplay frame
├── gemini_results.py            # Async Gemini call: race results from multiple scrolling frames
├── gemini_match_results.py      # Async Gemini call: final match results from CONGRATULATIONS screen
├── lorenzi_text.py              # Round-trip FinalStandings ↔ Lorenzi-style editor text
├── match_record.py              # Standardised JSON schema for persisted matches (history store)
├── table_generator.py           # Pillow-based Lorenzi-style results-table PNG renderer
├── capture/
│   └── video_source.py          # Camera enumeration (DirectShow) + VideoCapture wrapper
├── detection/
│   ├── __init__.py              # Shared Tesseract path auto-detection
│   ├── match_results.py         # MatchResultDetector: final CONGRATULATIONS! screen OCR (two-teams)
│   ├── match_settings.py        # MatchSettingsDetector + MatchSettings dataclass
│   ├── player_reader.py         # PlayerReader: dynamic grid detection + per-cell OCR
│   ├── race_finish.py           # RaceFinishDetector (FINISH! banner via HSV color detection)
│   ├── race_rank.py             # RaceRankDetector: crops rank indicator region for LLM reading
│   ├── race_results.py          # RaceResultDetector: post-race placement reading (teams & no-teams)
│   ├── track_select.py          # TrackSelectDetector (screen detection + OCR + fuzzy match)
│   └── tracks.py                # Canonical tuple of 30 track names
└── ui/
    ├── main_window.py           # MainWindow: top-level QTabWidget with Live View, Match History, and Settings tabs
    └── match_history.py         # MatchHistoryView + MatchDetailView: match list + per-race timeline with track icons

scripts/
├── backfill_match_records.py   # Rebuild match.json for legacy matches/ folders using Gemini + OCR
└── generate_table.py           # CLI: generate a table PNG for a match by ID (or most recent)

tests/
├── conftest.py                 # stub_fonts fixture: patches font loading for CI-safe rendering tests
├── test_match_record.py        # MatchRecord schema, round-trips, save/load, list_matches
├── test_match_results.py       # MatchResultDetector (CONGRATULATIONS screen OCR)
├── test_player_reader.py       # PlayerReader grid detection and cell OCR
├── test_race_finish.py         # RaceFinishDetector HSV banner detection
├── test_race_results.py        # RaceResultDetector placement parsing (teams & no-teams)
├── test_state_machine_persistence.py  # match_record save/load through state machine transitions
├── test_table_generator.py     # table_generator helpers + generate_table for all layout types
└── test_track_select.py        # TrackSelectDetector screen detection and fuzzy track matching
```

## Architecture
- **Detectors** are screen analyzers: `is_active(frame)` (fast check ~1ms) + `detect(frame)` (full OCR pipeline).
- **PlayerReader** dynamically detects the player grid (row brightness bands + column profiling), OCRs each cell with OTSU thresholding, and cleans avatar-icon noise via regex.
- **RaceResultDetector** reads placement/name pairs from the post-race results screen. Two preprocessing paths: hybrid threshold `(gray>170 & blue>100)` for no-teams mode, blur+Otsu on HSV V-channel for team-colour mode. Gold first-place bar handled with hybrid threshold in both modes. Uses `image_to_data` for word-level OCR with x-position zones to classify placement numbers, names, and scores. Gap-aware sequential placement fix infers missing placements from y-spacing. Adaptive-V cluster counting on a narrow strip (x=0.84-0.90) detects `+` signs to distinguish race results from overall standings. `has_race_results()` combines plus-cluster detection with sharp bar-transition counting to reject gameplay false positives.
- **MatchResultDetector.`_has_result_banner(frame, teams=...)`** is the readiness check for the final CONGRATULATIONS!/NICE TRY!/DRAW! screen. In `No Teams` mode it requires BOTH yellow text AND a red/orange diagonal stripe in the top 18%. In `Two Teams` mode the banner background takes the winning team's colour (or dark grey for DRAW), so the no-teams signature doesn't apply — instead `_has_two_team_result_banner` combines two checks: (1) the red/blue team-score panels below the banner are both strongly saturated, AND (2) the top ~10% of the frame contains a contiguous run of rows that are ≥85% a single banner colour (red, blue, or dark grey) spanning ≥8% of the strip. The row-run test is the key discriminator: mid-race overall standings, pre-race player lists, and FINISH! frames with a blue sky can hit the pixel-fraction threshold from scattered UI elements, but none of them produce a solid horizontal stripe.
- **GameStateMachine** orchestrates detectors based on current state. When a Gemini API key is configured, uses LLM calls instead of OCR for race rank, race results, and match results. All Gemini calls run on daemon threads via fire-and-forget pattern — the state machine transitions immediately and results are written back via callbacks with a `threading.Lock` protecting shared data. When no API key is present, falls back to the OCR pipeline.
- **Gemini modules** (`gemini_rank.py`, `gemini_results.py`, `gemini_match_results.py`): each encapsulates the prompt, API call, response parsing, and debug logging for one detection task. All share the same pattern: encode frame(s) as PNG, POST to `generateContent`, parse JSON (with markdown fence stripping as fallback), invoke callback with parsed result or `None` on failure.
- **main_window.py** owns the frame loop (30fps render) and delegates all game logic to the state machine. The central area is a top-level `QTabWidget` with three tabs: **Live View**, **Match History**, and **Settings**. The Live View tab is self-contained — it owns its own toolbar (video source dropdown, refresh, race/state labels, Gemini status dot, "Start Manual Match", "Advance State", "Reset State", "Capture Frame") above the video display, plus a 270px right-side panel with the match settings editor (syncs bidirectionally with state machine). The Match History tab hosts `MatchHistoryView` (constructed with `state_machine=self._state_machine` so it can identify and render the live match). The Settings tab hosts the Gemini API key + model configuration panel. When the history tab is showing, `_update_frame` short-circuits the BGR→RGB→QPixmap paint (detection still runs) and instead calls `MatchHistoryView.tick()` so the live timeline updates in place; switching to the history tab triggers `MatchHistoryView.refresh()` via `_on_tab_changed`.
- **match_history.py** renders persisted matches AND the currently-running match in the same UI. `MatchHistoryView` takes an optional `state_machine` reference; the live match is whichever record's `match_id == state_machine.current_match_id` while `state_machine.is_match_active`. The live row in the list gets a dark-red background, white text, an `●  LIVE` prefix on its own line, and an `N/M races` progress summary. The detail pane for a live match has three layers of indicator:
  1. A prominent **`_LiveStatusBanner`** at the top — dark red background, 2px red border, large "● LIVE MATCH IN PROGRESS" title, and a sub-line showing `Race N of M  ·  <state description>` (e.g. "Race 3 of 8  ·  Race in progress — waiting for FINISH"). The banner text comes from `_LIVE_STATUS_TEXT`, a `GameState.name → str` mapping kept in this module so it doesn't import the enum.
  2. A red-bordered, red-tinted **header card** with an oversized red `● LIVE` pill before the match id and a `Status: in progress` line in pink.
  3. Per-race cards adapted for live mode: `_RaceCard(race, *, live=True)` shows `Awaiting placements…` instead of `No placements recorded.` and a grey `Rank …` badge while waiting for the Gemini rank callback. `_PendingRaceCard` fills in races up to `settings.race_count` with a dashed-border placeholder. `_PendingFinalStandingsCard` replaces `_FinalStandingsCard` until standings arrive.

  Track icons are loaded via `TRACK_IMAGES` / `TRACK_ICONS_DIR`, scaled to 96px height (width auto-fits, no letterboxing), and cached in a module-level `_ICON_CACHE` keyed by track name. `MatchDetailView.set_record(record, *, live=False, live_status=None)` is the single rendering entry point. `refresh()` re-scans `matches/` via `list_matches()` and preserves selection by `match_id`. `tick()` is called from the main window's frame loop while the history tab is visible — it rebuilds the list when the live `match_id` changes (match start/end) and re-renders the detail pane when **either** the live match's `match.json` mtime changes **or** the state machine has stepped into a new state (so the status banner updates on transitions like → `WAITING_FOR_RACE_END` that don't trigger a disk write). Both checks debounce 30fps polling so the pane doesn't flicker. No new persistence path was needed: the state machine already calls `_save_match_record()` after every meaningful update, so the on-disk record is always live.
- **gemini_client.py** handles Gemini API key and model persistence in `.env` via `python-dotenv`. Health check uses a GET request to `/v1beta/models/{model}` (no tokens consumed). `_VerifyThread` runs the check off the main thread.

## Threading Model
- All detection runs on the main thread every 15th frame (~500ms).
- During WAITING_FOR_RACE_END, DETECTING_RACE_RANK, and READING_RACE_RESULTS, detection runs every 3rd frame (~100ms) for faster response to transient screens.
- Gemini API calls run on daemon `threading.Thread`s — never block the main thread. Callbacks write results back into `RaceInfo` or match data under `_races_lock`.
- **Stale callback handling**: each in-flight Gemini request captures `(match_seq, match_dir)` at dispatch time. `_match_seq` is a monotonic counter bumped on every new match start AND on `reset()`. When a callback fires, if `match_seq != self._match_seq` the state machine has moved on, so the callback re-routes to the **stale path**: `MatchRecord.load(match_dir) → mutate → save`. This writes the result to the original match's on-disk `match.json` without polluting the live state of the new match. The stale path lives in `_apply_stale_rank` / `_apply_stale_results` / `_apply_stale_match_results`. Missing/unreadable folders are dropped silently with a warning.

## State Machine Flow
1. **WAITING_FOR_MATCH** — polls `MatchSettingsDetector` for the "rules decided" screen
2. **MATCH_STARTED** — stores settings, waits 5 seconds, transitions to WAITING_FOR_TRACK_PICK
3. **WAITING_FOR_TRACK_PICK** — polls `TrackSelectDetector` (15s cooldown); on track detection, transitions to READING_PLAYERS_IN_RACE
4. **READING_PLAYERS_IN_RACE** — reads player names on the next frame (gives the player list time to load), transitions to WAITING_FOR_RACE_END
5. **WAITING_FOR_RACE_END** — polls `RaceFinishDetector` for the FINISH! banner, transitions to DETECTING_RACE_RANK
6. **DETECTING_RACE_RANK** — waits 1s after FINISH, captures frame, fires async Gemini rank call (`gemini_rank.py`), transitions to READING_RACE_RESULTS immediately
7. **READING_RACE_RESULTS** — two paths based on whether a Gemini API key is configured:
   - **Gemini path**: collects frames while `has_race_results()` returns True (combines plus-cluster + bar-transition checks), fires one async Gemini call with all frames on transition out (`gemini_results.py`)
   - **OCR path**: accumulates placements across frames via `RaceResultDetector.detect()` with quality-based overwriting
   - Transitions to WAITING_FOR_TRACK_PICK when overall standings detected, or FINALIZING_MATCH after the final race. 30s timeout.
8. **FINALIZING_MATCH** — two paths:
   - **Gemini path**: detects CONGRATULATIONS!/NICE TRY! banner, fires async Gemini call (`gemini_match_results.py`), transitions to WAITING_FOR_MATCH immediately
   - **OCR path**: polls `MatchResultDetector` for full OCR results, then transitions to WAITING_FOR_MATCH

## Detection Patterns
- **Track selection screen**: left 42% of frame is very dark (player list panel), right side is colorful map. Track name OCR'd from tight banner ROI at y=33-37%, x=52-85%, upscaled 3x. Fuzzy-matched against 30 canonical track names (difflib, cutoff 0.6).
- **Match settings screen**: bright white card in center (mean brightness >150), bottom banner at y=94-98% contains "The rules have been decided!", settings parsed from whole card OCR at y=34-86%. Each setting fuzzy-matched against known valid values.
- **Player names**: left panel 2-column grid. Rows detected via horizontal brightness bands (merged if gap < 6px, filtered if height < 40px). Column split from brightness profile dip. Each cell: skip 17% for avatar, OTSU threshold, `--psm 7` OCR. Names cleaned of leading avatar noise and trailing artefacts.
- **Race finish screen**: FINISH! banner detected via HSV masking for orange-yellow text (H 15-35, S>150, V>180) in center ROI (x=15-82%, y=33-58%). Validated by pixel ratio bounds (0.20-0.35) and requiring orange in all 5 vertical strips (rejects GO! text and partial animations).
- **Race results (no-teams)**: right-side result bars (x=0.56-0.98). Hybrid threshold `(gray>170) & (blue>100)` isolates white text on grey/gold bars. Words classified by x-position: placement numbers (<0.62), icons (0.62-0.66, skipped), names (0.66-0.82), scores/plus (>0.82). `+` detected via OCR word matching `^[+#]\d+$`.
- **Race results (teams)**: red/blue bars where text takes on bar colour (grayscale ~100). Gaussian blur + Otsu on HSV V-channel for name reading. Gold first-place bar top 10% replaced with hybrid threshold. `+` detected via adaptive-V cluster counting on a narrow strip (3+ clusters = race results). Minimum 3 detected name-rows required to filter gameplay frame noise.
- **Race results screen detection** (`has_race_results`): combines plus-cluster detection with a bar-transition check (counts sharp brightness transitions between adjacent rows in the result bar region — results screens have 8+ transitions from bar edges, gameplay has 0-2). Both must pass to confirm race results are visible; this prevents gameplay frames with colourful scenery from being misclassified as result screens.

## Match Settings UI
- **Live View right panel** (270px wide): dropdowns for Class, Teams, Items, COM, Intermission and a spinbox for Race Count. On startup, UI defaults are pushed to the state machine (settings always available even when advancing past match detection). When match settings are detected from video, UI updates to show detected values (label: "Detected"). Reset State wipes detected settings and re-applies current UI values as manual. The `teams` setting controls which race result preprocessing path is used.
- **Settings tab** (top-level): Gemini API key field (password-masked, with eye toggle to reveal), model name field (defaults to `gemma-3-27b-it`), Save button, and Verify Key button. Status label shows verification result in green/red/grey. A `●` dot indicator in the Live View toolbar reflects the API key status at a glance without switching tabs. Auto-verifies on startup if a key is stored. Also hosts a **Debug** group with an "Enable debug mode" checkbox — toggling it persists `DEBUG_MODE=true|false` to `.env` via `debug_config.py` and updates `GameStateMachine.debug_mode` live. The state machine reads the flag from `.env` on init; downstream code can branch on `state_machine.debug_mode` to emit additional per-race logging.

## Gemini API
- Key and model stored in `.env` (gitignored) via `python-dotenv`
- `gemini_client.py` exposes `load_api_key`, `save_api_key`, `load_model`, `save_model`, `verify_api_key`
- Health check: `GET /v1beta/models/{model}?key={key}` — no tokens consumed
- Default model: `gemma-3-27b-it`
- Verification runs on a background `QThread` to avoid blocking the UI; concurrent verify requests are ignored while one is in flight
- **Three async Gemini integrations** (all fire-and-forget on daemon threads):
  1. **Race rank** (`gemini_rank.py`): single gameplay frame → `{"rank": N}`. Determines the active user's placement (1-24). Stored in `RaceInfo.race_rank`.
  2. **Race results** (`gemini_results.py`): multiple scrolling result frames → structured JSON with teams, mode, and per-player placements. Stored in `RaceInfo.placements` and `RaceInfo.gemini_results`.
  3. **Match results** (`gemini_match_results.py`): single CONGRATULATIONS/NICE TRY frame → structured JSON with teams, scores, and final standings. Stored in `GameStateMachine.match_final_results` and `gemini_match_results`.
- All three modules strip markdown code fences from responses (Gemini sometimes wraps JSON in `` ```json ... ``` `` despite instructions) and retry parsing after stripping.
- Each call writes a text log (`gemini_rank.txt`, `gemini_results.txt`, `gemini_match_results.txt`) to the race's debug folder with prompt, model, raw response, and parsed result (or error details).
- When no API key is configured, all three fall back to OCR-based detection with no Gemini calls.

## Debug Frame Saving
- Each match creates a timestamped folder under `matches/` (gitignored)
- `match_settings.png`, `match_results.png` at match level
- Per-race subfolder `race_NN/` contains: `track.png`, `players.png`, `finish.png`, `rank.png`, `placement_NN.png` (race result frames)
- Gemini request/response logs saved alongside frames: `gemini_rank.txt`, `gemini_results.txt`, `gemini_match_results.txt`
- Race result frames saved to `debug_placements/` (gitignored) when new placements are captured (temporary debugging)

## Match History Persistence
- Each match folder under `matches/<timestamp>/` also contains a `match.json` written by the state machine — the standardised, on-disk representation of the match. This is the data backing the match history UI (including the live, in-progress match); the debug folder structure doubles as the history store.
- The schema lives in [`match_record.py`](src/mktracker/match_record.py): `MatchRecord` → `RaceRecord` → `PlayerPlacement`/`TeamGroup`, plus `FinalStandings` and `MatchSettingsRecord`. It is a superset of both detection paths — `mode` and `teams` are populated from Gemini when available and left `None` for OCR-only matches. Schema version is tracked in `SCHEMA_VERSION` for future migrations.
- `match_record.list_matches(matches_dir)` returns every persisted match newest-first; legacy folders without a `match.json` are skipped silently.
- `GameStateMachine._save_match_record()` snapshots state and writes the JSON atomically (tmp file → rename). It is called after every meaningful update: settings detected, race added, race rank/results callbacks, OCR placement finalisation, and final-standings arrival. Partial matches are preserved on disk if the match never finalises.
- `_match_started_at` is set when settings are detected OR when `start_manual_match()` is called explicitly; `_match_completed_at` is set when final standings actually arrive (OCR or Gemini). Both are cleared by `reset()` along with `_match_dir`.
- **Strict persistence rule**: `_save_match_record()` refuses to write unless `_match_started_at` is set. This is the explicit opt-in that prevents "ghost matches" — folders created lazily by frame-saving handlers after a manual `advance()` — from polluting the history store with empty/partial records. The only two ways to set `_match_started_at` are real settings detection in `_handle_waiting` or an explicit `start_manual_match()` call.
- **`start_manual_match()`** primes `_match_started_at`, `_match_dir`, and `_match_seq` from the UI's currently-pushed `_match_settings`. It's wired to a "Start Manual Match" button in the toolbar. Required when the user advances past `WAITING_FOR_MATCH` manually and wants the resulting session persisted. No-op if a match is already in progress (use Reset first to start over). Returns `True` on success, `False` if it was a no-op.

## Table Generator
- **`src/mktracker/table_generator.py`** — `generate_table(record: MatchRecord) -> bytes` returns a PNG of a Lorenzi-style match result table.
- Fonts (Roboto variable, RubikMonoOne, NotoSans JP) are downloaded from the Google Fonts GitHub repo on first use and cached in `src/mktracker/assets/fonts/` (gitignored).
- **Single-clan (FFA)**: player rows span the full image width; no clan tag or total score shown.
- **Multi-clan (teams)**: three-column layout — rank + tag on left, player rows in centre, clan total score on right. A black gap with a subtle divider line and score-differential label (`+N`) separates teams.
- Clan background colour is derived from the team tag via a character hash (matching Lorenzi's `table.js` hue algorithm). FFA matches use the match ID as the hash seed so each match gets a distinct colour.
- Row backgrounds are lightened with a 58% white blend over the clan colour for contrast; all name/rank text is pure black.
- Japanese/CJK player names (Hiragana, Katakana, fullwidth/halfwidth forms) automatically use NotoSans JP as a fallback font.
- `scripts/generate_table.py` is a standalone CLI for generating a table from any saved match: `uv run python -m scripts.generate_table [match_id]`. Saves `table.png` alongside the match folder.
- **Live integration**: `GameStateMachine._save_match_record()` renders `table.png` into the match folder whenever `record.final_standings` is set — both in the live save path and the stale-callback path (`_apply_stale_match_results`). The table is written **before** `match.json` so the match-history tick loop (mtime-gated) never sees an updated record without its table. Rendering failures are logged but never block the JSON save.
- **Table editor**: `src/mktracker/lorenzi_text.py` round-trips `FinalStandings` to/from the Lorenzi line format (tag on one line, then `<name> <score>` rows; blank lines separate teams; scores may be `+`/`-` expressions like `70+20+8`). In the Match History detail pane, the `_TableImageCard` has "Copy Table" and "Edit Table" buttons in its header. "Edit Table" opens `_TableEditDialog` (QPlainTextEdit prefilled via `standings_to_text`); Save parses via `text_to_standings`, overwrites `record.final_standings` (places re-derived by score desc, winner = highest-points team, mode inferred from team count), regenerates `table.png`, and writes `match.json`. "Copy Table" puts the full-resolution PNG on the system clipboard via `QGuiApplication.clipboard().setPixmap()`. Edits are disabled for live matches.

## Match Record Backfill
- Legacy match folders that predate the `match.json` persistence layer can be reconstructed via `scripts/backfill_match_records.py`. It walks `matches/`, skips folders that already have `match.json` (unless `--force`), and rebuilds the record from whatever frames are saved on disk.
- Runs **fresh Gemini calls** (not the cached `gemini_*.txt` logs) for race rank, race results, and final match results — it imports the prompts and HTTP/parse helpers (`_PROMPT`, `_encode_frame`, `_query_gemini`, `_parse_rank`, `_parse_results`) from the live modules to stay in lockstep. OCR is used for match settings, track name, and player names (same as the live app).
- Placement frames are sampled evenly (default max 12 per race, configurable via `--max-placement-frames`) to keep Gemini payloads under limits — sending all 40+ frames per race produced HTTP 400s.
- Overwrites the `gemini_*.txt` debug logs alongside the frames, so stale/errored logs from the original runs are replaced.
- Invocation: `python -m scripts.backfill_match_records [--matches-dir matches] [--force] [--max-placement-frames 12]`. Requires a configured Gemini API key (via `.env` or the Settings tab).

## Data Model
- **`RaceInfo`** (frozen dataclass): `track_name`, `players`, `placements` (from OCR or Gemini), `race_rank` (user's placement from Gemini, async), `gemini_results` (full structured Gemini response dict, async). Both `race_rank` and `gemini_results` start as `None`; background Gemini callbacks replace the entry in `_races[i]` via `dataclasses.replace()` (a new frozen instance) + single-index list assignment (GIL-atomic in CPython).
- **`GameStateMachine`** stores: `_match_final_results` (list of `(name, score)` tuples), `_gemini_match_results` (full structured Gemini response dict), `_match_started_at` and `_match_completed_at` (datetimes for the persisted record).
- **`MatchRecord`** (in `match_record.py`): the standardised, JSON-serialisable form of a match. Every save call snapshots the state machine into one of these and writes it to `<match_dir>/match.json`.

## Known Limitations
- **Special Unicode characters** (☆, π, ★, ♪, ⊃) in player names are not reliably OCR'd by Tesseract (Gemini handles these well)
- **Team-mode OCR quality**: text on coloured bars has only ~7-10 levels of V-channel contrast with the background, causing some names to be partially garbled
- **FINISH! detection timing**: the FINISH banner is brief and can be missed if no frame captures it during the fast-sampling window (~100ms intervals)

## Tests

Run the full suite with:
```bash
uv run pytest tests/
```

- **`conftest.py`** — defines the `stub_fonts` session fixture, which patches `_ensure_font` (normally downloads TTF files) and `PIL.ImageFont.truetype` (normally opens them) so that `generate_table` rendering tests run without real font files in CI. A pre-built `load_default(size=20)` instance is created *before* the patch to avoid the recursion that would result from `load_default` calling `truetype` internally.
- **`test_table_generator.py`** — 43 tests split into pure-function tests (no I/O: `_needs_cjk`, `_blend`, `_hsv2rgb`, `_clan_hsv`, `_build_clans`) and rendering integration tests (require `stub_fonts`: FFA, 2-, 3-, 4-team layouts, height scaling, CJK names, tied rankings, ISO date parsing).
- **`test_player_reader.py`** — reads 12 tracked player-list frames from `tests/fixtures/player_reader/race_NN_players.png` (sourced from a two-team match with a static roster). Verifies that `PlayerReader.read_players(frame, teams=True)` produces 12 names and fuzzy-matches each expected roster name via multiset containment (two distinct in-game "Kod49" players share a name).
- **`test_match_results.py`** — 34 tests covering all layouts (no-teams ≤12, no-teams 24p, two-team ≤12), plus false-positive rejection for gameplay, player lists, FINISH! frames with a blue sky, and mid-race overall standings. Banner-specific fixtures live alongside the full-frame fixtures in `tests/fixtures/match_results/`.
- All other test files use real fixture images from `tests/fixtures/` (tracked, no `matches/` dependency) and `testdata/` (gitignored).

## Test Data
- `testdata/trackselected/` — 6 screenshots of track selection (filenames = track names)
- `testdata/match_settings/` — 2 screenshots of match settings screens
- `testdata/track_names.txt` — reference list of 30 tracks (canonical list lives in `tracks.py`)
- `testdata/samplerace.mp4` — full race at 2560x1440
- `testdata/realsamplerace.mp4` — full race at 1920x1080 (real capture card output)
- `testdata/race_finish/` — 1 FINISH! screenshot + 4 negative frames (GO!, partial animation, track select, results)

## Conventions
- All game logic lives in `state_machine.py` and `detection/` — the UI just renders and forwards frames
- Debug/info logging via Python `logging` module (INFO level by default)
- State transitions logged as `State: X -> Y`
- Track detections logged as `Race N/total: Track Name` followed by player list
- Race result placements logged as numbered list when overall standings are detected
- OCR ROI coordinates are normalized (0-1) proportions of frame dimensions (1920x1080 reference)
- Tesseract `--psm 7` for single-line text (player cells), `--psm 6` for blocks (match settings, race results), `--psm 8` for single word

## Running
Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates `.venv/` and installs the project in editable mode; `uv.lock` is committed for reproducible installs.

```bash
uv sync
uv run mktracker
```

## GitHub
Private repo: github.com/n4t3m/mkworld-tracker (branch: main)
