# MKWorld Tracker

For continuing development on another computer or in a new coding session,
see [AGENTS.md](AGENTS.md) for current scope, code locations, portable setup,
test commands, external assets and recording spot checks.

A desktop app that watches your Switch capture card and automatically logs Mario Kart World matches (settings, tracks, players, race placements, and final standings) using a mix of computer vision, OCR, and the Gemini API.

This is intended to be used to track war and mogi data over a long period of time.

## Demo

**Match start detection**

https://github.com/user-attachments/assets/1bab91df-a4b8-4238-9d11-cdeda0bbe834

**Track selection**

https://github.com/user-attachments/assets/89d95382-ad98-4ff3-8d38-a708a3c63ebd

**Race results**

https://github.com/user-attachments/assets/c462f28f-29c9-477e-9c7b-23d69a2da6dd

**Team match scoring**

Per-race team scores and race winner are calculated automatically from the detected placements.

![Team match result example](assets/images/warresultexample.png)

**Match end**

https://github.com/user-attachments/assets/d7ac429f-694b-42a3-90be-5c84c20f4a71

**Results table**

When a match ends, a result table is generated automatically from the final standings. The table is viewable in the Match History tab and can be copied to the clipboard or edited in place.

![Generated results table example](assets/images/resultstable.png)

**Editing the results table**

If any names or scores were misread, the table can be edited directly from the Match History tab via the **Edit Table** button. The dialog a similar format to Lorenzi's table editor (team tag followed by `<name> <score>` rows, blank lines between teams); saving re-derives placements, regenerates the PNG, and updates the stored match record.

![Editing the results table](assets/images/edittable.png)

**Menu navigation**

https://github.com/user-attachments/assets/44d16668-a95c-4bb5-8190-1f4e8add526e

**Discord webhook notifications**

Live match updates can be sent to a Discord channel automatically via a Discord webhook. Once a webhook URL is configured, the app posts an embed at each key moment of a match — no manual interaction required.

Match start — posts the detected match settings along with a snapshot of the rules screen:

![Match start webhook embed](assets/images/webhook_matchstart.png)

Race results — When the match is played in team mode, posts per-team placements, the race winner, and the point delta after every race:

![Race result webhook embed](assets/images/webhook_raceresult.png)

Match end — posts the final standings with the generated results table attached as an image:

![Match end webhook embed](assets/images/webhook_table.png)

Each notification is individually configurable from the **Settings** tab. Paste your webhook URL, hit **Send Test** to confirm it's reachable, then toggle the checkboxes under **Webhook Events** to pick which moments get posted (match start, per-race results, match end).

![Discord webhook settings](assets/images/webhook_settings.png)

## Features

- **Live game state detection**: This program will automatically detect when a match starts and track all information about the race, track selection, placements, etc. No manual interaction from you is needed.
- **Match settings recognition**: Auto-reads and adapts to the number of teams and race count and gives specialized info based on that.
- **Track detection**: Automatically recognizes each of the tracks being played.

The course catalog includes all 48 courses through game version 1.9.0: the
original 30, ten SNES courses from 1.8.0, and eight numbered variants from
1.9.0. Live detection, video analysis, and the match-history course picker share
this catalog. Course numbers are preserved during OCR matching. Added courses
display their names without artwork until dedicated icons are available.
- **Per-race placements**: Records the full finishing order after every race. This accounts for different team modes.
- **Final match standings**: Captures the final scores and standings at the end of every match automatically.
- **Persistent match history**: Every match you play is saved automatically and viewable later in a Match History tab, with a per-race breakdown.
- **Live match view**: Watch your in-progress match update live in the history tab as each race finishes.
- **Gemini integration (optional)**: A free Gemini API key can be used to drastically improve the detected names throughout the program.
- **Cross-platform support**: Works on Windows, macOS, and Linux with any USB capture card that delivers a 1920x1080 feed from your Switch.

## Prerequisites

### Capture card

The app reads frames from a USB capture card via the platform's native video backend (DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux) and lets you pick the device from the source dropdown in the toolbar. Any card that can deliver a **1920x1080** feed from your Switch will work — the detection ROIs and OCR thresholds are tuned against that resolution.

### Tesseract OCR

The program utilizes the Tesseract binary for OCR. It must be installed separately prior to running this application.

- **Windows**: install via the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). The default install location (`C:\Program Files\Tesseract-OCR\`) is auto-detected. If you install it anywhere else, add that directory to your `PATH`.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr` (or your distro's equivalent)

Install Tesseract even when using trained checkpoints: it remains the fallback
for layouts the local models do not recognize and for several glyph checks.

### Gemini API key (optional but strongly recommended)

The app uses local OCR and, when configured, Gemini to detect game state.
Configured trained checkpoints handle many local text fields; Tesseract handles
unrecognized layouts and systems without those checkpoints. Live race rank uses
the dark checkpoint with local HUD-shape corroboration; Gemini handles frames
without a reliable local rank when an API key is configured. Race results use trained
readers first for 12-player two-team and 24-player team or no-team rooms with a full
roster; complete, roster-checked final cards do likewise. The 24-player
final-card model runs in the background for no-team and two-, three-, or
four-team rooms. Incomplete or unsupported reads use Gemini.

The Gemini models we use are **free** on Google AI Studio's free tier. Grab a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and paste it into the **Settings** tab in the app; it'll be persisted to a local `.env` file. Without a key, the app still runs but falls back to OCR for everything.

Suggested models (configurable from the Settings tab):

- `gemma-4-31b-it` _(default)_ — newer, larger Gemma; better at recovering noisy or partially obscured frames at the cost of a bit more latency.
- `gemma-3-27b-it` — open-weights Gemma 3; a solid lighter-weight baseline that's accurate enough across all three tasks.
- `gemini-3.1-flash-lite-preview` — the best of the three when you can get it, but free-tier capacity is often exhausted during peak hours and requests will fail with a quota error until demand dies down. Worth trying first; fall back to one of the Gemma models if you start seeing rate-limit errors.

## Running

Live View and Video Review each have their own **Worldwides**, **Time Trial**,
and **Custom Room** mode selector. Both default to **Custom Room**, which uses the
existing match tracker. **Worldwides and Time Trial are planned features and
are not implemented.** Their selector options are placeholders; video can be
viewed, but neither mode has a tracking or analysis pipeline. The mode is
locked during an active custom-room match; use Reset State to end that session
before switching modes.

This modified version has been tested on Windows. macOS has not yet been
tested, including capture-card access and OCR setup.

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run mktracker
```

### Recorded-video review

In **Custom Room**, **Race position** above the video reads the current frame's
HUD locally during playback and after seeking, including while paused. A dash
means the HUD is hidden or the shape is uncertain; the previous position is
not carried into menus or other frames. Faded indicators are checked with
additional low-saturation and brightness masks, with stricter shape agreement
to distinguish the numeral from its translucent outline and background. This
also works when seeking directly into a fade, without an earlier solid reading.
Fully invisible, heavily obscured or ambiguous indicators can still show a dash.
The reader searches a broad lower-right area for aligned numeral/suffix shapes;
it does not require the baseline or right edge from the recording with black
bars. Limited aspect tolerance handles padding followed by capture resizing.
Regression tests include original capture-card frames, side/top/bottom and
asymmetric padding at several resolutions. Some gradient fills remain uncertain.
No API key or full-video rescan is
needed. This reader supports **1st–24th in the single-player, full-screen
layout**. Templates for 13th–24th come from the 24-player neutral HUD in
`GxDrSh-Ee20`; tests include held-out frames from that recording. Some
gradient-filled and heavily faded indicators still show a dash. Split-screen
and other HUD layouts need additional reference footage.
It is connected only to Video Review, not live tracking, and does not yet save
a position history to the timeline.
Video Review previews 60 fps sources at up to 30 displayed frames per second
while following the source clock, so HUD matching does not slow playback.

**Coins** reads the current frame's two-digit counter locally in Custom Room
Video Review. It finds the coin icon in the lower-left region and uses its
location and size to crop the digits, accommodating capture padding instead
of assuming the recording's black bars. Tests cover direct capture-card frames,
colored/neutral panels, yellow digits at 20, and added side/top/bottom padding.
It reads 0–20, including zero and pickup digit animations. A dash
means the icon is missing or the digits are uncertain. It updates on
playback and direct paused seeks without keeping a previous count. Templates
are bundled; no API key, game font, OCR process or rescan is needed. Other HUD
layouts, extreme padding/cropping and heavily faded counters may remain unknown. Counts are not yet
saved to the timeline or connected to Live View.

Open the **Video Review** tab, choose **Custom Room**, then **Open Video…** and select an MP4 (or another
OpenCV-supported video). Analysis starts locally in the background. The video
opens paused; use **Play/Pause** and the seek bar to review it while analysis
continues. Playback currently shows video only, without audio. No Gemini key is
required, and video analysis does not send webhooks.

**Live View** contains capture-device controls, live match settings, and live
tracking. **Video Review** contains the file player, analysis controls, and saved
race timeline. Opening a recording never resets live match tracking, and changing
capture devices never clears the video timeline. Switching away from Video Review
pauses recorded playback; return and press Play to continue. A video passed on
the command line opens directly in Video Review.

The timeline lists each detected course preview, finish, and result screen.
Click an entry once to pause and jump to that moment. Course preview timestamps
mark the map/player-list screen, not the starting gun. Boundaries are sampled
every half second. Player-name OCR is a draft and may be incomplete or wrong;
the result-screen entry lets you inspect the footage directly.
Short clips containing only a race result screen appear as **Result clip** entries
even when the finish banner and course preview are outside the file. **Refresh
Names** works on these clips too.
Name OCR is tuned for a 1920×1080 source. Video Review shows a resolution
notice for smaller recordings because upscaling a 720p copy cannot restore
the fine lettering lost during capture or compression.

Selecting a timeline entry also shows that race's detected names in the results
panel. **Refresh Names** re-reads just the saved race result windows and updates
that panel, with progress counted by race. It keeps your race timestamps and
playback position. The local team-results reader isolates name and placement
columns, samples the scrolling table up to four times per second, and combines
repeated readings. A configured trained name model reads each located row;
without it, Video Review uses the existing OCR reader. Decorative clan symbols
and occasional similar letters can still be misread; names remain OCR drafts.
After a complete scan or **Refresh Names**, names repeated across several
course previews can resolve clear spelling errors in result rows. The original
OCR reading remains available in the result tooltip, and uncertain matches
stay marked for review.

Result comparison samples roughly four frames per second while the table is
visible. It gives clearer reads more weight, compares close spelling variants,
and counts an identical image only once. Several standings frames must agree
before result collection ends. Readings with insufficient or conflicting
evidence show **[review]**; hover over a name for its frame count and agreement.
Agreement describes consistency between OCR readings, not a probability that
the spelling is correct. Similar names actually seen together remain separate.

The reader also samples the course-preview player list across multiple frames.
That roster supports close spelling decisions and highlights result names that
do not clearly match a preview player. Preview OCR is checked rather than
treated as ground truth: ambiguous or missing matches stay marked for review,
and a roster name is never invented in a finishing row. Hover over the results
panel title to inspect the preview roster, or over a result name to see its
preview match.

### Game-font names and symbols

In **Settings → Game font**, select your local game TTF/OTF file. The optional
font recognizer builds templates from every visible character in its Unicode
map, including accented letters, Greek, Cyrillic, kana and private-use icons.
It compares multiple render sizes and component groups to recover unfamiliar
tags in course previews and team results. Strong shape matches can correct
OCR; ambiguous differences retain the original spelling and request review.
Visually equivalent characters (such as Latin/Greek capital A) share an
identity while retaining the OCR spelling. Distinct accents and symbols stay
separate. The results panel uses the supplied font to display custom icons.
Use **Refresh Names** to update previously saved readings.

Run `python scripts/audit_font_coverage.py --output analysis/font-audit --sizes 64,44,32`
to test every visible glyph in an unfamiliar tag. The supplied font has 748
visible glyphs: the current synthetic audit matches all at 64px and 44px, and
746 at 32px; the two small-size errors are rejected as ambiguous. This checks
rendered glyphs, not every possible compression, camera angle or background.

The font stays at the selected local path; its path is stored in the ignored
`.env` file as `GAME_FONT_PATH`. **Clear** disables font verification. Font
integration tests use that setting (or the environment variable of the same
name) and skip when no local reference font is available.

### Trained local OCR

The trained PaddleOCR recognizer reads names and ranks on dark race-result
screens during initial video analysis and **Refresh Names**. A separate colored
checkpoint handles colored result screens and preview rosters; set
`NAME_OCR_COLOR_CONFIG` and `NAME_OCR_COLOR_CHECKPOINT` to enable it. The colored model's
reads are checked against visible tag symbols and word spacing across the
board; ambiguous raised symbols remain marked for review. A complete colored
12-player board can stop sampling once each name has three supporting frames.
Video Review also stops reading a 12-player course-preview roster once two
frames establish all 12 names, reducing repeated model work during full scans.
The configured checkpoints also read course selections, room settings, live
player lists, and final-standings names and scores. Yellow and green match
standings use the colored checkpoint when available.
Video Review can also start a race from the trained model's read of the
countdown course card when the recording cuts away before the map preview.
The card must match a known course; its timestamp is treated as a sampled
race boundary. A real three-team
24-player race screen uses dark result bars, which the dark-model path detects
with its wider score column and row spacing. A separate four-team per-race
screen uses all four colored bars; its shifted grid now reaches the optional
colored model. The configured game font can check a few ambiguous `0/O` and
`1/I` result glyphs; short-name and dotted-initial corrections are marked
for review. Names remain drafts across recording conditions.
Set `NAME_OCR_PYTHON` to a separate Python executable with PaddlePaddle
installed, `NAME_OCR_PADDLEOCR` to the PaddleOCR source directory,
`NAME_OCR_CONFIG` to the trained YAML config, and `NAME_OCR_CHECKPOINT` to its
`.pdparams` file. Put these paths in the ignored `.env` file or environment;
`.env.example` lists the keys. The config's `character_dict_path` must also
point to the model's character list. The app keeps one worker loaded per
configured checkpoint while reading result frames.
If a checkpoint is unavailable or a screen layout is not recognized, the
existing OCR reader remains available. Live rank checks the trained dark model
against the visible HUD shape, using the local shape read when the model misses
and Gemini when no reliable local HUD rank is present. Across the two local
VODs, the local route read all 24 finish frames at the capture timing; the 11
24-player ranks with independent result-screen labels agreed. 12-player two-team and 24-player
team or no-team race boards can use the local model first with complete roster
evidence. Dark 24-player team boards without visible color bars continue to
the existing results reader. Complete 12-player two-team final cards also use
the local model.
No-team final cards use the dark model when every name agrees with a complete
prior race roster; obscured or partial cards continue to the existing fallback. The
Match History table-regeneration action also tries a roster-checked local model
read from its saved screenshot before using Gemini when configured. The
race-placement regeneration action does the same for verified 12-player
two-team, 24-player colored-team, and 24-player no-team saved result frames; incomplete reads retain
the Gemini fallback when an API key is configured. The
local model path handles the other live text and the no-key race and match
results. On complete colored team boards, the local path also
preserves team membership and calculates race and final team totals; partial
boards keep the existing flat result fallback.

### OCR development probes

Run these from the repository root after installing the dependencies. They are
developer scripts, excluded from the installed application. Inputs and output
directories are explicit; generated files under `analysis/` are ignored by Git.

The labeled name-crop corpus, training scripts, and held-out evaluation log live
in the separate private [mkworld-tracker-ocr](https://github.com/n4t3m/mkworld-tracker-ocr)
repository. This tracker contains
the runtime readers and screen-level regression fixtures; it runs without the
training checkout. Install selected checkpoints separately as described above.

Inspect result names at selected recording timestamps (in seconds):

```bash
uv run python scripts/probe_names.py "gameplay.mp4" --times 243 244.5 --output analysis/names
```

This writes normalized screenshots, deskewed name-row sheets and `readings.json`
using the current custom-room team-result reader. The game font is optional.

Inspect every visible result row in a screenshot with the full-font recognizer:

```bash
uv run python scripts/probe_font_symbols.py "results.jpg" --font "game-font.ttf" --output analysis/glyphs
```

Omit `--font` to use the configured game font. Add `--rows 1 2 9` to inspect
specific one-based visible rows (not racer ranks). Output includes row images
and `glyphs.json` with shape scores, margins, visual alternatives and verification
flags. Unverified glyph predictions are diagnostic candidates, not accepted OCR
corrections. Both probes expect custom-room team result screens; other game
screens may contain no detectable rows.

Timelines are saved in `video_timelines/` and reload when the same file is opened
again. Changing the source file invalidates its cached timeline. **Analyze Video**
runs a fresh scan. Seeking never mutates the saved timeline. Videos that begin
mid-race may have an unknown course and a start marked as needing review.
Unknown course labels are left unresolved unless they match the course list or
a confidently read console-prefixed preset label.

You can also open a recording directly:

```powershell
uv run mktracker "C:\path\to\gameplay.mp4"
```

This workspace's installed environment is `.venv-app`. To run it directly on
Windows:

```powershell
.\.venv-app\Scripts\python.exe -m mktracker.main "C:\path\to\gameplay.mp4"
```

Or use `Start-Tracker.ps1`, which selects the installed environment and sets the
working directory so saved timelines are found consistently:

```powershell
.\Start-Tracker.ps1 "C:\path\to\gameplay.mp4"
```
