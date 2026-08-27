# Marefat Video Builder

Upload a lesson clip, get it back with the Marefat branding wrapped around it —
and optional English or Farsi subtitles.

```
opening card (10s) → intro → your clip (+ animated logo) → outro
```

Everything runs on free, open-source tools. No API keys, no per-video cost.

## What it does

| Step | Tool | Notes |
|---|---|---|
| Branding | FFmpeg | normalises to 1280×720 / 30fps / stereo 48kHz before joining |
| Transcription | faster-whisper (`small`) | word-level timestamps, English |
| Translation | NLLB-200 distilled 600M | English → Farsi, reuses the English timings |
| Output | FFmpeg | soft subtitle tracks, or burned into the picture |

## Why the pieces are the way they are

**Everything is normalised before concatenation.** The source clips disagree —
the lessons are 720p/30fps/mono, the intro and outro are 1080p/25fps/stereo.
Concatenating those directly produces freezes and dropped audio. Every segment
is forced through one filter chain first.

**Subtitles are rebuilt from word timestamps, not Whisper's own segments.**
Whisper breaks segments mid-clause, which yields cues ending on a dangling
"and". `pipeline/subtitles.py` repacks the words at sentence boundaries.

**Translation never recomputes timings.** The Farsi track reuses the English
cue boundaries exactly, so the two stay in sync and translation stays cheap.

**Soft subtitles are the default.** The viewer can toggle them, and the video
player handles right-to-left shaping for Farsi. Burning in works too (tested,
renders correctly with Noto Naskh Arabic) but is permanent — use it when the
target platform ignores subtitle tracks.

## Deploying to Railway

1. Create a new project from this repo. The `Dockerfile` is detected
   automatically.
2. **Add a volume mounted at `/data`.** Without it, the ~3GB of models is
   re-downloaded on every restart. With it, only the first boot is slow.
3. Deploy. First boot takes several minutes while models download; after that
   a 3-minute video processes in roughly 90 seconds.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` — bigger is slower and more accurate |
| `TRANSLATE_MODEL` | `facebook/nllb-200-distilled-600M` | swap for `...-1.3B` for better Persian (needs ~5.5GB RAM) |
| `HF_HOME` | `/data/hf` | model cache — point at the volume |

### Memory

Whisper `small` and NLLB 600M loaded together need roughly 4GB. That fits
Railway's Hobby plan. Moving to NLLB 1.3B pushes it to around 7GB — check your
plan's ceiling before switching.

## Running locally

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
streamlit run app.py
```

FFmpeg must be on your PATH.

## A caution on the Farsi

NLLB-200 is a translation model, not a translator. It handles plain statements
well and flattens nuance in contemplative or metaphorical passages — which this
material has a lot of. Treat the Farsi output as a first draft: export the
`.srt`, correct it, and re-attach.

## Repository layout

```
app.py                 Streamlit interface
pipeline/assemble.py   step 1 — branding
pipeline/subtitles.py  step 2 — transcription, cue building, attaching
pipeline/translate.py  step 3 — English → Farsi
assets/                opening card, intro, outro, animated logo (~21MB)
```

The animated logo was re-encoded from a 177MB QuickTime RLE original to a 6.3MB
PNG-in-MOV at 400px — transparency intact, and small enough for git.
