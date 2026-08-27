# Marefat Video Builder

Upload a lesson clip, get it back with the Marefat branding wrapped around it —
and optional English or Farsi subtitles.

```
opening card (3s) → intro → your clip (+ animated logo) → outro
```

Everything runs on free, open-source tools. No API keys, no per-video cost.

## What it does

| Step | Tool | Notes |
|---|---|---|
| Branding | FFmpeg | normalises to 1280×720 / 30fps / stereo 48kHz before joining |
| Transcription | faster-whisper (`small`) | word-level timestamps, English |
| Translation | NLLB-200 distilled 1.3B (CTranslate2 int8) | English → Farsi, reuses the English timings |
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

**Sentences are translated, not subtitle lines.** A cue is a display unit, not
a grammatical one — roughly a third of them are half-sentences. NLLB was
trained on whole sentences, and given a fragment it invents the missing half.
In a measured run, "starve the ego" came back as "starve *yourself*", and
"a puzzle that *you* can manage" as "that *I* manage" — the fragment carried no
subject, so the model chose one. `pipeline/translate.py` joins consecutive cues
into sentences (51 cues → 37 sentences on a three-minute lesson), translates
those, then redistributes the Persian across the original cues in proportion to
how much English each carried. No split across languages is exact; each cue
holding roughly its share at the right moment beats translating fragments.

**A glossary fixes what context cannot.** Word-sense errors survive perfect
context — this material's "partner" is a spouse, and the model reads it as a
work colleague every time. `assets/glossary.json` is a Persian-to-Persian
correction list applied after translation, matched on whole words so `همکار`
never fires inside `همکاری`. Add an entry when a wrong rendering appears twice;
keep the note explaining why, so the next person can judge whether it still
holds.

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
| `TRANSLATE_MODEL` | `OpenNMT/nllb-200-distilled-1.3B-ct2-int8` | must be a **CTranslate2** conversion, not a PyTorch repo |
| `TRANSLATE_TOKENIZER` | `facebook/nllb-200-distilled-1.3B` | vocabulary only, ~17MB |
| `HF_HOME` | `/data/hf` | model cache — point at the volume |
| `CPU_LIMIT` | auto-detected | override the core count if detection is wrong |
| `TRANSLATE_BEAMS` | `2` | raise to 4 for slightly better Persian, double the time |
| `TRANSLATE_BATCH` | `8` | sentences translated at once; higher uses more memory |
| `GLOSSARY` | `assets/glossary.json` | path to the correction list |
| `SUB_FONTSIZE` / `SUB_MARGIN` | `14` / `22` | burned-in subtitle size and position |

### Why threads are capped

`os.cpu_count()` reports the **host's** cores and ignores the container's
cgroup quota. On an 8 vCPU service running on a much larger machine it can
return 64 — and every library that sizes its thread pool that way then spawns
64 workers for 8 usable cores.

That costs twice over: context-switch contention, and one memory arena per
thread. In practice it turned a ~3.5GB workload into 7GB against an 8GB
ceiling, and a job that should take two minutes ran for forty-five without
finishing, because the container kept being restarted.

`pipeline/runtime.py` reads the real limit from cgroup v2, then v1, then
scheduler affinity, and `pipeline/__init__.py` exports it as `OMP_NUM_THREADS`
*before* torch or ctranslate2 import — they read it at load time, so setting it
afterwards does nothing.

### Why CTranslate2 and not PyTorch

A Railway **Hobby volume is capped at 5GB** and cannot be resized past it. The
float32 PyTorch build of NLLB-1.3B is ~5.5GB, so it does not fit — with an
empty volume, let alone alongside Whisper.

The same checkpoint converted to CTranslate2 int8 is **~1.4GB**: smaller than
the 600M PyTorch model it replaces, while being the larger and better model.
CTranslate2 also arrives free with faster-whisper and outperforms PyTorch for
CPU inference, so the image drops `torch` entirely.

`TRANSLATE_MODEL` must therefore name a **CTranslate2 conversion**. Pointing it
at `facebook/nllb-200-...` will fail — those are PyTorch repos.

### Disk and memory

| | |
|---|---|
| Whisper `small` | ~0.5GB on disk, ~690MB resident |
| NLLB 1.3B ct2-int8 | ~1.4GB on disk |
| Tokenizer | ~17MB |
| **Volume total** | **~2GB** of the 5GB Hobby cap |

Whisper is released before the translator loads (measured: 654MB → 325MB), so
the two never occupy memory together.

### Reading the logs

Every stage prints its duration and the memory around it:

```
[runtime] cpu_limit=8 (os.cpu_count reports 64)
[stage] whisper transcribe (small) done in 41.2s
[mem] now 654 MB, peak 693 MB after transcribe
[mem] now 325 MB, peak 693 MB after unloading Whisper
[stage] translate 51 lines (beams=2, batch=4) done in 96.4s
```

If a run is slow, that tells you which stage and whether memory is climbing —
no need to read CPU graphs.

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
