"""Step 2 - English subtitles via Whisper (offline, no API).

Whisper is asked for word-level timestamps and subtitles are rebuilt from those
words at sentence boundaries. Its raw segments break mid-clause, which produces
cues ending on a dangling "and"; rebuilding avoids that.
"""
from __future__ import annotations

import gc
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass

from pipeline.runtime import cpu_limit, log_memory, stage

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")   # tiny|base|small|medium
# Burned-in appearance. A solid box beats an outline here: the footage is
# mostly white whiteboard, and white-on-white outlined text disappears.
# libass ignores BackColour alpha with BorderStyle=3, so the box is opaque.
SUB_FONTSIZE = int(os.environ.get("SUB_FONTSIZE", "14"))
SUB_MARGIN = int(os.environ.get("SUB_MARGIN", "22"))
SUB_STYLE = os.environ.get("SUB_STYLE", "")   # full override, wins if set

MAX_CUE = 84       # characters per subtitle
MAX_LINE = 42      # characters per displayed line (2 lines max)
MAX_GAP = 0.7      # a pause longer than this always starts a new cue
MIN_DUR = 1.0
MAX_DUR = 6.0

_model = None


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # cpu_limit(), not os.cpu_count(): see pipeline/runtime.py
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8",
                              cpu_threads=cpu_limit())
    return _model


def unload() -> None:
    """Drop Whisper from memory.

    Whisper and NLLB are never needed at the same moment, and holding both
    costs ~1GB against the container's ceiling. Transcription finishes before
    translation starts, so releasing here is free.
    """
    global _model
    if _model is not None:
        _model = None
        gc.collect()
        log_memory("after unloading Whisper")


def _join_fragments(words):
    """Whisper emits 'short' + '-circuit' as two tokens; glue them back."""
    out = []
    for w in words:
        if out and w["w"][:1] in ("-", "'", "’", ",", ".", "!", "?", ";", ":"):
            out[-1]["w"] += w["w"]
            out[-1]["e"] = w["e"]
        else:
            out.append(dict(w))
    return out


def _build_cues(words) -> list[Cue]:
    words = _join_fragments(words)
    cues, cur = [], []

    def flush():
        if not cur:
            return
        text = re.sub(r"\s+([,.!?;:])", r"\1", " ".join(w["w"] for w in cur))
        cues.append(Cue(cur[0]["s"], max(cur[-1]["e"], cur[0]["s"] + MIN_DUR), text))
        cur.clear()

    for w in words:
        pending = len(" ".join(x["w"] for x in cur)) + 1 + len(w["w"])
        gap = w["s"] - cur[-1]["e"] if cur else 0
        span = w["e"] - cur[0]["s"] if cur else 0
        if cur and (pending > MAX_CUE or gap > MAX_GAP or span > MAX_DUR):
            flush()
        cur.append(w)
        if re.search(r"[.!?]$", w["w"]):
            flush()
    flush()
    return cues


def wrap(t: str) -> str:
    if len(t) <= MAX_LINE:
        return t
    lines, cur = [], ""
    for w in t.split():
        if cur and len(cur) + 1 + len(w) > MAX_LINE:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        mid = len(t) // 2
        i = min((m.end() for m in re.finditer(r"\s", t)), key=lambda e: abs(e - mid))
        return t[:i].strip() + "\n" + t[i:].strip()
    return "\n".join(lines)


def _ts(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


# U+200F, zero width. A right-to-left script is only rendered right-to-left if
# the renderer decides the line's base direction is RTL - and text editors
# routinely default to LTR, which throws a trailing full stop to the wrong end
# of the line. libass gets this right on its own, so burned-in subtitles were
# never affected; the file being unreadable in an editor still is. One
# invisible character at the head of each line settles it everywhere.
RLM = "\u200f"


def write_srt(cues: list[Cue], path: str, rtl: bool = False) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            text = wrap(c.text)
            if rtl:
                text = "\n".join(RLM + line for line in text.split("\n"))
            f.write(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{text}\n\n")
    return path


def transcribe(video: str, progress=lambda pct, msg: None) -> list[Cue]:
    progress(0.05, "Extracting audio")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                       check=True)
        progress(0.15, f"Loading Whisper ({MODEL_SIZE})")
        model = _load()
        progress(0.25, "Transcribing")
        with stage(f"whisper transcribe ({MODEL_SIZE})"):
            segs, _info = model.transcribe(
                wav, language="en", beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,   # stops invented text over music
            )
            words = [{"w": w.word.strip(), "s": w.start, "e": w.end}
                     for sg in segs for w in (sg.words or []) if w.word.strip()]
    log_memory("after transcribe")
    progress(0.9, f"{len(words)} words transcribed")
    return _build_cues(words)


def attach(video: str, srt_paths: dict[str, str], out_path: str,
           burn: str | None = None) -> str:
    """srt_paths maps an ISO-639-2 code ('eng', 'fas') to a .srt file.

    burn = a language code to render into the picture (permanent), or None for
    soft tracks the viewer can toggle. Soft is safer for Farsi: the player does
    the right-to-left shaping instead of libass.
    """
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video]

    if burn:
        font = "Noto Naskh Arabic" if burn == "fas" else "DejaVu Sans"
        style = SUB_STYLE or (
            f"FontName={font},Fontsize={SUB_FONTSIZE},"
            "PrimaryColour=&H00FFFFFF,"      # white text
            "BackColour=&H00000000,"         # solid black box behind it
            "BorderStyle=3,"                 # 3 = box (not outline)
            "Outline=1.2,"                   # padding around the text
            "Shadow=0,"
            f"MarginV={SUB_MARGIN}"
        )
        srt = srt_paths[burn].replace("\\", "/").replace(":", r"\:")
        cmd += ["-vf", f"subtitles={srt}:force_style='{style}'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "copy"]
    else:
        for path in srt_paths.values():
            cmd += ["-i", path]
        cmd += ["-map", "0:v", "-map", "0:a"]
        for n in range(len(srt_paths)):
            cmd += ["-map", str(n + 1)]
        cmd += ["-c", "copy", "-c:s", "mov_text"]
        for n, lang in enumerate(srt_paths):
            cmd += [f"-metadata:s:s:{n}", f"language={lang}"]

    cmd.append(out_path)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")
    return out_path
