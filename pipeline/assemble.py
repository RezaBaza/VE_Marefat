"""Step 1 - assemble the branded video.

    opening picture (10s) -> intro -> main video (+ animated logo) -> outro

Every clip is forced through the same normalisation filter first, because
concatenating clips with different resolutions, frame rates or audio layouts
produces freezes and dropped sound.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# ---- house style --------------------------------------------------------
WIDTH, HEIGHT, FPS = 1280, 720, 30
AUDIO_RATE = 48000
PIC_SECONDS = 10          # how long the opening picture holds
LOGO_WIDTH = 170          # logo width in px
LOGO_MARGIN = 28          # distance from the top-right corner

VF = (
    f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
    f"setsar=1,fps={FPS},format=yuv420p"
)
VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
AENC = ["-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2"]


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")


def duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def assemble(main_video: str, out_path: str, picture: str | None = None,
             progress=lambda pct, msg: None) -> str:
    picture = picture or os.path.join(ASSETS, "opening_pic.jpg")
    intro = os.path.join(ASSETS, "intro.mp4")
    outro = os.path.join(ASSETS, "outro.mp4")
    logo = os.path.join(ASSETS, "logo.mov")

    with tempfile.TemporaryDirectory() as tmp:
        j = lambda n: os.path.join(tmp, n)

        progress(0.05, "Building the opening card")
        run(["ffmpeg", "-y", "-v", "error",
             "-loop", "1", "-t", str(PIC_SECONDS), "-i", picture,
             "-f", "lavfi", "-t", str(PIC_SECONDS),
             "-i", f"anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE}",
             "-vf", VF, *VENC, *AENC, "-shortest", j("01.mp4")])

        progress(0.15, "Normalising the intro")
        run(["ffmpeg", "-y", "-v", "error", "-i", intro, "-vf", VF,
             *VENC, *AENC, j("02.mp4")])

        progress(0.25, "Encoding the main video and overlaying the logo")
        run(["ffmpeg", "-y", "-v", "error", "-i", main_video, "-i", logo,
             "-filter_complex",
             f"[0:v]{VF}[base];"
             f"[1:v]scale={LOGO_WIDTH}:-1,format=rgba,fps={FPS}[lg];"
             f"[base][lg]overlay=W-w-{LOGO_MARGIN}:{LOGO_MARGIN}"
             f":eof_action=pass:format=auto[v]",
             "-map", "[v]", "-map", "0:a", *VENC, *AENC, j("03.mp4")])

        progress(0.75, "Normalising the outro")
        run(["ffmpeg", "-y", "-v", "error", "-i", outro, "-vf", VF,
             *VENC, *AENC, j("04.mp4")])

        progress(0.85, "Joining everything together")
        listing = j("list.txt")
        with open(listing, "w") as f:
            for n in ("01.mp4", "02.mp4", "03.mp4", "04.mp4"):
                f.write(f"file '{j(n)}'\n")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", listing, "-c", "copy", out_path])

    progress(0.95, "Video assembled")
    return out_path
