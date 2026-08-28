"""Marefat Video Builder - upload a clip, get it back branded and subtitled."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from pipeline import assemble as A
from pipeline import subtitles as S
from pipeline.runtime import stage

st.set_page_config(page_title="Marefat Video Builder", page_icon="🎬")

LANGS = {"English": "eng", "Farsi": "fas"}


# ----------------------------------------------------------------- helpers

def show_result(key: str) -> None:
    """Render a finished job from session state.

    Kept out of the button blocks on purpose: every button press - download
    buttons included - reruns this script from the top. Holding the result in
    a local variable meant a download click wiped the video the user had just
    waited minutes for.
    """
    result = st.session_state.get(key)
    if not result:
        return

    video = result["video"]
    st.success(f"Ready — {len(video) / 1e6:.1f} MB")
    st.video(video)

    st.download_button("Download video", video, type="primary",
                       file_name=f"{result['name']}.mp4", mime="video/mp4",
                       key=f"dl_video_{key}")

    for lang, blob in result.get("subtitles", {}).items():
        st.download_button(f"Download {lang}.srt", blob, mime="text/plain",
                           file_name=f"{result['name']}_{lang}.srt",
                           key=f"dl_{lang}_{key}")

    if st.button("Clear", key=f"clear_{key}"):
        st.session_state.pop(key, None)
        st.rerun()


def save_upload(upload, workdir: str, stem: str) -> str:
    path = os.path.join(workdir, stem + os.path.splitext(upload.name)[1])
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path


# -------------------------------------------------------------------- page

st.caption("PROOF OF CONCEPT")
st.title("Marefat Video Builder")
st.caption(
    "Adds the opening card, intro, animated logo and outro. "
    "Subtitles are optional."
)

with st.expander("راهنما  ·  How this works"):
    components.html(
        (Path(__file__).parent / "assets" / "info_panel.html").read_text("utf-8"),
        height=760,
        scrolling=True,
    )

tab_build, tab_subs = st.tabs([
    "ساخت ویدیو  ·  Build video",
    "جایگزینی زیرنویس  ·  Replace subtitles",
])


# ------------------------------------------------------------- tab 1: build

with tab_build:
    video_file = st.file_uploader(
        "Main video", type=["mp4", "mov", "mkv", "m4v"],
        help="The lesson itself. Everything else is added around it.",
    )

    col1, col2 = st.columns(2)
    with col1:
        pic_choice = st.radio("Opening picture",
                              ["Default Marefat card", "Upload my own"])
        custom_pic = None
        if pic_choice == "Upload my own":
            custom_pic = st.file_uploader("Opening picture",
                                          type=["jpg", "jpeg", "png"])
    with col2:
        sub_choice = st.radio(
            "Subtitles", ["None", "English", "Farsi", "Both"],
            help="Farsi is machine-translated from the English transcript.",
        )

    # Subtitles are attached as a toggleable track, never burned in here.
    # Burning re-encodes the whole video, and doing it now would spend those
    # minutes on text nobody has proofread yet. Burn once, on the corrected
    # version, in the second tab.
    st.caption(
        "Subtitles are added as a track you can switch on and off. "
        "To burn them into the picture, correct them first and use "
        "**Replace subtitles**."
    )

    if st.button("Build video", type="primary", disabled=video_file is None):
        st.session_state.pop("build", None)
        bar, status = st.progress(0.0), st.empty()

        def report(pct: float, msg: str):
            bar.progress(min(max(pct, 0.0), 1.0))
            status.write(msg)

        workdir = tempfile.mkdtemp()
        try:
            src = save_upload(video_file, workdir, "main")
            pic = save_upload(custom_pic, workdir, "pic") if custom_pic else None

            branded = os.path.join(workdir, "branded.mp4")
            with stage("assemble"):
                A.assemble(src, branded, picture=pic,
                           progress=lambda p, m: report(p * 0.4, m))
            result, subtitle_files = branded, {}

            if sub_choice != "None":
                cues = S.transcribe(branded,
                                    progress=lambda p, m: report(0.4 + p * 0.3, m))
                srts = {}
                if sub_choice in ("English", "Both"):
                    srts["eng"] = S.write_srt(cues, os.path.join(workdir, "en.srt"))

                if sub_choice in ("Farsi", "Both"):
                    # free Whisper before the translator loads, so the two
                    # never occupy the container's memory together
                    S.unload()
                    from pipeline import translate as T
                    fa = T.translate_cues(
                        cues, progress=lambda p, m: report(0.7 + p * 0.2, m))
                    srts["fas"] = S.write_srt(fa, os.path.join(workdir, "fa.srt"),
                                              rtl=True)

                report(0.92, "Attaching subtitles")
                result = S.attach(branded, srts,
                                  os.path.join(workdir, "final.mp4"))
                subtitle_files = {lang: open(p, "rb").read()
                                  for lang, p in srts.items()}

            report(1.0, "Done")
            with open(result, "rb") as f:
                st.session_state["build"] = {
                    "video": f.read(),
                    "subtitles": subtitle_files,
                    "name": f"{Path(video_file.name).stem}_marefat",
                }
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            raise
        finally:
            # every build wrote ~40MB here; without this the container's disk
            # fills up run by run until it dies
            shutil.rmtree(workdir, ignore_errors=True)

    show_result("build")


# --------------------------------------------------- tab 2: replace subtitles

with tab_subs:
    st.markdown(
        "Corrected a subtitle file by hand? Put it back on the video here — "
        "no need to rebuild, and this works days later, from any computer."
    )

    sub_video = st.file_uploader("Video", type=["mp4", "mov", "mkv", "m4v"],
                                 key="sv")
    sub_file = st.file_uploader("Corrected subtitle file (.srt)", type=["srt"],
                                key="ss")

    c1, c2 = st.columns(2)
    with c1:
        sub_lang = st.selectbox("Language of this file", list(LANGS))
    with c2:
        mode = st.radio(
            "How to add it",
            ["Toggleable track", "Burn into the picture"],
            help="Burning is permanent and re-encodes the video, so it takes "
                 "minutes. Needed for platforms that ignore subtitle tracks.",
        )

    ready = sub_video is not None and sub_file is not None
    if st.button("Apply subtitles", type="primary", disabled=not ready):
        st.session_state.pop("subs", None)
        status = st.empty()
        workdir = tempfile.mkdtemp()
        try:
            code = LANGS[sub_lang]
            vpath = save_upload(sub_video, workdir, "video")
            spath = os.path.join(workdir, f"{code}.srt")
            with open(spath, "wb") as f:
                f.write(sub_file.getbuffer())

            burning = mode.startswith("Burn")
            status.write("Re-encoding with burned-in subtitles…" if burning
                         else "Attaching subtitle track…")
            with stage("burn" if burning else "attach"):
                out = S.attach(vpath, {code: spath},
                               os.path.join(workdir, "out.mp4"),
                               burn=code if burning else None)

            with open(out, "rb") as f:
                st.session_state["subs"] = {
                    "video": f.read(),
                    "subtitles": {},
                    "name": f"{Path(sub_video.name).stem}_{code}",
                }
            status.write("Done")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    show_result("subs")
