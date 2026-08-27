"""Marefat Video Builder - upload a clip, get it back branded and subtitled."""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from pipeline import assemble as A
from pipeline import subtitles as S
from pipeline.runtime import cpu_limit, stage

st.set_page_config(page_title="Marefat Video Builder", page_icon="🎬")

# small line above the title, so "POC" reads as a status label rather than
# part of the product's name
st.caption("PROOF OF CONCEPT")
st.title("Marefat Video Builder")
st.caption(
    "Adds the opening card, intro, animated logo and outro. "
    "Subtitles are optional."
)

with st.expander("راهنما  ·  How this works"):
    st.markdown(
        """
<div dir="rtl" style="line-height:2;">

**ساختار هر ویدیو**

۱. تصویر آغازین — ۳ ثانیه &nbsp;·&nbsp; ۲. کلیپ ابتدایی — ۸ ثانیه &nbsp;·&nbsp;
۳. ویدیوی درس، با لوگوی متحرک در گوشه &nbsp;·&nbsp; ۴. کلیپ پایانی — ۱۲ ثانیه

همهٔ بخش‌ها پیش از اتصال به یک کیفیت، نرخ فریم و قالب صدای یکسان تبدیل می‌شوند تا
در محل اتصال پرش تصویر یا قطعی صدا رخ ندهد.

**زیرنویس**

گفتار انگلیسی نخست به متن تبدیل می‌شود و سپس همان متن به فارسی ترجمه می‌گردد.
زمان‌بندی هر دو زبان یکسان است. **ترجمه ماشینی است** و برای انتشار عمومی باید
بازبینی شود — فایل زیرنویس جداگانه قابل دانلود و ویرایش است.

هیچ ویدیو یا متنی به سرویس بیرونی فرستاده نمی‌شود؛ همهٔ پردازش روی همین سرور
انجام می‌گیرد.

</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
---

1. **Branding** — your clip is normalised to 1280×720 / 30fps / stereo 48kHz,
   then joined behind the opening card and intro and ahead of the outro. The
   animated logo plays once over the first seconds of your clip.
2. **Subtitles** *(optional)* — Whisper transcribes the finished video and
   subtitles are rebuilt from word-level timestamps at sentence boundaries.
3. **Farsi** *(optional)* — NLLB-200 translates the English lines. Timings are
   reused exactly, so the two languages stay in sync.

Nothing is sent to an external service — all three models run in this container.
"""
    )

video_file = st.file_uploader(
    "Main video", type=["mp4", "mov", "mkv", "m4v"],
    help="The lesson itself. Everything else is added around it.",
)

col1, col2 = st.columns(2)

with col1:
    pic_choice = st.radio("Opening picture", ["Default Marefat card", "Upload my own"])
    custom_pic = None
    if pic_choice == "Upload my own":
        custom_pic = st.file_uploader("Opening picture", type=["jpg", "jpeg", "png"])

with col2:
    sub_choice = st.radio(
        "Subtitles",
        ["None", "English", "Farsi", "Both"],
        help="Farsi is machine-translated from the English transcript.",
    )

burn_lang = None
if sub_choice != "None":
    burn = st.checkbox(
        "Burn subtitles into the picture",
        help="Off = a track the viewer can switch on and off (recommended, and "
             "safer for Farsi right-to-left text). On = permanently visible, "
             "needed for social media.",
    )
    if burn:
        if sub_choice == "Both":
            which = st.selectbox("Which language to burn in?", ["English", "Farsi"])
        else:
            which = sub_choice
        burn_lang = {"English": "eng", "Farsi": "fas"}[which]

st.divider()

if st.button("Build video", type="primary", disabled=video_file is None):
    bar = st.progress(0.0)
    status = st.empty()

    def report(pct: float, msg: str):
        bar.progress(min(max(pct, 0.0), 1.0))
        status.write(msg)

    workdir = tempfile.mkdtemp()
    try:
        src = os.path.join(workdir, "main" + os.path.splitext(video_file.name)[1])
        with open(src, "wb") as f:
            f.write(video_file.getbuffer())

        pic_path = None
        if custom_pic is not None:
            pic_path = os.path.join(workdir, "pic" + os.path.splitext(custom_pic.name)[1])
            with open(pic_path, "wb") as f:
                f.write(custom_pic.getbuffer())

        # ---- step 1: branding ------------------------------------------
        branded = os.path.join(workdir, "branded.mp4")
        with stage("assemble"):
            A.assemble(src, branded, picture=pic_path,
                       progress=lambda p, m: report(p * 0.4, m))
        result = branded

        # ---- step 2 + 3: subtitles -------------------------------------
        if sub_choice != "None":
            cues = S.transcribe(branded, progress=lambda p, m: report(0.4 + p * 0.3, m))

            srts: dict[str, str] = {}
            if sub_choice in ("English", "Both"):
                srts["eng"] = S.write_srt(cues, os.path.join(workdir, "en.srt"))

            if sub_choice in ("Farsi", "Both"):
                # Whisper is done; free it before the translator loads, so the
                # two never occupy the container's memory at the same time
                S.unload()
                from pipeline import translate as T
                fa = T.translate_cues(cues, progress=lambda p, m: report(0.7 + p * 0.2, m))
                srts["fas"] = S.write_srt(fa, os.path.join(workdir, "fa.srt"))

            report(0.92, "Attaching subtitles")
            subbed = os.path.join(workdir, "final.mp4")
            result = S.attach(branded, srts, subbed, burn=burn_lang)

            for lang, path in srts.items():
                st.download_button(
                    f"Download {lang}.srt",
                    open(path, "rb").read(),
                    file_name=f"{lang}.srt",
                    mime="text/plain",
                )

        report(1.0, "Done")
        with open(result, "rb") as f:
            data = f.read()

        st.success(f"Ready - {len(data) / 1e6:.1f} MB")
        st.video(data)
        st.download_button("Download video", data, file_name="marefat_final.mp4",
                           mime="video/mp4", type="primary")

    except Exception as exc:  # surfaced in the UI rather than only in logs
        st.error(f"Something went wrong: {exc}")
        raise
