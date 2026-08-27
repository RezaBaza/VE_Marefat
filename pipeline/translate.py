"""Step 3 - translate English subtitles to Farsi with NLLB-200 (open source).

Timings are never recomputed. The cues produced in step 2 keep their exact
start/end; only the text changes. That is why translation is cheap compared to
transcription.

Model is swappable through TRANSLATE_MODEL, so you can move up to
facebook/nllb-200-distilled-1.3B (better Persian, ~5.5GB RAM) without code
changes.
"""
from __future__ import annotations

import os

MODEL_NAME = os.environ.get("TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M")
SRC_LANG = "eng_Latn"
TGT_LANG = "pes_Arab"      # Western Persian, Arabic script
BATCH = 8

_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tok, _model


def _target_token_id(tok):
    """NLLB moved this around between transformers versions."""
    for getter in (
        lambda: tok.convert_tokens_to_ids(TGT_LANG),
        lambda: tok.lang_code_to_id[TGT_LANG],
    ):
        try:
            tid = getter()
            if tid is not None and tid >= 0:
                return tid
        except Exception:
            continue
    raise RuntimeError(f"could not resolve target language token {TGT_LANG}")


def translate_cues(cues, progress=lambda pct, msg: None):
    """Return new cues with Farsi text and identical timings."""
    import torch

    progress(0.05, f"Loading {MODEL_NAME}")
    tok, model = _load()
    forced = _target_token_id(tok)

    # subtitles are wrapped for display; translate them as flat sentences
    texts = [" ".join(c.text.split()) for c in cues]
    out: list[str] = []

    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=forced,
                                 max_length=256, num_beams=4)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        progress(0.05 + 0.9 * (i + len(batch)) / max(len(texts), 1),
                 f"Translated {min(i + BATCH, len(texts))}/{len(texts)} lines")

    from pipeline.subtitles import Cue
    return [Cue(c.start, c.end, t.strip()) for c, t in zip(cues, out)]
