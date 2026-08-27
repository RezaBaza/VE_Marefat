"""Step 3 - translate English subtitles to Farsi with NLLB-200 (open source).

Timings are never recomputed. The cues from step 2 keep their exact start/end;
only the text changes. That is why translation is cheap next to transcription.

Sized for a CPU container rather than a GPU:

* weights are int8-quantised at load, cutting memory roughly 4x and speeding
  up matmuls on CPU - a 600M seq2seq model in float32 needs ~2.4GB before any
  activations, which is most of an 8GB budget once Whisper has also run;
* `num_beams=2` instead of 4, halving the work for a difference that does not
  survive a human correction pass anyway;
* threads are capped to the cgroup limit (see pipeline/runtime.py).

Model is swappable via TRANSLATE_MODEL.
"""
from __future__ import annotations

import os

from pipeline.runtime import cpu_limit, log_memory, stage

MODEL_NAME = os.environ.get("TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M")
SRC_LANG = "eng_Latn"
TGT_LANG = "pes_Arab"      # Western Persian, Arabic script
BATCH = int(os.environ.get("TRANSLATE_BATCH", "4"))
BEAMS = int(os.environ.get("TRANSLATE_BEAMS", "2"))
QUANTIZE = os.environ.get("TRANSLATE_QUANTIZE", "1") != "0"

_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(cpu_limit())

        with stage(f"load {MODEL_NAME}"):
            _tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            model.eval()

            if QUANTIZE:
                with stage("quantise to int8"):
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8)
            _model = model
        log_memory("after loading translator")
    return _tok, _model


def _target_token_id(tok) -> int:
    """Resolve the Farsi language token.

    NLLB moved this between transformers versions, so both spellings are
    tried. The important part is rejecting the unknown-token id:
    `convert_tokens_to_ids` returns UNK rather than failing for a token it
    does not know, and forcing UNK as the first decoder token makes the model
    emit fluent nonsense until it hits max_length - slow *and* wrong, with
    nothing in the output to say the language was never selected.
    """
    unk = getattr(tok, "unk_token_id", None)

    for getter in (
        lambda: tok.convert_tokens_to_ids(TGT_LANG),
        lambda: tok.lang_code_to_id[TGT_LANG],
        lambda: tok.get_vocab()[TGT_LANG],
    ):
        try:
            tid = getter()
        except Exception:
            continue
        if tid is not None and tid >= 0 and tid != unk:
            return int(tid)

    raise RuntimeError(
        f"{MODEL_NAME} does not expose the language token {TGT_LANG!r}. "
        "Check that TRANSLATE_MODEL is an NLLB-200 checkpoint."
    )


def translate_cues(cues, progress=lambda pct, msg: None):
    """Return new cues with Farsi text and identical timings."""
    import torch

    progress(0.05, f"Loading {MODEL_NAME}")
    tok, model = _load()
    forced = _target_token_id(tok)

    # subtitles are wrapped across two lines for display; translate them flat
    texts = [" ".join(c.text.split()) for c in cues]
    out: list[str] = []

    with stage(f"translate {len(texts)} lines (beams={BEAMS}, batch={BATCH})"):
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=192)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=forced,
                                     max_new_tokens=128, num_beams=BEAMS)
            out += tok.batch_decode(gen, skip_special_tokens=True)
            done = min(i + BATCH, len(texts))
            progress(0.05 + 0.9 * done / max(len(texts), 1),
                     f"Translated {done}/{len(texts)} lines")

    log_memory("after translate")
    from pipeline.subtitles import Cue
    return [Cue(c.start, c.end, t.strip()) for c, t in zip(cues, out)]
