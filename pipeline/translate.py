"""Step 3 - translate English subtitles to Farsi with NLLB-200.

Timings are never recomputed. The cues from step 2 keep their exact start/end;
only the text changes. That is why translation is cheap next to transcription.

Runs the model through CTranslate2 rather than PyTorch. Three reasons:

* the 1.3B checkpoint ships pre-quantised to int8 at ~1.4GB, against ~5.5GB
  for the float32 PyTorch weights - which matters because a Railway Hobby
  volume is capped at 5GB, so the PyTorch version simply does not fit;
* CTranslate2 is already installed as a faster-whisper dependency, so this
  costs no new packages and lets the image drop torch entirely;
* it is markedly faster than PyTorch for inference on CPU, which is the only
  hardware this container has.

Only the tokenizer comes from the original Meta repo - a ~17MB download of
vocabulary files, not the weights.
"""
from __future__ import annotations

import os

from pipeline.runtime import cpu_limit, log_memory, stage

MODEL_NAME = os.environ.get(
    "TRANSLATE_MODEL", "OpenNMT/nllb-200-distilled-1.3B-ct2-int8")
TOKENIZER_NAME = os.environ.get(
    "TRANSLATE_TOKENIZER", "facebook/nllb-200-distilled-1.3B")

SRC_LANG = "eng_Latn"
TGT_LANG = "pes_Arab"      # Western Persian, Arabic script
BATCH = int(os.environ.get("TRANSLATE_BATCH", "8"))
BEAMS = int(os.environ.get("TRANSLATE_BEAMS", "2"))

_tok = _translator = None


def _load():
    global _tok, _translator
    if _translator is None:
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        with stage(f"fetch {MODEL_NAME}"):
            model_dir = snapshot_download(MODEL_NAME)

        with stage("load tokenizer + translator"):
            _tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, src_lang=SRC_LANG)
            _translator = ctranslate2.Translator(
                model_dir,
                device="cpu",
                compute_type="int8",
                # one model, several cores: parallelism belongs inside an op,
                # not across replicas of the model
                inter_threads=1,
                intra_threads=cpu_limit(),
            )
        log_memory("after loading translator")
    return _tok, _translator


def translate_cues(cues, progress=lambda pct, msg: None):
    """Return new cues with Farsi text and identical timings."""
    progress(0.05, f"Loading {MODEL_NAME}")
    tok, translator = _load()

    # subtitles are wrapped across two lines for display; translate them flat
    texts = [" ".join(c.text.split()) for c in cues]
    out: list[str] = []

    with stage(f"translate {len(texts)} lines (beams={BEAMS}, batch={BATCH})"):
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]

            # CTranslate2 works in tokens, not ids
            tokenized = [tok.convert_ids_to_tokens(tok.encode(t)) for t in batch]

            results = translator.translate_batch(
                tokenized,
                # forces the output language - the NLLB equivalent of
                # forced_bos_token_id in transformers
                target_prefix=[[TGT_LANG]] * len(batch),
                beam_size=BEAMS,
                max_decoding_length=192,
            )

            for res in results:
                # hypothesis starts with the language token we forced
                tokens = res.hypotheses[0]
                if tokens and tokens[0] == TGT_LANG:
                    tokens = tokens[1:]
                out.append(tok.decode(tok.convert_tokens_to_ids(tokens),
                                      skip_special_tokens=True))

            done = min(i + BATCH, len(texts))
            progress(0.05 + 0.9 * done / max(len(texts), 1),
                     f"Translated {done}/{len(texts)} lines")

    log_memory("after translate")
    from pipeline.subtitles import Cue
    return [Cue(c.start, c.end, t.strip()) for c, t in zip(cues, out)]
