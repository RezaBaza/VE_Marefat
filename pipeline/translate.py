"""Step 3 - translate English subtitles to Farsi with NLLB-200.

Timings are never recomputed. The cues from step 2 keep their exact start and
end; only the text changes.

**Sentences are translated, not subtitle lines.** A cue is a display unit, not
a grammatical one - "into a massive argument about the relationship." arrives
with no subject and no antecedent. NLLB was trained on whole sentences, and
fed a fragment it invents the missing half: in one measured run "starve the
ego" came back as "starve yourself", because nothing in the fragment said what
was being starved. So consecutive cues are joined into sentences, translated
whole, and the Persian is then redistributed across the original cues in
proportion to how much of the English each carried.

Runs the model through CTranslate2 rather than PyTorch. Three reasons:

* the 1.3B checkpoint ships pre-quantised to int8 at ~1.4GB, against ~5.5GB
  for the float32 PyTorch weights - and a Railway Hobby volume is capped at
  5GB, so the PyTorch version simply does not fit;
* CTranslate2 already arrives as a faster-whisper dependency, so this costs no
  new packages and lets the image drop torch entirely;
* it is markedly faster than PyTorch for CPU inference, which is all this
  container has.

Only the tokenizer comes from the original Meta repo - ~17MB of vocabulary
files, not the weights.
"""
from __future__ import annotations

import json
import os
import re

from pipeline.runtime import cpu_limit, log_memory, stage

MODEL_NAME = os.environ.get(
    "TRANSLATE_MODEL", "OpenNMT/nllb-200-distilled-1.3B-ct2-int8")
TOKENIZER_NAME = os.environ.get(
    "TRANSLATE_TOKENIZER", "facebook/nllb-200-distilled-1.3B")

SRC_LANG = "eng_Latn"
TGT_LANG = "pes_Arab"      # Western Persian, Arabic script
BATCH = int(os.environ.get("TRANSLATE_BATCH", "8"))
BEAMS = int(os.environ.get("TRANSLATE_BEAMS", "2"))

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assets")
GLOSSARY_PATH = os.environ.get("GLOSSARY", os.path.join(ASSETS, "glossary.json"))

# Persian letters, for whole-word matching. Arabic-script words are not
# delimited by \b in a way Python's re can use, so neighbours are checked
# explicitly instead.
_PERSIAN = r"؀-ۿ‌"

_tok = _translator = None
_glossary: list[tuple[re.Pattern, str]] | None = None


# ----------------------------------------------------------------- model

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


# -------------------------------------------------------------- glossary

def _load_glossary():
    """Compile the correction list, longest phrase first."""
    global _glossary
    if _glossary is None:
        try:
            with open(GLOSSARY_PATH, encoding="utf-8") as f:
                terms = json.load(f).get("terms", [])
        except FileNotFoundError:
            terms = []

        # longest first, so a phrase wins over a word nested inside it
        terms.sort(key=lambda t: len(t["from"]), reverse=True)
        _glossary = [
            (re.compile(f"(?<![{_PERSIAN}]){re.escape(t['from'])}(?![{_PERSIAN}])"),
             t["to"])
            for t in terms if t.get("from") and t.get("to")
        ]
        if _glossary:
            print(f"[glossary] {len(_glossary)} corrections loaded", flush=True)
    return _glossary


def apply_glossary(text: str) -> str:
    for pattern, replacement in _load_glossary():
        text = pattern.sub(replacement, text)
    return text


# ------------------------------------------------------------- sentences

_SENTENCE_END = re.compile(r"[.!?…]['\"»]?\s*$")


def group_into_sentences(cues):
    """Consecutive cues that together make one sentence.

    A cue ending in terminal punctuation closes the group. Anything still open
    at the end forms a final group, so nothing is dropped when the transcript
    ends mid-thought.
    """
    groups, current = [], []
    for cue in cues:
        current.append(cue)
        if _SENTENCE_END.search(cue.text.strip()):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def split_proportionally(text: str, weights: list[int]) -> list[str]:
    """Divide translated text across cues by each cue's share of the English.

    Word order differs between the languages, so no split is exact. What this
    guarantees is that each cue holds roughly its share of the sentence and
    stays on screen at the right moment - which beats translating the fragments
    separately, where the model invents the missing context outright.
    """
    n = len(weights)
    if n == 1:
        return [text]

    words = text.split()
    if len(words) <= n:                       # too short to divide sensibly
        return [text] + [""] * (n - 1)

    total = sum(weights) or 1
    parts, taken = [], 0
    for i in range(n - 1):
        cumulative = sum(weights[:i + 1]) / total
        target = round(len(words) * cumulative)
        # leave at least one word for every cue still to come
        target = max(taken + 1, min(target, len(words) - (n - i - 1)))
        parts.append(" ".join(words[taken:target]))
        taken = target
    parts.append(" ".join(words[taken:]))
    return parts


# ------------------------------------------------------------ translate

def _translate_batch(texts: list[str]) -> list[str]:
    tok, translator = _load()
    tokenized = [tok.convert_ids_to_tokens(tok.encode(t)) for t in texts]
    results = translator.translate_batch(
        tokenized,
        # forces the output language - the CTranslate2 equivalent of
        # forced_bos_token_id in transformers
        target_prefix=[[TGT_LANG]] * len(texts),
        beam_size=BEAMS,
        max_decoding_length=256,
    )
    out = []
    for res in results:
        tokens = res.hypotheses[0]
        if tokens and tokens[0] == TGT_LANG:
            tokens = tokens[1:]
        out.append(tok.decode(tok.convert_tokens_to_ids(tokens),
                              skip_special_tokens=True).strip())
    return out


def translate_cues(cues, progress=lambda pct, msg: None):
    """Return new cues with Farsi text and identical timings."""
    from pipeline.subtitles import Cue

    progress(0.05, f"Loading {MODEL_NAME}")
    _load()

    groups = group_into_sentences(cues)
    sentences = [" ".join(" ".join(c.text.split()) for c in g) for g in groups]
    print(f"[translate] {len(cues)} cues -> {len(sentences)} sentences", flush=True)

    translated: list[str] = []
    with stage(f"translate {len(sentences)} sentences "
               f"(beams={BEAMS}, batch={BATCH})"):
        for i in range(0, len(sentences), BATCH):
            translated += _translate_batch(sentences[i:i + BATCH])
            done = min(i + BATCH, len(sentences))
            progress(0.05 + 0.85 * done / max(len(sentences), 1),
                     f"Translated {done}/{len(sentences)} sentences")

    progress(0.92, "Applying glossary")
    out: list[Cue] = []
    for group, farsi in zip(groups, translated):
        farsi = apply_glossary(farsi)
        weights = [max(len(c.text), 1) for c in group]
        for cue, piece in zip(group, split_proportionally(farsi, weights)):
            out.append(Cue(cue.start, cue.end, piece.strip()))

    log_memory("after translate")
    return out
