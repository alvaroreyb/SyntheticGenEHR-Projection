
"""
Unified clinical NER pipeline for Spanish patient notes.

This script combines four extraction workflows over Spanish clinical .txt files:
1. Token-classification NER with BSC models.
2. Token-classification NER with MedSpaner models.
3. Regex-based Gleason score extraction.
4. Regex-based PSA value extraction.

The pipeline reads .txt files from the configured input folders, writes workflow-
specific JSON and CSV outputs under OUT_DIR/etiquetas/, and generates merged
Brat .ann files.

Main robustness improvements in this version:
- Long documents are processed in character chunks with overlap instead of
  tokenizing the full document at once.
- Per-patient, per-model, and per-chunk progress is printed.
- Text is normalized before inference.
- Runtime errors during inference are caught so one problematic file does not
  block the entire pipeline.
- Batch size is automatically reduced if inference fails for a chunk.
"""

import argparse
import os
import re
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


DEFAULT_UNTAGGED_DIR = ""

UNTAGGED_DIR = os.environ.get("GENEHR_UNTAGGED_DIR", DEFAULT_UNTAGGED_DIR)

WF1_INPUT_DIR = WF2_INPUT_DIR = WF3_INPUT_DIR = WF4_INPUT_DIR = UNTAGGED_DIR
OUT_DIR = ANN_DIR = os.path.join(UNTAGGED_DIR, "SilverMentions")
WF1_MODELS = [
    "BSC-NLP4BIA/bsc-bio-ehr-es-livingner-species",
    "BSC-NLP4BIA/bsc-bio-ehr-es-livingner-humano",
    "BSC-NLP4BIA/bsc-bio-ehr-es-medprocner",
    "BSC-NLP4BIA/bsc-bio-ehr-es-symptemist",
    "BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
    "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
]

WF2_MODELS = [
    "medspaner/roberta-es-clinical-trials-cases-medic-attr",
    "medspaner/xlm-roberta-large-spanish-trials-cases-temp-ent",
    "medspaner/roberta-es-clinical-trials-cases-neg-spec",
]

MIN_SCORE = 0.85
MAX_SCORE = 1.00
WINDOW_STRIDE_TOKENS = 192
BATCH_WINDOWS_WF1 = 16
BATCH_WINDOWS_WF2 = 8

SNIPPET_WINDOW = 35
MAX_CHARS_PER_DOC = 400_000

NER_CHAR_CHUNK_SIZE = 40_000
NER_CHAR_CHUNK_OVERLAP = 2_000
MAX_WINDOWS_PER_CHUNK = 2000
EMPTY_TEXT_PLACEHOLDER = ""

_ETIQUETAS = os.path.join(OUT_DIR, "etiquetas") if OUT_DIR else ""

WF1_JSON_DIR = os.path.join(_ETIQUETAS, "BSC", "json_por_paciente")
WF1_OUT_CSV = os.path.join(_ETIQUETAS, "BSC", "menciones_ner_BSC.csv")

WF2_JSON_DIR = os.path.join(_ETIQUETAS, "medspaner", "json_por_paciente")
WF2_OUT_CSV = os.path.join(_ETIQUETAS, "medspaner", "menciones_ner_medspaner.csv")

WF3_OUT_CSV = os.path.join(_ETIQUETAS, "gleason", "gleason_mentions_simple.csv")
WF4_OUT_CSV = os.path.join(_ETIQUETAS, "psa", "psa_mentions_simple.csv")


def configure_paths(untagged_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Configure input/output paths after parsing command-line arguments.
    """
    global UNTAGGED_DIR, WF1_INPUT_DIR, WF2_INPUT_DIR, WF3_INPUT_DIR, WF4_INPUT_DIR
    global OUT_DIR, ANN_DIR, _ETIQUETAS
    global WF1_JSON_DIR, WF1_OUT_CSV, WF2_JSON_DIR, WF2_OUT_CSV, WF3_OUT_CSV, WF4_OUT_CSV

    UNTAGGED_DIR = untagged_dir
    WF1_INPUT_DIR = WF2_INPUT_DIR = WF3_INPUT_DIR = WF4_INPUT_DIR = UNTAGGED_DIR
    OUT_DIR = ANN_DIR = output_dir or os.path.join(UNTAGGED_DIR, "SilverMentions")
    _ETIQUETAS = os.path.join(OUT_DIR, "etiquetas")

    WF1_JSON_DIR = os.path.join(_ETIQUETAS, "BSC", "json_por_paciente")
    WF1_OUT_CSV = os.path.join(_ETIQUETAS, "BSC", "menciones_ner_BSC.csv")

    WF2_JSON_DIR = os.path.join(_ETIQUETAS, "medspaner", "json_por_paciente")
    WF2_OUT_CSV = os.path.join(_ETIQUETAS, "medspaner", "menciones_ner_medspaner.csv")

    WF3_OUT_CSV = os.path.join(_ETIQUETAS, "gleason", "gleason_mentions_simple.csv")
    WF4_OUT_CSV = os.path.join(_ETIQUETAS, "psa", "psa_mentions_simple.csv")


def ensure_configured_directories() -> None:
    """
    Create configured output directories.
    """
    for d in [
        WF1_JSON_DIR,
        WF2_JSON_DIR,
        os.path.join(_ETIQUETAS, "gleason"),
        os.path.join(_ETIQUETAS, "psa"),
        ANN_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


_INTRASPAN_GLUE = re.compile(r"^[\s\-\u2010\u2011\u2012\u2013\u2014\/,;:.()]+$")

GLEASON_KW = r"(?:\bgleason\b|puntuaci[oó]n\s+de\s+gleason|score\s+de\s+gleason|escala\s+de\s+gleason)"
ROMAN_TOT_RE = r"(?:X|IX|IV|V?I{0,3})"
TOT_RE = rf"(?:10|[2-9]|{ROMAN_TOT_RE})"
PS_RE = r"(?:[1-5])"
SEP_RE = r"(?:\+|/)"

SUM_TOTAL_RE = re.compile(
    rf"(?P<prefix>{GLEASON_KW})(?:\s*[:=]?\s*)"
    rf"(?P<p1>{PS_RE})\s*{SEP_RE}\s*(?P<p2>{PS_RE})"
    rf"(?:\s*(?:=|→|->)\s*(?P<tot>{TOT_RE}))?",
    flags=re.IGNORECASE,
)

TOTAL_THEN_SUM_RE = re.compile(
    rf"(?P<prefix>{GLEASON_KW})(?:\s*[:=]?\s*)"
    rf"(?P<tot>{TOT_RE})"
    rf"(?:\s*[\(\[\{{]\s*(?P<p1>{PS_RE})\s*{SEP_RE}\s*(?P<p2>{PS_RE})\s*[\)\]\}}])",
    flags=re.IGNORECASE,
)

SUM_IN_PARENS_RE = re.compile(
    rf"(?P<prefix>{GLEASON_KW})(?:\s*[:=]?\s*)"
    rf"(?P<tot>{TOT_RE})?"
    rf"(?:\s*[\(\[\{{]\s*(?P<p1>{PS_RE})\s*{SEP_RE}\s*(?P<p2>{PS_RE})\s*[\)\]\}}])",
    flags=re.IGNORECASE,
)

SCORE_ONLY_RE = re.compile(
    rf"(?P<prefix>{GLEASON_KW})(?:\s*[:=]?\s*)(?P<tot>{TOT_RE})\b",
    flags=re.IGNORECASE,
)

_UNIT_RE = r"(?:ng\s*/\s*mL|ng\s*/\s*ml|ng\s*/\s*L|ng\s*/\s*l|ng\s*mL|ng\s*ml|ng\s*L|ng\s*l)"
_NUM_RE = r"(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:[.,]\d+)?"

_VALUE_WITH_UNIT_RE = re.compile(
    rf"(?P<value>{_NUM_RE})\s*(?P<unit>{_UNIT_RE})\b",
    flags=re.IGNORECASE,
)


def clear_ann_dir(ann_dir: str) -> None:
    """
    Remove all .ann files from the annotation output directory.

    Arguments:
        ann_dir: Directory containing Brat .ann files.
    Return:
        None.
    """
    if not os.path.isdir(ann_dir):
        return

    removed = 0
    for fn in os.listdir(ann_dir):
        path = os.path.join(ann_dir, fn)
        if os.path.isfile(path) and fn.lower().endswith(".ann"):
            os.remove(path)
            removed += 1

    print(f"  Removed {removed} existing .ann files from {ann_dir}")


def read_txt_folder(input_dir: str) -> pd.DataFrame:
    """
    Collect all .txt files recursively and build a DataFrame.

    Arguments:
        input_dir: Root folder to search for .txt files.
    Return:
        DataFrame with columns patient_id, text, file_name, file_path.
    """
    txt_files = sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(input_dir)
        for fn in files
        if fn.lower().endswith(".txt")
    )

    rows = []
    for pid, path in enumerate(txt_files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                text = f.read()

        rows.append(
            {
                "patient_id": pid,
                "text": text,
                "file_name": os.path.basename(path),
                "file_path": path,
            }
        )

    print(f"  Found {len(rows)} .txt files in {input_dir}")
    return pd.DataFrame(rows)


def write_ann(file_path: str, entities: List[Dict]) -> None:
    """
    Write a Brat standoff .ann file for a source .txt file.

    Arguments:
        file_path: Path to the source .txt file.
        entities: Merged entity list. Each entity must contain label, start, end, text.
    Return:
        None.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    ann_path = os.path.join(ANN_DIR, stem + ".ann")

    sorted_ents = sorted(entities, key=lambda e: (int(e["start"]), int(e["end"])))

    lines = []
    for i, e in enumerate(sorted_ents, start=1):
        label = str(e["label"]).replace(" ", "_")
        text = str(e["text"]).replace("\n", " ").replace("\r", " ")
        lines.append(f"T{i}\t{label} {e['start']} {e['end']}\t{text}")

    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def clean_span_text(s: str) -> str:
    """
    Normalize whitespace in an extracted span.

    Arguments:
        s: Raw span string.
    Return:
        Cleaned span string.
    """
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


clean_text_token = clean_span_text


def normalize_input_text(text: str) -> str:
    """
    Normalize raw document text before model inference.

    Arguments:
        text: Raw input document text.
    Return:
        Normalized document text.
    """
    t = str(text) if text is not None else ""
    t = t.replace("\x00", " ")
    t = t.replace("\ufeff", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t if t else EMPTY_TEXT_PLACEHOLDER


def get_effective_max_length(tokenizer, model, cap: int = 0) -> int:
    """
    Resolve the effective maximum sequence length for a model/tokenizer pair.

    Arguments:
        tokenizer: Hugging Face tokenizer instance.
        model: Hugging Face model instance.
        cap: Optional hard upper bound. Use 0 for no cap.
    Return:
        Effective maximum input length.
    """
    model_max = getattr(model.config, "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)

    candidates = []
    if isinstance(model_max, int) and model_max > 0:
        candidates.append(model_max)
    if isinstance(tok_max, int) and tok_max > 0 and tok_max < 1_000_000:
        candidates.append(tok_max)

    result = int(min(candidates)) if candidates else 512
    if cap > 0:
        result = min(result, cap)

    return result


def softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax over the last tensor dimension.

    Arguments:
        x: Logits tensor.
    Return:
        Probability tensor.
    """
    return torch.softmax(x, dim=-1)


def aggregate_bio_spans(
    full_text: str,
    offsets,
    label_ids,
    label_scores,
    id2label: dict,
    avg_scores: bool = False,
) -> List[Dict]:
    """
    Convert per-token BIO predictions into character-level entity spans.

    Arguments:
        full_text: Complete document text.
        offsets: Token offset list.
        label_ids: Predicted label ids per token.
        label_scores: Confidence scores per token.
        id2label: Mapping from label id to label string.
        avg_scores: If True, average token scores across the span. Otherwise use the minimum.
    Return:
        List of entity dictionaries with start, end, label, score, text.
    """
    entities = []
    current = None

    def close_current():
        nonlocal current
        if current is None:
            return

        if avg_scores and "scores_list" in current and current["scores_list"]:
            current["score"] = sum(current["scores_list"]) / len(current["scores_list"])
            del current["scores_list"]

        if current["end"] > current["start"]:
            span_text = clean_text_token(full_text[current["start"]: current["end"]])
            if span_text:
                current["text"] = span_text
                entities.append(current)

        current = None

    for idx, ((st, en), lab_id, sc) in enumerate(zip(offsets, label_ids, label_scores)):
        if st is None or en is None or (st == 0 and en == 0):
            if idx > 0:
                close_current()
            continue

        if en <= st:
            continue

        lab = id2label.get(int(lab_id), str(lab_id))
        if lab == "O":
            close_current()
            continue

        if "-" in lab:
            prefix, typ = lab.split("-", 1)
            prefix = prefix.upper()
        else:
            prefix, typ = "B", lab

        if current is not None and current["label"] == typ:
            is_contiguous = st == current["end"]

            if is_contiguous or prefix == "I":
                if not is_contiguous and prefix == "I":
                    gap = full_text[current["end"]: st]
                    if gap and not _INTRASPAN_GLUE.match(gap):
                        close_current()
                        current = None

                if current is not None:
                    current["end"] = int(max(current["end"], en))
                    if avg_scores:
                        current["scores_list"].append(float(sc))
                    else:
                        current["score"] = min(current["score"], float(sc))
                    continue

            elif prefix == "B":
                gap_ok = True
                if st > current["end"]:
                    gap = full_text[current["end"]: st]
                    gap_ok = bool(_INTRASPAN_GLUE.match(gap)) if gap else True

                if gap_ok:
                    current["end"] = int(max(current["end"], en))
                    if avg_scores:
                        current["scores_list"].append(float(sc))
                    else:
                        current["score"] = min(current["score"], float(sc))
                    continue
                else:
                    close_current()
            else:
                close_current()
        else:
            close_current()

        if avg_scores:
            current = {
                "start": int(st),
                "end": int(en),
                "label": typ,
                "scores_list": [float(sc)],
                "text": "",
            }
        else:
            current = {
                "start": int(st),
                "end": int(en),
                "label": typ,
                "score": float(sc),
                "text": "",
            }

    close_current()
    return entities


def deduplicate_entities_by_span_label(entities: List[Dict]) -> List[Dict]:
    """
    Remove duplicate entities when they share the same label and either the same
    start offset or the same end offset.

    If two entities with the same label overlap under this rule, only the one
    with the longest character span is preserved.

    Arguments:
        entities: List of entity dictionaries.
    Return:
        Deduplicated list of entity dictionaries.
    """
    kept: Dict[Tuple, Dict] = {}

    for ent in entities:
        ent_label = str(ent["label"])
        ent_start = int(ent["start"])
        ent_end = int(ent["end"])
        ent_length = ent_end - ent_start

        found_duplicate = False

        for (label, s, e), prev_ent in list(kept.items()):
            same_label = ent_label == label
            same_start_or_end = ent_start == s or ent_end == e

            if same_label and same_start_or_end:
                prev_length = e - s
                if ent_length > prev_length:
                    del kept[(label, s, e)]
                    kept[(ent_label, ent_start, ent_end)] = ent
                found_duplicate = True
                break

        if not found_duplicate:
            kept[(ent_label, ent_start, ent_end)] = ent

    result = list(kept.values())
    return sorted(result, key=lambda x: (int(x["start"]), int(x["end"]), str(x["label"])))


def deduplicate_entities_exact(entities: List[Dict]) -> List[Dict]:
    """
    Remove exact duplicate entities using (label, start, end, text).

    Arguments:
        entities: List of entity dictionaries.
    Return:
        Deduplicated list of entity dictionaries.
    """
    seen = set()
    out = []

    for ent in sorted(entities, key=lambda x: (int(x["start"]), int(x["end"]), str(x["label"]), str(x.get("text", "")))):
        key = (
            str(ent["label"]),
            int(ent["start"]),
            int(ent["end"]),
            clean_span_text(ent.get("text", "")),
        )
        if key not in seen:
            seen.add(key)
            out.append(ent)

    return out


def build_char_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Split a long document into overlapping character chunks.

    Arguments:
        text: Full input text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Character overlap between consecutive chunks.
    Return:
        List of (start, end) character ranges.
    """
    n = len(text)
    if n == 0:
        return [(0, 0)]

    if chunk_size <= 0:
        return [(0, n)]

    if overlap < 0:
        overlap = 0

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append((start, end))
        if end >= n:
            break
        start += step

    return chunks


def safe_tokenize_overflow(
    tokenizer,
    full_text: str,
    max_len: int,
    stride_tokens: int,
) -> Dict:
    """
    Tokenize a text chunk with overflow handling and perform basic safety checks.

    Arguments:
        tokenizer: Hugging Face tokenizer.
        full_text: Input text chunk.
        max_len: Maximum token length per window.
        stride_tokens: Token overlap between windows.
    Return:
        Tokenizer output dictionary.
    """
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
        stride=stride_tokens,
        return_overflowing_tokens=True,
        padding=False,
        return_tensors=None,
    )

    n_windows = len(enc["input_ids"])
    if n_windows > MAX_WINDOWS_PER_CHUNK:
        raise RuntimeError(
            f"Too many overflow windows for one chunk: {n_windows}. "
            f"Reduce NER_CHAR_CHUNK_SIZE or revise the input text."
        )

    return enc


def run_inference_batch(
    model,
    tokenizer,
    input_ids_slice,
    attn_mask_slice,
    device,
) -> Tuple[torch.Tensor, int]:
    """
    Run one forward pass for a batch of tokenized windows.

    Arguments:
        model: Hugging Face token-classification model.
        tokenizer: Matching tokenizer.
        input_ids_slice: Token ids for a subset of windows.
        attn_mask_slice: Attention masks for a subset of windows.
        device: Torch device.
    Return:
        Tuple containing probability tensor and the batch size actually used.
    """
    batch_padded = tokenizer.pad(
        {
            "input_ids": input_ids_slice,
            "attention_mask": attn_mask_slice,
        },
        padding=True,
        return_tensors="pt",
    )

    input_ids = batch_padded["input_ids"].to(device)
    attention_mask = batch_padded["attention_mask"].to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = softmax_lastdim(out.logits)

    return probs, input_ids.shape[0]


def run_model_overflow_note(
    model,
    tokenizer,
    full_text: str,
    max_len: int,
    stride_tokens: int,
    device,
    batch_windows: int,
    avg_scores: bool = False,
) -> List[Dict]:
    """
    Run token-classification inference over a text chunk using overlapping windows.

    Arguments:
        model: Hugging Face token-classification model.
        tokenizer: Matching tokenizer.
        full_text: Input text chunk.
        max_len: Maximum token length per window.
        stride_tokens: Overlap between windows in tokens.
        device: Torch device.
        batch_windows: Number of windows processed per forward pass.
        avg_scores: Passed to span aggregation.
    Return:
        List of entity dictionaries filtered by score and deduplicated.
    """
    if not full_text:
        return []

    id2label = model.config.id2label
    enc = safe_tokenize_overflow(
        tokenizer=tokenizer,
        full_text=full_text,
        max_len=max_len,
        stride_tokens=stride_tokens,
    )

    input_ids_all = enc["input_ids"]
    attn_mask_all = enc["attention_mask"]
    offsets_all = enc["offset_mapping"]

    all_ents = []
    n = len(input_ids_all)
    i = 0

    while i < n:
        current_batch = min(batch_windows, n - i)
        success = False
        last_error = None

        while current_batch >= 1 and not success:
            try:
                j = i + current_batch
                probs, used_batch = run_inference_batch(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_slice=input_ids_all[i:j],
                    attn_mask_slice=attn_mask_all[i:j],
                    device=device,
                )

                for b in range(used_batch):
                    probs_b = probs[b]
                    label_ids = torch.argmax(probs_b, dim=-1).tolist()
                    label_scores = torch.max(probs_b, dim=-1).values.tolist()
                    offsets = offsets_all[i + b]

                    ents = aggregate_bio_spans(
                        full_text=full_text,
                        offsets=offsets,
                        label_ids=label_ids[: len(offsets)],
                        label_scores=label_scores[: len(offsets)],
                        id2label=id2label,
                        avg_scores=avg_scores,
                    )

                    for e in ents:
                        sc = float(e["score"])
                        if MIN_SCORE <= sc <= MAX_SCORE:
                            all_ents.append(e)

                i = j
                success = True

            except RuntimeError as e:
                last_error = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_batch = current_batch // 2

        if not success:
            raise RuntimeError(f"Inference failed for chunk window block starting at {i}: {last_error}")

    all_ents = deduplicate_entities_by_span_label(all_ents)
    return all_ents


def run_model_chunked_document(
    model,
    tokenizer,
    full_text: str,
    max_len: int,
    stride_tokens: int,
    device,
    batch_windows: int,
    avg_scores: bool = False,
    verbose_prefix: str = "",
) -> List[Dict]:
    """
    Run NER over a complete document by splitting it into overlapping character chunks.

    Arguments:
        model: Hugging Face token-classification model.
        tokenizer: Matching tokenizer.
        full_text: Full input document.
        max_len: Maximum token length per overflow window.
        stride_tokens: Overlap between token windows.
        device: Torch device.
        batch_windows: Number of windows processed per forward pass.
        avg_scores: Score aggregation strategy for BIO span aggregation.
        verbose_prefix: Prefix used for progress printing.
    Return:
        List of entity dictionaries with document-level absolute offsets.
    """
    text = normalize_input_text(full_text)
    if not text:
        return []

    char_chunks = build_char_chunks(
        text=text,
        chunk_size=NER_CHAR_CHUNK_SIZE,
        overlap=NER_CHAR_CHUNK_OVERLAP,
    )

    all_entities = []
    total_chunks = len(char_chunks)

    for chunk_idx, (chunk_start, chunk_end) in enumerate(char_chunks, start=1):
        chunk_text = text[chunk_start:chunk_end]
        t0 = time.time()
        print(
            f"{verbose_prefix}chunk {chunk_idx}/{total_chunks} "
            f"chars={chunk_start}:{chunk_end} len={len(chunk_text)}"
        )

        chunk_entities = run_model_overflow_note(
            model=model,
            tokenizer=tokenizer,
            full_text=chunk_text,
            max_len=max_len,
            stride_tokens=stride_tokens,
            device=device,
            batch_windows=batch_windows,
            avg_scores=avg_scores,
        )

        for ent in chunk_entities:
            all_entities.append(
                {
                    "start": int(ent["start"]) + chunk_start,
                    "end": int(ent["end"]) + chunk_start,
                    "label": ent["label"],
                    "score": float(ent["score"]),
                    "text": clean_span_text(text[int(ent["start"]) + chunk_start:int(ent["end"]) + chunk_start]),
                }
            )

        elapsed = time.time() - t0

    all_entities = deduplicate_entities_by_span_label(all_entities)
    all_entities = deduplicate_entities_exact(all_entities)
    return all_entities


def load_models(model_names: List[str], device, use_fp16: bool, max_len_cap: int = 0) -> Dict[str, Dict]:
    """
    Load token-classification models and tokenizers.

    Arguments:
        model_names: List of Hugging Face model identifiers.
        device: Torch device.
        use_fp16: If True, cast models to float16 after loading.
        max_len_cap: Optional cap for the effective maximum length.
    Return:
        Mapping from model name to tokenizer, model, and resolved max_len.
    """
    pack = {}

    for name in model_names:
        print(f"  Loading {name}")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        mod = AutoModelForTokenClassification.from_pretrained(name)
        mod.eval()
        mod.to(device)

        if use_fp16:
            mod = mod.to(dtype=torch.float16)

        max_len = get_effective_max_length(tok, mod, cap=max_len_cap)
        print(f"    max_len={max_len}")

        pack[name] = {
            "tokenizer": tok,
            "model": mod,
            "max_len": max_len,
        }

    return pack


def process_patients(
    df: pd.DataFrame,
    models_pack: Dict[str, Dict],
    json_dir: str,
    out_csv: str,
    batch_windows: int,
    avg_scores: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Run NER over all patient rows and write JSON plus CSV outputs.

    Arguments:
        df: Input DataFrame. Must contain patient_id and text.
        models_pack: Loaded models from load_models().
        json_dir: Output directory for per-patient JSON files.
        out_csv: Output CSV path.
        batch_windows: Number of windows per inference batch.
        avg_scores: Score aggregation strategy for span construction.
    Return:
        Dictionary mapping file_path to entity lists for later .ann generation.
    """
    if "patient_id" not in df.columns or "text" not in df.columns:
        raise ValueError("DataFrame must have columns: patient_id, text")

    all_rows = []
    model_names = list(models_pack.keys())
    device = next(iter(models_pack.values()))["model"].device
    ann_bucket: Dict[str, List[Dict]] = defaultdict(list)

    total_rows = len(df)

    for idx, r in df.iterrows():
        pid = int(r["patient_id"])
        age = r.get("age", None)
        file_name = r.get("file_name", None)
        file_path = r.get("file_path", None)
        raw_text = str(r["text"]) if r["text"] is not None else ""
        text = normalize_input_text(raw_text)

        print("")
        print(f"[PATIENT {idx + 1}/{total_rows}] patient_id={pid} file={file_name} chars={len(text)}")

        entities_all = []
        errors = []

        for model_idx, (model_name, pack) in enumerate(models_pack.items(), start=1):
            print(f"  [MODEL {model_idx}/{len(models_pack)}] {model_name}")

            try:
                ents = run_model_chunked_document(
                    model=pack["model"],
                    tokenizer=pack["tokenizer"],
                    full_text=text,
                    max_len=pack["max_len"],
                    stride_tokens=WINDOW_STRIDE_TOKENS,
                    device=device,
                    batch_windows=batch_windows,
                    avg_scores=avg_scores,
                    verbose_prefix="    ",
                )

                for e in ents:
                    entities_all.append(
                        {
                            "text": e["text"],
                            "label": e["label"],
                            "score": float(e["score"]),
                            "start": int(e["start"]),
                            "end": int(e["end"]),
                            "model": model_name,
                        }
                    )

                print(f"  [MODEL DONE] {model_name} entities={len(ents)}")

            except Exception as e:
                err = f"{model_name}: {type(e).__name__}: {str(e)}"
                errors.append(err)
                print(f"  [MODEL ERROR] {err}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        entities_all = sorted(
            deduplicate_entities_exact(entities_all),
            key=lambda x: (x["start"], x["end"], x["model"], x["label"]),
        )

        if file_path:
            ann_bucket[file_path].extend(entities_all)

        out_obj = {
            "patient_id": pid,
            "age": age,
            "file_name": file_name,
            "models": model_names,
            "min_score": MIN_SCORE,
            "max_score": MAX_SCORE,
            "batch_windows": batch_windows,
            "window_stride_tokens": WINDOW_STRIDE_TOKENS,
            "ner_char_chunk_size": NER_CHAR_CHUNK_SIZE,
            "ner_char_chunk_overlap": NER_CHAR_CHUNK_OVERLAP,
            "num_entities": len(entities_all),
            "errors": errors,
            "entities": entities_all,
        }

        out_path = os.path.join(json_dir, f"patient_{pid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        for ent in entities_all:
            all_rows.append(
                {
                    "patient_id": pid,
                    "age": age,
                    "file_name": file_name,
                    "model": ent["model"],
                    "label": ent["label"],
                    "entity_text": ent["text"],
                    "score": ent["score"],
                    "start": ent["start"],
                    "end": ent["end"],
                }
            )

        print(f"[PATIENT DONE] patient_id={pid} file={file_name} entities={len(entities_all)} errors={len(errors)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  CSV: {out_csv}")
    print(f"  JSON dir: {json_dir}")

    return dict(ann_bucket)


def _read_text(path: str) -> str:
    """
    Read a text file using UTF-8 with Latin-1 fallback.

    Arguments:
        path: File path.
    Return:
        File content as string.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            return f.read()


def _roman_to_int(s: str) -> Optional[int]:
    """
    Convert a Roman numeral between I and X into an integer.

    Arguments:
        s: Roman numeral string.
    Return:
        Integer value if valid, otherwise None.
    """
    if not s:
        return None

    t = str(s).strip().upper()
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    prev = 0

    for ch in reversed(t):
        v = values.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
            prev = v

    if not (1 <= total <= 10):
        return None

    canonical = {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
        8: "VIII",
        9: "IX",
        10: "X",
    }[total]

    return total if t == canonical else None


def _parse_total(raw: Optional[str]) -> Optional[int]:
    """
    Parse a Gleason total written in Arabic or Roman notation.

    Arguments:
        raw: Raw captured total string.
    Return:
        Integer total if valid, otherwise None.
    """
    if not raw:
        return None

    s = str(raw).strip()
    if s.isdigit():
        v = int(s)
        return v if 2 <= v <= 10 else None

    return _roman_to_int(s)


def _make_snippet(text: str, start: int, end: int, window: int) -> str:
    """
    Build a local context snippet around a matched span.

    Arguments:
        text: Full document text.
        start: Start offset.
        end: End offset.
        window: Number of surrounding characters to include on each side.
    Return:
        Cleaned snippet string.
    """
    a = max(0, start - window)
    b = min(len(text), end + window)
    snip = text[a:b].replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", snip).strip()


def _add_gleason_match(out: List[Dict], t: str, m: re.Match, pattern_type: str) -> None:
    """
    Normalize a Gleason regex match and append it to the accumulator.

    Arguments:
        out: Output accumulator list.
        t: Full text.
        m: Regex match object.
        pattern_type: Name of the regex pattern used.
    Return:
        None.
    """
    start, end = m.start(), m.end()
    gd = m.groupdict()

    p1 = int(gd["p1"]) if gd.get("p1") else None
    p2 = int(gd["p2"]) if gd.get("p2") else None
    tot = _parse_total(gd.get("tot"))
    inferred = tot if tot is not None else ((p1 + p2) if (p1 and p2) else None)

    out.append(
        {
            "label": "GLEASON",
            "gleason_raw": t[start:end],
            "pattern_type": pattern_type,
            "primary": p1,
            "secondary": p2,
            "total": inferred,
            "start": start,
            "end": end,
            "snippet": _make_snippet(t, start, end, SNIPPET_WINDOW),
            "text": t[start:end],
        }
    )


def extract_gleason_mentions(text: str) -> List[Dict]:
    """
    Extract Gleason mentions from a clinical text.

    Arguments:
        text: Clinical narrative.
    Return:
        List of Gleason mention dictionaries.
    """
    t = normalize_input_text(str(text)[:MAX_CHARS_PER_DOC])
    out: List[Dict] = []
    used: List[Tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < y and b > x for x, y in used)

    for name, rx in [
        ("total_then_sum", TOTAL_THEN_SUM_RE),
        ("sum_in_parens", SUM_IN_PARENS_RE),
        ("sum_total", SUM_TOTAL_RE),
        ("score_only", SCORE_ONLY_RE),
    ]:
        for m in rx.finditer(t):
            if overlaps(m.start(), m.end()):
                continue
            _add_gleason_match(out, t, m, name)
            used.append((m.start(), m.end()))

    out.sort(key=lambda d: (d["start"], d["end"]))
    return out


def run_gleason_folder(input_dir: str, out_csv: str) -> Dict[str, List[Dict]]:
    """
    Run Gleason extraction over all .txt files in a folder.

    Arguments:
        input_dir: Input folder containing .txt files.
        out_csv: Output CSV path.
    Return:
        Dictionary mapping file_path to Gleason entities for .ann merging.
    """
    txt_files = sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(input_dir)
        for fn in files
        if fn.lower().endswith(".txt")
    )

    rows: List[Dict] = []
    total = len(txt_files)
    ann_bucket: Dict[str, List[Dict]] = defaultdict(list)

    for idx, path in enumerate(txt_files, start=1):
        text = _read_text(path)
        matches = extract_gleason_mentions(text)
        print(f"  [{idx}/{total}] {os.path.basename(path)} | gleason_mentions={len(matches)}")

        for j, rec in enumerate(matches, start=1):
            rows.append(
                {
                    "file_name": os.path.basename(path),
                    "file_path": path,
                    "match_index": j,
                    **rec,
                }
            )
            ann_bucket[path].append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  CSV: {out_csv}")
    return dict(ann_bucket)


def _to_float(num_str: str) -> Optional[float]:
    """
    Parse a numeric string with Spanish or mixed thousand/decimal separators.

    Arguments:
        num_str: Raw numeric string.
    Return:
        Parsed float, or None if the input is empty.
    """
    if not num_str:
        return None

    s = str(num_str).strip().replace(" ", "")
    if not s:
        return None

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", s):
        return float(s.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", s):
        return float(s.replace(",", ""))
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            return float(s.replace(".", "").replace(",", "."))
        return float(s.replace(",", ""))
    if "," in s:
        return float(s.replace(",", "."))
    return float(s)


def _norm_unit(unit: str) -> str:
    """
    Normalize PSA unit strings to canonical forms.

    Arguments:
        unit: Raw unit string.
    Return:
        Canonical unit string.
    """
    u = re.sub(r"\s+", "", unit or "").lower()
    if "ng/ml" in u or "ngml" in u:
        return "ng/mL"
    if "ng/l" in u or re.fullmatch(r"ngl", u):
        return "ng/L"
    return unit.strip() if unit else ""


def _convert_to_ng_ml(value: float, unit_norm: str) -> Optional[float]:
    """
    Convert a PSA numeric value to ng/mL when possible.

    Arguments:
        value: Numeric value.
        unit_norm: Normalized canonical unit.
    Return:
        Converted value in ng/mL, or None if the unit is unsupported.
    """
    if value is None:
        return None
    if unit_norm == "ng/mL":
        return float(value)
    if unit_norm == "ng/L":
        return float(value) / 1000.0
    return None


def extract_psa_values(text: str) -> List[Dict]:
    """
    Extract PSA-compatible numeric values followed by supported units.

    Arguments:
        text: Clinical narrative.
    Return:
        List of PSA mention dictionaries.
    """
    t = normalize_input_text(str(text)[:MAX_CHARS_PER_DOC])
    out: List[Dict] = []

    for m in _VALUE_WITH_UNIT_RE.finditer(t):
        value_raw = m.group("value")
        unit_raw = m.group("unit")
        unit_norm = _norm_unit(unit_raw)
        value_num = _to_float(value_raw)
        value_ng_ml = _convert_to_ng_ml(value_num, unit_norm) if value_num is not None else None
        start, end = m.start(), m.end()

        out.append(
            {
                "label": "PSA",
                "value_raw": value_raw,
                "value_num": value_num,
                "unit_raw": unit_raw,
                "unit_norm": unit_norm,
                "value_ng_ml": value_ng_ml,
                "start": start,
                "end": end,
                "snippet": _make_snippet(t, start, end, SNIPPET_WINDOW),
                "text": t[start:end],
            }
        )

    out.sort(key=lambda d: (d["start"], d["end"]))
    return out


def run_psa_folder(input_dir: str, out_csv: str) -> Dict[str, List[Dict]]:
    """
    Run PSA extraction over all .txt files in a folder.

    Arguments:
        input_dir: Input folder containing .txt files.
        out_csv: Output CSV path.
    Return:
        Dictionary mapping file_path to PSA entities for .ann merging.
    """
    txt_files = sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(input_dir)
        for fn in files
        if fn.lower().endswith(".txt")
    )

    rows: List[Dict] = []
    total = len(txt_files)
    ann_bucket: Dict[str, List[Dict]] = defaultdict(list)

    for idx, path in enumerate(txt_files, start=1):
        text = _read_text(path)
        matches = extract_psa_values(text)
        print(f"  [{idx}/{total}] {os.path.basename(path)} | psa_values={len(matches)}")

        for j, rec in enumerate(matches, start=1):
            rows.append(
                {
                    "file_name": os.path.basename(path),
                    "file_path": path,
                    "match_index": j,
                    **rec,
                }
            )
            ann_bucket[path].append(rec)

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  CSV: {out_csv}")
    return dict(ann_bucket)


def merge_and_write_ann_files(
    buckets: List[Dict[str, List[Dict]]],
    all_txt_paths: List[str],
) -> None:
    """
    Merge entities from all workflows by source file and write .ann files.

    Arguments:
        buckets: One dictionary per workflow, keyed by file_path.
        all_txt_paths: All source .txt paths seen across workflows.
    Return:
        None.
    """
    merged: Dict[str, List[Dict]] = defaultdict(list)

    for bucket in buckets:
        for path, ents in bucket.items():
            merged[path].extend(ents)

    for path in all_txt_paths:
        ents = merged.get(path, [])
        ents = deduplicate_entities_exact(ents)
        write_ann(path, ents)
        print(f"  ann: {os.path.splitext(os.path.basename(path))[0]}.ann  entities={len(ents)}")


def main() -> None:
    """
    Run the full entity extraction pipeline from command-line arguments.
    """
    global MIN_SCORE

    parser = argparse.ArgumentParser(description="Run the silver-standard entity extraction pipeline.")
    parser.add_argument("--untagged-dir", required=True, help="Directory containing raw .txt notes.")
    parser.add_argument(
        "--output-dir",
        help="Directory where SilverMentions outputs will be written. Defaults to <untagged-dir>/SilverMentions.",
    )
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="Minimum confidence score for NER mentions.")
    args = parser.parse_args()

    MIN_SCORE = args.min_score
    configure_paths(args.untagged_dir, args.output_dir)
    ensure_configured_directories()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    print(f"Device: {device}")

    all_txt_paths: List[str] = []

    print("\n=== WF1: BSC models")
    df_wf1 = read_txt_folder(WF1_INPUT_DIR)
    all_txt_paths.extend(df_wf1["file_path"].tolist())
    models_wf1 = load_models(WF1_MODELS, device, use_fp16, max_len_cap=0)
    ann_wf1 = process_patients(
        df=df_wf1,
        models_pack=models_wf1,
        json_dir=WF1_JSON_DIR,
        out_csv=WF1_OUT_CSV,
        batch_windows=BATCH_WINDOWS_WF1,
        avg_scores=True,
    )
    del models_wf1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== WF2: MedSpaner models")
    df_wf2 = read_txt_folder(WF2_INPUT_DIR)
    all_txt_paths.extend(df_wf2["file_path"].tolist())
    models_wf2 = load_models(WF2_MODELS, device, use_fp16, max_len_cap=512)
    ann_wf2 = process_patients(
        df=df_wf2,
        models_pack=models_wf2,
        json_dir=WF2_JSON_DIR,
        out_csv=WF2_OUT_CSV,
        batch_windows=BATCH_WINDOWS_WF2,
        avg_scores=False,
    )
    del models_wf2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Detectando Gleason")
    ann_wf3 = run_gleason_folder(WF3_INPUT_DIR, WF3_OUT_CSV)
    all_txt_paths.extend(list(ann_wf3.keys()))

    print("\n=== Detectando PSA")
    ann_wf4 = run_psa_folder(WF4_INPUT_DIR, WF4_OUT_CSV)
    all_txt_paths.extend(list(ann_wf4.keys()))

    print("\n=== Cleaning previous .ann files")
    clear_ann_dir(ANN_DIR)

    print("\n=== Writing .ann files")
    merge_and_write_ann_files(
        buckets=[ann_wf1, ann_wf2, ann_wf3, ann_wf4],
        all_txt_paths=list(dict.fromkeys(all_txt_paths)),
    )

    print(f"\nDone. All outputs under: {_ETIQUETAS}")


if __name__ == "__main__":
    main()
