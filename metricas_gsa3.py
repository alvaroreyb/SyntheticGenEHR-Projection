"""
Compare Brat .ann files from a silver-standard directory against predicted .ann files.

This script performs entity-level NER evaluation using a relaxed match criterion:
- same label
- and same start offset OR same end offset

If a predicted entity matches a silver entity under this rule, it is counted as
a true positive. Unmatched predicted entities are counted as false positives.
Unmatched silver entities are counted as false negatives.

The script reads paired .ann files from two directories, compares their entities,
computes global and per-file metrics, and exports detailed CSV reports.

Generated outputs:
1. detailed_entity_comparisons.csv
   One row per comparison outcome (TP, FP, FN).

2. file_level_metrics.csv
   One row per file with entity-level metrics.

3. label_level_metrics.csv
   One row per label aggregated across all files.

4. global_metrics.csv
   Single-row global summary.

Accuracy note:
At entity level, true negatives are usually undefined in NER. This script computes
accuracy as:
    TP / (TP + FP + FN)
which is an entity-level accuracy-like score based on the evaluated prediction space.
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd

SILVER_ANN_DIR = ""
PRED_ANN_DIR = ""
OUT_DIR = ""

NORMALIZE_TEXT = True
CASE_SENSITIVE_TEXT = False


@dataclass(frozen=True)
class AnnEntity:
    file_name: str
    entity_id: str
    label: str
    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


def ensure_output_dir(path: str) -> None:
    """
    Create the output directory if it does not exist.

    Arguments:
        path: Output directory path.
    Return:
        None.
    """
    os.makedirs(path, exist_ok=True)


def normalize_span_text(text: str) -> str:
    """
    Normalize span text to make comparisons more stable across annotation sources.

    Arguments:
        text: Raw span text from the annotation file.
    Return:
        Normalized span text.
    """
    value = str(text) if text is not None else ""
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()

    if not CASE_SENSITIVE_TEXT:
        value = value.lower()

    return value


def parse_ann_line(file_name: str, line: str) -> Optional[AnnEntity]:
    """
    Parse one Brat text-bound annotation line into an AnnEntity object.

    Supported format:
        T1<TAB>LABEL START END<TAB>TEXT

    Discontinuous spans such as:
        T1<TAB>LABEL 0 4;10 15<TAB>text
    are skipped.

    Arguments:
        file_name: Base file name for traceability.
        line: Raw line from the .ann file.
    Return:
        Parsed AnnEntity if the line is valid and supported, otherwise None.
    """
    line = line.strip()
    if not line or not line.startswith("T"):
        return None

    parts = line.split("\t")
    if len(parts) < 3:
        return None

    entity_id = parts[0].strip()
    ann_info = parts[1].strip()
    text = "\t".join(parts[2:]).strip()

    if ";" in ann_info:
        return None

    info_parts = ann_info.split()
    if len(info_parts) < 3:
        return None

    label = info_parts[0]

    try:
        start = int(info_parts[1])
        end = int(info_parts[2])
    except ValueError:
        return None

    if end <= start:
        return None

    if NORMALIZE_TEXT:
        text = normalize_span_text(text)

    return AnnEntity(
        file_name=file_name,
        entity_id=entity_id,
        label=label,
        start=start,
        end=end,
        text=text,
    )


def read_ann_file(path: str) -> List[AnnEntity]:
    """
    Read a Brat .ann file and extract all supported text-bound entities.

    Arguments:
        path: Path to the .ann file.
    Return:
        List of parsed AnnEntity objects.
    """
    entities: List[AnnEntity] = []

    if not path or not os.path.isfile(path):
        return entities

    file_name = os.path.basename(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            entity = parse_ann_line(file_name=file_name, line=line)
            if entity is not None:
                entities.append(entity)

    entities.sort(key=lambda x: (x.start, x.end, x.label, x.text, x.entity_id))
    return entities


def list_ann_files(directory: str) -> Dict[str, str]:
    """
    Collect all .ann files in a directory and index them by file name.

    Arguments:
        directory: Directory containing .ann files.
    Return:
        Dictionary mapping file name to absolute path.
    """
    files = {}

    if not os.path.isdir(directory):
        return files

    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".ann"):
            files[name] = os.path.join(directory, name)

    return files


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Perform protected division returning zero when the denominator is zero.

    Arguments:
        numerator: Numerator value.
        denominator: Denominator value.
    Return:
        Division result or zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute entity-level evaluation metrics from TP, FP, and FN counts.

    Accuracy is computed as:
        TP / (TP + FP + FN)

    Arguments:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
    Return:
        Dictionary with precision, recall, f1_score, and accuracy.
    """
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp, tp + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def entities_match_relaxed(silver_entity: AnnEntity, pred_entity: AnnEntity) -> Tuple[bool, str]:
    """
    Decide whether a silver entity and a predicted entity match under the relaxed rule.

    A match is considered true if:
    - both entities share the same label
    - and they share the same start offset OR the same end offset

    The returned match_type indicates which boundary matched.

    Arguments:
        silver_entity: Reference entity.
        pred_entity: Predicted entity.
    Return:
        Tuple containing:
        - Boolean indicating whether they match.
        - String describing the match type.
    """
    if silver_entity.label != pred_entity.label:
        return False, "label_mismatch"

    same_start = silver_entity.start == pred_entity.start
    same_end = silver_entity.end == pred_entity.end

    if same_start and same_end:
        return True, "same_start_and_end"
    if same_start:
        return True, "same_start"
    if same_end:
        return True, "same_end"

    return False, "boundary_mismatch"


def match_entities_greedily(
    silver_entities: List[AnnEntity],
    pred_entities: List[AnnEntity],
) -> Tuple[List[Tuple[AnnEntity, AnnEntity, str]], List[AnnEntity], List[AnnEntity]]:
    """
    Perform greedy one-to-one matching between silver and predicted entities.

    Each predicted entity can match at most one silver entity, and each silver
    entity can match at most one predicted entity.

    Matching priority is:
    1. Same label and same start and end.
    2. Same label and same start only.
    3. Same label and same end only.

    Arguments:
        silver_entities: Reference entities for one file.
        pred_entities: Predicted entities for one file.
    Return:
        Tuple containing:
        - List of matched pairs with match type.
        - List of unmatched silver entities.
        - List of unmatched predicted entities.
    """
    matches: List[Tuple[AnnEntity, AnnEntity, str]] = []
    used_silver = set()
    used_pred = set()

    candidate_pairs: List[Tuple[int, int, int, int, int, str]] = []

    for i, silver_entity in enumerate(silver_entities):
        for j, pred_entity in enumerate(pred_entities):
            is_match, match_type = entities_match_relaxed(silver_entity, pred_entity)
            if not is_match:
                continue

            if match_type == "same_start_and_end":
                priority = 0
            elif match_type == "same_start":
                priority = 1
            else:
                priority = 2

            span_distance = abs(silver_entity.start - pred_entity.start) + abs(silver_entity.end - pred_entity.end)
            length_distance = abs(silver_entity.length - pred_entity.length)

            candidate_pairs.append((priority, span_distance, length_distance, i, j, match_type))

    candidate_pairs.sort()

    for _, _, _, i, j, match_type in candidate_pairs:
        if i in used_silver or j in used_pred:
            continue

        used_silver.add(i)
        used_pred.add(j)
        matches.append((silver_entities[i], pred_entities[j], match_type))

    unmatched_silver = [ent for idx, ent in enumerate(silver_entities) if idx not in used_silver]
    unmatched_pred = [ent for idx, ent in enumerate(pred_entities) if idx not in used_pred]

    return matches, unmatched_silver, unmatched_pred


def compare_entities(
    file_name: str,
    silver_entities: List[AnnEntity],
    pred_entities: List[AnnEntity],
) -> Tuple[List[Dict], Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Compare silver and predicted entities for one file using relaxed matching.

    A true positive is assigned when a silver entity and a predicted entity share:
    - the same label
    - and the same start OR the same end

    The function also builds a detailed comparison table and per-label counts.

    Arguments:
        file_name: File being evaluated.
        silver_entities: Reference entities.
        pred_entities: Predicted entities.
    Return:
        Tuple containing:
        - Detailed comparison rows.
        - Aggregate counts for the file.
        - Per-label TP, FP, and FN counts.
    """
    rows: List[Dict] = []
    label_counts: Dict[str, Dict[str, int]] = {}

    matches, unmatched_silver, unmatched_pred = match_entities_greedily(
        silver_entities=silver_entities,
        pred_entities=pred_entities,
    )

    file_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "silver_entities": len(silver_entities),
        "pred_entities": len(pred_entities),
    }

    def ensure_label(label: str) -> None:
        """
        Ensure that the per-label counter exists before updating it.

        Arguments:
            label: Entity label.
        Return:
            None.
        """
        if label not in label_counts:
            label_counts[label] = {"tp": 0, "fp": 0, "fn": 0}

    for silver_entity, pred_entity, match_type in matches:
        ensure_label(silver_entity.label)

        rows.append(
            {
                "file_name": file_name,
                "status": "TP",
                "match_type": match_type,
                "label": silver_entity.label,
                "silver_entity_id": silver_entity.entity_id,
                "pred_entity_id": pred_entity.entity_id,
                "silver_start": silver_entity.start,
                "silver_end": silver_entity.end,
                "pred_start": pred_entity.start,
                "pred_end": pred_entity.end,
                "silver_text": silver_entity.text,
                "pred_text": pred_entity.text,
                "same_label": 1,
                "same_start": 1 if silver_entity.start == pred_entity.start else 0,
                "same_end": 1 if silver_entity.end == pred_entity.end else 0,
                "same_text": 1 if silver_entity.text == pred_entity.text else 0,
            }
        )

        file_counts["tp"] += 1
        label_counts[silver_entity.label]["tp"] += 1

    for silver_entity in unmatched_silver:
        ensure_label(silver_entity.label)

        rows.append(
            {
                "file_name": file_name,
                "status": "FN",
                "match_type": "unmatched_silver",
                "label": silver_entity.label,
                "silver_entity_id": silver_entity.entity_id,
                "pred_entity_id": "",
                "silver_start": silver_entity.start,
                "silver_end": silver_entity.end,
                "pred_start": "",
                "pred_end": "",
                "silver_text": silver_entity.text,
                "pred_text": "",
                "same_label": 0,
                "same_start": 0,
                "same_end": 0,
                "same_text": 0,
            }
        )

        file_counts["fn"] += 1
        label_counts[silver_entity.label]["fn"] += 1

    for pred_entity in unmatched_pred:
        ensure_label(pred_entity.label)

        rows.append(
            {
                "file_name": file_name,
                "status": "FP",
                "match_type": "unmatched_pred",
                "label": pred_entity.label,
                "silver_entity_id": "",
                "pred_entity_id": pred_entity.entity_id,
                "silver_start": "",
                "silver_end": "",
                "pred_start": pred_entity.start,
                "pred_end": pred_entity.end,
                "silver_text": "",
                "pred_text": pred_entity.text,
                "same_label": 0,
                "same_start": 0,
                "same_end": 0,
                "same_text": 0,
            }
        )

        file_counts["fp"] += 1
        label_counts[pred_entity.label]["fp"] += 1

    return rows, file_counts, label_counts


def aggregate_label_counts(
    global_label_counts: Dict[str, Dict[str, int]],
    file_label_counts: Dict[str, Dict[str, int]],
) -> None:
    """
    Merge per-file label counts into the global label count accumulator.

    Arguments:
        global_label_counts: Global accumulator for label counts.
        file_label_counts: Per-file label counts.
    Return:
        None.
    """
    for label, counts in file_label_counts.items():
        if label not in global_label_counts:
            global_label_counts[label] = {"tp": 0, "fp": 0, "fn": 0}

        global_label_counts[label]["tp"] += counts["tp"]
        global_label_counts[label]["fp"] += counts["fp"]
        global_label_counts[label]["fn"] += counts["fn"]


def evaluate_ann_directories(
    silver_dir: str,
    pred_dir: str,
    out_dir: str,
) -> None:
    """
    Evaluate all paired .ann files from the silver and prediction directories.

    Files are paired by identical file name, and only files present in both
    directories are evaluated.

    Arguments:
        silver_dir: Directory containing silver-standard .ann files.
        pred_dir: Directory containing predicted .ann files.
        out_dir: Directory where CSV outputs will be written.
    Return:
        None.
    """
    ensure_output_dir(out_dir)

    silver_files = list_ann_files(silver_dir)
    pred_files = list_ann_files(pred_dir)

    all_file_names = sorted(set(silver_files.keys()) & set(pred_files.keys()))

    detailed_rows: List[Dict] = []
    file_rows: List[Dict] = []
    global_label_counts: Dict[str, Dict[str, int]] = {}

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_silver_entities = 0
    global_pred_entities = 0

    print(f"Silver .ann files: {len(silver_files)}")
    print(f"Predicted .ann files: {len(pred_files)}")
    print(f"Common files to evaluate: {len(all_file_names)}")

    for idx, file_name in enumerate(all_file_names, start=1):
        silver_path = silver_files.get(file_name)
        pred_path = pred_files.get(file_name)

        silver_entities = read_ann_file(silver_path) if silver_path else []
        pred_entities = read_ann_file(pred_path) if pred_path else []

        comp_rows, file_counts, file_label_counts = compare_entities(
            file_name=file_name,
            silver_entities=silver_entities,
            pred_entities=pred_entities,
        )

        metrics = compute_metrics(
            tp=file_counts["tp"],
            fp=file_counts["fp"],
            fn=file_counts["fn"],
        )

        detailed_rows.extend(comp_rows)
        aggregate_label_counts(global_label_counts, file_label_counts)

        file_rows.append(
            {
                "file_name": file_name,
                "silver_file_present": 1 if silver_path is not None else 0,
                "pred_file_present": 1 if pred_path is not None else 0,
                "silver_entities": file_counts["silver_entities"],
                "pred_entities": file_counts["pred_entities"],
                "tp": file_counts["tp"],
                "fp": file_counts["fp"],
                "fn": file_counts["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            }
        )

        global_tp += file_counts["tp"]
        global_fp += file_counts["fp"]
        global_fn += file_counts["fn"]
        global_silver_entities += file_counts["silver_entities"]
        global_pred_entities += file_counts["pred_entities"]

        print(
            f"[{idx}/{len(all_file_names)}] {file_name} | "
            f"silver={file_counts['silver_entities']} pred={file_counts['pred_entities']} "
            f"tp={file_counts['tp']} fp={file_counts['fp']} fn={file_counts['fn']}"
        )

    label_rows: List[Dict] = []
    for label in sorted(global_label_counts.keys()):
        tp = global_label_counts[label]["tp"]
        fp = global_label_counts[label]["fp"]
        fn = global_label_counts[label]["fn"]

        metrics = compute_metrics(tp=tp, fp=fp, fn=fn)

        label_rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            }
        )

    global_metrics = compute_metrics(tp=global_tp, fp=global_fp, fn=global_fn)

    global_row = {
        "silver_dir": silver_dir,
        "pred_dir": pred_dir,
        "evaluated_files": len(all_file_names),
        "silver_entities": global_silver_entities,
        "pred_entities": global_pred_entities,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "precision": global_metrics["precision"],
        "recall": global_metrics["recall"],
        "f1_score": global_metrics["f1_score"],
        "accuracy": global_metrics["accuracy"],
        "tp_rule": "same_label_and_same_start_or_end",
    }

    detailed_df = pd.DataFrame(detailed_rows)
    file_df = pd.DataFrame(file_rows)
    label_df = pd.DataFrame(label_rows)
    global_df = pd.DataFrame([global_row])

    detailed_csv = os.path.join(out_dir, "detailed_entity_comparisons.csv")
    file_csv = os.path.join(out_dir, "file_level_metrics.csv")
    label_csv = os.path.join(out_dir, "label_level_metrics.csv")
    global_csv = os.path.join(out_dir, "global_metrics.csv")

    detailed_df.to_csv(detailed_csv, index=False, encoding="utf-8")
    file_df.to_csv(file_csv, index=False, encoding="utf-8")
    label_df.to_csv(label_csv, index=False, encoding="utf-8")
    global_df.to_csv(global_csv, index=False, encoding="utf-8")

    print("")
    print("Global results")
    print(f"TP: {global_tp}")
    print(f"FP: {global_fp}")
    print(f"FN: {global_fn}")
    print(f"Precision: {global_metrics['precision']:.6f}")
    print(f"Recall: {global_metrics['recall']:.6f}")
    print(f"F1-score: {global_metrics['f1_score']:.6f}")
    print(f"Accuracy: {global_metrics['accuracy']:.6f}")
    print("")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"File-level CSV: {file_csv}")
    print(f"Label-level CSV: {label_csv}")
    print(f"Global CSV: {global_csv}")


def main() -> None:
    """
    Evaluate BRAT entity annotations from command-line arguments.
    """
    global SILVER_ANN_DIR, PRED_ANN_DIR, OUT_DIR, NORMALIZE_TEXT, CASE_SENSITIVE_TEXT

    parser = argparse.ArgumentParser(description="Evaluate predicted BRAT .ann files against a silver directory.")
    parser.add_argument("--silver-ann-dir", required=True, help="Directory containing silver .ann files.")
    parser.add_argument("--pred-ann-dir", required=True, help="Directory containing predicted .ann files.")
    parser.add_argument("--output-dir", required=True, help="Directory where metric CSV files will be written.")
    parser.add_argument(
        "--no-normalize-text",
        action="store_true",
        help="Disable text normalisation before comparing matched spans.",
    )
    parser.add_argument(
        "--case-sensitive-text",
        action="store_true",
        help="Use case-sensitive text comparison when text normalisation is enabled.",
    )
    args = parser.parse_args()

    SILVER_ANN_DIR = args.silver_ann_dir
    PRED_ANN_DIR = args.pred_ann_dir
    OUT_DIR = args.output_dir
    NORMALIZE_TEXT = not args.no_normalize_text
    CASE_SENSITIVE_TEXT = args.case_sensitive_text

    evaluate_ann_directories(
        silver_dir=SILVER_ANN_DIR,
        pred_dir=PRED_ANN_DIR,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
