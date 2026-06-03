"""
brat_to_inline_tags_event_based.py

Converts BRAT standoff annotations (.ann) + raw text (.txt) into an inline-tagged plain text.

Behavior:
- Tags are inserted strictly by offsets:
  - At each boundary position pos:
    1) write all closing tags for spans ending at pos
    2) write all opening tags for spans starting at pos
    3) write the next character (if any)
- No stack / no re-open logic. This intentionally allows non-XML crossing patterns like:
  <NORMALIZABLES><Negated>estrógeno</NORMALIZABLES></Negated>

This matches the "open-start / close-end" requirement even when multiple labels share the same span.

Outputs: OUT_DIR/<basename>.tagged.txt for each (.txt,.ann) pair.
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


INPUT_DIR = ""
OUT_DIR = ""

LABEL_PRIORITY: Dict[str, int] = {
    "NORMALIZABLES": 0,
    "Negated": 1,
    "Speculated": 2,
    "Spec_cue": 3,
    "Neg_cue": 4,
}

DEFAULT_PRIORITY = 100


@dataclass(frozen=True)
class EntitySpan:
    label: str
    start: int
    end: int


def prio(label: str) -> int:
    """
    Priority for tie-breaking among labels (lower first).

    Arguments: label (str): label name.
    Return: int: priority value.
    """
    return LABEL_PRIORITY.get(label, DEFAULT_PRIORITY)


def escape_text(s: str) -> str:
    """
    Escapes raw text so it does not interfere with tag syntax.

    Arguments: s (str): raw text.
    Return: str: escaped text.
    """
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def sanitize_tag(label: str) -> str:
    """
    Normalizes a BRAT label into a safe tag name.

    Arguments: label (str): entity label.
    Return: str: safe tag name.
    """
    tag = label.strip().replace(" ", "_")
    tag = re.sub(r"[^A-Za-z0-9_:-]", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    if not tag:
        tag = "ENTIDAD"
    if re.match(r"^[0-9]", tag):
        tag = f"X_{tag}"
    return tag


def open_tag(label: str) -> str:
    """
    Builds an opening tag.

    Arguments: label (str): entity label.
    Return: str: opening tag.
    """
    t = sanitize_tag(label)
    return f"<{t}>"


def close_tag(label: str) -> str:
    """
    Builds a closing tag.

    Arguments: label (str): entity label.
    Return: str: closing tag.
    """
    t = sanitize_tag(label)
    return f"</{t}>"


def parse_brat_ann(ann_path: str) -> List[EntitySpan]:
    """
    Parses BRAT .ann file for text-bound annotations (T-lines).

    Arguments: ann_path (str): path to .ann.
    Return: List[EntitySpan]: spans (end-exclusive).
    """
    spans: List[EntitySpan] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or not line.startswith("T"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            meta = parts[1].strip()
            meta_parts = meta.split()
            if len(meta_parts) < 3:
                continue

            label = meta_parts[0]
            offsets_str = " ".join(meta_parts[1:]).strip()

            segs = [seg.strip() for seg in offsets_str.split(";")] if ";" in offsets_str else [offsets_str]
            for seg in segs:
                if not seg:
                    continue
                m = re.match(r"^(\d+)\s+(\d+)$", seg)
                if not m:
                    continue
                start, end = int(m.group(1)), int(m.group(2))
                if end <= start:
                    continue
                spans.append(EntitySpan(label=label, start=start, end=end))

    return spans


def build_event_maps(spans: List[EntitySpan]) -> Tuple[Dict[int, List[EntitySpan]], Dict[int, List[EntitySpan]]]:
    """
    Builds open/close event maps.

    Arguments: spans (List[EntitySpan]): spans.
    Return: (opens_by_pos, closes_by_pos)
    """
    opens_by_pos: Dict[int, List[EntitySpan]] = {}
    closes_by_pos: Dict[int, List[EntitySpan]] = {}

    for sp in spans:
        opens_by_pos.setdefault(sp.start, []).append(sp)
        closes_by_pos.setdefault(sp.end, []).append(sp)

    for pos, lst in opens_by_pos.items():
        opens_by_pos[pos] = sorted(
            lst,
            key=lambda x: (-x.end, prio(x.label), x.label),
        )

    for pos, lst in closes_by_pos.items():
        closes_by_pos[pos] = sorted(
            lst,
            key=lambda x: (-x.start, prio(x.label), x.label),
        )

    return opens_by_pos, closes_by_pos


def tag_text_event_based(text: str, spans: List[EntitySpan]) -> str:
    """
    Inserts tags by start/end events without enforcing stack nesting.

    Arguments:
        text (str): raw text.
        spans (List[EntitySpan]): entity spans.
    Return:
        str: tagged text.
    """
    if not spans:
        return escape_text(text)

    opens_by_pos, closes_by_pos = build_event_maps(spans)

    out: List[str] = []
    n = len(text)

    for pos in range(0, n + 1):
        if pos in closes_by_pos:
            for sp in closes_by_pos[pos]:
                out.append(close_tag(sp.label))

        if pos in opens_by_pos:
            for sp in opens_by_pos[pos]:
                out.append(open_tag(sp.label))

        if pos < n:
            out.append(escape_text(text[pos]))

    return "".join(out)


def find_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Finds (.txt, .ann) pairs by basename in a directory.

    Arguments: input_dir (str): directory containing files.
    Return: List[Tuple[str, str]]: list of (txt_path, ann_path).
    """
    files = os.listdir(input_dir)
    txt_map: Dict[str, str] = {}
    ann_map: Dict[str, str] = {}

    for fn in files:
        p = os.path.join(input_dir, fn)
        low = fn.lower()
        if low.endswith(".txt"):
            txt_map[os.path.splitext(fn)[0]] = p
        elif low.endswith(".ann"):
            ann_map[os.path.splitext(fn)[0]] = p

    keys = sorted(set(txt_map.keys()) & set(ann_map.keys()))
    return [(txt_map[k], ann_map[k]) for k in keys]


def main() -> None:
    """
    Runs conversion for every (.txt,.ann) pair in INPUT_DIR and writes outputs to OUT_DIR.

    Arguments: None
    Return: None
    """
    global INPUT_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Convert BRAT .txt/.ann pairs into inline-tagged text.")
    parser.add_argument("--input-dir", required=True, help="Directory containing paired .txt and .ann files.")
    parser.add_argument("--output-dir", required=True, help="Directory where inline-tagged files will be written.")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUT_DIR = args.output_dir

    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = find_pairs(INPUT_DIR)
    print(f"Found {len(pairs)} pairs in {INPUT_DIR}")

    for txt_path, ann_path in pairs:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(OUT_DIR, f"{base}t.txt")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        spans = parse_brat_ann(ann_path)
        tagged = tag_text_event_based(text, spans)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tagged)

        print(f"Wrote {out_path} (spans={len(spans)})")


if __name__ == "__main__":
    main()
