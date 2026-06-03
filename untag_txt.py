"""
This script converts inline-tagged plain text back into BRAT standoff annotations (.ann)
and raw text (.txt).

It is the inverse of an event-based inline tag conversion where tags were inserted
strictly by offsets and crossing tag patterns were allowed. The parser scans the tagged
text from left to right, reconstructs the plain text, and rebuilds BRAT entity spans
using tag open and close events at the current plain-text offset.

Behavior:
- Opening tag <LABEL>: starts a new entity at the current plain-text offset.
- Closing tag </LABEL>: closes the most recently opened entity with that label.
- Escaped text entities (&amp;, &lt;, &gt;) are decoded back into raw text.
- Invalid or malformed tags are treated as plain text so the parser always advances.

Outputs:
- OUT_DIR/<basename>.txt
- OUT_DIR/<basename>.ann
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


INPUT_DIR = ""
OUT_DIR = ""


@dataclass(frozen=True)
class EntitySpan:
    label: str
    start: int
    end: int
    text: str


TAG_PATTERN = re.compile(r"<(/?)([A-Za-z0-9_:-]+)>")
ESCAPE_PATTERN = re.compile(r"&(?:amp|lt|gt);")


def unescape_text(s: str) -> str:
    """
    Decodes escaped text fragments produced by the inline tagging step.

    Arguments:
        s (str): Escaped text fragment.
    Return:
        str: Decoded raw text fragment.
    """
    return (
        s.replace("&lt;", "<")
         .replace("&gt;", ">")
         .replace("&amp;", "&")
    )


def decode_next_fragment(tagged_text: str, start_idx: int) -> Tuple[str, int]:
    """
    Decodes the next text fragment starting at the current position.

    This function handles escaped entities and single raw characters. It always
    advances the cursor by at least one character.

    Arguments:
        tagged_text (str): Full tagged content.
        start_idx (int): Current cursor position.
    Return:
        Tuple[str, int]: Decoded fragment and next cursor position.
    """
    escape_match = ESCAPE_PATTERN.match(tagged_text, start_idx)
    if escape_match:
        encoded = escape_match.group(0)
        return unescape_text(encoded), escape_match.end()

    return tagged_text[start_idx], start_idx + 1


def parse_inline_tagged_text(tagged_text: str) -> Tuple[str, List[EntitySpan], Dict[str, int]]:
    """
    Parses inline-tagged text and reconstructs plain text plus BRAT spans.

    Invalid tags are treated as raw text so the parser cannot stall on malformed input.

    Arguments:
        tagged_text (str): Inline-tagged content.
    Return:
        Tuple[str, List[EntitySpan], Dict[str, int]]: Plain text, extracted entity spans,
        and parsing statistics.
    """
    plain_parts: List[str] = []
    spans: List[EntitySpan] = []
    open_spans: Dict[str, List[int]] = {}

    i = 0
    plain_pos = 0
    n = len(tagged_text)

    malformed_tag_starts = 0
    unmatched_closing_tags = 0
    unclosed_open_tags = 0

    while i < n:
        if tagged_text[i] == "<":
            tag_match = TAG_PATTERN.match(tagged_text, i)

            if tag_match:
                is_closing = bool(tag_match.group(1))
                label = tag_match.group(2)

                if is_closing:
                    if label in open_spans and open_spans[label]:
                        start = open_spans[label].pop()
                        end = plain_pos
                        if end > start:
                            entity_text = "".join(plain_parts)[start:end]
                            spans.append(EntitySpan(label=label, start=start, end=end, text=entity_text))
                    else:
                        unmatched_closing_tags += 1
                else:
                    open_spans.setdefault(label, []).append(plain_pos)

                i = tag_match.end()
                continue

            malformed_tag_starts += 1

        decoded, next_i = decode_next_fragment(tagged_text, i)
        plain_parts.append(decoded)
        plain_pos += len(decoded)
        i = next_i

    for positions in open_spans.values():
        unclosed_open_tags += len(positions)

    plain_text = "".join(plain_parts)
    spans.sort(key=lambda x: (x.start, x.end, x.label, x.text))

    stats = {
        "malformed_tag_starts": malformed_tag_starts,
        "unmatched_closing_tags": unmatched_closing_tags,
        "unclosed_open_tags": unclosed_open_tags,
    }

    return plain_text, spans, stats


def format_brat_ann(spans: List[EntitySpan]) -> str:
    """
    Formats extracted spans into BRAT .ann T-lines.

    Arguments:
        spans (List[EntitySpan]): Entity spans.
    Return:
        str: BRAT annotation content.
    """
    lines: List[str] = []

    for idx, sp in enumerate(spans, start=1):
        mention = sp.text.replace("\n", " ")
        lines.append(f"T{idx}\t{sp.label} {sp.start} {sp.end}\t{mention}")

    return "\n".join(lines) + ("\n" if lines else "")


def find_tagged_files(input_dir: str) -> List[str]:
    """
    Finds tagged text files to process.

    Arguments:
        input_dir (str): Directory containing tagged files.
    Return:
        List[str]: Sorted list of tagged file paths.
    """
    tagged_files: List[str] = []

    for fn in os.listdir(input_dir):
        low = fn.lower()
        if low.endswith(".txt"):
            tagged_files.append(os.path.join(input_dir, fn))

    return sorted(tagged_files)


def build_output_base_name(tagged_path: str) -> str:
    """
    Builds the output basename from the tagged file name.

    Arguments:
        tagged_path (str): Path to tagged file.
    Return:
        str: Normalized basename for output files.
    """
    base = os.path.splitext(os.path.basename(tagged_path))[0]

    if base.endswith(".tagged"):
        return base[:-7]

    if base.endswith("t"):
        return base[:-1]

    return base


def write_restored_files(tagged_path: str, out_dir: str) -> None:
    """
    Restores .txt and .ann files from one tagged file.

    Arguments:
        tagged_path (str): Input tagged file path.
        out_dir (str): Output directory.
    Return:
        None
    """
    with open(tagged_path, "r", encoding="utf-8") as f:
        tagged_text = f.read()

    plain_text, spans, stats = parse_inline_tagged_text(tagged_text)
    ann_content = format_brat_ann(spans)

    base = build_output_base_name(tagged_path)
    txt_out = os.path.join(out_dir, f"{base}.txt")
    ann_out = os.path.join(out_dir, f"{base}.ann")

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(plain_text)

    with open(ann_out, "w", encoding="utf-8") as f:
        f.write(ann_content)

    print(f"Wrote {txt_out}")
    print(
        f"Wrote {ann_out} (spans={len(spans)}, malformed_tag_starts={stats['malformed_tag_starts']}, "
        f"unmatched_closing_tags={stats['unmatched_closing_tags']}, unclosed_open_tags={stats['unclosed_open_tags']})"
    )


def main() -> None:
    """
    Runs the inverse conversion for all tagged files in INPUT_DIR.

    Arguments:
        None
    Return:
        None
    """
    global INPUT_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Convert inline-tagged text into BRAT .txt/.ann files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing inline-tagged .txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory where restored .txt/.ann files will be written.")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUT_DIR = args.output_dir

    os.makedirs(OUT_DIR, exist_ok=True)

    tagged_files = find_tagged_files(INPUT_DIR)
    print(f"Found {len(tagged_files)} tagged files in {INPUT_DIR}")

    for tagged_path in tagged_files:
        write_restored_files(tagged_path, OUT_DIR)


if __name__ == "__main__":
    main()
