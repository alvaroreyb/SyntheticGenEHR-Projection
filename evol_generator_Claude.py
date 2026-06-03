"""
claude_evolutivo_generator_delimited.py

This script generates Spanish hospital-style clinical progress notes from source
clinical cases stored as .txt files using the Anthropic Messages API (Claude).

The implementation uses client.messages.create() with configurable model selection.
For models supporting extended thinking (claude-3-7-sonnet and earlier), the script
can enable it via the ENABLE_THINKING environment variable. For Claude 4+ models,
adaptive thinking is used via the effort parameter.

For each input case, the script prompts the model to produce a reformulated
clinical progress note bounded by ---INICIO--- and ---FIN--- delimiters.
The final note is extracted from those delimiters when possible. If they are
missing, the script falls back to the cleaned raw model output when it is long
enough.

The script writes generated notes to OUT_DIR, stores failed generations as JSON
files under an _errors directory, and creates a CSV execution log.
"""

import argparse
import os
import time
import json
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import anthropic


INPUT_DIR = ""
OUT_DIR = ""
LOG_CSV_PATH = ""

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6").strip()
REQUEST_TIMEOUT_SECONDS = 600

MAX_OUTPUT_TOKENS = 6000
OVERWRITE = False
MIN_OUTPUT_CHARS = 80

SYSTEM_PROMPT = (
    "Actúa como un médico especialista en Urología u Oncología que redacta notas evolutivas clínicas "
    "en una historia clínica electrónica hospitalaria en español. "
    "Devuelve únicamente la nota clínica solicitada, sin explicaciones ni metatexto."
)

USER_PROMPT_TEMPLATE = """
Se te proporcionará un caso clínico completamente anotado mediante etiquetas XML que identifican entidades clínicas (por ejemplo: `<ENFERMEDAD>`, `<Date>`, `<GLEASON>`, etc.). A partir de este caso clínico, genera un texto equivalente con el estilo y estructura propios de la narrativa de historia clínica de un paciente de cáncer de próstata, tal y como podría aparecer en un sistema de historia clínica real de un centro sanitario.

Las posibles etiquetas XML que identifican entidades clínicas son las siguientes: [`<SINTOMA>`, `<PROCEDIMIENTO>`, `<ENFERMEDAD>`, `<Age>`, `<Date>`, `<Dose>`, `<Duration>`, `<Frequency>`, `<Neg_cue>`, `<Negated>`, `<Spec_cue>`, `<Speculated>`, `<Time>`, `<PROTEINAS>`, `<UNCLEAR>`, `<SPECIES>`, `<HUMAN>`, `<NORMALIZABLES>` `<GLEASON>`, `<PSA>`].

El texto generado debe:

1. Mantener estrictamente la información y los datos clínicamente relevantes desde el punto de vista del manejo médico urológico.
2.Prescindir única y exclusivamente de datos que no tengan relevancia ni significación clínica desde el punto de vista del manejo (diagnóstico, tratamiento, plan de actuación, estadificación, etc.) de la patología de cáncer de próstata.
3. No añadir datos nuevos que no estén explícitamente presentes.
4. No omitir información clínicamente relevante para la patología de cáncer de próstata.
5. No inferir resultados analíticos, radiológicos, de estadiaje, de PSA, de Gleason, ni datos anatomopatológicos si no aparecen en el caso clínico original.
6. Reformular completamente el texto (no copiar frases literalmente) con el estilo y estructura de narrativa de historia clínica.
7. Utilizar estilo clínico habitual en historias clínicas de pacientes reales de cáncer de próstata:
    - Utiliza las frases más breves posibles, directas y clínicamente informativas.
    - Minimiza y condensa al máximo el texto generado, omitiendo todo contenido prescindible, evitando redundancias y priorizando frases breves, compactas y de alta densidad informativa.
    - Evita el lenguaje narrativo o académico del caso clínico original, propio de artículos científicos pero no de historia clínica.
    - No incluyas comentarios explicativos, conectores entre frases ni texto de relleno prescindible.
    - Evita el uso de verbos en las frases siempre que sea posible (p. ej., "Evolución favorable con PSA 0.04 a las 4 semanas" mejor que "Presenta evolución favorable con PSA 0.04 a las 4 semanas").
    - Emplea las abreviaturas clínicas habituales cuando sea apropiado (p. ej., ADC, dx, tto, pte, RT, UCI, ADT, etc.).
    - Utiliza terminología médica habitual en la práctica clínica hospitalaria española.
8. Adapta, en la medida de lo posible, la prosa narrativa extensa del caso clínico original en un formato parecido a este estilo (sin que las secciones se muestre de forma explícita):
    - Información paciente.
    - Diagnóstico inicial + fecha + tratamiento inicial.
    - Progresión + momento temporal + manejo.
    - Complicación/evoluciones + hallazgos + actuación.
    - Situación final + evolución clínica.
9. Mantener coherencia temporal, tal y como está reflejada en el caso clínico original (fechas, intervalos temporales, etc.).
10. Conservar las dosis (sin necesidad específica de mantener las unidades), frecuencias, procedimientos y diagnósticos del caso clínico original.
11. Contener las entidades etiquetadas de la misma manera que se te han proporcionado, es decir siguiendo una estructura de etiquetas similar a XML

No expliques lo que haces. Devuelve únicamente la nota clínica generada con las entidades debidamente identificadas e insertadas.

---INICIO---
(aquí la nota)
---FIN---
CASO CLÍNICO ORIGINAL:
{CASE_TEXT}


"""

@dataclass
class RunResult:
    """
    Stores execution metadata for each processed input file.

    Arguments:
        None
    Return:
        None
    """
    input_path: str
    output_path: str
    status: str
    error: str
    latency_s: float
    input_chars: int
    output_chars: int
    model: str
    prompt_hash: str
    response_hash: str


def ensure_dir(path: str) -> None:
    """
    Creates a directory if it does not already exist.

    Arguments:
        path (str): Directory path to create.
    Return:
        None
    """
    os.makedirs(path, exist_ok=True)


def iter_txt_files(root_dir: str) -> List[str]:
    """
    Recursively collects all .txt files under the provided root directory.

    Arguments:
        root_dir (str): Root directory to scan.
    Return:
        List[str]: Sorted list of .txt file paths.
    """
    out: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(base, fn))
    return sorted(out)


def sha1_text(text: str) -> str:
    """
    Computes the SHA1 hash of a text string.

    Arguments:
        text (str): Input text.
    Return:
        str: SHA1 hexadecimal digest.
    """
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_relpath(path: str, start: str) -> str:
    """
    Computes a stable relative path and falls back to basename if needed.

    Arguments:
        path (str): Input absolute path.
        start (str): Base directory used for relative path computation.
    Return:
        str: Relative path or basename fallback.
    """
    try:
        rel = os.path.relpath(path, start)
        if rel.startswith(".."):
            return os.path.basename(path)
        return rel
    except Exception:
        return os.path.basename(path)


def output_path_for_input(input_path: str) -> str:
    """
    Resolves the output .txt path corresponding to an input case.

    Arguments:
        input_path (str): Source input file path.
    Return:
        str: Destination output file path.
    """
    rel = safe_relpath(input_path, INPUT_DIR)
    out_path = os.path.join(OUT_DIR, rel)
    return os.path.splitext(out_path)[0] + ".txt"


def write_text_atomic(path: str, text: str) -> None:
    """
    Writes text atomically to the target path.

    Arguments:
        path (str): Destination file path.
        text (str): Content to write.
    Return:
        None
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def extract_delimited_note(text: str) -> str:
    """
    Extracts the note contained between ---INICIO--- and ---FIN--- delimiters.

    Arguments:
        text (str): Model output text.
    Return:
        str: Extracted note or empty string when delimiters are missing.
    """
    match = re.search(r"(?s)---INICIO---\s*(.*?)\s*---FIN---", text)
    if not match:
        return ""
    return match.group(1).strip()


def extract_text_from_response(response: anthropic.types.Message) -> str:
    """
    Extracts all visible text_block content from an Anthropic Message response,
    skipping thinking blocks.

    Arguments:
        response (anthropic.types.Message): Anthropic API response object.
    Return:
        str: Concatenated text content from all text blocks.
    """
    pieces: List[str] = []
    for block in response.content:
        if block.type == "text":
            if block.text.strip():
                pieces.append(block.text.strip())
    return "\n".join(pieces).strip()


def resolve_note(content: str) -> Tuple[str, str]:
    """
    Resolves the final note from raw model content using a cascade strategy.

    Arguments:
        content (str): Raw generated content.
    Return:
        Tuple[str, str]: Final note text and resolution strategy label.
    """
    if content:
        note = extract_delimited_note(content)
        if note:
            return note, "delimited_content"

        if len(content) >= MIN_OUTPUT_CHARS:
            return content, "full_content"

    return "", "no_output"


def response_status_details(response: anthropic.types.Message) -> str:
    """
    Builds a compact diagnostic string from the Anthropic Message response.

    Arguments:
        response (anthropic.types.Message): Anthropic API response object.
    Return:
        str: Diagnostic summary string.
    """
    stop_reason = getattr(response, "stop_reason", "unknown") or "unknown"
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", "unknown") if usage else "unknown"
    output_tokens = getattr(usage, "output_tokens", "unknown") if usage else "unknown"
    cache_read = getattr(usage, "cache_read_input_tokens", "") if usage else ""
    cache_creation = getattr(usage, "cache_creation_input_tokens", "") if usage else ""

    diag = (
        f"stop_reason={stop_reason} "
        f"input_tokens={input_tokens} "
        f"output_tokens={output_tokens}"
    )
    if cache_read != "":
        diag += f" cache_read={cache_read}"
    if cache_creation != "":
        diag += f" cache_creation={cache_creation}"
    return diag


def call_claude(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> anthropic.types.Message:
    """
    Sends a generation request to the Anthropic Messages API.
    Uses client.messages.create() with system + user message structure.

    Arguments:
        client (anthropic.Anthropic): Initialized Anthropic client.
        model (str): Claude model name.
        system_prompt (str): System instruction text.
        user_prompt (str): User message text.
    Return:
        anthropic.types.Message: Raw Anthropic response object.
    """
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return response


def generate_note(
    client: anthropic.Anthropic,
    model_name: str,
    case_text: str,
) -> Tuple[str, Dict[str, Any], str, str]:
    """
    Generates a clinical progress note for a given source case using Claude.

    Arguments:
        client (anthropic.Anthropic): Initialized Anthropic client.
        model_name (str): Claude model name to use.
        case_text (str): Source clinical case text.
    Return:
        Tuple[str, Dict[str, Any], str, str]:
            - final resolved note,
            - raw response dump,
            - status string,
            - diagnostic text.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(CASE_TEXT=case_text)
    raw_response = call_claude(
        client=client,
        model=model_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    try:
        raw_dump = raw_response.model_dump()
    except Exception:
        raw_dump = {"raw_response_repr": repr(raw_response)}

    raw_text = extract_text_from_response(raw_response)
    diag = response_status_details(raw_response)

    print(f"    response_chars={len(raw_text)}")
    print(f"    {diag}")

    note, strategy = resolve_note(raw_text)
    print(f"    resolution_strategy={strategy}")

    if note:
        return note, raw_dump, "ok", diag

    return "", raw_dump, "empty_output", diag


def write_log_csv(rows: List[RunResult], csv_path: str) -> None:
    """
    Writes the execution summary to a CSV log file.

    Arguments:
        rows (List[RunResult]): Collected run results.
        csv_path (str): Destination CSV path.
    Return:
        None
    """
    header = [
        "input_path",
        "output_path",
        "status",
        "error",
        "latency_s",
        "input_chars",
        "output_chars",
        "model",
        "prompt_hash",
        "response_hash",
    ]

    ensure_dir(os.path.dirname(csv_path) or ".")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rr in rows:
            vals = [
                rr.input_path.replace('"', '""'),
                rr.output_path.replace('"', '""'),
                rr.status.replace('"', '""'),
                rr.error.replace('"', '""'),
                f"{rr.latency_s:.3f}",
                str(rr.input_chars),
                str(rr.output_chars),
                rr.model,
                rr.prompt_hash,
                rr.response_hash,
            ]
            f.write(",".join(f'"{v}"' for v in vals) + "\n")


def main() -> None:
    """
    Processes all .txt source cases and writes generated progress notes to disk.

    Arguments:
        None
    Return:
        None
    """
    global INPUT_DIR, OUT_DIR, LOG_CSV_PATH, CLAUDE_MODEL, OVERWRITE

    parser = argparse.ArgumentParser(description="Generate clinical notes with Claude.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source .txt clinical cases.")
    parser.add_argument("--output-dir", required=True, help="Directory where generated notes will be written.")
    parser.add_argument("--log-csv", help="CSV execution log path. Defaults to <output-dir>/log.csv.")
    parser.add_argument("--model", default=CLAUDE_MODEL, help="Claude model name.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even when outputs already exist.")
    args = parser.parse_args()

    INPUT_DIR = args.input_dir
    OUT_DIR = args.output_dir
    LOG_CSV_PATH = args.log_csv or os.path.join(OUT_DIR, "log.csv")
    CLAUDE_MODEL = args.model
    OVERWRITE = args.overwrite

    ensure_dir(OUT_DIR)
    err_dir = os.path.join(OUT_DIR, "_errors")
    ensure_dir(err_dir)

    all_files = iter_txt_files(INPUT_DIR)
    files = [
        in_path
        for in_path in all_files
        if OVERWRITE or not os.path.exists(output_path_for_input(in_path))
    ]
    existing_count = len(all_files) - len(files)

    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUT_DIR:   {OUT_DIR}")
    print(f"MODEL:     {CLAUDE_MODEL}")
    print(f"Found {len(all_files)} source .txt files")
    print(f"Existing outputs: {existing_count}")
    print(f"Pending cases:    {len(files)}")

    if not files:
        print("No pending cases. Nothing to generate.")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    results: List[RunResult] = []

    for i, in_path in enumerate(files, 1):
        out_path = output_path_for_input(in_path)
        ensure_dir(os.path.dirname(out_path))

        with open(in_path, "r", encoding="utf-8", errors="replace") as f:
            case_text = f.read().strip()

        print(f"\n[{i}/{len(files)}] IN:  {in_path}")
        print(f"[{i}/{len(files)}] OUT: {out_path}")
        print(f"[{i}/{len(files)}] in_chars={len(case_text)} overwrite={OVERWRITE}")

        if not case_text:
            results.append(
                RunResult(
                    input_path=in_path,
                    output_path=out_path,
                    status="empty_input",
                    error="Input .txt is empty after strip().",
                    latency_s=0.0,
                    input_chars=0,
                    output_chars=0,
                    model=CLAUDE_MODEL,
                    prompt_hash="",
                    response_hash="",
                )
            )
            print(f"[{i}/{len(files)}] SKIP: empty input")
            continue

        if os.path.exists(out_path) and not OVERWRITE:
            results.append(
                RunResult(
                    input_path=in_path,
                    output_path=out_path,
                    status="skipped_existing",
                    error="",
                    latency_s=0.0,
                    input_chars=len(case_text),
                    output_chars=0,
                    model=CLAUDE_MODEL,
                    prompt_hash="",
                    response_hash="",
                )
            )
            print(f"[{i}/{len(files)}] SKIP: output exists and OVERWRITE=False")
            continue

        prompt_hash = sha1_text(case_text + SYSTEM_PROMPT + USER_PROMPT_TEMPLATE)

        t0 = time.time()
        status = "ok"
        err = ""
        output = ""
        raw_json: Optional[Dict[str, Any]] = None

        try:
            output, raw_json, status, diag = generate_note(client, CLAUDE_MODEL, case_text)
            if status != "ok" and diag:
                err = diag
        except anthropic.RateLimitError as e:
            status = "rate_limit_error"
            err = str(e)
        except anthropic.APITimeoutError as e:
            status = "timeout_error"
            err = str(e)
        except anthropic.APIStatusError as e:
            status = f"api_error_{e.status_code}"
            err = str(e)
        except Exception as e:
            status = "error"
            err = str(e)

        latency = time.time() - t0

        if status == "ok":
            write_text_atomic(out_path, output.strip() + "\n")
            print(f"[{i}/{len(files)}] WROTE: out_chars={len(output)} latency={latency:.2f}s")
        else:
            base = os.path.splitext(os.path.basename(in_path))[0]
            err_path = os.path.join(err_dir, f"{base}.{status}.error.json")
            dump = {
                "input_path": in_path,
                "output_path": out_path,
                "status": status,
                "error": err,
                "model": CLAUDE_MODEL,
                "latency_s": latency,
                "input_chars": len(case_text),
                "raw_json": raw_json,
            }
            write_text_atomic(err_path, json.dumps(dump, ensure_ascii=False, indent=2) + "\n")
            print(f"[{i}/{len(files)}] {status.upper()}: {err if err else 'no_output'}")
            print(f"[{i}/{len(files)}] Saved error: {err_path}")

        results.append(
            RunResult(
                input_path=in_path,
                output_path=out_path,
                status=status,
                error=err,
                latency_s=latency,
                input_chars=len(case_text),
                output_chars=len(output),
                model=CLAUDE_MODEL,
                prompt_hash=prompt_hash,
                response_hash=sha1_text(output) if output else "",
            )
        )

    write_log_csv(results, LOG_CSV_PATH)
    print(f"\nWrote log: {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()
