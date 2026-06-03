"""
llm_as_judge_ollama_resume.py

This script evaluates generated Spanish clinical progress notes as independent
clinical documents using an Ollama-compatible chat endpoint.

It keeps the generated-notes directory, output directory, Ollama configuration,
system prompt and user prompt from the reference-free evaluator, while adding the
resumable execution, strict result validation, JSONL/CSV persistence and retry
logic used in the second workflow.

Inputs:
    GENERATED_NOTES_ROOT_DIR: Directory containing generated note .txt files.

Outputs:
    OUTPUT_DIR/results.jsonl       Clean successful evaluation records.
    OUTPUT_DIR/results.csv         Flattened successful results.
    OUTPUT_DIR/parse_errors.jsonl  Processing, parsing and API errors.
    OUTPUT_DIR/debug.jsonl         Optional raw responses when enabled.
"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests


GENERATED_NOTES_ROOT_DIR = ""
OUTPUT_DIR = ""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "Kimi-k2.6:cloud").strip()

REQUEST_TIMEOUT_SECONDS = 600
REQUEST_SLEEP_SECONDS = 0.2
MAX_RETRIES = 3

WRITE_DEBUG_JSONL = False

OLLAMA_FORMAT_JSON = os.environ.get("OLLAMA_FORMAT_JSON", "1").strip().lower() not in {"0", "false", "no"}
OLLAMA_THINK = os.environ.get("OLLAMA_THINK")
OLLAMA_OPTIONS_JSON = os.environ.get("OLLAMA_OPTIONS_JSON", "").strip()

RESULTS_JSONL_PATH = ""
RESULTS_CSV_PATH = ""
PARSE_ERRORS_JSONL_PATH = ""
DEBUG_JSONL_PATH = ""


def configure_output_paths(output_dir: str) -> None:
    """
    Configure result file paths after parsing command-line arguments.
    """
    global OUTPUT_DIR, RESULTS_JSONL_PATH, RESULTS_CSV_PATH, PARSE_ERRORS_JSONL_PATH, DEBUG_JSONL_PATH

    OUTPUT_DIR = output_dir
    RESULTS_JSONL_PATH = os.path.join(OUTPUT_DIR, "results.jsonl")
    RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
    PARSE_ERRORS_JSONL_PATH = os.path.join(OUTPUT_DIR, "parse_errors.jsonl")
    DEBUG_JSONL_PATH = os.path.join(OUTPUT_DIR, "debug.jsonl")

SCORE_KEYS = [
    "actualizada_temporalmente_correcta",
    "precisa",
    "exhaustiva",
    "util",
    "organizada",
    "comprensible",
    "concisa",
    "sintetizada",
    "consistencia_interna",
]

BOOLEAN_KEYS = [
    "error_seguridad_grave",
    "contradicciones_clinicas_insalvables",
    "human_generated",
]

SYSTEM_PROMPT = """
Actúa como un médico especialista en Urología u Oncología que evalúa notas evolutivas clínicas sintéticas con amplia experiencia en documentación clínica hospitalaria.

Evalúa la calidad clínica de la NOTA GENERADA (NG) como documento clínico independiente.

CRITERIOS DE EVALUACIÓN

1. Actualizada (temporalmente correcta)
Evalúa si la NG sigue un orden lógico y temporal en cuanto a evolución clínica, eventos, tratamientos y procedimientos.

2. Precisa
Evalúa si todos los datos presentes en la NG son clínicamente correctos. Penaliza alucinaciones o invención de datos.

3. Exhaustiva
Evalúa si la NG contiene los datos clínicos relevantes ÚNICAMENTE PARA LA ENFERMEDAD DE CÁNCER DE PRÓSTATA.

4. Útil
Evalúa si la NG contiene información clínicamente relevante para la toma de decisiones.

5. Organizada
Evalúa la estructura lógica y la organización del documento.

6. Comprensible
Evalúa la claridad del lenguaje y el uso adecuado de terminología médica.

7. Concisa
Evalúa si la NG evita redundancias innecesarias.

8. Sintetizada
Evalúa si la información clínica está integrada de forma coherente.

9. Consistencia interna
Evalúa si existen contradicciones clínicas dentro de la NG.

No trates acrónimos y acortamientos como errores, pueden ser parte de la sintetización del texto.

PREGUNTAS ADICIONALES

Existe algún error de seguridad grave:
errores clínicos potencialmente peligrosos (dosis incorrectas, errores de estadiaje, procedimientos inseguros, etc.).
Respuesta: SI/NO

Existe alguna contradicción clínica insalvable:
contradicciones que hacen la nota clínicamente incoherente.
Respuesta: SI/NO

Esta nota podría haber sido generada por un humano:
Respuesta: SI/NO

REGLAS IMPORTANTES
- No penalices cambios de estilo o reordenación si la información clínica es fiel.
- Penaliza fuertemente alucinaciones clínicas.
- Penaliza omisiones relevantes.
- Si detectas alucinaciones importantes, "precisa" ≤ 2.
- Si detectas contradicciones internas importantes, "consistencia_interna" ≤ 2.
- Cuando la puntuación de un criterio sea 5, el campo "justification" debe ser "".

ESCALA DE EVALUACIÓN

5 (Excelente): cumplimiento total del criterio evaluado.
4 (Adecuado): errores menores que no afectan a la seguridad clínica ni a la interpretación.
3 (Regular): errores moderados, pero el significado clínico general se mantiene.
2 (Deficiente): errores significativos u omisiones relevantes.
1 (Inaceptable): error crítico, información falsa peligrosa o incoherencia grave.

FORMATO DE RESPUESTA

Devuelve exclusivamente un JSON válido con esta estructura exacta:

{
  "actualizada_temporalmente_correcta": {"score": int, "justification": string},
  "precisa": {"score": int, "justification": string},
  "exhaustiva": {"score": int, "justification": string},
  "util": {"score": int, "justification": string},
  "organizada": {"score": int, "justification": string},
  "comprensible": {"score": int, "justification": string},
  "concisa": {"score": int, "justification": string},
  "sintetizada": {"score": int, "justification": string},
  "consistencia_interna": {"score": int, "justification": string},
  "error_seguridad_grave": boolean,
  "contradicciones_clinicas_insalvables": boolean,
  "human_generated": boolean
}

No incluyas texto fuera del JSON.
Cuando score sea 5, justification debe ser "".
"""

USER_PROMPT_TEMPLATE = """{generated_note}"""


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Arguments:
        path: Directory path.
    Return:
        None.
    """
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> str:
    """
    Read a UTF-8 text file with replacement for invalid characters.

    Arguments:
        path: File path.
    Return:
        File content without leading or trailing whitespace.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        return file.read().strip()


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    """
    Append a JSON object to a JSONL file and flush it to disk.

    Arguments:
        path: JSONL output path.
        item: JSON-serializable dictionary.
    Return:
        None.
    """
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_csv_snapshot(rows: List[Dict[str, Any]], path: str) -> None:
    """
    Write the current successful evaluation rows to CSV.

    Arguments:
        rows: Flattened successful evaluation rows.
        path: CSV output path.
    Return:
        None.
    """
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    """
    Parse an optional boolean environment value.

    Arguments:
        value: Raw environment value.
    Return:
        Parsed boolean or None when the input is absent.
    """
    if value is None:
        return None

    text = value.strip().lower()

    if text in {"1", "true", "yes", "si", "sí"}:
        return True

    if text in {"0", "false", "no"}:
        return False

    return None


def load_ollama_options() -> Dict[str, Any]:
    """
    Load optional Ollama generation parameters from OLLAMA_OPTIONS_JSON.

    Arguments:
        None.
    Return:
        Dictionary with Ollama options.
    """
    if not OLLAMA_OPTIONS_JSON:
        return {}

    try:
        options = json.loads(OLLAMA_OPTIONS_JSON)
    except Exception as error:
        raise ValueError(f"Invalid OLLAMA_OPTIONS_JSON: {error}") from error

    if not isinstance(options, dict):
        raise ValueError("OLLAMA_OPTIONS_JSON must contain a JSON object.")

    return options


def build_ollama_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Build the Ollama-compatible /api/chat request payload.

    Arguments:
        messages: Chat messages.
    Return:
        Request payload dictionary.
    """
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    if OLLAMA_FORMAT_JSON:
        payload["format"] = "json"

    parsed_think = parse_optional_bool(OLLAMA_THINK)

    if parsed_think is not None:
        payload["think"] = parsed_think

    options = load_ollama_options()

    if options:
        payload["options"] = options

    return payload


def is_valid_generated_filename(filename: str) -> bool:
    """
    Check whether a filename is a candidate generated note.

    Arguments:
        filename: Filename.
    Return:
        True if the filename should be evaluated.
    """
    if not filename.lower().endswith(".txt"):
        return False

    banned_terms = [
        "report",
        "analysis",
        "stats",
        "summary",
        "metric",
        "metrics",
        "log",
        "readme",
    ]
    return not any(term in filename.lower() for term in banned_terms)


def get_default_strategy_name(root_dir: str) -> str:
    """
    Derive the default strategy name from the generated notes root directory.

    Arguments:
        root_dir: Generated notes root directory.
    Return:
        Strategy name.
    """
    return os.path.basename(os.path.normpath(root_dir))


def is_inside_directory(path: str, directory: str) -> bool:
    """
    Check whether one path is inside another directory.

    Arguments:
        path: Candidate path.
        directory: Parent directory.
    Return:
        True when path is the same directory or inside it.
    """
    path_abs = os.path.abspath(path)
    directory_abs = os.path.abspath(directory)
    return path_abs == directory_abs or path_abs.startswith(directory_abs + os.sep)


def should_skip_directory(path: str, output_dir: str) -> bool:
    """
    Decide whether a directory should be skipped during discovery.

    Arguments:
        path: Candidate directory path.
        output_dir: Output directory path.
    Return:
        True if the directory should not be scanned for notes.
    """
    return is_inside_directory(path, output_dir)


def discover_notes() -> List[Tuple[str, str, str, str]]:
    """
    Discover generated note files recursively.

    Arguments:
        None.
    Return:
        List of tuples with strategy, filename, relative path and filepath.
    """
    notes: List[Tuple[str, str, str, str]] = []
    root = GENERATED_NOTES_ROOT_DIR

    if not os.path.isdir(root):
        print(f"Generated notes directory not found: {root}")
        return notes

    default_strategy = get_default_strategy_name(root)

    for base, dirs, files in os.walk(root):
        dirs[:] = [
            dirname
            for dirname in dirs
            if not should_skip_directory(os.path.join(base, dirname), OUTPUT_DIR)
        ]

        if should_skip_directory(base, OUTPUT_DIR):
            continue

        relative_dir = os.path.relpath(base, root)
        strategy = default_strategy if relative_dir == "." else relative_dir.split(os.sep)[0]

        for filename in sorted(files):
            if not is_valid_generated_filename(filename):
                continue

            filepath = os.path.join(base, filename)
            relative_path = os.path.relpath(filepath, root).replace("\\", "/")
            notes.append((strategy, filename, relative_path, filepath))

    return sorted(set(notes), key=lambda item: (item[0], item[2]))


def make_key(strategy: Any, filename: Any, relative_path: Any = "") -> Tuple[str, str, str]:
    """
    Build a stable evaluation key.

    Arguments:
        strategy: Strategy name.
        filename: Generated note filename.
        relative_path: Path relative to the generated notes root.
    Return:
        Key tuple used to identify an evaluated note.
    """
    clean_strategy = str(strategy or "").strip()
    clean_filename = os.path.basename(str(filename or "").strip())
    clean_relative_path = str(relative_path or "").strip().replace("\\", "/")
    return clean_strategy, clean_filename, clean_relative_path


def candidate_keys(strategy: Any, filename: Any, relative_path: Any = "") -> Set[Tuple[str, str, str]]:
    """
    Build candidate keys for exact and legacy resume matching.

    Arguments:
        strategy: Strategy name.
        filename: Generated note filename.
        relative_path: Path relative to the generated notes root.
    Return:
        Set of candidate keys.
    """
    clean_strategy = str(strategy or "").strip()
    clean_filename = os.path.basename(str(filename or "").strip())
    clean_relative_path = str(relative_path or "").strip().replace("\\", "/")

    keys = {
        make_key(clean_strategy, clean_filename, clean_relative_path),
        make_key(clean_strategy, clean_filename, ""),
        make_key("", clean_filename, ""),
    }

    if clean_relative_path:
        keys.add(make_key("", clean_filename, clean_relative_path))

    return keys


def has_error_value(value: Any) -> bool:
    """
    Check whether a parse error field contains a real error.

    Arguments:
        value: Candidate error value.
    Return:
        True if the value represents an error.
    """
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null"}


def has_valid_scores_in_json(result: Any) -> bool:
    """
    Check whether a structured JSON result contains all valid score fields.

    Arguments:
        result: Candidate result object.
    Return:
        True if all score fields are present and valid.
    """
    if not isinstance(result, dict) or "parse_error" in result:
        return False

    for key in SCORE_KEYS:
        value = result.get(key)

        if not isinstance(value, dict):
            return False

        try:
            score = int(value.get("score"))
        except Exception:
            return False

        if score < 1 or score > 5:
            return False

    return True


def has_valid_scores_in_csv_row(row: Dict[str, Any]) -> bool:
    """
    Check whether a CSV row contains all valid score columns.

    Arguments:
        row: CSV row as a dictionary.
    Return:
        True if all required scores are present and valid.
    """
    if has_error_value(row.get("parse_error")):
        return False

    for key in SCORE_KEYS:
        column = f"{key}_score"

        if column not in row:
            return False

        try:
            score = int(float(str(row[column]).strip()))
        except Exception:
            return False

        if score < 1 or score > 5:
            return False

    return True


def load_completed_from_jsonl(path: str) -> Set[Tuple[str, str, str]]:
    """
    Load completed evaluation keys from a JSONL results file.

    Arguments:
        path: JSONL results path.
    Return:
        Set of completed evaluation keys.
    """
    completed: Set[Tuple[str, str, str]] = set()

    if not os.path.exists(path):
        return completed

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            if not isinstance(record, dict):
                continue

            if not has_valid_scores_in_json(record.get("result")):
                continue

            strategy = record.get("strategy", "")
            filename = record.get("filename", record.get("generated_filename", ""))
            relative_path = record.get("relative_path", "")
            completed.update(candidate_keys(strategy, filename, relative_path))

    return completed


def load_completed_from_csv(path: str) -> Set[Tuple[str, str, str]]:
    """
    Load completed evaluation keys from a CSV results file.

    Arguments:
        path: CSV results path.
    Return:
        Set of completed evaluation keys.
    """
    completed: Set[Tuple[str, str, str]] = set()

    if not os.path.exists(path):
        return completed

    try:
        dataframe = pd.read_csv(path)
    except Exception:
        return completed

    if dataframe.empty:
        return completed

    dataframe = dataframe.where(pd.notna(dataframe), None)

    for row in dataframe.to_dict(orient="records"):
        if not has_valid_scores_in_csv_row(row):
            continue

        strategy = row.get("strategy", "")
        filename = row.get("filename", row.get("generated_filename", ""))
        relative_path = row.get("relative_path", "")
        completed.update(candidate_keys(strategy, filename, relative_path))

    return completed


def load_existing_csv_rows(path: str) -> List[Dict[str, Any]]:
    """
    Load previous valid CSV rows to preserve them on new writes.

    Arguments:
        path: CSV results path.
    Return:
        Existing successful CSV rows.
    """
    if not os.path.exists(path):
        return []

    try:
        dataframe = pd.read_csv(path)
    except Exception:
        return []

    if dataframe.empty:
        return []

    dataframe = dataframe.where(pd.notna(dataframe), None)
    rows = dataframe.to_dict(orient="records")
    return [row for row in rows if has_valid_scores_in_csv_row(row)]


def is_completed(
    completed: Set[Tuple[str, str, str]],
    strategy: str,
    filename: str,
    relative_path: str,
) -> bool:
    """
    Check whether a generated note has already been evaluated.

    Arguments:
        completed: Set of completed keys.
        strategy: Strategy name.
        filename: Generated note filename.
        relative_path: Relative generated note path.
    Return:
        True if the note is already completed.
    """
    return bool(completed.intersection(candidate_keys(strategy, filename, relative_path)))


def strip_thinking(text: str) -> str:
    """
    Remove model thinking blocks from a response.

    Arguments:
        text: Raw model response text.
    Return:
        Text without thinking blocks.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_output(text: str) -> Any:
    """
    Parse model output as JSON with embedded-object recovery.

    Arguments:
        text: Raw model text.
    Return:
        Parsed JSON object.
    """
    cleaned = strip_thinking(text)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError("Could not parse model output as JSON.")

    return json.loads(match.group(0))


def normalize_bool(value: Any) -> bool:
    """
    Normalize a candidate value to boolean.

    Arguments:
        value: Candidate boolean value.
    Return:
        Normalized boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "1", "yes", "si", "sí"}:
            return True

        if text in {"false", "0", "no"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return False


def validate_result(result: Any) -> Dict[str, Any]:
    """
    Validate and normalize the model evaluation result.

    Arguments:
        result: Parsed model output.
    Return:
        Normalized evaluation dictionary.
    """
    if not isinstance(result, dict):
        return {"parse_error": f"unexpected_type_{type(result).__name__}"}

    validated: Dict[str, Any] = {}

    for key in SCORE_KEYS:
        value = result.get(key, {})

        if not isinstance(value, dict):
            value = {}

        score = value.get("score")
        justification = value.get("justification", "")

        try:
            score = int(score)
        except Exception:
            score = None

        if score is not None and not 1 <= score <= 5:
            score = None

        if not isinstance(justification, str):
            justification = ""

        if score == 5:
            justification = ""

        validated[key] = {
            "score": score,
            "justification": justification.strip(),
        }

    for key in BOOLEAN_KEYS:
        validated[key] = normalize_bool(result.get(key, False))

    return validated


def extract_text_from_ollama_response(response: Dict[str, Any]) -> str:
    """
    Extract assistant text from an Ollama /api/chat response.

    Arguments:
        response: Raw response dictionary.
    Return:
        Assistant message content.
    """
    message = response.get("message", {})

    if isinstance(message, dict):
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

    content = response.get("response", "")

    if isinstance(content, str):
        return content.strip()

    return ""


def call_ollama(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Send a chat completion request to the Ollama-compatible API.

    Arguments:
        messages: Chat messages.
    Return:
        Raw response dictionary.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = build_ollama_payload(messages)

    start = time.time()
    print("Sending request to Ollama API")

    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()
    print(f"Ollama response received in {time.time() - start:.2f} seconds")
    return data


def evaluate_note(generated_note: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Evaluate a single generated note as an independent clinical document.

    Arguments:
        generated_note: Generated clinical note.
    Return:
        Validated result, raw model text and raw response dictionary.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(generated_note=generated_note)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Attempt {attempt}/{MAX_RETRIES}")
            response = call_ollama(messages)
            raw_text = extract_text_from_ollama_response(response)

            if not raw_text:
                raise ValueError("Empty model response.")

            print(f"Raw response length: {len(raw_text)} characters")
            parsed = parse_json_output(raw_text)
            result = validate_result(parsed)

            if not has_valid_scores_in_json(result):
                raise ValueError("Model response does not contain all valid scores.")

            return result, raw_text, response

        except Exception as error:
            last_error = error
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {error}")

            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_SLEEP_SECONDS * attempt)

    raise RuntimeError(str(last_error))


def flatten_result(
    filename: str,
    strategy: str,
    relative_path: str,
    filepath: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten a nested evaluation result into one CSV row.

    Arguments:
        filename: Generated note filename.
        strategy: Strategy name.
        relative_path: Relative generated note path.
        filepath: Absolute generated note path.
        result: Evaluation result.
    Return:
        Flat dictionary for CSV output.
    """
    row: Dict[str, Any] = {
        "filename": filename,
        "strategy": strategy,
        "relative_path": relative_path,
        "filepath": filepath,
        "model": OLLAMA_MODEL,
        "provider": "ollama",
    }

    for key in SCORE_KEYS:
        value = result.get(key, {})
        row[f"{key}_score"] = value.get("score")
        row[f"{key}_justification"] = value.get("justification", "")

    for key in BOOLEAN_KEYS:
        row[key] = result.get(key)

    if "parse_error" in result:
        row["parse_error"] = result["parse_error"]

    return row


def build_success_record(
    filename: str,
    strategy: str,
    relative_path: str,
    filepath: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the JSONL record for a successful evaluation.

    Arguments:
        filename: Generated note filename.
        strategy: Strategy name.
        relative_path: Relative generated note path.
        filepath: Absolute generated note path.
        result: Evaluation result.
    Return:
        JSON-serializable record.
    """
    return {
        "filename": filename,
        "strategy": strategy,
        "relative_path": relative_path,
        "filepath": filepath,
        "model": OLLAMA_MODEL,
        "provider": "ollama",
        "result": result,
    }


def build_error_record(
    filename: str,
    strategy: str,
    relative_path: str,
    filepath: str,
    error: Exception,
) -> Dict[str, Any]:
    """
    Build the JSONL record for a failed evaluation.

    Arguments:
        filename: Generated note filename.
        strategy: Strategy name.
        relative_path: Relative generated note path.
        filepath: Absolute generated note path.
        error: Exception raised during evaluation.
    Return:
        JSON-serializable error record.
    """
    return {
        "filename": filename,
        "strategy": strategy,
        "relative_path": relative_path,
        "filepath": filepath,
        "model": OLLAMA_MODEL,
        "provider": "ollama",
        "parse_error": str(error),
    }


def print_startup_summary(note_files: List[Tuple[str, str, str, str]], rows: List[Dict[str, Any]]) -> None:
    """
    Print the initial execution summary.

    Arguments:
        note_files: Discovered generated note files.
        rows: Existing successful CSV rows.
    Return:
        None.
    """
    print(f"Found generated notes: {len(note_files)}")
    print(f"Existing CSV rows:     {len(rows)}")
    print(f"Model:                 {OLLAMA_MODEL}")
    print(f"Provider:              ollama")
    print(f"Ollama URL:            {OLLAMA_BASE_URL.rstrip('/')}/api/chat")
    print(f"Output CSV:            {RESULTS_CSV_PATH}")
    print(f"Output JSONL:          {RESULTS_JSONL_PATH}")

    if note_files:
        print("Sample files:")

        for item in note_files[:10]:
            print(f"  {item[2]}")


def main() -> None:
    """
    Run batch reference-free evaluation and skip previously completed notes.

    Arguments:
        None.
    Return:
        None.
    """
    global GENERATED_NOTES_ROOT_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, REQUEST_SLEEP_SECONDS, WRITE_DEBUG_JSONL

    parser = argparse.ArgumentParser(description="Evaluate generated clinical notes with Kimi through Ollama.")
    parser.add_argument("--generated-notes-dir", required=True, help="Root directory containing generated .txt notes.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation results will be written.")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama base URL.")
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=REQUEST_SLEEP_SECONDS,
        help="Seconds to sleep between requests.",
    )
    parser.add_argument("--debug-jsonl", action="store_true", help="Write raw model responses to debug.jsonl.")
    args = parser.parse_args()

    GENERATED_NOTES_ROOT_DIR = args.generated_notes_dir
    configure_output_paths(args.output_dir)
    OLLAMA_MODEL = args.model
    OLLAMA_BASE_URL = args.base_url
    REQUEST_SLEEP_SECONDS = args.request_sleep
    WRITE_DEBUG_JSONL = args.debug_jsonl

    if not GENERATED_NOTES_ROOT_DIR:
        raise RuntimeError("GENERATED_NOTES_ROOT_DIR is not configured.")

    if not OUTPUT_DIR:
        raise RuntimeError("OUTPUT_DIR is not configured.")

    if not OLLAMA_MODEL:
        raise RuntimeError("OLLAMA_MODEL is not configured.")

    ensure_dir(OUTPUT_DIR)

    note_files = discover_notes()
    completed = load_completed_from_jsonl(RESULTS_JSONL_PATH).union(load_completed_from_csv(RESULTS_CSV_PATH))
    rows = load_existing_csv_rows(RESULTS_CSV_PATH)

    print_startup_summary(note_files, rows)

    if not note_files:
        print("No generated note files found. Exiting.")
        write_csv_snapshot(rows, RESULTS_CSV_PATH)
        return

    processed = 0
    skipped = 0
    errors = 0

    for index, (strategy, filename, relative_path, filepath) in enumerate(note_files, 1):
        print(f"\n[{index}/{len(note_files)}] {relative_path}")
        print(f"Strategy: {strategy}")
        print(f"Path:     {filepath}")

        if is_completed(completed, strategy, filename, relative_path):
            skipped += 1
            print("Already evaluated. Skipping.")
            continue

        try:
            generated_note = read_text(filepath)
            print(f"Generated characters: {len(generated_note)}")

            result, raw_text, raw_response = evaluate_note(generated_note)
            record = build_success_record(filename, strategy, relative_path, filepath, result)
            append_jsonl(RESULTS_JSONL_PATH, record)

            if WRITE_DEBUG_JSONL:
                append_jsonl(
                    DEBUG_JSONL_PATH,
                    {
                        "filename": filename,
                        "strategy": strategy,
                        "relative_path": relative_path,
                        "filepath": filepath,
                        "model": OLLAMA_MODEL,
                        "provider": "ollama",
                        "raw_text": raw_text,
                        "raw_response": raw_response,
                    },
                )

            rows.append(flatten_result(filename, strategy, relative_path, filepath, result))
            write_csv_snapshot(rows, RESULTS_CSV_PATH)
            completed.update(candidate_keys(strategy, filename, relative_path))
            processed += 1

            print("Saved evaluation.")

        except Exception as error:
            errors += 1
            append_jsonl(
                PARSE_ERRORS_JSONL_PATH,
                build_error_record(filename, strategy, relative_path, filepath, error),
            )
            print(f"Saved error: {error}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    print(f"\nProcessed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"CSV:       {RESULTS_CSV_PATH}")
    print(f"JSONL:     {RESULTS_JSONL_PATH}")
    print(f"Errors:    {PARSE_ERRORS_JSONL_PATH}")


if __name__ == "__main__":
    main()
