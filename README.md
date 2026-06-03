# Generation and Evaluation of Realistic Synthetic Clinical Progress Notes for Prostate Cancer Using Large Language Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Version](https://img.shields.io/badge/Version-v0.1.0-informational)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![Domain](https://img.shields.io/badge/Domain-Spanish%20Clinical%20NLP-purple)
![Clinical Focus](https://img.shields.io/badge/Clinical%20Focus-Prostate%20Cancer-darkred)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides a research-oriented software framework for the **generation, evaluation, annotation conversion, clinical entity extraction, and entity-level assessment of synthetic Spanish hospital-style progress notes** focused on prostate cancer.

The project integrates **Large Language Model-based clinical note generation**, **LLM-as-a-judge evaluation**, and **biomedical entity extraction** using BSC models, MedSpaNER models, and rule-based clinical pattern detection for PSA and Gleason score mentions. It is intended for scientific experimentation, benchmarking, and reproducibility studies in Spanish clinical Natural Language Processing.

---

## Overview

This repository implements a complete experimental pipeline for transforming plain-text prostate cancer clinical cases into realistic synthetic clinical progress notes, evaluating their clinical quality, extracting relevant biomedical entities, and computing entity-level metrics against silver-standard Brat annotations.

The pipeline supports:

- generation of Spanish hospital-style clinical progress notes;
- evaluation of generated notes using an LLM-as-a-judge workflow;
- conversion between inline XML-like entity tags and Brat standoff annotations;
- extraction of clinically relevant entities from original and generated notes;
- export of structured annotations for statistical analysis and manual review;
- computation of global, file-level, and label-level entity metrics.

The software is designed for research workflows involving clinical text generation, clinical summarisation, information preservation, synthetic clinical documentation, and Spanish biomedical information extraction.

---

## Main Objectives

The main objectives of this repository are:

1. **Synthetic clinical text generation**  
   Generate realistic Spanish hospital-style progress notes from source prostate cancer clinical cases.

2. **Clinical quality evaluation**  
   Assess generated notes using a structured LLM-as-a-judge evaluation workflow with Kimi through an Ollama-compatible endpoint.

3. **Clinical entity extraction**  
   Extract relevant clinical entities using neural biomedical NLP models and rule-based extraction methods.

4. **Entity-level performance assessment**  
   Compare generated or predicted entity annotations against silver-standard Brat annotations.

5. **Reproducible experimentation**  
   Provide a modular and traceable framework for model comparison, evaluation, and downstream clinical NLP analysis.

---

## Pipeline Overview

```text
Original clinical cases (.txt)
        |
        v
Synthetic progress-note generation
        |
        |-- OpenAI Responses API
        |-- Anthropic Messages API
        |-- Ollama-compatible local or cloud models
        |
        v
Generated clinical progress notes (.txt)
        |
        v
LLM-as-a-judge clinical evaluation
        |
        v
Evaluation outputs (.jsonl, .csv, debug artifacts)
        |
        v
Annotation conversion and entity extraction
        |
        |-- Inline XML-like tags
        |-- Brat standoff annotations
        |-- BSC biomedical token-classification models
        |-- MedSpaNER clinical token-classification models
        |-- PSA regular-expression extraction
        |-- Gleason regular-expression extraction
        |
        v
Structured annotations (.csv, .json, .ann)
        |
        v
Entity-level metrics (.csv)
```

---

## Repository Contents

### `evol_generator_openai.py`

Generates Spanish hospital-style clinical progress notes from source clinical cases using the OpenAI Responses API or an OpenAI-compatible Zai endpoint.

Main functionality:

- recursively reads `.txt` clinical cases from an input directory;
- prompts the selected model to generate a hospital-style clinical progress note;
- supports OpenAI models such as `gpt-5.4-nano`;
- supports Zai/GLM models through `--provider zai`;
- extracts generated content from `---INICIO---` and `---FIN---` delimiters;
- applies fallback extraction when delimiters are unavailable;
- writes generated notes to an output directory;
- stores failed generations as structured JSON files;
- exports a CSV execution log.

---

### `evol_generator_ollama.py`

Generates Spanish hospital-style clinical progress notes from source clinical cases using an Ollama-compatible endpoint.

Main functionality:

- recursively reads `.txt` clinical cases from an input directory;
- supports local, remote, and cloud Ollama-compatible models;
- extracts generated content from delimiters;
- skips existing outputs unless `--overwrite` is provided;
- stores failed generations as structured JSON files;
- exports a CSV execution log.

---

### `evol_generator_Claude.py`

Generates Spanish hospital-style clinical progress notes from source clinical cases using the Anthropic Messages API.

Main functionality:

- recursively reads `.txt` clinical cases from an input directory;
- prompts Claude models to generate clinical progress notes;
- extracts generated content from delimiters;
- supports resumable execution by skipping existing outputs;
- stores failed generations as structured JSON files;
- exports a CSV execution log.

---

### `llm_as_judge_Kimi.py`

Evaluates generated clinical progress notes as independent clinical documents using Kimi through an Ollama-compatible endpoint.

Main functionality:

- recursively discovers generated `.txt` notes;
- evaluates multiple clinical quality dimensions on a structured 1-5 scale;
- records severe safety errors and unresolvable clinical contradictions;
- records whether the note could plausibly have been human-generated;
- stores compact JSONL evaluation records;
- exports CSV summaries for visual overview analysis;
- stores parsing and debugging artifacts;
- supports resumable execution.

---

### `tagged_txt.py`

Converts Brat standoff annotations (`.ann`) and raw text (`.txt`) into inline XML-like tagged text.

Main functionality:

- reads paired `.txt` and `.ann` files;
- inserts tags strictly according to Brat offsets;
- supports overlapping and crossing annotation patterns;
- escapes raw text to avoid tag conflicts;
- exports inline-tagged `.txt` files.

---

### `untag_txt.py`

Converts inline XML-like tagged text back into Brat standoff annotations and raw text.

Main functionality:

- parses inline-tagged `.txt` files;
- reconstructs plain text;
- rebuilds Brat-compatible entity spans;
- reports malformed, unmatched, and unclosed tags;
- exports `.txt` and `.ann` files.

---

### `entidades_all.py`

Runs the clinical entity extraction pipeline over original or generated clinical notes.

Main functionality:

- applies BSC biomedical and clinical token-classification models;
- applies MedSpaNER token-classification models;
- detects Gleason score mentions using regular expressions;
- detects PSA mentions using regular expressions;
- merges neural and rule-based outputs;
- exports per-patient JSON files;
- exports CSV mention tables;
- exports Brat-compatible `.ann` files.

---

### `metricas_gsa3.py`

Computes entity-level NER metrics by comparing predicted Brat annotations against a silver-standard annotation directory.

Main functionality:

- reads paired `.ann` files from two directories;
- applies a relaxed match criterion based on same label and same start or end offset;
- computes precision, recall, F1-score, and entity-level accuracy;
- exports detailed comparison rows;
- exports file-level, label-level, and global metric CSV files.

---

## Input Data

All scripts are designed to process plain-text `.txt` files or Brat-compatible `.ann` files.

- `evol_generator_openai.py`: source clinical cases in `.txt` format.
- `evol_generator_ollama.py`: source clinical cases in `.txt` format.
- `evol_generator_Claude.py`: source clinical cases in `.txt` format.
- `llm_as_judge_Kimi.py`: generated progress notes in `.txt` format.
- `tagged_txt.py`: paired `.txt` and `.ann` Brat files.
- `untag_txt.py`: inline-tagged `.txt` files.
- `entidades_all.py`: raw clinical notes in `.txt` format.
- `metricas_gsa3.py`: silver-standard and predicted `.ann` files.

Nested directory structures are supported where applicable.

---

## Output Data

Depending on the executed script, the repository may generate:

- `.txt` files containing synthetic clinical progress notes;
- `.csv` files containing execution logs, evaluation summaries, extracted mention tables, and entity metrics;
- `.json` files containing error files, debug files, and per-patient extraction outputs;
- `.jsonl` files containing compact LLM-as-a-judge evaluation records;
- `.ann` files containing Brat-compatible standoff annotations.

---

## Requirements

### Python Version

Recommended and supported version:

```text
Python 3.10+
```

The codebase was designed primarily for Python 3.10-based research environments.

---

### Core Dependencies

Core Python dependencies include:

```text
anthropic
openai
pandas
requests
torch
transformers
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

### NER Models

The NER pipeline requires access to the following models:

```python
"BSC-NLP4BIA/bsc-bio-ehr-es-livingner-species"
"BSC-NLP4BIA/bsc-bio-ehr-es-livingner-humano"
"BSC-NLP4BIA/bsc-bio-ehr-es-medprocner"
"BSC-NLP4BIA/bsc-bio-ehr-es-symptemist"
"BSC-NLP4BIA/bsc-bio-ehr-es-distemist"
"PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer"
"medspaner/roberta-es-clinical-trials-cases-medic-attr"
"medspaner/xlm-roberta-large-spanish-trials-cases-temp-ent"
"medspaner/roberta-es-clinical-trials-cases-neg-spec"
```

---

### Ollama

Ollama is required for Ollama-compatible generation and Kimi-based LLM-as-a-judge evaluation.

Install Ollama with:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Check the installed version:

```bash
ollama --version
```

For cloud models, subscription requirements and availability depend on the selected backend and execution date.

---

## Hardware Requirements

Hardware requirements depend on the selected workflow.

### API-Based Workflows

When using hosted APIs such as OpenAI or Anthropic, the scripts can be executed in a standard CPU environment.

Recommended minimum:

```text
CPU: 4 cores
RAM: 8 GB
GPU: Not required
```

### Local Model Inference and Entity Extraction

When using Ollama-served models or transformer-based NER models locally, a CUDA-compatible GPU is recommended.

Recommended configuration:

```text
CPU: 8 cores or higher
RAM: 24 GB or higher
GPU: CUDA-compatible GPU recommended
VRAM: Model-dependent
```

The exact requirements depend on the selected LLM, quantisation level, batch size, NER model configuration, and sequence length.

---

## Environment Variables

API credentials must be provided through environment variables.

```bash
export OPENAI_API_KEY="your_openai_api_key"
export NEBIUS_API_KEY="your_nebius_or_zai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

Optional LLM-as-a-judge controls:

```bash
export OLLAMA_FORMAT_JSON="1"
export OLLAMA_THINK="false"
export OLLAMA_OPTIONS_JSON='{"temperature": 0.0}'
```

---

## Usage

### Generate Synthetic Notes with OpenAI

```bash
python evol_generator_openai.py \
  --input-dir /path/to/cases \
  --output-dir /path/to/generated_notes \
  --provider openai \
  --model gpt-5.4-nano
```

Optional arguments:

```text
--overwrite
```

Regenerates outputs even when files already exist.

```text
--log-csv /path/to/log.csv
```

Specifies a custom CSV log path.

---

### Generate Synthetic Notes with Zai

```bash
python evol_generator_openai.py \
  --input-dir /path/to/cases \
  --output-dir /path/to/generated_notes \
  --provider zai \
  --model zai-org/GLM-5 \
  --base-url https://api.tokenfactory.nebius.com/v1/
```

---

### Generate Synthetic Notes with Ollama

```bash
python evol_generator_ollama.py \
  --input-dir /path/to/cases \
  --output-dir /path/to/generated_notes \
  --model qwen3.5:35b-a3b \
  --base-url http://127.0.0.1:11434
```

This mode is intended for local or self-hosted model inference using an Ollama-compatible endpoint.

---

### Generate Synthetic Notes with Claude

```bash
python evol_generator_Claude.py \
  --input-dir /path/to/cases \
  --output-dir /path/to/generated_notes \
  --model claude-sonnet-4-6
```

---

### Evaluate Generated Notes with Kimi

```bash
python llm_as_judge_Kimi.py \
  --generated-notes-dir /path/to/generated_notes \
  --output-dir /path/to/evaluation \
  --model Kimi-k2.6:cloud
```

Optional argument:

```text
--debug-jsonl
```

Stores raw model responses in `debug.jsonl`.

---

### Convert Brat Annotations to Inline Tags

```bash
python tagged_txt.py \
  --input-dir /path/to/brat_pairs \
  --output-dir /path/to/inline_tagged
```

---

### Convert Inline Tags Back to Brat

```bash
python untag_txt.py \
  --input-dir /path/to/inline_tagged \
  --output-dir /path/to/restored_brat
```

---

### Run Clinical Entity Extraction

```bash
python entidades_all.py \
  --untagged-dir /path/to/notes \
  --output-dir /path/to/SilverMentions \
  --min-score 0.85
```

This module exports structured entity extraction results in CSV, JSON, and Brat-compatible `.ann` formats.

---

### Compute Entity-Level Metrics

```bash
python metricas_gsa3.py \
  --silver-ann-dir /path/to/silver_ann \
  --pred-ann-dir /path/to/predicted_ann \
  --output-dir /path/to/EntEval
```

---

## Example End-to-End Workflow

```bash
python evol_generator_openai.py \
  --input-dir data/cases \
  --output-dir outputs/generated_gpt54nano \
  --provider openai \
  --model gpt-5.4-nano

python llm_as_judge_Kimi.py \
  --generated-notes-dir outputs/generated_gpt54nano \
  --output-dir outputs/evaluation_gpt54nano \
  --model Kimi-k2.6:cloud

python untag_txt.py \
  --input-dir outputs/generated_gpt54nano \
  --output-dir outputs/generated_gpt54nano_brat

python entidades_all.py \
  --untagged-dir outputs/generated_gpt54nano_brat \
  --output-dir outputs/generated_gpt54nano_brat/SilverMentions

python metricas_gsa3.py \
  --silver-ann-dir outputs/generated_gpt54nano_brat/SilverMentions \
  --pred-ann-dir outputs/generated_gpt54nano_brat \
  --output-dir outputs/generated_gpt54nano_brat/EntEval
```

This workflow performs:

1. synthetic clinical progress-note generation;
2. LLM-based clinical quality evaluation;
3. conversion into Brat-compatible files;
4. clinical entity extraction;
5. entity-level metric computation.

---

## Recommended Repository Structure

```text
repository/
  data/
    cases/
  outputs/
    GS3/
      Model/
        generated/
        brat/
        entities/
        evaluation/
    GS4/
      Model/
        generated/
        brat/
        entities/
        evaluation/
  logs/
  errors/
  PaperCode/
    evol_generator_openai.py
    evol_generator_ollama.py
    evol_generator_Claude.py
    llm_as_judge_Kimi.py
    tagged_txt.py
    untag_txt.py
    entidades_all.py
    metricas_gsa3.py
    requirements.txt
    README.md
```

---

## Reproducibility

Experimental results may vary depending on:

- model version;
- API behaviour;
- local inference backend;
- prompt revisions;
- decoding parameters;
- hardware configuration;
- dependency versions;
- input data versioning;
- annotation conversion strategy;
- entity-matching criterion.

For reproducible experimentation, it is recommended to record:

- input dataset version;
- model identifiers;
- provider or backend;
- prompt template version;
- decoding configuration;
- Python version;
- dependency versions;
- execution logs;
- output directories;
- generated notes;
- evaluation outputs;
- annotation and metric outputs.

Recommended reproducibility files:

```text
requirements.txt
environment.yml
metadata.json
run_config.json
```

---

## Data Governance and Privacy

This repository is intended for research workflows involving clinical text.

Any use of real or sensitive clinical data must comply with applicable legal, ethical, and institutional requirements. Users are responsible for ensuring that:

- all clinical data are properly anonymised or de-identified;
- data processing complies with applicable data protection regulations;
- ethics committee or institutional review board requirements are satisfied where applicable;
- access to data and generated outputs is appropriately restricted;
- API-based processing is compatible with the data governance framework of the project.

Clinical data should not be sent to external APIs unless this is explicitly permitted by the corresponding data governance protocol.

---

## Clinical Safety Statement

The generated notes, evaluations, and extracted annotations are intended exclusively for research and evaluation purposes.

They must not be used for:

- clinical diagnosis;
- treatment recommendation;
- direct patient management;
- replacement of professional clinical documentation;
- automated clinical decision-making.

All generated clinical content should be reviewed by qualified professionals before any possible downstream clinical interpretation.

---

## Known Limitations

The repository has the following limitations:

- generated outputs may contain omissions, hallucinations, or clinical inconsistencies;
- LLM-as-a-judge evaluation may be sensitive to prompt design and evaluator model behaviour;
- entity extraction performance depends on the coverage and limitations of the selected NER models;
- regex-based extraction of PSA and Gleason mentions may not capture all possible linguistic variants;
- entity-level metrics depend on the selected relaxed matching rule;
- the repository is designed for research workflows and has not been validated as clinical software.

---

## Versioning

Current software version:

```text
v0.1.0
```

Version meaning:

- `0.1.0`: initial research version.
- `0.x.x`: experimental research-stage development.
- `1.0.0`: stable release after formal validation and documentation review.

---

## Suggested Citation

If you use this repository in academic work, please cite the associated manuscript or software release.

```bibtex
@software{synthetic_clinical_progress_notes_2026,
  title        = {Generation and Evaluation of Realistic Synthetic Clinical Progress Notes for Prostate Cancer Using Large Language Models},
  author       = {Rey-Blanes, Alvaro and Moreno-Barea, Francisco J. and Veredas, Francisco J.},
  year         = {2026},
  version      = {0.1.0},
  note         = {Research software for Spanish clinical NLP and synthetic clinical note generation}
}
```

---

## License

This repository is released under the MIT License.

Before public release, verify that the selected license is compatible with all institutional, data governance, and third-party dependency requirements.

---

## Contact

For questions regarding the repository, reproducibility, or research use, please contact the corresponding author or repository maintainer.

```text
Maintainer: Alvaro Rey-Blanes
Institution: Inteligencia Computacional en Biomedicina, Universidad de Malaga
Email: alvaroreyb@uma.es
```
