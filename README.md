# Decimal comparison failures in language models

This repository contains the code, datasets, model outputs, model revisions,
held-out tests, cross-format probes, representation analyses, causal
interventions, and figures for a study of decimal comparison behavior in
Llama 3.1 8B, Qwen3 4B, and Gemma 2 9B.

The unpublished LessWrong draft is intentionally not included here.

## Main findings

- Llama 3.1 8B compares ordinary integers almost perfectly but is highly
  sensitive to decimal formatting, prompt wording, and numeral presentation
  order.
- Qwen3 4B and Gemma 2 9B are much more accurate on matched decimal tasks.
- Qwen passes the fixed-site causal-generalization criterion on two new prompt
  templates and an incompatible-value donor test.
- Gemma passes the donor test, but its easy-source rescue does not generalize
  across both new prompt templates.
- Canonicalization improves average broad-format accuracy in all three models,
  but harms some format families and is not a universal fix.

See `RESULTS_REPORT.md` for the complete result summary and
`MECHANISTIC_REPORT.md` for the mechanistic evidence and claim boundaries.

## Exact models

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`

The quantized behavioral controls use Ollama's `llama3.1:8b` package (ID
`46e0c10c039e`, weight SHA-256
`667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`)
and `gemma2:9b` package (ID `ff02c3702f32`, weight SHA-256
`ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373`).
They are reported separately from the official full-precision checkpoints.
Gemma's official template rejects a system role, so the same instruction was
prepended to the user message and this mode is recorded in every output row.
All runs use greedy decoding.

## Repository structure

- `data/`: deterministic generated datasets.
- `results/`: raw responses, causal rows, and checked analysis outputs.
- `scripts/`: generation, evaluation, analysis, and figure code.
- `figures/`: publication figures.
- `activations/`: row metadata for the regenerable activation arrays.

## Setup

Analysis and MLX inference were run on Apple silicon with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MLX is Apple-silicon-specific. The behavioral Ollama evaluator uses a local
Ollama server and Python's standard library.

## Useful commands

```bash
python scripts/make_datasets.py --n 1000
python scripts/build_report.py
python scripts/confirmatory_analysis.py
python scripts/analyze_causal_generalization.py
python scripts/analyze_token_decomposition.py
python scripts/analyze_format_robustness.py
```

The three activation arrays total about 1.7 GB and are excluded from the public
repository. Their exact row metadata and resume-safe extraction script are
included. Regenerate them with `scripts/extract_mechanistic_activations.py` if
you want to rerun the activation analyses.
