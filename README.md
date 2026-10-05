# Numerical format sensitivity in language models

This repository contains the code, datasets, model outputs, model revisions,
held-out tests, downstream evaluations, mitigation tests, cross-format probes,
representation analyses, causal interventions, and figures for a study of
numerical format sensitivity in Llama 3.1 8B, Qwen3 4B, and Gemma 2 9B. A
separately labeled reasoning-enabled scaling control uses a local 4-bit Gemma 3
27B checkpoint.

The manuscript is intentionally kept separate from this artifact repository.

## Main findings

- On 213 untouched GSM8K, FinQA, and TAT-QA problems, four exact-value numeral
  forms produced different normalized predictions on 69.5% to 80.3% of pooled
  items after exact decimal scoring. Two of 27 prespecified domain-level
  accuracy contrasts survived Holm correction.
- A selective policy that normalizes leading-zero and scientific notation while
  preserving negative decimals, long fractions, and signed zero improved fresh
  600-case accuracy by 15.2 to 18.8 points across all three models.
- On 200 further FinQA and TAT-QA problems, that frozen policy transferred for
  Qwen (+4.50 points, 95% CI +2.00 to +7.00) but not Llama (+0.83 points,
  95% CI -1.83 to +3.67). Blanket normalization was only 0.5 points higher
  than selective normalization for either model, with intervals spanning zero.
  Gemma 2 9B is excluded from this native-precision mitigation confirmation
  because its run is incomplete.
- Reasoning-enabled confirmations on all 200 bases show pooled
  scientific-minus-canonical changes of -7.0 points for Llama (95% CI -13.0
  to -1.0), -8.5 for Qwen (-13.5 to -3.5), and -1.0 for a separately labeled
  4-bit Gemma 3 27B scaling control (-7.5 to +5.5). No domain-level contrast
  survives within-model Holm correction.
- Under the same reasoning condition, the frozen selective policy recovers
  4.50 points for Llama (95% CI 1.17 to 7.83), 3.83 for Qwen (1.33 to 6.50),
  and an inconclusive 0.83 for Gemma 3 (-2.50 to 4.17).
- Llama 3.1 8B compares ordinary integers almost perfectly but is highly
  sensitive to decimal formatting, prompt wording, and numeral presentation
  order.
- Qwen3 4B and Gemma 2 9B are much more accurate on matched decimal tasks.
- In a new fully paired test with equality-label position controlled, Qwen and
  Gemma still showed large padded-first disadvantages under constrained-label
  scoring. Llama was poor in both orders and strongly label biased.
- In a fresh ten-template confirmation with 1,500 paired items per model, the
  pooled padded-first effect was negative for all three models. Only Gemma met
  the strict requirement that every template-level interval point in the same
  direction.
- Qwen passes the fixed-site causal-generalization criterion on two new prompt
  templates and an incompatible-value donor test.
- Gemma passes the donor test, but its easy-source rescue does not generalize
  across both new prompt templates.
- In a final 150-case test with 50 matched random-site controls per component
  and case, only Qwen's joint numeral patch passes. Gemma and Llama have
  positive padded-only effects that cancel or reverse under joint patching.
- Canonicalization improves average broad-format accuracy in all three models,
  but harms some format families and is not a universal fix.
- In the earlier 200-problem GSM8K and FinQA test, canonical and padded versions
  produced different numeric predictions in 52% to 65% of matched pairs. No
  model-domain accuracy contrast passed the six-test multiplicity correction.
- A frozen Qwen decomposition selected four attention heads at layer 2 using 30
  discovery cases. Their joint aligned-minus-random margin effect was 0.646 on
  90 held-out cases, with a 95% bootstrap interval of [0.486, 0.820], and the
  interval remained positive in all three unseen templates.

See `RESULTS_REPORT.md` for the complete result summary and
`MECHANISTIC_REPORT.md` for the mechanistic evidence and claim boundaries.

Downstream accuracy uses exact decimal equality, requires percent units to
match, and applies no rounding tolerance. The scoring correction and audit are
included with the reproducibility artifacts.

## Exact models

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`

The separate scaling control uses
`mlx-community/gemma-3-text-27b-it-4bit`, revision
`feccbf793f8404211939458acfa9b857f22a9fe4`. It is not treated as an
exact-precision comparison with the three checkpoints above.

The quantized behavioral controls use Ollama's `llama3.1:8b` package (ID
`46e0c10c039e`, weight SHA-256
`667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`)
and `gemma2:9b` package (ID `ff02c3702f32`, weight SHA-256
`ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373`).
They are reported separately from the official unquantized checkpoints.
Gemma's official template rejects a system role, so the same instruction was
prepended to the user message and this mode is recorded in every output row.
All runs use greedy decoding.

## Repository structure

- `data/`: deterministic generated datasets.
- `results/`: raw responses, causal rows, and checked analysis outputs.
- `scripts/`: generation, evaluation, analysis, and figure code.
- `figures/`: publication figures.
- `activations/`: row metadata for the regenerable activation arrays.

The transformed GSM8K, FinQA, and TAT-QA subsets retain the upstream MIT
notices in `THIRD_PARTY_NOTICES.md`. That notice does not set a license for the
rest of this repository.

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
python scripts/validate_paired_generalization.py
python scripts/analyze_paired_generalization.py
python scripts/analyze_prompt_robustness_confirmation.py
python scripts/finalize_prompt_robustness_analysis.py
python scripts/analyze_causal_random_site_confirmation.py
python scripts/validate_final_confirmation.py
python scripts/build_final_confirmation_figures.py
python scripts/analyze_downstream_invariance.py
python scripts/analyze_qwen_component_causal.py
python scripts/build_downstream_figure.py
python scripts/build_qwen_component_figure.py
python scripts/analyze_numeric_invariance.py
python scripts/analyze_selective_normalization.py
python scripts/build_numeric_invariance_figures.py
python scripts/analyze_downstream_mitigation_postcrash.py
python scripts/build_downstream_mitigation_figure.py
python scripts/analyze_reasoning_confirmation.py
python scripts/build_reasoning_confirmation_figure.py
python scripts/analyze_reasoning_cross_model.py
python scripts/build_reasoning_cross_model_figure.py
```

The three activation arrays total about 1.7 GB and are excluded from the public
repository. Their exact row metadata and resume-safe extraction script are
included. Regenerate them with `scripts/extract_mechanistic_activations.py` if
you want to rerun the activation analyses.
