# Reasoning-enabled cross-model replication

Frozen before generating or inspecting any reasoning-enabled Llama or Gemma
output on this dataset.

## Question

Do exact-value format sensitivity and the frozen selective-normalization effect
observed under Qwen reasoning also appear in the other two official checkpoints
used throughout the paper?

## Models and data

- Llama: `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`
- Gemma: `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`
- Precision: official native unquantized checkpoints
- Data: all 200 base problems and 800 rows in
  `data/downstream_mitigation_confirmation.jsonl`
- Decoding: greedy, maximum 256 generated tokens

The dataset and Qwen reasoning results have already been observed. No
reasoning-enabled output from Llama or Gemma on this dataset has been generated
or inspected.

## Prompt and scoring

The instruction allows at most three short calculation lines and requires a
final line in the form `Final answer: <number>`. Gemma's official template does
not accept a system role, so the identical instruction is prepended to its user
message. The scorer accepts only the final marked answer, removes commas and a
currency symbol, supports scientific notation, compares by exact decimal value,
requires the percent marker to match, and applies no rounding tolerance.

## Primary analysis

For each model separately and for FinQA and TAT-QA separately, compare padded,
leading-zero, and scientific forms with canonical. Report paired accuracy
changes, 10,000 base-clustered bootstraps, exact McNemar tests, and Holm
correction across the six source-by-form contrasts within model. Also report a
secondary Holm correction across all 12 contrasts from both new models.

Report four-form prediction disagreement, robust accuracy, parse failures, and
pooled descriptive results. Do not convert a pooled result into a primary
confirmatory claim.

## Mitigation analysis

As a prespecified secondary analysis, apply the already frozen selective policy
to the three noncanonical forms: preserve padding and map leading-zero and
scientific forms to canonical. Report pooled and source-specific paired accuracy
changes with 10,000 base-clustered bootstraps. Compare blanket normalization
with selective normalization as a secondary contrast.

## Cross-model interpretation

The existing Qwen result is not part of the new inferential family because it
was already observed. Compare effect directions and intervals descriptively
across Qwen, Llama, and Gemma. Claim cross-model replication only for an outcome
whose new-model evidence supports that wording.

## Execution safety

Run one model per process, never concurrently. Use batch size 1, a 10 GB MLX
cache limit, and clear the MLX cache after every prompt. Run Llama first, then
Gemma only after the Llama process exits. Preserve completed rows for resume.
Stop on a repeated memory error, severe system instability, or low-memory
warning. Do not raise batch size during the confirmatory runs.
