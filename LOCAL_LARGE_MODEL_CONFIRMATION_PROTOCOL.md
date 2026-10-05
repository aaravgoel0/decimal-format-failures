# Local larger-model reasoning confirmation

Frozen before downloading the checkpoint or generating any local larger-model
output.

## Motivation and status of earlier attempts

The exact full-precision Gemma 2 9B reasoning run exceeded the safe memory
available on the 24 GB Apple silicon host and produced no usable outcome. A
separate hosted Gemma 3 27B run then completed 364 of 800 rows before Hugging
Face stopped the remaining requests because the account's included inference
credits were depleted. The incomplete hosted outcomes are preserved but will
not be analyzed or combined with this study.

This study uses a local quantized conversion of Gemma 3 27B. It is a scaling
control, not an exact-precision comparison with the three primary checkpoints.

## Model

- Repository: `mlx-community/gemma-3-text-27b-it-4bit`
- Immutable revision: `feccbf793f8404211939458acfa9b857f22a9fe4`
- Source family: `google/gemma-3-27b-it`
- Precision: 4-bit MLX conversion
- Backend: MLX LM, serial generation only

The paper must label this model as quantized and must not merge its results
with the partial hosted full-precision run.

## Data, prompt, decoding, and scoring

- All 200 bases and 800 rows in
  `data/downstream_mitigation_confirmation.jsonl`
- The same short-calculation instruction and final-answer marker used for Qwen
  and Llama
- Greedy decoding
- Maximum generation of 256 tokens
- Batch size 1
- Strict final-marker parsing
- Exact decimal equality
- Exact agreement on whether the answer is a percentage
- No rounding tolerance

## Safety constraints

- MLX memory limit at most 18 GiB
- MLX buffer cache disabled
- One-row pilot before the full run
- Stop after the first memory failure
- Do not raise macOS wired-memory limits
- Do not run another large model concurrently

The one-row pilot remains part of the final run if its settings are identical.

## Analysis

Use the same six FinQA and TAT-QA domain-by-form contrasts, 10,000
base-clustered bootstraps, exact McNemar tests, and within-model Holm correction
as the Qwen and Llama reasoning confirmations. The selective policy remains a
prespecified secondary analysis. A secondary 12-test Holm family combines the
six new Llama contrasts and six local Gemma 3 contrasts. Cross-model summaries
must disclose quantization.

