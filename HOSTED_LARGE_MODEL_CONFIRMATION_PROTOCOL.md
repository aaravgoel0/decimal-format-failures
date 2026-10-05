# Hosted larger-model reasoning confirmation

Frozen before generating or inspecting any hosted larger-model output.

## Motivation

The official native Gemma 2 9B checkpoint cannot generate the reasoning
condition safely on the 24 GB local Mac. Both the one-element batch path and
the lower-memory serial path fail before row 1 with a Metal insufficient-memory
error. Raising the Mac's wired-memory limit is excluded for system safety.

Rather than substitute a quantized checkpoint, this study adds the stronger
generality control requested by review: a current, substantially larger Gemma
family model on the same frozen downstream dataset.

## Model and provider

- Model: `google/gemma-3-27b-it`
- Provider: DeepInfra through Hugging Face Inference Providers
- Provider route status at freeze: live
- Published routed price at freeze: $0.08 per million input tokens and $0.16
  per million output tokens
- Revision pinning: not exposed by the hosted provider

This is reported as a hosted scaling control, not as an exact-revision
replication of the local Gemma 2 checkpoint.

## Data, prompt, and decoding

- Data: all 200 base problems and 800 rows in
  `data/downstream_mitigation_confirmation.jsonl`
- Instruction: identical short-calculation and final-marker wording used in the
  local Qwen and Llama reasoning confirmations
- Chat mode: instruction prepended to one user message
- Maximum generation: 256 tokens
- Temperature: 0
- Seed: 20261004
- Concurrency: at most four requests

The scorer accepts only the final marked numeric answer, compares by exact
decimal value, requires the percent marker to match, and applies no rounding
tolerance.

## Analysis

Use the same six domain-by-form primary contrasts, 10,000 base-clustered
bootstraps, exact McNemar tests, and within-model Holm correction as the other
reasoning confirmations. Report pooled results descriptively. Apply the frozen
selective policy as a prespecified secondary analysis and compare it with
blanket normalization.

Compare Qwen 4B, Llama 8B, and hosted Gemma 27B descriptively. Generalization
claims must disclose the provider backend and lack of revision pinning.

## Execution integrity

Requests are resume-safe. Raw responses, provider, requested decoding fields,
finish reason, usage metadata, errors, and timing are preserved row by row.
Transient requests may be retried up to three times with backoff. Failed rows
are not silently omitted.
