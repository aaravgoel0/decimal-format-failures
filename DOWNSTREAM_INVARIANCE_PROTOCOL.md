# Frozen downstream format-invariance protocol

Frozen before generating any model output for this experiment.

## Question

Do mathematically equivalent trailing-zero changes alter performance on
multi-step word problems and financial reasoning, rather than only on direct
decimal comparisons?

## Sources and sampling

- GSM8K official test split at source commit
  `3101c7d5072418e28b9008a6636bde82a006892c`.
- FinQA official test split at source commit
  `0f16e2867befa6840783e58be38c9efb9229d742`.
- Select the first 100 test items in source order from each dataset that contain
  at least one plain decimal literal in the evaluated text and have a numeric
  answer.
- FinQA receives only the annotated gold evidence and question. This measures
  numerical reasoning with evidence supplied, not retrieval.

## Paired transformation

Every selected item produces two prompts. The canonical prompt strips trailing
zeros while retaining at least one fractional digit. The padded prompt appends
two, three, or four zeros according only to source index. All decimal literals
in the evaluated question and supplied evidence are transformed together. The
words, numerical values, correct answer, item order, and chat template are
otherwise identical.

## Models

Use the same pinned official native-precision checkpoints as the main study:

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

Use greedy decoding and a 96-token ceiling. Ask for a final numeric answer only.
Preserve raw generations. Parse the final numeric expression without requiring
the model to reproduce insignificant trailing zeros. Percentage answers retain
their percent scale.

## Outcomes

For each model and domain, report:

- canonical and padded accuracy;
- paired padded-minus-canonical accuracy difference;
- canonical-correct to padded-incorrect damage rate;
- canonical-incorrect to padded-correct gain rate;
- numerical prediction disagreement rate;
- 95 percent intervals from 10,000 paired item bootstraps;
- exact two-sided McNemar tests for accuracy changes.

The primary claim is downstream format sensitivity if at least one model-domain
cell has a nonzero paired accuracy interval after Holm correction across the six
model-domain cells. Prediction disagreement is descriptive and does not by
itself establish an accuracy effect.

## Claim boundary

This test estimates sensitivity on decimal-bearing subsets of GSM8K and FinQA.
It does not estimate performance on either complete benchmark and must not be
reported as a standard benchmark score.
