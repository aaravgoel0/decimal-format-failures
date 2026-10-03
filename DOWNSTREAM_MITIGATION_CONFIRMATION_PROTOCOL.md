# Frozen downstream mitigation confirmation protocol

Frozen on 2026-10-02 at 22:26 UTC, before generating this block's dataset or
any study-row model output. A one-prompt provider feasibility check used an
unrelated instruction and did not expose any study item.

## Question

Does the previously selected, model-independent normalization policy improve
accuracy and prediction stability on fresh realistic numerical-reasoning
items, rather than only on synthetic comparisons?

## Sources and untouched sampling

- FinQA official test split at commit
  `0f16e2867befa6840783e58be38c9efb9229d742`.
- TAT-QA official development split at commit
  `870accc41953dcde885aabeb963d94aabdc0fbc3`.
- Eligibility, answer parsing, context construction, and literal rewriting are
  identical to the frozen numerical-invariance study.
- Skip the first 200 eligible FinQA items. The first 100 were used by the
  paired downstream study and the next 100 by the four-format study.
- Skip the first 100 eligible TAT-QA items used by the four-format study.
- Select the next 100 eligible items from each source in source order.
- The resulting confirmation set contains 200 base problems and 800 prompts.
  It is disjoint from every earlier downstream block by source identifier.

## Exact-value forms

Each base problem has four value-equivalent versions: `canonical`, `padded`,
`leading_zero`, and `scientific`. Every eligible decimal literal in a prompt
is rewritten together. The generator must verify exact `Decimal` equality for
every literal and preserve the gold answer.

## Frozen policies

The policies were chosen from earlier synthetic results and do not depend on
this confirmation set or on any model in this block.

- `original`: retain the observed noncanonical representation.
- `selective`: preserve `padded`; map `leading_zero` and `scientific` to the
  byte-identical canonical prompt.
- `blanket`: map all three noncanonical forms to the byte-identical canonical
  prompt.

Canonical outputs are reused when a policy produces the exact same prompt.
This avoids stochastic or provider variability and makes the number of model
calls auditable. It is not counted as a separate model run.

## Primary models

Use the same native-precision checkpoints, chat templates, greedy decoding,
and 96-token ceiling as the earlier downstream study:

- `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

## Secondary hosted scale control

Use the official `Qwen/Qwen2.5-72B-Instruct` repository through one explicitly
recorded Hugging Face inference provider. This is a hosted serving control,
not an exact-checkpoint comparison: the provider may use an optimized serving
build and does not attest the repository revision loaded behind the endpoint.
Before study rows, require ten unrelated pilot prompts to complete without an
error and a repeated pilot prompt to return identical greedy output three
times. Preserve the public repository revision, provider name, request times,
raw responses, token usage, and the serving limitation. If the gate fails, do
not run study rows and report the failed feasibility check.

## Outcomes and inference

The evaluation unit is a source problem. For each model, report:

- accuracy for each form under the original policy;
- original, selective, and blanket accuracy across the 600 noncanonical rows;
- the selective-minus-original paired accuracy change overall, by source, and
  by input form;
- 10,000 percentile bootstrap intervals resampling base problems, never
  individual prompt rows;
- paired sign-flip randomization tests at the base-problem level, with Holm
  correction across the two source-specific tests for each model;
- exact McNemar counts as a descriptive row-level sensitivity analysis;
- three-form prediction disagreement, robust accuracy, and mean unique numeric
  predictions before and after each policy;
- parse failures and execution errors.

The primary confirmation criterion is fixed per model: the pooled 95 percent
base-bootstrap interval for selective minus original accuracy must exclude
zero on the positive side, and neither source-specific point estimate may be
negative. Blanket normalization is a prespecified comparison, not the target
policy. The hosted 72B result is reported separately from the three primary
checkpoints.

## Claim boundaries

- These are fixed direct-answer subsets, not full benchmark scores.
- FinQA uses annotated gold evidence, so it does not measure retrieval.
- TAT-QA uses the annotated relevant context and development split.
- Normalization is deterministic preprocessing. It does not show that a model
  learned numerical invariance.
- Accuracy gains apply only to the tested policy, formats, tasks, and prompts.
- The hosted 72B endpoint cannot establish an immutable-weight scale trend.
