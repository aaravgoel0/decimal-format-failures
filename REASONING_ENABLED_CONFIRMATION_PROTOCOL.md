# Reasoning-enabled downstream confirmation

Frozen before inspecting any reasoning-enabled output.

## Question

Does exact-value format sensitivity persist when the strongest completed model
is allowed to reason before returning its final answer?

## Model and data

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Precision: official native unquantized checkpoint
- Data: all 200 base problems and 800 rows in
  `data/downstream_mitigation_confirmation.jsonl`
- Decoding: greedy, maximum 384 generated tokens

The data were frozen for the earlier mitigation study. They have been observed
under final-answer-only prompting, but no reasoning-enabled output has been
generated or inspected.

## Prompt and scoring

The system instruction asks the model to work step by step and end with
`Final answer: <number>`. Percentage questions additionally require the percent
sign. The scorer accepts only the final marked answer, removes commas and a
currency symbol, supports scientific notation, compares by exact decimal
value, requires the percent marker to match, and applies no rounding tolerance.

## Primary analysis

For FinQA and TAT-QA separately, compare padded, leading-zero, and scientific
forms with the canonical form. Report paired accuracy changes, 10,000
base-clustered bootstraps, exact McNemar tests, and Holm correction across the
six source-by-form contrasts. Also report four-form prediction disagreement,
robust accuracy, and pooled descriptive results.

## Mitigation analysis

As a prespecified secondary analysis, apply the already frozen selective policy
to the three noncanonical forms: preserve padding and map leading-zero and
scientific forms to canonical. Report its pooled and source-specific paired
accuracy changes with 10,000 base-clustered bootstraps. Also compare blanket
normalization with selective normalization as a secondary contrast.

## Safety and stopping

The run uses a 12 GB MLX memory limit and clears the cache every ten prompts.
Stop on repeated memory errors or system instability. Do not run Gemma or a
larger checkpoint locally for this confirmation.
