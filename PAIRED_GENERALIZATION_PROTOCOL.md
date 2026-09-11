# Frozen paired order and label-position generalization protocol

Frozen on 2026-09-11 before model inference and before inspecting any outcome
from this dataset.

## Purpose

The earlier held-out zero-padding dataset balanced numeral presentation order,
but it did not present every exact numerical item in both orders, and equality
always occupied label position 3. This experiment independently crosses numeral
order, equality-label position, prompt wording, fractional digit, and padding
length while pairing the two numeral orders within every base item.

## Models

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

The official checkpoint weights are loaded without weight quantization. No
substitute release, size, or model family is permitted.

## Dataset

The frozen dataset contains 300 base items and 600 prompts. Each base item is
shown once with the padded numeral first and once with the padded numeral
second. The two rows in a pair are identical in every other respect.

The 300 base items are the complete crossing of:

- 2 prompt templates not used in the original confirmation;
- 3 positions for the correct equality label;
- 10 fractional digits, 0 through 9;
- 5 padding lengths, 1 through 5 trailing zeros.

Whole-number components 201 through 500 are unique across base items and are
disjoint from all earlier behavioral, probe, and causal datasets. The row order
is deterministically shuffled only after the complete factorial dataset is
constructed.

## Decoding and recorded outcomes

The primary outcome is exact greedy response accuracy. Invalid or non-exact
responses count as incorrect and are reported separately. The evaluator also
records the three answer-label logits and their constrained argmax as a
secondary diagnostic. Each row records the exact model revision, chat-template
mode, software versions, token IDs, sequence length, and decoding settings.

## Primary analyses

For each model:

1. Report overall accuracy and accuracy by numeral order.
2. Estimate the paired padded-first minus padded-second accuracy difference.
3. Bootstrap the 300 base-item pairs 10,000 times for a two-sided 95% interval.
4. Run an exact two-sided McNemar test on discordant paired outcomes.
5. Report whether the order-effect interval excludes zero.

Across models, report paired differences in the order effect with 10,000
base-item bootstraps. No fixed accuracy threshold is treated as a primary test.

## Secondary analyses

- Accuracy and order effects within each equality-label position.
- Accuracy and order effects within each prompt template.
- Fractional-digit and zero-count strata.
- Exact-response versus constrained-label agreement.
- A three-model comparison of answer-label-position sensitivity.

These secondary results are descriptive. No layer, subgroup, prompt, or label
position may be selected after outcome inspection and presented as primary.

## Claim boundary

This experiment can establish whether numeral-order sensitivity generalizes
when item identity and answer-label position are controlled. It cannot identify
an internal numerical algorithm or causal circuit.
