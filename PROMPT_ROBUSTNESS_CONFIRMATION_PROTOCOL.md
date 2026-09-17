# Frozen prompt-robustness confirmation protocol

Frozen on 2026-09-13 before generating or inspecting any outcomes from this
dataset.

## Purpose

Measure how strongly trailing-zero order sensitivity varies across prompt
wording while separating mathematical choice from response-format compliance.
This is a new confirmation set, disjoint in numerical values and prompt text
from every earlier behavioral dataset.

## Models

Only the three exact checkpoints already used in the paper are permitted:

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

Weights must be the official unquantized checkpoints. No replacement model or
smaller checkpoint is allowed.

## Dataset

The dataset is the complete crossing of:

- 10 frozen, semantically equivalent prompt templates;
- 3 equality-label positions;
- 10 fractional digits;
- 5 padding lengths; and
- 2 numeral orders.

This gives 1,500 paired base items and 3,000 prompts per model. Whole-number
components 1001 through 2500 are unique across base items and disjoint from all
earlier datasets. Within every pair, only numeral order changes.

## Outcomes

The primary outcome is the argmax among the next-token logits for the three
answer labels. This directly measures the forced mathematical choice and is
defined before model evaluation. Exact greedy response compliance is a
separate secondary outcome. The evaluator records raw text, parse status,
token IDs, numeral-token counts, prompt length, model revision, and chat
template mode on every row.

## Primary analysis

For each model:

1. Estimate the paired padded-first minus padded-second difference over all
   1,500 base items.
2. Use 10,000 base-item bootstraps for a two-sided 95 percent interval.
3. Report the order effect separately for each of the 10 prompt templates.
4. Report the median, minimum, and maximum prompt-specific order effect.
5. Treat generalization as established only if at least 8 of 10 prompt-specific
   intervals exclude zero in the same direction and the pooled interval also
   excludes zero.

## Secondary analysis

- Label-position, digit, and padding-length strata.
- Exact response compliance and parsed accuracy, kept separate from the forced
  choice outcome.
- Tokenization strata based on canonical token count, padded token count,
  count difference, and token-sequence overlap.
- A fixed-effect logistic model with prompt, label, padding length, digit, and
  numeral order, using base-item clustered standard errors.

All secondary and tokenization analyses are descriptive unless explicitly
identified as prespecified above.

