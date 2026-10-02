# Frozen numerical invariance and mitigation protocol

Frozen before generating the new datasets or inspecting any model output from
this study block.

## Motivation

The prior experiments establish a trailing-zero failure, broad synthetic
format sensitivity, and downstream instability. This block tests the broader
claim that value-preserving numerical rewrites can change answers on realistic
reasoning tasks, and evaluates two practical mitigation strategies.

## Study 1: untouched downstream rewrite benchmark

### Sources and sampling

- GSM8K official test split at commit
  `3101c7d5072418e28b9008a6636bde82a006892c`.
- FinQA official test split at commit
  `0f16e2867befa6840783e58be38c9efb9229d742`.
- Skip the first 100 eligible decimal-bearing items from each source because
  they were used in the prior paired downstream experiment.
- Select the next 100 eligible items from each source in source order.
- FinQA uses only the annotated gold evidence and question. This measures
  reasoning with supplied evidence, not retrieval.

### Value-preserving forms

Every base item produces four prompts. All decimal literals in the evaluated
text are rewritten together.

1. `canonical`: plain decimal notation with redundant trailing zeros removed.
2. `padded`: the canonical fraction followed by two, three, or four zeros,
   determined only by source index.
3. `leading_zero`: two redundant zeros before the integer part of every plain
   decimal literal.
4. `scientific`: normalized base-10 scientific notation with the exact same
   Decimal value.

Words, numerical values, correct answers, source order, and model instructions
remain fixed. The generator must verify exact Decimal equality for every
transformed literal.

### Primary models

Use the same exact native-precision checkpoints as the prior study:

- `meta-llama/Meta-Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- `google/gemma-2-9b-it`, revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`.

Use official chat templates, greedy decoding, and a 96-token ceiling. Preserve
raw output and parsing status.

### Secondary scale control

Attempt `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` only as a labeled
quantized scale sensitivity analysis. Before any study-row execution, require
successful loading and deterministic output on ten unrelated pilot prompts.
Record the immutable repository revision and quantization metadata. Failure of
the feasibility gate is reported and does not alter the primary design.

### Outcomes

For each model and domain, report:

- accuracy by form;
- canonical-versus-form paired accuracy changes with 10,000 item bootstraps;
- exact two-sided McNemar tests with Holm correction across the three
  noncanonical forms, two domains, and all primary models;
- four-form numeric prediction disagreement rate;
- robust accuracy, defined as the fraction of base items answered correctly in
  all four forms;
- the mean number of unique normalized numeric predictions per base item;
- parse failures by form.

The primary invariance claim passes for a model-domain cell when its four-form
prediction-disagreement interval excludes zero. Accuracy-change claims require
the corresponding Holm-adjusted paired test to pass.

## Study 2: mitigation

Two mitigation baselines are fixed before outcomes:

1. `canonical input`: apply the verified parser and send the canonical prompt
   regardless of the observed input form. This guarantees identical model
   input for all equivalent variants. Report its accuracy as canonical-form
   accuracy and do not present zero disagreement as a learned capability.
2. `four-form plurality`: choose the most frequent normalized numeric answer
   across the four raw forms. Ties are resolved by the canonical-form answer.
   Compare plurality accuracy with canonical-form accuracy using paired
   bootstrap intervals and exact McNemar tests.

### Fresh selective-normalization confirmation

Generate 600 new synthetic comparisons, 120 per previously defined family,
with whole numbers beginning at 1301 and three prompt templates not used in the
prior broad-format dataset. The previously observed 500-case dataset is the
discovery set. Its result motivates one frozen, model-independent policy:

- normalize leading-zero and scientific-notation inputs;
- preserve negative-decimal, long-fraction, and signed-zero inputs.

Evaluate original, full-canonical, and selective-normalized prompts. The
selective policy passes for a model when its pooled paired accuracy interval is
positive and neither changed family has a negative point estimate. Unchanged
families are checked for byte-identical prompts. Report the full-canonical
baseline even when it fails.

## Claim boundaries

- The downstream subset is not a standard full-benchmark score.
- A deterministic canonicalizer enforces input invariance by construction; it
  does not prove that the model learned invariance.
- A quantized 30B run is a scale sensitivity analysis, not a native-precision
  comparison and not a frontier-model claim.
- The selective policy was derived after inspecting the earlier broad-format
  outcomes, so only its new 600-case confirmation is held out.
