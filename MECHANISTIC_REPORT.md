# Mechanistic analysis report

## Scope and controls

All analyses use the exact commit-pinned, full-precision Llama 3.1 8B, Qwen3
4B Instruct 2507, and Gemma 2 9B checkpoints listed in `README.md`. The fixed
1,200-row activation set separates training (whole numbers 21–40), validation
(41–50), and test values (51–60).
Residual states were extracted at every layer at the numeral-final and answer
positions. Probe penalties and descriptive layers were selected on validation
data only. Test inference uses 1,000 label permutations and 10,000 grouped
bootstraps over numerical values.

## Cross-format probes

Value probes often transferred across canonical and padded forms, but not
uniformly. At the validation-selected numeral-final layer, canonical-to-padded
Spearman correlations were 0.763 for Llama, 0.098 for Qwen, and 0.404 for Gemma;
the Qwen interval crossed zero despite a nominal permutation p-value of 0.040.
Reverse-transfer correlations were 0.637, 0.443, and 0.598. At the answer
position, canonical-to-padded correlations were 0.644, 0.572, and 0.637, while
reverse transfer was 0.255, 0.830, and 0.596. Llama's answer-position reverse
interval crossed zero. These are exploratory full-curve results, not evidence
for a single privileged layer.

Equality probes were near-perfect at some answer-position layers in Qwen and
Gemma, but numeral-final reverse transfer was at chance for Qwen and weak,
permutation-nonsignificant for Llama and Gemma. This dependence on direction
and readout position argues against reducing the result to “the models encode
equality” without qualification.

The exact Llama greedy first-token behavior parsed on all 300 test prompts.
It labeled every canonical and nearby-unequal test row correctly, but every
padded-equivalent test row incorrectly. Consequently, some correct/incorrect
subgroup probe metrics are undefined because behavioral success is completely
confounded with form and class; the report preserves those cells as null rather
than inventing a comparison. The applicable value-probe subgroups and 10,000
bootstrap intervals are in `results/llama_probe_behavior_breakdown.json`.

## Controlled representation geometry

After nuisance regression fitted only on training wholes (token count,
absolute numeral position, prompt order, whole number, and fractional digit),
canonical–padded cosine similarity was compared with canonical–nearby-unequal
similarity on held-out values. The analysis was run separately for target-first
and target-second prompts.

Qwen and Gemma had positive equivalence-minus-nearby intervals in both prompt
orders at every layer, at both the numeral-final and answer positions. Llama did
not pass that criterion at the numeral-final position in both orders; it did
pass at the answer position from layers 9 and 15–31. This is evidence
that Qwen and Gemma's residual geometry tracks numerical equivalence beyond the
specified surface-form controls, while Llama's numeral-local geometry does not
show the same controlled, order-stable pattern. RSA and linear CKA curves are
reported as secondary descriptive statistics. Geometry alone remains
correlational.

## Matched causal residual interchange

Each model received the same 45 equal-decimal easy-order/hard-order pairs.
Every layer was swept with exact residual replacement at aligned numeral
tokens, the answer position, and an equal number of deterministic random
non-numeral positions. Digits 1–4 supplied 20 discovery cases for layer
selection; digits 5–9 supplied 25 untouched held-out cases.

| Model | Selected layer | Aligned effect | Random effect | Aligned − random | 95% bootstrap CI | Flip rate |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 8 | +0.405 | +0.465 | −0.060 | [−0.530, +0.365] | 20% |
| Qwen3 4B | 2 | +1.475 | −4.545 | +6.020 | [+3.440, +9.135] | 0% |
| Gemma 2 9B | 7 | +2.975 | +0.433 | +2.543 | [+1.710, +3.425] | 8% |

Llama fails the prespecified criterion: aligned numeral patches do not outperform
random-position patches on held-out cases. Qwen and Gemma pass the stated
criterion. The Qwen contrast requires a specific caveat: random patches reduce
the margin sharply, so the large contrast is not six units of direct rescue,
and no held-out case flips from incorrect to correct. Gemma supplies the
cleanest positive intervention: aligned patches improve the margin by 2.975 on
average, outperform random patches, and flip 8% of cases.

## Fixed-site causal generalization on new prompts and values

The selected Qwen layer 2 and Gemma layer 7 were carried forward without
reselection to 100 new cases, whole numbers 101–200, and two unseen prompt
templates. Aligned numeral replacement was again compared with an equal-size
deterministic random-position replacement, with 10,000 case bootstraps.

Qwen passed the prespecified criterion pooled and within both templates. Its pooled
easy-source aligned-minus-random effect was +3.756 (95% interval +2.416 to
+5.288); the relation-statement and direct-choice effects were +0.963 and
+6.550, both with intervals above zero. Gemma was positive pooled (+1.167,
+0.309 to +2.081) and on the relation template (+2.781, +1.650 to +4.011), but
not on direct choice (-0.448, -1.629 to +0.710). Gemma therefore fails the
strict cross-template generalization criterion.

An incompatible-value donor test selectively corrupted the correct equality
margin in both models and both templates. Pooled aligned-minus-random effects
were -43.349 (-44.853 to -41.620) for Qwen and -8.620 (-9.458 to -7.731) for
Gemma; correct-to-incorrect flip rates were 94% and 56%. These tests strengthen
the Qwen localization result and establish selective numerical corruption in
Gemma even though its easy-source rescue remains prompt-contingent.

## Token-level decomposition

This analysis was designed after the joint outcomes were inspected and is
therefore follow-up evidence. At the same fixed layers, padded-numeral and
short-numeral states were patched separately, each against an equal-size random
control. Qwen's easy-source contrasts were +1.123 for padded tokens and +3.084
for short tokens; Gemma's were +4.076 and +0.722. Incompatible-donor corruption
was concentrated in the short numeral for both models: -43.283 for Qwen and
-8.499 for Gemma, while padded-donor effects were approximately zero.

For Qwen, the pooled joint effect was statistically compatible with the sum of
the two component contrasts. For Gemma easy-source rescue, joint minus component
sum was -3.631 (-4.752 to -2.538). This is reported as descriptive
non-additivity, not a factorial interaction, because the necessary two-factor
intervention was not run.

## Broad-format robustness

A separate fixed 500-case dataset balanced negative decimals, leading zeros,
long fractions, scientific notation, and signed zero; equality class, larger
side, prompt template, and correct label position were controlled. Original-form
accuracy was 80.0% for Qwen, 59.4% for Gemma, and 48.8% for Llama. Paired
canonicalization increased accuracy by 12.4, 25.2, and 30.6 points, respectively,
with overall bootstrap intervals excluding zero.

No model passed the stronger prespecified criterion requiring no harmed family.
Qwen declined on long fractions and signed zero, Gemma declined on long
fractions, and Llama declined on signed zero. Canonicalization is therefore an
aggregate engineering improvement rather than a universally safe fix. The raw
subgroups also reveal strong prompt and label-position dependence, particularly
for Gemma and Llama.

Broad-format decoding was greedy throughout. Qwen and Llama used a 16-token
output ceiling; Gemma's ceiling was reduced from 16 to 4 during execution for
throughput. All saved Gemma responses terminate within the shorter budget, but
the ceiling itself was not recorded row by row. This is a disclosed procedural
limitation rather than an excluded result.

## Conclusion

The blocks converge on a comparative result, not a universal decimal circuit.
Qwen has the strongest generalization evidence: controlled geometry, the
original held-out patch, both new prompt templates, and selective donor
corruption all agree. Gemma has controlled geometry and selective corruption,
but its positive easy-source patch is prompt-contingent on untouched prompts.
Llama's numeral-final geometry is not stable across prompt order and its
aligned patches fail the random-position control. Late answer-position signals
in all models must not be mistaken for a numeral-specific mechanism. Full layer
curves, raw case rows, selection logic, permutations, and bootstrap outputs are
retained in `results/`.
