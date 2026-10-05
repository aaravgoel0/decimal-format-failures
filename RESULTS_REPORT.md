# Decimal comparison results report

## Scope

This report covers unequal integers, misleading decimal pairs, equal
zero-padded decimal pairs, prompt-robustness confirmations, held-out tests,
unquantized-checkpoint controls, reasoning-enabled downstream confirmations,
representation analyses, and causal interventions.

## Primary results

| Model | Integer | Misleading decimal | Equal zero-padding |
|---|---:|---:|---:|
| Llama 3.1 8B | 99.8% | 54.6% | 57.5% |
| Qwen3 4B Instruct 2507 | 100.0% | 99.6% | 99.6% |
| Gemma 2 9B | 100.0% | 95.7% | 93.05% |

These are the pinned official checkpoints. Integer and misleading-decimal
cells contain 1,000 valid responses. Equal zero-padding cells contain 2,000
held-out responses. Full confidence intervals and source filenames are in
`results/summary.json`.

## Numeral presentation and prompt effects

On the zero-padding task, accuracy when the padded representation appeared
first versus second was:

| Model | Padded first | Padded second |
|---|---:|---:|
| Llama 3.1 8B | 15.5% | 99.5% |
| Qwen3 4B Instruct 2507 | 99.2% | 100.0% |
| Gemma 2 9B | 86.1% | 100.0% |

Official unquantized Llama accuracy across three prompt variants was 57.5%,
3.5%, and 24.65%. These effects show that prompt wording and numeral
presentation order are major components of the observed failure. The correct
response remains option 3 when the two numerals swap, so this experiment does
not test answer-label position.

## Fully paired numeral-order and label-position control

A later frozen dataset contains 300 base items and 600 prompts per model. Each
exact item appears with the padded numeral first and second while prompt,
values, padding length, and answer-label position remain fixed. Equality is
balanced across labels 1, 2, and 3, and the two prompt templates were not used
in the earlier confirmation.

The preregistered primary required an exact one-number generation. Qwen had
61.33% strict accuracy with 232 non-exact responses. Gemma and Llama produced
explanatory text on all 600 rows, so both scored 0% under that strict rule. This
is an instruction-following result, not a useful comparison of their numerical
choices.

The prespecified evaluator also recorded answer-label logits. Their constrained
argmax gives the following secondary diagnostic:

| Model | Overall | Padded first | Padded second | Paired first-minus-second | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Qwen3 4B | 86.83% | 73.67% | 100.00% | -26.33 points | -31.33 to -21.33 |
| Gemma 2 9B | 65.00% | 30.00% | 100.00% | -70.00 points | -75.00 to -65.00 |
| Llama 3.1 8B | 21.00% | 23.00% | 19.00% | +4.00 points | +0.33 to +7.67 |

Qwen and Gemma's padded-first disadvantages appeared at all three equality-label
positions, so answer-label position does not explain the numeral-order effect.
Llama instead showed a large preference for label 3. Prompt dependence remained
large: Qwen was perfect in both orders on the first template, while its
padded-first accuracy fell to 47.33% on the second; Gemma's padded-first scores
were 47.33% and 12.67%. Five transient Gemma Metal out-of-memory attempts were
preserved in a separate error log and rerun under identical settings. The final
files contain 600 unique valid rows per model.

## Fresh ten-template prompt confirmation

A final frozen dataset contains 1,500 paired items and 3,000 prompts per model.
It crosses ten new prompt templates, all three equality-label positions, ten
fractional digits, five padding lengths, and both numeral orders. The outcome
is the argmax over the exact next-token logits for labels 1, 2, and 3.

| Model | Padded first | Padded second | Paired effect | 95% paired bootstrap CI | Templates with a negative interval |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 11.33% | 25.67% | -14.33 points | -16.40 to -12.33 | 7/10 |
| Qwen3 4B | 87.27% | 100.00% | -12.73 points | -14.47 to -11.07 | 7/10 |
| Gemma 2 9B | 60.33% | 100.00% | -39.67 points | -42.13 to -37.20 | 10/10 |

The pooled order effect is negative for every model and at every equality-label
position. Only Gemma passes the prespecified cross-template criterion. Qwen and
Llama each have seven template-level intervals below zero, so their pooled
effects should not be described as template-invariant.

## Initial held-out analysis

A fixed 2,000-row factorial dataset used unseen whole-number components
(21–99), ten fractional digits, one through five appended zeros, and perfectly
balanced presentation order.

| Model | Accuracy | 95% CI | Padded first | Padded second |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 45.25% | 43.08–47.44% | 2.5% | 88.0% |
| Qwen3 4B Instruct 2507 | 99.60% | 99.21–99.80% | 99.2% | 100.0% |
| Gemma 2 9B | 90.70% | 89.35–91.90% | 81.4% | 100.0% |

These were the initially analyzed quantized Llama and Gemma runs, alongside
official Qwen. They are retained as a prespecified sensitivity analysis, not
as the primary cross-model comparison. All prespecified overall thresholds and
all three directional numeral-order tests survived Holm correction. The
secondary digit-1-versus-digit-0 prediction also
appeared in every model. The registered ordinary logistic interaction was
invalid because perfect padded-second performance in Qwen and Gemma caused
complete separation; this failure is reported rather than interpreted.

Qwen's 2,000 held-out rows and its three 1,000-row exploratory or control
datasets were rerun from the pinned official revision to add checkpoint and
software provenance fields. Across all 5,000 rows, IDs, predictions, and raw
responses reproduced exactly.

The two prespecified held-out Llama prompt-robustness runs are also complete.
On the identical 2,000 examples, prompt variants 0, 1, and 2 scored 45.25%,
0.60%, and 12.15%, respectively, with 2,000 exact parses in every run. These
were robustness analyses, not members of the primary confirmatory family.

The same three prompts were subsequently run on the official unquantized
Llama checkpoint. Variants 0, 1, and 2 scored 57.50%, 3.50%, and 24.65%, with
2,000 exact parses per condition and no errors. Precision improved all three
conditions but did not remove the large prompt dependence.

## Quantization sensitivity

The primary held-out prompt was rerun on the official unquantized checkpoints,
pinned to immutable Hugging Face revisions, using the identical 2,000 rows.

| Model | Quantized | Official unquantized | Change | Unquantized padded first | Unquantized padded second |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 45.25% | 57.50% | +12.25 points | 15.5% | 99.5% |
| Gemma 2 9B | 90.70% | 93.05% | +2.35 points | 86.1% | 100.0% |

Paired exact McNemar tests found 245 versus 0 discordant Llama rows
(`p=3.5e-74`) and 50 versus 3 Gemma rows (`p=5.5e-12`). The unquantized checkpoints
therefore change the aggregate result, especially for Llama, while preserving
the central numeral-order instability.

Official unquantized Llama exploratory controls are also complete. Integer
accuracy remained 99.8%; misleading-decimal accuracy rose from 41.9% quantized
to 54.6% on the unquantized checkpoint (95% CI 51.5–57.7%).

Official unquantized Gemma controls scored 100.0% on integers and 95.7% on
misleading decimals (95% CI 94.3–96.8%), with 1,000 unique valid rows in each
final result file. Four transient Metal out-of-memory rows were removed and
successfully retried before analysis.

## Downstream value-preserving pairs

The downstream test transforms 100 decimal-bearing GSM8K problems and 100
decimal-bearing FinQA problems into matched canonical and zero-padded versions.
Only decimal surface form changes. Exact decimal scoring accepts equivalent
textual precision, requires the percent marker to match, and applies no
rounding tolerance. The primary family contains the six model-domain paired
accuracy contrasts and uses Holm correction.

| Model | Domain | Pairs | Canonical accuracy | Padded accuracy | Numeric disagreement |
|---|---|---:|---:|---:|---:|
| Llama 3.1 8B | GSM8K | 100 | 22.0% | 24.0% | 60.0% |
| Llama 3.1 8B | FinQA | 100 | 13.0% | 14.0% | 65.0% |
| Qwen3 4B | GSM8K | 100 | 32.0% | 22.0% | 61.0% |
| Qwen3 4B | FinQA | 100 | 9.0% | 9.0% | 52.0% |
| Gemma 2 9B | GSM8K | 100 | 8.0% | 9.0% | 55.0% |
| Gemma 2 9B | FinQA | 99 | 13.1% | 10.1% | 63.6% |

No primary accuracy contrast passed the multiplicity correction. The large
disagreement rates therefore establish prediction instability, not a uniform
accuracy penalty. One incomplete Gemma pair is excluded under the documented
complete-pair rule.

## Untouched four-format benchmark

On 213 further GSM8K, FinQA, and TAT-QA problems, pooled disagreement across
canonical, padded, leading-zero, and scientific forms is 80.3% for Llama,
69.5% for Qwen, and 76.5% for Gemma. Four-form robust accuracy is 12.2%, 20.2%,
and 17.0%, respectively. Two of 27 prespecified contrasts survive Holm
correction: Qwen on TAT-QA under scientific notation and Gemma on FinQA under
scientific notation. All other corrected intervals include zero.

## Fresh downstream mitigation confirmation

The frozen selective-normalization policy was carried into 100 further FinQA
test problems and 100 further TAT-QA development problems, all disjoint from
the earlier downstream sets. Each base has canonical, padded, leading-zero,
and scientific-notation forms. The policy preserves padded inputs and maps the
other two noncanonical forms to the byte-identical canonical prompt.

| Model | Original | Selective | Blanket | Selective change | 95% base-clustered CI | Frozen criterion |
|---|---:|---:|---:|---:|---:|---|
| Llama 3.1 8B | 24.67% | 25.50% | 26.00% | +0.83 points | -1.83 to +3.67 | Fail |
| Qwen3 4B | 28.50% | 33.00% | 33.50% | +4.50 points | +2.00 to +7.00 | Pass |

Qwen's point estimate is positive in both sources: +4.33 points in FinQA and
+4.67 in TAT-QA. Both source-specific sign-flip tests have Holm-adjusted
`p < 0.035`. Llama's interval crosses zero. Blanket normalization exceeds the
selective policy by only 0.5 points for either model; both paired intervals are
-0.83 to +1.83. The result supports realistic-task transfer for Qwen, not a
model-general mitigation claim.

The native-precision Gemma 2 9B execution is incomplete and excluded from this
mitigation analysis because its missingness is nonrandom. The preserved
attempts and deviation record are included in the supplement.

## Reasoning-enabled downstream confirmations

The same 200 FinQA and TAT-QA bases and 800 exact-value forms are evaluated
with at most three short calculation lines before a strictly marked final
answer. Qwen3 4B and Llama 3.1 8B use pinned official native-precision
checkpoints. The separately labeled Gemma 3 27B scaling control uses the pinned
local 4-bit MLX revision `feccbf793f8404211939458acfa9b857f22a9fe4`.
All 2,400 analyzed rows complete without an execution error.

| Model | Canonical | Padded | Leading zero | Scientific | Four-form disagreement | Four-form robust |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 38.5% | 37.0% | 32.0% | 31.5% | 69.5% | 19.5% |
| Qwen3 4B | 57.0% | 54.5% | 54.0% | 48.5% | 43.5% | 40.0% |
| Gemma 3 27B 4-bit | 50.0% | 50.5% | 48.5% | 49.0% | 58.0% | 32.0% |

| Model | Scientific minus canonical | 95% CI | Selective minus original | 95% CI |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | -7.00 points | -13.00 to -1.00 | +4.50 points | +1.17 to +7.83 |
| Qwen3 4B | -8.50 points | -13.50 to -3.50 | +3.83 points | +1.33 to +6.50 |
| Gemma 3 27B 4-bit | -1.00 points | -7.50 to +5.50 | +0.83 points | -2.50 to +4.17 |

No domain-by-form contrast survives a model's six-test Holm correction, and no
Llama or Gemma 3 contrast survives the secondary combined 12-test correction.
The pooled result therefore shows sensitivity under stronger prompting for
Llama and Qwen, but not a universal effect in the larger quantized control.
The frozen selective policy has positive pooled intervals for Llama and Qwen
and an inconclusive interval for Gemma 3. An incomplete hosted Gemma 3 attempt
is preserved for execution transparency but excluded from all outcome analysis.

## Further analyses

See `MECHANISTIC_REPORT.md` for the cross-format probes, representation
geometry, causal interventions, donor tests, token decomposition, final
many-random-site confirmation, Qwen component decomposition, and broad-format
robustness results.

## Interpretation boundary

The results establish a structured behavioral failure and a large
cross-model difference. They do not establish a particular internal numeric
representation or causal circuit. Attention averaging and ordinary logit-lens
plots are not treated as causal evidence.

## Artifacts

- `data/`: deterministic generated datasets.
- `results/`: raw responses and checked summaries.
- `figures/`: publication figures.
- `scripts/`: generation, evaluation, analysis, and figure code.
- `MECHANISTIC_REPORT.md`: controlled probes, geometry, and causal comparison.
