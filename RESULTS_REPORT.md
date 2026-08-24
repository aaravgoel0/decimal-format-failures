# Decimal comparison results report

## Scope

This report covers unequal integers, misleading decimal pairs, equal
zero-padded decimal pairs, two additional prompt variants, held-out tests,
full-precision controls, representation analyses, and causal interventions.

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

Official full-precision Llama accuracy across three prompt variants was 57.5%,
3.5%, and 24.65%. These effects show that prompt wording and numeral
presentation order are major components of the observed failure. The correct
response remains option 3 when the two numerals swap, so this experiment does
not test answer-label position.

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

The two prespecified held-out Llama prompt-robustness runs are also complete.
On the identical 2,000 examples, prompt variants 0, 1, and 2 scored 45.25%,
0.60%, and 12.15%, respectively, with 2,000 exact parses in every run. These
were robustness analyses, not members of the primary confirmatory family.

The same three prompts were subsequently run on the official full-precision
Llama checkpoint. Variants 0, 1, and 2 scored 57.50%, 3.50%, and 24.65%, with
2,000 exact parses per condition and no errors. Precision improved all three
conditions but did not remove the large prompt dependence.

## Quantization sensitivity

The primary held-out prompt was rerun on the official unquantized checkpoints,
pinned to immutable Hugging Face revisions, using the identical 2,000 rows.

| Model | Quantized | Full precision | Change | Full padded first | Full padded second |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 45.25% | 57.50% | +12.25 points | 15.5% | 99.5% |
| Gemma 2 9B | 90.70% | 93.05% | +2.35 points | 86.1% | 100.0% |

Paired exact McNemar tests found 245 versus 0 discordant Llama rows
(`p=3.5e-74`) and 50 versus 3 Gemma rows (`p=5.5e-12`). Full precision
therefore changes the aggregate result, especially for Llama, while preserving
the central numeral-order instability.

Official full-precision Llama exploratory controls are also complete. Integer
accuracy remained 99.8%; misleading-decimal accuracy rose from 41.9% quantized
to 54.6% full precision (95% CI 51.5–57.7%).

Official full-precision Gemma controls scored 100.0% on integers and 95.7% on
misleading decimals (95% CI 94.3–96.8%), with 1,000 unique valid rows in each
final result file. Four transient Metal out-of-memory rows were removed and
successfully retried before analysis.

## Further analyses

See `MECHANISTIC_REPORT.md` for the cross-format probes, representation
geometry, causal interventions, donor tests, token decomposition, and
broad-format robustness results.

## Interpretation boundary

The results establish a structured behavioral failure and a large
cross-model difference. It does not establish a particular internal numeric
representation or causal circuit. Attention averaging and ordinary logit-lens
plots are not treated as causal evidence.

## Artifacts

- `data/`: deterministic generated datasets.
- `results/`: raw responses and checked summaries.
- `figures/`: publication figures.
- `scripts/`: generation, evaluation, analysis, and figure code.
- `MECHANISTIC_REPORT.md`: controlled probes, geometry, and causal comparison.
