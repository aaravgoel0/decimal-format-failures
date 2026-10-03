# Downstream scoring correction

The original downstream evaluators parsed answers into decimal strings but compared
those strings textually. This incorrectly marked numerically equal forms such as
`55` and `55.0` as different. The raw prompts and model responses were not affected.

The corrected analysis applies one scoring rule to every saved downstream response:

- remove commas and currency symbols before extraction;
- extract the final numeric token, including scientific notation;
- compare the prediction and reference with exact decimal arithmetic;
- require the percent marker to match the reference;
- apply no rounding tolerance.

The corrected analyses recompute correctness from the preserved parsed prediction and
reference rather than trusting the stale derived `correct` field in the raw rows. The
raw row files remain unchanged to preserve the original execution record and hashes.
`results/downstream_scoring_audit.json` reports the before-and-after counts and a
deterministic set of examples covering decimal formatting, percentages, scientific
notation, punctuation, near misses, and parse failures.

All 36 deterministic spot-audit examples in that file were manually checked against
the stated rule. Numerically equivalent strings were accepted, percent-unit
mismatches and rounded near misses were rejected, scientific notation was interpreted
by value, and parse failures remained incorrect. No inconsistency was found.

The correction raises absolute direct-answer accuracy, lowers disagreement after
numerically equivalent output strings are collapsed, and changes the four-format
multiplicity result from seven to two significant contrasts. It does not change the
central qualitative result: exact-value rewrites still cause high prediction
disagreement, low four-form robust accuracy, and significant scientific-notation
damage in held-out domain-model cells. The fresh downstream mitigation still passes
for Qwen and not for Llama.
