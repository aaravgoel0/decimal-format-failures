# Paired generalization implementation amendment

Frozen on 2026-09-11 after 35 Llama rows had been saved, before any aggregate
result was calculated and before Qwen or Gemma inference began.

The protocol already states that a non-exact response counts as incorrect. The
first evaluator version inherited a parser that could recover an option number
from a longer response and would have credited that response. The evaluator was
stopped and corrected so `correct` is true only when `parse_status` is `exact`
and the prediction matches the answer.

Of the 35 saved rows, 27 had `parse_status` equal to `recovered` and eight were
invalid. The first version had credited two recovered responses. A deterministic
repair set `correct` to false for all 35 rows, as the frozen protocol requires.
The raw responses, parsed predictions, constrained-label predictions, and all
other stored fields remain unchanged. No hypothesis, dataset row, model,
prompt, or analysis rule changed.

After the completed Llama run showed that many generations included prose
instead of only an option number, the analysis output was expanded to report
the same paired order summaries for the already-recorded constrained-label
logit outcome. This does not replace the preregistered exact-response primary
outcome. The constrained paired summaries are secondary and were added after
seeing Llama's response-format behavior but before inspecting any Qwen or Gemma
outcomes.

After the first 600 Gemma attempts completed, the integrity validator found
four Metal out-of-memory execution errors. Before calculating any three-model
accuracy result, those failed attempt rows were copied unchanged to
`results/execution_errors_paired_order_generalization_gemma.jsonl`, removed
from the primary result file, and scheduled for exact rerun under the same
checkpoint and settings. Hardware failures are not model responses and are not
scored as incorrect. The replacement rows retain their original dataset IDs.

Three of the four first retries succeeded. The retry for `paired-124-o2` hit
the same Metal out-of-memory error a second time. That fifth failed execution
attempt was appended to the error log and removed from the primary result file
before a second exact retry. No model outcome for that row had been observed.
