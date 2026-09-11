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
