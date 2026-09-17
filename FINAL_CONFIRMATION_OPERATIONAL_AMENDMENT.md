# Operational amendment for prompt confirmation

Frozen on 2026-09-13 after 80 Qwen rows had executed, but before any outcome
row, response, accuracy, or model comparison was inspected or summarized. The
partial file was removed from the active results and is not used.

The primary next-token constrained-label outcome remains evaluated on all
3,000 prompts per model exactly as registered. Free-text generation is a
secondary compliance measurement and was the dominant runtime cost. It is now
evaluated on a deterministic 20 percent paired subset. A base item is included
when `(digit + template_index + equality_label + zeros) mod 5 = 0`. This
selects 300 of 1,500 base items and both numeral orders, giving 600 generated
responses per model. It is exactly balanced across template, equality label,
and padding length, and selects two of ten digits within every such cell.

The subset rule depends only on the frozen dataset index and was specified
without inspecting model outcomes. All primary analyses, template strata,
tokenization fields, checkpoint restrictions, and stopping criteria are
unchanged.

An additional 134 Qwen rows then executed one at a time, again without outcome
inspection. That second partial file was also removed from the active results.
To reduce pure execution overhead, prompts of identical token length are now
evaluated in batches of up to eight. This changes neither input tokens nor logits,
decoding, scoring, cases, or analyses. The final active result files are
generated from a clean start with the batched evaluator.

A third partial run produced 96 Qwen rows before a balance audit detected that
the earlier index-based compliance subset co-varied with padding length. No
outcome was inspected. That partial file was removed, the balanced subset rule
above was frozen, and every active result is again generated from a clean
start.

Gemma's first batched attempt wrote four rows and then exhausted GPU memory
because the full batch-logit tensor remained allocated during free-text
generation. No outcome was inspected. That partial file was removed. The
evaluator now converts the three required label logits to CPU memory and
releases the full-vocabulary tensor before generating compliance responses.
This is a memory-lifetime correction only and does not change any numerical
operation used to define the primary prediction.

Gemma still exhausted available GPU memory after 23 rows at batch size four.
No outcome was inspected, and that partial file was removed. Gemma is therefore
run at batch size one, matching the stable execution pattern used for its prior
official-checkpoint evaluations. Batch size is an execution parameter only and
is recorded on every result row.

Gemma's next batch-size-one attempt was interrupted after two rows because
free-text generation still made the confirmation impractically slow. No outcome
was inspected, and that partial file was removed. The confirmation now omits new
free-text generation for every model. Its sole outcome is the registered argmax
over the exact next-token logits for the single-token labels 1, 2, and 3 on all
3,000 prompts per model. The existing paired behavioral experiment already
reports free-text compliance, while the purpose of this confirmation is prompt
and label robustness. This uniform change was made for all three models before
any confirmation accuracy or model comparison was inspected.

A complete 3,000-row Qwen file produced under the superseded generation-heavy
implementation was also removed from active results before any accuracy or
response was inspected. It is not analyzed. All three active files are produced
from clean starts with the final uniform evaluator.

The evaluator and causal-confirmation runner now compute only the three needed
output-token logits by selecting the corresponding rows of the tied embedding or
untied language-model head. This is algebraically identical to constructing the
full vocabulary logits and then selecting labels 1, 2, and 3. It removes an
otherwise unnecessary full-vocabulary allocation, with no change to inputs,
weights, layer operations, logits, predictions, margins, cases, or analyses.
Before any active confirmation run, the selected-output implementation was
compared with the original full-vocabulary implementation on the exact Qwen
revision. The three logits matched bit for bit in float32 conversion (maximum
absolute difference 0.0), and the argmax matched.

Before any causal-confirmation outcome was generated, aligned and random patch
mappings were grouped into shared downstream-forward batches. The baseline and
easy-source states were likewise grouped into one downstream batch. Every
registered mapping is still evaluated separately and every effect is stored.
This changes only execution grouping, not cases, states, patch locations,
weights, logits, margins, controls, seeds, or analyses.

After Qwen and Llama completed and after 19 Gemma cases were saved, the pending
Gemma cases were ordered by tokenized sequence length to reuse compiled Metal
kernels across equal-shape inputs. No causal outcome was inspected. The runner
sorts the completed output back to case order before analysis. Execution order
does not change prompts, activations, mappings, logits, controls, seeds, stored
effects, stopping rules, or inference.

After 105 Gemma cases were saved, memory diagnostics showed that macOS was
paging model weights because the custom forward runner had not requested MLX's
recommended wired-memory limit. The runner now calls `mx.set_wired_limit` with
the device's `max_recommended_working_set_size` before loading the model, as
MLX-LM's standard generation path does. This is a memory-residency setting. It
does not change model weights, tensor dtypes, operations, logits, cases,
patches, controls, seeds, stored effects, or analyses. No causal outcome was
inspected when this setting was added.

The wired-memory setting was then tested at batch sizes 64, 16, 1, and 4.
Those attempts produced no completed row: the larger batches raised Metal
out-of-memory errors, while the smaller attempts were interrupted before a row
completed. The setting was removed, so every retained causal-confirmation row
was produced by the same non-wired runner. The failed attempts do not enter any
analysis.
