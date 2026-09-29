# Downstream invariance operational amendment

This amendment was fixed after execution errors were observed but before any
accuracy or prediction outcomes were inspected.

The frozen evaluator completed 1,198 of 1,200 generations. Both conditions for
one Gemma FinQA item (`finqa-036`) repeatedly failed with the same Metal
out-of-memory error under the frozen native-unquantized checkpoint and 96-token
generation ceiling. No model output was recovered for either condition.

The analysis therefore excludes that complete matched pair from Gemma's FinQA
and pooled summaries. Gemma has 99 FinQA pairs and 199 pooled pairs; all other
model-domain cells retain 100 pairs. The failure rows remain in the raw output.
No prompt, checkpoint, numerical value, successful row, or statistical
criterion was changed. The original frozen evaluator is preserved as
`scripts/evaluate_downstream_invariance_frozen.py`; the operational evaluator
only adds resume handling and an optional MLX allocation-limit argument.

