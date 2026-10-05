# Reasoning-enabled cross-model operational amendment

Frozen after 35 Llama rows completed and before any accuracy, format-effect,
mitigation-effect, confidence-interval, or hypothesis-test result was computed.

The initial batch-1 Llama run was stopped because it averaged roughly 24 seconds
per row and would have required more than five hours for that model alone. The
Mac remained responsive, but the 15 GB checkpoint left limited unused memory.

The Llama run may resume at batch size 2. The 35 completed batch-1 rows remain
in the confirmatory output; every row records its actual batch size. Greedy
decoding is deterministic, and this change does not alter the model, revision,
dataset, prompts, 256-token cap, scorer, analysis, bootstrap count, or Holm
families. Cache clearing remains after every batch. Gemma remains restricted to
batch size 1.

Before this amendment, only row count, elapsed time, response length, parse
status, and system memory pressure were inspected. Correctness and aggregate
effects were not inspected.
