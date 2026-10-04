# Reasoning-enabled confirmation operational amendment

Frozen after the first 25 feasibility rows and before any aggregate accuracy or
paired result was computed.

The original serial 384-token runner was stopped because it averaged about 12
seconds per prompt and eight of 25 responses reached the token cap before the
required final-answer marker. The partial rows are preserved as
`results/reasoning_confirmation_feasibility_pilot.jsonl` and are excluded from
all analysis.

The confirmatory run makes two execution-only changes:

- request at most three short calculation lines and cap generation at 256
  tokens;
- use MLX batch generation with four prompts per batch.

The model, checkpoint revision, 800-row dataset, greedy decoding, exact decimal
scoring rule, percent-unit rule, primary comparisons, bootstrap count, Holm
family, and mitigation analyses remain unchanged. The run retains the 12 GB
memory cap and stops if batching causes a memory error.
