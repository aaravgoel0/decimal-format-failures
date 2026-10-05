# Gemma memory-safety amendment

Frozen after the first Gemma attempt failed on row 1 with a Metal insufficient
memory error and before any Gemma output row or outcome existed.

Gemma will use MLX's serial `generate` path rather than `batch_generate` with a
one-element batch. Its free-buffer cache is disabled with
`mx.set_cache_limit(0)`, and the cache is still cleared after every prompt. The
batch size remains 1. The model, immutable revision, native precision, dataset,
prompt, 256-token cap, greedy decoding, scorer, statistical analysis, and stop
rules are unchanged.

This is the only retry after the initial memory error. If the serial path also
produces a memory error, local Gemma execution stops rather than repeatedly
pressuring the Mac.
