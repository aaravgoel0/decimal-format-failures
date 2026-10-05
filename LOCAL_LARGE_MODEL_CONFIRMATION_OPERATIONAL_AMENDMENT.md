# Local larger-model operational amendment

Recorded after the pinned checkpoint downloaded and loaded, but before any
model generation or outcome was produced.

The original evaluator checked `model.args.quantization` after loading. Gemma 3
stores the conversion metadata in the pinned repository's `config.json` under
`quantization` and `quantization_config`, so the guard returned `None` and
stopped before the one-row pilot. The amended evaluator reads the immutable
configuration file and requires `bits` to equal 4 before loading and
generation. No prompt, decoding, scoring, memory, model, revision, or analysis
setting changes.
