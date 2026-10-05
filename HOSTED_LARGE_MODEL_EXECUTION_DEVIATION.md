# Hosted larger-model execution deviation

The hosted `google/gemma-3-27b-it` scaling control did not complete and is
excluded from all outcome analyses.

- Planned rows: 800
- Successful request rows: 364
- Failed request rows: 436
- Failure class: HTTP 402 from Hugging Face Inference Providers
- Provider message: monthly included inference credits depleted
- Amount charged by this project: none
- Partial outcomes inspected for aggregate results: no
- Partial outcomes used in the manuscript: no

The runner preserved every attempt and refused to construct an analysis file.
The successful requests used approximately 104,615 input tokens and 21,094
output tokens. At the frozen routed price this represents about $0.0117 of the
account's included credit. No endpoint, subscription, payment method, or
paid-provider account was created.

The subsequent local Gemma 3 27B 4-bit study is a separately frozen experiment.
It does not merge, continue, or compare individual rows with this incomplete
hosted run.
