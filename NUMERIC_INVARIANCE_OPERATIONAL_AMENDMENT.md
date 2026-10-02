# Numerical invariance operational amendment

Recorded after source-file inspection and before any model output from this
study block.

## Trigger

The frozen protocol requested the next 100 eligible decimal-bearing GSM8K test
items after excluding the 100 previously used items. The exact source contains
only 113 eligible items in total, leaving 13 untouched items. The frozen
generator stopped with an error and produced no downstream dataset.

## Amendment

Preserve all 13 remaining GSM8K items and the planned next 100 FinQA items. Add
the first 100 eligible numeric-answer questions from the official TAT-QA
development split at revision
`870accc41953dcde885aabeb963d94aabdc0fbc3`. TAT-QA is used because its answers
are available in the development split and it tests numerical reasoning over
financial tables and related text.

For TAT-QA:

- include the full table when the annotated answer source contains `table`;
- include only the annotated relevant paragraphs when the answer source
  contains `text`;
- always include the question;
- require an answer that parses as one numeric value;
- preserve the annotated percentage scale;
- select eligible questions in official source order;
- apply the same four exact value-preserving rewrites to all included decimal
  literals.

The amended benchmark therefore contains 213 untouched base items and 852
prompts across three domains. The primary paired-test family expands from 18 to
27 tests: three noncanonical forms, three domains, and three primary models.
All other frozen methods, outcomes, mitigations, and claim boundaries remain
unchanged.

## Execution-resume correction

During the Gemma run, one generation attempt failed with a Metal out-of-memory
error and the process later stopped after 66 stored attempts. Before resuming,
the evaluator was corrected so a successful retry replaces the failed attempt
in the ordered result file while the failed attempt is retained in a separate
execution-error log. The retry uses the same model revision, prompt, chat
template, greedy decoding, 96-token ceiling, and parser. The MLX memory cap is
reduced from 22 GB to 18 GB. This changes memory scheduling only, not the model
or evaluation procedure.

The 16 GB retry proved too restrictive for the native Gemma weights and caused
a cascade of failed TAT-QA generations. Those attempts remain in the execution
error log and are excluded from analysis. The final retry returns to an 18 GB
cap and clears the MLX cache after every prompt rather than every 20 prompts.
This is an execution-stability change only. All prompts, weights, decoding,
token ceilings, parsing, and scoring remain frozen.

The final unresolved Gemma failures are confined to 44 prompts from 13 TAT-QA
base items. They are retried as isolated base-item batches in fresh processes,
so a memory fault cannot contaminate later rows. Batch isolation changes only
process lifetime and cache state. It does not change the evaluated rows or any
model, prompt, generation, parsing, or scoring setting.

Isolated retries recovered one additional prompt, leaving 43 failed prompts
across the same 13 TAT-QA base items. Raising the memory ceiling or quantizing
the key-value cache would alter the execution conditions and increase machine
risk. The final analysis therefore applies the frozen complete-base rule: all
13 affected TAT-QA bases are excluded for Gemma, leaving 87 Gemma TAT-QA bases
and all GSM8K and FinQA bases. Qwen and Llama retain all 213 bases. All failed
attempts are preserved and the unequal domain sample size is reported.

## Source integrity

- GSM8K source SHA-256:
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- FinQA source SHA-256:
  `831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc`
- TAT-QA source SHA-256:
  `8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5`
- TAT-QA license SHA-256:
  `46e27ffbc49c3fd44a9595c2213f8a5f4319a81b53238e49ce1eec39e7e25662`
