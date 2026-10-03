# Downstream mitigation execution deviation

## Event

The frozen downstream mitigation confirmation completed all 800 rows for Qwen3
4B and Llama 3.1 8B. The Gemma 2 9B run preserved 752 attempt lines before a
macOS restart on 2026-10-02 at 21:35:29 America/Los_Angeles. The evaluator was
using the preregistered native-precision checkpoint and an 18 GB MLX memory
limit. MLX repeatedly warned that the model required approximately 17.6 GB,
close to its recommended 18.2 GB maximum for this host.

## Safety decision

The Gemma run will not be resumed on this 24 GB Mac. No local 30B or larger
model will be loaded. At the first post-restart check, system memory was 84
percent free, swap input and output were zero, 84 GB of disk space was free,
and no evaluator process remained.

## Preserved data

- Qwen: 800 of 800 rows, zero recorded execution errors.
- Llama: 800 of 800 rows, zero recorded execution errors.
- Gemma: 752 preserved attempt lines, including 169 explicit Metal
  out-of-memory errors. Interrupted restarts produced repeated attempts,
  leaving 637 unique prompt IDs.
- Fifty-seven prompt IDs have repeated attempts; five have different raw and
  parsed predictions across attempts despite temperature-zero decoding. The
  evaluator's frozen rule retains the first successful attempt.
- Applying the evaluator's frozen recovery rule leaves 583 successful unique
  rows and 54 error-only prompt IDs. Only 142 bases have all four successful
  forms: 98 FinQA and 44 TAT-QA. Seven further bases are partial. Missingness
  reflects memory failure and source-order truncation, not random sampling.

## Analysis rule

The two complete models retain the frozen analysis and confirmation criterion.
Gemma is not assigned a confirmatory pass or failure, and no accuracy estimate
from its incomplete run is used in the manuscript. The preserved attempts are
retained only to document the failure.
The incomplete run must not be silently treated as a complete three-model
confirmation.

## Hosted scale control

The official Qwen2.5 72B hosted pilot completed nine unrelated prompts, then
Hugging Face returned HTTP 402 because the account's included inference credits
were depleted. No study row was sent. The hosted gate failed, and the scale
control was not executed.
