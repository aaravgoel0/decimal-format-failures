# Frozen Qwen component-level causal protocol

Frozen before generating or inspecting any outcomes from this dataset.

## Purpose

Determine whether Qwen's previously confirmed layer-2 joint numeral effect is
carried by the attention sublayer, the MLP sublayer, or a reproducible subset of
attention heads.

## Fixed model and layer

Use native unquantized `Qwen/Qwen3-4B-Instruct-2507` at immutable revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The zero-based intervention layer
is fixed at 2 from the earlier discovery and confirmation experiments. No layer
reselection is permitted.

## Fresh data

Use 120 new equivalent-decimal pairs with whole numbers 1001 through 1120 and
three prompt templates not used in earlier causal tests. Cases 0 through 29 are
the discovery split, with 10 cases per template. Cases 30 through 119 are the
held-out split, with 30 cases per template.

## Interventions

All interventions import states from the padded-second source into the
padded-first target at both aligned numeral spans.

- Whole-block output: replace the complete layer-2 output at aligned positions.
- Attention output: replace the pre-output-projection attention vector at
  aligned positions, then recompute the MLP from the intervened residual.
- MLP output: replace the layer-2 MLP update at aligned positions while keeping
  the target attention computation.
- Attention heads: replace one head's pre-output-projection vector at aligned
  positions, then apply the original output projection and recompute the MLP.

Every intervention is compared with equal-size non-numeral random-position
patches using the same source state. The final answer position is excluded.

## Head discovery and held-out test

Evaluate all 32 layer-2 attention heads on the 30 discovery cases using five
deterministic random-position sets per case. Rank heads by mean aligned effect
minus the case-level random mean. Carry the top four heads forward regardless
of sign. The ranking and selected head IDs are fixed before held-out execution.

On the 90 held-out cases, jointly patch the four selected heads. Compare that
patch, the full attention patch, the MLP patch, and the whole-block patch with
20 unique matched random-position sets per case. Store every random effect.

## Outcomes and criteria

The outcome is the correct equality-label logit margin. For each held-out
intervention, report the mean aligned effect minus random mean and a 10,000-case
bootstrap interval, the one-sided empirical randomization p value, behavior
flips, and template-specific intervals.

The selected-head result passes only if its pooled interval excludes zero
positively, the randomization p value is at most 0.05, and all three template
intervals exclude zero positively. Attention and MLP results are prespecified
secondary comparisons. A successful selected-head test localizes causal
influence but does not by itself establish a complete numerical algorithm.
