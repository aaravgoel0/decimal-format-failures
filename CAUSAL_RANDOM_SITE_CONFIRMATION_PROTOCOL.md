# Frozen many-random-site causal confirmation protocol

Frozen on 2026-09-13 before generating or inspecting any outcomes from this
dataset.

## Purpose

Test whether previously selected causal sites outperform an empirical
distribution of random non-numeral sites on fresh values and prompts. The same
experiment separately patches the canonical numeral, padded numeral, and both
numerals, providing a confirmatory component comparison.

## Models and fixed sites

No new layer selection is permitted.

- Llama 3.1 8B Instruct, exact revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, zero-based layer 8.
- Qwen3 4B Instruct 2507, exact revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, zero-based layer 2.
- Gemma 2 9B Instruct, exact revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`, zero-based layer 7.

Only official unquantized weights are allowed.

## Dataset

The dataset contains 150 fresh cases using unique whole-number components 701
through 850, fractional digits 0 through 9, padding lengths 1 through 5, and
three prompt templates not used by the original causal layer sweep. Each case
contains an easy padded-second source and a hard padded-first target. Source and
target have the same total sequence length. No outcomes from these cases may be
used to change a layer or prompt.

## Interventions and controls

At the fixed layer, replace target residual states with source states at:

1. the canonical numeral tokens only;
2. the padded numeral tokens only; and
3. both numeral spans jointly.

For each component and each case, generate 50 unique equal-size random-position
sets from positions that are non-numeral in both source and target and exclude
the final answer position. The seed is fixed by case and component. The same
source activation replacement is performed at each random set. Store every
random effect, not only its mean.

## Outcomes and inference

The outcome is the change in the correct equality-label logit margin relative
to the unpatched hard target.

For each model and component:

- compute the case-level aligned effect minus the mean of its 50 random-site
  effects;
- bootstrap 150 cases 10,000 times for a two-sided 95 percent interval;
- compare the mean aligned effect with 10,000 null means formed by drawing one
  random-site effect per case;
- report incorrect-to-correct flips for aligned and random interventions; and
- report the fraction of cases in which the aligned effect exceeds at least 95
  percent of its random-site distribution.

The joint patch is primary. It passes only if its aligned-minus-random interval
excludes zero positively and its randomization p value is at most 0.05. The
short-only and padded-only tests are secondary but prespecified. Their direct
difference receives a paired bootstrap interval. No circuit claim is allowed.

