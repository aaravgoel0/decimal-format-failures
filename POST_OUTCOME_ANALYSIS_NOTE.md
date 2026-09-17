# Post-outcome analysis note

The frozen prompt-confirmation analysis completed on 2026-09-14. Its primary
paired bootstrap and exact McNemar analyses ran as specified. The auxiliary
fully adjusted logistic regressions for Qwen and Gemma encountered complete
separation because padded-second accuracy was 100 percent. Statsmodels returned
finite iteration-dependent coefficients but non-finite standard errors,
intervals, and p values.

No alternative inferential model was substituted. The finalizer replaces each
non-finite auxiliary fit with an explicit `not_estimable_complete_separation`
record. It leaves every primary estimate, bootstrap interval, exact test,
template result, stratum, and the finite Llama logistic fit unchanged. This
correction was made after outcomes were available and is not part of the frozen
confirmatory plan.
