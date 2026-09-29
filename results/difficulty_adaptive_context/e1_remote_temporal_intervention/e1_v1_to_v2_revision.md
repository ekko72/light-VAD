# E1-RTI-v1 to E1-RTI-v2 Revision

## ORIGINAL_STATUS

`INCONCLUSIVE_OR_INVALID`

E1-RTI-v1 was invalidated before any intervention was executed because the
mandatory full-population C1 reproduction check failed.

## ORIGINAL_RUNNER_SHA256

`A03A0939DF7D54C9195C05930A2BDC4AE76896BC8CF55F91A9567270003397F9`

The exact original runner is preserved as:

`results/difficulty_adaptive_context/e1_remote_temporal_intervention/e1_v1_archive/run_e1_v1.py`

## ORIGINAL_FAILURE

The frozen v1 C1 command completed twice with the same result:

- `passed=false`
- `max_abs_full_error=0.00019294023513793945`
- `max_abs_embedded_short_error=0.00016936659812927246`
- `tolerance=1e-06`

The invalid result is preserved at:

- `results/difficulty_adaptive_context/e1_remote_temporal_intervention/e1_c1_validation.json`
- `results/difficulty_adaptive_context/e1_remote_temporal_intervention/e1_final_summary.json`
- `results/difficulty_adaptive_context/e1_remote_temporal_intervention/e1_v1_archive/`

## ROOT_CAUSE_DIAGNOSTIC

The frozen reference evaluates the classifier on the same 2,000-frame scoring
chunks used by the streaming reference. The v1 E1 manual scoring path instead
evaluated the classifier on the full concatenated encoded sequence. The
encoded features and manual refinement arithmetic were otherwise consistent.

The diagnostic record is:

- `results/difficulty_adaptive_context/e1_remote_temporal_intervention/diagnostic_c1_root_cause.md`
- SHA-256: `EAC068D1AD662005AC3DE57F64185A5E31D2F3E2A25F724238BF1A134BFC3951`

## FULL_POPULATION_DIAGNOSTIC

- records = 1024
- frames = 553532
- corrected Full max abs error = 2.384185791015625e-07
- corrected embedded Short max abs error = 0.0
- tolerance = 1e-06

The full diagnostic record is:

- `results/difficulty_adaptive_context/e1_remote_temporal_intervention/diagnostic_c1_chunked_classifier_full.json`
- SHA-256: `77DD581AB9912091BE20F58A22F6DFBF93BC8A330A0A1366E1DBBE5A49AE75F0`

## AUTHORIZED_CHANGE

Create `E1-RTI-v2` with one implementation correction:

`_short_logits_for_encoded` must evaluate
`model.short_model.classifier` on the same configured 2,000-frame scoring
chunks used by the frozen reference path, concatenate the chunk logits in
time order, and pass the result through the existing C1 and intervention
scoring implementation unchanged.

## NON_SCIENTIFIC_EXECUTION_FIX

The first formal v2 run after the C1 pass stopped before intervention on a
pre-existing call-site typo: `_run_formal_analysis` called
`_write_rf_dependency_audit`, while the existing function is named
`write_rf_dependency_audit`. The call site was corrected to invoke the
existing function. No arguments, artifact content, intervention behavior,
endpoint, threshold, gate, or scientific definition changed. C1 is rerun and
the v2 protocol is refrozen after this correction so the frozen runner hash
covers the executable implementation.

## UNCHANGED_PROTOCOL_COMPONENTS

The checkpoint, dataset, evaluation population, feature pipeline, RF384
dependency definition, LOCAL/REMOTE definitions, intervention definitions,
intervention seeds, matching rules, loss definitions, proper-scoring metrics,
bootstrap procedure and sample count, taxonomy definitions, stratification
bins, primary endpoints, decision gates, reproduction tolerance, decision
threshold, router, model architecture, and all training behavior remain
unchanged.

CONFIRMATION:
NO_INTERVENTION_EXECUTED_BEFORE_REFREEZE=true

NEW_FINAL_OOD_TOUCHED=false

RESULT_DRIVEN_SCIENTIFIC_CHANGE=false
