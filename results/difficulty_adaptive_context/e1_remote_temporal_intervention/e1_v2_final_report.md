# E1 Remote Temporal Information Intervention Study

- Status: `REMOTE_INFORMATION_CONDITIONAL`
- Protocol status: `FROZEN_BEFORE_E1_OUTCOME_ANALYSIS`
- Primary question: Does RF384 benefit depend on correctly related remote temporal information?
- C1 FULL reproduced: `true`

## Provenance

- `PROTOCOL_VERSION = E1-RTI-v2`
- `V1_INVALID_RETAINED = true`
- `SCIENTIFIC_PROTOCOL_CHANGED = false`
- `IMPLEMENTATION_CORRECTION = true`
- `INTERVENTION_EXECUTED_BEFORE_V2_FREEZE = false`
- `NEW_FINAL_OOD_TOUCHED = false`
- `TRAINING_PERFORMED = false`
- `NEXT_EXPERIMENT_AUTHORIZED = false`

E1-RTI-v1 was invalidated before intervention because the manual scoring path did not reproduce the frozen Full baseline within the preregistered tolerance. A diagnostic performed without executing interventions isolated the discrepancy to classifier chunking semantics. After explicit authorization, E1-RTI-v2 changed only the classifier scoring path to reproduce the frozen 2,000-frame chunked reference. The scientific hypotheses, interventions, endpoints, thresholds, gates, and evaluation population were unchanged.

## Primary results

- C3 REMOTE_PERMUTE delta log-loss `0.00023030` with source-cluster 95% CI `[-0.00018318, 0.00062663]`.
- C4 MATCHED_REPLACE delta log-loss `0.03213133` with source-cluster 95% CI `[0.02451276, 0.03988860]`.
- C2 REMOTE_ZERO destructive-control delta log-loss `0.02067777`.
- C5 LOCAL_PERMUTE sensitivity-control delta log-loss `-0.00113240`.

## Correction survival

- Original R prevalence: `0.01334563`.
- FULL survival: `1.00000000`.
- PERMUTE survival: `0.85124341`.
- MATCHED survival: `0.32629992`.

## A9/AE2 reconciliation

`T2`: The evidence points to a small, high-value correction subset, but source or realization uncertainty prevents the stronger aggregate claim.

## Audit and boundary

- The analysis is inference-only on the frozen A9/AE population.
- No training, threshold change, router change, or architecture change was performed.
- `NEW_FINAL_OOD_TOUCHED = false`.
- `TRAINING_PERFORMED = false`.
- `NEXT_EXPERIMENT_AUTHORIZED = false`.
- The experiment stops here; no E2, CAR, A-v3, A14 rerun, or background-reference experiment was started.

## Supported wording

The tested data are consistent with some dependence on correctly related remote temporal information, but the pre-registered conditions for a supported conclusion are not met.

## Alternatives retained

- intervention distribution shift;
- RF384 topology/processing effect;
- feature-statistics mismatch;
- boundary/annotation uncertainty;
- small high-value subset;
- source/domain heterogeneity.
