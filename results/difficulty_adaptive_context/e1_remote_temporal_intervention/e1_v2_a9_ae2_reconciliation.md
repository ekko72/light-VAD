# A9/AE2 reconciliation

AE2 reported a stable sufficient span of approximately `69.22` frames, while A9 observed a large RF384 advantage over RF64. E1 tested whether the RF384 advantage is explained by correctly related remote temporal information.

Reconciliation classification: `T2`.

## Evidence

- C3 REMOTE_PERMUTE mean delta log-loss: `0.00023030` (source-cluster CI `[-0.00018318, 0.00062663]`).
- C4 MATCHED_REPLACE mean delta log-loss: `0.03213133` (source-cluster CI `[0.02451276, 0.03988860]`).
- Original-R correction loss under C3: `0.14875659`.
- Original-R correction loss under C4: `0.67370008`.

## Interpretation

The evidence points to a small, high-value correction subset, but source or realization uncertainty prevents the stronger aggregate claim.

The T1/T2/T3/T4 labels are interpretations of this frozen E1 analysis. They do not turn a bounded intervention effect into a claim that 384 contiguous frames are necessary or that long-term semantic context has been identified.

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
