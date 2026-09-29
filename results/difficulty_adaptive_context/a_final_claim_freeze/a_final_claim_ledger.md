# A Final Claim Ledger

Ledger ID: `A-FINAL-CLAIM-LEDGER-v1`

Freeze date: `2026-09-22`

Freeze scope: close A mechanism exploration, preserve all official states, and
freeze the A-v2 method and final-OOD protocol before any `NEW_FINAL_OOD` access.

## 1. Official Claims

| Claim | Frozen status | Evidence boundary |
| --- | --- | --- |
| A-v1 | `CONDITIONAL_GO` | Frozen phase-A2 confirmatory evidence. The historical internal `formal_status=FORMAL_GO` is retained as historical metadata only, but the user-mandated official A-v1 status is `CONDITIONAL_GO`. |
| A-v2 | `CONDITIONAL_GO` | Five positive M2 seed deltas, positive paired point estimate, but G1 fails because the source-cluster 95% CI lower bound is below zero. |
| CAR | `STRONG_CROSS_ARCH_REPLICATION` | Passed all five frozen CAR gates. |
| E1 | `REMOTE_INFORMATION_CONDITIONAL` | E1-RTI-v2. Remote identity/compatibility matters, but exact temporal ordering is not established. |
| E2 | `INCONCLUSIVE_OR_INVALID` | Required matching-validity gate failed. Exploratory transition/onset observations remain exploratory and are not upgraded. |
| E3 | `ROBUST_VALUE_STRUCTURE` | Decision sparsity and event structure survive the frozen robustness audit, while temporal-value sparsity is not supported in the broad form. |
| E4 | `INCONCLUSIVE_OR_INVALID` | Official result remains unchanged. The post-hoc `STABLE_REFINABLE_CORE` finding is diagnostic-only. |

## 2. E4 Diagnostic Boundary

E4 post-hoc repair/decomposition evidence may be cited only with the label:

`DIAGNOSTIC_ONLY_NOT_AN_OFFICIAL_UPGRADE`

It cannot:

1. Replace `E4=INCONCLUSIVE_OR_INVALID`.
2. Establish a new confirmatory mechanism claim.
3. Change the A-v1 or A-v2 state.
4. Change the final method, endpoint, gates, or OOD protocol.
5. Authorize a new search or experiment.

## 3. A-v2 Final Method

The final candidate method is A-v2 `M2` at budget `0.05`.

| Item | Frozen value |
| --- | --- |
| Seeds | `17, 23, 41, 59, 71` |
| Checkpoints | Five frozen M2 `best` state payloads |
| Input | `X3`, dimension `330` |
| Causal window | `25` frames |
| Normalization | TRAIN-only mean/variance, `ddof=0` |
| Split seed | `20260920` |
| Selection | Top `ceil(0.05 * DEV frames)` on DEV, threshold applied once to evaluation |
| Comparator | Frozen uncertainty `M0`, score `-abs(short_score - 0.5)` |
| Primary endpoint | `U(B)=mean(selected * v)` at `B=0.05` |
| Primary estimate | `Delta_U_primary=0.0005167364201845739` |
| Source-cluster 95% CI | `[-0.00017398191060556133, 0.001320609542074035]` |
| Seed-positive count | `5/5` |
| Seen delta | `-0.00012815246857800049` |
| Unseen delta | `0.002440321621875295` |
| Primary activation | `0.05731` |
| Maximum activation | `0.10861728766468762` |

Interpretation: positive point estimate with incomplete gate evidence. The
official A-v2 state therefore remains `CONDITIONAL_GO`.

## 4. Preserved Mechanism Claims

| Claim | Status | Required wording |
| --- | --- | --- |
| Decision-changing refinability is sparse | Supported | Sparse decision-changing refinability, not "all temporal value is sparse." |
| Probabilistic value is broader than R | Supported | Reported as proper-scoring value; decision-unchanged positive frames are not relabeled as R or "refinable." |
| Hard/not-refinable value structure | Supported | Hard cases remain distinct from decision-changing R frames. |
| Ground-truth onset structure | Supported | Report separately from model-posterior transition structure. |
| Posterior-transition structure | Supported | Model-posterior evidence, not a substitute for ground-truth onset evidence. |
| Boundary annotation artifact explains the effect | Not supported | Narrow boundary exclusion does not explain away the structure. |
| Exact temporal ordering is established | Not supported | E1 remains conditional and E2 remains invalid. |
| Stable-refinable core is confirmatory | Not supported | E4 result is diagnostic only. |

## 5. Freeze Actions

The following are frozen:

1. Official claim states.
2. Final method, five checkpoints, router score, threshold rule, and budget.
3. Data splits, preprocessing, and feature definition.
4. Primary metric, comparator, aggregate test, bootstrap method, and source
   clusters.
5. Seen/unseen and other allowed subgroup definitions.
6. GO, CONDITIONAL, NO-GO, and invalidity rules.
7. Missing-data and execution-failure handling.
8. Evaluation code identities and SHA256 values.

## 6. Stop Boundary

`STOP_MECHANISM_EXPLORATION=true`.

Forbidden after this ledger:

1. E5.
2. Router search.
3. RF search.
4. New backbone.
5. New feature.
6. Retraining, fine-tuning, checkpoint replacement, or seed selection.
7. Threshold or gate tuning.
8. Any change based on OOD outcomes.

`NEXT_EXPERIMENT_AUTHORIZED=false`.

`NEW_FINAL_OOD_TOUCHED=false`.

The final OOD protocol is frozen but unexecuted. The dataset/manifest remains:

`UNBOUND_AT_FREEZE_NOT_ACCESSED`.

## 7. Post-Execution Experimental Closure

The A experimental program is closed:

`A_EXPERIMENTATION_COMPLETE=true`

`NEXT_EXPERIMENT_AUTHORIZED=false`

The authorized final-OOD execution could not proceed because the bound
manifest declared:

`FINAL_OOD_STATUS=NOT_EXECUTABLE_NO_UNUSED_CONFIRMATORY_DATASET`

The manifest contained `candidate_dataset=null` and stated that the available
LibriSpeech test speakers had already been used for A-v1 evaluation. No
confirmatory dataset was loaded and no final-OOD outcome was computed:

`FINAL_OOD_OUTCOME_EXPOSED=false`

`SCIENTIFIC_BLINDNESS_PRESERVED=true`

The final method state is unchanged:

`A_v2_STATUS=CONDITIONAL_GO`

`M2_FINAL_OOD_SUPERIORITY=UNRESOLVED`

No pre-reserved, unused confirmatory final-OOD dataset exists. The final
untouched-OOD superiority claim was therefore not tested. This result must
not be interpreted as support for or refutation of M2.

The following remain forbidden:

1. Re-splitting or selecting development/OOD data as a new `NEW_FINAL_OOD`.
2. Creating, filtering, or tuning a dataset for final confirmation.
3. Rerunning A14 or the A-v2 OOD evaluation.
4. Continuing M2, router, threshold, backbone, or RF search.
5. Starting E5.
