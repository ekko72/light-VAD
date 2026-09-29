# A Final OOD Protocol

Protocol ID: `A-FINAL-OOD-v1`

Freeze date: `2026-09-22`

Status at freeze: `FROZEN_BEFORE_NEW_FINAL_OOD_ACCESS`

This protocol is prewritten before the `NEW_FINAL_OOD` dataset or manifest is
bound, read, hashed, previewed, summarized, or executed. The dataset state at
freeze is:

`UNBOUND_AT_FREEZE_NOT_ACCESSED`

No result from this protocol exists at freeze time.

## 1. Frozen Scientific State

The following official states are carried forward without modification:

| Study | Official status |
| --- | --- |
| A-v1 | `CONDITIONAL_GO` |
| A-v2 | `CONDITIONAL_GO` |
| CAR | `STRONG_CROSS_ARCH_REPLICATION` |
| E1 | `REMOTE_INFORMATION_CONDITIONAL` |
| E2 | `INCONCLUSIVE_OR_INVALID` |
| E3 | `ROBUST_VALUE_STRUCTURE` |
| E4 | `INCONCLUSIVE_OR_INVALID` |

E4 post-hoc decomposition evidence is diagnostic only. It cannot upgrade the
official E4 state, substitute for a confirmatory result, or create a new
mechanism claim.

`STOP_MECHANISM_EXPLORATION=true`.

The following are forbidden: E5, router search, RF search, a new backbone, new
features, retraining, checkpoint selection, threshold tuning, and any change to
this protocol after OOD outcome inspection.

## 2. Final Candidate Method

Final method: A-v2 `M2`, refinement budget `B=0.05`.

The fixed seed set is `17, 23, 41, 59, 71`. All five frozen M2 checkpoints are
required. Every seed uses the same frozen A-v2 recipe and the same primary
selection rule:

1. Score every frame with `sigmoid(primary R head)`.
2. Calibrate on the frozen DEV split by selecting the top
   `ceil(0.05 * DEV frames)` scores.
3. Freeze the resulting score threshold.
4. Apply that threshold once to the final OOD population without
   recalibration.

Comparator: frozen uncertainty routing `M0`.

M0 score: `-abs(short_score - 0.5)`.

M2 and M0 operate under the same frame population, budget, labels, and source
clusters. `AlwaysRefine` is a compute reference only and is not a scientific
comparator.

The final method uses:

| Component | Frozen value |
| --- | --- |
| Input | A-v2 `X3`, dimension `330` |
| Causal window | `25` frames |
| Normalization | TRAIN-only mean/variance, `ddof=0` |
| Decision threshold | `0.5` |
| Primary budget | `0.05` |
| Budgets audited for activation | `0.01, 0.02, 0.05, 0.10` |
| Split seed | `20260920` |
| Split | TRAIN `24` speakers / `313,993` frames, DEV `8` / `126,522`, INTERNAL_TEST `8` / `113,017` |
| Training seeds | `17, 23, 41, 59, 71` |

No checkpoint may be retrained, replaced, fine-tuned, or selected again.

## 3. Primary Endpoint

For each source cluster `c` and training seed `s`, define the per-frame signed
value:

`v = 1[short prediction wrong] - 1[RF384 prediction wrong]`

For a routing method `m` under budget `B`, define:

`U(m, s, B) = mean(selected_m(B, s) * v)`

The primary endpoint is the paired source-cluster utility difference:

`Delta_U_primary = mean_s(U(M2, s, 0.05) - U(M0, s, 0.05))`

The five seed-level paired differences are averaged before source-cluster
bootstrap inference. Seeds are not treated as independent bootstrap samples.

Positive `Delta_U_primary` means that M2 improves the selected-frame utility
over frozen uncertainty routing at the same budget.

## 4. Aggregate Test

Primary inference is a paired source-cluster percentile bootstrap:

| Item | Frozen rule |
| --- | --- |
| Cluster | Frozen OOD `source_key` |
| M2 vs M0 pairing | Same resampled source clusters |
| Seed aggregation | Average the five per-seed paired differences first |
| Resamples | `2000` |
| Seed | `20260921` |
| Interval | Two-sided `95%` percentile CI |
| Return | Point estimate, lower bound, upper bound |

Speaker-cluster bootstrap may be reported only as sensitivity. It cannot
replace the source-cluster primary analysis.

## 5. Allowed Subgroups

Only previously frozen subgroup definitions may be reported:

| Dimension | Frozen levels or rule |
| --- | --- |
| Seen/unseen | `seen`, `unseen` |
| SNR | `-5, 0, 5, 10, 15, 20` |
| Noise domain | `clean`, `noise` |
| Condition | Existing frozen CAR/E2 condition labels only |
| Onset distance | `0`, `1-2`, `3-5`, `6-10`, `11-25`, `26+` |
| Offset distance | Same frozen bins as onset distance |
| Posterior-transition distance | Same frozen bins as onset distance |

No subgroup may be created, split, merged, or selected after OOD outcome
inspection. A cell with fewer than `1000` OOD frames is descriptive only and is
not eligible for the G3 harm gate.

## 6. Frozen Gates

G1, primary utility:

`Delta_U_primary > 0` and source-cluster `95%` CI lower bound `> 0`.

G2, seed consistency:

At least `4/5` per-seed paired deltas are strictly positive.

G3, domain and subgroup safety:

Unseen aggregate delta is at least `0`, and no eligible subgroup cell has
delta utility at or below `-0.005`.

G4, activation control:

Primary-budget OOD activation is in `[0.03, 0.075]`, and activation for every
frozen budget `0.01, 0.02, 0.05, 0.10` is at most `0.15`.

G5, compute control:

The frozen latency estimate must be at least `0.20 ms/frame` below
`AlwaysRefine` and at least `10%` below `AlwaysRefine`. This gate uses the
frozen A-v2/A11 engineering estimate; it is not a new model metric and cannot
authorize router tuning.

## 7. Final Decision Rule

`METHOD_GO`:

The five-checkpoint M2 ensemble passes G1 through G5.

`CONDITIONAL_GO`:

The paired point estimate is positive, but one or more of G1 through G5 are
incomplete or fail for a reason other than an invalidity condition below.

`METHOD_NO_GO`:

The point estimate is not positive, or activation/compute control fails while
the primary utility is also unsupported.

`INCONCLUSIVE_OR_INVALID`:

Any integrity, reproduction, alignment, data, inference, or execution failure
listed in Section 8 occurs. This state takes precedence over all other states.

No post-OOD threshold, seed, checkpoint, subgroup, comparator, bootstrap, or
endpoint search is permitted.

## 8. Fail-Closed Execution Rules

The OOD run must return `INCONCLUSIVE_OR_INVALID` without a substitute
analysis if any of the following occurs:

1. The OOD manifest, dataset payload, feature cache, preprocessing inputs, or
   source metadata are missing.
2. Any required SHA256 is missing or differs from its frozen value.
3. Baseline or checkpoint reproduction fails.
4. Frame order, labels, masks, source IDs, or prediction rows are misaligned.
5. Any of the five required M2 seeds is missing, unreadable, or has mismatched
   metadata.
6. The seed set, budget, feature dimension, causal window, normalization,
   threshold, or split fingerprint differs from the frozen method.
7. Source clusters are missing or empty, or the paired bootstrap cannot use
   identical clusters for M2 and M0.
8. The OOD population overlaps a previously used evaluation population or is
   not demonstrably the predeclared untouched final-holdout population.
9. The frozen evaluation code has changed.
10. An execution failure prevents completion of all five seeds and the
    primary source-cluster bootstrap.

On failure, do not impute data, drop a seed, retrain, retune, fall back to a
different router, partially evaluate, or report a positive outcome. Preserve
the failure record and stop.

## 9. Runtime Data Binding

At freeze time the OOD manifest path and hash are intentionally unbound:

`UNBOUND_AT_FREEZE_NOT_ACCESSED`

After explicit authorization to execute, and before opening any OOD sample,
the execution preflight must record:

1. Absolute OOD manifest path.
2. Manifest SHA256.
3. Dataset payload SHA256, if separately supplied.
4. Source-cluster metadata SHA256.
5. Feature/preprocessing artifact SHA256 values.
6. Verification that no hash or population overlaps prior TRAIN, DEV,
   INTERNAL_TEST, A9/AE/E1/E2/E3/E4, CAR, A14, or any previous OOD use.

Runtime binding is integrity metadata. It does not permit changing the
endpoint, comparator, method, seed set, gate, subgroup, or inference rule.

## 10. Frozen Execution Code

The canonical scoring and inference implementation is the frozen A-v2 code
identified in `a_final_freeze_manifest.json`, especially:

`reproductions/difficulty_adaptive_context/run_a2_method_study.py`

and:

`reproductions/difficulty_adaptive_context/recover_a2_method_study.py`

The A-v2 evaluation functions `taxonomy_from_scores`,
`paired_cluster_bootstrap`, `evaluate_primary`, and `assess_primary` define
the endpoint, taxonomy, bootstrap, and gates. No alternative implementation
may compute the primary result.

Data transport into the same frozen frame array schema is permitted only if
it is value-preserving, hash-bound before scoring, and cannot alter labels,
predictions, masks, source IDs, or features. Any analytic prediction,
recalibration, threshold change, or feature operation is forbidden.

If a value-preserving OOD adapter is required, it must be hashed and verified
before OOD access. The adapter may load and align frozen arrays only; it
cannot implement scoring or inference logic.

## 11. Required Outputs

An authorized OOD execution must produce:

1. OOD preflight manifest with all runtime binding hashes.
2. Per-seed metric rows.
3. Seed-averaged paired delta.
4. Source-cluster bootstrap distribution and CI.
5. G1 through G5 gate evaluations.
6. Seen/unseen and allowed subgroup results.
7. Final status under the Section 7 rule.
8. Hash of the unchanged protocol and frozen evaluation code.

No OOD output may be produced during the freeze task. The next experiment
remains unauthorized at freeze completion.
