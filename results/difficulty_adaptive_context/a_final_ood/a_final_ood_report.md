# A Final OOD Execution Report

Execution ID: `A-FINAL-OOD-EXECUTION-v1`

Execution status: `INVALID_EXECUTION`

Final status: `INCONCLUSIVE_OR_INVALID`

## Outcome

The authorized execution stopped during runtime preflight. The frozen OOD
manifest at:

`results/difficulty_adaptive_context/a2_method_study/a2_new_final_ood_manifest.json`

has SHA256:

`1097F834A8D59EC5F63197D07A8AEF81A39F65F51C7907EF73EED28FD971B6E7`

and declares:

`NO_UNUSED_CONFIRMATORY_DATASET_AVAILABLE`.

Its `candidate_dataset` field is `null`. The manifest states that the
available LibriSpeech test speakers were already used for A-v1 evaluation and
that no new untouched confirmatory dataset is present.

No dataset payload was loaded. No OOD sample was read or scored. No baseline
reproduction, checkpoint loading, inference, bootstrap, subgroup analysis, or
gate evaluation was attempted.

## Fail-Closed Decision

This is a Section 8, item 1 failure: the OOD dataset payload is missing.
Execution therefore returns `INCONCLUSIVE_OR_INVALID`. No substitute dataset or
alternative analysis was used.

The authorized manifest access is recorded as `NEW_FINAL_OOD_CONSUMED=true`.
It cannot be treated as an untouched test after this execution.

## Preserved Frozen State

| Item | Value |
| --- | --- |
| Protocol | `A-FINAL-OOD-v1` |
| Protocol SHA256 | `4D9BDD2A6381E28F8E50B43352F5A0AA33CA916188839727FD2EB3F609A910E8` |
| Freeze manifest SHA256 | `FB6EB5F005BE596308E444644A19CE34B54262700FEC2779707D91F38B82AF43` |
| Primary result | Not computed |
| Primary CI | Not computed |
| Final gates | Not evaluated |
| Seen result | Not computed |
| Unseen result | Not computed |
| Maximum activation | Not computed |
| Training performed | `false` |
| Protocol deviations | `[]` |
| Next experiment authorized | `false` |

## Claim Boundary

This execution supports no new A claim. It does not refute the frozen A-v2
result or any preserved mechanism claim. The pre-existing official states
remain unchanged, and A-v2 remains `CONDITIONAL_GO`.
