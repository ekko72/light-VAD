# FINAL_OOD_INVALID_EXECUTION_AUDIT

Audit ID: `FINAL-OOD-INVALID-EXECUTION-AUDIT-v1`

Audit mode: read-only review of existing execution records.

## Exact Failure Point

The execution reached runtime preflight and successfully verified the frozen
protocol and freeze-manifest identities. It then hash-bound the authorized
OOD manifest before reading it.

The failure occurred during OOD candidate resolution. The OOD manifest
declared:

`NO_UNUSED_CONFIRMATORY_DATASET_AVAILABLE`

and contained:

`candidate_dataset: null`

The manifest stated that the available LibriSpeech test speakers were already
used for A-v1 evaluation and that no new untouched confirmatory dataset was
present. This is fail-closed rule Section 8, item 1: the OOD dataset payload
was missing.

Execution stopped at this point. No substitute dataset was used.

## Audit Fields

FAILURE_STAGE=RUNTIME_PREFLIGHT_OOD_MANIFEST_CANDIDATE_RESOLUTION

FAILURE_REASON=NO_UNUSED_CONFIRMATORY_DATASET_AVAILABLE; candidate_dataset=null; no untouched final-holdout dataset available

FINAL_OOD_FILES_OPENED=manifest_only

FINAL_OOD_AUDIO_READ=false

FINAL_OOD_LABELS_READ=false

FINAL_OOD_METADATA_READ=true; manifest status and candidate metadata only

MODEL_INFERENCE_STARTED=false

N_FRAMES_INFERRED=0

N_SOURCES_INFERRED=0

ANY_PER_FRAME_RESULT_PRODUCED=false

ANY_PER_SOURCE_RESULT_PRODUCED=false

ANY_AGGREGATE_RESULT_PRODUCED=false

ANY_OOD_OUTCOME_EXPOSED_TO_USER=false

ANY_OOD_INFORMATION_USED_FOR_MODEL_OR_PROTOCOL_CHANGE=false

HASH_MISMATCH=false

MISSING_FILE_OR_PATH=true; candidate_dataset=null

CODE_OR_ENVIRONMENT_FAILURE=false

SCIENTIFIC_BLINDNESS_PRESERVED=true

RERUN_WITH_IDENTICAL_FROZEN_PROTOCOL_SCIENTIFICALLY_VALID=false

## Consumption Boundary

`NEW_FINAL_OOD_CONSUMED=true` remains unchanged. No rerun was performed. The
absence of per-frame or aggregate OOD results means no scientific outcome was
exposed, but the consumed access state still prohibits treating this holdout
as untouched.

NEXT_EXPERIMENT_AUTHORIZED=false
