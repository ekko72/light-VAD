# -*- coding: utf-8 -*-
"""Execute E2 remote-context compatibility decomposition.

E2 is an inference-only mechanism audit of the frozen E1 RF384 checkpoint.
Matching decisions use manifest metadata, labels, and waveform energy only.
No model output or loss is consulted while constructing donors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from reproductions.difficulty_adaptive_context import run_e1 as e1
from reproductions.difficulty_adaptive_context.data import (
    causal_frame_bounds,
    causal_frame_labels,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE, read_manifest
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "E2-RCCD-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_E2_OUTCOME_ANALYSIS"
RUNNER_MODULE = "reproductions.difficulty_adaptive_context.run_e2"

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e2_remote_context_compatibility"
)
PROTOCOL_PATH = OUTPUT_ROOT / "e2_protocol_freeze.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "e2_protocol_sha256.txt"
RUNNER_HASH_PATH = OUTPUT_ROOT / "e2_runner_sha256.txt"
EXECUTION_MANIFEST_PATH = OUTPUT_ROOT / "e2_execution_manifest.json"
BASELINE_PATH = OUTPUT_ROOT / "e2_baseline_reproduction.json"
FEASIBILITY_PATH = OUTPUT_ROOT / "e2_matching_feasibility.csv"
MATCH_QUALITY_PATH = OUTPUT_ROOT / "e2_match_quality.csv"
FRAME_RESULTS_PATH = OUTPUT_ROOT / "e2_frame_results.parquet"
CONDITION_RESULTS_PATH = OUTPUT_ROOT / "e2_condition_results.csv"
PRIMARY_CONTRASTS_PATH = OUTPUT_ROOT / "e2_primary_contrasts.csv"
CLUSTER_BOOTSTRAP_PATH = OUTPUT_ROOT / "e2_cluster_bootstrap.csv"
CORRECTION_SURVIVAL_PATH = OUTPUT_ROOT / "e2_correction_survival.csv"
TAXONOMY_TRANSITIONS_PATH = OUTPUT_ROOT / "e2_taxonomy_transitions.csv"
EVENT_STRATA_PATH = OUTPUT_ROOT / "e2_event_strata.csv"
ONSET_OFFSET_PATH = OUTPUT_ROOT / "e2_onset_offset_asymmetry.csv"
SEEN_UNSEEN_PATH = OUTPUT_ROOT / "e2_seen_unseen.csv"
SOURCE_INFLUENCE_PATH = OUTPUT_ROOT / "e2_source_influence.csv"
DONOR_SEED_CONSISTENCY_PATH = OUTPUT_ROOT / "e2_donor_seed_consistency.csv"
E1_RECONCILIATION_PATH = OUTPUT_ROOT / "e2_e1_reconciliation.md"
FINAL_SUMMARY_PATH = OUTPUT_ROOT / "e2_final_summary.json"
FINAL_REPORT_PATH = OUTPUT_ROOT / "e2_final_report.md"
CLAIM_FREEZE_PATH = OUTPUT_ROOT / "e2_claim_freeze.md"
FIGURES_ROOT = OUTPUT_ROOT / "e2_figures"
BASELINE_SCORE_CACHE_PATH = OUTPUT_ROOT / "_e2_baseline_scores.npz"
E1_C4_AUDIT_PATH = OUTPUT_ROOT / "e2_e1_c4_replay.json"

E1_RESULTS_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e1_remote_temporal_intervention"
)
E1_PROTOCOL_PATH = E1_RESULTS_ROOT / "e1_v2_protocol_freeze.json"
E1_PROTOCOL_SHA256_PATH = E1_RESULTS_ROOT / "e1_v2_protocol_sha256.txt"
E1_RUNNER_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "run_e1.py"
)
E1_RUNNER_SHA256_PATH = E1_RESULTS_ROOT / "e1_v2_runner_sha256.txt"
E1_FINAL_SUMMARY_PATH = E1_RESULTS_ROOT / "e1_v2_final_summary.json"
E1_C1_PATH = E1_RESULTS_ROOT / "e1_v2_c1_validation.json"
E1_FRAME_RESULTS_PATH = E1_RESULTS_ROOT / "e1_v2_frame_results.parquet"

EXPECTED_E1_PROTOCOL_SHA256 = (
    "D61BB571BA60454AC0AF28491AEF278967B53D7764E58C8BBA797C1313C886E6"
)
EXPECTED_E1_RUNNER_SHA256 = (
    "A429DCBF8ADF4FF9A4C1601F75D941E623A171A81F6FA084B7D7FAFBA3265BB3"
)
E1_C4_REFERENCE_DELTA = 0.032131332797710244

# E1 full-population baseline reproduction.
E1_EXPECTED_TOTAL_FRAMES = 553_532
E1_EXPECTED_TEST_FRAMES = 298_300
E2_EXPECTED_NOISY_TEST_FRAMES = 240_565
E2_EXPECTED_NOISY_TEST_RECORDS = 396
E2_EXPECTED_NOISY_TEST_SOURCES = 96
E2_EXPECTED_NOISY_TEST_SPEAKERS = 20
BASELINE_TOLERANCE = 1e-6

# Frozen inference and endpoint settings.
VALID_START = e1.VALID_START
DECISION_THRESHOLD = e1.DECISION_THRESHOLD
UNCERTAINTY_THRESHOLD = e1.GATE_THRESHOLD
REMOTE_TAPS = e1.REMOTE_TAPS
REMOTE_WINDOW_TAPS = e1.REMOTE_WINDOW_TAPS
DONOR_SEEDS = tuple(int(value) for value in e1.INTERVENTION_SEEDS)
BOOTSTRAP_REPEATS = 2_000
SOURCE_BOOTSTRAP_SEED = 20_260_921
SPEAKER_BOOTSTRAP_OFFSET = 500_000
LOGLOSS_EPSILON = 1e-12

# Matching rules are frozen before model outcome scoring.
MAX_DONOR_RECORDS_PER_SEED_SHORTLIST = 5
MAX_DONOR_ANCHORS = 96
ANCHOR_TOP_K = 5
SPEECH_STATE_BINS = (
    (0.00, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.01),
)
SPEECH_STATE_NAMES = ("low", "medium", "high", "very_high")
C6_MIN_SPEECH_STATE_MISMATCH = 0.25
MATERIAL_CONTRAST_DELTA = 0.005
MIN_PRIMARY_FRAMES = 1_000
MIN_PRIMARY_SOURCES = 20
MIN_MATCH_SUCCESS_RATE = 0.80
MAX_TOP_DONOR_SHARE = 0.20
MIN_SEED_POSITIVE_COUNT = 4
B1_MIN_CONTRAST = MATERIAL_CONTRAST_DELTA
B2_MIN_CONTRAST = MATERIAL_CONTRAST_DELTA
B4_MIN_EVENT_EXCESS = MATERIAL_CONTRAST_DELTA
SPEECH_STATE_DOMINANCE_FRACTION = 0.60
E1_C4_AUDIT_TOLERANCE = 1e-6
EVENT_DISTANCE_WINDOW = 5
STABLE_RUN_MIN = 21

C0_FULL = "C0_FULL"
C1_SAME_UTT_DIFFERENT_TIME = "C1_SAME_UTT_DIFFERENT_TIME"
C2_SAME_NOISE_INSTANCE = "C2_SAME_NOISE_INSTANCE"
C3_SAME_NOISE_CLASS = "C3_SAME_NOISE_CLASS"
C4_SAME_CLASS_WRONG_SNR = "C4_SAME_CLASS_WRONG_SNR"
C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR = (
    "C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR"
)
C6_SPEECH_STATE_MISMATCH = "C6_SPEECH_STATE_MISMATCH"
C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE = (
    "C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE"
)

CONDITION_ORDER = (
    C0_FULL,
    C1_SAME_UTT_DIFFERENT_TIME,
    C2_SAME_NOISE_INSTANCE,
    C3_SAME_NOISE_CLASS,
    C4_SAME_CLASS_WRONG_SNR,
    C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR,
    C6_SPEECH_STATE_MISMATCH,
    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
)
REPLACEMENT_CONDITIONS = CONDITION_ORDER[1:]
CONDITION_SHORT = {
    C0_FULL: "C0",
    C1_SAME_UTT_DIFFERENT_TIME: "C1",
    C2_SAME_NOISE_INSTANCE: "C2",
    C3_SAME_NOISE_CLASS: "C3",
    C4_SAME_CLASS_WRONG_SNR: "C4",
    C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR: "C5",
    C6_SPEECH_STATE_MISMATCH: "C6",
    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE: "C7",
}
CONDITION_ROLE = {
    C0_FULL: "baseline",
    C1_SAME_UTT_DIFFERENT_TIME: "utterance_identity",
    C2_SAME_NOISE_INSTANCE: "noise_instance",
    C3_SAME_NOISE_CLASS: "noise_class_control",
    C4_SAME_CLASS_WRONG_SNR: "level_mismatch",
    C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR: "noise_class_mismatch",
    C6_SPEECH_STATE_MISMATCH: "speech_state_mismatch",
    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE: "best_generic_control",
}

PRIMARY_CONTRASTS = (
    (
        "P1_UTTERANCE",
        C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
        C1_SAME_UTT_DIFFERENT_TIME,
        "C7 minus C1; positive means different-source replacement is more harmful",
    ),
    (
        "P2_NOISE_INSTANCE",
        C3_SAME_NOISE_CLASS,
        C2_SAME_NOISE_INSTANCE,
        "C3 minus C2; positive means class-only replacement is more harmful",
    ),
    (
        "P3_NOISE_CLASS",
        C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR,
        C3_SAME_NOISE_CLASS,
        "C5 minus C3; positive means class mismatch adds damage",
    ),
    (
        "P4_SNR",
        C4_SAME_CLASS_WRONG_SNR,
        C3_SAME_NOISE_CLASS,
        "C4 minus C3; positive means SNR mismatch adds damage",
    ),
    (
        "P5_SPEECH_STATE",
        C6_SPEECH_STATE_MISMATCH,
        C3_SAME_NOISE_CLASS,
        "C6 minus C3; positive means speech-state mismatch adds damage",
    ),
    (
        "P6_RESIDUAL_IDENTITY",
        C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
        C0_FULL,
        "C7 minus FULL; positive means residual replacement damage remains",
    ),
)

PRIMARY_CONDITIONS = (
    C1_SAME_UTT_DIFFERENT_TIME,
    C2_SAME_NOISE_INSTANCE,
    C3_SAME_NOISE_CLASS,
    C4_SAME_CLASS_WRONG_SNR,
    C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR,
    C6_SPEECH_STATE_MISMATCH,
    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
)

ALLOWED_E2_STATUSES = (
    "BACKGROUND_REFERENCE_SUPPORTED",
    "BACKGROUND_REFERENCE_CONDITIONAL",
    "RESIDUAL_CONTEXT_IDENTITY_EFFECT",
    "SPEECH_STATE_COMPATIBILITY_SUPPORTED",
    "COMPATIBILITY_UNRESOLVED",
    "INCONCLUSIVE_OR_INVALID",
)

FIGURE_FILES = (
    "figure1_condition_delta_logloss.png",
    "figure2_primary_contrasts.png",
    "figure3_correction_survival.png",
    "figure4_event_asymmetry.png",
    "figure5_seen_unseen.png",
)

REQUIRED_OUTPUT_FILES = (
    "e2_protocol_freeze.json",
    "e2_protocol_sha256.txt",
    "e2_runner_sha256.txt",
    "e2_execution_manifest.json",
    "e2_baseline_reproduction.json",
    "e2_matching_feasibility.csv",
    "e2_match_quality.csv",
    "e2_frame_results.parquet",
    "e2_condition_results.csv",
    "e2_primary_contrasts.csv",
    "e2_cluster_bootstrap.csv",
    "e2_correction_survival.csv",
    "e2_taxonomy_transitions.csv",
    "e2_event_strata.csv",
    "e2_onset_offset_asymmetry.csv",
    "e2_seen_unseen.csv",
    "e2_source_influence.csv",
    "e2_donor_seed_consistency.csv",
    "e2_e1_reconciliation.md",
    "e2_figures/",
    "e2_e1_c4_replay.json",
    "e2_final_summary.json",
    "e2_final_report.md",
    "e2_claim_freeze.md",
)

REQUIRED_SUMMARY_KEYS = (
    "E2_STATUS",
    "BASELINE_REPRODUCED",
    "MATCHING_FEASIBILITY_BY_CONDITION",
    "SAME_UTT_EFFECT",
    "SAME_NOISE_INSTANCE_EFFECT",
    "SAME_NOISE_CLASS_EFFECT",
    "WRONG_SNR_EFFECT",
    "DIFFERENT_CLASS_EFFECT",
    "SPEECH_STATE_MISMATCH_EFFECT",
    "BEST_METADATA_MATCHED_EFFECT",
    "P1_UTTERANCE_CONTRAST",
    "P2_NOISE_INSTANCE_CONTRAST",
    "P3_NOISE_CLASS_CONTRAST",
    "P4_SNR_CONTRAST",
    "P5_SPEECH_STATE_CONTRAST",
    "P6_RESIDUAL_IDENTITY_EFFECT",
    "ONSET_EFFECT",
    "TRANSITION_EFFECT",
    "OFFSET_EFFECT",
    "ONSET_MINUS_OFFSET",
    "SEEN_EFFECT",
    "UNSEEN_EFFECT",
    "R_SURVIVAL_BY_CONDITION",
    "SOURCE_LOSO_STABILITY",
    "DONOR_SEED_STABILITY",
    "E1_RECONCILIATION",
    "PROTOCOL_DEVIATIONS",
    "TRAINING_PERFORMED",
    "NEW_FINAL_OOD_TOUCHED",
    "NEXT_EXPERIMENT_AUTHORIZED",
)

EVENT_STRATA = {
    "S1_ONSET_NEAR": "ground-truth speech with onset distance 0-5 frames",
    "S2_TRANSITION_NEAR": "Short-posterior switch age 0-5 frames",
    "S3_OFFSET_NEAR": "post-speech non-speech with offset distance 0-5 frames",
    "S4_STABLE_SPEECH": "speech run length at least 21 frames",
    "S5_STABLE_SILENCE": "silence run length at least 21 frames",
}

EVENT_MATCH_RULES = {
    "S1_ONSET_NEAR": (
        "match each speech frame with onset distance 0-5 to the nearest "
        "non-overlapping stable-speech frame in the same item, same "
        "ground-truth label, and stable speech run length at least 21"
    ),
    "S2_TRANSITION_NEAR": (
        "match each Short-posterior transition frame with absolute switch "
        "age 0-5 to the nearest non-overlapping stable frame in the same "
        "item, same ground-truth label, and stable run length at least 21"
    ),
    "S3_OFFSET_NEAR": (
        "match each non-speech frame with offset distance 0-5 to the "
        "nearest non-overlapping stable-silence frame in the same item, "
        "same ground-truth label, and stable silence run length at "
        "least 21"
    ),
}

EVENT_STRATA_PROVENANCE = {
    "source_protocols": [
        "reproductions/difficulty_adaptive_context/ax3_protocol.json",
        "reproductions/difficulty_adaptive_context/ax3_freeze_manifest.json",
        "reproductions/difficulty_adaptive_context/ae_protocol_freeze.json",
        "results/cross_architecture_replication/car_protocol_freeze.json",
        "results/cross_architecture_replication/car_protocol_sha256.txt",
    ],
    "bindings": {
        "S1_ONSET_NEAR": {
            "source": (
                "ax3_protocol.json#event_definitions.dimensions."
                "onset_distance"
            ),
            "imported_definition": (
                "ground-truth speech frames with frames-since-onset in "
                "frozen bins 0, 1-2, or 3-5"
            ),
            "distance_window": 5,
        },
        "S2_TRANSITION_NEAR": {
            "source": (
                "ax3_protocol.json#event_definitions.dimensions."
                "recent_posterior_transition"
            ),
            "imported_definition": (
                "age of the most recent Short-prediction switch in the "
                "current utterance in frozen bins 0, 1-2, or 3-5"
            ),
            "distance_window": 5,
        },
        "S3_OFFSET_NEAR": {
            "source": (
                "ax3_protocol.json#event_definitions.dimensions."
                "offset_distance"
            ),
            "imported_definition": (
                "post-speech ground-truth non-speech frames in frozen bins "
                "0, 1-2, or 3-5"
            ),
            "distance_window": 5,
        },
        "S4_STABLE_SPEECH": {
            "source": (
                "ax3_protocol.json#event_definitions.dimensions."
                "speech_run_length"
            ),
            "imported_definition": (
                "ground-truth speech run length at least 21 frames, using "
                "the frozen 21-50 and longer run bins"
            ),
            "minimum_run_length": 21,
        },
        "S5_STABLE_SILENCE": {
            "source": (
                "ax3_protocol.json#event_definitions.dimensions."
                "silence_run_length"
            ),
            "imported_definition": (
                "ground-truth silence run length at least 21 frames, using "
                "the frozen 21-50 and longer run bins"
            ),
            "minimum_run_length": 21,
        },
    },
    "import_rule": (
        "E2 imports these definitions from already frozen CAR/AE/AX3 "
        "protocols and does not redefine windows or run thresholds after "
        "outcome inspection."
    ),
}

FROZEN_INPUT_PATHS = {
    "rf384_checkpoint": e1.RF384_CHECKPOINT,
    "short_checkpoint": e1.SHORT_CHECKPOINT,
    "rf384_reference": e1.RF384_REFERENCE,
    "test_manifest": e1.MANIFEST_PATH,
    "e1_protocol": E1_PROTOCOL_PATH,
    "e1_runner": E1_RUNNER_PATH,
    "e1_final_summary": E1_FINAL_SUMMARY_PATH,
    "e1_c1_validation": E1_C1_PATH,
    "e1_frame_results": E1_FRAME_RESULTS_PATH,
}
FROZEN_SOURCE_PATHS = (
    Path("reproductions/difficulty_adaptive_context/adaptive_model.py"),
    Path("reproductions/difficulty_adaptive_context/data.py"),
    Path("reproductions/difficulty_adaptive_context/evaluate_adaptive.py"),
    Path("reproductions/difficulty_adaptive_context/analyze_context_gate.py"),
    Path("reproductions/marblenet_vad/model.py"),
    Path("reproductions/marblenet_vad/features.py"),
    Path("reproductions/marblenet_vad/dataset.py"),
    Path("reproductions/difficulty_adaptive_context/ax3_protocol.json"),
    Path("reproductions/difficulty_adaptive_context/ax3_freeze_manifest.json"),
    Path("reproductions/difficulty_adaptive_context/ae_protocol_freeze.json"),
    Path(
        "results/cross_architecture_replication/"
        "car_protocol_freeze.json"
    ),
    Path(
        "results/cross_architecture_replication/"
        "car_protocol_sha256.txt"
    ),
)


@dataclass(frozen=True)
class ManifestMeta:
    sample_id: str
    source_key: str
    speaker_id: str
    noise_name: str
    noise_source: str
    noise_start_sample: int
    snr_db: float
    noise_instance: tuple[str, int]
    clean_duration_s: float
    speech_duration_s: float
    silence_duration_s: float


@dataclass(frozen=True)
class MatchRecord:
    record: e1.Record
    meta: ManifestMeta
    target_positions: np.ndarray
    target_speech_state: np.ndarray
    target_rms: np.ndarray
    anchor_positions: np.ndarray
    anchor_speech_state: np.ndarray
    anchor_rms: np.ndarray
    anchor_indices: np.ndarray
    representative_indices: np.ndarray
    mean_remote_speech_state: float
    median_remote_rms: float

    @property
    def item_index(self) -> int:
        return int(self.record.item_index)

    @property
    def source_key(self) -> str:
        return self.record.source_key

    @property
    def speaker_id(self) -> str:
        return self.record.speaker_id

    @property
    def noise_name(self) -> str:
        return self.record.noise_name

    @property
    def condition_value(self) -> str:
        return self.record.condition


@dataclass(frozen=True)
class Assignment:
    condition: str
    seed: int
    target_item_index: int
    donor_item_index: int
    target_positions: np.ndarray
    donor_anchors: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.target_positions.size)


def _output_path(name: str) -> Path:
    return OUTPUT_ROOT / name.rstrip("/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_hashes(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({Path(value).resolve() for value in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: {path}")
        records.append(
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return records


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return e1._max_abs_error(actual, expected)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_status_short() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"frozen input not found: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _stable_hash_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        "big",
    )


def _speech_state_bin(fraction: float) -> int:
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("speech-state fraction must be in [0, 1]")
    for index, (low, high) in enumerate(SPEECH_STATE_BINS):
        if low <= value < high:
            return int(index)
    return int(len(SPEECH_STATE_BINS) - 1)


def _speech_state_name(index: int) -> str:
    return str(SPEECH_STATE_NAMES[int(index)])


def _log_ratio(value: float, reference: float) -> float:
    return abs(
        math.log((max(float(value), 0.0) + 1e-12) / (
            max(float(reference), 0.0) + 1e-12
        ))
    )


def _frame_rms(
    waveform: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    starts, ends = causal_frame_bounds(int(n_frames))
    starts = np.clip(starts, 0, waveform.size)
    ends = np.clip(ends, 0, waveform.size)
    result = np.zeros(int(n_frames), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end > start:
            values = waveform[int(start) : int(end)]
            result[index] = float(np.sqrt(np.mean(values * values)))
    return result


def _remote_features(
    *,
    labels: np.ndarray,
    frame_rms: np.ndarray,
    frame_index: int,
) -> tuple[float, float, bool]:
    positions = [
        int(frame_index) - int(tap)
        for tap in REMOTE_WINDOW_TAPS
        if int(frame_index) >= int(tap)
    ]
    if not positions:
        return 0.0, 0.0, False
    state = float(
        np.mean(
            np.asarray(labels, dtype=np.float64)[
                np.asarray(positions, dtype=np.int64)
            ]
        )
    )
    rms = float(np.mean(np.asarray(frame_rms, dtype=np.float64)[positions]))
    return state, rms, True


def _manifest_meta_for_population(
    population: e1.Population,
) -> dict[int, ManifestMeta]:
    rows = read_manifest(e1.MANIFEST_PATH)
    by_key: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("sample_id", "")),
            str(row.get("noise_name", "")),
            str(row.get("snr_db", "")),
        )
        by_key[key].append(row)
    result: dict[int, ManifestMeta] = {}
    for record in population.records:
        if record.is_clean:
            continue
        key = (
            str(record.item.sample_id),
            str(record.noise_name),
            str(record.condition),
        )
        matching = by_key.get(key, [])
        if len(matching) != 1:
            raise ValueError(
                "could not uniquely map noisy evaluation item "
                f"{record.item_index} {key!r} to manifest; "
                f"found {len(matching)} rows"
            )
        row = matching[0]
        noise_source = str(row.get("noise_source", ""))
        noise_start = int(row.get("noise_start_sample", "-1"))
        result[int(record.item_index)] = ManifestMeta(
            sample_id=record.item.sample_id,
            source_key=record.source_key,
            speaker_id=record.speaker_id,
            noise_name=record.noise_name,
            noise_source=noise_source,
            noise_start_sample=noise_start,
            snr_db=float(row.get("snr_db", "nan")),
            noise_instance=(noise_source, noise_start),
            clean_duration_s=float(row.get("clean_duration_s", "nan")),
            speech_duration_s=float(row.get("speech_duration_s", "nan")),
            silence_duration_s=float(row.get("silence_duration_s", "nan")),
        )
    return result


def _noisy_test_records(population: e1.Population) -> list[e1.Record]:
    return [
        record
        for record in population.records
        if not record.is_clean
        and np.any(e1._record_test_positions(record, population.test_mask))
    ]


def _all_test_records(population: e1.Population) -> list[e1.Record]:
    return [
        record
        for record in population.records
        if np.any(e1._record_test_positions(record, population.test_mask))
    ]


def _build_match_records(
    population: e1.Population,
    metas: Mapping[int, ManifestMeta],
) -> dict[int, MatchRecord]:
    result: dict[int, MatchRecord] = {}
    for record in _noisy_test_records(population):
        waveform = e1.read_int16_audio(record.item.audio_path)
        full_labels = causal_frame_labels(
            np.load(record.item.label_path),
            record.n_frames,
        )
        frame_rms = _frame_rms(waveform, record.n_frames)
        target_positions = e1._record_test_positions(
            record,
            population.test_mask,
        )
        target_state = np.empty(target_positions.size, dtype=np.float64)
        target_rms = np.empty(target_positions.size, dtype=np.float64)
        for row, frame_index in enumerate(target_positions):
            state, rms, available = _remote_features(
                labels=full_labels,
                frame_rms=frame_rms,
                frame_index=int(frame_index),
            )
            if not available:
                raise RuntimeError("test target has no available remote tap")
            target_state[row] = state
            target_rms[row] = rms

        anchor_positions = np.arange(
            max(VALID_START, max(REMOTE_WINDOW_TAPS)),
            record.n_frames,
            dtype=np.int64,
        )
        anchor_state = np.empty(anchor_positions.size, dtype=np.float64)
        anchor_rms = np.empty(anchor_positions.size, dtype=np.float64)
        for row, frame_index in enumerate(anchor_positions):
            state, rms, available = _remote_features(
                labels=full_labels,
                frame_rms=frame_rms,
                frame_index=int(frame_index),
            )
            if not available:
                raise RuntimeError("legal donor anchor lacks remote history")
            anchor_state[row] = state
            anchor_rms[row] = rms
        order = np.lexsort(
            (
                anchor_rms,
                anchor_state,
                np.asarray(
                    [_speech_state_bin(value) for value in anchor_state],
                    dtype=np.int64,
                ),
            )
        )
        if anchor_positions.size <= MAX_DONOR_ANCHORS:
            representative = order.astype(np.int64)
        else:
            picks = np.rint(
                np.linspace(
                    0,
                    anchor_positions.size - 1,
                    MAX_DONOR_ANCHORS,
                )
            ).astype(np.int64)
            representative = order[picks].astype(np.int64)
        result[int(record.item_index)] = MatchRecord(
            record=record,
            meta=metas[int(record.item_index)],
            target_positions=target_positions,
            target_speech_state=target_state,
            target_rms=target_rms,
            anchor_positions=anchor_positions,
            anchor_speech_state=anchor_state,
            anchor_rms=anchor_rms,
            anchor_indices=np.arange(
                anchor_positions.size,
                dtype=np.int64,
            ),
            representative_indices=representative,
            mean_remote_speech_state=float(np.mean(anchor_state)),
            median_remote_rms=float(np.median(anchor_rms)),
        )
    return result


def _same_noise_instance(target: MatchRecord, donor: MatchRecord) -> bool:
    return bool(target.meta.noise_instance == donor.meta.noise_instance)


def _different_utterance(target: MatchRecord, donor: MatchRecord) -> bool:
    return bool(target.item_index != donor.item_index)


def _different_source(target: MatchRecord, donor: MatchRecord) -> bool:
    return bool(target.source_key != donor.source_key)


def _different_speaker(target: MatchRecord, donor: MatchRecord) -> bool:
    return bool(target.speaker_id != donor.speaker_id)


def _record_cost(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    condition: str,
    seed: int,
) -> float:
    snr_diff = abs(float(target.meta.snr_db) - float(donor.meta.snr_db))
    state_diff = abs(
        float(target.mean_remote_speech_state)
        - float(donor.mean_remote_speech_state)
    )
    rms_diff = _log_ratio(
        donor.median_remote_rms,
        target.median_remote_rms,
    )
    duration_diff = _log_ratio(
        float(donor.record.n_frames),
        float(target.record.n_frames),
    )
    speaker_penalty = 0.0 if _different_speaker(target, donor) else 0.20
    base = state_diff + 0.25 * rms_diff + 0.05 * duration_diff
    if condition == C3_SAME_NOISE_CLASS:
        return base + speaker_penalty
    if condition == C4_SAME_CLASS_WRONG_SNR:
        return (
            base
            + 0.05 * abs(snr_diff - 10.0)
            + speaker_penalty
        )
    if condition == C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR:
        return base + 0.50 * snr_diff + speaker_penalty
    if condition == C6_SPEECH_STATE_MISMATCH:
        return (
            0.25 * rms_diff
            + 0.05 * duration_diff
            + speaker_penalty
        )
    if condition == C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE:
        return (
            base
            + 0.10 * snr_diff
            + speaker_penalty
        )
    if condition == C2_SAME_NOISE_INSTANCE:
        return base + speaker_penalty
    return base + _stable_hash_int(
        PROTOCOL_ID,
        condition,
        int(seed),
        target.meta.sample_id,
        donor.meta.sample_id,
    ) / float(2**64)


def _hard_candidate_ok(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    condition: str,
) -> bool:
    if not _different_utterance(target, donor):
        return False
    if condition == C2_SAME_NOISE_INSTANCE:
        return bool(
            _same_noise_instance(target, donor)
            and target.condition_value == donor.condition_value
        )
    if condition == C3_SAME_NOISE_CLASS:
        return bool(
            donor.noise_name == target.noise_name
            and donor.condition_value == target.condition_value
            and not _same_noise_instance(target, donor)
            and _different_source(target, donor)
        )
    if condition == C4_SAME_CLASS_WRONG_SNR:
        return bool(
            donor.noise_name == target.noise_name
            and abs(
                float(target.meta.snr_db) - float(donor.meta.snr_db)
            )
            >= 10.0
            and not _same_noise_instance(target, donor)
            and _different_source(target, donor)
        )
    if condition == C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR:
        return bool(
            donor.noise_name != target.noise_name
            and donor.condition_value == target.condition_value
            and _different_source(target, donor)
        )
    if condition == C6_SPEECH_STATE_MISMATCH:
        mean_mismatch = abs(
            float(donor.mean_remote_speech_state)
            - float(target.mean_remote_speech_state)
        )
        return bool(
            donor.noise_name == target.noise_name
            and donor.condition_value == target.condition_value
            and _different_source(target, donor)
            and mean_mismatch >= C6_MIN_SPEECH_STATE_MISMATCH
        )
    if condition == C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE:
        return bool(
            donor.noise_name == target.noise_name
            and donor.condition_value == target.condition_value
            and _different_source(target, donor)
            and not _same_noise_instance(target, donor)
        )
    return False


def _ordered_candidates(
    *,
    target: MatchRecord,
    donors: Sequence[MatchRecord],
    condition: str,
    seed: int,
) -> list[MatchRecord]:
    candidates = [
        donor
        for donor in donors
        if _hard_candidate_ok(
            target=target,
            donor=donor,
            condition=condition,
        )
    ]
    candidates.sort(
        key=lambda donor: (
            _record_cost(
                target=target,
                donor=donor,
                condition=condition,
                seed=seed,
            ),
            _stable_hash_int(
                PROTOCOL_ID,
                condition,
                target.meta.sample_id,
                donor.meta.sample_id,
            ),
        )
    )
    return candidates


def _seed_shortlist(
    candidates: Sequence[MatchRecord],
    *,
    target: MatchRecord,
    condition: str,
    seed: int,
) -> list[MatchRecord]:
    shortlist = list(candidates[:MAX_DONOR_RECORDS_PER_SEED_SHORTLIST])
    if not shortlist:
        return []
    index = _stable_hash_int(
        PROTOCOL_ID,
        "DONOR_RECORD",
        condition,
        int(seed),
        target.meta.sample_id,
    ) % len(shortlist)
    return [shortlist[index]]


def _anchor_cost_matrix(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    condition: str,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    donor_state = donor.anchor_speech_state[anchor_indices]
    donor_bins = np.asarray(
        [_speech_state_bin(value) for value in donor_state],
        dtype=np.int64,
    )
    target_bins = np.asarray(
        [_speech_state_bin(value) for value in target.target_speech_state],
        dtype=np.int64,
    )
    state_diff = np.abs(
        donor_state[None, :] - target.target_speech_state[:, None]
    )
    bin_mismatch = (
        donor_bins[None, :] != target_bins[:, None]
    ).astype(np.float64)
    rms_diff = np.abs(
        np.log(
            (
                donor.anchor_rms[anchor_indices][None, :]
                + 1e-12
            )
            / (target.target_rms[:, None] + 1e-12)
        )
    )
    if condition == C6_SPEECH_STATE_MISMATCH:
        valid = (
            bin_mismatch.astype(bool)
            & (state_diff >= C6_MIN_SPEECH_STATE_MISMATCH)
        )
        cost = -state_diff + 0.05 * rms_diff
        return np.where(valid, cost, np.inf)
    return 4.0 * bin_mismatch + state_diff + 0.25 * rms_diff


def _ordered_finite_candidates(
    *,
    cost_row: np.ndarray,
    finite: np.ndarray,
    tie_breaker: Callable[[int], int],
) -> np.ndarray:
    cost_values = np.asarray(cost_row, dtype=np.float64)[finite]
    local_order = np.argsort(cost_values, kind="stable")
    if cost_values.size > 1:
        sorted_costs = cost_values[local_order]
        equal = sorted_costs[1:] == sorted_costs[:-1]
        if not np.any(equal):
            return finite[local_order]
        starts = np.flatnonzero(np.r_[True, ~equal])
        ends = np.r_[starts[1:], sorted_costs.size]
        tie_mask = (ends - starts) > 1
        for start, end in zip(starts[tie_mask], ends[tie_mask]):
            group = local_order[start:end]
            hash_values = np.fromiter(
                (
                    tie_breaker(int(finite[int(local_index)]))
                    for local_index in group
                ),
                dtype=np.uint64,
                count=int(end - start),
            )
            group_order = np.argsort(hash_values, kind="stable")
            local_order[start:end] = group[group_order]
    return finite[local_order]


def _same_utterance_legal_mask(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    anchors = donor.anchor_positions[anchor_indices]
    target_positions = np.asarray(
        target.target_positions,
        dtype=np.int64,
    )
    anchors = np.asarray(anchors, dtype=np.int64)
    taps = np.asarray(REMOTE_WINDOW_TAPS, dtype=np.int64)

    # Away from the initial partial-window frames, two remote windows overlap
    # exactly when their anchor difference is a difference between taps.
    deltas = anchors[None, :] - target_positions[:, None]
    tap_deltas = np.unique(
        (taps[:, None] - taps[None, :]).reshape(-1)
    )
    latest_donor = anchors - int(np.min(taps))
    legal = (
        (anchors[None, :] != target_positions[:, None])
        & (latest_donor[None, :] < target_positions[:, None])
        & ~np.isin(deltas, tap_deltas)
    )

    full_window = (
        (target_positions >= int(np.max(taps)))[:, None]
        & (anchors >= int(np.max(taps)))[None, :]
    )
    for row, column in zip(*np.nonzero(~full_window)):
        target_frame = int(target_positions[row])
        anchor = int(anchors[column])
        original_positions = {
            target_frame - int(tap)
            for tap in taps
            if target_frame >= int(tap)
        }
        donor_positions = {
            anchor - int(tap)
            for tap in taps
            if anchor >= int(tap)
        }
        legal[row, column] = bool(
            anchor != target_frame
            and max(donor_positions) < target_frame
            and not donor_positions.intersection(original_positions)
        )
    return legal


def _choose_anchor_assignments(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    condition: str,
    seed: int,
    anchor_indices: np.ndarray,
    legal_mask: np.ndarray | None = None,
) -> Assignment | None:
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    if anchor_indices.size == 0:
        return None
    cost = _anchor_cost_matrix(
        target=target,
        donor=donor,
        condition=condition,
        anchor_indices=anchor_indices,
    )
    if legal_mask is not None:
        legal_mask = np.asarray(legal_mask, dtype=bool)
        if legal_mask.shape != cost.shape:
            raise ValueError("same-utterance legal mask has wrong shape")
        cost = np.where(legal_mask, cost, np.inf)
    valid_rows: list[int] = []
    chosen_anchors: list[int] = []
    for row in range(cost.shape[0]):
        finite = np.flatnonzero(np.isfinite(cost[row]))
        if finite.size == 0:
            continue
        if condition == C6_SPEECH_STATE_MISMATCH:
            order = finite[
                np.argsort(cost[row, finite], kind="stable")
            ]
        else:
            order = _ordered_finite_candidates(
                cost_row=cost[row],
                finite=finite,
                tie_breaker=lambda index: _stable_hash_int(
                    PROTOCOL_ID,
                    "ANCHOR",
                    condition,
                    int(seed),
                    target.meta.sample_id,
                    int(target.target_positions[row]),
                    int(anchor_indices[index]),
                ),
            )
        shortlist = order[: min(ANCHOR_TOP_K, order.size)]
        pick = shortlist[
            _stable_hash_int(
                PROTOCOL_ID,
                "ANCHOR_PICK",
                condition,
                int(seed),
                target.meta.sample_id,
                int(target.target_positions[row]),
            )
            % shortlist.size
        ]
        valid_rows.append(int(row))
        chosen_anchors.append(int(donor.anchor_positions[pick]))
    if not valid_rows:
        return None
    return Assignment(
        condition=str(condition),
        seed=int(seed),
        target_item_index=target.item_index,
        donor_item_index=donor.item_index,
        target_positions=np.asarray(
            target.target_positions[valid_rows],
            dtype=np.int64,
        ),
        donor_anchors=np.asarray(chosen_anchors, dtype=np.int64),
    )


def build_match_plan(
    *,
    population: e1.Population,
    match_records: Mapping[int, MatchRecord],
) -> tuple[
    dict[int, dict[str, dict[int, Assignment]]],
    list[dict[str, Any]],
]:
    records = list(match_records.values())
    plan: dict[int, dict[str, dict[int, Assignment]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    quality_rows: list[dict[str, Any]] = []
    for target in records:
        # C1 is a same-utterance alternate causal segment.
        for seed in DONOR_SEEDS:
            legal = _same_utterance_legal_mask(
                target=target,
                donor=target,
                anchor_indices=target.anchor_indices,
            )
            assignment = _choose_anchor_assignments(
                target=target,
                donor=target,
                condition=C1_SAME_UTT_DIFFERENT_TIME,
                seed=int(seed),
                anchor_indices=target.anchor_indices,
                legal_mask=legal,
            )
            if assignment is not None:
                plan[target.item_index][
                    C1_SAME_UTT_DIFFERENT_TIME
                ][int(seed)] = assignment
                quality_rows.append(
                    _quality_row(
                        target=target,
                        donor=target,
                        assignment=assignment,
                        condition=C1_SAME_UTT_DIFFERENT_TIME,
                    )
                )

        for condition in (
            C2_SAME_NOISE_INSTANCE,
            C3_SAME_NOISE_CLASS,
            C4_SAME_CLASS_WRONG_SNR,
            C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR,
            C6_SPEECH_STATE_MISMATCH,
            C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
        ):
            first_seed = int(DONOR_SEEDS[0])
            candidates = _ordered_candidates(
                target=target,
                donors=records,
                condition=condition,
                seed=first_seed,
            )
            if not candidates:
                continue
            for seed in DONOR_SEEDS:
                chosen = _seed_shortlist(
                    candidates,
                    target=target,
                    condition=condition,
                    seed=int(seed),
                )
                if not chosen:
                    continue
                donor = chosen[0]
                assignment = _choose_anchor_assignments(
                    target=target,
                    donor=donor,
                    condition=condition,
                    seed=int(seed),
                    anchor_indices=donor.representative_indices,
                )
                if assignment is None:
                    continue
                plan[target.item_index][condition][int(seed)] = assignment
                quality_rows.append(
                    _quality_row(
                        target=target,
                        donor=donor,
                        assignment=assignment,
                        condition=condition,
                    )
                )
    return plan, quality_rows


def _quality_row(
    *,
    target: MatchRecord,
    donor: MatchRecord,
    assignment: Assignment,
    condition: str,
) -> dict[str, Any]:
    target_lookup = {
        int(frame_index): row
        for row, frame_index in enumerate(target.target_positions)
    }
    target_rows = np.asarray(
        [target_lookup[int(frame_index)] for frame_index in assignment.target_positions],
        dtype=np.int64,
    )
    donor_anchor_rows = np.searchsorted(
        donor.anchor_positions,
        assignment.donor_anchors,
    )
    if not np.array_equal(
        donor.anchor_positions[donor_anchor_rows],
        assignment.donor_anchors,
    ):
        raise ValueError("assignment contains a non-anchor donor position")
    state_diffs = np.abs(
        target.target_speech_state[target_rows]
        - donor.anchor_speech_state[donor_anchor_rows]
    )
    rms_diffs = np.abs(
        np.log(
            (donor.anchor_rms[donor_anchor_rows] + 1e-12)
            / (target.target_rms[target_rows] + 1e-12)
        )
    )

    intended_class_match = condition in {
        C2_SAME_NOISE_INSTANCE,
        C3_SAME_NOISE_CLASS,
        C4_SAME_CLASS_WRONG_SNR,
        C6_SPEECH_STATE_MISMATCH,
    }
    return {
        "condition": str(condition),
        "seed": int(assignment.seed),
        "target_item_index": int(target.item_index),
        "target_sample_id": target.meta.sample_id,
        "target_source_key": target.source_key,
        "target_speaker_id": target.speaker_id,
        "target_noise_name": target.noise_name,
        "target_snr_db": float(target.meta.snr_db),
        "target_noise_instance": str(target.meta.noise_instance),
        "donor_item_index": int(donor.item_index),
        "donor_sample_id": donor.meta.sample_id,
        "donor_source_key": donor.source_key,
        "donor_speaker_id": donor.speaker_id,
        "donor_noise_name": donor.noise_name,
        "donor_snr_db": float(donor.meta.snr_db),
        "donor_noise_instance": str(donor.meta.noise_instance),
        "matched_frames": int(assignment.frames),
        "snr_abs_difference": abs(
            float(target.meta.snr_db) - float(donor.meta.snr_db)
        ),
        "rms_abs_log_difference_mean": float(np.mean(rms_diffs)),
        "speech_state_abs_difference_mean": float(np.mean(state_diffs)),
        "noise_class_match": bool(donor.noise_name == target.noise_name),
        "noise_instance_match": bool(
            donor.meta.noise_instance == target.meta.noise_instance
        ),
        "source_overlap": bool(donor.source_key == target.source_key),
        "speaker_overlap": bool(donor.speaker_id == target.speaker_id),
        "different_utterance": bool(
            donor.item_index != target.item_index
        ),
        "same_utterance": bool(
            donor.item_index == target.item_index
        ),
        "intended_class_match": bool(
            donor.noise_name == target.noise_name
            if intended_class_match
            else donor.noise_name != target.noise_name
            if condition == C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR
            else True
        ),
        # Every row describes one target-to-donor assignment. Condition-level
        # uniqueness is reported separately by the feasibility table.
        "donor_instance_count": 1,
        "single_donor_assignment": True,
    }


def _population_counts(population: e1.Population) -> dict[str, int]:
    test_mask = np.asarray(population.test_mask, dtype=bool)
    noisy_test = test_mask & (population.condition != "clean")
    return {
        "frames": int(population.labels.size),
        "test_frames": int(np.count_nonzero(test_mask)),
        "test_sources": int(
            len(set(np.asarray(population.source_key)[test_mask]))
        ),
        "test_speakers": int(
            len(set(np.asarray(population.speaker_ids)[test_mask]))
        ),
        "noisy_test_frames": int(np.count_nonzero(noisy_test)),
        "noisy_test_sources": int(
            len(set(np.asarray(population.source_key)[noisy_test]))
        ),
        "noisy_test_speakers": int(
            len(set(np.asarray(population.speaker_ids)[noisy_test]))
        ),
        "noisy_test_records": int(
            len(_noisy_test_records(population))
        ),
    }


def _validate_population(population: e1.Population) -> dict[str, int]:
    counts = _population_counts(population)
    expected = {
        "frames": E1_EXPECTED_TOTAL_FRAMES,
        "test_frames": E1_EXPECTED_TEST_FRAMES,
        "test_sources": E2_EXPECTED_NOISY_TEST_SOURCES,
        "test_speakers": E2_EXPECTED_NOISY_TEST_SPEAKERS,
        "noisy_test_frames": E2_EXPECTED_NOISY_TEST_FRAMES,
        "noisy_test_sources": E2_EXPECTED_NOISY_TEST_SOURCES,
        "noisy_test_speakers": E2_EXPECTED_NOISY_TEST_SPEAKERS,
        "noisy_test_records": E2_EXPECTED_NOISY_TEST_RECORDS,
    }
    if counts != expected:
        raise ValueError(
            f"E2 population mismatch: {counts}, expected {expected}"
        )
    return counts


def _e2_test_global_indices(population: e1.Population) -> np.ndarray:
    indices: list[np.ndarray] = []
    for record in _noisy_test_records(population):
        local = e1._record_test_positions(record, population.test_mask)
        indices.append(
            (record.start + local - VALID_START).astype(np.int64)
        )
    result = np.concatenate(indices)
    if result.size != E2_EXPECTED_NOISY_TEST_FRAMES:
        raise ValueError(
            f"E2 test index list has {result.size} frames, expected "
            f"{E2_EXPECTED_NOISY_TEST_FRAMES}"
        )
    return result


def _assignment_count(plan: Mapping[int, Mapping[str, Mapping[int, Assignment]]]) -> int:
    return sum(
        len(seed_map)
        for target in plan.values()
        for seed_map in target.values()
    )


def _feasibility_row(
    *,
    condition: str,
    target_count: int,
    eligible_target_frames: int,
    eligible_sources: set[str],
    quality: pd.DataFrame,
) -> dict[str, Any]:
    if quality.empty:
        return {
            "condition": condition,
            "eligible_target_count": int(target_count),
            "eligible_target_frames": int(eligible_target_frames),
            "eligible_source_count": int(len(eligible_sources)),
            "matched_target_count": 0,
            "matched_frames": 0,
            "matched_frame_coverage": 0.0,
            "unmatched_target_count": int(target_count),
            "assignment_count": 0,
            "donor_item_count": 0,
            "donor_source_count": 0,
            "unique_donor_instance_count": 0,
            "match_success_rate": 0.0,
            "top_donor_assignment_share": 1.0,
            "snr_abs_difference_mean": float("nan"),
            "snr_abs_difference_max": float("nan"),
            "rms_abs_log_difference_mean": float("nan"),
            "speech_state_abs_difference_mean": float("nan"),
            "noise_class_match_rate": 0.0,
            "noise_instance_match_rate": 0.0,
            "source_overlap_rate": 0.0,
            "speaker_overlap_rate": 0.0,
            "different_utterance_rate": 0.0,
            "feasibility_status": "INCOMPARABLE",
            "feasibility_reason": "zero feasible matches",
        }
    weights = np.asarray(quality["matched_frames"], dtype=np.float64)
    weight_sum = float(np.sum(weights))

    def weighted(column: str, *, boolean: bool = False) -> float:
        values = quality[column].to_numpy(dtype=np.float64)
        if boolean:
            values = values.astype(np.float64)
        return float(np.sum(values * weights) / weight_sum)

    targets = set(quality["target_item_index"].astype(int))
    target_sources = set(
        quality["target_source_key"].astype(str)
    )
    donors = quality["donor_item_index"].astype(int)
    donor_counts = donors.value_counts()
    top_donor_share = (
        float(donor_counts.iloc[0]) / float(len(quality))
        if len(quality)
        else 1.0
    )
    donor_sources = set(quality["donor_source_key"].astype(str))
    donor_instances = set(
        quality["donor_noise_instance"].astype(str)
    )
    match_success_rate = float(len(targets)) / float(target_count)
    matched_frame_coverage = (
        float(weight_sum) / float(eligible_target_frames)
        if eligible_target_frames
        else 0.0
    )
    class_match = weighted("noise_class_match", boolean=True)
    instance_match = weighted("noise_instance_match", boolean=True)
    source_overlap = weighted("source_overlap", boolean=True)
    speaker_overlap = weighted("speaker_overlap", boolean=True)
    different_utterance = weighted(
        "different_utterance",
        boolean=True,
    )
    snr_mean = weighted("snr_abs_difference")
    snr_max = float(quality["snr_abs_difference"].max())
    reasons: list[str] = []
    if match_success_rate < MIN_MATCH_SUCCESS_RATE:
        reasons.append("match success below 0.80")
    if len(target_sources) < MIN_PRIMARY_SOURCES:
        reasons.append("target source diversity below 20")
    if condition not in {
        C1_SAME_UTT_DIFFERENT_TIME,
        C2_SAME_NOISE_INSTANCE,
    }:
        if len(donor_sources) < MIN_PRIMARY_SOURCES:
            reasons.append("donor source diversity below 20")
        if top_donor_share > MAX_TOP_DONOR_SHARE:
            reasons.append("donor concentration above 0.20")
    if condition in {
        C3_SAME_NOISE_CLASS,
        C4_SAME_CLASS_WRONG_SNR,
        C6_SPEECH_STATE_MISMATCH,
        C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
    }:
        if class_match < 0.995:
            reasons.append("intended noise-class match rate below 0.995")
    if condition == C2_SAME_NOISE_INSTANCE and instance_match < 0.995:
        reasons.append("intended noise-instance match rate below 0.995")
    if condition in {
        C3_SAME_NOISE_CLASS,
        C6_SPEECH_STATE_MISMATCH,
    }:
        if snr_max > 1e-9:
            reasons.append("intended same-SNR matching failed")
    if condition == C4_SAME_CLASS_WRONG_SNR and snr_mean < 10.0:
        reasons.append("mean SNR mismatch below 10 dB")
    if condition == C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR and class_match > 0.005:
        reasons.append("different-class separation failed")
    if condition == C6_SPEECH_STATE_MISMATCH:
        state_mismatch = weighted("speech_state_abs_difference_mean")
        if state_mismatch < C6_MIN_SPEECH_STATE_MISMATCH:
            reasons.append("speech-state mismatch below frozen minimum")
    if condition not in {
        C1_SAME_UTT_DIFFERENT_TIME,
        C2_SAME_NOISE_INSTANCE,
    }:
        if source_overlap > 0.005:
            reasons.append("source separation failed")
    if condition != C1_SAME_UTT_DIFFERENT_TIME:
        if different_utterance < 0.995:
            reasons.append("different-utterance matching failed")
    status = "PRIMARY_INTERPRETABLE" if not reasons else "INCOMPARABLE"
    return {
        "condition": condition,
        "eligible_target_count": int(target_count),
        "eligible_target_frames": int(eligible_target_frames),
        "eligible_source_count": int(len(eligible_sources)),
        "matched_target_count": int(len(targets)),
        "matched_frames": int(weight_sum),
        "matched_frame_coverage": matched_frame_coverage,
        "unmatched_target_count": int(target_count - len(targets)),
        "assignment_count": int(len(quality)),
        "donor_item_count": int(donors.nunique()),
        "donor_source_count": int(len(donor_sources)),
        "unique_donor_instance_count": int(len(donor_instances)),
        "match_success_rate": match_success_rate,
        "top_donor_assignment_share": top_donor_share,
        "snr_abs_difference_mean": snr_mean,
        "snr_abs_difference_max": snr_max,
        "rms_abs_log_difference_mean": weighted(
            "rms_abs_log_difference_mean"
        ),
        "speech_state_abs_difference_mean": weighted(
            "speech_state_abs_difference_mean"
        ),
        "noise_class_match_rate": class_match,
        "noise_instance_match_rate": instance_match,
        "source_overlap_rate": source_overlap,
        "speaker_overlap_rate": speaker_overlap,
        "different_utterance_rate": different_utterance,
        "feasibility_status": status,
        "feasibility_reason": "; ".join(reasons),
    }


def build_feasibility(
    *,
    match_records: Mapping[int, MatchRecord],
    quality_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    quality = pd.DataFrame(quality_rows)
    if not quality.empty:
        quality["condition"] = quality["condition"].astype(str)
    target_count = len(match_records)
    eligible_sources = {
        str(record.source_key) for record in match_records.values()
    }
    eligible_target_frames = int(
        sum(record.target_positions.size for record in match_records.values())
    )
    rows = [
        _feasibility_row(
            condition=condition,
            target_count=target_count,
            eligible_target_frames=eligible_target_frames,
            eligible_sources=eligible_sources,
            quality=(
                quality.loc[quality["condition"] == condition].copy()
                if not quality.empty
                else pd.DataFrame()
            ),
        )
        for condition in REPLACEMENT_CONDITIONS
    ]
    frame = pd.DataFrame(rows)
    frame["condition_order"] = frame["condition"].map(
        {condition: index for index, condition in enumerate(PRIMARY_CONDITIONS)}
    )
    frame = frame.sort_values("condition_order").drop(
        columns=["condition_order"]
    )
    return frame.reset_index(drop=True)


def _feasibility_status_map(
    feasibility: pd.DataFrame,
) -> dict[str, str]:
    return {
        str(row["condition"]): str(row["feasibility_status"])
        for _, row in feasibility.iterrows()
    }


def _outcome_stage_paths() -> tuple[Path, ...]:
    return (
        BASELINE_PATH,
        E1_C4_AUDIT_PATH,
        FRAME_RESULTS_PATH,
        CONDITION_RESULTS_PATH,
        PRIMARY_CONTRASTS_PATH,
        CLUSTER_BOOTSTRAP_PATH,
        CORRECTION_SURVIVAL_PATH,
        TAXONOMY_TRANSITIONS_PATH,
        EVENT_STRATA_PATH,
        ONSET_OFFSET_PATH,
        SEEN_UNSEEN_PATH,
        SOURCE_INFLUENCE_PATH,
        DONOR_SEED_CONSISTENCY_PATH,
        E1_RECONCILIATION_PATH,
        FINAL_SUMMARY_PATH,
        FINAL_REPORT_PATH,
        CLAIM_FREEZE_PATH,
        FIGURES_ROOT,
    )


def _assert_metadata_stage_before_freeze(*, mode: str) -> None:
    frozen = [
        path
        for path in (PROTOCOL_PATH, PROTOCOL_HASH_PATH, RUNNER_HASH_PATH)
        if path.exists()
    ]
    if frozen:
        raise RuntimeError(
            f"{mode} must precede protocol freeze; found "
            + ", ".join(str(path) for path in frozen)
        )
    outcomes = [path for path in _outcome_stage_paths() if path.exists()]
    if outcomes:
        raise RuntimeError(
            f"{mode} must precede outcome scoring; found "
            + ", ".join(str(path) for path in outcomes)
        )


def _metadata_feasibility_snapshot() -> dict[str, Any]:
    missing = [
        str(path)
        for path in (FEASIBILITY_PATH, MATCH_QUALITY_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "metadata-only feasibility must complete before protocol "
            f"freeze; missing: {missing}"
        )
    feasibility = pd.read_csv(FEASIBILITY_PATH)
    required_columns = {"condition", "feasibility_status"}
    missing_columns = sorted(required_columns - set(feasibility.columns))
    if missing_columns:
        raise ValueError(
            "matching feasibility table is missing columns: "
            f"{missing_columns}"
        )
    if feasibility["condition"].astype(str).duplicated().any():
        raise ValueError("matching feasibility contains duplicate conditions")
    statuses = _feasibility_status_map(feasibility)
    expected_conditions = set(REPLACEMENT_CONDITIONS)
    if set(statuses) != expected_conditions:
        raise ValueError(
            "matching feasibility conditions do not match E2 conditions: "
            f"{sorted(statuses)}"
        )
    allowed_statuses = {"PRIMARY_INTERPRETABLE", "INCOMPARABLE"}
    invalid = {
        condition: status
        for condition, status in statuses.items()
        if status not in allowed_statuses
    }
    if invalid:
        raise ValueError(
            f"matching feasibility has invalid statuses: {invalid}"
        )
    return {
        "stage": "METADATA_ONLY_FEASIBILITY_COMPLETE",
        "outcome_inspection_during_feasibility": False,
        "condition_statuses": {
            condition: statuses[condition]
            for condition in REPLACEMENT_CONDITIONS
        },
        "artifacts": [
            _file_record(FEASIBILITY_PATH),
            _file_record(MATCH_QUALITY_PATH),
        ],
    }


def _protocol_payload(
    *,
    feasibility_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_inputs = {
        name: _file_record(path)
        for name, path in FROZEN_INPUT_PATHS.items()
    }
    frozen_sources = {
        str(path).replace("\\", "/"): _file_record(REPO_ROOT / path)
        for path in FROZEN_SOURCE_PATHS
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "runner_module": RUNNER_MODULE,
        "execution_order": [
            "metadata_only_feasibility",
            "protocol_freeze",
            "c1_full_reproduction_gate",
            "e1_c4_reference_audit",
            "c0_c7_outcome_scoring_once",
            "verify",
        ],
        "research_boundary": {
            "training_performed": False,
            "threshold_tuning": False,
            "router_or_architecture_search": False,
            "new_final_ood_touched": False,
            "e1_artifacts_may_be_modified": False,
            "e3_or_e4_started": False,
            "tiny_gru_interventions": False,
        },
        "primary_question": (
            "Why does replacing the original remote context strongly damage "
            "RF384 refinement while permuting the original remote context "
            "does not?"
        ),
        "population": {
            "reconstruction": (
                "build_evaluation_items(row_sample=1080, seed=17, "
                "include_clean=True)"
            ),
            "expected_records": 1_024,
            "expected_frames": E1_EXPECTED_TOTAL_FRAMES,
            "expected_test_frames": E1_EXPECTED_TEST_FRAMES,
            "expected_noisy_test_records": E2_EXPECTED_NOISY_TEST_RECORDS,
            "expected_noisy_test_frames": E2_EXPECTED_NOISY_TEST_FRAMES,
            "expected_noisy_test_sources": E2_EXPECTED_NOISY_TEST_SOURCES,
            "expected_noisy_test_speakers": E2_EXPECTED_NOISY_TEST_SPEAKERS,
            "valid_start": int(VALID_START),
            "decision_threshold": float(DECISION_THRESHOLD),
            "uncertainty_threshold": float(UNCERTAINTY_THRESHOLD),
        },
        "frozen_inputs": frozen_inputs,
        "frozen_source_files": frozen_sources,
        "metadata_feasibility": dict(feasibility_snapshot),
        "baseline_gate": {
            "condition": C0_FULL,
            "population": "all E2 noisy-test frames",
            "max_abs_full_error_max": BASELINE_TOLERANCE,
            "max_abs_embedded_short_error_max": BASELINE_TOLERANCE,
            "stop_if_failed": True,
            "e1_c4_replay_required_before_interventions": True,
            "e1_c4_reference_delta_logloss": E1_C4_REFERENCE_DELTA,
            "e1_c4_audit_tolerance": E1_C4_AUDIT_TOLERANCE,
        },
        "remote_definition": {
            "remote_taps": list(REMOTE_TAPS),
            "remote_window_order": list(REMOTE_WINDOW_TAPS),
            "local_and_current_preserved": True,
            "future_information_used": False,
        },
        "matching": {
            "noise_identity": "(noise_source, noise_start_sample)",
            "speech_state_bins": [
                list(bounds) for bounds in SPEECH_STATE_BINS
            ],
            "speech_state_names": list(SPEECH_STATE_NAMES),
            "candidate_pool": "E2 noisy test records only",
            "donor_selection_uses_model_outcomes": False,
            "max_donor_records_per_seed_shortlist": (
                MAX_DONOR_RECORDS_PER_SEED_SHORTLIST
            ),
            "max_donor_anchors": MAX_DONOR_ANCHORS,
            "anchor_top_k": ANCHOR_TOP_K,
            "c1_rule": (
                "same utterance, different causal anchor, remote vector "
                "positions disjoint from the original remote positions, "
                "and donor causal window strictly before target frame"
            ),
            "c2_rule": (
                "same noise instance and same SNR; different utterance"
            ),
            "c3_rule": (
                "same noise class and same SNR; different noise instance, "
                "utterance, and source"
            ),
            "c4_rule": (
                "same noise class; different noise instance, utterance, "
                "and source; absolute SNR difference at least 10 dB"
            ),
            "c5_rule": (
                "different noise class; same SNR; different utterance and "
                "source; RMS/state costs minimized"
            ),
            "c6_rule": (
                "same noise class and SNR; different utterance and source; "
                "remote speech-state mismatch at least 0.25"
            ),
            "c7_rule": (
                "same noise class and SNR; different source, noise "
                "instance, and utterance; simultaneous cost matching for "
                "speech-state composition, RMS, and duration"
            ),
            "b4_event_pairing": EVENT_MATCH_RULES,
            "b4_stable_event_distance_clearance_frames": (
                EVENT_DISTANCE_WINDOW
            ),
            "b4_stable_run_minimum_frames": STABLE_RUN_MIN,
            "event_strata_provenance": EVENT_STRATA_PROVENANCE,
            "b4_matching_variable": (
                "Short uncertainty abs(short_score - 0.5), with frame "
                "index only as a deterministic tie break"
            ),
            "pre_outcome_implementation_corrections": [
                (
                    "Remote speech-state features now index full-history "
                    "labels at the original causal frame positions."
                ),
                (
                    "C7 now requires same noise class and SNR as hard "
                    "candidate constraints before deterministic metadata "
                    "cost matching."
                ),
            ],
            "donor_seeds": list(DONOR_SEEDS),
        },
        "feasibility_gate": {
            "minimum_match_success_rate": MIN_MATCH_SUCCESS_RATE,
            "minimum_primary_sources": MIN_PRIMARY_SOURCES,
            "maximum_top_donor_assignment_share": MAX_TOP_DONOR_SHARE,
            "c6_minimum_speech_state_mismatch": (
                C6_MIN_SPEECH_STATE_MISMATCH
            ),
            "condition_failure_status": "INCOMPARABLE",
            "outcome_inspection_during_feasibility": False,
        },
        "scoring": {
            "encoder_chunk_frames": 2_000,
            "classifier_chunk_frames": 2_000,
            "primary_endpoint": "delta log-loss relative to C0 FULL",
            "secondary_endpoint": "delta Brier relative to C0 FULL",
            "donor_realizations_averaged_per_frame_before_bootstrap": True,
            "donor_draws_treated_as_independent_frames": False,
        },
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": SOURCE_BOOTSTRAP_SEED,
            "speaker_seed_offset": SPEAKER_BOOTSTRAP_OFFSET,
            "primary_cluster": "source",
            "confidence_interval": 0.95,
        },
        "conditions": {
            condition: {
                "short_name": CONDITION_SHORT[condition],
                "role": CONDITION_ROLE[condition],
            }
            for condition in CONDITION_ORDER
        },
        "primary_contrasts": [
            {
                "name": name,
                "left": left,
                "right": right,
                "definition": definition,
            }
            for name, left, right, definition in PRIMARY_CONTRASTS
        ],
        "event_strata": EVENT_STRATA,
        "event_rules": {
            "onset_near": "ground-truth speech, onset distance 0-5",
            "transition_near": "Short posterior switch age 0-5",
            "offset_near": "post-speech non-speech, offset distance 0-5",
            "stable_speech": "ground-truth speech run length at least 21",
            "stable_silence": "ground-truth silence run length at least 21",
        },
        "event_matching": EVENT_MATCH_RULES,
        "decision_rule": {
            "material_contrast_delta_logloss": MATERIAL_CONTRAST_DELTA,
            "B1": (
                "P1 C7-C1 at least 0.005 with source CI lower bound > 0"
            ),
            "B2": (
                "P3 C5-C3 or P4 C4-C3 at least 0.005 with source "
                "CI lower bound > 0"
            ),
            "B3": (
                "speech-state contrast P5 does not explain at least 60% "
                "of positive P6 residual damage"
            ),
            "B4": (
                "C7 onset or transition excess over its matched stable "
                "stratum has source CI lower bound > 0 and at least 0.005"
            ),
            "B5": (
                "relevant LOSO signs are consistent and positive donor-seed "
                "count is at least four"
            ),
            "supported": "all B1-B5 pass",
            "conditional": (
                "background-related point estimates are material but one or "
                "more primary CIs cross zero or source/donor stability fails"
            ),
            "residual": (
                "B1 and B2 fail while P6 C7-FULL is material with CI lower "
                "bound > 0"
            ),
            "speech_state": (
                "P5 is material with CI lower bound > 0 and explains at "
                "least 60% of positive P6"
            ),
            "unresolved": (
                "no mechanism separates the replacement conditions"
            ),
            "invalid": "matching, baseline, or intervention validity fails",
        },
        "required_figures": list(FIGURE_FILES),
        "stop_rule": {
            "next_experiment_authorized": False,
            "final_response": "exactly one allowed E2 status token",
        },
    }


def _freeze_protocol() -> int:
    _assert_metadata_stage_before_freeze(mode="protocol freeze")
    feasibility_snapshot = _metadata_feasibility_snapshot()
    payload = _protocol_payload(
        feasibility_snapshot=feasibility_snapshot,
    )
    _write_json(PROTOCOL_PATH, payload)
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    PROTOCOL_HASH_PATH.write_text(
        protocol_hash + "\n",
        encoding="ascii",
    )
    runner_hash = _sha256_file(Path(__file__))
    RUNNER_HASH_PATH.write_text(
        runner_hash + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "mode": "freeze",
                "protocol_sha256": protocol_hash,
                "runner_sha256": runner_hash,
                "output": str(PROTOCOL_PATH),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _validate_frozen_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("E2 protocol has not been frozen")
    if not PROTOCOL_HASH_PATH.exists() or not RUNNER_HASH_PATH.exists():
        raise FileNotFoundError("E2 hash freeze files are missing")
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    recorded_protocol_hash = PROTOCOL_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    if protocol_hash != recorded_protocol_hash:
        raise ValueError("E2 protocol hash mismatch")
    runner_hash = _sha256_file(Path(__file__))
    recorded_runner_hash = RUNNER_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    if runner_hash != recorded_runner_hash:
        raise ValueError("E2 runner changed after protocol freeze")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong E2 protocol identifier")
    if payload.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError("wrong E2 protocol status")
    if _sha256_file(E1_RUNNER_PATH) != EXPECTED_E1_RUNNER_SHA256:
        raise ValueError("frozen E1 runner hash mismatch")
    if _sha256_file(E1_PROTOCOL_PATH) != EXPECTED_E1_PROTOCOL_SHA256:
        raise ValueError("frozen E1 protocol hash mismatch")
    metadata = payload.get("metadata_feasibility")
    if not isinstance(metadata, Mapping):
        raise ValueError("E2 protocol lacks metadata feasibility snapshot")
    if metadata.get("outcome_inspection_during_feasibility") is not False:
        raise ValueError("metadata feasibility outcome-isolation flag is invalid")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("E2 protocol feasibility artifacts are invalid")
    expected_paths = {
        str(FEASIBILITY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        str(MATCH_QUALITY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    recorded_paths = {str(item.get("path")) for item in artifacts}
    if recorded_paths != expected_paths:
        raise ValueError("E2 protocol feasibility artifact paths are invalid")
    for item in artifacts:
        path = REPO_ROOT / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"frozen metadata feasibility artifact is missing: {path}"
            )
        if _sha256_file(path) != str(item["sha256"]):
            raise ValueError(
                f"metadata feasibility artifact hash mismatch: {path}"
            )
        if int(item["size_bytes"]) != int(path.stat().st_size):
            raise ValueError(
                f"metadata feasibility artifact size mismatch: {path}"
            )
    return payload


def _e1_artifact_snapshot() -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _file_record(
            path
        )
        for path in sorted(E1_RESULTS_ROOT.rglob("*"))
        if path.is_file()
    }


def _encode_records(
    *,
    population: e1.Population,
    item_indices: Iterable[int],
    model: Any,
    frontend: Any,
    device: torch.device,
    chunk_frames: int,
) -> dict[int, torch.Tensor]:
    records_by_item = {
        int(record.item_index): record for record in population.records
    }
    result: dict[int, torch.Tensor] = {}
    ordered = sorted({int(value) for value in item_indices})
    started = time.time()
    for position, item_index in enumerate(ordered, start=1):
        record = records_by_item[int(item_index)]
        waveform = e1.read_int16_audio(record.item.audio_path)
        encoded = e1._encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=int(chunk_frames),
        )
        result[int(item_index)] = encoded.detach().cpu()
        if position % 25 == 0 or position == len(ordered):
            print(
                f"encoded record {position}/{len(ordered)} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
    return result


def _score_base(
    *,
    record: e1.Record,
    encoded: torch.Tensor,
    short_logits: torch.Tensor,
    model: Any,
    target_positions: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    targets = torch.as_tensor(
        np.asarray(target_positions, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    branches = list(model.refinement.branches)
    local_sum = None
    for branch in branches[:-1]:
        windows = e1._gather_branch_windows(encoded, targets, branch)
        contribution = e1._branch_contribution(windows, branch)
        local_sum = (
            contribution
            if local_sum is None
            else local_sum + contribution
        )
    if local_sum is None:
        raise RuntimeError("RF384 has no local refinement branches")
    remote_windows = e1._gather_branch_windows(
        encoded,
        targets,
        branches[-1],
    )
    full = e1._score_branch_sum(
        local_sum
        + e1._branch_contribution(remote_windows, branches[-1]),
        short_logits,
        targets,
        model,
    )
    embedded = torch.softmax(
        short_logits[0, :, targets].transpose(0, 1),
        dim=-1,
    )[:, 1]
    return (
        full.detach().cpu().numpy().astype(np.float64),
        embedded.detach().cpu().numpy().astype(np.float64),
    )


def _run_baseline_mode(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    _validate_frozen_protocol()
    population = e1._load_population()
    counts = _validate_population(population)
    model, frontend, _config = e1._load_model(device)
    rows: list[dict[str, Any]] = []
    global_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    short_parts: list[np.ndarray] = []
    max_full = 0.0
    max_short = 0.0
    started = time.time()
    target_records = _noisy_test_records(population)
    for position, record in enumerate(target_records, start=1):
        targets = e1._record_test_positions(record, population.test_mask)
        waveform = e1.read_int16_audio(record.item.audio_path)
        encoded = e1._encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=int(args.chunk_frames),
        )
        short_logits = e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=int(args.chunk_frames),
        )
        full, embedded = _score_base(
            record=record,
            encoded=encoded,
            short_logits=short_logits,
            model=model,
            target_positions=targets,
            device=device,
        )
        offset = targets - VALID_START
        expected_full = record.frozen_full_scores[offset]
        expected_short = record.short_scores[offset]
        full_error = _max_abs_error(full, expected_full)
        short_error = _max_abs_error(embedded, expected_short)
        max_full = max(max_full, full_error)
        max_short = max(max_short, short_error)
        global_parts.append(
            (record.start + offset).astype(np.int64)
        )
        full_parts.append(full)
        short_parts.append(embedded)
        rows.append(
            {
                "item_index": int(record.item_index),
                "sample_id": record.item.sample_id,
                "frames": int(targets.size),
                "max_abs_full_error": full_error,
                "max_abs_embedded_short_error": short_error,
            }
        )
        if position % 25 == 0 or position == len(target_records):
            print(
                f"baseline record {position}/{len(target_records)} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
    global_index = np.concatenate(global_parts)
    full_scores = np.concatenate(full_parts)
    short_scores = np.concatenate(short_parts)
    expected_global = _e2_test_global_indices(population)
    if not np.array_equal(global_index, expected_global):
        raise ValueError("baseline cache frame order does not match E2 mask")
    passed = bool(
        max_full <= BASELINE_TOLERANCE
        and max_short <= BASELINE_TOLERANCE
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_sha256": _sha256_file(Path(__file__)),
        "completed_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "population": counts,
        "frames": int(global_index.size),
        "records": len(rows),
        "max_abs_full_error": float(max_full),
        "max_abs_embedded_short_error": float(max_short),
        "tolerance": BASELINE_TOLERANCE,
        "passed": passed,
        "elapsed_seconds": float(time.time() - started),
        "record_rows": rows,
        "training_performed": False,
        "new_final_ood_touched": False,
    }
    _write_json(BASELINE_PATH, payload)
    if not passed:
        raise RuntimeError(
            "E2 C0 FULL baseline gate failed: "
            f"full={max_full}, short={max_short}, "
            f"tolerance={BASELINE_TOLERANCE}"
        )
    np.savez_compressed(
        BASELINE_SCORE_CACHE_PATH,
        global_index=global_index,
        full_scores=full_scores,
        short_scores=short_scores,
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _load_baseline_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not BASELINE_PATH.exists() or not BASELINE_SCORE_CACHE_PATH.exists():
        raise FileNotFoundError("E2 baseline gate has not been run")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if not bool(baseline.get("passed")):
        raise ValueError("E2 baseline gate did not pass")
    if baseline.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("baseline cache belongs to a different E2 protocol")
    if baseline.get("runner_sha256") != _sha256_file(Path(__file__)):
        raise ValueError("baseline cache belongs to a different E2 runner")
    with np.load(BASELINE_SCORE_CACHE_PATH, allow_pickle=False) as payload:
        return (
            np.asarray(payload["global_index"], dtype=np.int64),
            np.asarray(payload["full_scores"], dtype=np.float64),
            np.asarray(payload["short_scores"], dtype=np.float64),
        )


def _seed_column(condition: str, seed: int) -> str:
    return f"score_{CONDITION_SHORT[condition]}_seed{int(seed)}"


def _score_column(condition: str) -> str:
    return f"score_{CONDITION_SHORT[condition]}"


def _delta_logloss_column(condition: str) -> str:
    return f"delta_logloss_{CONDITION_SHORT[condition]}"


def _delta_brier_column(condition: str) -> str:
    return f"delta_brier_{CONDITION_SHORT[condition]}"


def _taxonomy_column(condition: str) -> str:
    return f"taxonomy_{CONDITION_SHORT[condition]}"


def _delta_logloss_seed_column(condition: str, seed: int) -> str:
    return f"delta_logloss_{CONDITION_SHORT[condition]}_seed{int(seed)}"


def _local_branch_sum(
    *,
    encoded: torch.Tensor,
    targets: torch.Tensor,
    model: Any,
) -> torch.Tensor:
    value = None
    for branch in list(model.refinement.branches)[:-1]:
        windows = e1._gather_branch_windows(encoded, targets, branch)
        contribution = e1._branch_contribution(windows, branch)
        value = (
            contribution
            if value is None
            else value + contribution
        )
    if value is None:
        raise RuntimeError("RF384 has no local branches")
    return value


def _score_replaced_positions(
    *,
    encoded: torch.Tensor,
    short_logits: torch.Tensor,
    model: Any,
    target_positions: np.ndarray,
    donor_encoded: torch.Tensor,
    donor_anchors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    targets_np = np.asarray(target_positions, dtype=np.int64)
    targets = torch.as_tensor(
        targets_np,
        dtype=torch.long,
        device=device,
    )
    branch = list(model.refinement.branches)[-1]
    local_sum = _local_branch_sum(
        encoded=encoded,
        targets=targets,
        model=model,
    )
    target_windows = e1._gather_branch_windows(
        encoded,
        targets,
        branch,
    )
    donor_targets = torch.as_tensor(
        np.asarray(donor_anchors, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    donor_windows = e1._gather_branch_windows(
        donor_encoded,
        donor_targets,
        branch,
    )
    modified = e1._replace_remote_windows(
        target_windows,
        donor_windows,
        targets_np,
    )
    score = e1._score_branch_sum(
        local_sum + e1._branch_contribution(modified, branch),
        short_logits,
        targets,
        model,
    )
    return score.detach().cpu().numpy().astype(np.float64)


def _score_match_record(
    *,
    target: MatchRecord,
    encoded: torch.Tensor,
    short_logits: torch.Tensor,
    model: Any,
    device: torch.device,
    donor_encoded_cpu: Mapping[int, torch.Tensor],
    target_positions: np.ndarray,
    assignments: Mapping[str, Mapping[int, Assignment]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[int, dict[str, np.ndarray]]],
]:
    base_full, base_short = _score_base(
        record=target.record,
        encoded=encoded,
        short_logits=short_logits,
        model=model,
        target_positions=target_positions,
        device=device,
    )
    seed_scores: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for condition in REPLACEMENT_CONDITIONS:
        condition_assignments = assignments.get(condition, {})
        condition_scores: dict[int, dict[str, np.ndarray]] = {}
        for seed in DONOR_SEEDS:
            assignment = condition_assignments.get(int(seed))
            if assignment is None:
                continue
            donor = donor_encoded_cpu[int(assignment.donor_item_index)].to(
                device
            )
            positions = np.asarray(
                assignment.target_positions,
                dtype=np.int64,
            )
            condition_scores[int(seed)] = {
                "positions": positions,
                "scores": _score_replaced_positions(
                    encoded=encoded,
                    short_logits=short_logits,
                    model=model,
                    target_positions=positions,
                    donor_encoded=donor,
                    donor_anchors=np.asarray(
                        assignment.donor_anchors,
                        dtype=np.int64,
                    ),
                    device=device,
                ),
            }
        seed_scores[condition] = condition_scores
    return (
        {
            "full": base_full,
            "embedded_short": base_short,
        },
        seed_scores,
    )


def _frame_row_for_match_record(
    *,
    population: e1.Population,
    target: MatchRecord,
    target_positions: np.ndarray,
    base: Mapping[str, np.ndarray],
    seed_scores: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    baseline_lookup: Mapping[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    record = target.record
    targets = np.asarray(target_positions, dtype=np.int64)
    offset = targets - VALID_START
    labels = record.labels[offset]
    short_scores = record.short_scores[offset]
    full_scores = np.asarray(base["full"], dtype=np.float64)
    embedded_short = np.asarray(base["embedded_short"], dtype=np.float64)
    global_index = (record.start + offset).astype(np.int64)
    if _max_abs_error(embedded_short, short_scores) > BASELINE_TOLERANCE:
        raise RuntimeError("embedded Short changed during E2 scoring")
    if _max_abs_error(
        full_scores,
        record.frozen_full_scores[offset],
    ) > BASELINE_TOLERANCE:
        raise RuntimeError("C0 FULL changed during E2 scoring")
    for row_index, (index, full) in enumerate(
        zip(global_index, full_scores)
    ):
        cached_full, cached_short = baseline_lookup[int(index)]
        if (
            abs(float(full) - cached_full) > BASELINE_TOLERANCE
            or abs(float(short_scores[row_index]) - cached_short)
            > BASELINE_TOLERANCE
        ):
            raise RuntimeError("live C0 scores disagree with baseline cache")

    onset_distance = record.event_features["onset_distance"][offset]
    offset_distance = record.event_features["offset_distance"][offset]
    transition_distance = record.event_features[
        "posterior_transition_distance"
    ][offset]
    speech_run = e1._run_lengths(record.labels == 1)[offset].astype(
        np.float64
    )
    silence_run = e1._run_lengths(record.labels == 0)[offset].astype(
        np.float64
    )
    unseen = bool(record.noise_name in e1.UNSEEN_NOISE)
    rows: list[dict[str, Any]] = []
    for row_index, global_value in enumerate(global_index):
        row: dict[str, Any] = {
            "global_index": int(global_value),
            "item_index": int(record.item_index),
            "sample_id": record.item.sample_id,
            "source_key": record.source_key,
            "speaker_id": record.speaker_id,
            "noise_name": record.noise_name,
            "condition": record.condition,
            "seen": bool(not unseen),
            "unseen": bool(unseen),
            "seen_group": "unseen" if unseen else "seen",
            "frame_in_utterance": int(targets[row_index]),
            "label": int(labels[row_index]),
            "short_score": float(short_scores[row_index]),
            "embedded_short_score": float(embedded_short[row_index]),
            "full_score": float(full_scores[row_index]),
            "selected": bool(population.selected[int(global_value)]),
            "uncertainty": float(abs(short_scores[row_index] - 0.5)),
            "onset_distance": float(onset_distance[row_index]),
            "offset_distance": float(offset_distance[row_index]),
            "posterior_transition_distance": float(
                transition_distance[row_index]
            ),
            "speech_run_length": float(speech_run[row_index]),
            "silence_run_length": float(silence_run[row_index]),
            "full_correct": bool(
                e1._correct(
                    np.asarray([labels[row_index]]),
                    np.asarray([full_scores[row_index]]),
                )[0]
            ),
            "short_correct": bool(
                e1._correct(
                    np.asarray([labels[row_index]]),
                    np.asarray([short_scores[row_index]]),
                )[0]
            ),
            "full_taxonomy": str(
                e1._taxonomy(
                    np.asarray([labels[row_index]]),
                    np.asarray([short_scores[row_index]]),
                    np.asarray([full_scores[row_index]]),
                )[0]
            ),
        }
        row[_score_column(C0_FULL)] = float(full_scores[row_index])
        row[_delta_logloss_column(C0_FULL)] = 0.0
        row[_delta_brier_column(C0_FULL)] = 0.0
        row[_taxonomy_column(C0_FULL)] = row["full_taxonomy"]
        position_rows = {
            int(position): index
            for index, position in enumerate(targets)
        }
        for condition in REPLACEMENT_CONDITIONS:
            condition_seed_scores = seed_scores.get(condition, {})
            per_seed_full_length: list[np.ndarray] = []
            for seed in DONOR_SEEDS:
                seed_payload = condition_seed_scores.get(int(seed))
                seed_column = _seed_column(condition, int(seed))
                delta_column = _delta_logloss_seed_column(
                    condition,
                    int(seed),
                )
                if seed_payload is None:
                    row[seed_column] = float("nan")
                    row[delta_column] = float("nan")
                    continue
                seed_positions = np.asarray(
                    seed_payload["positions"],
                    dtype=np.int64,
                )
                seed_values = np.asarray(
                    seed_payload["scores"],
                    dtype=np.float64,
                )
                full_length = np.full(
                    targets.size,
                    np.nan,
                    dtype=np.float64,
                )
                for seed_row, frame in enumerate(seed_positions):
                    target_row = position_rows[int(frame)]
                    full_length[target_row] = seed_values[seed_row]
                per_seed_full_length.append(full_length)
                row[seed_column] = float(full_length[row_index])
                if np.isfinite(full_length[row_index]):
                    row[delta_column] = float(
                        e1._delta_logloss(
                            np.asarray([labels[row_index]]),
                            np.asarray([full_scores[row_index]]),
                            np.asarray([full_length[row_index]]),
                        )[0]
                    )
                else:
                    row[delta_column] = float("nan")
            if per_seed_full_length:
                with np.errstate(invalid="ignore"):
                    combined = np.nanmean(
                        np.stack(per_seed_full_length, axis=0),
                        axis=0,
                    )
            else:
                combined = np.full(
                    targets.size,
                    np.nan,
                    dtype=np.float64,
                )
            row[_score_column(condition)] = (
                float(combined[row_index])
                if np.isfinite(combined[row_index])
                else float("nan")
            )
            score = combined[row_index]
            if np.isfinite(score):
                row[_delta_logloss_column(condition)] = float(
                    e1._delta_logloss(
                        np.asarray([labels[row_index]]),
                        np.asarray([full_scores[row_index]]),
                        np.asarray([score]),
                    )[0]
                )
                row[_delta_brier_column(condition)] = float(
                    e1._delta_brier(
                        np.asarray([full_scores[row_index]]),
                        np.asarray([score]),
                        np.asarray([labels[row_index]]),
                    )[0]
                )
                row[_taxonomy_column(condition)] = str(
                    e1._taxonomy(
                        np.asarray([labels[row_index]]),
                        np.asarray([short_scores[row_index]]),
                        np.asarray([score]),
                    )[0]
                )
            else:
                row[_delta_logloss_column(condition)] = float("nan")
                row[_delta_brier_column(condition)] = float("nan")
                row[_taxonomy_column(condition)] = None
        rows.append(row)
    return rows


def _collect_e2_frame_results(
    *,
    population: e1.Population,
    match_records: Mapping[int, MatchRecord],
    plan: Mapping[int, Mapping[str, Mapping[int, Assignment]]],
    model: Any,
    frontend: Any,
    device: torch.device,
    chunk_frames: int,
) -> pd.DataFrame:
    cache_global, cache_full, cache_short = _load_baseline_cache()
    expected_global = _e2_test_global_indices(population)
    if not np.array_equal(cache_global, expected_global):
        raise ValueError("baseline cache does not align with E2 test mask")
    baseline_lookup = {
        int(index): (float(full), float(short))
        for index, full, short in zip(
            cache_global,
            cache_full,
            cache_short,
        )
    }
    encoded_cpu = _encode_records(
        population=population,
        item_indices=match_records.keys(),
        model=model,
        frontend=frontend,
        device=device,
        chunk_frames=int(chunk_frames),
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    targets = _noisy_test_records(population)
    for position, record in enumerate(targets, start=1):
        target = match_records[int(record.item_index)]
        target_positions = e1._record_test_positions(
            record,
            population.test_mask,
        )
        encoded = encoded_cpu[int(record.item_index)].to(device)
        short_logits = e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=int(chunk_frames),
        )
        base, seed_scores = _score_match_record(
            target=target,
            encoded=encoded,
            short_logits=short_logits,
            model=model,
            device=device,
            donor_encoded_cpu=encoded_cpu,
            target_positions=target_positions,
            assignments=plan.get(int(record.item_index), {}),
        )
        rows.extend(
            _frame_row_for_match_record(
                population=population,
                target=target,
                target_positions=target_positions,
                base=base,
                seed_scores=seed_scores,
                baseline_lookup=baseline_lookup,
            )
        )
        if position % 20 == 0 or position == len(targets):
            print(
                f"scored E2 record {position}/{len(targets)} "
                f"({len(rows)} frames, {time.time() - started:.1f}s)",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    if len(frame) != E2_EXPECTED_NOISY_TEST_FRAMES:
        raise ValueError(
            f"E2 scoring produced {len(frame)} frames, expected "
            f"{E2_EXPECTED_NOISY_TEST_FRAMES}"
        )
    return frame


def _run_feasibility_mode(args: argparse.Namespace) -> int:
    _assert_metadata_stage_before_freeze(mode="feasibility")
    population = e1._load_population()
    _validate_population(population)
    metas = _manifest_meta_for_population(population)
    match_records = _build_match_records(population, metas)
    plan, quality_rows = build_match_plan(
        population=population,
        match_records=match_records,
    )
    feasibility = build_feasibility(
        match_records=match_records,
        quality_rows=quality_rows,
    )
    _write_csv(FEASIBILITY_PATH, feasibility.to_dict(orient="records"))
    _write_csv(MATCH_QUALITY_PATH, quality_rows)
    print(
        json.dumps(
            {
                "mode": "feasibility",
                "target_count": len(match_records),
                "assignment_count": _assignment_count(plan),
                "conditions": feasibility.set_index("condition")[
                    "feasibility_status"
                ].to_dict(),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _run_e1_c4_audit_mode(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    _validate_frozen_protocol()
    _load_baseline_cache()
    population = e1._load_population()
    e1._validate_population(population, require_c1=True)
    donor_selection, _quality = e1._build_seed_donors(
        population.records,
        population.test_mask,
    )
    records_by_item = {
        int(record.item_index): record
        for record in population.records
    }
    needed_donors = {
        int(donor)
        for selected in donor_selection.values()
        for donor in selected.values()
    }
    model, frontend, _config = e1._load_model(device)
    donor_encoded_cpu = _encode_records(
        population=population,
        item_indices=needed_donors,
        model=model,
        frontend=frontend,
        device=device,
        chunk_frames=int(args.chunk_frames),
    )
    target_records = [
        record
        for record in population.records
        if np.any(e1._record_test_positions(record, population.test_mask))
    ]
    labels_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    speaker_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    c4_parts: list[np.ndarray] = []
    max_full_error = 0.0
    started = time.time()
    for position, record in enumerate(target_records, start=1):
        target_positions = e1._record_test_positions(
            record,
            population.test_mask,
        )
        waveform = e1.read_int16_audio(record.item.audio_path)
        encoded = e1._encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=int(args.chunk_frames),
        )
        donor_selection_for_record = {
            int(seed): int(
                donor_selection[int(seed)][int(record.item_index)]
            )
            for seed in DONOR_SEEDS
        }
        scores = e1._score_record(
            record=record,
            encoded=encoded,
            short_logits=e1._short_logits_for_encoded(
                encoded,
                model,
                chunk_frames=int(args.chunk_frames),
            ),
            model=model,
            device=device,
            c3_seeds=(),
            c4_seeds=DONOR_SEEDS,
            c5_seeds=(),
            donor_encoded={
                int(item_index): donor_encoded_cpu[int(item_index)].to(
                    device
                )
                for item_index in set(
                    donor_selection_for_record.values()
                )
            },
            donor_selection=donor_selection_for_record,
            donor_records=records_by_item,
            target_local_indices=target_positions,
        )
        full = np.asarray(scores["full"], dtype=np.float64)
        offset = target_positions - VALID_START
        max_full_error = max(
            max_full_error,
            _max_abs_error(
                full,
                record.frozen_full_scores[offset],
            ),
        )
        seed_scores = np.stack(
            [
                np.asarray(
                    scores[f"C4_{int(seed)}"],
                    dtype=np.float64,
                )
                for seed in DONOR_SEEDS
            ],
            axis=0,
        )
        labels_parts.append(record.labels[offset])
        source_parts.append(
            np.full(target_positions.size, record.source_key, dtype=object)
        )
        speaker_parts.append(
            np.full(target_positions.size, record.speaker_id, dtype=object)
        )
        full_parts.append(full)
        c4_parts.append(np.mean(seed_scores, axis=0))
        if position % 25 == 0 or position == len(target_records):
            print(
                f"E1 C4 audit record {position}/{len(target_records)} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
    if max_full_error > BASELINE_TOLERANCE:
        raise RuntimeError("E1 C4 audit did not reproduce E1 FULL")
    labels = np.concatenate(labels_parts)
    source_keys = np.concatenate(source_parts).astype(str)
    speaker_ids = np.concatenate(speaker_parts).astype(str)
    full = np.concatenate(full_parts)
    c4 = np.concatenate(c4_parts)
    delta = e1._delta_logloss(labels, full, c4)
    stats = e1._mean_and_ci(
        delta,
        source_keys,
        speaker_ids,
        seed=SOURCE_BOOTSTRAP_SEED,
    )
    passed = bool(
        abs(stats["mean"] - E1_C4_REFERENCE_DELTA)
        <= E1_C4_AUDIT_TOLERANCE
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_sha256": _sha256_file(Path(__file__)),
        "completed_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "device": str(device),
        "frames": int(labels.size),
        "records": len(target_records),
        "donors_encoded": len(donor_encoded_cpu),
        "max_abs_full_error": float(max_full_error),
        "replayed_delta_logloss": float(stats["mean"]),
        "replayed_source_ci95": [
            float(stats["ci95_low"]),
            float(stats["ci95_high"]),
        ],
        "reference_delta_logloss": E1_C4_REFERENCE_DELTA,
        "absolute_error": float(
            abs(stats["mean"] - E1_C4_REFERENCE_DELTA)
        ),
        "tolerance": E1_C4_AUDIT_TOLERANCE,
        "passed": passed,
        "elapsed_seconds": float(time.time() - started),
        "training_performed": False,
        "new_final_ood_touched": False,
    }
    _write_json(E1_C4_AUDIT_PATH, payload)
    if not passed:
        raise RuntimeError(
            "E1 C4 replay did not match the frozen reference: "
            f"replayed={stats['mean']}, reference={E1_C4_REFERENCE_DELTA}"
        )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=np.float64))


def _condition_stats_rows(
    frame: pd.DataFrame,
    *,
    feasibility_status: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    statistics: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    labels = frame["label"].to_numpy(dtype=np.float64)
    full = frame[_score_column(C0_FULL)].to_numpy(dtype=np.float64)
    short = frame["short_score"].to_numpy(dtype=np.float64)
    for condition in CONDITION_ORDER:
        delta = frame[_delta_logloss_column(condition)].to_numpy(
            dtype=np.float64
        )
        brier = frame[_delta_brier_column(condition)].to_numpy(
            dtype=np.float64
        )
        mask = _finite_mask(delta)
        if condition == C0_FULL:
            mask = np.ones(frame.shape[0], dtype=bool)
        if not np.any(mask):
            stats = {
                "condition": condition,
                "short_name": CONDITION_SHORT[condition],
                "role": CONDITION_ROLE[condition],
                "feasibility_status": "INCOMPARABLE",
                "frames": 0,
                "targets": 0,
                "sources": 0,
                "speakers": 0,
                "delta_logloss": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "speaker_ci95_low": float("nan"),
                "speaker_ci95_high": float("nan"),
                "delta_brier": float("nan"),
                "delta_brier_ci95_low": float("nan"),
                "delta_brier_ci95_high": float("nan"),
                "full_logloss": float("nan"),
                "intervention_logloss": float("nan"),
                "original_r_prevalence": float("nan"),
                "r_survival": float("nan"),
            }
        else:
            source = frame.loc[mask, "source_key"].astype(str).to_numpy()
            speaker = frame.loc[mask, "speaker_id"].astype(str).to_numpy()
            metrics = e1._mean_and_ci(
                delta[mask],
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED,
            )
            brier_metrics = e1._mean_and_ci(
                brier[mask],
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED + 10_000,
            )
            original_r = (
                (~e1._correct(labels[mask], short[mask]))
                & e1._correct(labels[mask], full[mask])
            )
            intervention = frame.loc[
                mask,
                _score_column(condition),
            ].to_numpy(dtype=np.float64)
            r_survival = (
                float(
                    np.mean(
                        e1._correct(
                            labels[mask][original_r],
                            intervention[original_r],
                        )
                    )
                )
                if np.any(original_r)
                else float("nan")
            )
            stats = {
                "condition": condition,
                "short_name": CONDITION_SHORT[condition],
                "role": CONDITION_ROLE[condition],
                "feasibility_status": (
                    "BASELINE"
                    if condition == C0_FULL
                    else feasibility_status.get(condition, "INCOMPARABLE")
                ),
                "frames": int(np.count_nonzero(mask)),
                "targets": int(
                    frame.loc[mask, "item_index"].nunique()
                ),
                "sources": int(len(set(source))),
                "speakers": int(len(set(speaker))),
                "delta_logloss": float(metrics["mean"]),
                "ci95_low": float(metrics["ci95_low"]),
                "ci95_high": float(metrics["ci95_high"]),
                "speaker_ci95_low": float(metrics["speaker_ci95_low"]),
                "speaker_ci95_high": float(metrics["speaker_ci95_high"]),
                "delta_brier": float(brier_metrics["mean"]),
                "delta_brier_ci95_low": float(
                    brier_metrics["ci95_low"]
                ),
                "delta_brier_ci95_high": float(
                    brier_metrics["ci95_high"]
                ),
                "full_logloss": float(
                    e1._logloss(labels[mask], full[mask])
                ),
                "intervention_logloss": float(
                    e1._logloss(labels[mask], intervention)
                ),
                "original_r_prevalence": float(np.mean(original_r)),
                "r_survival": r_survival,
            }
        statistics[condition] = stats
        rows.append(stats)
    return statistics, rows


def _contrast_effect(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    left_values = frame[_delta_logloss_column(left)].to_numpy(
        dtype=np.float64
    )
    right_values = frame[_delta_logloss_column(right)].to_numpy(
        dtype=np.float64
    )
    mask = _finite_mask(left_values) & _finite_mask(right_values)
    return (
        left_values - right_values,
        mask,
        left,
        right,
    )


def _contrast_rows(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    statistics: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for name, left, right, definition in PRIMARY_CONTRASTS:
        values, mask, _left, _right = _contrast_effect(
            frame,
            left=left,
            right=right,
        )
        if not np.any(mask):
            stats = {
                "contrast": name,
                "left_condition": left,
                "right_condition": right,
                "definition": definition,
                "status": "INCOMPARABLE",
                "frames": 0,
                "targets": 0,
                "sources": 0,
                "speakers": 0,
                "effect": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "speaker_ci95_low": float("nan"),
                "speaker_ci95_high": float("nan"),
            }
        else:
            source = frame.loc[mask, "source_key"].astype(str).to_numpy()
            speaker = frame.loc[mask, "speaker_id"].astype(str).to_numpy()
            metrics = e1._mean_and_ci(
                values[mask],
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED + 20_000,
            )
            stats = {
                "contrast": name,
                "left_condition": left,
                "right_condition": right,
                "definition": definition,
                "status": "PRIMARY_INTERPRETABLE",
                "frames": int(np.count_nonzero(mask)),
                "targets": int(
                    frame.loc[mask, "item_index"].nunique()
                ),
                "sources": int(len(set(source))),
                "speakers": int(len(set(speaker))),
                "effect": float(metrics["mean"]),
                "ci95_low": float(metrics["ci95_low"]),
                "ci95_high": float(metrics["ci95_high"]),
                "speaker_ci95_low": float(metrics["speaker_ci95_low"]),
                "speaker_ci95_high": float(metrics["speaker_ci95_high"]),
            }
        statistics[name] = stats
        rows.append(stats)
    return statistics, rows


def _cluster_bootstrap_rows(
    condition_stats: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        stats = condition_stats[condition]
        rows.append(
            {
                "kind": "condition",
                "name": condition,
                "left": condition,
                "right": C0_FULL,
                "status": stats["feasibility_status"],
                "frames": stats["frames"],
                "sources": stats["sources"],
                "estimate": stats["delta_logloss"],
                "source_ci95_low": stats["ci95_low"],
                "source_ci95_high": stats["ci95_high"],
                "speaker_ci95_low": stats["speaker_ci95_low"],
                "speaker_ci95_high": stats["speaker_ci95_high"],
            }
        )
    for name, _left, _right, _definition in PRIMARY_CONTRASTS:
        stats = contrasts[name]
        rows.append(
            {
                "kind": "primary_contrast",
                "name": name,
                "left": stats["left_condition"],
                "right": stats["right_condition"],
                "status": stats["status"],
                "frames": stats["frames"],
                "sources": stats["sources"],
                "estimate": stats["effect"],
                "source_ci95_low": stats["ci95_low"],
                "source_ci95_high": stats["ci95_high"],
                "speaker_ci95_low": stats["speaker_ci95_low"],
                "speaker_ci95_high": stats["speaker_ci95_high"],
            }
        )
    return rows


def _condition_values(
    frame: pd.DataFrame,
    condition: str,
    *,
    delta: bool = True,
) -> np.ndarray:
    column = (
        _delta_logloss_column(condition)
        if delta
        else _score_column(condition)
    )
    return frame[column].to_numpy(dtype=np.float64)


def _seed_delta_values(
    frame: pd.DataFrame,
    condition: str,
    seed: int,
) -> np.ndarray:
    if condition == C0_FULL:
        return np.zeros(len(frame), dtype=np.float64)
    return frame[
        _delta_logloss_seed_column(condition, int(seed))
    ].to_numpy(dtype=np.float64)


def _statistics_for_values(
    values: np.ndarray,
    frame: pd.DataFrame,
    *,
    mask: np.ndarray | None = None,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return {
            "frames": 0.0,
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "speaker_ci95_low": float("nan"),
            "speaker_ci95_high": float("nan"),
        }
    stats = e1._mean_and_ci(
        values[mask],
        frame.loc[mask, "source_key"].astype(str).to_numpy(),
        frame.loc[mask, "speaker_id"].astype(str).to_numpy(),
        seed=int(seed),
    )
    return {
        "frames": float(np.count_nonzero(mask)),
        "mean": float(stats["mean"]),
        "ci95_low": float(stats["ci95_low"]),
        "ci95_high": float(stats["ci95_high"]),
        "speaker_ci95_low": float(stats["speaker_ci95_low"]),
        "speaker_ci95_high": float(stats["speaker_ci95_high"]),
    }


def _event_masks(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    onset_distance = frame["onset_distance"].to_numpy(dtype=np.float64)
    offset_distance = frame["offset_distance"].to_numpy(dtype=np.float64)
    transition_distance = frame[
        "posterior_transition_distance"
    ].to_numpy(dtype=np.float64)
    speech_run = frame["speech_run_length"].to_numpy(dtype=np.float64)
    silence_run = frame["silence_run_length"].to_numpy(dtype=np.float64)
    event_masks = {
        "S1_ONSET_NEAR": (
            (labels == 1)
            & (onset_distance >= 0.0)
            & (onset_distance <= EVENT_DISTANCE_WINDOW)
        ),
        "S2_TRANSITION_NEAR": (
            (transition_distance >= 0.0)
            & (transition_distance <= EVENT_DISTANCE_WINDOW)
        ),
        "S3_OFFSET_NEAR": (
            (labels == 0)
            & (offset_distance >= 0.0)
            & (offset_distance <= EVENT_DISTANCE_WINDOW)
        ),
        "S4_STABLE_SPEECH": (
            (labels == 1) & (speech_run >= STABLE_RUN_MIN)
        ),
        "S5_STABLE_SILENCE": (
            (labels == 0) & (silence_run >= STABLE_RUN_MIN)
        ),
    }
    same_label_run = np.where(
        labels == 1,
        speech_run,
        silence_run,
    )
    stable_masks = {
        "S1_ONSET_NEAR": (
            (labels == 1)
            & (speech_run >= STABLE_RUN_MIN)
            & ~event_masks["S1_ONSET_NEAR"]
        ),
        "S2_TRANSITION_NEAR": (
            (same_label_run >= STABLE_RUN_MIN)
            & ~event_masks["S2_TRANSITION_NEAR"]
        ),
        "S3_OFFSET_NEAR": (
            (labels == 0)
            & (silence_run >= STABLE_RUN_MIN)
            & ~event_masks["S3_OFFSET_NEAR"]
        ),
        "S4_STABLE_SPEECH": event_masks["S4_STABLE_SPEECH"],
        "S5_STABLE_SILENCE": event_masks["S5_STABLE_SILENCE"],
    }
    return event_masks, stable_masks


def _match_event_strata(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair event frames with pre-outcome matched stable frames.

    Pairing uses Short uncertainty first and the absolute frame index only
    as a deterministic tie-break. Event frames are excluded from the stable
    pool for their own stratum.
    """

    working = frame.reset_index(drop=True)
    event_masks, stable_masks = _event_masks(working)
    labels = working["label"].to_numpy(dtype=np.int64)
    item_index = working["item_index"].to_numpy(dtype=np.int64)
    frame_index = working["frame_in_utterance"].to_numpy(dtype=np.int64)
    uncertainty = working["uncertainty"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {
        "same_item": True,
        "same_label": True,
        "stable_run_verified": True,
        "event_frames_excluded": True,
        "no_self_match": True,
    }
    for stratum in ("S1_ONSET_NEAR", "S2_TRANSITION_NEAR", "S3_OFFSET_NEAR"):
        event_mask = event_masks[stratum]
        stable_mask = stable_masks[stratum]
        for item in np.unique(item_index[event_mask]):
            item_mask = item_index == item
            event_rows = np.flatnonzero(event_mask & item_mask)
            stable_rows = np.flatnonzero(stable_mask & item_mask)
            for event_row in event_rows:
                candidates = stable_rows[
                    labels[stable_rows] == labels[event_row]
                ]
                if candidates.size == 0:
                    continue
                uncertainty_difference = np.abs(
                    uncertainty[candidates] - uncertainty[event_row]
                )
                order = np.lexsort(
                    (
                        frame_index[candidates],
                        uncertainty_difference,
                    )
                )
                stable_row = int(candidates[int(order[0])])
                rows.append(
                    {
                        "stratum": stratum,
                        "event_row": int(event_row),
                        "stable_row": stable_row,
                        "item_index": int(item),
                        "event_frame": int(frame_index[event_row]),
                        "stable_frame": int(frame_index[stable_row]),
                        "label": int(labels[event_row]),
                        "event_uncertainty": float(uncertainty[event_row]),
                        "stable_uncertainty": float(uncertainty[stable_row]),
                        "uncertainty_difference": float(
                            uncertainty_difference[int(order[0])]
                        ),
                    }
                )
                checks["same_item"] &= bool(
                    item_index[stable_row] == item
                )
                checks["same_label"] &= bool(
                    labels[stable_row] == labels[event_row]
                )
                checks["event_frames_excluded"] &= bool(
                    stable_mask[stable_row] and not event_mask[stable_row]
                )
                checks["no_self_match"] &= bool(
                    event_row != stable_row
                )
                if stratum == "S1_ONSET_NEAR":
                    run_ok = (
                        working.loc[stable_row, "speech_run_length"]
                        >= STABLE_RUN_MIN
                    )
                elif stratum == "S3_OFFSET_NEAR":
                    run_ok = (
                        working.loc[stable_row, "silence_run_length"]
                        >= STABLE_RUN_MIN
                    )
                else:
                    column = (
                        "speech_run_length"
                        if int(labels[stable_row]) == 1
                        else "silence_run_length"
                    )
                    run_ok = (
                        working.loc[stable_row, column] >= STABLE_RUN_MIN
                    )
                checks["stable_run_verified"] &= bool(run_ok)
    pair_frame = pd.DataFrame(rows)
    event_counts = {
        stratum: int(np.count_nonzero(mask))
        for stratum, mask in event_masks.items()
    }
    pair_counts = (
        pair_frame["stratum"].value_counts().to_dict()
        if not pair_frame.empty
        else {}
    )
    checks.update(
        {
            "event_frame_counts": event_counts,
            "matched_event_counts": {
                stratum: int(pair_counts.get(stratum, 0))
                for stratum in (
                    "S1_ONSET_NEAR",
                    "S2_TRANSITION_NEAR",
                    "S3_OFFSET_NEAR",
                )
            },
            "all_event_frames_matched": bool(
                all(
                    int(pair_counts.get(stratum, 0))
                    == int(event_counts[stratum])
                    for stratum in (
                        "S1_ONSET_NEAR",
                        "S2_TRANSITION_NEAR",
                        "S3_OFFSET_NEAR",
                    )
                )
            ),
        }
    )
    return pair_frame, checks


def _event_strata_rows(
    frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    working = frame.reset_index(drop=True)
    event_masks, _stable_masks = _event_masks(working)
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for stratum_index, stratum in enumerate(
        (
            "S1_ONSET_NEAR",
            "S2_TRANSITION_NEAR",
            "S3_OFFSET_NEAR",
            "S4_STABLE_SPEECH",
            "S5_STABLE_SILENCE",
        )
    ):
        event_mask = event_masks[stratum]
        pair_subset = (
            pair_frame.loc[pair_frame["stratum"] == stratum]
            if not pair_frame.empty
            else pd.DataFrame()
        )
        for condition_index, condition in enumerate(CONDITION_ORDER):
            values = _condition_values(working, condition)
            stats = _statistics_for_values(
                values,
                working,
                mask=event_mask,
                seed=(
                    SOURCE_BOOTSTRAP_SEED
                    + 30_000
                    + stratum_index * 100
                    + condition_index
                ),
            )
            row: dict[str, Any] = {
                "condition": condition,
                "short_name": CONDITION_SHORT[condition],
                "stratum": stratum,
                "definition": EVENT_STRATA[stratum],
                "frames": int(stats["frames"]),
                "delta_logloss": stats["mean"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "speaker_ci95_low": stats["speaker_ci95_low"],
                "speaker_ci95_high": stats["speaker_ci95_high"],
                "matched_stable_frames": 0,
                "matched_stable_delta_logloss": float("nan"),
                "matched_stable_ci95_low": float("nan"),
                "matched_stable_ci95_high": float("nan"),
                "excess_over_matched_stable": float("nan"),
                "excess_ci95_low": float("nan"),
                "excess_ci95_high": float("nan"),
            }
            if not pair_subset.empty:
                event_rows = pair_subset["event_row"].to_numpy(dtype=np.int64)
                stable_rows = pair_subset[
                    "stable_row"
                ].to_numpy(dtype=np.int64)
                event_values = values[event_rows]
                stable_values = values[stable_rows]
                paired_values = event_values - stable_values
                event_stats = _statistics_for_values(
                    event_values,
                    working.loc[event_rows].reset_index(drop=True),
                    seed=SOURCE_BOOTSTRAP_SEED + 31_000 + stratum_index,
                )
                stable_stats = _statistics_for_values(
                    stable_values,
                    working.loc[event_rows].reset_index(drop=True),
                    seed=SOURCE_BOOTSTRAP_SEED + 32_000 + stratum_index,
                )
                excess_stats = _statistics_for_values(
                    paired_values,
                    working.loc[event_rows].reset_index(drop=True),
                    seed=(
                        SOURCE_BOOTSTRAP_SEED
                        + 33_000
                        + stratum_index * 10
                        + condition_index
                    ),
                )
                row.update(
                    {
                        "frames": int(event_stats["frames"]),
                        "delta_logloss": event_stats["mean"],
                        "ci95_low": event_stats["ci95_low"],
                        "ci95_high": event_stats["ci95_high"],
                        "matched_stable_frames": int(
                            stable_stats["frames"]
                        ),
                        "matched_stable_delta_logloss": (
                            stable_stats["mean"]
                        ),
                        "matched_stable_ci95_low": (
                            stable_stats["ci95_low"]
                        ),
                        "matched_stable_ci95_high": (
                            stable_stats["ci95_high"]
                        ),
                        "excess_over_matched_stable": (
                            excess_stats["mean"]
                        ),
                        "excess_ci95_low": excess_stats["ci95_low"],
                        "excess_ci95_high": excess_stats["ci95_high"],
                    }
                )
            rows.append(row)
            lookup[(condition, stratum)] = row
    return rows, lookup


def _onset_offset_rows(
    frame: pd.DataFrame,
    event_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    working = frame.reset_index(drop=True)
    event_masks, _stable_masks = _event_masks(working)
    condition = C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE
    values = _condition_values(working, condition)
    onset_values = values[event_masks["S1_ONSET_NEAR"]]
    offset_values = values[event_masks["S3_OFFSET_NEAR"]]
    transition_values = values[event_masks["S2_TRANSITION_NEAR"]]
    onset_stats = _statistics_for_values(
        onset_values,
        working.loc[event_masks["S1_ONSET_NEAR"]].reset_index(drop=True),
        seed=SOURCE_BOOTSTRAP_SEED + 40_000,
    )
    offset_stats = _statistics_for_values(
        offset_values,
        working.loc[event_masks["S3_OFFSET_NEAR"]].reset_index(drop=True),
        seed=SOURCE_BOOTSTRAP_SEED + 40_001,
    )
    transition_stats = _statistics_for_values(
        transition_values,
        working.loc[event_masks["S2_TRANSITION_NEAR"]].reset_index(drop=True),
        seed=SOURCE_BOOTSTRAP_SEED + 40_002,
    )
    paired_difference = np.concatenate(
        (onset_values, -offset_values)
    )
    paired_sources = np.concatenate(
        (
            working.loc[
                event_masks["S1_ONSET_NEAR"], "source_key"
            ]
            .astype(str)
            .to_numpy(),
            working.loc[
                event_masks["S3_OFFSET_NEAR"], "source_key"
            ]
            .astype(str)
            .to_numpy(),
        )
    )
    paired_speakers = np.concatenate(
        (
            working.loc[
                event_masks["S1_ONSET_NEAR"], "speaker_id"
            ]
            .astype(str)
            .to_numpy(),
            working.loc[
                event_masks["S3_OFFSET_NEAR"], "speaker_id"
            ]
            .astype(str)
            .to_numpy(),
        )
    )
    difference_stats = e1._mean_and_ci(
        paired_difference,
        paired_sources,
        paired_speakers,
        seed=SOURCE_BOOTSTRAP_SEED + 40_003,
    )
    onset_row = event_lookup[(condition, "S1_ONSET_NEAR")]
    offset_row = event_lookup[(condition, "S3_OFFSET_NEAR")]
    row = {
        "outcome_condition": condition,
        "onset_frames": int(onset_stats["frames"]),
        "transition_frames": int(transition_stats["frames"]),
        "offset_frames": int(offset_stats["frames"]),
        "onset_effect": onset_stats["mean"],
        "onset_ci95_low": onset_stats["ci95_low"],
        "onset_ci95_high": onset_stats["ci95_high"],
        "transition_effect": transition_stats["mean"],
        "transition_ci95_low": transition_stats["ci95_low"],
        "transition_ci95_high": transition_stats["ci95_high"],
        "offset_effect": offset_stats["mean"],
        "offset_ci95_low": offset_stats["ci95_low"],
        "offset_ci95_high": offset_stats["ci95_high"],
        "onset_minus_offset": float(difference_stats["mean"]),
        "onset_minus_offset_ci95_low": float(
            difference_stats["ci95_low"]
        ),
        "onset_minus_offset_ci95_high": float(
            difference_stats["ci95_high"]
        ),
        "onset_excess_over_matched_stable": onset_row[
            "excess_over_matched_stable"
        ],
        "offset_excess_over_matched_stable": offset_row[
            "excess_over_matched_stable"
        ],
    }
    return [row], row


def _seen_unseen_rows(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    seen_mask = frame["seen"].to_numpy(dtype=bool)
    unseen_mask = frame["unseen"].to_numpy(dtype=bool)
    for condition_index, condition in enumerate(REPLACEMENT_CONDITIONS):
        values = _condition_values(frame, condition)
        seen_stats = _statistics_for_values(
            values,
            frame,
            mask=seen_mask,
            seed=SOURCE_BOOTSTRAP_SEED + 50_000 + condition_index,
        )
        unseen_stats = _statistics_for_values(
            values,
            frame,
            mask=unseen_mask,
            seed=SOURCE_BOOTSTRAP_SEED + 50_100 + condition_index,
        )
        difference_values = np.concatenate(
            (
                values[unseen_mask & np.isfinite(values)],
                -values[seen_mask & np.isfinite(values)],
            )
        )
        difference_sources = np.concatenate(
            (
                frame.loc[
                    unseen_mask & np.isfinite(values), "source_key"
                ]
                .astype(str)
                .to_numpy(),
                frame.loc[
                    seen_mask & np.isfinite(values), "source_key"
                ]
                .astype(str)
                .to_numpy(),
            )
        )
        difference_speakers = np.concatenate(
            (
                frame.loc[
                    unseen_mask & np.isfinite(values), "speaker_id"
                ]
                .astype(str)
                .to_numpy(),
                frame.loc[
                    seen_mask & np.isfinite(values), "speaker_id"
                ]
                .astype(str)
                .to_numpy(),
            )
        )
        difference_stats = e1._mean_and_ci(
            difference_values,
            difference_sources,
            difference_speakers,
            seed=SOURCE_BOOTSTRAP_SEED + 50_200 + condition_index,
        )
        row = {
            "condition": condition,
            "short_name": CONDITION_SHORT[condition],
            "seen_frames": int(seen_stats["frames"]),
            "unseen_frames": int(unseen_stats["frames"]),
            "seen_effect": seen_stats["mean"],
            "seen_ci95_low": seen_stats["ci95_low"],
            "seen_ci95_high": seen_stats["ci95_high"],
            "unseen_effect": unseen_stats["mean"],
            "unseen_ci95_low": unseen_stats["ci95_low"],
            "unseen_ci95_high": unseen_stats["ci95_high"],
            "unseen_minus_seen": float(difference_stats["mean"]),
            "unseen_minus_seen_ci95_low": float(
                difference_stats["ci95_low"]
            ),
            "unseen_minus_seen_ci95_high": float(
                difference_stats["ci95_high"]
            ),
        }
        rows.append(row)
        lookup[condition] = row
    return rows, lookup


def _loso_contrast_rows(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    source_all = frame["source_key"].astype(str).to_numpy()
    for contrast_index, (
        name,
        left,
        right,
        _definition,
    ) in enumerate(PRIMARY_CONTRASTS):
        values, mask, _left, _right = _contrast_effect(
            frame,
            left=left,
            right=right,
        )
        full_estimate = (
            float(np.mean(values[mask])) if np.any(mask) else float("nan")
        )
        source_values = source_all[mask]
        estimates: list[float] = []
        for source in sorted(set(source_values)):
            source_mask = source_values != source
            if not np.any(source_mask):
                continue
            estimate = float(np.mean(values[mask][source_mask]))
            estimates.append(estimate)
            rows.append(
                {
                    "row_type": "leave_one_source",
                    "contrast": name,
                    "left_condition": left,
                    "right_condition": right,
                    "excluded_source": source,
                    "included_sources": int(
                        len(set(source_values)) - 1
                    ),
                    "frames": int(np.count_nonzero(mask & (source_all != source))),
                    "estimate": estimate,
                }
            )
        positive = all(value > 0.0 for value in estimates)
        negative = all(value < 0.0 for value in estimates)
        summary = {
            "row_type": "summary",
            "contrast": name,
            "left_condition": left,
            "right_condition": right,
            "excluded_source": "__NONE__",
            "included_sources": int(len(set(source_values))),
            "frames": int(np.count_nonzero(mask)),
            "estimate": full_estimate,
            "min_loso_estimate": (
                float(min(estimates)) if estimates else float("nan")
            ),
            "max_loso_estimate": (
                float(max(estimates)) if estimates else float("nan")
            ),
            "sign_consistent": bool(
                estimates and (positive or negative)
            ),
            "all_loso_positive": bool(estimates and positive),
            "all_loso_negative": bool(estimates and negative),
            "loso_source_count": len(estimates),
        }
        rows.insert(
            0,
            summary,
        )
        lookup[name] = summary
    return rows, lookup


def _donor_seed_rows(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    condition_summary: dict[str, dict[str, Any]] = {}
    contrast_summary: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(REPLACEMENT_CONDITIONS):
        seed_effects: list[float] = []
        for seed_index, seed in enumerate(DONOR_SEEDS):
            values = _seed_delta_values(frame, condition, int(seed))
            mask = np.isfinite(values)
            estimate = (
                float(np.mean(values[mask]))
                if np.any(mask)
                else float("nan")
            )
            seed_effects.append(estimate)
            rows.append(
                {
                    "row_type": "condition_seed",
                    "condition_or_contrast": condition,
                    "contrast": "",
                    "seed": int(seed),
                    "frames": int(np.count_nonzero(mask)),
                    "estimate": estimate,
                    "direction_positive": bool(
                        np.isfinite(estimate) and estimate > 0.0
                    ),
                }
            )
        finite_effects = [
            effect for effect in seed_effects if np.isfinite(effect)
        ]
        positive_count = int(
            sum(effect > 0.0 for effect in finite_effects)
        )
        summary = {
            "row_type": "condition_summary",
            "condition_or_contrast": condition,
            "contrast": "",
            "seed": -1,
            "frames": int(
                np.count_nonzero(
                    np.isfinite(_condition_values(frame, condition))
                )
            ),
            "estimate": (
                float(np.mean(_condition_values(frame, condition)[
                    np.isfinite(_condition_values(frame, condition))
                ]))
                if np.any(np.isfinite(_condition_values(frame, condition)))
                else float("nan")
            ),
            "positive_seed_count": positive_count,
            "finite_seed_count": len(finite_effects),
            "required_positive_seed_count": MIN_SEED_POSITIVE_COUNT,
            "seed_direction_stable": bool(
                positive_count >= MIN_SEED_POSITIVE_COUNT
            ),
        }
        rows.append(summary)
        condition_summary[condition] = summary

    for contrast_index, (
        name,
        left,
        right,
        _definition,
    ) in enumerate(PRIMARY_CONTRASTS):
        seed_effects: list[float] = []
        for seed in DONOR_SEEDS:
            left_values = _seed_delta_values(frame, left, int(seed))
            right_values = _seed_delta_values(frame, right, int(seed))
            values = left_values - right_values
            mask = np.isfinite(values)
            estimate = (
                float(np.mean(values[mask]))
                if np.any(mask)
                else float("nan")
            )
            seed_effects.append(estimate)
            rows.append(
                {
                    "row_type": "contrast_seed",
                    "condition_or_contrast": name,
                    "contrast": name,
                    "seed": int(seed),
                    "frames": int(np.count_nonzero(mask)),
                    "estimate": estimate,
                    "direction_positive": bool(
                        np.isfinite(estimate) and estimate > 0.0
                    ),
                }
            )
        finite_effects = [
            effect for effect in seed_effects if np.isfinite(effect)
        ]
        positive_count = int(
            sum(effect > 0.0 for effect in finite_effects)
        )
        summary = {
            "row_type": "contrast_summary",
            "condition_or_contrast": name,
            "contrast": name,
            "seed": -1,
            "frames": int(
                np.count_nonzero(
                    _finite_mask(
                        _contrast_effect(
                            frame,
                            left=left,
                            right=right,
                        )[0]
                    )
                )
            ),
            "estimate": float(
                np.mean(
                    _contrast_effect(
                        frame,
                        left=left,
                        right=right,
                    )[0][
                        _contrast_effect(
                            frame,
                            left=left,
                            right=right,
                        )[1]
                    ]
                )
            ),
            "positive_seed_count": positive_count,
            "finite_seed_count": len(finite_effects),
            "required_positive_seed_count": MIN_SEED_POSITIVE_COUNT,
            "seed_direction_stable": bool(
                positive_count >= MIN_SEED_POSITIVE_COUNT
            ),
        }
        rows.append(summary)
        contrast_summary[name] = summary
    return rows, {
        "conditions": condition_summary,
        "contrasts": contrast_summary,
    }


def _correction_survival_rows(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    short_scores = frame["short_score"].to_numpy(dtype=np.float64)
    full_scores = frame[_score_column(C0_FULL)].to_numpy(dtype=np.float64)
    full_taxonomy = e1._taxonomy(labels, short_scores, full_scores)
    rows: list[dict[str, Any]] = []
    survival_by_condition: dict[str, float] = {}
    for condition_index, condition in enumerate(CONDITION_ORDER):
        scores = frame[_score_column(condition)].to_numpy(dtype=np.float64)
        valid = np.isfinite(scores)
        intervention_taxonomy = np.full(
            len(frame),
            "",
            dtype="<U2",
        )
        intervention_taxonomy[valid] = e1._taxonomy(
            labels[valid],
            short_scores[valid],
            scores[valid],
        )
        intervention_correct = np.zeros(len(frame), dtype=bool)
        intervention_correct[valid] = e1._correct(
            labels[valid],
            scores[valid],
        )
        for original_state in ("R", "H"):
            original_mask = (full_taxonomy == original_state) & valid
            denominator = int(np.count_nonzero(original_mask))
            if denominator == 0:
                for intervention_state in ("SS", "R", "I", "H"):
                    rows.append(
                        {
                            "condition": condition,
                            "short_name": CONDITION_SHORT[condition],
                            "original_taxonomy": original_state,
                            "intervention_taxonomy": intervention_state,
                            "original_frames": 0,
                            "count": 0,
                            "proportion": float("nan"),
                            "correct_survival": float("nan"),
                        }
                    )
                continue
            for intervention_state in ("SS", "R", "I", "H"):
                count = int(
                    np.count_nonzero(
                        original_mask
                        & (intervention_taxonomy == intervention_state)
                    )
                )
                rows.append(
                    {
                        "condition": condition,
                        "short_name": CONDITION_SHORT[condition],
                        "original_taxonomy": original_state,
                        "intervention_taxonomy": intervention_state,
                        "original_frames": denominator,
                        "count": count,
                        "proportion": float(count) / float(denominator),
                        "correct_survival": float(
                            np.mean(intervention_correct[original_mask])
                        ),
                    }
                )
            if original_state == "R":
                survival_by_condition[condition] = float(
                    np.mean(intervention_correct[original_mask])
                )
    return rows, survival_by_condition


def _taxonomy_transition_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    short_scores = frame["short_score"].to_numpy(dtype=np.float64)
    full_scores = frame[_score_column(C0_FULL)].to_numpy(dtype=np.float64)
    full_taxonomy = e1._taxonomy(labels, short_scores, full_scores)
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        scores = frame[_score_column(condition)].to_numpy(dtype=np.float64)
        valid = np.isfinite(scores)
        intervention_taxonomy = np.full(len(frame), "", dtype="<U2")
        intervention_taxonomy[valid] = e1._taxonomy(
            labels[valid],
            short_scores[valid],
            scores[valid],
        )
        for full_state in ("SS", "R", "I", "H"):
            denominator = int(
                np.count_nonzero(valid & (full_taxonomy == full_state))
            )
            for intervention_state in ("SS", "R", "I", "H"):
                count = int(
                    np.count_nonzero(
                        valid
                        & (full_taxonomy == full_state)
                        & (intervention_taxonomy == intervention_state)
                    )
                )
                rows.append(
                    {
                        "condition": condition,
                        "short_name": CONDITION_SHORT[condition],
                        "full_taxonomy": full_state,
                        "intervention_taxonomy": intervention_state,
                        "full_state_frames": denominator,
                        "count": count,
                        "proportion": (
                            float(count) / float(denominator)
                            if denominator
                            else float("nan")
                        ),
                    }
                )
    return rows


def _decision_status(
    *,
    baseline: Mapping[str, Any],
    e1_c4_replay: Mapping[str, Any],
    feasibility: pd.DataFrame,
    frame: pd.DataFrame,
    condition_stats: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    event_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    loso: Mapping[str, Mapping[str, Any]],
    donor_seed: Mapping[str, Any],
    e1_unchanged: bool,
) -> dict[str, Any]:
    feasibility_status = _feasibility_status_map(feasibility)
    core_conditions = (
        C1_SAME_UTT_DIFFERENT_TIME,
        C3_SAME_NOISE_CLASS,
        C6_SPEECH_STATE_MISMATCH,
        C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
    )
    core_matching_valid = all(
        feasibility_status.get(condition) == "PRIMARY_INTERPRETABLE"
        for condition in core_conditions
    )
    intervention_valid = bool(
        _max_abs_error(
            frame["embedded_short_score"].to_numpy(dtype=np.float64),
            frame["short_score"].to_numpy(dtype=np.float64),
        )
        <= BASELINE_TOLERANCE
        and _max_abs_error(
            frame[_score_column(C0_FULL)].to_numpy(dtype=np.float64),
            frame["full_score"].to_numpy(dtype=np.float64),
        )
        <= BASELINE_TOLERANCE
    )
    validity = {
        "baseline_passed": bool(baseline.get("passed")),
        "e1_c4_replay_passed": bool(e1_c4_replay.get("passed")),
        "core_matching_valid": core_matching_valid,
        "intervention_valid": intervention_valid,
        "e1_artifacts_unchanged": bool(e1_unchanged),
    }
    valid = all(validity.values())

    p1 = contrasts["P1_UTTERANCE"]
    p2 = contrasts["P2_NOISE_INSTANCE"]
    p3 = contrasts["P3_NOISE_CLASS"]
    p4 = contrasts["P4_SNR"]
    p5 = contrasts["P5_SPEECH_STATE"]
    p6 = contrasts["P6_RESIDUAL_IDENTITY"]

    def material_positive(stats: Mapping[str, Any]) -> bool:
        return bool(
            str(stats.get("status")) == "PRIMARY_INTERPRETABLE"
            and float(stats["effect"]) >= MATERIAL_CONTRAST_DELTA
            and float(stats["ci95_low"]) > 0.0
        )

    def material_point(stats: Mapping[str, Any]) -> bool:
        return bool(
            str(stats.get("status")) == "PRIMARY_INTERPRETABLE"
            and float(stats["effect"]) >= MATERIAL_CONTRAST_DELTA
        )

    b1 = material_positive(p1)
    b2 = material_positive(p3) or material_positive(p4)
    p5_effect = float(p5["effect"])
    p6_effect = float(p6["effect"])
    speech_state_fraction = (
        max(p5_effect, 0.0) / p6_effect
        if p6_effect > 0.0
        else float("inf")
    )
    b3 = bool(
        p6_effect > 0.0
        and speech_state_fraction < SPEECH_STATE_DOMINANCE_FRACTION
    )
    b4 = False
    b4_region = ""
    for stratum in ("S1_ONSET_NEAR", "S2_TRANSITION_NEAR"):
        event = event_lookup[
            (C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE, stratum)
        ]
        passed = bool(
            float(event["excess_over_matched_stable"])
            >= B4_MIN_EVENT_EXCESS
            and float(event["excess_ci95_low"]) > 0.0
        )
        if passed:
            b4 = True
            b4_region = stratum
            break

    relevant_contrasts: list[str] = []
    if material_point(p1):
        relevant_contrasts.append("P1_UTTERANCE")
    if material_point(p3):
        relevant_contrasts.append("P3_NOISE_CLASS")
    if material_point(p4):
        relevant_contrasts.append("P4_SNR")
    if not relevant_contrasts:
        relevant_contrasts.append("P1_UTTERANCE")
    contrast_seed = donor_seed["contrasts"]
    b5 = bool(
        all(
            bool(loso[name]["sign_consistent"])
            and bool(loso[name]["all_loso_positive"])
            and int(
                contrast_seed[name]["positive_seed_count"]
            )
            >= MIN_SEED_POSITIVE_COUNT
            for name in relevant_contrasts
        )
    )
    speech_state_dominates = bool(
        material_positive(p5)
        and p6_effect > 0.0
        and speech_state_fraction >= SPEECH_STATE_DOMINANCE_FRACTION
    )
    residual = bool(
        not b2
        and not speech_state_dominates
        and p6_effect >= MATERIAL_CONTRAST_DELTA
        and float(p6["ci95_low"]) > 0.0
    )
    favorable_point = bool(
        material_point(p1)
        or material_point(p3)
        or material_point(p4)
        or float(
            event_lookup[
                (
                    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
                    "S1_ONSET_NEAR",
                )
            ]["excess_over_matched_stable"]
        )
        >= B4_MIN_EVENT_EXCESS
        or float(
            event_lookup[
                (
                    C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
                    "S2_TRANSITION_NEAR",
                )
            ]["excess_over_matched_stable"]
        )
        >= B4_MIN_EVENT_EXCESS
        or p6_effect > 0.0
    )
    if not valid:
        status = "INCONCLUSIVE_OR_INVALID"
    elif speech_state_dominates:
        status = "SPEECH_STATE_COMPATIBILITY_SUPPORTED"
    elif b1 and b2 and b3 and b4 and b5:
        status = "BACKGROUND_REFERENCE_SUPPORTED"
    elif residual:
        status = "RESIDUAL_CONTEXT_IDENTITY_EFFECT"
    elif favorable_point:
        status = "BACKGROUND_REFERENCE_CONDITIONAL"
    else:
        status = "COMPATIBILITY_UNRESOLVED"
    return {
        "E2_STATUS": status,
        "VALIDITY": validity,
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "B4": b4,
        "B4_REGION": b4_region,
        "B5": b5,
        "SPEECH_STATE_DOMINATES": speech_state_dominates,
        "RESIDUAL_CONTEXT_IDENTITY": residual,
        "SPEECH_STATE_FRACTION_OF_P6": speech_state_fraction,
        "RELEVANT_B5_CONTRASTS": relevant_contrasts,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "P6": p6,
    }


def _reconciliation_classification(
    *,
    condition_stats: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    def material(name: str) -> bool:
        stats = contrasts[name]
        return bool(
            float(stats["effect"]) >= MATERIAL_CONTRAST_DELTA
            and float(stats["ci95_low"]) > 0.0
        )

    def favorable(name: str) -> bool:
        return bool(float(contrasts[name]["effect"]) > 0.0)

    c1 = condition_stats[C1_SAME_UTT_DIFFERENT_TIME]
    c7 = condition_stats[C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE]
    same_utterance_near_full = bool(
        abs(float(c1["delta_logloss"])) <= MATERIAL_CONTRAST_DELTA
    )
    replacement_harmful = bool(
        float(c7["delta_logloss"]) >= MATERIAL_CONTRAST_DELTA
    )
    r1 = (
        "SUPPORTED"
        if same_utterance_near_full and replacement_harmful
        else "PARTIALLY_SUPPORTED"
        if favorable("P1_UTTERANCE")
        else "NOT_SUPPORTED"
    )
    classifications = {
        "R1_exact_temporal_order_unimportant_context_statistics_matter": r1,
        "R2_same_utterance_identity_matters": (
            "SUPPORTED"
            if material("P1_UTTERANCE")
            else "PARTIALLY_SUPPORTED"
            if favorable("P1_UTTERANCE")
            else "NOT_SUPPORTED"
        ),
        "R3_background_or_noise_instance_matters": (
            "SUPPORTED"
            if material("P2_NOISE_INSTANCE")
            else "PARTIALLY_SUPPORTED"
            if favorable("P2_NOISE_INSTANCE")
            else "NOT_SUPPORTED"
        ),
        "R4_noise_class_matters": (
            "SUPPORTED"
            if material("P3_NOISE_CLASS")
            else "PARTIALLY_SUPPORTED"
            if favorable("P3_NOISE_CLASS")
            else "NOT_SUPPORTED"
        ),
        "R5_snr_or_level_matters": (
            "SUPPORTED"
            if material("P4_SNR")
            else "PARTIALLY_SUPPORTED"
            if favorable("P4_SNR")
            else "NOT_SUPPORTED"
        ),
        "R6_speech_state_composition_matters": (
            "SUPPORTED"
            if material("P5_SPEECH_STATE")
            else "PARTIALLY_SUPPORTED"
            if favorable("P5_SPEECH_STATE")
            else "NOT_SUPPORTED"
        ),
        "R7_coarse_metadata_do_not_explain": (
            "SUPPORTED"
            if material("P6_RESIDUAL_IDENTITY")
            else "PARTIALLY_SUPPORTED"
            if favorable("P6_RESIDUAL_IDENTITY")
            else "NOT_SUPPORTED"
        ),
    }
    if not any(
        value != "NOT_SUPPORTED" for value in classifications.values()
    ):
        classifications["R8_insufficient_evidence"] = "SUPPORTED"
    else:
        classifications["R8_insufficient_evidence"] = (
            "PARTIALLY_SUPPORTED"
        )
    return classifications


def _summary_effect(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "estimate": stats.get("delta_logloss", stats.get("effect")),
        "ci95_low": stats.get("ci95_low"),
        "ci95_high": stats.get("ci95_high"),
        "status": stats.get(
            "feasibility_status",
            stats.get("status"),
        ),
    }


def _build_final_summary(
    *,
    baseline: Mapping[str, Any],
    e1_c4_replay: Mapping[str, Any],
    feasibility: pd.DataFrame,
    condition_stats: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    onset_offset: Mapping[str, Any],
    seen_unseen_rows: Sequence[Mapping[str, Any]],
    survival_by_condition: Mapping[str, float],
    loso: Mapping[str, Mapping[str, Any]],
    donor_seed: Mapping[str, Any],
    decision: Mapping[str, Any],
    reconciliation: Mapping[str, Mapping[str, str]],
    protocol_deviations: Sequence[str],
) -> dict[str, Any]:
    feasibility_status = _feasibility_status_map(feasibility)
    seen_lookup = {
        str(row["condition"]): row for row in seen_unseen_rows
    }
    event_lookup = {
        (str(row["condition"]), str(row["stratum"])): row
        for row in event_rows
    }
    return {
        "E2_STATUS": decision["E2_STATUS"],
        "BASELINE_REPRODUCED": bool(baseline.get("passed")),
        "E1_C4_REPLAY_REPRODUCED": bool(e1_c4_replay.get("passed")),
        "MATCHING_FEASIBILITY_BY_CONDITION": feasibility_status,
        "SAME_UTT_EFFECT": _summary_effect(
            condition_stats[C1_SAME_UTT_DIFFERENT_TIME]
        ),
        "SAME_NOISE_INSTANCE_EFFECT": _summary_effect(
            condition_stats[C2_SAME_NOISE_INSTANCE]
        ),
        "SAME_NOISE_CLASS_EFFECT": _summary_effect(
            condition_stats[C3_SAME_NOISE_CLASS]
        ),
        "WRONG_SNR_EFFECT": _summary_effect(
            condition_stats[C4_SAME_CLASS_WRONG_SNR]
        ),
        "DIFFERENT_CLASS_EFFECT": _summary_effect(
            condition_stats[C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR]
        ),
        "SPEECH_STATE_MISMATCH_EFFECT": _summary_effect(
            condition_stats[C6_SPEECH_STATE_MISMATCH]
        ),
        "BEST_METADATA_MATCHED_EFFECT": _summary_effect(
            condition_stats[
                C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE
            ]
        ),
        "P1_UTTERANCE_CONTRAST": _summary_effect(
            contrasts["P1_UTTERANCE"]
        ),
        "P2_NOISE_INSTANCE_CONTRAST": _summary_effect(
            contrasts["P2_NOISE_INSTANCE"]
        ),
        "P3_NOISE_CLASS_CONTRAST": _summary_effect(
            contrasts["P3_NOISE_CLASS"]
        ),
        "P4_SNR_CONTRAST": _summary_effect(contrasts["P4_SNR"]),
        "P5_SPEECH_STATE_CONTRAST": _summary_effect(
            contrasts["P5_SPEECH_STATE"]
        ),
        "P6_RESIDUAL_IDENTITY_EFFECT": _summary_effect(
            contrasts["P6_RESIDUAL_IDENTITY"]
        ),
        "ONSET_EFFECT": {
            "estimate": onset_offset["onset_effect"],
            "ci95_low": onset_offset["onset_ci95_low"],
            "ci95_high": onset_offset["onset_ci95_high"],
        },
        "TRANSITION_EFFECT": {
            "estimate": onset_offset["transition_effect"],
            "ci95_low": onset_offset["transition_ci95_low"],
            "ci95_high": onset_offset["transition_ci95_high"],
        },
        "OFFSET_EFFECT": {
            "estimate": onset_offset["offset_effect"],
            "ci95_low": onset_offset["offset_ci95_low"],
            "ci95_high": onset_offset["offset_ci95_high"],
        },
        "ONSET_MINUS_OFFSET": {
            "estimate": onset_offset["onset_minus_offset"],
            "ci95_low": onset_offset[
                "onset_minus_offset_ci95_low"
            ],
            "ci95_high": onset_offset[
                "onset_minus_offset_ci95_high"
            ],
        },
        "ONSET_MATCHED_STABLE_EXCESS": event_lookup[
            (
                C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
                "S1_ONSET_NEAR",
            )
        ]["excess_over_matched_stable"],
        "TRANSITION_MATCHED_STABLE_EXCESS": event_lookup[
            (
                C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
                "S2_TRANSITION_NEAR",
            )
        ]["excess_over_matched_stable"],
        "SEEN_EFFECT": {
            condition: {
                "estimate": row["seen_effect"],
                "ci95_low": row["seen_ci95_low"],
                "ci95_high": row["seen_ci95_high"],
            }
            for condition, row in seen_lookup.items()
        },
        "UNSEEN_EFFECT": {
            condition: {
                "estimate": row["unseen_effect"],
                "ci95_low": row["unseen_ci95_low"],
                "ci95_high": row["unseen_ci95_high"],
            }
            for condition, row in seen_lookup.items()
        },
        "R_SURVIVAL_BY_CONDITION": dict(survival_by_condition),
        "SOURCE_LOSO_STABILITY": {
            name: {
                "full_estimate": stats["estimate"],
                "min_loso_estimate": stats["min_loso_estimate"],
                "max_loso_estimate": stats["max_loso_estimate"],
                "sign_consistent": stats["sign_consistent"],
                "all_loso_positive": stats["all_loso_positive"],
                "loso_source_count": stats["loso_source_count"],
            }
            for name, stats in loso.items()
        },
        "DONOR_SEED_STABILITY": {
            "conditions": donor_seed["conditions"],
            "contrasts": donor_seed["contrasts"],
        },
        "E1_RECONCILIATION": dict(reconciliation),
        "DECISION_DETAILS": dict(decision),
        "PROTOCOL_DEVIATIONS": list(protocol_deviations),
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
    }


def _finite_interval(
    estimate: Any,
    low: Any,
    high: Any,
) -> tuple[float, float, float]:
    mean = float(estimate)
    lower = float(low)
    upper = float(high)
    if not all(math.isfinite(value) for value in (mean, lower, upper)):
        return float("nan"), float("nan"), float("nan")
    return mean, lower, upper


def _write_figure_condition_delta_logloss(
    *,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    labels = [CONDITION_SHORT[condition] for condition in CONDITION_ORDER]
    values = [
        _finite_interval(
            condition_statistics[condition]["delta_logloss"],
            condition_statistics[condition]["ci95_low"],
            condition_statistics[condition]["ci95_high"],
        )
        for condition in CONDITION_ORDER
    ]
    means = np.asarray([value[0] for value in values], dtype=np.float64)
    lower = np.asarray([value[1] for value in values], dtype=np.float64)
    upper = np.asarray([value[2] for value in values], dtype=np.float64)
    errors = np.vstack(
        (
            np.where(np.isfinite(means - lower), means - lower, 0.0),
            np.where(np.isfinite(upper - means), upper - means, 0.0),
        )
    )
    colors = [
        "#6B7280"
        if condition == C0_FULL
        else "#3F6D8E"
        if condition_statistics[condition]["feasibility_status"]
        == "PRIMARY_INTERPRETABLE"
        else "#B7A27A"
        for condition in CONDITION_ORDER
    ]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, means, color=colors, width=0.66)
    axis.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.15,
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean delta log-loss vs C0 FULL")
    axis.set_title("E2 condition effects with source-cluster 95% CI")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_primary_contrasts(
    *,
    contrasts: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    names = [name for name, _left, _right, _definition in PRIMARY_CONTRASTS]
    values = [
        _finite_interval(
            contrasts[name]["effect"],
            contrasts[name]["ci95_low"],
            contrasts[name]["ci95_high"],
        )
        for name in names
    ]
    means = np.asarray([value[0] for value in values], dtype=np.float64)
    lower = np.asarray([value[1] for value in values], dtype=np.float64)
    upper = np.asarray([value[2] for value in values], dtype=np.float64)
    errors = np.vstack(
        (
            np.where(np.isfinite(means - lower), means - lower, 0.0),
            np.where(np.isfinite(upper - means), upper - means, 0.0),
        )
    )
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(names))
    axis.bar(positions, means, color="#B56B4A", width=0.64)
    axis.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.15,
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.axhline(
        MATERIAL_CONTRAST_DELTA,
        color="#8A3B2E",
        linewidth=1.0,
        linestyle="--",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [name.split("_", 1)[0] for name in names],
    )
    axis.set_ylabel("Contrast delta log-loss")
    axis.set_title("E2 preregistered primary contrasts")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_correction_survival(
    *,
    survival_by_condition: Mapping[str, float],
    path: Path,
) -> None:
    labels = [CONDITION_SHORT[condition] for condition in CONDITION_ORDER]
    values = [
        float(survival_by_condition.get(condition, float("nan")))
        for condition in CONDITION_ORDER
    ]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, values, color="#4E7A62", width=0.66)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("P(correct | original RF384 R)")
    axis.set_title("Correction survival by replacement condition")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_event_asymmetry(
    *,
    onset_offset: Mapping[str, Any],
    path: Path,
) -> None:
    labels = ("onset", "transition", "offset")
    values = (
        _finite_interval(
            onset_offset["onset_effect"],
            onset_offset["onset_ci95_low"],
            onset_offset["onset_ci95_high"],
        ),
        _finite_interval(
            onset_offset["transition_effect"],
            onset_offset["transition_ci95_low"],
            onset_offset["transition_ci95_high"],
        ),
        _finite_interval(
            onset_offset["offset_effect"],
            onset_offset["offset_ci95_low"],
            onset_offset["offset_ci95_high"],
        ),
    )
    means = np.asarray([value[0] for value in values], dtype=np.float64)
    lower = np.asarray([value[1] for value in values], dtype=np.float64)
    upper = np.asarray([value[2] for value in values], dtype=np.float64)
    errors = np.vstack(
        (
            np.where(np.isfinite(means - lower), means - lower, 0.0),
            np.where(np.isfinite(upper - means), upper - means, 0.0),
        )
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(len(labels))
    axis.bar(positions, means, color="#7B6A9A", width=0.62)
    axis.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.15,
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("C7 delta log-loss")
    axis.set_title("Onset, transition, and offset compatibility effects")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_seen_unseen(
    *,
    seen_unseen_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    lookup = {
        str(row["condition"]): row for row in seen_unseen_rows
    }
    labels = [
        CONDITION_SHORT[condition]
        for condition in REPLACEMENT_CONDITIONS
    ]
    seen = np.asarray(
        [
            float(lookup[condition]["seen_effect"])
            for condition in REPLACEMENT_CONDITIONS
        ],
        dtype=np.float64,
    )
    unseen = np.asarray(
        [
            float(lookup[condition]["unseen_effect"])
            for condition in REPLACEMENT_CONDITIONS
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    axis.bar(
        positions - width / 2.0,
        seen,
        width=width,
        color="#4F7596",
        label="seen",
    )
    axis.bar(
        positions + width / 2.0,
        unseen,
        width=width,
        color="#C08457",
        label="unseen",
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean delta log-loss vs C0 FULL")
    axis.set_title("Seen and unseen replacement effects")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figures(
    *,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    survival_by_condition: Mapping[str, float],
    onset_offset: Mapping[str, Any],
    seen_unseen_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    paths = tuple(FIGURES_ROOT / name for name in FIGURE_FILES)
    _write_figure_condition_delta_logloss(
        condition_statistics=condition_statistics,
        path=paths[0],
    )
    _write_figure_primary_contrasts(
        contrasts=contrasts,
        path=paths[1],
    )
    _write_figure_correction_survival(
        survival_by_condition=survival_by_condition,
        path=paths[2],
    )
    _write_figure_event_asymmetry(
        onset_offset=onset_offset,
        path=paths[3],
    )
    _write_figure_seen_unseen(
        seen_unseen_rows=seen_unseen_rows,
        path=paths[4],
    )
    return list(paths)


def _status_interpretation(status: str) -> str:
    interpretations = {
        "BACKGROUND_REFERENCE_SUPPORTED": (
            "Within the frozen E2 condition set, preserving a compatible "
            "background reference provided more RF384 value than the "
            "controlled mismatched replacements, with the preregistered "
            "event and stability checks satisfied."
        ),
        "BACKGROUND_REFERENCE_CONDITIONAL": (
            "Background compatibility has favorable point estimates, but "
            "at least one primary interval crosses zero or a source/donor "
            "stability check remains unresolved. The stronger supported "
            "status is not established."
        ),
        "RESIDUAL_CONTEXT_IDENTITY_EFFECT": (
            "The best metadata-matched different-source replacement remains "
            "harmful while the controlled background contrasts do not "
            "separate the tested mechanism dimensions. A residual "
            "context/identity effect remains."
        ),
        "SPEECH_STATE_COMPATIBILITY_SUPPORTED": (
            "Deliberate mismatch of remote speech/non-speech composition "
            "explains most of the positive residual replacement damage in "
            "the frozen decision rule."
        ),
        "COMPATIBILITY_UNRESOLVED": (
            "The realistic replacement conditions do not separate the "
            "tested compatibility factors under the frozen decision rule."
        ),
        "INCONCLUSIVE_OR_INVALID": (
            "The frozen baseline, matching-validity, intervention-validity, "
            "or E1-preservation gate failed. No scientific status is "
            "established."
        ),
    }
    return interpretations.get(
        str(status),
        "No supported interpretation is available.",
    )


def _condition_effect_line(
    condition: str,
    stats: Mapping[str, Any],
) -> str:
    return (
        f"- {CONDITION_SHORT[condition]} {condition}: "
        f"`{float(stats['delta_logloss']):.8f}` "
        f"(source CI "
        f"`[{float(stats['ci95_low']):.8f}, "
        f"{float(stats['ci95_high']):.8f}]`, "
        f"`{stats['feasibility_status']}`)"
    )


def _write_e1_reconciliation(
    *,
    classifications: Mapping[str, Mapping[str, str]],
    condition_statistics: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    e1_c4_replay: Mapping[str, Any],
) -> None:
    lines = [
        "# E2/E1 remote-context reconciliation",
        "",
        "E1 observed that permutation of remote windows was near FULL, "
        "ZERO and MATCHED_REPLACE were harmful, and exact temporal ordering "
        "was not established as important. E2 decomposes that MATCHED_REPLACE "
        "penalty using constrained donor matching.",
        "",
        "## Frozen E1 reference checks",
        "",
        f"- E1 C4 replay passed: "
        f"`{str(bool(e1_c4_replay.get('passed'))).lower()}`.",
        f"- Replayed C4 delta log-loss: "
        f"`{float(e1_c4_replay.get('replayed_delta_logloss', float('nan'))):.8f}`.",
        f"- Frozen E1 C4 delta log-loss: "
        f"`{float(E1_C4_REFERENCE_DELTA):.8f}`.",
        "",
        "## E2 condition evidence",
        "",
        _condition_effect_line(
            C1_SAME_UTT_DIFFERENT_TIME,
            condition_statistics[C1_SAME_UTT_DIFFERENT_TIME],
        ),
        _condition_effect_line(
            C2_SAME_NOISE_INSTANCE,
            condition_statistics[C2_SAME_NOISE_INSTANCE],
        ),
        _condition_effect_line(
            C3_SAME_NOISE_CLASS,
            condition_statistics[C3_SAME_NOISE_CLASS],
        ),
        _condition_effect_line(
            C4_SAME_CLASS_WRONG_SNR,
            condition_statistics[C4_SAME_CLASS_WRONG_SNR],
        ),
        _condition_effect_line(
            C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR,
            condition_statistics[C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR],
        ),
        _condition_effect_line(
            C6_SPEECH_STATE_MISMATCH,
            condition_statistics[C6_SPEECH_STATE_MISMATCH],
        ),
        _condition_effect_line(
            C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
            condition_statistics[
                C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE
            ],
        ),
        "",
        "## Primary contrasts",
        "",
    ]
    for name, _left, _right, definition in PRIMARY_CONTRASTS:
        stats = contrasts[name]
        lines.append(
            f"- {name}: `{float(stats['effect']):.8f}` "
            f"(source CI `[{float(stats['ci95_low']):.8f}, "
            f"{float(stats['ci95_high']):.8f}]`). {definition}"
        )
    lines.extend(
        [
            "",
            "## Reconciliation classifications",
            "",
        ]
    )
    for name, value in classifications.items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The classifications are bounded to the frozen E2 population "
            "and MarbleNet RF384 intervention.",
            "- E1 PERMUTE behavior is not reinterpreted as proof that exact "
            "temporal ordering is irrelevant.",
            "- No training, threshold tuning, router search, architecture "
            "search, NEW_FINAL_OOD access, or E1 modification is authorized "
            "by this reconciliation.",
            "",
        ]
    )
    _write_markdown(E1_RECONCILIATION_PATH, lines)


def _write_final_report(
    *,
    summary: Mapping[str, Any],
    feasibility: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    onset_offset: Mapping[str, Any],
    seen_unseen_rows: Sequence[Mapping[str, Any]],
    e1_c4_replay: Mapping[str, Any],
    reconciliation_explanation: str,
) -> None:
    lines = [
        "# E2 Remote-Context Compatibility Decomposition",
        "",
        f"- Status: `{summary['E2_STATUS']}`",
        f"- Protocol: `{PROTOCOL_ID}`",
        "- Baseline reproduced: "
        f"`{str(bool(summary['BASELINE_REPRODUCED'])).lower()}`",
        "- E1 C4 replay reproduced: "
        f"`{str(bool(summary['E1_C4_REPLAY_REPRODUCED'])).lower()}`",
        "- Training performed: `false`",
        "- NEW_FINAL_OOD touched: `false`",
        "- Next experiment authorized: `false`",
        "",
        "## Method boundary",
        "",
        "E2 is an inference-only decomposition of the frozen MarbleNet "
        "RF384 checkpoint. Donors are constrained using manifest metadata, "
        "labels, and waveform energy before model outcomes are inspected. "
        "Target/local evidence and the current frame are unchanged.",
        "",
        "## Matching feasibility",
        "",
    ]
    for row in feasibility.to_dict(orient="records"):
        lines.append(
            f"- {row['condition']}: `{row['feasibility_status']}`; "
            f"eligible targets `{int(row['eligible_target_count'])}`, "
            f"eligible sources `{int(row['eligible_source_count'])}`, "
            f"matched coverage `{float(row['matched_frame_coverage']):.6f}`, "
            f"unique donor instances "
            f"`{int(row['unique_donor_instance_count'])}`."
        )
    lines.extend(
        [
            "",
            "## Condition effects",
            "",
        ]
    )
    for condition in CONDITION_ORDER:
        lines.append(
            _condition_effect_line(
                condition,
                condition_statistics[condition],
            )
        )
    lines.extend(
        [
            "",
            "## Primary contrasts",
            "",
        ]
    )
    for name, _left, _right, definition in PRIMARY_CONTRASTS:
        stats = contrasts[name]
        lines.append(
            f"- {name}: `{float(stats['effect']):.8f}` "
            f"(source CI `[{float(stats['ci95_low']):.8f}, "
            f"{float(stats['ci95_high']):.8f}]`). {definition}"
        )
    lines.extend(
        [
            "",
            "## Event and seen/unseen summaries",
            "",
            f"- Onset C7 effect: "
            f"`{float(onset_offset['onset_effect']):.8f}` "
            f"(source CI `[{float(onset_offset['onset_ci95_low']):.8f}, "
            f"{float(onset_offset['onset_ci95_high']):.8f}]`).",
            f"- Transition C7 effect: "
            f"`{float(onset_offset['transition_effect']):.8f}` "
            f"(source CI "
            f"`[{float(onset_offset['transition_ci95_low']):.8f}, "
            f"{float(onset_offset['transition_ci95_high']):.8f}]`).",
            f"- Offset C7 effect: "
            f"`{float(onset_offset['offset_effect']):.8f}` "
            f"(source CI `[{float(onset_offset['offset_ci95_low']):.8f}, "
            f"{float(onset_offset['offset_ci95_high']):.8f}]`).",
            f"- Onset minus offset: "
            f"`{float(onset_offset['onset_minus_offset']):.8f}`.",
        ]
    )
    for row in seen_unseen_rows:
        lines.append(
            f"- {row['condition']}: seen "
            f"`{float(row['seen_effect']):.8f}`, unseen "
            f"`{float(row['unseen_effect']):.8f}`, unseen-minus-seen "
            f"`{float(row['unseen_minus_seen']):.8f}`."
        )
    lines.extend(
        [
            "",
            "## E1 reconciliation",
            "",
            f"E1 C4 replay delta log-loss: "
            f"`{float(e1_c4_replay['replayed_delta_logloss']):.8f}` "
            f"(frozen reference `{E1_C4_REFERENCE_DELTA:.8f}`).",
            "",
            reconciliation_explanation,
            "",
            "## Decision details",
            "",
        ]
    )
    for key, value in summary["DECISION_DETAILS"].items():
        if key in {
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "SPEECH_STATE_DOMINATES",
            "RESIDUAL_CONTEXT_IDENTITY",
        }:
            lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- E2 does not train a background encoder, context classifier, "
            "router, attention module, embedding, or normalization module.",
            "- E2 does not modify RF384 or Tiny GRU.",
            "- E2 does not open NEW_FINAL_OOD and does not modify E1.",
            "- E2 does not authorize E3, E4, A-v3, A14 work, or any follow-on "
            "experiment.",
            "- The result is specific to the frozen MarbleNet RF384 "
            "checkpoint and the E2 eligible population unless stated "
            "otherwise.",
            "",
        ]
    )
    _write_markdown(FINAL_REPORT_PATH, lines)


def _write_claim_freeze(
    *,
    summary: Mapping[str, Any],
    condition_statistics: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    reconciliation_classifications: Mapping[str, Mapping[str, str]],
) -> None:
    supported = _status_interpretation(str(summary["E2_STATUS"]))
    partial = [
        f"{name}: {value}"
        for name, value in reconciliation_classifications.items()
        if value == "PARTIALLY_SUPPORTED"
    ]
    if str(summary["E2_STATUS"]) == "BACKGROUND_REFERENCE_CONDITIONAL":
        partial.append(
            "The background-compatibility mechanism has favorable point "
            "estimates but does not satisfy the preregistered supported gate."
        )
    lines = [
        "OBSERVATIONS",
        f"- E2_STATUS: {summary['E2_STATUS']}",
        f"- Baseline reproduced: "
        f"{str(bool(summary['BASELINE_REPRODUCED'])).lower()}",
        f"- E1 C4 replay reproduced: "
        f"{str(bool(summary['E1_C4_REPLAY_REPRODUCED'])).lower()}",
        f"- Matching feasibility: "
        f"{json.dumps(summary['MATCHING_FEASIBILITY_BY_CONDITION'], sort_keys=True)}",
        f"- C1 same-utterance effect: "
        f"{summary['SAME_UTT_EFFECT']}",
        f"- C2 same-noise-instance effect: "
        f"{summary['SAME_NOISE_INSTANCE_EFFECT']}",
        f"- C3 same-noise-class effect: "
        f"{summary['SAME_NOISE_CLASS_EFFECT']}",
        f"- C4 wrong-SNR effect: {summary['WRONG_SNR_EFFECT']}",
        f"- C5 different-class effect: {summary['DIFFERENT_CLASS_EFFECT']}",
        f"- C6 speech-state-mismatch effect: "
        f"{summary['SPEECH_STATE_MISMATCH_EFFECT']}",
        f"- C7 best-metadata-matched effect: "
        f"{summary['BEST_METADATA_MATCHED_EFFECT']}",
        f"- P1 utterance contrast: {summary['P1_UTTERANCE_CONTRAST']}",
        f"- P2 noise-instance contrast: "
        f"{summary['P2_NOISE_INSTANCE_CONTRAST']}",
        f"- P3 noise-class contrast: {summary['P3_NOISE_CLASS_CONTRAST']}",
        f"- P4 SNR contrast: {summary['P4_SNR_CONTRAST']}",
        f"- P5 speech-state contrast: "
        f"{summary['P5_SPEECH_STATE_CONTRAST']}",
        f"- P6 residual identity effect: "
        f"{summary['P6_RESIDUAL_IDENTITY_EFFECT']}",
        f"- Onset/transition/offset C7 effects: "
        f"{summary['ONSET_EFFECT']} / "
        f"{summary['TRANSITION_EFFECT']} / "
        f"{summary['OFFSET_EFFECT']}",
        f"- Source LOSO stability: {summary['SOURCE_LOSO_STABILITY']}",
        f"- Donor-seed stability: {summary['DONOR_SEED_STABILITY']}",
        f"- Protocol deviations: {summary['PROTOCOL_DEVIATIONS']}",
        "",
        "SUPPORTED_INTERPRETATIONS",
        supported,
        "",
        "PARTIALLY_SUPPORTED_INTERPRETATIONS",
    ]
    if partial:
        lines.extend([f"- {value}" for value in partial])
    else:
        lines.append("- None under the frozen classification.")
    lines.extend(
        [
            "",
            "ALTERNATIVE_EXPLANATIONS",
            "- Same-utterance identity may reflect background realization, "
            "channel, gain, speaker/context state, mixture construction, or "
            "utterance-specific normalization rather than noise estimation.",
            "- Residual replacement damage may reflect latent identity or "
            "processing effects not captured by the available metadata.",
            "- Matching variables may be noisy or insufficiently precise, "
            "leaving residual confounding despite constrained matching.",
            "- Source/domain heterogeneity and finite donor seed count may "
            "limit stability of the observed effects.",
            "- The conclusions are bounded to frozen MarbleNet RF384 "
            "checkpoint behavior.",
            "",
            "UNSUPPORTED_CLAIMS",
            "- \"RF384 estimates noise.\"",
            "- \"Long context works by noise adaptation.\"",
            "- \"Background reference is the universal mechanism.\"",
            "- \"Exact temporal ordering is irrelevant.\"",
            "- \"The same mechanism explains all refinable frames.\"",
            "- \"The mechanism generalizes to all VAD architectures.\"",
            "- \"E2 proves why Tiny GRU benefits from long history.\"",
            "- Any follow-on experiment or NEW_FINAL_OOD access is authorized.",
            "",
        ]
    )
    _write_markdown(CLAIM_FREEZE_PATH, lines)


def _load_e1_c4_audit() -> dict[str, Any]:
    if not E1_C4_AUDIT_PATH.exists():
        raise FileNotFoundError("E2 E1 C4 replay has not been run")
    payload = json.loads(E1_C4_AUDIT_PATH.read_text(encoding="utf-8"))
    if not bool(payload.get("passed")):
        raise ValueError("E2 E1 C4 replay did not pass")
    if payload.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("E2 E1 C4 replay belongs to a different protocol")
    if payload.get("runner_sha256") != _sha256_file(Path(__file__)):
        raise ValueError("E2 E1 C4 replay belongs to a different runner")
    return payload


def _write_execution_manifest(
    *,
    population: e1.Population,
    frame: pd.DataFrame,
    baseline: Mapping[str, Any],
    e1_c4_replay: Mapping[str, Any],
    figure_paths: Sequence[Path],
    e1_before: Mapping[str, Mapping[str, Any]],
    e1_after: Mapping[str, Mapping[str, Any]],
    runtime_seconds: float,
    device: torch.device,
    command: Sequence[str],
) -> dict[str, Any]:
    artifact_paths = [
        _output_path(name)
        for name in REQUIRED_OUTPUT_FILES
        if name not in {
            "e2_execution_manifest.json",
            "e2_figures/",
        }
    ]
    artifact_paths.extend(figure_paths)
    artifacts = _artifact_hashes(artifact_paths)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_sha256": _sha256_file(Path(__file__)),
        "status": "COMPLETE",
        "protocol_status": PROTOCOL_STATUS,
        "command": list(command),
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "runtime_seconds": float(runtime_seconds),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "population": _population_counts(population),
        "scored_frames": int(len(frame)),
        "baseline": {
            "passed": bool(baseline.get("passed")),
            "max_abs_full_error": float(
                baseline["max_abs_full_error"]
            ),
            "max_abs_embedded_short_error": float(
                baseline["max_abs_embedded_short_error"]
            ),
        },
        "e1_c4_replay": {
            "passed": bool(e1_c4_replay.get("passed")),
            "replayed_delta_logloss": float(
                e1_c4_replay["replayed_delta_logloss"]
            ),
            "absolute_error": float(e1_c4_replay["absolute_error"]),
            "tolerance": float(e1_c4_replay["tolerance"]),
        },
        "e1_artifacts": {
            "artifact_count": len(e1_after),
            "before": dict(e1_before),
            "after": dict(e1_after),
            "unchanged": bool(dict(e1_before) == dict(e1_after)),
        },
        "figures": [str(path.name) for path in figure_paths],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "training_performed": False,
        "new_final_ood_touched": False,
        "next_experiment_authorized": False,
    }
    _write_json(EXECUTION_MANIFEST_PATH, payload)
    return payload


def _verify_required_artifacts() -> dict[str, Any]:
    missing = [
        name
        for name in REQUIRED_OUTPUT_FILES
        if not _output_path(name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing required E2 artifacts: {missing}")
    figures = (
        sorted(
            path.name
            for path in FIGURES_ROOT.glob("*.png")
            if path.is_file()
        )
        if FIGURES_ROOT.exists()
        else []
    )
    if figures != list(FIGURE_FILES):
        raise ValueError(
            f"expected exactly five frozen figure files, got {figures}"
        )

    summary = json.loads(FINAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    missing_keys = sorted(set(REQUIRED_SUMMARY_KEYS) - set(summary))
    if missing_keys:
        raise ValueError(
            f"e2_final_summary.json is missing keys: {missing_keys}"
        )
    if summary["TRAINING_PERFORMED"] is not False:
        raise ValueError("TRAINING_PERFORMED must be false")
    if summary["NEW_FINAL_OOD_TOUCHED"] is not False:
        raise ValueError("NEW_FINAL_OOD_TOUCHED must be false")
    if summary["NEXT_EXPERIMENT_AUTHORIZED"] is not False:
        raise ValueError("NEXT_EXPERIMENT_AUTHORIZED must be false")
    if summary["E2_STATUS"] not in ALLOWED_E2_STATUSES:
        raise ValueError(f"unknown E2 status: {summary['E2_STATUS']}")

    manifest = json.loads(
        EXECUTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("execution manifest protocol identifier mismatch")
    if manifest.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("execution manifest protocol hash mismatch")
    if manifest.get("runner_sha256") != _sha256_file(Path(__file__)):
        raise ValueError("execution manifest runner hash mismatch")
    if manifest.get("training_performed") is not False:
        raise ValueError("manifest TRAINING_PERFORMED must be false")
    if manifest.get("new_final_ood_touched") is not False:
        raise ValueError("manifest NEW_FINAL_OOD_TOUCHED must be false")
    if manifest.get("next_experiment_authorized") is not False:
        raise ValueError("manifest NEXT_EXPERIMENT_AUTHORIZED must be false")

    for item in manifest.get("artifacts", []):
        path = REPO_ROOT / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        if _sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"artifact hash mismatch: {path}")
        if int(item["size_bytes"]) != int(path.stat().st_size):
            raise ValueError(f"artifact size mismatch: {path}")

    e1_record = manifest.get("e1_artifacts")
    if not isinstance(e1_record, Mapping):
        raise ValueError("execution manifest lacks E1 artifact snapshot")
    e1_before = e1_record.get("before")
    e1_after = e1_record.get("after")
    if not isinstance(e1_before, Mapping) or not isinstance(
        e1_after,
        Mapping,
    ):
        raise ValueError("execution manifest E1 snapshots are invalid")
    if dict(e1_before) != dict(e1_after):
        raise ValueError("E1 artifacts changed during E2 execution")
    if _e1_artifact_snapshot() != dict(e1_after):
        raise ValueError("E1 artifacts changed after E2 execution")

    frame = pd.read_parquet(
        FRAME_RESULTS_PATH,
        columns=["global_index"],
    )
    if len(frame) != E2_EXPECTED_NOISY_TEST_FRAMES:
        raise ValueError(
            f"frame results contain {len(frame)} rows, expected "
            f"{E2_EXPECTED_NOISY_TEST_FRAMES}"
        )
    return {
        "figures": figures,
        "summary": summary,
        "manifest": manifest,
    }


def _write_analysis_outputs(
    *,
    population: e1.Population,
    frame: pd.DataFrame,
    feasibility: pd.DataFrame,
    condition_stats: Mapping[str, Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    onset_offset: Mapping[str, Any],
    seen_unseen_rows: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
    donor_seed_rows: Sequence[Mapping[str, Any]],
    e1_c4_replay: Mapping[str, Any],
    decision: Mapping[str, Any],
    reconciliation: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    survival_by_condition = {
        str(row["condition"]): float(row["correct_survival"])
        for row in correction_rows
        if str(row.get("original_taxonomy")) == "R"
    }
    donor_seed = {
        "conditions": {
            str(row["condition_or_contrast"]): row
            for row in donor_seed_rows
            if str(row.get("row_type")) == "condition_summary"
        },
        "contrasts": {
            str(row["condition_or_contrast"]): row
            for row in donor_seed_rows
            if str(row.get("row_type")) == "contrast_summary"
        },
    }
    summary = _build_final_summary(
        baseline={
            "passed": bool(
                json.loads(
                    BASELINE_PATH.read_text(encoding="utf-8")
                ).get("passed")
            )
        },
        e1_c4_replay=e1_c4_replay,
        feasibility=feasibility,
        condition_stats=condition_stats,
        contrasts=contrasts,
        event_rows=event_rows,
        onset_offset=onset_offset,
        seen_unseen_rows=seen_unseen_rows,
        survival_by_condition=survival_by_condition,
        loso={
            name: stats
            for name, stats in (
                (str(row["contrast"]), row)
                for row in loso_rows
                if str(row.get("row_type")) == "summary"
            )
        },
        donor_seed=donor_seed,
        decision=decision,
        reconciliation=reconciliation,
        protocol_deviations=(),
    )
    _write_json(FINAL_SUMMARY_PATH, summary)
    _write_csv(CONDITION_RESULTS_PATH, condition_rows)
    _write_csv(PRIMARY_CONTRASTS_PATH, contrast_rows)
    _write_csv(CLUSTER_BOOTSTRAP_PATH, cluster_rows)
    _write_csv(CORRECTION_SURVIVAL_PATH, correction_rows)
    _write_csv(TAXONOMY_TRANSITIONS_PATH, taxonomy_rows)
    _write_csv(EVENT_STRATA_PATH, event_rows)
    _write_csv(ONSET_OFFSET_PATH, [onset_offset])
    _write_csv(SEEN_UNSEEN_PATH, seen_unseen_rows)
    _write_csv(SOURCE_INFLUENCE_PATH, loso_rows)
    _write_csv(DONOR_SEED_CONSISTENCY_PATH, donor_seed_rows)
    _write_e1_reconciliation(
        classifications=reconciliation,
        condition_statistics=condition_stats,
        contrasts=contrasts,
        e1_c4_replay=e1_c4_replay,
    )
    _write_final_report(
        summary=summary,
        feasibility=feasibility,
        condition_statistics=condition_stats,
        contrasts=contrasts,
        onset_offset=onset_offset,
        seen_unseen_rows=seen_unseen_rows,
        e1_c4_replay=e1_c4_replay,
        reconciliation_explanation=_status_interpretation(
            str(summary["E2_STATUS"])
        ),
    )
    _write_claim_freeze(
        summary=summary,
        condition_statistics=condition_stats,
        contrasts=contrasts,
        reconciliation_classifications=reconciliation,
    )
    return summary


def _run_formal_analysis(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    started = time.time()
    _validate_frozen_protocol()
    population = e1._load_population()
    _validate_population(population)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    _load_baseline_cache()
    e1_c4_replay = _load_e1_c4_audit()
    e1_before = _e1_artifact_snapshot()

    metas = _manifest_meta_for_population(population)
    match_records = _build_match_records(population, metas)
    plan, quality_rows = build_match_plan(
        population=population,
        match_records=match_records,
    )
    feasibility = build_feasibility(
        match_records=match_records,
        quality_rows=quality_rows,
    )
    _write_csv(
        FEASIBILITY_PATH,
        feasibility.to_dict(orient="records"),
    )
    _write_csv(MATCH_QUALITY_PATH, quality_rows)

    model, frontend, _config = e1._load_model(device)
    frame = _collect_e2_frame_results(
        population=population,
        match_records=match_records,
        plan=plan,
        model=model,
        frontend=frontend,
        device=device,
        chunk_frames=int(args.chunk_frames),
    )
    frame.to_parquet(FRAME_RESULTS_PATH, index=False)

    condition_stats, condition_rows = _condition_stats_rows(
        frame,
        feasibility_status=_feasibility_status_map(feasibility),
    )
    contrasts, contrast_rows = _contrast_rows(frame)
    cluster_rows = _cluster_bootstrap_rows(condition_stats, contrasts)
    correction_rows, _survival = _correction_survival_rows(frame)
    taxonomy_rows = _taxonomy_transition_rows(frame)
    pair_frame, _event_checks = _match_event_strata(frame)
    event_rows, event_lookup = _event_strata_rows(frame, pair_frame)
    _onset_rows, onset_offset = _onset_offset_rows(frame, event_lookup)
    seen_rows, _seen_lookup = _seen_unseen_rows(frame)
    loso_rows, loso_lookup = _loso_contrast_rows(frame)
    donor_rows, donor_summary = _donor_seed_rows(frame)

    e1_after = _e1_artifact_snapshot()
    decision = _decision_status(
        baseline=baseline,
        e1_c4_replay=e1_c4_replay,
        feasibility=feasibility,
        frame=frame,
        condition_stats=condition_stats,
        contrasts=contrasts,
        event_lookup=event_lookup,
        loso=loso_lookup,
        donor_seed=donor_summary,
        e1_unchanged=(e1_before == e1_after),
    )
    reconciliation = _reconciliation_classification(
        condition_stats=condition_stats,
        contrasts=contrasts,
    )
    summary = _write_analysis_outputs(
        population=population,
        frame=frame,
        feasibility=feasibility,
        condition_stats=condition_stats,
        condition_rows=condition_rows,
        contrasts=contrasts,
        contrast_rows=contrast_rows,
        cluster_rows=cluster_rows,
        correction_rows=correction_rows,
        taxonomy_rows=taxonomy_rows,
        event_rows=event_rows,
        onset_offset=onset_offset,
        seen_unseen_rows=seen_rows,
        loso_rows=loso_rows,
        donor_seed_rows=donor_rows,
        e1_c4_replay=e1_c4_replay,
        decision=decision,
        reconciliation=reconciliation,
    )
    survival_by_condition = {
        str(row["condition"]): float(row["correct_survival"])
        for row in correction_rows
        if str(row.get("original_taxonomy")) == "R"
    }
    figure_paths = _write_figures(
        condition_statistics=condition_stats,
        contrasts=contrasts,
        survival_by_condition=survival_by_condition,
        onset_offset=onset_offset,
        seen_unseen_rows=seen_rows,
    )
    _write_execution_manifest(
        population=population,
        frame=frame,
        baseline=baseline,
        e1_c4_replay=e1_c4_replay,
        figure_paths=figure_paths,
        e1_before=e1_before,
        e1_after=e1_after,
        runtime_seconds=time.time() - started,
        device=device,
        command=[sys.executable, "-m", RUNNER_MODULE, *sys.argv[1:]],
    )
    _verify_required_artifacts()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _run_verify_mode(args: argparse.Namespace) -> int:
    _validate_frozen_protocol()
    population = e1._load_population()
    _validate_population(population)
    result = _verify_required_artifacts()
    print(
        json.dumps(
            {
                "mode": "verify",
                "status": result["summary"]["E2_STATUS"],
                "figures": result["figures"],
                "verified": True,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen E2 remote-context compatibility "
            "decomposition."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "freeze_protocol",
            "feasibility",
            "baseline",
            "e1_c4_audit",
            "run",
            "verify",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-frames", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "freeze_protocol":
        return _freeze_protocol()
    device = resolve_device(str(args.device))
    if args.mode == "feasibility":
        return _run_feasibility_mode(args)
    if args.mode == "baseline":
        return _run_baseline_mode(args, device=device)
    if args.mode == "e1_c4_audit":
        return _run_e1_c4_audit_mode(args, device=device)
    if args.mode == "run":
        return _run_formal_analysis(args, device=device)
    if args.mode == "verify":
        return _run_verify_mode(args)
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
