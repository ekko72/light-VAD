# -*- coding: utf-8 -*-
"""Execute the frozen E4 Refinability Stability Decomposition (RSD).

E4 is analysis-only.  It uses the five already frozen CAR Tiny-GRU
Short/HFULL replicate pairs and does not train, fine-tune, change
checkpoints, alter interventions, or inspect the new final OOD set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "E4-RSD-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_E4_OUTCOME_ANALYSIS"
RUNNER_MODULE = "reproductions.difficulty_adaptive_context.run_e4"

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e4_refinability_stability_decomposition"
)
PROTOCOL_PATH = OUTPUT_ROOT / "e4_protocol_freeze.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "e4_protocol_sha256.txt"
RUNNER_HASH_PATH = OUTPUT_ROOT / "e4_runner_sha256.txt"
EXECUTION_MANIFEST_PATH = OUTPUT_ROOT / "e4_execution_manifest.json"
BASELINE_PATH = OUTPUT_ROOT / "e4_baseline_reproduction.json"

CAR_RESULTS_ROOT = REPO_ROOT / "results" / "cross_architecture_replication"
CAR_SEED_SCORE_PATHS = {
    seed: CAR_RESULTS_ROOT / "evaluations" / f"seed{seed}" / "scores.npz"
    for seed in (41, 59, 71, 83, 97)
}
CAR_TAXONOMY_REFERENCE_PATH = (
    CAR_RESULTS_ROOT / "car1_taxonomy_by_seed.csv"
)
CAR_AGREEMENT_REFERENCE_PATH = (
    CAR_RESULTS_ROOT / "car5_seed_agreement.csv"
)
CAR_FINAL_SUMMARY_PATH = CAR_RESULTS_ROOT / "car_final_summary.json"
CAR_RUNNER_PATH = (
    REPO_ROOT
    / "reproductions"
    / "cross_architecture_replication"
    / "run_car.py"
)
AE_MECHANISM_RUNNER_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "run_ae_mechanism.py"
)
RF384_REFERENCE_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
E3_FINAL_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e3_boundary_value_robustness"
    / "e3_final_summary.json"
)
TEST_MANIFEST_PATH = (
    REPO_ROOT
    / "data"
    / "librivad"
    / "manifests"
    / "LibriSpeech_test_medium.tsv"
)
DATA_ROOT = REPO_ROOT / "data" / "librivad"
LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"

MODEL_SEEDS = (41, 59, 71, 83, 97)
DECISION_THRESHOLD = 0.50
LOGLOSS_EPSILON = 1e-7
EXPECTED_TOTAL_FRAMES = 553_532
EXPECTED_TEST_FRAMES = 298_300
EXPECTED_TEST_SOURCES = 96
EXPECTED_TEST_SPEAKERS = 20
EXPECTED_EVALUATION_ROWS = 1_080
VALID_START = 382
FRAME_HOP = 160
MISSING_DISTANCE = 51.0

STABILITY_CATEGORY_ORDER = (
    "NEVER_R",
    "UNSTABLE_R",
    "STABLE_R",
    "CORE_R",
)
EVENT_BINS = ("0", "1", "2-4", "5-10", "11-25", "26-50", ">50")
EVENT_DIMENSIONS = (
    "onset_distance",
    "offset_distance",
    "posterior_transition_distance",
)
UNSEEN_NOISE = ("SSN_noise", "Street_noise", "Transport_noise")
SNR_ORDER = ("-5", "0", "5", "10", "15", "20")

PERMUTATION_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED_BASE = 20_260_921
BASELINE_TOLERANCE = 1e-6

REQUIRED_OUTPUT_FILES = (
    "e4_protocol_freeze.json",
    "e4_protocol_sha256.txt",
    "e4_runner_sha256.txt",
    "e4_execution_manifest.json",
    "e4_baseline_reproduction.json",
    "e4_kr_distribution.csv",
    "e4_stability_taxonomy.csv",
    "e4_null_overlap.csv",
    "e4_permutation_overlap.csv",
    "e4_proper_scoring_stability.csv",
    "e4_short_difficulty.csv",
    "e4_event_stability.csv",
    "e4_condition_stability.csv",
    "e4_disagreement_decomposition.csv",
    "e4_short_long_instability.csv",
    "e4_bootstrap.csv",
    "e4_source_influence.csv",
    "e4_final_summary.json",
    "e4_final_report.md",
    "e4_claim_freeze.md",
)

FIGURE_FILES = (
    "figure1_kr_distribution.png",
    "figure2_mean_vlog_by_stability.png",
    "figure3_non_r_fate_decomposition.png",
    "figure4_event_stability.png",
    "figure5_seen_unseen_stability.png",
)

ALLOWED_STATUSES = (
    "STABLE_REFINABLE_CORE",
    "POPULATION_STABLE_INSTANCE_RELATIVE",
    "MOSTLY_MODEL_RELATIVE_REFINABILITY",
    "INCONCLUSIVE_OR_INVALID",
    "NOT_EXECUTABLE_FROM_FROZEN_DATA",
)

# These rules are deliberately frozen before E4 outcome analysis.  The
# "nontrivial" thresholds are population shares, not outcome-selected cutoffs.
NONTRIVIAL_STABLE_R_MIN = 0.002
NONTRIVIAL_CORE_R_MIN = 0.0005
NONTRIVIAL_MIN_STABLE_SOURCES = 20
NONTRIVIAL_MIN_CORE_SOURCES = 10
NEAR_NULL_ABS_EXCESS_MAX = 0.0001
NEAR_NULL_RATIO_MAX = 1.25
CLAIM_C_STABLE_SHARE_MAX = 0.50
CLAIM_C_CORE_SHARE_MAX = 0.25

G3_MIN_ABS_DIFFERENCE = {
    "posterior_transition_0_vs_beyond_stable": 0.001,
    "onset_0_vs_beyond_stable": 0.001,
    "offset_0_vs_beyond_stable": 0.001,
    "stable_minus_unstable_short_margin": 0.005,
    "seen_vs_unseen_stable": 0.001,
    "stable_vs_unstable_non_r_i_share": 0.01,
}


@dataclass
class E4Data:
    seeds: tuple[int, ...]
    labels_full: np.ndarray
    test_mask: np.ndarray
    test_index: np.ndarray
    labels: np.ndarray
    source_keys: np.ndarray
    speaker_ids: np.ndarray
    conditions: np.ndarray
    noise_names: np.ndarray
    short_scores: dict[int, np.ndarray]
    long_scores: dict[int, np.ndarray]
    taxonomy: dict[int, dict[str, np.ndarray]]
    values: dict[int, np.ndarray]
    segments: tuple[tuple[int, int], ...]
    event_features: dict[str, np.ndarray]
    source_index: "SourceIndex"


@dataclass(frozen=True)
class SourceIndex:
    names: np.ndarray
    codes: np.ndarray
    count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
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
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"frozen input is missing: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _artifact_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            continue
        records.append(_file_record(path))
    return records


def _runner_hash() -> str:
    return _sha256_file(Path(__file__).resolve())


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _stable_hash_seed(label: str) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_SEED_BASE}:{label}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _frame_bce(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.clip(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        LOGLOSS_EPSILON,
        1.0 - LOGLOSS_EPSILON,
    )
    return -(
        labels * np.log(scores)
        + (1.0 - labels) * np.log1p(-scores)
    )


def _frame_brier(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return (scores - labels) ** 2


def _taxonomy_from_scores(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    labels_bool = np.asarray(labels, dtype=np.int64).reshape(-1) == 1
    short_correct = (
        np.asarray(short_scores, dtype=np.float64).reshape(-1)
        >= DECISION_THRESHOLD
    ) == labels_bool
    long_correct = (
        np.asarray(long_scores, dtype=np.float64).reshape(-1)
        >= DECISION_THRESHOLD
    ) == labels_bool
    ss = short_correct & long_correct
    r = (~short_correct) & long_correct
    i = (~short_correct) & (~long_correct)
    h = short_correct & (~long_correct)
    short_loss = _frame_bce(labels, short_scores)
    long_loss = _frame_bce(labels, long_scores)
    return {
        "short_correct": short_correct,
        "long_correct": long_correct,
        "SS": ss,
        "R": r,
        "I": i,
        "H": h,
        "short_loss": short_loss,
        "long_loss": long_loss,
        "v_log": short_loss - long_loss,
        "v_brier": _frame_brier(labels, short_scores)
        - _frame_brier(labels, long_scores),
    }


def _taxonomy_codes(taxonomy: Mapping[str, np.ndarray]) -> np.ndarray:
    codes = np.full(np.asarray(taxonomy["R"]).size, -1, dtype=np.int8)
    for code, name in enumerate(("SS", "R", "I", "H")):
        mask = np.asarray(taxonomy[name], dtype=bool)
        if np.any(codes[mask] >= 0):
            raise ValueError("taxonomy categories overlap")
        codes[mask] = code
    if np.any(codes < 0):
        raise ValueError("taxonomy categories are not exhaustive")
    return codes


def _source_index(values: np.ndarray) -> SourceIndex:
    names, codes = np.unique(
        np.asarray(values, dtype=str),
        return_inverse=True,
    )
    return SourceIndex(
        names=np.asarray(names, dtype=str),
        codes=np.asarray(codes, dtype=np.int64),
        count=int(names.size),
    )


def _validate_replicate_alignment(
    *,
    labels: np.ndarray,
    test_mask: np.ndarray,
    replicates: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    labels = np.asarray(labels)
    test_mask = np.asarray(test_mask, dtype=bool)
    checks: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        if seed not in replicates:
            checks.append(
                {
                    "seed": int(seed),
                    "check": "replicate_present",
                    "passed": False,
                }
            )
            continue
        arrays = replicates[seed]
        checks.append(
            {
                "seed": int(seed),
                "check": "labels_equal",
                "passed": bool(
                    np.array_equal(
                        np.asarray(arrays["labels"]),
                        labels,
                    )
                ),
            }
        )
        checks.append(
            {
                "seed": int(seed),
                "check": "test_mask_equal",
                "passed": bool(
                    np.array_equal(
                        np.asarray(arrays["test_mask"], dtype=bool),
                        test_mask,
                    )
                ),
            }
        )
    passed = all(bool(item["passed"]) for item in checks)
    if not passed:
        failed = [item for item in checks if not item["passed"]]
        raise ValueError(f"replicate alignment failed: {failed[:5]}")
    return {
        "passed": True,
        "checks": checks,
        "n_frames_total": int(labels.size),
        "n_frames_test": int(np.count_nonzero(test_mask)),
        "n_replicates": int(len(MODEL_SEEDS)),
    }


def _nearest_signed_distance(
    positions: np.ndarray,
    length: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    result = np.full(int(length), MISSING_DISTANCE, dtype=np.float64)
    if positions.size == 0:
        return result
    for frame in range(int(length)):
        distances = np.abs(positions - frame)
        nearest = int(np.argmin(distances))
        result[frame] = float(frame - positions[nearest])
    return result


def _nearest_absolute_distance(
    positions: np.ndarray,
    length: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    result = np.full(int(length), MISSING_DISTANCE, dtype=np.float64)
    if positions.size == 0:
        return result
    for frame in range(int(length)):
        result[frame] = float(np.min(np.abs(positions - frame)))
    return result


def _build_event_features(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    segments: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    features = {
        "onset_distance": np.full(labels.size, MISSING_DISTANCE, dtype=np.float64),
        "offset_distance": np.full(labels.size, MISSING_DISTANCE, dtype=np.float64),
        "posterior_transition_distance": np.full(
            labels.size,
            MISSING_DISTANCE,
            dtype=np.float64,
        ),
    }
    for start, end in segments:
        start = int(start)
        end = int(end)
        local_labels = labels[start:end]
        local_scores = short_scores[start:end]
        onset = np.flatnonzero(
            (local_labels[1:] == 1) & (local_labels[:-1] == 0)
        ) + 1
        offset = np.flatnonzero(
            (local_labels[1:] == 0) & (local_labels[:-1] == 1)
        ) + 1
        short_state = local_scores >= DECISION_THRESHOLD
        transitions = np.flatnonzero(
            short_state[1:] != short_state[:-1]
        ) + 1
        features["onset_distance"][start:end] = _nearest_signed_distance(
            onset,
            local_labels.size,
        )
        features["offset_distance"][start:end] = _nearest_signed_distance(
            offset,
            local_labels.size,
        )
        features["posterior_transition_distance"][start:end] = (
            _nearest_absolute_distance(
                transitions,
                local_labels.size,
            )
        )
    return features


def _reconstruct_segments(
    *,
    labels: np.ndarray,
    test_manifest: Path = TEST_MANIFEST_PATH,
    data_root: Path = DATA_ROOT,
    librispeech_root: Path = LIBRISPEECH_ROOT,
) -> tuple[tuple[int, int], ...]:
    from reproductions.difficulty_adaptive_context.data import (
        FRAME_HOP as DATA_FRAME_HOP,
        audio_frame_count,
        build_evaluation_items,
    )

    items = build_evaluation_items(
        test_manifest,
        generated_root=data_root / "generated",
        label_root=data_root / "labels",
        librispeech_root=librispeech_root,
        row_sample=EXPECTED_EVALUATION_ROWS,
        seed=17,
        include_clean=True,
    )
    segments: list[tuple[int, int]] = []
    cursor = 0
    for item in items:
        n_frames = audio_frame_count(item.audio_path) // DATA_FRAME_HOP + 1
        if n_frames <= VALID_START:
            continue
        length = n_frames - VALID_START
        segments.append((cursor, cursor + length))
        cursor += length
    if cursor != int(np.asarray(labels).size):
        raise ValueError(
            f"reconstructed utterance frames {cursor} != frozen {labels.size}"
        )
    return tuple(segments)


def _load_frozen_data() -> E4Data:
    reference = _load_npz(RF384_REFERENCE_PATH)
    required_reference = {
        "labels",
        "short_scores",
        "test_mask",
        "source_key",
        "speaker_ids",
        "condition",
        "noise_name",
    }
    missing = sorted(required_reference - set(reference))
    if missing:
        raise ValueError(f"RF384 bundle is missing arrays: {missing}")
    labels_full = np.asarray(reference["labels"], dtype=np.int64)
    test_mask = np.asarray(reference["test_mask"], dtype=bool)
    source_key_full = np.asarray(reference["source_key"], dtype=str)
    speaker_full = np.asarray(reference["speaker_ids"], dtype=str)
    condition_full = np.asarray(reference["condition"], dtype=str)
    noise_full = np.asarray(reference["noise_name"], dtype=str)
    reference_short = np.asarray(reference["short_scores"], dtype=np.float64)

    if labels_full.size != EXPECTED_TOTAL_FRAMES:
        raise ValueError("RF384 total frame count differs from the freeze")
    if int(np.count_nonzero(test_mask)) != EXPECTED_TEST_FRAMES:
        raise ValueError("RF384 test frame count differs from the freeze")
    if int(np.unique(source_key_full[test_mask]).size) != EXPECTED_TEST_SOURCES:
        raise ValueError("RF384 source count differs from the freeze")
    if int(np.unique(speaker_full[test_mask]).size) != EXPECTED_TEST_SPEAKERS:
        raise ValueError("RF384 speaker count differs from the freeze")

    replicate_arrays: dict[int, dict[str, np.ndarray]] = {}
    for seed in MODEL_SEEDS:
        path = CAR_SEED_SCORE_PATHS[seed]
        if not path.exists():
            raise FileNotFoundError(f"missing frozen CAR scores: {path}")
        payload = _load_npz(path)
        for name in ("labels", "test_mask", "H64", "HFULL"):
            if name not in payload:
                raise ValueError(f"{path} is missing {name}")
        replicate_arrays[seed] = {
            "labels": np.asarray(payload["labels"]),
            "test_mask": np.asarray(payload["test_mask"], dtype=bool),
            "H64": np.asarray(payload["H64"], dtype=np.float64),
            "HFULL": np.asarray(payload["HFULL"], dtype=np.float64),
        }
    alignment = _validate_replicate_alignment(
        labels=labels_full,
        test_mask=test_mask,
        replicates=replicate_arrays,
    )

    test_index = np.flatnonzero(test_mask)
    labels = labels_full[test_index]
    source_keys = source_key_full[test_index]
    speaker_ids = speaker_full[test_index]
    conditions = condition_full[test_index]
    noise_names = noise_full[test_index]
    short_scores: dict[int, np.ndarray] = {}
    long_scores: dict[int, np.ndarray] = {}
    taxonomy: dict[int, dict[str, np.ndarray]] = {}
    values: dict[int, np.ndarray] = {}
    for seed in MODEL_SEEDS:
        short = replicate_arrays[seed]["H64"][test_index]
        long = replicate_arrays[seed]["HFULL"][test_index]
        if not np.all(np.isfinite(short)):
            raise ValueError(f"non-finite H64 values for seed {seed}")
        if not np.all(np.isfinite(long)):
            raise ValueError(f"non-finite HFULL values for seed {seed}")
        short_scores[seed] = short
        long_scores[seed] = long
        tax = _taxonomy_from_scores(labels, short, long)
        taxonomy[seed] = tax
        values[seed] = np.asarray(tax["v_log"], dtype=np.float64)

    segments = _reconstruct_segments(labels=labels_full)
    event_features_full = _build_event_features(
        labels=labels_full,
        short_scores=reference_short,
        segments=segments,
    )
    event_features = {
        name: values_full[test_index]
        for name, values_full in event_features_full.items()
    }
    return E4Data(
        seeds=MODEL_SEEDS,
        labels_full=labels_full,
        test_mask=test_mask,
        test_index=test_index,
        labels=labels,
        source_keys=source_keys,
        speaker_ids=speaker_ids,
        conditions=conditions,
        noise_names=noise_names,
        short_scores=short_scores,
        long_scores=long_scores,
        taxonomy=taxonomy,
        values=values,
        segments=segments,
        event_features=event_features,
        source_index=_source_index(source_keys),
    )


def _seed_taxonomy_rows(data: E4Data) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in data.seeds:
        tax = data.taxonomy[seed]
        values = np.asarray(data.values[seed], dtype=np.float64)
        rows.append(
            {
                "scope": "seed",
                "seed": int(seed),
                "frames": int(data.labels.size),
                "P_SS": float(np.mean(tax["SS"])),
                "P_R": float(np.mean(tax["R"])),
                "P_I": float(np.mean(tax["I"])),
                "P_H": float(np.mean(tax["H"])),
                "NetRefinability": float(np.mean(tax["R"]) - np.mean(tax["H"])),
                "mean_v_log": float(np.mean(values)),
                "mean_v_brier": float(
                    np.mean(data.taxonomy[seed]["v_brier"])
                ),
            }
        )
    rows.append(
        {
            "scope": "aggregate",
            "seed": "aggregate",
            "frames": int(data.labels.size),
            "P_SS": float(np.mean([np.mean(data.taxonomy[s]["SS"]) for s in data.seeds])),
            "P_R": float(np.mean([np.mean(data.taxonomy[s]["R"]) for s in data.seeds])),
            "P_I": float(np.mean([np.mean(data.taxonomy[s]["I"]) for s in data.seeds])),
            "P_H": float(np.mean([np.mean(data.taxonomy[s]["H"]) for s in data.seeds])),
            "NetRefinability": float(
                np.mean([np.mean(data.taxonomy[s]["R"]) for s in data.seeds])
                - np.mean([np.mean(data.taxonomy[s]["H"]) for s in data.seeds])
            ),
            "mean_v_log": float(
                np.mean(
                    [
                        np.mean(data.values[s])
                        for s in data.seeds
                    ]
                )
            ),
            "mean_v_brier": float(
                np.mean(
                    [
                        np.mean(data.taxonomy[s]["v_brier"])
                        for s in data.seeds
                    ]
                )
            ),
        }
    )
    return rows


def _pairwise_taxonomy_rows(data: E4Data) -> list[dict[str, Any]]:
    codes = {seed: _taxonomy_codes(data.taxonomy[seed]) for seed in data.seeds}
    refinable = {
        seed: np.asarray(data.taxonomy[seed]["R"], dtype=bool)
        for seed in data.seeds
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(data.seeds, 2):
        left_r = refinable[left]
        right_r = refinable[right]
        union = int(np.count_nonzero(left_r | right_r))
        intersection = int(np.count_nonzero(left_r & right_r))
        rows.append(
            {
                "row_type": "pair",
                "left_seed": int(left),
                "right_seed": int(right),
                "frames": int(data.labels.size),
                "taxonomy_agreement": float(
                    np.mean(codes[left] == codes[right])
                ),
                "r_jaccard": (
                    float(intersection / union)
                    if union
                    else None
                ),
                "r_intersection_frames": intersection,
                "r_union_frames": union,
            }
        )
    return rows


def _baseline_reproduction(data: E4Data) -> dict[str, Any]:
    seed_rows = _seed_taxonomy_rows(data)
    aggregate = next(row for row in seed_rows if row["scope"] == "aggregate")
    pairwise = _pairwise_taxonomy_rows(data)
    reference_taxonomy = pd.read_csv(CAR_TAXONOMY_REFERENCE_PATH)
    reference_pairs = pd.read_csv(CAR_AGREEMENT_REFERENCE_PATH)
    reference_summary = _load_json(CAR_FINAL_SUMMARY_PATH)

    errors: list[dict[str, Any]] = []
    count_checks: list[dict[str, Any]] = []
    reference_seed = {
        int(row["seed"]): row
        for _, row in reference_taxonomy[
            (reference_taxonomy["scope"] == "seed")
            & (reference_taxonomy["group"] == "all")
        ].iterrows()
    }
    for row in seed_rows:
        if row["scope"] != "seed":
            continue
        seed = int(row["seed"])
        ref = reference_seed.get(seed)
        if ref is None:
            count_checks.append(
                {"seed": seed, "check": "reference_row", "passed": False}
            )
            continue
        count_checks.append(
            {
                "seed": seed,
                "check": "frames",
                "observed": int(row["frames"]),
                "expected": int(ref["frames"]),
                "passed": int(row["frames"]) == int(ref["frames"]),
            }
        )
        for metric, column in (
            ("P_SS", "p_SS"),
            ("P_R", "p_R"),
            ("P_I", "p_I"),
            ("P_H", "p_H"),
            ("NetRefinability", "availability_net"),
            ("mean_v_log", "mean_value"),
        ):
            observed = float(row[metric])
            expected = float(ref[column])
            errors.append(
                {
                    "scope": "seed",
                    "seed": seed,
                    "metric": metric,
                    "observed": observed,
                    "expected": expected,
                    "abs_error": abs(observed - expected),
                }
            )

    reference_pair_lookup = {
        (int(row["left_seed"]), int(row["right_seed"])): row
        for _, row in reference_pairs[
            reference_pairs["row_type"] == "pair"
        ].iterrows()
    }
    for row in pairwise:
        key = (int(row["left_seed"]), int(row["right_seed"]))
        ref = reference_pair_lookup.get(key)
        if ref is None:
            count_checks.append(
                {
                    "pair": key,
                    "check": "reference_pair",
                    "passed": False,
                }
            )
            continue
        count_checks.append(
            {
                "pair": key,
                "check": "frames",
                "observed": int(row["frames"]),
                "expected": int(ref["frames"]),
                "passed": int(row["frames"]) == int(ref["frames"]),
            }
        )
        for metric, column in (
            ("taxonomy_agreement", "taxonomy_agreement"),
            ("r_jaccard", "r_jaccard"),
        ):
            observed = float(row[metric])
            expected = float(ref[column])
            errors.append(
                {
                    "scope": "pair",
                    "left_seed": int(row["left_seed"]),
                    "right_seed": int(row["right_seed"]),
                    "metric": metric,
                    "observed": observed,
                    "expected": expected,
                    "abs_error": abs(observed - expected),
                }
            )

    reference_aggregate = reference_summary["car1_aggregate"]
    for metric, column in (
        ("P_SS", "p_SS"),
        ("P_R", "p_R"),
        ("P_I", "p_I"),
        ("P_H", "p_H"),
        ("NetRefinability", "availability_net"),
        ("mean_v_log", "mean_value"),
    ):
        observed = float(aggregate[metric])
        expected = float(reference_aggregate[column])
        errors.append(
            {
                "scope": "aggregate",
                "metric": metric,
                "observed": observed,
                "expected": expected,
                "abs_error": abs(observed - expected),
            }
        )

    max_abs_error = max(
        [float(item["abs_error"]) for item in errors],
        default=float("inf"),
    )
    count_pass = all(bool(item["passed"]) for item in count_checks)
    reproduced = bool(
        errors
        and count_checks
        and count_pass
        and max_abs_error <= BASELINE_TOLERANCE
    )
    return {
        "BASELINE_REPRODUCED": reproduced,
        "FRAME_ALIGNMENT_VERIFIED": True,
        "tolerance": BASELINE_TOLERANCE,
        "max_abs_error": max_abs_error,
        "count_checks_passed": count_pass,
        "reference_paths": {
            "taxonomy": str(CAR_TAXONOMY_REFERENCE_PATH.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "agreement": str(CAR_AGREEMENT_REFERENCE_PATH.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "summary": str(CAR_FINAL_SUMMARY_PATH.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
        },
        "errors": errors,
        "count_checks": count_checks,
        "seed_taxonomy": [
            row for row in seed_rows if row["scope"] == "seed"
        ],
        "aggregate_taxonomy": aggregate,
        "pairwise": pairwise,
    }


def _replicate_matrix(data: E4Data, category: str) -> np.ndarray:
    """Return a replicate-by-frame boolean matrix for one taxonomy class."""
    return np.stack(
        [
            np.asarray(data.taxonomy[seed][category], dtype=bool)
            for seed in data.seeds
        ],
        axis=0,
    )


def _count_matrix_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=bool)
    return np.sum(matrix, axis=0, dtype=np.int64)


def _stability_masks(k_r: np.ndarray) -> dict[str, np.ndarray]:
    k_r = np.asarray(k_r, dtype=np.int64)
    return {
        "NEVER_R": k_r == 0,
        "UNSTABLE_R": (k_r >= 1) & (k_r <= 2),
        "STABLE_R": k_r >= 3,
        "CORE_R": k_r >= 4,
        "EVER_R": k_r >= 1,
        "STABLE_K34": (k_r >= 3) & (k_r <= 4),
    }


def _independence_k_pmf(probabilities: Sequence[float]) -> np.ndarray:
    pmf = np.asarray([1.0], dtype=np.float64)
    for probability in probabilities:
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError("independence probability must be in [0, 1]")
        pmf = np.convolve(pmf, np.asarray([1.0 - p, p]), mode="full")
    return pmf


def _k_distribution_rows(
    *,
    k_r: np.ndarray,
    k_i: np.ndarray,
    k_h: np.ndarray,
    independence_pmf: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_frames = int(np.asarray(k_r).size)
    for variable, counts in (
        ("K_R", np.asarray(k_r, dtype=np.int64)),
        ("K_I", np.asarray(k_i, dtype=np.int64)),
        ("K_H", np.asarray(k_h, dtype=np.int64)),
    ):
        for k in range(len(MODEL_SEEDS) + 1):
            frame_count = int(np.count_nonzero(counts == k))
            row: dict[str, Any] = {
                "variable": variable,
                "k": int(k),
                "frames": frame_count,
                "P": float(frame_count / n_frames),
            }
            if variable == "K_R":
                row["independence_expected_P"] = float(
                    independence_pmf[k]
                )
            rows.append(row)
    return rows


def _taxonomy_summary_rows(
    *,
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    mean_v_log: np.ndarray,
    mean_v_brier: np.ndarray,
) -> list[dict[str, Any]]:
    n_frames = int(np.asarray(k_r).size)
    r_assignment_total = int(np.sum(k_r))
    rows: list[dict[str, Any]] = []
    for category in STABILITY_CATEGORY_ORDER:
        mask = np.asarray(masks[category], dtype=bool)
        frame_count = int(np.count_nonzero(mask))
        r_assignments = int(np.sum(k_r[mask]))
        ever = np.asarray(masks["EVER_R"], dtype=bool)
        rows.append(
            {
                "category": category,
                "frames": frame_count,
                "P_of_all_frames": float(frame_count / n_frames),
                "P_given_ever_R": (
                    float(frame_count / int(np.count_nonzero(ever)))
                    if np.any(ever)
                    else float("nan")
                ),
                "R_assignments": r_assignments,
                "share_of_R_assignments": (
                    float(r_assignments / r_assignment_total)
                    if r_assignment_total
                    else float("nan")
                ),
                "mean_v_log": float(
                    np.mean(mean_v_log[mask]) if frame_count else float("nan")
                ),
                "mean_v_brier": float(
                    np.mean(mean_v_brier[mask])
                    if frame_count
                    else float("nan")
                ),
            }
        )
    return rows


def _permute_within_strata(
    mask: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute a boolean mask within fixed strata, preserving each count."""
    mask = np.asarray(mask, dtype=bool)
    strata = np.asarray(strata)
    result = np.zeros(mask.size, dtype=bool)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        selected = int(np.count_nonzero(mask[indices]))
        if selected:
            permuted = rng.permutation(indices)
            result[permuted[:selected]] = True
    return result


def _permutation_overlap_rows(
    *,
    r_matrix: np.ndarray,
    source_keys: np.ndarray,
    conditions: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_keys = np.asarray(source_keys, dtype=str)
    conditions = np.asarray(conditions, dtype=str)
    strata = np.asarray(
        [
            f"{source}\x1f{condition}"
            for source, condition in zip(source_keys, conditions)
        ],
        dtype=str,
    )
    rows: list[dict[str, Any]] = []
    for permutation_seed in PERMUTATION_SEEDS:
        rng = np.random.default_rng(int(permutation_seed))
        permuted = np.stack(
            [
                _permute_within_strata(r_matrix[index], strata, rng)
                for index in range(r_matrix.shape[0])
            ],
            axis=0,
        )
        k = _count_matrix_rows(permuted)
        rows.append(
            {
                "permutation_seed": int(permutation_seed),
                "P_K_R_GE_3": float(np.mean(k >= 3)),
                "P_K_R_GE_4": float(np.mean(k >= 4)),
                "P_K_R_GE_5": float(np.mean(k >= 5)),
                "P_STABLE_R": float(np.mean(k >= 3)),
                "P_CORE_R": float(np.mean(k >= 4)),
            }
        )
    summary: list[dict[str, Any]] = []
    for metric in ("P_K_R_GE_3", "P_K_R_GE_4"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        low, high = np.quantile(values, [0.025, 0.975])
        summary.append(
            {
                "metric": metric,
                "permutation_count": int(values.size),
                "permutation_mean": float(np.mean(values)),
                "permutation_median": float(np.median(values)),
                "permutation_ci95_low": float(low),
                "permutation_ci95_high": float(high),
            }
        )
    return rows, summary


def _null_overlap_rows(
    *,
    k_r: np.ndarray,
    independence_pmf: np.ndarray,
    permutation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_ge_3 = float(np.mean(np.asarray(k_r) >= 3))
    observed_ge_4 = float(np.mean(np.asarray(k_r) >= 4))
    independence_ge_3 = float(np.sum(independence_pmf[3:]))
    independence_ge_4 = float(np.sum(independence_pmf[4:]))
    for label, observed, independence, metric in (
        (
            "K_R_GE_3",
            observed_ge_3,
            independence_ge_3,
            "P_K_R_GE_3",
        ),
        (
            "K_R_GE_4",
            observed_ge_4,
            independence_ge_4,
            "P_K_R_GE_4",
        ),
    ):
        permutation_values = np.asarray(
            [float(row[metric]) for row in permutation_rows],
            dtype=np.float64,
        )
        permutation_mean = float(np.mean(permutation_values))
        reference = max(float(independence), permutation_mean)
        rows.append(
            {
                "statistic": label,
                "observed": observed,
                "independence_expected": float(independence),
                "permutation_mean": permutation_mean,
                "permutation_ci95_low": float(
                    np.quantile(permutation_values, 0.025)
                ),
                "permutation_ci95_high": float(
                    np.quantile(permutation_values, 0.975)
                ),
                "null_reference": reference,
                "observed_minus_null": observed - reference,
                "exceeds_null_point": bool(observed > reference),
            }
        )
    return rows


def _source_sums(
    values: np.ndarray,
    mask: np.ndarray | None,
    sources: SourceIndex,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        selected = np.ones(values.size, dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool)
    if values.size != selected.size:
        raise ValueError("values and mask lengths differ")
    weighted = np.where(selected, values, 0.0)
    sums = np.bincount(
        sources.codes,
        weights=weighted,
        minlength=sources.count,
    )
    counts = np.bincount(
        sources.codes,
        weights=selected.astype(np.float64),
        minlength=sources.count,
    )
    return sums, counts


def _ratio_from_sums(
    sums: np.ndarray,
    counts: np.ndarray,
    selected_sources: np.ndarray | None = None,
) -> float:
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if selected_sources is None:
        selected_sources = np.arange(sums.size, dtype=np.int64)
    denominator = float(np.sum(counts[selected_sources]))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(sums[selected_sources]) / denominator)


def _difference_from_sums(
    left_sums: np.ndarray,
    left_counts: np.ndarray,
    right_sums: np.ndarray,
    right_counts: np.ndarray,
    selected_sources: np.ndarray | None = None,
) -> float:
    left = _ratio_from_sums(left_sums, left_counts, selected_sources)
    right = _ratio_from_sums(right_sums, right_counts, selected_sources)
    return left - right if np.isfinite(left) and np.isfinite(right) else float("nan")


def _bootstrap_ratio(
    values: np.ndarray,
    mask: np.ndarray | None,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    sums, counts = _source_sums(values, mask, sources)
    all_sources = np.arange(sources.count, dtype=np.int64)
    estimate = _ratio_from_sums(sums, counts, all_sources)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sources.count, size=sources.count)
        estimates[index] = _ratio_from_sums(sums, counts, selected)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_difference(
    values: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    left_sums, left_counts = _source_sums(values, left_mask, sources)
    right_sums, right_counts = _source_sums(values, right_mask, sources)
    all_sources = np.arange(sources.count, dtype=np.int64)
    estimate = _difference_from_sums(
        left_sums,
        left_counts,
        right_sums,
        right_counts,
        all_sources,
    )
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sources.count, size=sources.count)
        estimates[index] = _difference_from_sums(
            left_sums,
            left_counts,
            right_sums,
            right_counts,
            selected,
        )
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_excess(
    indicator: np.ndarray,
    null_value: float,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    centered = np.asarray(indicator, dtype=np.float64) - float(null_value)
    return _bootstrap_ratio(
        centered,
        np.ones(centered.size, dtype=bool),
        sources,
        seed=seed,
    )


def _frame_observables(data: E4Data) -> dict[str, np.ndarray]:
    labels = np.asarray(data.labels, dtype=np.int64)
    short_scores = np.stack(
        [data.short_scores[seed] for seed in data.seeds],
        axis=0,
    )
    long_scores = np.stack(
        [data.long_scores[seed] for seed in data.seeds],
        axis=0,
    )
    short_correct = (
        short_scores >= DECISION_THRESHOLD
    ) == labels[None, :].astype(bool)
    long_correct = (
        long_scores >= DECISION_THRESHOLD
    ) == labels[None, :].astype(bool)
    v_log = np.stack(
        [data.values[seed] for seed in data.seeds],
        axis=0,
    )
    v_brier = np.stack(
        [
            np.asarray(data.taxonomy[seed]["v_brier"], dtype=np.float64)
            for seed in data.seeds
        ],
        axis=0,
    )
    short_logloss = np.stack(
        [
            np.asarray(data.taxonomy[seed]["short_loss"], dtype=np.float64)
            for seed in data.seeds
        ],
        axis=0,
    )
    short_wrong = ~short_correct
    conditional_denominator = np.sum(short_wrong, axis=0)
    conditional_numerator = np.sum(short_wrong & long_correct, axis=0)
    conditional_probability = np.divide(
        conditional_numerator,
        conditional_denominator,
        out=np.full(conditional_denominator.shape, np.nan, dtype=np.float64),
        where=conditional_denominator > 0,
    )
    return {
        "short_scores": short_scores,
        "long_scores": long_scores,
        "short_correct": short_correct,
        "long_correct": long_correct,
        "v_log": v_log,
        "v_brier": v_brier,
        "short_logloss": short_logloss,
        "mean_v_log": np.mean(v_log, axis=0),
        "median_v_log": np.median(v_log, axis=0),
        "mean_v_brier": np.mean(v_brier, axis=0),
        "positive_value_count": np.sum(v_log > 0.0, axis=0),
        "mean_short_posterior": np.mean(short_scores, axis=0),
        "mean_short_margin": np.mean(np.abs(short_scores - 0.5), axis=0),
        "mean_short_correctness": np.mean(short_correct, axis=0),
        "mean_short_logloss": np.mean(short_logloss, axis=0),
        "short_wrong_count": np.sum(short_wrong, axis=0),
        "long_correct_count": np.sum(long_correct, axis=0),
        "conditional_long_correct_given_short_wrong": conditional_probability,
    }


def _proper_scoring_rows(
    *,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in STABILITY_CATEGORY_ORDER:
        mask = np.asarray(masks[category], dtype=bool)
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "category": category,
                "frames": count,
                "mean_v_log": float(
                    np.mean(observables["mean_v_log"][mask])
                    if count
                    else float("nan")
                ),
                "median_v_log": float(
                    np.median(observables["median_v_log"][mask])
                    if count
                    else float("nan")
                ),
                "mean_v_brier": float(
                    np.mean(observables["mean_v_brier"][mask])
                    if count
                    else float("nan")
                ),
                "mean_positive_value_count": float(
                    np.mean(observables["positive_value_count"][mask])
                    if count
                    else float("nan")
                ),
                "positive_frame_share": float(
                    np.mean(observables["mean_v_log"][mask] > 0.0)
                    if count
                    else float("nan")
                ),
            }
        )
    return rows


def _short_difficulty_rows(
    *,
    k_r: np.ndarray,
    k_i: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    group_masks = {
        "UNSTABLE_R": masks["UNSTABLE_R"],
        "STABLE_R": masks["STABLE_R"],
        "CORE_R": masks["CORE_R"],
        "I_DOMINANT": (
            (np.asarray(k_i, dtype=np.int64) >= 3)
            & (np.asarray(k_r, dtype=np.int64) == 0)
        ),
    }
    rows: list[dict[str, Any]] = []
    for group, mask in group_masks.items():
        mask = np.asarray(mask, dtype=bool)
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "group": group,
                "frames": count,
                "mean_short_posterior": float(
                    np.mean(observables["mean_short_posterior"][mask])
                    if count
                    else float("nan")
                ),
                "mean_abs_margin_from_0_5": float(
                    np.mean(observables["mean_short_margin"][mask])
                    if count
                    else float("nan")
                ),
                "mean_short_correctness": float(
                    np.mean(observables["mean_short_correctness"][mask])
                    if count
                    else float("nan")
                ),
                "mean_short_logloss": float(
                    np.mean(observables["mean_short_logloss"][mask])
                    if count
                    else float("nan")
                ),
                "mean_v_log": float(
                    np.mean(observables["mean_v_log"][mask])
                    if count
                    else float("nan")
                ),
            }
        )
    return rows


def _event_bin(value: float) -> str:
    distance = abs(float(value))
    if distance == 0.0:
        return "0"
    if distance <= 1.0:
        return "1"
    if distance <= 4.0:
        return "2-4"
    if distance <= 10.0:
        return "5-10"
    if distance <= 25.0:
        return "11-25"
    if distance <= 50.0:
        return "26-50"
    return ">50"


def _event_stability_rows(
    *,
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
    event_features: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in EVENT_DIMENSIONS:
        values = np.asarray(event_features[dimension], dtype=np.float64)
        for event_bin in EVENT_BINS:
            mask = np.asarray(
                [_event_bin(value) == event_bin for value in values],
                dtype=bool,
            )
            count = int(np.count_nonzero(mask))
            rows.append(
                {
                    "dimension": dimension,
                    "event_bin": event_bin,
                    "frames": count,
                    "P_K_R_GE_1": float(
                        np.mean(masks["EVER_R"][mask])
                        if count
                        else float("nan")
                    ),
                    "P_STABLE_R": float(
                        np.mean(masks["STABLE_R"][mask])
                        if count
                        else float("nan")
                    ),
                    "P_CORE_R": float(
                        np.mean(masks["CORE_R"][mask])
                        if count
                        else float("nan")
                    ),
                    "mean_K_R": float(
                        np.mean(k_r[mask]) if count else float("nan")
                    ),
                    "mean_v_log": float(
                        np.mean(observables["mean_v_log"][mask])
                        if count
                        else float("nan")
                    ),
                    "mean_v_brier": float(
                        np.mean(observables["mean_v_brier"][mask])
                        if count
                        else float("nan")
                    ),
                }
            )
    return rows


def _condition_stability_rows(
    *,
    data: E4Data,
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    conditions = np.asarray(data.conditions, dtype=str)
    noise = np.asarray(data.noise_names, dtype=str)
    clean = noise == "Clean"
    unseen = np.isin(noise, np.asarray(UNSEEN_NOISE, dtype=str))
    seen = (~clean) & (~unseen)
    group_masks: list[tuple[str, str, np.ndarray]] = [
        ("seen_unseen", "seen", seen),
        ("seen_unseen", "unseen", unseen),
        ("seen_unseen", "clean", clean),
    ]
    for snr in SNR_ORDER:
        group_masks.append(
            (
                "snr",
                snr,
                (~clean) & (conditions == snr),
            )
        )
    for noise_name in sorted(np.unique(noise).tolist()):
        group_masks.append(
            (
                "noise_domain",
                str(noise_name),
                noise == noise_name,
            )
        )
    rows: list[dict[str, Any]] = []
    for group_type, group, mask in group_masks:
        mask = np.asarray(mask, dtype=bool)
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "frames": count,
                "P_K_R_GE_1": float(
                    np.mean(masks["EVER_R"][mask])
                    if count
                    else float("nan")
                ),
                "P_STABLE_R": float(
                    np.mean(masks["STABLE_R"][mask])
                    if count
                    else float("nan")
                ),
                "P_CORE_R": float(
                    np.mean(masks["CORE_R"][mask])
                    if count
                    else float("nan")
                ),
                "mean_K_R": float(
                    np.mean(k_r[mask]) if count else float("nan")
                ),
                "mean_v_log": float(
                    np.mean(observables["mean_v_log"][mask])
                    if count
                    else float("nan")
                ),
            }
        )
    return rows, {"seen": seen, "unseen": unseen, "clean": clean}


def _disagreement_decomposition_rows(
    *,
    k_r: np.ndarray,
    r_matrix: np.ndarray,
    taxonomy_matrices: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    k_r = np.asarray(k_r, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    selections: list[tuple[str, np.ndarray]] = [
        (str(k), k_r == k) for k in (1, 2, 3, 4)
    ]
    selections.append(("1-4", (k_r >= 1) & (k_r <= 4)))
    for label, frame_mask in selections:
        frame_mask = np.asarray(frame_mask, dtype=bool)
        non_r = ~np.asarray(r_matrix[:, frame_mask], dtype=bool)
        if non_r.size == 0:
            continue
        counts = {
            category: int(
                np.count_nonzero(
                    non_r
                    & np.asarray(
                        taxonomy_matrices[category][:, frame_mask],
                        dtype=bool,
                    )
                )
            )
            for category in ("SS", "I", "H")
        }
        denominator = int(sum(counts.values()))
        rows.append(
            {
                "K_R_category": label,
                "frames": int(np.count_nonzero(frame_mask)),
                "non_R_assignments": denominator,
                "P_non_R_SS": (
                    float(counts["SS"] / denominator)
                    if denominator
                    else float("nan")
                ),
                "P_non_R_I": (
                    float(counts["I"] / denominator)
                    if denominator
                    else float("nan")
                ),
                "P_non_R_H": (
                    float(counts["H"] / denominator)
                    if denominator
                    else float("nan")
                ),
                "non_R_SS_count": counts["SS"],
                "non_R_I_count": counts["I"],
                "non_R_H_count": counts["H"],
            }
        )
    return rows


def _short_long_instability_rows(
    *,
    k_r: np.ndarray,
    observables: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    k_r = np.asarray(k_r, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for k in range(len(MODEL_SEEDS) + 1):
        mask = k_r == k
        count = int(np.count_nonzero(mask))
        conditional = observables["conditional_long_correct_given_short_wrong"][
            mask
        ]
        finite = conditional[np.isfinite(conditional)]
        rows.append(
            {
                "K_R": int(k),
                "frames": count,
                "mean_short_wrong_count": float(
                    np.mean(observables["short_wrong_count"][mask])
                    if count
                    else float("nan")
                ),
                "mean_long_correct_count": float(
                    np.mean(observables["long_correct_count"][mask])
                    if count
                    else float("nan")
                ),
                "mean_P_long_correct_given_short_wrong": float(
                    np.mean(finite) if finite.size else float("nan")
                ),
                "mean_short_wrong_rate": float(
                    np.mean(observables["short_wrong_count"][mask] / len(MODEL_SEEDS))
                    if count
                    else float("nan")
                ),
                "mean_long_correct_rate": float(
                    np.mean(
                        observables["long_correct_count"][mask]
                        / len(MODEL_SEEDS)
                    )
                    if count
                    else float("nan")
                ),
            }
        )
    return rows


def _bootstrap_row(
    statistic: str,
    estimate: float,
    low: float,
    high: float,
    *,
    frames: int,
) -> dict[str, Any]:
    return {
        "statistic": statistic,
        "estimate": float(estimate),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "frames": int(frames),
        "direction_supported": bool(
            np.isfinite(low)
            and np.isfinite(high)
            and (low > 0.0 or high < 0.0)
        ),
    }


def _bootstrap_stability_rows(
    *,
    data: E4Data,
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
    event_features: Mapping[str, np.ndarray],
    condition_masks: Mapping[str, np.ndarray],
    null_rows: Sequence[Mapping[str, Any]],
    r_matrix: np.ndarray,
    taxonomy_matrices: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    sources = data.source_index
    stable = np.asarray(masks["STABLE_R"], dtype=bool)
    unstable = np.asarray(masks["UNSTABLE_R"], dtype=bool)
    core = np.asarray(masks["CORE_R"], dtype=bool)
    rows: list[dict[str, Any]] = []

    estimate, low, high = _bootstrap_ratio(
        stable.astype(np.float64),
        None,
        sources,
        seed=_stable_hash_seed("bootstrap:P_STABLE_R"),
    )
    rows.append(_bootstrap_row("P_STABLE_R", estimate, low, high, frames=stable.size))

    estimate, low, high = _bootstrap_ratio(
        core.astype(np.float64),
        None,
        sources,
        seed=_stable_hash_seed("bootstrap:P_CORE_R"),
    )
    rows.append(_bootstrap_row("P_CORE_R", estimate, low, high, frames=core.size))

    null_lookup = {
        str(row["statistic"]): row for row in null_rows
    }
    for statistic, indicator_name in (
        ("observed_minus_null_K_GE_3", "K_R_GE_3"),
        ("observed_minus_null_K_GE_4", "K_R_GE_4"),
    ):
        null_value = float(
            null_lookup[indicator_name]["null_reference"]
        )
        indicator = (
            np.asarray(k_r) >= (3 if indicator_name.endswith("3") else 4)
        ).astype(np.float64)
        estimate, low, high = _bootstrap_excess(
            indicator,
            null_value,
            sources,
            seed=_stable_hash_seed(f"bootstrap:{statistic}"),
        )
        rows.append(
            _bootstrap_row(
                statistic,
                estimate,
                low,
                high,
                frames=indicator.size,
            )
        )

    estimate, low, high = _bootstrap_ratio(
        observables["mean_v_log"],
        stable,
        sources,
        seed=_stable_hash_seed("bootstrap:stable_mean_v_log"),
    )
    rows.append(
        _bootstrap_row(
            "stable_mean_v_log",
            estimate,
            low,
            high,
            frames=int(np.count_nonzero(stable)),
        )
    )
    estimate, low, high = _bootstrap_ratio(
        observables["mean_v_log"],
        unstable,
        sources,
        seed=_stable_hash_seed("bootstrap:unstable_mean_v_log"),
    )
    rows.append(
        _bootstrap_row(
            "unstable_mean_v_log",
            estimate,
            low,
            high,
            frames=int(np.count_nonzero(unstable)),
        )
    )
    estimate, low, high = _bootstrap_difference(
        observables["mean_v_log"],
        stable,
        unstable,
        sources,
        seed=_stable_hash_seed("bootstrap:stable_minus_unstable_vlog"),
    )
    rows.append(
        _bootstrap_row(
            "stable_minus_unstable_v_log",
            estimate,
            low,
            high,
            frames=int(np.count_nonzero(stable | unstable)),
        )
    )
    estimate, low, high = _bootstrap_difference(
        observables["mean_short_margin"],
        stable,
        unstable,
        sources,
        seed=_stable_hash_seed(
            "bootstrap:stable_minus_unstable_short_margin"
        ),
    )
    rows.append(
        _bootstrap_row(
            "stable_minus_unstable_short_margin",
            estimate,
            low,
            high,
            frames=int(np.count_nonzero(stable | unstable)),
        )
    )

    event_contrasts = {
        "posterior_transition_0_vs_beyond_stable": (
            "posterior_transition_distance"
        ),
        "onset_0_vs_beyond_stable": "onset_distance",
        "offset_0_vs_beyond_stable": "offset_distance",
    }
    for statistic, dimension in event_contrasts.items():
        values = np.asarray(event_features[dimension], dtype=np.float64)
        exact = np.asarray(
            [_event_bin(value) == "0" for value in values],
            dtype=bool,
        )
        beyond = ~exact
        estimate, low, high = _bootstrap_difference(
            stable.astype(np.float64),
            exact,
            beyond,
            sources,
            seed=_stable_hash_seed(f"bootstrap:{statistic}"),
        )
        rows.append(
            _bootstrap_row(
                statistic,
                estimate,
                low,
                high,
                frames=int(np.count_nonzero(exact | beyond)),
            )
        )

    estimate, low, high = _bootstrap_difference(
        stable.astype(np.float64),
        condition_masks["seen"],
        condition_masks["unseen"],
        sources,
        seed=_stable_hash_seed("bootstrap:seen_vs_unseen_stable"),
    )
    rows.append(
        _bootstrap_row(
            "seen_vs_unseen_stable",
            estimate,
            low,
            high,
            frames=int(
                np.count_nonzero(
                    condition_masks["seen"] | condition_masks["unseen"]
                )
            ),
        )
    )

    n_frames = int(np.asarray(k_r).size)
    n_replicates = len(data.seeds)
    slot_sources = SourceIndex(
        names=sources.names,
        codes=np.repeat(sources.codes, n_replicates),
        count=sources.count,
    )
    slot_non_r = (~np.asarray(r_matrix, dtype=bool)).T.reshape(-1)
    slot_i = np.asarray(taxonomy_matrices["I"], dtype=bool).T.reshape(-1)
    slot_stable_k34 = np.repeat(
        np.asarray(masks["STABLE_K34"], dtype=bool),
        n_replicates,
    )
    slot_unstable = np.repeat(
        np.asarray(masks["UNSTABLE_R"], dtype=bool),
        n_replicates,
    )
    estimate, low, high = _bootstrap_difference(
        slot_i.astype(np.float64),
        slot_stable_k34 & slot_non_r,
        slot_unstable & slot_non_r,
        slot_sources,
        seed=_stable_hash_seed(
            "bootstrap:stable_vs_unstable_non_r_i_share"
        ),
    )
    rows.append(
        _bootstrap_row(
            "stable_vs_unstable_non_r_i_share",
            estimate,
            low,
            high,
            frames=int(
                np.count_nonzero(
                    (slot_stable_k34 | slot_unstable) & slot_non_r
                )
            ),
        )
    )
    return rows


def _loso_ratio(
    values: np.ndarray,
    mask: np.ndarray | None,
    sources: SourceIndex,
) -> list[float]:
    sums, counts = _source_sums(values, mask, sources)
    estimates: list[float] = []
    for source_index in range(sources.count):
        selected = np.asarray(
            [
                index
                for index in range(sources.count)
                if index != source_index
            ],
            dtype=np.int64,
        )
        estimate = _ratio_from_sums(sums, counts, selected)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    return estimates


def _loso_difference(
    values: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    sources: SourceIndex,
) -> list[float]:
    left_sums, left_counts = _source_sums(values, left_mask, sources)
    right_sums, right_counts = _source_sums(values, right_mask, sources)
    estimates: list[float] = []
    for source_index in range(sources.count):
        selected = np.asarray(
            [
                index
                for index in range(sources.count)
                if index != source_index
            ],
            dtype=np.int64,
        )
        estimate = _difference_from_sums(
            left_sums,
            left_counts,
            right_sums,
            right_counts,
            selected,
        )
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    return estimates


def _loso_row(
    statistic: str,
    full_estimate: float,
    estimates: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(estimates, dtype=np.float64)
    if values.size == 0:
        sign_consistent = False
        low = high = float("nan")
    else:
        low = float(np.min(values))
        high = float(np.max(values))
        if full_estimate > 0.0:
            sign_consistent = bool(np.all(values > 0.0))
        elif full_estimate < 0.0:
            sign_consistent = bool(np.all(values < 0.0))
        else:
            sign_consistent = bool(np.all(values == 0.0))
    return {
        "statistic": statistic,
        "full_estimate": float(full_estimate),
        "loso_count": int(values.size),
        "loso_min": low,
        "loso_max": high,
        "sign_consistent": sign_consistent,
    }


def _source_influence_rows(
    *,
    data: E4Data,
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    observables: Mapping[str, np.ndarray],
    event_features: Mapping[str, np.ndarray],
    condition_masks: Mapping[str, np.ndarray],
    r_matrix: np.ndarray,
    taxonomy_matrices: Mapping[str, np.ndarray],
    null_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sources = data.source_index
    stable = np.asarray(masks["STABLE_R"], dtype=bool)
    unstable = np.asarray(masks["UNSTABLE_R"], dtype=bool)
    rows: list[dict[str, Any]] = []
    full = float(np.mean(stable))
    rows.append(
        _loso_row(
            "P_STABLE_R",
            full,
            _loso_ratio(stable.astype(np.float64), None, sources),
        )
    )
    null_lookup = {str(row["statistic"]): row for row in null_rows}
    for statistic, key, threshold in (
        ("observed_minus_null_K_GE_3", "K_R_GE_3", 3),
        ("observed_minus_null_K_GE_4", "K_R_GE_4", 4),
    ):
        null_value = float(null_lookup[key]["null_reference"])
        indicator = (np.asarray(k_r) >= threshold).astype(np.float64)
        full_estimate = float(np.mean(indicator) - null_value)
        rows.append(
            _loso_row(
                statistic,
                full_estimate,
                _loso_ratio(
                    indicator - null_value,
                    None,
                    sources,
                ),
            )
        )
    for statistic, values, left, right in (
        (
            "stable_minus_unstable_v_log",
            observables["mean_v_log"],
            stable,
            unstable,
        ),
        (
            "stable_minus_unstable_short_margin",
            observables["mean_short_margin"],
            stable,
            unstable,
        ),
    ):
        full_estimate = float(
            np.mean(values[left]) - np.mean(values[right])
        )
        rows.append(
            _loso_row(
                statistic,
                full_estimate,
                _loso_difference(values, left, right, sources),
            )
        )
    for statistic, dimension in (
        (
            "posterior_transition_0_vs_beyond_stable",
            "posterior_transition_distance",
        ),
        ("onset_0_vs_beyond_stable", "onset_distance"),
        ("offset_0_vs_beyond_stable", "offset_distance"),
    ):
        values = np.asarray(event_features[dimension], dtype=np.float64)
        exact = np.asarray(
            [_event_bin(value) == "0" for value in values],
            dtype=bool,
        )
        full_estimate = float(
            np.mean(stable[exact]) - np.mean(stable[~exact])
        )
        rows.append(
            _loso_row(
                statistic,
                full_estimate,
                _loso_difference(
                    stable.astype(np.float64),
                    exact,
                    ~exact,
                    sources,
                ),
            )
        )
    full_estimate = float(
        np.mean(stable[condition_masks["seen"]])
        - np.mean(stable[condition_masks["unseen"]])
    )
    rows.append(
        _loso_row(
            "seen_vs_unseen_stable",
            full_estimate,
            _loso_difference(
                stable.astype(np.float64),
                condition_masks["seen"],
                condition_masks["unseen"],
                sources,
            ),
        )
    )

    n_replicates = len(data.seeds)
    slot_sources = SourceIndex(
        names=sources.names,
        codes=np.repeat(sources.codes, n_replicates),
        count=sources.count,
    )
    slot_non_r = (~np.asarray(r_matrix, dtype=bool)).T.reshape(-1)
    slot_i = np.asarray(taxonomy_matrices["I"], dtype=bool).T.reshape(-1)
    slot_stable_k34 = np.repeat(
        np.asarray(masks["STABLE_K34"], dtype=bool),
        n_replicates,
    )
    slot_unstable = np.repeat(
        np.asarray(masks["UNSTABLE_R"], dtype=bool),
        n_replicates,
    )
    left = slot_stable_k34 & slot_non_r
    right = slot_unstable & slot_non_r
    full_estimate = float(
        np.mean(slot_i[left]) - np.mean(slot_i[right])
    )
    rows.append(
        _loso_row(
            "stable_vs_unstable_non_r_i_share",
            full_estimate,
            _loso_difference(
                slot_i.astype(np.float64),
                left,
                right,
                slot_sources,
            ),
        )
    )
    return rows


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    value: str,
    *,
    value_key: str = "statistic",
) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get(value_key)) == value:
            return row
    raise KeyError(f"missing row {value_key}={value}")


def _candidate_passes(
    row: Mapping[str, Any],
    minimum_abs_difference: float,
) -> bool:
    estimate = float(row.get("estimate", float("nan")))
    low = float(row.get("ci95_low", float("nan")))
    high = float(row.get("ci95_high", float("nan")))
    return bool(
        np.isfinite(estimate)
        and np.isfinite(low)
        and np.isfinite(high)
        and abs(estimate) >= float(minimum_abs_difference)
        and (low > 0.0 or high < 0.0)
    )


def _build_gates_and_claims(
    *,
    baseline: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    k_r: np.ndarray,
    masks: Mapping[str, np.ndarray],
    null_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
    source_codes: np.ndarray,
) -> dict[str, Any]:
    source_codes = np.asarray(source_codes, dtype=np.int64)
    if source_codes.size != np.asarray(k_r).size:
        raise ValueError("source_codes and k_r lengths differ")
    p_stable = float(np.mean(masks["STABLE_R"]))
    p_core = float(np.mean(masks["CORE_R"]))
    p_ever = float(np.mean(masks["EVER_R"]))
    p_stable_given_ever = (
        p_stable / p_ever if p_ever > 0.0 else float("nan")
    )
    p_core_given_ever = (
        p_core / p_ever if p_ever > 0.0 else float("nan")
    )
    stable_sources = int(
        np.unique(
            source_codes[np.asarray(masks["STABLE_R"], dtype=bool)]
        ).size
    )
    core_sources = int(
        np.unique(
            source_codes[np.asarray(masks["CORE_R"], dtype=bool)]
        ).size
    )
    null3 = _row_lookup(null_rows, "statistic", "K_R_GE_3")
    null4 = _row_lookup(null_rows, "statistic", "K_R_GE_4")
    boot3 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_3",
    )
    boot4 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_4",
    )
    stable_mean_row = _row_lookup(
        bootstrap_rows,
        "statistic",
        "stable_mean_v_log",
    )
    contrast_row = _row_lookup(
        bootstrap_rows,
        "statistic",
        "stable_minus_unstable_v_log",
    )
    loso_contrast = _row_lookup(
        loso_rows,
        "statistic",
        "stable_minus_unstable_v_log",
    )
    aggregate = next(
        (
            row
            for row in seed_rows
            if str(row.get("scope")) == "aggregate"
        ),
        {},
    )
    seed_net_positive = bool(
        seed_rows
        and all(
            float(row.get("NetRefinability", float("nan"))) > 0.0
            for row in seed_rows
            if str(row.get("scope")) == "seed"
        )
    )
    population_reproducible = bool(
        baseline.get("BASELINE_REPRODUCED", False)
        and seed_net_positive
        and float(aggregate.get("mean_v_log", float("nan"))) > 0.0
    )
    g1 = bool(
        bool(null3.get("exceeds_null_point", False))
        and float(boot3.get("ci95_low", float("nan"))) > 0.0
    )
    g2 = bool(
        float(stable_mean_row.get("estimate", float("nan"))) > 0.0
        and float(contrast_row.get("estimate", float("nan"))) > 0.0
        and float(contrast_row.get("ci95_low", float("nan"))) > 0.0
        and bool(loso_contrast.get("sign_consistent", False))
    )
    candidates = {
        name: _candidate_passes(
            _row_lookup(bootstrap_rows, "statistic", name),
            minimum,
        )
        for name, minimum in G3_MIN_ABS_DIFFERENCE.items()
    }
    event_candidates = {
        key: value
        for key, value in candidates.items()
        if key.endswith("_stable")
        and key
        in {
            "posterior_transition_0_vs_beyond_stable",
            "onset_0_vs_beyond_stable",
            "offset_0_vs_beyond_stable",
        }
    }
    g3 = bool(any(candidates.values()))
    nontrivial_stable = bool(
        p_stable >= NONTRIVIAL_STABLE_R_MIN
        and p_core >= NONTRIVIAL_CORE_R_MIN
        and stable_sources >= NONTRIVIAL_MIN_STABLE_SOURCES
        and core_sources >= NONTRIVIAL_MIN_CORE_SOURCES
    )
    observed_excess = float(null3.get("observed_minus_null", float("nan")))
    near_null = bool(
        abs(observed_excess) <= NEAR_NULL_ABS_EXCESS_MAX
        or (
            float(null3.get("observed", float("nan")))
            <= float(null3.get("null_reference", float("nan")))
            * NEAR_NULL_RATIO_MAX
            and float(boot3.get("ci95_low", float("nan"))) <= 0.0
        )
    )
    if not population_reproducible:
        status = "INCONCLUSIVE_OR_INVALID"
    elif g1 and g2 and nontrivial_stable:
        status = "STABLE_REFINABLE_CORE"
    elif g1 or observed_excess > 0.0:
        status = "POPULATION_STABLE_INSTANCE_RELATIVE"
    elif near_null or observed_excess <= 0.0:
        status = "MOSTLY_MODEL_RELATIVE_REFINABILITY"
    else:
        status = "POPULATION_STABLE_INSTANCE_RELATIVE"

    if population_reproducible:
        claim_a = "SUPPORTED"
    else:
        claim_a = "NOT_SUPPORTED"
    if status == "STABLE_REFINABLE_CORE":
        claim_b = "SUPPORTED"
    elif g1:
        claim_b = "CONDITIONAL"
    else:
        claim_b = "NOT_SUPPORTED"
    if population_reproducible and (
        not g1
        or p_stable_given_ever <= CLAIM_C_STABLE_SHARE_MAX
        or p_core_given_ever <= CLAIM_C_CORE_SHARE_MAX
    ):
        claim_c = "SUPPORTED"
    elif g1:
        claim_c = "CONDITIONAL"
    else:
        claim_c = "NOT_SUPPORTED"
    short_margin_pass = bool(candidates["stable_minus_unstable_short_margin"])
    non_short_candidates = [
        value
        for key, value in candidates.items()
        if key != "stable_minus_unstable_short_margin"
    ]
    if short_margin_pass and not any(non_short_candidates):
        claim_d = "SUPPORTED"
    elif not short_margin_pass:
        claim_d = "NOT_SUPPORTED"
    else:
        claim_d = "UNRESOLVED"
    if population_reproducible and (
        p_stable_given_ever <= CLAIM_C_STABLE_SHARE_MAX
        or p_core_given_ever <= CLAIM_C_CORE_SHARE_MAX
    ):
        claim_e = "SUPPORTED_AS_HYPOTHESIS"
    else:
        claim_e = "UNRESOLVED"

    event_result = (
        "ROBUST_DIFFERENCE"
        if any(event_candidates.values())
        else "NO_ROBUST_DIFFERENCE"
    )
    condition_result = (
        "ROBUST_DIFFERENCE"
        if candidates["seen_vs_unseen_stable"]
        else "NO_ROBUST_DIFFERENCE"
    )
    short_result = (
        "ROBUST_DIFFERENCE"
        if short_margin_pass
        else "NO_ROBUST_DIFFERENCE"
    )
    disagreement_result = (
        "ROBUST_DIFFERENCE"
        if candidates["stable_vs_unstable_non_r_i_share"]
        else "NO_ROBUST_DIFFERENCE"
    )
    return {
        "E4_STATUS": status,
        "POPULATION_REPRODUCIBLE": population_reproducible,
        "P_STABLE_R": p_stable,
        "P_CORE_R": p_core,
        "P_EVER_R": p_ever,
        "P_STABLE_GIVEN_EVER_R": p_stable_given_ever,
        "P_CORE_GIVEN_EVER_R": p_core_given_ever,
        "NONTRIVIAL_STABLE": nontrivial_stable,
        "NEAR_NULL": near_null,
        "G1_STABLE_CORE": g1,
        "G2_STABLE_CORE_VALUE": g2,
        "G3_STRUCTURED_INSTABILITY": g3,
        "G1_DETAILS": {
            "observed_K_GE_3": float(null3["observed"]),
            "null_reference_K_GE_3": float(null3["null_reference"]),
            "observed_minus_null_K_GE_3": float(
                null3["observed_minus_null"]
            ),
            "observed_minus_null_ci95_low": float(
                boot3["ci95_low"]
            ),
            "observed_K_GE_4": float(null4["observed"]),
            "null_reference_K_GE_4": float(null4["null_reference"]),
            "observed_minus_null_K_GE_4": float(
                null4["observed_minus_null"]
            ),
            "observed_minus_null_ci95_low_K_GE_4": float(
                boot4["ci95_low"]
            ),
        },
        "G2_DETAILS": {
            "stable_mean_v_log": float(stable_mean_row["estimate"]),
            "stable_minus_unstable_v_log": float(contrast_row["estimate"]),
            "stable_minus_unstable_v_log_ci95_low": float(
                contrast_row["ci95_low"]
            ),
            "loso_sign_consistent": bool(
                loso_contrast.get("sign_consistent", False)
            ),
        },
        "G3_DETAILS": candidates,
        "EVENT_STABILITY_RESULT": event_result,
        "CONDITION_STABILITY_RESULT": condition_result,
        "SHORT_DIFFICULTY_RESULT": short_result,
        "DISAGREEMENT_STRUCTURE_RESULT": disagreement_result,
        "CLAIM_A": claim_a,
        "CLAIM_B": claim_b,
        "CLAIM_C": claim_c,
        "CLAIM_D": claim_d,
        "CLAIM_E": claim_e,
        "STABLE_SOURCE_COUNT": int(stable_sources),
        "CORE_SOURCE_COUNT": int(core_sources),
        "NONTRIVIAL_SOURCE_SUPPORT": bool(
            stable_sources >= NONTRIVIAL_MIN_STABLE_SOURCES
            and core_sources >= NONTRIVIAL_MIN_CORE_SOURCES
        ),
    }


def _protocol_payload() -> dict[str, Any]:
    frozen_inputs: dict[str, Path] = {
        "rf384_frame_reference": RF384_REFERENCE_PATH,
        "car_taxonomy_reference": CAR_TAXONOMY_REFERENCE_PATH,
        "car_seed_agreement_reference": CAR_AGREEMENT_REFERENCE_PATH,
        "car_final_summary": CAR_FINAL_SUMMARY_PATH,
        "test_manifest": TEST_MANIFEST_PATH,
        "e3_final_summary": E3_FINAL_SUMMARY_PATH,
        "car_runner": CAR_RUNNER_PATH,
        "ae_mechanism_runner": AE_MECHANISM_RUNNER_PATH,
    }
    for seed, path in CAR_SEED_SCORE_PATHS.items():
        frozen_inputs[f"car_seed_{seed}_scores"] = path
    return {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PROTOCOL_STATUS": PROTOCOL_STATUS,
        "CREATED_UTC": datetime.now(timezone.utc).isoformat(),
        "ANALYSIS_ONLY": True,
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "REPLICATE_IDENTITIES": list(MODEL_SEEDS),
        "PRIMARY_MODEL": "RF384_SHORT_vs_HFULL",
        "PRIMARY_ARCHITECTURE": "TinyGRU_CAR_replicates",
        "POPULATION": {
            "test_frames": EXPECTED_TEST_FRAMES,
            "test_sources": EXPECTED_TEST_SOURCES,
            "test_speakers": EXPECTED_TEST_SPEAKERS,
            "total_frames": EXPECTED_TOTAL_FRAMES,
            "valid_start": VALID_START,
            "frame_hop": FRAME_HOP,
            "labels_and_test_mask_source": str(
                RF384_REFERENCE_PATH.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "score_source": (
                "results/cross_architecture_replication/evaluations/"
                "seed{seed}/scores.npz"
            ),
        },
        "METADATA_PROVENANCE": {
            "source_and_condition_metadata": (
                "The CAR score bundles do not contain source or condition "
                "metadata. Source, speaker, condition, and noise identity are "
                "taken from the authoritative frozen RF384 frame reference "
                "after verifying that labels and test_mask are identical "
                "across all five CAR score bundles."
            ),
            "replicate_alignment_checks": (
                "labels_equal and test_mask_equal for each of the five "
                "frozen CAR score bundles"
            ),
        },
        "PRE_FREEZE_INPUT_CHECK": {
            "status": "PASSED_INPUT_INTEGRITY_ONLY",
            "n_frames": 298_300,
            "n_sources": 96,
            "n_speakers": 20,
            "n_segments": 1024,
            "baseline_max_abs_error": 1.1102230246251565e-16,
            "aggregate_taxonomy": {
                "P_SS": 0.8621870600067046,
                "P_R": 0.020581293999329536,
                "P_I": 0.10800536372779082,
                "P_H": 0.009226282266174992,
                "NetRefinability": 0.011355011733154544,
                "mean_v_log": 0.020043899306062256,
                "mean_v_brier": 0.0070263799128756745,
            },
            "note": (
                "This was an input-integrity and reproduction check only. "
                "No E4 stability endpoint was inspected before freeze."
            ),
        },
        "DECISION_THRESHOLD": DECISION_THRESHOLD,
        "TAXONOMY": {
            "SS": "Short correct and Full correct",
            "R": "Short wrong and Full correct",
            "I": "Short wrong and Full wrong",
            "H": "Short correct and Full wrong",
            "NetRefinability": "P(R) - P(H)",
        },
        "STABILITY_CATEGORIES": {
            "NEVER_R": "K_R = 0",
            "UNSTABLE_R": "K_R = 1 or 2",
            "STABLE_R": "K_R = 3, 4, or 5",
            "CORE_R": "K_R = 4 or 5",
        },
        "PROPER_SCORING": {
            "primary": "v_log(t) = LogLoss(Short,t) - LogLoss(Full,t)",
            "secondary": "v_brier(t) = Brier(Short,t) - Brier(Full,t)",
            "logloss_epsilon": LOGLOSS_EPSILON,
        },
        "OBSERVABLES": [
            "mean_v_log",
            "median_v_log",
            "positive_value_count",
            "mean_short_posterior",
            "mean_abs_margin_from_0.5",
            "mean_short_correctness",
            "mean_short_logloss",
            "short_wrong_count",
            "long_correct_count",
            "conditional_long_correct_given_short_wrong",
        ],
        "EVENT_DEFINITIONS": {
            "bins": list(EVENT_BINS),
            "dimensions": list(EVENT_DIMENSIONS),
            "onset_and_offset": (
                "signed frame distance to the nearest ground-truth onset or "
                "offset inside each reconstructed evaluation utterance"
            ),
            "posterior_transition": (
                "absolute frame distance to the nearest RF384 Short "
                "threshold-0.5 transition inside each reconstructed utterance"
            ),
            "missing_distance": MISSING_DISTANCE,
        },
        "CONDITION_DEFINITIONS": {
            "seen": (
                "non-clean frames whose noise domain is not one of the frozen "
                "unseen noise domains"
            ),
            "unseen": list(UNSEEN_NOISE),
            "snr_bins": list(SNR_ORDER),
            "noise_domains": (
                "the existing frozen noise_name values, including Clean"
            ),
        },
        "PERMUTATION_NULL": {
            "seeds": list(PERMUTATION_SEEDS),
            "method": (
                "one realization per seed; independently permute each "
                "replicate's R membership within source x condition strata "
                "while preserving replicate-level R prevalence"
            ),
            "primary_comparison": (
                "observed P(K_R >= 3) and P(K_R >= 4) versus the maximum of "
                "the independence reference and permutation mean"
            ),
        },
        "BOOTSTRAP": {
            "method": "source-cluster percentile bootstrap",
            "repeats": BOOTSTRAP_REPEATS,
            "confidence": 0.95,
            "seed_base": BOOTSTRAP_SEED_BASE,
        },
        "PRIMARY_GATES": {
            "G1_STABLE_CORE": (
                "observed P(K_R >= 3) exceeds the frozen independence/"
                "permutation reference and source-cluster 95% CI lower bound "
                "for excess overlap is > 0"
            ),
            "G2_STABLE_CORE_VALUE": (
                "STABLE_R mean cross-seed v_log > 0, stable-minus-unstable "
                "v_log contrast > 0 with CI lower bound > 0, and leave-one-"
                "source-out sign consistency"
            ),
            "G3_STRUCTURED_INSTABILITY": (
                "at least one preregistered event, margin, condition, or "
                "disagreement contrast is robust in the frozen direction"
            ),
        },
        "NONTRIVIAL_RULES": {
            "stable_R_min": NONTRIVIAL_STABLE_R_MIN,
            "core_R_min": NONTRIVIAL_CORE_R_MIN,
            "min_stable_sources": NONTRIVIAL_MIN_STABLE_SOURCES,
            "min_core_sources": NONTRIVIAL_MIN_CORE_SOURCES,
            "near_null_abs_excess_max": NEAR_NULL_ABS_EXCESS_MAX,
            "near_null_ratio_max": NEAR_NULL_RATIO_MAX,
            "claim_c_stable_share_max": CLAIM_C_STABLE_SHARE_MAX,
            "claim_c_core_share_max": CLAIM_C_CORE_SHARE_MAX,
        },
        "G3_MIN_ABS_DIFFERENCE": dict(G3_MIN_ABS_DIFFERENCE),
        "FINAL_STATUS_RULES": {
            "STABLE_REFINABLE_CORE": (
                "G1 PASS, G2 PASS, and nontrivial stable/core R population"
            ),
            "POPULATION_STABLE_INSTANCE_RELATIVE": (
                "population-level R prevalence/value remains reproducible, "
                "but individual-frame stable overlap is limited or mixed "
                "while still exceeding null expectation"
            ),
            "MOSTLY_MODEL_RELATIVE_REFINABILITY": (
                "population-level phenomenon remains reproducible, but "
                "individual R membership is close to the frozen null "
                "reference and no meaningful stable core is supported"
            ),
            "INCONCLUSIVE_OR_INVALID": (
                "failed reproduction, alignment, integrity, or required "
                "statistical validity"
            ),
            "NOT_EXECUTABLE_FROM_FROZEN_DATA": (
                "required aligned frozen predictions do not exist and "
                "execution would require retraining"
            ),
        },
        "FROZEN_INPUTS": {
            name: _file_record(path)
            for name, path in frozen_inputs.items()
        },
        "RUNNER": {
            "module": RUNNER_MODULE,
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "sha256": _runner_hash(),
        },
    }


def _freeze_protocol() -> int:
    if PROTOCOL_PATH.exists() or RUNNER_HASH_PATH.exists():
        _validate_frozen_protocol()
        print(
            json.dumps(
                {
                    "mode": "freeze_protocol",
                    "status": "ALREADY_FROZEN",
                    "protocol_sha256": _sha256_file(PROTOCOL_PATH),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    payload = _protocol_payload()
    _write_json(PROTOCOL_PATH, payload)
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    PROTOCOL_HASH_PATH.write_text(protocol_hash + "\n", encoding="ascii")
    runner_hash = _runner_hash()
    RUNNER_HASH_PATH.write_text(runner_hash + "\n", encoding="ascii")
    print(
        json.dumps(
            {
                "mode": "freeze_protocol",
                "status": "FROZEN",
                "protocol_sha256": protocol_hash,
                "runner_sha256": runner_hash,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _validate_frozen_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"E4 protocol is not frozen: {PROTOCOL_PATH}"
        )
    if not PROTOCOL_HASH_PATH.exists() or not RUNNER_HASH_PATH.exists():
        raise FileNotFoundError("E4 freeze hash files are missing")
    payload = _load_json(PROTOCOL_PATH)
    if payload.get("PROTOCOL_ID") != PROTOCOL_ID:
        raise ValueError("E4 protocol id does not match the runner")
    if payload.get("PROTOCOL_STATUS") != PROTOCOL_STATUS:
        raise ValueError("E4 protocol status does not match the runner")
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    recorded_protocol_hash = PROTOCOL_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    runner_hash = _runner_hash()
    recorded_runner_hash = RUNNER_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    if protocol_hash != recorded_protocol_hash:
        raise ValueError("E4 protocol hash does not match its freeze file")
    if runner_hash != recorded_runner_hash:
        raise ValueError("E4 runner hash does not match its freeze file")
    return {
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner_hash,
    }


def _read_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            "E4 baseline reproduction must be run before analysis"
        )
    return _load_json(BASELINE_PATH)


def _build_final_summary(
    *,
    data: E4Data,
    baseline: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    k_distribution_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
    permutation_summary: Sequence[Mapping[str, Any]],
    proper_rows: Sequence[Mapping[str, Any]],
    short_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
    short_long_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    k_r_rows = {
        int(row["k"]): row
        for row in k_distribution_rows
        if str(row.get("variable")) == "K_R"
    }
    null3 = _row_lookup(null_rows, "statistic", "K_R_GE_3")
    null4 = _row_lookup(null_rows, "statistic", "K_R_GE_4")
    boot3 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_3",
    )
    boot4 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_4",
    )
    stable_mean = _row_lookup(
        bootstrap_rows,
        "statistic",
        "stable_mean_v_log",
    )
    unstable_mean = _row_lookup(
        bootstrap_rows,
        "statistic",
        "unstable_mean_v_log",
    )
    contrast = _row_lookup(
        bootstrap_rows,
        "statistic",
        "stable_minus_unstable_v_log",
    )
    disagreement = next(
        (
            row
            for row in disagreement_rows
            if str(row.get("K_R_category")) == "1-4"
        ),
        {},
    )
    loso_stable = _row_lookup(
        loso_rows,
        "statistic",
        "P_STABLE_R",
    )
    deviations = [
        (
            "CAR score bundles do not embed source/condition metadata; "
            "source and condition identity were taken from the authoritative "
            "RF384 frame reference after replicate label/test-mask alignment "
            "verification."
        ),
        (
            "Pre-freeze input integrity reproduction passed at max_abs_error "
            "1.1102230246251565e-16; this was recorded before stability "
            "outcome inspection and is not an E4 result."
        ),
    ]
    return {
        "E4_STATUS": str(gates["E4_STATUS"]),
        "BASELINE_REPRODUCED": bool(
            baseline.get("BASELINE_REPRODUCED", False)
        ),
        "FRAME_ALIGNMENT_VERIFIED": bool(
            baseline.get("FRAME_ALIGNMENT_VERIFIED", False)
        ),
        "N_FRAMES": int(data.labels.size),
        "N_SOURCES": int(data.source_index.count),
        "N_REPLICATES": int(len(data.seeds)),
        "P_KR_0": float(k_r_rows[0]["P"]),
        "P_KR_1": float(k_r_rows[1]["P"]),
        "P_KR_2": float(k_r_rows[2]["P"]),
        "P_KR_3": float(k_r_rows[3]["P"]),
        "P_KR_4": float(k_r_rows[4]["P"]),
        "P_KR_5": float(k_r_rows[5]["P"]),
        "P_STABLE_R": float(gates["P_STABLE_R"]),
        "P_CORE_R": float(gates["P_CORE_R"]),
        "P_STABLE_GIVEN_EVER_R": float(
            gates["P_STABLE_GIVEN_EVER_R"]
        ),
        "P_CORE_GIVEN_EVER_R": float(gates["P_CORE_GIVEN_EVER_R"]),
        "EXPECTED_STABLE_R_NULL": float(null3["null_reference"]),
        "OBSERVED_MINUS_NULL_STABLE_R": float(
            null3["observed_minus_null"]
        ),
        "OBSERVED_MINUS_NULL_CI": [
            float(boot3["ci95_low"]),
            float(boot3["ci95_high"]),
        ],
        "G1_STABLE_CORE": bool(gates["G1_STABLE_CORE"]),
        "G2_STABLE_CORE_VALUE": bool(gates["G2_STABLE_CORE_VALUE"]),
        "G3_STRUCTURED_INSTABILITY": bool(
            gates["G3_STRUCTURED_INSTABILITY"]
        ),
        "STABLE_R_MEAN_VLOG": float(stable_mean["estimate"]),
        "UNSTABLE_R_MEAN_VLOG": float(unstable_mean["estimate"]),
        "STABLE_MINUS_UNSTABLE_VLOG": float(contrast["estimate"]),
        "STABLE_MINUS_UNSTABLE_VLOG_CI": [
            float(contrast["ci95_low"]),
            float(contrast["ci95_high"]),
        ],
        "R_NONREPLICATE_TO_SS": float(
            disagreement.get("P_non_R_SS", float("nan"))
        ),
        "R_NONREPLICATE_TO_I": float(
            disagreement.get("P_non_R_I", float("nan"))
        ),
        "R_NONREPLICATE_TO_H": float(
            disagreement.get("P_non_R_H", float("nan"))
        ),
        "EVENT_STABILITY_RESULT": str(
            gates["EVENT_STABILITY_RESULT"]
        ),
        "CONDITION_STABILITY_RESULT": str(
            gates["CONDITION_STABILITY_RESULT"]
        ),
        "SHORT_DIFFICULTY_RESULT": str(
            gates["SHORT_DIFFICULTY_RESULT"]
        ),
        "DISAGREEMENT_STRUCTURE_RESULT": str(
            gates["DISAGREEMENT_STRUCTURE_RESULT"]
        ),
        "SOURCE_LOSO_STABILITY": list(loso_rows),
        "SOURCE_LOSO_SIGN_CONSISTENT": bool(
            loso_stable.get("sign_consistent", False)
        ),
        "CLAIM_A": str(gates["CLAIM_A"]),
        "CLAIM_B": str(gates["CLAIM_B"]),
        "CLAIM_C": str(gates["CLAIM_C"]),
        "CLAIM_D": str(gates["CLAIM_D"]),
        "CLAIM_E": str(gates["CLAIM_E"]),
        "PROTOCOL_DEVIATIONS": deviations,
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "E2_STATUS": "INCONCLUSIVE_OR_INVALID",
        "E3_STATUS": "ROBUST_VALUE_STRUCTURE",
        "E4_DETAILS": {
            "G1": dict(gates["G1_DETAILS"]),
            "G2": dict(gates["G2_DETAILS"]),
            "G3": dict(gates["G3_DETAILS"]),
            "NONTRIVIAL_STABLE": bool(gates["NONTRIVIAL_STABLE"]),
            "NEAR_NULL": bool(gates["NEAR_NULL"]),
            "STABLE_SOURCE_COUNT": int(gates["STABLE_SOURCE_COUNT"]),
            "CORE_SOURCE_COUNT": int(gates["CORE_SOURCE_COUNT"]),
            "NONTRIVIAL_SOURCE_SUPPORT": bool(
                gates["NONTRIVIAL_SOURCE_SUPPORT"]
            ),
            "OBSERVED_MINUS_NULL_K_GE_4": float(
                null4["observed_minus_null"]
            ),
            "OBSERVED_MINUS_NULL_CI_K_GE_4": [
                float(boot4["ci95_low"]),
                float(boot4["ci95_high"]),
            ],
        },
        "STABILITY_TAXONOMY": list(taxonomy_rows),
        "K_DISTRIBUTION": list(k_distribution_rows),
        "NULL_OVERLAP": list(null_rows),
        "PERMUTATION_SUMMARY": list(permutation_summary),
        "PROPER_SCORING": list(proper_rows),
        "SHORT_DIFFICULTY": list(short_rows),
        "EVENT_STABILITY": list(event_rows),
        "CONDITION_STABILITY": list(condition_rows),
        "DISAGREEMENT_DECOMPOSITION": list(disagreement_rows),
        "SHORT_LONG_INSTABILITY": list(short_long_rows),
        "BOOTSTRAP": list(bootstrap_rows),
    }


def _status_interpretation(status: str) -> str:
    if status == "STABLE_REFINABLE_CORE":
        return (
            "A reproducible stable refinable core exists across independently "
            "trained model pairs. This does not establish sample-intrinsic or "
            "architecture-independent refinability."
        )
    if status == "POPULATION_STABLE_INSTANCE_RELATIVE":
        return (
            "Population-level refinability is reproducible, while individual "
            "refinable-frame identity is only partially stable across models."
        )
    if status == "MOSTLY_MODEL_RELATIVE_REFINABILITY":
        return (
            "The population-level phenomenon is reproducible, but the "
            "identity of individual decision-changing frames is largely "
            "model-relative. This does not say that temporal value itself is "
            "model-relative."
        )
    if status == "INCONCLUSIVE_OR_INVALID":
        return (
            "The frozen inputs or required statistical validity checks did "
            "not pass, so no E4 mechanism interpretation is made."
        )
    return (
        "The required aligned frozen predictions are not executable without "
        "retraining, so no E4 mechanism interpretation is made."
    )


def _write_report(
    *,
    summary: Mapping[str, Any],
    k_distribution_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
) -> None:
    status = str(summary["E4_STATUS"])
    null3 = _row_lookup(null_rows, "statistic", "K_R_GE_3")
    null4 = _row_lookup(null_rows, "statistic", "K_R_GE_4")
    boot3 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_3",
    )
    boot4 = _row_lookup(
        bootstrap_rows,
        "statistic",
        "observed_minus_null_K_GE_4",
    )
    lines = [
        "# E4 Refinability Stability Decomposition",
        "",
        f"- E4 status: `{status}`",
        f"- Baseline reproduced: `{summary['BASELINE_REPRODUCED']}`",
        f"- Frame alignment verified: `{summary['FRAME_ALIGNMENT_VERIFIED']}`",
        f"- Frames: `{summary['N_FRAMES']}`",
        f"- Sources: `{summary['N_SOURCES']}`",
        f"- Replicates: `{summary['N_REPLICATES']}`",
        "",
        "## Primary Stability Endpoints",
        "",
        (
            f"P(STABLE_R)={float(summary['P_STABLE_R']):.8f}, "
            f"P(CORE_R)={float(summary['P_CORE_R']):.8f}, "
            f"P(STABLE_R | K_R>=1)="
            f"{float(summary['P_STABLE_GIVEN_EVER_R']):.8f}, "
            f"P(CORE_R | K_R>=1)="
            f"{float(summary['P_CORE_GIVEN_EVER_R']):.8f}."
        ),
        "",
        (
            f"Observed P(K_R>=3)={float(null3['observed']):.8f} versus "
            f"frozen null={float(null3['null_reference']):.8f}; "
            f"excess={float(null3['observed_minus_null']):.8f}, "
            f"source-cluster CI95=[{float(boot3['ci95_low']):.8f}, "
            f"{float(boot3['ci95_high']):.8f}]."
        ),
        (
            f"Observed P(K_R>=4)={float(null4['observed']):.8f} versus "
            f"frozen null={float(null4['null_reference']):.8f}; "
            f"excess={float(null4['observed_minus_null']):.8f}, "
            f"source-cluster CI95=[{float(boot4['ci95_low']):.8f}, "
            f"{float(boot4['ci95_high']):.8f}]."
        ),
        "",
        "## Gates And Claims",
        "",
        f"- G1 stable core: `{summary['G1_STABLE_CORE']}`",
        f"- G2 stable core value: `{summary['G2_STABLE_CORE_VALUE']}`",
        (
            "- G3 structured instability: "
            f"`{summary['G3_STRUCTURED_INSTABILITY']}`"
        ),
        f"- Claim A: `{summary['CLAIM_A']}`",
        f"- Claim B: `{summary['CLAIM_B']}`",
        f"- Claim C: `{summary['CLAIM_C']}`",
        f"- Claim D: `{summary['CLAIM_D']}`",
        f"- Claim E: `{summary['CLAIM_E']}`",
        "",
        "## Stability Categories",
        "",
        "| category | frames | P(all) | P(given ever R) | mean v_log |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in taxonomy_rows:
        lines.append(
            f"| {row['category']} | {int(row['frames'])} | "
            f"{float(row['P_of_all_frames']):.8f} | "
            f"{float(row['P_given_ever_R']):.8f} | "
            f"{float(row['mean_v_log']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## K_R Distribution",
            "",
            "| K_R | frames | observed P | independence expected P |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in k_distribution_rows:
        if str(row.get("variable")) != "K_R":
            continue
        lines.append(
            f"| {int(row['k'])} | {int(row['frames'])} | "
            f"{float(row['P']):.8f} | "
            f"{float(row['independence_expected_P']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Non-R Fate Among Frames With K_R 1-4",
            "",
            "| K_R category | frames | non-R assignments | P(SS) | P(I) | P(H) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in disagreement_rows:
        lines.append(
            f"| {row['K_R_category']} | {int(row['frames'])} | "
            f"{int(row['non_R_assignments'])} | "
            f"{float(row['P_non_R_SS']):.8f} | "
            f"{float(row['P_non_R_I']):.8f} | "
            f"{float(row['P_non_R_H']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _status_interpretation(status),
            "",
            (
                "E3 remains `ROBUST_VALUE_STRUCTURE`; E4 studies only the "
                "stability of decision-changing R membership and does not "
                "reinterpret probabilistic-value frames as R."
            ),
            "",
            (
                "Claim E remains hypothesis-level. E4 cannot establish that "
                "R instability causes M2's limited OOD gain."
            ),
            "",
            "No training, checkpoint change, OOD access, or next experiment "
            "was performed.",
        ]
    )
    _write_markdown(OUTPUT_ROOT / "e4_final_report.md", lines)
    claim_lines = [
        "# E4 Claim Freeze",
        "",
        f"E4_STATUS={status}",
        f"CLAIM_A={summary['CLAIM_A']}",
        f"CLAIM_B={summary['CLAIM_B']}",
        f"CLAIM_C={summary['CLAIM_C']}",
        f"CLAIM_D={summary['CLAIM_D']}",
        f"CLAIM_E={summary['CLAIM_E']}",
        "",
        "Claim E is frozen only as a hypothesis-level interpretation.",
        "These labels were frozen after the pre-registered E4 analysis.",
    ]
    _write_markdown(OUTPUT_ROOT / "e4_claim_freeze.md", claim_lines)


def _write_figures(
    *,
    k_distribution_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    k_frame = pd.DataFrame(k_distribution_rows)
    k_r_frame = k_frame[k_frame["variable"].astype(str) == "K_R"].copy()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        k_r_frame["k"],
        k_r_frame["P"],
        label="Observed",
    )
    axis.plot(
        k_r_frame["k"],
        k_r_frame["independence_expected_P"],
        marker="o",
        label="Independence reference",
    )
    axis.set_xlabel("K_R")
    axis.set_ylabel("Frame probability")
    axis.set_title("Distribution of replicate refinability counts")
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[0]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    taxonomy_frame = pd.DataFrame(taxonomy_rows).set_index("category")
    categories = list(STABILITY_CATEGORY_ORDER)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        np.arange(len(categories)),
        [
            float(taxonomy_frame.loc[category, "mean_v_log"])
            for category in categories
        ],
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(categories)))
    axis.set_xticklabels(categories, rotation=20, ha="right")
    axis.set_ylabel("Mean cross-seed v_log")
    axis.set_title("Proper-scoring value by stability category")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[1]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    disagreement = next(
        row
        for row in disagreement_rows
        if str(row.get("K_R_category")) == "1-4"
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    fate_labels = ["SS", "I", "H"]
    fate_values = [
        float(disagreement["P_non_R_SS"]),
        float(disagreement["P_non_R_I"]),
        float(disagreement["P_non_R_H"]),
    ]
    axis.bar(np.arange(len(fate_labels)), fate_values)
    axis.set_xticks(np.arange(len(fate_labels)))
    axis.set_xticklabels(fate_labels)
    axis.set_ylabel("Conditional proportion")
    axis.set_title("Non-R fate for frames with K_R 1-4")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[2]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    event_frame = pd.DataFrame(event_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, dimension in zip(axes, EVENT_DIMENSIONS):
        subset = event_frame[
            event_frame["dimension"].astype(str) == dimension
        ].set_index("event_bin").reindex(EVENT_BINS)
        axis.bar(
            np.arange(len(EVENT_BINS)),
            subset["P_STABLE_R"].to_numpy(dtype=np.float64),
        )
        axis.set_xticks(np.arange(len(EVENT_BINS)))
        axis.set_xticklabels(EVENT_BINS, rotation=45, ha="right")
        axis.set_title(dimension)
        axis.set_xlabel("distance bin")
    axes[0].set_ylabel("P(STABLE_R)")
    fig.suptitle("Stable refinability by frozen event bin")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[3]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    condition_frame = pd.DataFrame(condition_rows)
    seen_unseen = condition_frame[
        condition_frame["group_type"].astype(str) == "seen_unseen"
    ].set_index("group")
    condition_labels = ["seen", "unseen"]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        np.arange(len(condition_labels)),
        [
            float(seen_unseen.loc[label, "P_STABLE_R"])
            for label in condition_labels
        ],
    )
    axis.set_xticks(np.arange(len(condition_labels)))
    axis.set_xticklabels(condition_labels)
    axis.set_ylabel("P(STABLE_R)")
    axis.set_title("Stable refinability by seen/unseen condition")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[4]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)
    return figure_paths


def _write_execution_manifest(
    *,
    protocol_hash: str,
    runner_hash: str,
    baseline: Mapping[str, Any],
    summary: Mapping[str, Any],
    runtime_seconds: float,
    figure_paths: Sequence[Path],
) -> None:
    output_paths = [
        OUTPUT_ROOT / name
        for name in REQUIRED_OUTPUT_FILES
        if name != EXECUTION_MANIFEST_PATH.name
    ]
    output_paths.extend(figure_paths)
    manifest = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PROTOCOL_STATUS": PROTOCOL_STATUS,
        "PROTOCOL_SHA256": protocol_hash,
        "RUNNER_SHA256": runner_hash,
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "BASELINE_REPRODUCED": bool(
            baseline.get("BASELINE_REPRODUCED", False)
        ),
        "FRAME_ALIGNMENT_VERIFIED": bool(
            baseline.get("FRAME_ALIGNMENT_VERIFIED", False)
        ),
        "E4_STATUS": str(summary.get("E4_STATUS", "UNKNOWN")),
        "RUNTIME_SECONDS": float(runtime_seconds),
        "PYTHON": sys.version,
        "PLATFORM": platform.platform(),
        "GIT_COMMIT": _git_commit(),
        "GIT_STATUS_SHORT": _git_status_short(),
        "ARTIFACTS": _artifact_records(output_paths),
    }
    _write_json(EXECUTION_MANIFEST_PATH, manifest)


def _verify_required_artifacts() -> dict[str, Any]:
    missing = [
        name
        for name in REQUIRED_OUTPUT_FILES
        if not (OUTPUT_ROOT / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "E4 required artifacts are missing: " + ", ".join(missing)
        )
    figure_paths = [OUTPUT_ROOT / name for name in FIGURE_FILES]
    missing_figures = [path.name for path in figure_paths if not path.exists()]
    if missing_figures:
        raise FileNotFoundError(
            "E4 core figures are missing: " + ", ".join(missing_figures)
        )
    manifest = _load_json(EXECUTION_MANIFEST_PATH)
    for record in manifest.get("ARTIFACTS", []):
        path = REPO_ROOT / str(record["path"])
        if not path.exists():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        if _sha256_file(path) != str(record["sha256"]).upper():
            raise ValueError(f"manifest artifact hash mismatch: {path}")
    summary = _load_json(OUTPUT_ROOT / "e4_final_summary.json")
    if str(summary.get("E4_STATUS")) not in ALLOWED_STATUSES:
        raise ValueError("E4 final summary contains an unknown status")
    return {
        "summary": summary,
        "figures": [path.name for path in figure_paths],
    }


def _run_baseline_mode() -> int:
    _validate_frozen_protocol()
    data = _load_frozen_data()
    baseline = _baseline_reproduction(data)
    _write_json(BASELINE_PATH, baseline)
    print(json.dumps(baseline, indent=2), flush=True)
    return 0 if baseline["BASELINE_REPRODUCED"] else 2


def _run_formal_analysis() -> int:
    started = time.time()
    freeze = _validate_frozen_protocol()
    baseline = _read_baseline()
    data = _load_frozen_data()
    if not bool(baseline.get("BASELINE_REPRODUCED", False)):
        summary = {
            "E4_STATUS": "INCONCLUSIVE_OR_INVALID",
            "BASELINE_REPRODUCED": False,
            "FRAME_ALIGNMENT_VERIFIED": bool(
                baseline.get("FRAME_ALIGNMENT_VERIFIED", False)
            ),
            "TRAINING_PERFORMED": False,
            "NEW_FINAL_OOD_TOUCHED": False,
            "NEXT_EXPERIMENT_AUTHORIZED": False,
            "PROTOCOL_DEVIATIONS": [],
        }
        _write_json(OUTPUT_ROOT / "e4_final_summary.json", summary)
        _write_markdown(
            OUTPUT_ROOT / "e4_final_report.md",
            [
                "# E4 Refinability Stability Decomposition",
                "",
                "Baseline reproduction failed. E4 is "
                "`INCONCLUSIVE_OR_INVALID` and the analysis stopped before "
                "outcome interpretation.",
            ],
        )
        return 2

    r_matrix = _replicate_matrix(data, "R")
    i_matrix = _replicate_matrix(data, "I")
    h_matrix = _replicate_matrix(data, "H")
    k_r = _count_matrix_rows(r_matrix)
    k_i = _count_matrix_rows(i_matrix)
    k_h = _count_matrix_rows(h_matrix)
    masks = _stability_masks(k_r)
    independence_pmf = _independence_k_pmf(
        [float(np.mean(r_matrix[index])) for index in range(len(data.seeds))]
    )
    k_distribution_rows = _k_distribution_rows(
        k_r=k_r,
        k_i=k_i,
        k_h=k_h,
        independence_pmf=independence_pmf,
    )
    observables = _frame_observables(data)
    taxonomy_rows = _taxonomy_summary_rows(
        k_r=k_r,
        masks=masks,
        mean_v_log=observables["mean_v_log"],
        mean_v_brier=observables["mean_v_brier"],
    )
    permutation_rows, permutation_summary = _permutation_overlap_rows(
        r_matrix=r_matrix,
        source_keys=data.source_keys,
        conditions=data.conditions,
    )
    null_rows = _null_overlap_rows(
        k_r=k_r,
        independence_pmf=independence_pmf,
        permutation_rows=permutation_rows,
    )
    proper_rows = _proper_scoring_rows(
        masks=masks,
        observables=observables,
    )
    short_rows = _short_difficulty_rows(
        k_r=k_r,
        k_i=k_i,
        masks=masks,
        observables=observables,
    )
    event_rows = _event_stability_rows(
        k_r=k_r,
        masks=masks,
        observables=observables,
        event_features=data.event_features,
    )
    condition_rows, condition_masks = _condition_stability_rows(
        data=data,
        k_r=k_r,
        masks=masks,
        observables=observables,
    )
    disagreement_rows = _disagreement_decomposition_rows(
        k_r=k_r,
        r_matrix=r_matrix,
        taxonomy_matrices={
            "SS": _replicate_matrix(data, "SS"),
            "I": i_matrix,
            "H": h_matrix,
        },
    )
    short_long_rows = _short_long_instability_rows(
        k_r=k_r,
        observables=observables,
    )
    bootstrap_rows = _bootstrap_stability_rows(
        data=data,
        k_r=k_r,
        masks=masks,
        observables=observables,
        event_features=data.event_features,
        condition_masks=condition_masks,
        null_rows=null_rows,
        r_matrix=r_matrix,
        taxonomy_matrices={"I": i_matrix},
    )
    loso_rows = _source_influence_rows(
        data=data,
        k_r=k_r,
        masks=masks,
        observables=observables,
        event_features=data.event_features,
        condition_masks=condition_masks,
        r_matrix=r_matrix,
        taxonomy_matrices={"I": i_matrix},
        null_rows=null_rows,
    )
    gates = _build_gates_and_claims(
        baseline=baseline,
        seed_rows=baseline["seed_taxonomy"],
        k_r=k_r,
        masks=masks,
        null_rows=null_rows,
        bootstrap_rows=bootstrap_rows,
        loso_rows=loso_rows,
        source_codes=data.source_index.codes,
    )
    summary = _build_final_summary(
        data=data,
        baseline=baseline,
        seed_rows=baseline["seed_taxonomy"],
        k_distribution_rows=k_distribution_rows,
        taxonomy_rows=taxonomy_rows,
        null_rows=null_rows,
        permutation_summary=permutation_summary,
        proper_rows=proper_rows,
        short_rows=short_rows,
        event_rows=event_rows,
        condition_rows=condition_rows,
        disagreement_rows=disagreement_rows,
        short_long_rows=short_long_rows,
        bootstrap_rows=bootstrap_rows,
        loso_rows=loso_rows,
        gates=gates,
    )
    _write_csv(
        OUTPUT_ROOT / "e4_kr_distribution.csv",
        k_distribution_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e4_stability_taxonomy.csv",
        taxonomy_rows,
    )
    _write_csv(OUTPUT_ROOT / "e4_null_overlap.csv", null_rows)
    _write_csv(
        OUTPUT_ROOT / "e4_permutation_overlap.csv",
        permutation_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e4_proper_scoring_stability.csv",
        proper_rows,
    )
    _write_csv(OUTPUT_ROOT / "e4_short_difficulty.csv", short_rows)
    _write_csv(OUTPUT_ROOT / "e4_event_stability.csv", event_rows)
    _write_csv(
        OUTPUT_ROOT / "e4_condition_stability.csv",
        condition_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e4_disagreement_decomposition.csv",
        disagreement_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e4_short_long_instability.csv",
        short_long_rows,
    )
    _write_csv(OUTPUT_ROOT / "e4_bootstrap.csv", bootstrap_rows)
    _write_csv(
        OUTPUT_ROOT / "e4_source_influence.csv",
        loso_rows,
    )
    _write_json(OUTPUT_ROOT / "e4_final_summary.json", summary)
    _write_report(
        summary=summary,
        k_distribution_rows=k_distribution_rows,
        taxonomy_rows=taxonomy_rows,
        null_rows=null_rows,
        bootstrap_rows=bootstrap_rows,
        disagreement_rows=disagreement_rows,
    )
    figure_paths = _write_figures(
        k_distribution_rows=k_distribution_rows,
        taxonomy_rows=taxonomy_rows,
        disagreement_rows=disagreement_rows,
        event_rows=event_rows,
        condition_rows=condition_rows,
    )
    _write_execution_manifest(
        protocol_hash=str(freeze["protocol_sha256"]),
        runner_hash=str(freeze["runner_sha256"]),
        baseline=baseline,
        summary=summary,
        runtime_seconds=time.time() - started,
        figure_paths=figure_paths,
    )
    _verify_required_artifacts()
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _run_verify_mode() -> int:
    _validate_frozen_protocol()
    result = _verify_required_artifacts()
    print(
        json.dumps(
            {
                "mode": "verify",
                "status": result["summary"]["E4_STATUS"],
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
            "Execute the frozen E4 refinability stability decomposition."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("freeze_protocol", "baseline", "run", "verify"),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "freeze_protocol":
        return _freeze_protocol()
    if args.mode == "baseline":
        return _run_baseline_mode()
    if args.mode == "run":
        return _run_formal_analysis()
    if args.mode == "verify":
        return _run_verify_mode()
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
