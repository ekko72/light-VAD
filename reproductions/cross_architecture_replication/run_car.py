# -*- coding: utf-8 -*-
"""Execute the frozen A-v2 Cross-Architecture Replication (CAR) study.

The runner is stage-oriented:

* ``freeze`` writes and hashes the protocol before any model training.
* ``train`` trains exactly five fresh Tiny-GRU models.
* ``audit`` verifies the causal-history implementation.
* ``evaluate`` scores every trained model at all frozen horizons.
* ``analyze`` writes CAR1-CAR5 and the final gate report.

No command trains a router or reads the sealed final-OOD split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    FrameChunkDataset,
    audio_frame_count,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.run_ae_mechanism import (
    CELL_ORDER,
    DISTANCE_BINS,
    DURATION_BINS,
    FEATURE_ORDER,
    SNR_ORDER,
    VALID_START,
    _build_group_masks,
    _feature_bin_mask,
    _segment_event_features,
)
from reproductions.difficulty_adaptive_context.train_context import (
    learning_rate_at,
    resolve_device,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.prior_factored_vad.models import TinyGRUConfig, TinyGRUVAD


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "reproductions" / "cross_architecture_replication"
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "results" / "cross_architecture_replication"
)
PROTOCOL_PATH = DEFAULT_RESULTS_DIR / "car_protocol_freeze.json"
PROTOCOL_HASH_PATH = DEFAULT_RESULTS_DIR / "car_protocol_sha256.txt"

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_MANIFEST_ROOT = DEFAULT_DATA_ROOT / "manifests"
DEFAULT_TRAIN_MANIFEST = (
    DEFAULT_MANIFEST_ROOT / "LibriSpeech_train_medium.tsv"
)
DEFAULT_VAL_MANIFEST = DEFAULT_MANIFEST_ROOT / "LibriSpeech_val_medium.tsv"
DEFAULT_TEST_MANIFEST = (
    DEFAULT_MANIFEST_ROOT / "LibriSpeech_test_medium.tsv"
)
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
RF384_PREDICTION_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
AE3_FEATURE_BINS_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae_mechanism"
    / "ae3_feature_bins.csv"
)
AE1_CONDITION_AVAILABILITY_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae_mechanism"
    / "ae1_condition_availability.csv"
)
AE2_HORIZON_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae_mechanism"
    / "ae2_horizon_summary.csv"
)
AE_MECHANISM_RUNNER_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "run_ae_mechanism.py"
)
B15_RUNNER_PATH = (
    REPO_ROOT / "reproductions" / "prior_factored_vad" / "run_b15.py"
)
MODELS_PATH = (
    REPO_ROOT / "reproductions" / "prior_factored_vad" / "models.py"
)
DATA_UTILS_PATH = (
    REPO_ROOT / "reproductions" / "difficulty_adaptive_context" / "data.py"
)
FEATURES_PATH = (
    REPO_ROOT / "reproductions" / "marblenet_vad" / "features.py"
)

PROTOCOL_ID = "A-v2-CAR-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_CAR_TRAINING"
TEST_MANIFEST_SHA256 = (
    "267CD1A018216E62552BC00B088F2667AD2113B1D1EC0E954A4F01854C1C8A39"
)
RF384_SHA256 = (
    "F25749E568D8DF6370FD2F52262EDF6169AFEB6DA30F1BAA6F46E80CD68EEC03"
)
RF384_SIZE_BYTES = 10_702_416

MODEL_SEEDS = (41, 59, 71, 83, 97)
ROW_SELECTION_SEED = 17
VALIDATION_ROW_SEED = 18
HORIZONS = (64, 128, 256, 384, None)
WINDOW_HORIZONS = (64, 128, 256, 384)
HORIZON_LABELS = {
    64: "H64",
    128: "H128",
    256: "H256",
    384: "H384",
    None: "HFULL",
}
PRIMARY_SHORT = 64
PRIMARY_LONG = None
SECONDARY_LONG = 384
DECISION_THRESHOLD = 0.5
BOOTSTRAP_REPEATS = 2_000
SPEAKER_SEED_OFFSET = 700_000
EXPECTED_TOTAL_FRAMES = 553_532
EXPECTED_TEST_FRAMES = 298_300
EXPECTED_CALIBRATION_FRAMES = 255_232
EXPECTED_TEST_SOURCES = 96
EXPECTED_TEST_SPEAKERS = 20
EXPECTED_EVALUATION_ROWS = 1_080
EXPECTED_TRAIN_CHUNKS = 8_286
EXPECTED_VALIDATION_CHUNKS = 318
UNSEEN_NOISE = ("SSN_noise", "Street_noise", "Transport_noise")
CONDITION_CELLS = tuple(
    f"{domain}/{snr}" for domain in ("seen", "unseen") for snr in SNR_ORDER
)
R3_MIN_PASSING_STRUCTURES = 2
R4_RANGE_THRESHOLD = 0.002
R5_MAX_SEED_POSITIVE_SHARE = 0.50
LONG_HISTORY_SHARE_THRESHOLD = 0.50
SPARSE_PREVALENCE_RATIO_THRESHOLD = 0.10
VALUE_SPARSITY_EPSILON = 1e-8
MEANINGFUL_VALUE_RATE_THRESHOLD = 0.10
NON_MONOTONE_MIN_RATE = 0.05
BOOTSTRAP_SEED_BASE = 20_260_920
CAR_FIGURE_FILES = (
    "car1_taxonomy.png",
    "car2_horizon_demand.png",
    "car2_nonmonotonicity.png",
    "car3_event_profiles.png",
    "car4_condition_heatmap.png",
    "car5_seed_agreement.png",
    "car_gate_status.png",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify_hash(path: Path, expected: str, *, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != str(expected).upper():
        raise ValueError(f"{role} hash mismatch: {actual} != {expected}")
    return actual


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    )
    return bool(output.strip())


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _horizon_key(horizon: int | None) -> str:
    return HORIZON_LABELS[horizon]


def _resolve_protocol_path(path: str | Path | None = None) -> Path:
    return PROTOCOL_PATH if path is None else Path(path)


def load_car_protocol(
    path: str | Path | None = None,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    protocol_path = _resolve_protocol_path(path)
    payload = _read_json(protocol_path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported CAR protocol id")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("CAR protocol is not frozen before training")
    digest = file_sha256(protocol_path)
    hash_path = (
        PROTOCOL_HASH_PATH
        if protocol_path.resolve() == PROTOCOL_PATH.resolve()
        else protocol_path.with_name("car_protocol_sha256.txt")
    )
    if not hash_path.is_file():
        raise FileNotFoundError(hash_path)
    recorded = hash_path.read_text(encoding="utf-8").strip().split()[0]
    if recorded.upper() != digest:
        raise ValueError("CAR protocol hash file does not match protocol")
    if verify_hashes:
        verify_locked_hashes(payload)
    return payload


def verify_locked_hashes(protocol: Mapping[str, Any]) -> None:
    for role, record in protocol["source_hashes"].items():
        _verify_hash(
            REPO_ROOT / str(record["path"]),
            str(record["sha256"]),
            role=role,
        )
    for role, record in protocol["data_hashes"].items():
        _verify_hash(
            REPO_ROOT / str(record["path"]),
            str(record["sha256"]),
            role=role,
        )


def _marble_bin_value(
    rows: Sequence[Mapping[str, str]],
    *,
    feature: str,
    bins: Sequence[str],
    group: str = "all",
) -> float:
    selected: list[tuple[float, float]] = []
    for row in rows:
        if (
            str(row.get("feature", "")) != feature
            or str(row.get("group", "")) != group
            or str(row.get("bin", "")) not in set(bins)
        ):
            continue
        frames = _float_or_none(row.get("frames"))
        availability = _float_or_none(row.get("availability_net"))
        if (
            frames is None
            or frames <= 0.0
            or availability is None
        ):
            continue
        selected.append((frames, availability))
    total = sum(frames for frames, _ in selected)
    if total <= 0.0:
        raise ValueError(f"no MarbleNet frames for {feature}/{bins}")
    numerator = sum(
        frames * availability for frames, availability in selected
    )
    return numerator / total


def _r3_contrast_records() -> dict[str, Any]:
    rows = _read_csv(AE3_FEATURE_BINS_PATH)
    definitions = {
        "transition": {
            "feature": "posterior_transition_distance",
            "high_bins": ("0",),
            "reference_bins": (
                "1",
                "2-4",
                "5-10",
                "11-25",
                "26-50",
                ">50",
            ),
            "selected_only": False,
        },
        "persistence": {
            "feature": "uncertainty_persistence",
            "high_bins": ("11-25",),
            "reference_bins": ("1",),
            "selected_only": True,
        },
        "onset": {
            "feature": "onset_distance",
            "high_bins": ("0",),
            "reference_bins": ("1",),
            "selected_only": False,
        },
    }
    records: dict[str, Any] = {}
    for name, definition in definitions.items():
        high = _marble_bin_value(
            rows,
            feature=str(definition["feature"]),
            bins=definition["high_bins"],
        )
        reference = _marble_bin_value(
            rows,
            feature=str(definition["feature"]),
            bins=definition["reference_bins"],
        )
        records[name] = {
            **definition,
            "marble_net_high_availability": high,
            "marble_net_reference_availability": reference,
            "marble_net_contrast": high - reference,
            "predicted_direction": "positive",
            "support_rule": (
                "point_contrast_positive_and_source_cluster_"
                "ci95_lower_strictly_positive"
            ),
        }
    return records


def build_protocol_payload() -> dict[str, Any]:
    source_paths = {
        "car_runner": PACKAGE_ROOT / "run_car.py",
        "car_tests": PACKAGE_ROOT / "test_run_car.py",
        "b15_runner": B15_RUNNER_PATH,
        "tiny_gru_models": MODELS_PATH,
        "frame_data": DATA_UTILS_PATH,
        "mfcc_features": FEATURES_PATH,
        "ae_mechanism_runner": AE_MECHANISM_RUNNER_PATH,
        "ae3_feature_bins": AE3_FEATURE_BINS_PATH,
        "ae1_condition_availability": AE1_CONDITION_AVAILABILITY_PATH,
        "ae2_horizon_summary": AE2_HORIZON_SUMMARY_PATH,
    }
    data_paths = {
        "train_manifest": DEFAULT_TRAIN_MANIFEST,
        "validation_manifest": DEFAULT_VAL_MANIFEST,
        "test_manifest": DEFAULT_TEST_MANIFEST,
        "rf384_prediction_bundle": RF384_PREDICTION_PATH,
    }
    source_hashes = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    data_hashes = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in data_paths.items()
    }
    if data_hashes["test_manifest"]["sha256"] != TEST_MANIFEST_SHA256:
        raise ValueError("test manifest does not match the frozen A9 manifest")
    if data_hashes["rf384_prediction_bundle"]["sha256"] != RF384_SHA256:
        raise ValueError("RF384 prediction bundle does not match the freeze")
    if (
        data_hashes["rf384_prediction_bundle"]["size_bytes"]
        != RF384_SIZE_BYTES
    ):
        raise ValueError("RF384 prediction bundle size differs from the freeze")
    return {
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "source_hashes": source_hashes,
        "data_hashes": data_hashes,
        "scope": {
            "scientific_question": (
                "Does the sparse, hard-not-refinable, temporally "
                "structured marginal value observed with MarbleNet "
                "reproduce in a recurrent Tiny-GRU backbone?"
            ),
            "forbidden": [
                "router_training",
                "method_search",
                "architecture_search",
                "hyperparameter_search",
                "seed_selection",
                "H512",
                "frame_bootstrap_as_primary",
                "NEW_FINAL_OOD_access",
            ],
        },
        "architecture": {
            "class": "TinyGRUVAD",
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": 2,
            "dropout": 0.0,
            "feature_frontend": (
                "MfccFrontend(MfccConfig(causal=True))"
            ),
        },
        "training": {
            "recipe": "B15",
            "dataset": "LibriSpeech_train_medium.tsv",
            "validation_dataset": "LibriSpeech_val_medium.tsv",
            "row_selection_seed": ROW_SELECTION_SEED,
            "validation_row_seed": VALIDATION_ROW_SEED,
            "model_seeds": list(MODEL_SEEDS),
            "model_seed_controls": [
                "initialization",
                "training_loader_shuffle",
                "training_chunk_resampling",
            ],
            "excluded_training_noise": list(UNSEEN_NOISE),
            "train_rows": 8_640,
            "validation_rows": 432,
            "train_chunks": EXPECTED_TRAIN_CHUNKS,
            "validation_chunks": EXPECTED_VALIDATION_CHUNKS,
            "context_seconds": 4.0,
            "target_seconds": 4.0,
            "context_frames": 400,
            "target_frames": 400,
            "batch_size": 128,
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "epochs": 40,
            "warmup_ratio": 0.0,
            "hold_ratio": 0.0,
            "decay_power": 2.0,
            "gradient_clip_norm": 5.0,
            "early_stopping": "none",
            "checkpoint_selection": "best_validation_auc",
            "num_workers": 0,
            "loss": "two_class_softmax_cross_entropy",
        },
        "evaluation": {
            "test_manifest": str(
                DEFAULT_TEST_MANIFEST.relative_to(REPO_ROOT)
            ),
            "test_manifest_sha256": TEST_MANIFEST_SHA256,
            "a9_row_sample": EXPECTED_EVALUATION_ROWS,
            "a9_evaluation_seed": 17,
            "valid_start_frame": VALID_START,
            "prediction_bundle": str(
                RF384_PREDICTION_PATH.relative_to(REPO_ROOT)
            ),
            "prediction_bundle_sha256": RF384_SHA256,
            "expected_total_frames": EXPECTED_TOTAL_FRAMES,
            "expected_test_frames": EXPECTED_TEST_FRAMES,
            "expected_calibration_frames": EXPECTED_CALIBRATION_FRAMES,
            "expected_test_source_clusters": EXPECTED_TEST_SOURCES,
            "expected_test_speaker_clusters": EXPECTED_TEST_SPEAKERS,
            "decision_threshold": DECISION_THRESHOLD,
            "horizons": ["H64", "H128", "H256", "H384", "HFULL"],
            "primary_short": "H64",
            "primary_long": "HFULL",
            "secondary_long": "H384",
            "forbidden_horizons": ["H512"],
            "reset_semantics": (
                "For target frame t and finite horizon H, initialize the "
                "GRU hidden state to zero at max(0,t-H+1), process frames "
                "through t, and score only t. H includes the target frame. "
                "HFULL initializes at the utterance start and streams "
                "causally. All horizons share one trained model."
            ),
            "future_frames_forbidden": True,
            "same_weights_all_horizons": True,
        },
        "taxonomy": {
            "threshold": DECISION_THRESHOLD,
            "definitions": {
                "SS": "H64 correct and HFULL correct",
                "R": "H64 wrong and HFULL correct",
                "I": "H64 wrong and HFULL wrong",
                "H": "H64 correct and HFULL wrong",
            },
            "value": "v(t)=loss_H64(t)-loss_HFULL(t)",
            "net_refinability": "P(R)-P(H)",
        },
        "bootstrap": {
            "primary_unit": "source_cluster",
            "source_clusters": "source_key",
            "speaker_unit": "speaker_sensitivity_only",
            "speaker_clusters": "speaker_ids",
            "repeats": BOOTSTRAP_REPEATS,
            "interval": "percentile_95",
            "frame_bootstrap_primary": False,
            "aggregate_definition": (
                "Equal-weight mean of the five seed-specific frame "
                "indicators followed by source-cluster resampling."
            ),
        },
        "car2_definitions": {
            "ordered_horizons": ["H64", "H128", "H256", "H384", "HFULL"],
            "signed_incremental_value": (
                "loss(H64)-loss(horizon), reported for each horizon"
            ),
            "adjacent_delta_value": (
                "value(horizon)-value(previous_horizon), with H64 as zero"
            ),
            "first_correct_horizon": (
                "Earliest horizon whose thresholded prediction is correct; "
                "-1 when no evaluated horizon is correct."
            ),
            "stable_sufficient_horizon": (
                "Earliest horizon that is correct and remains correct at "
                "every later horizon; -1 when no such suffix exists."
            ),
            "non_monotone_prediction": (
                "Correct at a horizon and incorrect at any immediately "
                "later horizon."
            ),
            "non_monotone_rate_denominator": (
                "Frames incorrect at H64, matching AE2's not-short subset."
            ),
            "long_history_share": (
                "(loss_H64-loss_H384)/(loss_H64-loss_HFULL), when the "
                "denominator is positive; long history is not dominant "
                "when this share is at least 0.50."
            ),
            "cross_architecture_unit_mismatch": (
                "The Tiny-GRU long-history share uses "
                "(loss_H64-loss_H384)/(loss_H64-loss_HFULL). "
                "The frozen MarbleNet AE2 comparison uses "
                "oracle_net_utility_rf384/oracle_net_utility_rf512. "
                "These are direction-only diagnostics and their raw "
                "values are not numerically interchangeable."
            ),
        },
        "car3_definitions": {
            "event_bins": "AE3 frozen feature bins imported without changes",
            "persistence_denominator": "selected frames only",
            "contrast_direction": "positive",
            "centered_profiles": {
                "transition": (
                    "posterior_transition_distance bins 0 through >50"
                ),
                "onset": (
                    "signed onset distance split into before/at/after, "
                    "then the frozen absolute-distance bins"
                ),
                "offset": (
                    "signed offset distance split into before/at/after, "
                    "then the frozen absolute-distance bins"
                ),
            },
            "cross_architecture_metrics": [
                "bin_sign_agreement",
                "bin_spearman_rank_correlation",
                "qualitative_peak_location_agreement",
            ],
        },
        "car4_definitions": {
            "primary_cells": list(CONDITION_CELLS),
            "noise_scope": (
                "Additional descriptive rows by noise_name; the R4 gate "
                "uses only seen/unseen x SNR cells."
            ),
            "marble_cell_mapping": (
                "MarbleNet cell availability and taxonomy are computed "
                "from the frozen AE3 posterior_transition_distance rows "
                "with frame-count weighting. MarbleNet horizon diagnostics "
                "use the frozen AE2 rows for the same cell. Noise-name "
                "rows remain descriptive because the frozen AE1-AE3 "
                "tables do not contain those names."
            ),
            "range": "max(cell mean v)-min(cell mean v)",
            "cross_architecture_metrics": [
                "cell_sign_agreement",
                "cell_spearman_rank_correlation",
                "seen_unseen_direction_agreement",
                "snr_trend_agreement",
            ],
        },
        "car5_definitions": {
            "sign_agreement": (
                "Exact agreement of sign(v), including zero, over all "
                "test frames."
            ),
            "refinable_jaccard": (
                "Intersection over union of per-seed R frame sets."
            ),
            "taxonomy_agreement": (
                "Exact SS/R/I/H agreement over all test frames."
            ),
            "leave_one_seed_out": (
                "Mean v over the remaining four seeds, frame by frame."
            ),
            "positive_share": (
                "Seed count of v>0 divided by the sum of v>0 counts over "
                "all five seeds."
            ),
        },
        "cross_architecture_classification": {
            "values": [
                "REPLICATED",
                "PARTIALLY_REPLICATED",
                "NOT_REPLICATED",
                "INCOMPARABLE",
            ],
            "sparse_prevalence_ratio_threshold": (
                SPARSE_PREVALENCE_RATIO_THRESHOLD
            ),
            "long_history_share_threshold": LONG_HISTORY_SHARE_THRESHOLD,
            "value_sparsity_epsilon": VALUE_SPARSITY_EPSILON,
            "meaningful_value_rate_threshold": (
                MEANINGFUL_VALUE_RATE_THRESHOLD
            ),
            "non_monotone_min_rate": NON_MONOTONE_MIN_RATE,
            "rule": (
                "Qualitative structure is classified before raw effect "
                "sizes; incomparable units remain INCOMPARABLE."
            ),
        },
        "r3_contrasts": _r3_contrast_records(),
        "r3_pass_rule": {
            "minimum_structures_passing": R3_MIN_PASSING_STRUCTURES,
            "structure_passes": (
                "point contrast positive, MarbleNet contrast positive, "
                "and source-cluster bootstrap 95% lower bound > 0"
            ),
        },
        "condition_heterogeneity": {
            "cells": list(CONDITION_CELLS),
            "metric": "mean_v",
            "statistic": "max(cell_mean_v)-min(cell_mean_v)",
            "minimum_point_range": R4_RANGE_THRESHOLD,
            "support_rule": (
                "source-cluster bootstrap 95% lower bound of the range > 0"
            ),
        },
        "seed_dominance": {
            "rule": (
                "full mean v and every leave-one-seed-out mean v have the "
                "same positive sign, and no seed contributes more than "
                "50% of positive-v frame counts"
            ),
            "maximum_positive_share": R5_MAX_SEED_POSITIVE_SHARE,
        },
        "gates": {
            "R1": (
                "5/5 Tiny-GRU seeds aggregate NetRefinability > 0"
            ),
            "R2": (
                "P(R) source-cluster 95% lower bound > 0 and P(I)>P(R)"
            ),
            "R3": (
                "at least two of transition/persistence/onset contrasts "
                "pass the frozen direction and cluster-support rule"
            ),
            "R4": (
                "condition-cell mean-v range meets the frozen point "
                "threshold and cluster-support rule"
            ),
            "R5": "seed-dominance rule passes",
            "strong": "R1-R5 all pass",
            "conditional": "exactly four pass",
            "no_go": "three or fewer pass",
        },
        "final_status_values": [
            "STRONG_CROSS_ARCH_REPLICATION",
            "CROSS_ARCH_CONDITIONAL",
            "CROSS_ARCH_NO_GO",
        ],
        "outputs": {
            "tables": [
                "car_gru_training_runs.csv",
                "car1_taxonomy_by_seed.csv",
                "car1_taxonomy_by_condition.csv",
                "car1_bootstrap.csv",
                "car2_history_demand.csv",
                "car2_nonmonotonicity.csv",
                "car3_event_structure.csv",
                "car3_event_comparison.csv",
                "car3_gate_support.csv",
                "car4_condition_dependence.csv",
                "car4_cross_architecture.csv",
                "car5_seed_agreement.csv",
                "car_cross_architecture_summary.csv",
            ],
            "json": [
                "car_causality_audit.json",
                "car_final_summary.json",
                "car_execution_manifest.json",
            ],
            "reports": [
                "car_final_report.md",
                "car_claim_freeze.md",
            ],
            "directories": [
                "car3_event_profiles",
                "car_figures",
                "checkpoints",
                "evaluations",
            ],
            "figures": list(CAR_FIGURE_FILES),
        },
        "safety": {
            "NEXT_METHOD_SEARCH_AUTHORIZED": False,
            "NEW_FINAL_OOD_TOUCHED": False,
        },
    }


def write_protocol_freeze(*, force: bool = False) -> Path:
    if PROTOCOL_PATH.exists() and not force:
        raise FileExistsError(
            f"protocol freeze already exists: {PROTOCOL_PATH}"
        )
    if PROTOCOL_HASH_PATH.exists() and not force:
        raise FileExistsError(
            f"protocol hash already exists: {PROTOCOL_HASH_PATH}"
        )
    payload = build_protocol_payload()
    _write_json(PROTOCOL_PATH, payload)
    digest = file_sha256(PROTOCOL_PATH)
    PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="ascii")
    load_car_protocol(verify_hashes=True)
    return PROTOCOL_PATH


@dataclass
class CarReference:
    protocol: dict[str, Any]
    labels: np.ndarray
    short_scores: np.ndarray
    full_adaptive_scores: np.ndarray
    long_scores: np.ndarray
    selected: np.ndarray
    test_mask: np.ndarray
    calibration_mask: np.ndarray
    source_key: np.ndarray
    speaker_ids: np.ndarray
    condition: np.ndarray
    noise_name: np.ndarray
    source_order: tuple[str, ...]
    speaker_order: tuple[str, ...]
    group_masks: dict[str, np.ndarray]
    cell_labels: np.ndarray
    segments: list[tuple[int, int]]
    segment_audio_paths: list[Path]
    event_features: dict[str, np.ndarray]


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def _build_event_features(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    selected: np.ndarray,
    segments: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    features = {
        name: np.zeros(labels.size, dtype=np.float64)
        for name in FEATURE_ORDER
    }
    for start, end in segments:
        local = _segment_event_features(
            labels[start:end],
            short_scores[start:end],
            selected[start:end],
        )
        for name, values in local.items():
            features[name][start:end] = values
    return features


def load_rf384_reference(
    protocol: Mapping[str, Any],
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    librispeech_root: str | Path = DEFAULT_LIBRISPEECH_ROOT,
) -> CarReference:
    record = protocol["data_hashes"]["rf384_prediction_bundle"]
    path = REPO_ROOT / str(record["path"])
    _verify_hash(path, str(record["sha256"]), role="RF384 prediction bundle")
    arrays = _load_npz_arrays(path)
    required = {
        "labels",
        "short_scores",
        "full_adaptive_scores",
        "long_scores",
        "embedded_short_scores",
        "selected",
        "test_mask",
        "calibration_mask",
        "source_key",
        "speaker_ids",
        "condition",
        "noise_name",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"RF384 bundle is missing arrays: {missing}")
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    short_scores = np.asarray(arrays["short_scores"], dtype=np.float64)
    full_adaptive_scores = np.asarray(
        arrays["full_adaptive_scores"], dtype=np.float64
    )
    long_scores = np.asarray(arrays["long_scores"], dtype=np.float64)
    embedded = np.asarray(
        arrays["embedded_short_scores"], dtype=np.float64
    )
    if not np.array_equal(short_scores, embedded):
        raise ValueError("RF384 short_scores differ from embedded short scores")
    selected = np.asarray(arrays["selected"], dtype=bool)
    test_mask = np.asarray(arrays["test_mask"], dtype=bool)
    calibration_mask = np.asarray(
        arrays["calibration_mask"], dtype=bool
    )
    source_key = np.asarray(arrays["source_key"], dtype=str)
    speaker_ids = np.asarray(arrays["speaker_ids"], dtype=str)
    condition = np.asarray(arrays["condition"], dtype=str)
    noise_name = np.asarray(arrays["noise_name"], dtype=str)
    lengths = {
        value.size
        for value in (
            labels,
            short_scores,
            full_adaptive_scores,
            long_scores,
            selected,
            test_mask,
            calibration_mask,
            source_key,
            speaker_ids,
            condition,
            noise_name,
        )
    }
    if lengths != {EXPECTED_TOTAL_FRAMES}:
        raise ValueError(f"RF384 array lengths differ: {sorted(lengths)}")
    if int(np.count_nonzero(test_mask)) != EXPECTED_TEST_FRAMES:
        raise ValueError("RF384 test-frame count differs from the freeze")
    if (
        int(np.count_nonzero(calibration_mask))
        != EXPECTED_CALIBRATION_FRAMES
    ):
        raise ValueError("RF384 calibration-frame count differs from freeze")
    if int(np.count_nonzero(test_mask & calibration_mask)):
        raise ValueError("test and calibration masks overlap")

    items = build_evaluation_items(
        DEFAULT_TEST_MANIFEST,
        generated_root=Path(data_root) / "generated",
        label_root=Path(data_root) / "labels",
        librispeech_root=librispeech_root,
        row_sample=EXPECTED_EVALUATION_ROWS,
        seed=17,
        include_clean=True,
    )
    segments: list[tuple[int, int]] = []
    segment_audio_paths: list[Path] = []
    cursor = 0
    for item in items:
        n_frames = audio_frame_count(item.audio_path) // FRAME_HOP + 1
        if n_frames <= VALID_START:
            continue
        length = n_frames - VALID_START
        segments.append((cursor, cursor + length))
        segment_audio_paths.append(item.audio_path)
        cursor += length
    if cursor != labels.size:
        raise ValueError(
            f"reconstructed utterance frames {cursor} != frozen {labels.size}"
        )

    group_masks = _build_group_masks(
        test_mask=test_mask,
        condition=condition,
        noise_name=noise_name,
    )
    cell_labels = np.asarray(
        [
            (
                "clean"
                if str(c) == "clean"
                else (
                    f"unseen/{str(c)}"
                    if str(n) in set(UNSEEN_NOISE)
                    else f"seen/{str(c)}"
                )
            )
            for c, n in zip(condition, noise_name)
        ],
        dtype=object,
    )
    source_order = tuple(
        sorted({str(value) for value in source_key[test_mask]})
    )
    speaker_order = tuple(
        sorted({str(value) for value in speaker_ids[test_mask]})
    )
    if len(source_order) != EXPECTED_TEST_SOURCES:
        raise ValueError("test source-cluster count differs from the freeze")
    if len(speaker_order) != EXPECTED_TEST_SPEAKERS:
        raise ValueError("test speaker-cluster count differs from the freeze")
    event_features = _build_event_features(
        labels=labels,
        short_scores=short_scores,
        selected=selected,
        segments=segments,
    )
    return CarReference(
        protocol=dict(protocol),
        labels=labels,
        short_scores=short_scores,
        selected=selected,
        test_mask=test_mask,
        calibration_mask=calibration_mask,
        source_key=source_key,
        speaker_ids=speaker_ids,
        condition=condition,
        noise_name=noise_name,
        source_order=source_order,
        speaker_order=speaker_order,
        group_masks=group_masks,
        cell_labels=cell_labels,
        segments=segments,
        segment_audio_paths=segment_audio_paths,
        event_features=event_features,
    )


class FixedRowsFrameChunkDataset(FrameChunkDataset):
    """Frame chunks with fixed row selection and seed-specific resampling."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        generated_root: str | Path,
        label_root: str | Path,
        context_samples: int,
        target_samples: int,
        row_seed: int,
        chunk_seed: int,
        row_sample: int | None,
        exclude_noise_names: set[str] | None = None,
        sample_offset: int = 0,
        resample_chunks: bool = False,
    ) -> None:
        super().__init__(
            manifest,
            generated_root=generated_root,
            label_root=label_root,
            context_samples=context_samples,
            target_samples=target_samples,
            row_sample=row_sample,
            seed=int(row_seed),
            exclude_noise_names=exclude_noise_names,
            sample_offset=sample_offset,
            resample_chunks=resample_chunks,
        )
        self.row_seed = int(row_seed)
        self.chunk_seed = int(chunk_seed)
        self.seed = int(chunk_seed)
        self._build_chunks(0)


def _build_loader(
    dataset: FrameChunkDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def _frame_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    context_samples: int,
    criterion: nn.Module,
) -> torch.Tensor:
    target_start = int(context_samples) // FRAME_HOP
    target_logits = logits[:, :, target_start:]
    if target_logits.shape[-1] != labels.shape[-1]:
        raise ValueError("target-frame mismatch in training loss")
    return criterion(
        target_logits.transpose(1, 2).reshape(-1, target_logits.shape[1]),
        labels.reshape(-1),
    )


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predictions = scores >= DECISION_THRESHOLD
    return {
        "frames": int(labels.size),
        "speech": int(np.count_nonzero(labels == 1)),
        "silence": int(np.count_nonzero(labels == 0)),
        "error": float(np.mean(predictions != labels)),
        "f1": (
            float(f1_score(labels, predictions, zero_division=0))
            if labels.size and np.unique(labels).size > 1
            else None
        ),
        "auc": (
            float(roc_auc_score(labels, scores))
            if labels.size and np.unique(labels).size > 1
            else None
        ),
    }


def _validate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    frontend: nn.Module,
    device: torch.device,
    *,
    context_samples: int,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    frame_count = 0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for waveforms, labels in loader:
            features = frontend(waveforms.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = _frame_loss(
                logits,
                labels,
                context_samples=context_samples,
                criterion=criterion,
            )
            probabilities = torch.softmax(
                logits[
                    :, :, context_samples // FRAME_HOP :
                ].transpose(1, 2),
                dim=-1,
            )[..., 1]
            loss_sum += float(loss.detach().cpu()) * int(labels.numel())
            frame_count += int(labels.numel())
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))
            all_scores.append(
                probabilities.detach().cpu().numpy().reshape(-1)
            )
    result = _binary_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_scores),
    )
    result["loss"] = loss_sum / max(frame_count, 1)
    return result


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "seed": int(seed),
        "history": list(history),
        "model_config": {
            "architecture": "causal_tiny_gru",
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": 2,
            "dropout": 0.0,
            "context_samples": 64_000,
            "target_samples": 64_000,
        },
    }


def _load_model_from_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> TinyGRUVAD:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyGRUVAD(TinyGRUConfig())
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def train_one_seed(
    seed: int,
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    seed_dir = results_dir / "checkpoints" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    history_path = seed_dir / "history.json"
    best_path = seed_dir / "best.pt"
    last_path = seed_dir / "last.pt"
    if history_path.is_file() and best_path.is_file():
        payload = _read_json(history_path)
        return {
            "seed": int(seed),
            "status": "already_complete",
            "best_checkpoint": str(best_path.relative_to(REPO_ROOT)),
            "best_epoch": int(payload.get("best_epoch", -1)),
            "best_val_auc": payload.get("best_val_auc"),
            "history_path": str(history_path.relative_to(REPO_ROOT)),
        }
    if last_path.is_file() and not resume:
        raise FileExistsError(f"partial training exists: {last_path}")

    _set_seed(seed)
    train_cfg = protocol["training"]
    context_samples = int(round(float(train_cfg["context_seconds"]) * 16_000))
    target_samples = int(round(float(train_cfg["target_seconds"]) * 16_000))
    train_dataset = FixedRowsFrameChunkDataset(
        DEFAULT_TRAIN_MANIFEST,
        generated_root=DEFAULT_DATA_ROOT / "generated",
        label_root=DEFAULT_DATA_ROOT / "labels",
        context_samples=context_samples,
        target_samples=target_samples,
        row_seed=ROW_SELECTION_SEED,
        chunk_seed=seed,
        row_sample=int(train_cfg["train_rows"]),
        exclude_noise_names=set(UNSEEN_NOISE),
        resample_chunks=True,
    )
    val_dataset = FixedRowsFrameChunkDataset(
        DEFAULT_VAL_MANIFEST,
        generated_root=DEFAULT_DATA_ROOT / "generated",
        label_root=DEFAULT_DATA_ROOT / "labels",
        context_samples=context_samples,
        target_samples=target_samples,
        row_seed=VALIDATION_ROW_SEED,
        chunk_seed=VALIDATION_ROW_SEED,
        row_sample=int(train_cfg["validation_rows"]),
        exclude_noise_names=set(UNSEEN_NOISE),
        sample_offset=10_000,
        resample_chunks=False,
    )
    if len(train_dataset.rows) != int(train_cfg["train_rows"]):
        raise RuntimeError("training row count differs from the recipe")
    if len(val_dataset.rows) != int(train_cfg["validation_rows"]):
        raise RuntimeError("validation row count differs from the recipe")
    if len(train_dataset) != int(train_cfg["train_chunks"]):
        raise RuntimeError("training chunk count differs from the recipe")
    if len(val_dataset) != int(train_cfg["validation_chunks"]):
        raise RuntimeError("validation chunk count differs from the recipe")
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        seed=seed,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        seed=VALIDATION_ROW_SEED,
    )
    model = TinyGRUVAD(TinyGRUConfig()).to(device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if last_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
    else:
        start_epoch = 0
        global_step = 0
        history = []
    total_steps = int(train_cfg["epochs"]) * len(train_loader)
    best_val_auc = max(
        (
            float(record["val"]["auc"])
            for record in history
            if record.get("val", {}).get("auc") is not None
        ),
        default=-math.inf,
    )
    best_epoch = max(
        (
            int(record["epoch"])
            for record in history
            if record.get("val", {}).get("auc") is not None
            and float(record["val"]["auc"]) == best_val_auc
        ),
        default=-1,
    )
    print(
        f"seed={seed} device={device} train_chunks={len(train_dataset)} "
        f"val_chunks={len(val_dataset)} steps_per_epoch={len(train_loader)} "
        f"start_epoch={start_epoch}",
        flush=True,
    )
    started = time.time()
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        train_dataset.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        frame_count = 0
        for waveforms, labels in train_loader:
            lr = learning_rate_at(
                global_step,
                total_steps,
                max_lr=float(train_cfg["learning_rate"]),
                min_lr=float(train_cfg["learning_rate"]),
                warmup_ratio=float(train_cfg["warmup_ratio"]),
                hold_ratio=float(train_cfg["hold_ratio"]),
                power=float(train_cfg["decay_power"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            features = frontend(waveforms.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = _frame_loss(
                logits,
                labels,
                context_samples=context_samples,
                criterion=criterion,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(labels.numel())
            frame_count += int(labels.numel())
            global_step += 1
        train_metrics = {
            "loss": loss_sum / max(frame_count, 1),
            "frames": frame_count,
        }
        val_metrics = _validate_model(
            model,
            val_loader,
            criterion,
            frontend,
            device,
            context_samples=context_samples,
        )
        record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        improved = (
            val_metrics["auc"] is not None
            and float(val_metrics["auc"]) > best_val_auc
        )
        if improved:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = int(epoch)
        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                seed=seed,
                history=history,
            ),
            last_path,
        )
        if improved:
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    seed=seed,
                    history=history,
                ),
                best_path,
            )
        print(
            f"seed={seed} epoch {epoch + 1}/{train_cfg['epochs']} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_f1={val_metrics['f1']:.5f} "
            f"val_auc={val_metrics['auc']:.5f}",
            flush=True,
        )
    _write_json(
        history_path,
        {
            "seed": int(seed),
            "row_selection_seed": ROW_SELECTION_SEED,
            "validation_row_seed": VALIDATION_ROW_SEED,
            "chunk_seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "global_step": int(global_step),
            "elapsed_seconds": float(time.time() - started),
            "history": history,
        },
    )
    return {
        "seed": int(seed),
        "status": "trained",
        "best_checkpoint": str(best_path.relative_to(REPO_ROOT)),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "elapsed_seconds": float(time.time() - started),
        "history_path": str(history_path.relative_to(REPO_ROOT)),
    }


def train_all(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
    resume: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = results_dir / "car_gru_training_runs.csv"
    for seed in MODEL_SEEDS:
        record = train_one_seed(
            seed,
            protocol=protocol,
            results_dir=results_dir,
            device=device,
            resume=resume,
        )
        rows.append(record)
        _write_csv(csv_path, rows)
    return rows


def _score_window_batch(
    model: TinyGRUVAD,
    features: torch.Tensor,
    targets: Sequence[int],
    *,
    horizon: int,
) -> np.ndarray:
    if horizon not in WINDOW_HORIZONS:
        raise ValueError(f"invalid finite horizon: {horizon}")
    if not targets:
        return np.empty(0, dtype=np.float64)
    n_frames = int(features.shape[2])
    lengths = np.empty(len(targets), dtype=np.int64)
    padded = torch.zeros(
        (len(targets), int(horizon), features.shape[1]),
        dtype=features.dtype,
        device=features.device,
    )
    for index, target in enumerate(targets):
        target = int(target)
        if target < 0 or target >= n_frames:
            raise ValueError("target frame outside the sequence")
        start = max(0, target - int(horizon) + 1)
        sequence = features[0, :, start : target + 1].transpose(0, 1)
        length = int(sequence.shape[0])
        lengths[index] = length
        padded[index, :length, :] = sequence
    packed = nn.utils.rnn.pack_padded_sequence(
        padded,
        torch.from_numpy(lengths),
        batch_first=True,
        enforce_sorted=False,
    )
    _, hidden = model.encoder(packed)
    logits = model.head(hidden[-1])
    return (
        torch.softmax(logits, dim=-1)[:, 1]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def score_horizon(
    model: TinyGRUVAD,
    features: torch.Tensor,
    targets: Sequence[int],
    *,
    horizon: int,
    batch_size: int = 512,
) -> np.ndarray:
    result = np.empty(len(targets), dtype=np.float64)
    for start in range(0, len(targets), int(batch_size)):
        end = min(len(targets), start + int(batch_size))
        result[start:end] = _score_window_batch(
            model,
            features,
            targets[start:end],
            horizon=horizon,
        )
    return result


def encode_stream_chunks(
    model: TinyGRUVAD,
    features: torch.Tensor,
    *,
    chunk_size: int = 2_048,
) -> torch.Tensor:
    if features.dim() != 3 or features.shape[0] != 1:
        raise ValueError("encode_stream_chunks expects [1,C,T] features")
    n_frames = int(features.shape[2])
    hidden = torch.zeros(
        model.config.num_layers,
        1,
        model.config.hidden_size,
        dtype=features.dtype,
        device=features.device,
    )
    outputs: list[torch.Tensor] = []
    for start in range(0, n_frames, int(chunk_size)):
        end = min(n_frames, start + int(chunk_size))
        sequence = features[0, :, start:end].transpose(0, 1).unsqueeze(0)
        encoded, hidden = model.encoder(sequence, hidden)
        outputs.append(encoded)
    if not outputs:
        return torch.empty(
            (1, 0, model.config.hidden_size),
            dtype=features.dtype,
            device=features.device,
        )
    return torch.cat(outputs, dim=1)


def _score_full_history(
    model: TinyGRUVAD,
    features: torch.Tensor,
    targets: Sequence[int],
    *,
    chunk_size: int = 2_048,
) -> np.ndarray:
    hidden = encode_stream_chunks(model, features, chunk_size=chunk_size)
    logits = model.forward_from_hidden(hidden)
    probabilities = torch.softmax(logits.transpose(1, 2), dim=-1)[
        ..., 1
    ].reshape(-1)
    indices = torch.as_tensor(targets, dtype=torch.long)
    return (
        probabilities[indices].detach().cpu().numpy().astype(np.float64)
    )


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    reference: CarReference,
    output_dir: Path,
    device: torch.device,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scores.npz"
    if output_path.is_file():
        raise FileExistsError(f"evaluation already exists: {output_path}")
    model = _load_model_from_checkpoint(checkpoint, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    scores: dict[str, np.ndarray] = {
        _horizon_key(horizon): np.full(
            reference.labels.size,
            np.nan,
            dtype=np.float32,
        )
        for horizon in HORIZONS
    }
    for segment_index, ((global_start, global_end), audio_path) in enumerate(
        zip(reference.segments, reference.segment_audio_paths),
        start=1,
    ):
        waveform = read_int16_audio(audio_path)
        tensor = (
            torch.from_numpy(waveform.astype(np.float32) * INT16_SCALE)
            .unsqueeze(0)
            .to(device)
        )
        features = frontend(tensor)
        length = int(global_end - global_start)
        global_indices = np.arange(global_start, global_end, dtype=np.int64)
        keep = reference.test_mask[global_indices]
        local_targets = VALID_START + np.flatnonzero(keep)
        if local_targets.size:
            for horizon in WINDOW_HORIZONS:
                values = score_horizon(
                    model,
                    features,
                    local_targets,
                    horizon=horizon,
                )
                scores[_horizon_key(horizon)][global_indices[keep]] = values
            full_values = _score_full_history(
                model,
                features,
                local_targets,
            )
            scores["HFULL"][global_indices[keep]] = full_values
        if segment_index % 100 == 0 or segment_index == len(
            reference.segments
        ):
            print(
                f"evaluated {segment_index}/{len(reference.segments)} "
                "utterances",
                flush=True,
            )
    for name, values in scores.items():
        if not np.all(np.isfinite(values[reference.test_mask])):
            raise RuntimeError(f"{name} contains non-finite test scores")
    np.savez_compressed(
        output_path,
        **scores,
        test_mask=reference.test_mask,
        labels=reference.labels,
    )
    _write_json(
        output_dir / "evaluation.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "scores": str(output_path),
            "horizons": [HORIZON_LABELS[item] for item in HORIZONS],
            "test_frames": int(np.count_nonzero(reference.test_mask)),
        },
    )
    return output_path


def evaluate_all(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
) -> None:
    reference = load_rf384_reference(protocol)
    for seed in MODEL_SEEDS:
        checkpoint = (
            results_dir / "checkpoints" / f"seed{seed}" / "best.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output_dir = results_dir / "evaluations" / f"seed{seed}"
        if (output_dir / "scores.npz").is_file():
            print(f"seed={seed} evaluation already complete", flush=True)
            continue
        evaluate_checkpoint(
            checkpoint,
            reference=reference,
            output_dir=output_dir,
            device=device,
        )


def run_causality_audit(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    audit_path = results_dir / "car_causality_audit.json"
    if audit_path.is_file():
        raise FileExistsError(f"causality audit already exists: {audit_path}")
    _set_seed(20260920)
    features = torch.randn(
        1,
        64,
        700,
        dtype=torch.float32,
        device=device,
    )
    seed_records: list[dict[str, Any]] = []
    all_pass = True
    for seed in MODEL_SEEDS:
        checkpoint = (
            results_dir / "checkpoints" / f"seed{seed}" / "best.pt"
        )
        model = _load_model_from_checkpoint(checkpoint, device)
        checks: dict[str, Any] = {}
        semantics_passes = True
        future_passes = True
        reset_passes = True
        for horizon in WINDOW_HORIZONS:
            for target in (400, 100, 20):
                value = _score_window_batch(
                    model,
                    features,
                    [target],
                    horizon=horizon,
                )[0]
                start = max(0, target - horizon + 1)
                sequence = features[
                    0, :, start : target + 1
                ].transpose(0, 1)
                logits = model(
                    sequence.transpose(0, 1).unsqueeze(0)
                )[:, :, -1]
                explicit = float(
                    torch.softmax(logits, dim=-1)[0, 1].cpu()
                )
                semantics_passes &= bool(
                    np.allclose(value, explicit, rtol=0.0, atol=1e-7)
                )
                future = features.clone()
                future[:, :, target + 1 :] += 100.0
                future_value = _score_window_batch(
                    model,
                    future,
                    [target],
                    horizon=horizon,
                )[0]
                future_passes &= bool(
                    np.allclose(
                        value,
                        future_value,
                        rtol=0.0,
                        atol=1e-7,
                    )
                )
                reset = features.clone()
                if start > 0:
                    reset[:, :, :start] += 100.0
                reset_value = _score_window_batch(
                    model,
                    reset,
                    [target],
                    horizon=horizon,
                )[0]
                reset_passes &= bool(
                    np.allclose(
                        value,
                        reset_value,
                        rtol=0.0,
                        atol=1e-7,
                    )
                )
        targets = [20, 150, 251, 377]
        batched = _score_window_batch(
            model,
            features,
            targets,
            horizon=128,
        )
        separate = np.asarray(
            [
                _score_window_batch(
                    model,
                    features,
                    [target],
                    horizon=128,
                )[0]
                for target in targets
            ]
        )
        batch_passes = bool(
            np.allclose(batched, separate, rtol=0.0, atol=0.0)
        )
        full = model.forward_features(features)
        chunked = encode_stream_chunks(model, features, chunk_size=97)
        chunk_passes = bool(
            torch.equal(full, chunked)
            or torch.allclose(full, chunked, rtol=1e-6, atol=1e-7)
        )
        checks = {
            "target_only_reset_semantics": bool(semantics_passes),
            "future_frame_isolation": bool(future_passes),
            "reset_isolation": bool(reset_passes),
            "batch_isolation": bool(batch_passes),
            "chunk_equivalence_with_carried_hidden_state": bool(
                chunk_passes
            ),
        }
        passed = all(checks.values())
        all_pass &= passed
        seed_records.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint),
                "checks": checks,
                "pass": bool(passed),
            }
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "created_at": _utc_now(),
        "device": str(device),
        "seeds": seed_records,
        "all_pass": bool(all_pass),
        "status": "PASS" if all_pass else "INVALID",
    }
    _write_json(audit_path, result)
    if not all_pass:
        raise RuntimeError("CAR causality audit failed; claims are forbidden")
    return result


@dataclass
class CarAnalysisData:
    protocol: dict[str, Any]
    reference: CarReference
    scores: dict[int, dict[str, np.ndarray]]
    losses: dict[int, dict[str, np.ndarray]]
    taxonomy: dict[int, dict[str, np.ndarray]]
    values: dict[int, np.ndarray]
    marble_taxonomy: dict[str, np.ndarray]
    marble_losses: dict[str, np.ndarray]
    marble_value: np.ndarray
    source_codes: np.ndarray
    speaker_codes: np.ndarray


def _load_evaluation_scores(
    results_dir: Path,
    seed: int,
    reference: CarReference,
) -> dict[str, np.ndarray]:
    path = results_dir / "evaluations" / f"seed{seed}" / "scores.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    arrays = _load_npz_arrays(path)
    required = {_horizon_key(horizon) for horizon in HORIZONS}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"evaluation seed {seed} is missing {missing}")
    scores: dict[str, np.ndarray] = {}
    for key in required:
        values = np.asarray(arrays[key], dtype=np.float64)
        if values.size != reference.labels.size:
            raise ValueError(f"{key} has the wrong frame count for seed {seed}")
        if not np.all(np.isfinite(values[reference.test_mask])):
            raise ValueError(f"{key} has non-finite test scores for seed {seed}")
        scores[key] = values
    return scores


def _frame_bce(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.clip(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        1e-7,
        1.0 - 1e-7,
    )
    return -(
        labels * np.log(scores)
        + (1.0 - labels) * np.log1p(-scores)
    )


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
    short_sufficient = short_correct & long_correct
    refinable = (~short_correct) & long_correct
    irreducible = (~short_correct) & (~long_correct)
    harm = short_correct & (~long_correct)
    short_loss = _frame_bce(labels, short_scores)
    long_loss = _frame_bce(labels, long_scores)
    return {
        "short_correct": short_correct,
        "long_correct": long_correct,
        "SS": short_sufficient,
        "R": refinable,
        "I": irreducible,
        "H": harm,
        "short_loss": short_loss,
        "long_loss": long_loss,
        "value": short_loss - long_loss,
    }


def _cluster_codes(
    values: np.ndarray,
    order: Sequence[str],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if valid_mask is None:
        valid = np.ones(values.size, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != values.size:
            raise ValueError("cluster values and valid mask differ in size")
    lookup = {str(name): index for index, name in enumerate(order)}
    result = np.full(values.size, -1, dtype=np.int64)
    selected_values = values[valid]
    selected_codes = np.asarray(
        [lookup.get(str(value), -1) for value in selected_values],
        dtype=np.int64,
    )
    if np.any(selected_codes < 0):
        missing = sorted(
            {
                str(value)
                for value, code in zip(selected_values, selected_codes)
                if int(code) < 0
            }
        )
        raise ValueError(f"unknown cluster values: {missing[:5]}")
    result[valid] = selected_codes
    return result


def _bootstrap_cluster_weights(
    n_clusters: int,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int,
) -> np.ndarray:
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(int(seed))
    weights = np.empty((int(n_clusters), int(repeats)), dtype=np.int16)
    for repeat in range(int(repeats)):
        sampled = rng.integers(
            0,
            int(n_clusters),
            size=int(n_clusters),
        )
        weights[:, repeat] = np.bincount(
            sampled,
            minlength=int(n_clusters),
        )
    return weights


def _group_bootstrap_seed(group: str, unit: str) -> int:
    return _stable_seed("car-bootstrap", unit, group, BOOTSTRAP_SEED_BASE)


def _cluster_weights_for_group(
    group: str,
    *,
    n_clusters: int,
    unit: str,
) -> np.ndarray:
    return _bootstrap_cluster_weights(
        n_clusters,
        repeats=BOOTSTRAP_REPEATS,
        seed=_group_bootstrap_seed(group, unit),
    )


def _cluster_sums(
    values: np.ndarray,
    mask: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(mask, dtype=bool)
    selected_values = np.asarray(values, dtype=np.float64)
    selected_codes = np.asarray(codes, dtype=np.int64)
    if not (selected.size == selected_values.size == selected_codes.size):
        raise ValueError("cluster metric arrays must have equal lengths")
    if np.any(selected_codes[selected] < 0):
        raise ValueError("selected cluster codes must be non-negative")
    if np.any(selected_codes[selected] >= int(n_clusters)):
        raise ValueError("selected cluster code exceeds the cluster count")
    sums = np.bincount(
        selected_codes[selected],
        weights=selected_values[selected],
        minlength=int(n_clusters),
    ).astype(np.float64)
    counts = np.bincount(
        selected_codes[selected],
        minlength=int(n_clusters),
    ).astype(np.float64)
    return sums, counts


def _percentile_interval(
    values: np.ndarray,
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _point_ratio(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
) -> float | None:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return None
    return float(numerator) / denominator_value


def _point_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    local = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(np.mean(local)) if local.size else None


def _mean_or_none(values: Iterable[float | int | None]) -> float | None:
    local = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(local)) if local else None


def _bootstrap_mean_metrics(
    *,
    data: CarAnalysisData,
    group: str,
    mask: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    unit: str,
    masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float | None]]:
    codes = (
        data.source_codes
        if unit == "source_cluster"
        else data.speaker_codes
    )
    n_clusters = (
        len(data.reference.source_order)
        if unit == "source_cluster"
        else len(data.reference.speaker_order)
    )
    weights = _cluster_weights_for_group(
        group,
        n_clusters=n_clusters,
        unit=unit,
    )
    result: dict[str, dict[str, float | None]] = {}
    for name, values in metrics.items():
        metric_mask = (
            np.asarray(masks[name], dtype=bool)
            if masks is not None and name in masks
            else np.asarray(mask, dtype=bool)
        )
        sums, counts = _cluster_sums(
            values,
            metric_mask,
            codes,
            n_clusters,
        )
        sampled_counts = counts @ weights
        sampled_sums = sums @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            replicates = np.where(
                sampled_counts > 0.0,
                sampled_sums / sampled_counts,
                np.nan,
            )
        low, high = _percentile_interval(replicates)
        result[name] = {
            "estimate": _point_mean(values, metric_mask),
            "ci95_low": low,
            "ci95_high": high,
        }
    return result


def _bootstrap_ratio_metrics(
    *,
    data: CarAnalysisData,
    group: str,
    numerators: Mapping[str, np.ndarray],
    denominators: Mapping[str, np.ndarray],
    unit: str,
) -> dict[str, dict[str, float | None]]:
    codes = (
        data.source_codes
        if unit == "source_cluster"
        else data.speaker_codes
    )
    n_clusters = (
        len(data.reference.source_order)
        if unit == "source_cluster"
        else len(data.reference.speaker_order)
    )
    weights = _cluster_weights_for_group(
        group,
        n_clusters=n_clusters,
        unit=unit,
    )
    result: dict[str, dict[str, float | None]] = {}
    for name, numerator in numerators.items():
        if name not in denominators:
            raise ValueError(f"missing ratio denominator for {name}")
        numerator_mask = np.asarray(numerator, dtype=bool)
        denominator_mask = np.asarray(denominators[name], dtype=bool)
        if numerator_mask.size != denominator_mask.size:
            raise ValueError("ratio numerator and denominator differ in size")
        if np.any(numerator_mask & ~denominator_mask):
            raise ValueError(
                f"ratio numerator for {name} is not a subset of its "
                "denominator"
            )
        numerator_sums, _ = _cluster_sums(
            numerator_mask.astype(np.float64),
            numerator_mask,
            codes,
            n_clusters,
        )
        denominator_sums, _ = _cluster_sums(
            denominator_mask.astype(np.float64),
            denominator_mask,
            codes,
            n_clusters,
        )
        sampled_numerators = numerator_sums @ weights
        sampled_denominators = denominator_sums @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            replicates = np.where(
                sampled_denominators > 0.0,
                sampled_numerators / sampled_denominators,
                np.nan,
            )
        low, high = _percentile_interval(replicates)
        numerator_count = float(np.count_nonzero(numerator_mask))
        denominator_count = float(np.count_nonzero(denominator_mask))
        estimate = (
            numerator_count / denominator_count
            if denominator_count > 0.0
            else None
        )
        result[name] = {
            "estimate": estimate,
            "ci95_low": low,
            "ci95_high": high,
        }
    return result


def _bootstrap_difference(
    *,
    data: CarAnalysisData,
    group: str,
    focal_mask: np.ndarray,
    reference_mask: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | None]:
    n_clusters = len(data.reference.source_order)
    weights = _cluster_weights_for_group(
        group,
        n_clusters=n_clusters,
        unit="source_cluster",
    )
    focal_sums, focal_counts = _cluster_sums(
        values,
        focal_mask,
        data.source_codes,
        n_clusters,
    )
    reference_sums, reference_counts = _cluster_sums(
        values,
        reference_mask,
        data.source_codes,
        n_clusters,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        focal = np.where(
            focal_counts @ weights > 0.0,
            (focal_sums @ weights) / (focal_counts @ weights),
            np.nan,
        )
        reference = np.where(
            reference_counts @ weights > 0.0,
            (reference_sums @ weights) / (reference_counts @ weights),
            np.nan,
        )
    replicates = focal - reference
    low, high = _percentile_interval(replicates)
    point_focal = _point_mean(values, focal_mask)
    point_reference = _point_mean(values, reference_mask)
    point = (
        None
        if point_focal is None or point_reference is None
        else float(point_focal - point_reference)
    )
    return {
        "difference": point,
        "ci95_low": low,
        "ci95_high": high,
        "focal_mean": point_focal,
        "reference_mean": point_reference,
    }


def _bootstrap_range(
    *,
    data: CarAnalysisData,
    group: str,
    masks: Sequence[np.ndarray],
    values: np.ndarray,
) -> dict[str, float | None]:
    n_clusters = len(data.reference.source_order)
    weights = _cluster_weights_for_group(
        group,
        n_clusters=n_clusters,
        unit="source_cluster",
    )
    means: list[np.ndarray] = []
    points: list[float] = []
    for mask in masks:
        sums, counts = _cluster_sums(
            values,
            mask,
            data.source_codes,
            n_clusters,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            means.append(
                np.where(
                    counts @ weights > 0.0,
                    (sums @ weights) / (counts @ weights),
                    np.nan,
                )
            )
        point = _point_mean(values, mask)
        if point is not None:
            points.append(float(point))
    if not points:
        return {
            "range": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    matrix = np.stack(means, axis=0)
    with np.errstate(invalid="ignore"):
        replicates = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0)
    low, high = _percentile_interval(replicates)
    return {
        "range": float(max(points) - min(points)),
        "ci95_low": low,
        "ci95_high": high,
    }


def load_car_analysis_data(
    protocol: Mapping[str, Any],
    *,
    results_dir: Path,
) -> CarAnalysisData:
    reference = load_rf384_reference(protocol)
    scores = {
        seed: _load_evaluation_scores(results_dir, seed, reference)
        for seed in MODEL_SEEDS
    }
    losses: dict[int, dict[str, np.ndarray]] = {}
    taxonomy: dict[int, dict[str, np.ndarray]] = {}
    values: dict[int, np.ndarray] = {}
    for seed in MODEL_SEEDS:
        losses[seed] = {
            key: _frame_bce(reference.labels, values_by_horizon)
            for key, values_by_horizon in scores[seed].items()
        }
        taxonomy[seed] = _taxonomy_from_scores(
            reference.labels,
            scores[seed]["H64"],
            scores[seed]["HFULL"],
        )
        values[seed] = taxonomy[seed]["value"]
    marble_taxonomy = _taxonomy_from_scores(
        reference.labels,
        reference.short_scores,
        reference.full_adaptive_scores,
    )
    marble_losses = {
        "short": _frame_bce(reference.labels, reference.short_scores),
        "long": _frame_bce(
            reference.labels,
            reference.full_adaptive_scores,
        ),
    }
    marble_value = marble_losses["short"] - marble_losses["long"]
    source_codes = _cluster_codes(
        reference.source_key,
        reference.source_order,
        valid_mask=reference.test_mask,
    )
    speaker_codes = _cluster_codes(
        reference.speaker_ids,
        reference.speaker_order,
        valid_mask=reference.test_mask,
    )
    return CarAnalysisData(
        protocol=dict(protocol),
        reference=reference,
        scores=scores,
        losses=losses,
        taxonomy=taxonomy,
        values=values,
        marble_taxonomy=marble_taxonomy,
        marble_losses=marble_losses,
        marble_value=marble_value,
        source_codes=source_codes,
        speaker_codes=speaker_codes,
    )


def _aggregate_indicator(
    seed_arrays: Mapping[int, Mapping[str, np.ndarray]],
    key: str,
) -> np.ndarray:
    return np.mean(
        np.stack(
            [
                np.asarray(seed_arrays[seed][key], dtype=np.float64)
                for seed in MODEL_SEEDS
            ],
            axis=0,
        ),
        axis=0,
    )


def _aggregate_values(
    values_by_seed: Mapping[int, np.ndarray],
) -> np.ndarray:
    return np.mean(
        np.stack(
            [
                np.asarray(values_by_seed[seed], dtype=np.float64)
                for seed in MODEL_SEEDS
            ],
            axis=0,
        ),
        axis=0,
    )


def _condition_groups(reference: CarReference) -> list[str]:
    noise_groups = [
        f"noise/{name}"
        for name in sorted(
            {
                str(value)
                for value in reference.noise_name[reference.test_mask]
                if str(value) != "Clean"
            }
        )
    ]
    return [
        "all",
        "clean",
        "seen",
        "unseen",
        *CONDITION_CELLS,
        *noise_groups,
    ]


def _mask_for_group(reference: CarReference, group: str) -> np.ndarray:
    if group in reference.group_masks:
        return np.asarray(reference.group_masks[group], dtype=bool)
    if group.startswith("noise/"):
        noise_name = group.split("/", 1)[1]
        return reference.test_mask & (reference.noise_name == noise_name)
    raise ValueError(f"unknown CAR group: {group}")


def _taxonomy_metrics(
    taxonomy: Mapping[str, np.ndarray],
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    selected = np.asarray(mask, dtype=bool)
    frames = int(np.count_nonzero(selected))
    if frames == 0:
        return {
            "frames": 0,
            "p_SS": None,
            "p_R": None,
            "p_I": None,
            "p_H": None,
            "availability_net": None,
            "mean_value": None,
        }
    rates = {
        name: float(
            np.sum(
                np.asarray(taxonomy[name], dtype=np.float64)[selected]
            )
            / frames
        )
        for name in ("SS", "R", "I", "H")
    }
    return {
        "frames": frames,
        "p_SS": rates["SS"],
        "p_R": rates["R"],
        "p_I": rates["I"],
        "p_H": rates["H"],
        "availability_net": float(rates["R"] - rates["H"]),
        "mean_value": _point_mean(values, mask),
    }


def _aggregate_taxonomy(
    taxonomy: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: _aggregate_indicator(taxonomy, key)
        for key in (
            "short_correct",
            "long_correct",
            "SS",
            "R",
            "I",
            "H",
            "short_loss",
            "long_loss",
            "value",
        )
    }


def _taxonomy_bootstrap(
    data: CarAnalysisData,
    *,
    group: str,
    mask: np.ndarray,
    taxonomy: Mapping[str, np.ndarray],
    values: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    metrics = {
        "p_SS": np.asarray(taxonomy["SS"], dtype=np.float64),
        "p_R": np.asarray(taxonomy["R"], dtype=np.float64),
        "p_I": np.asarray(taxonomy["I"], dtype=np.float64),
        "p_H": np.asarray(taxonomy["H"], dtype=np.float64),
        "availability_net": (
            np.asarray(taxonomy["R"], dtype=np.float64)
            - np.asarray(taxonomy["H"], dtype=np.float64)
        ),
        "mean_value": np.asarray(values, dtype=np.float64),
    }
    return _bootstrap_mean_metrics(
        data=data,
        group=group,
        mask=mask,
        metrics=metrics,
        unit="source_cluster",
    )


def _taxonomy_sensitivity_bootstrap(
    data: CarAnalysisData,
    *,
    group: str,
    mask: np.ndarray,
    taxonomy: Mapping[str, np.ndarray],
    values: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    metrics = {
        "p_SS": np.asarray(taxonomy["SS"], dtype=np.float64),
        "p_R": np.asarray(taxonomy["R"], dtype=np.float64),
        "p_I": np.asarray(taxonomy["I"], dtype=np.float64),
        "p_H": np.asarray(taxonomy["H"], dtype=np.float64),
        "availability_net": (
            np.asarray(taxonomy["R"], dtype=np.float64)
            - np.asarray(taxonomy["H"], dtype=np.float64)
        ),
        "mean_value": np.asarray(values, dtype=np.float64),
    }
    return _bootstrap_mean_metrics(
        data=data,
        group=group,
        mask=mask,
        metrics=metrics,
        unit="speaker_sensitivity",
    )


def _taxonomy_output_row(
    data: CarAnalysisData,
    *,
    scope: str,
    seed: int | str,
    group: str,
    taxonomy: Mapping[str, np.ndarray],
    values: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask = _mask_for_group(data.reference, group)
    point = _taxonomy_metrics(taxonomy, values, mask)
    source = _taxonomy_bootstrap(
        data,
        group=group,
        mask=mask,
        taxonomy=taxonomy,
        values=values,
    )
    speaker = _taxonomy_sensitivity_bootstrap(
        data,
        group=group,
        mask=mask,
        taxonomy=taxonomy,
        values=values,
    )
    row: dict[str, Any] = {
        "scope": scope,
        "seed": seed,
        "group": group,
        **point,
    }
    bootstrap_rows: list[dict[str, Any]] = []
    for metric in (
        "p_SS",
        "p_R",
        "p_I",
        "p_H",
        "availability_net",
        "mean_value",
    ):
        source_metric = source[metric]
        speaker_metric = speaker[metric]
        row[f"source_ci95_low_{metric}"] = source_metric["ci95_low"]
        row[f"source_ci95_high_{metric}"] = source_metric["ci95_high"]
        row[f"speaker_ci95_low_{metric}"] = speaker_metric["ci95_low"]
        row[f"speaker_ci95_high_{metric}"] = speaker_metric["ci95_high"]
        bootstrap_rows.append(
            {
                "scope": scope,
                "seed": seed,
                "group": group,
                "metric": metric,
                "estimate": source_metric["estimate"],
                "source_ci95_low": source_metric["ci95_low"],
                "source_ci95_high": source_metric["ci95_high"],
                "speaker_ci95_low": speaker_metric["ci95_low"],
                "speaker_ci95_high": speaker_metric["ci95_high"],
            }
        )
    return row, bootstrap_rows


def run_car1(
    data: CarAnalysisData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute sparse-refinability taxonomy and cluster uncertainty."""
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    aggregate_values = _aggregate_values(data.values)
    scopes: list[tuple[str, int | str, Mapping[str, np.ndarray], np.ndarray]] = [
        ("seed", seed, data.taxonomy[seed], data.values[seed])
        for seed in MODEL_SEEDS
    ]
    scopes.append(
        ("aggregate", "aggregate", aggregate_taxonomy, aggregate_values)
    )
    by_seed: list[dict[str, Any]] = []
    by_condition: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for scope, seed, taxonomy, values in scopes:
        for group in _condition_groups(data.reference):
            row, local_bootstrap = _taxonomy_output_row(
                data,
                scope=scope,
                seed=seed,
                group=group,
                taxonomy=taxonomy,
                values=values,
            )
            by_condition.append(row)
            bootstrap_rows.extend(local_bootstrap)
            if scope == "seed":
                by_seed.append(row)
    _write_csv(output_dir / "car1_taxonomy_by_seed.csv", by_seed)
    _write_csv(
        output_dir / "car1_taxonomy_by_condition.csv",
        by_condition,
    )
    _write_csv(output_dir / "car1_bootstrap.csv", bootstrap_rows)
    return {
        "by_seed": by_seed,
        "by_condition": by_condition,
        "bootstrap": bootstrap_rows,
        "aggregate": next(
            row
            for row in by_condition
            if row["scope"] == "aggregate" and row["group"] == "all"
        ),
    }


def _horizon_correctness(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    labels_bool = np.asarray(labels, dtype=np.int64).reshape(-1) == 1
    return np.stack(
        [
            (
                np.asarray(scores[_horizon_key(horizon)], dtype=np.float64)
                >= DECISION_THRESHOLD
            )
            == labels_bool
            for horizon in HORIZONS
        ],
        axis=1,
    )


def _horizon_diagnostics(
    correctness: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    correctness = np.asarray(correctness, dtype=bool)
    first_correct = np.full(correctness.shape[0], -1, dtype=np.int16)
    stable_sufficient = np.full(correctness.shape[0], -1, dtype=np.int16)
    for index in range(correctness.shape[1]):
        missing = (first_correct < 0) & correctness[:, index]
        first_correct[missing] = index
    suffix = np.ones(correctness.shape[0], dtype=bool)
    for index in range(correctness.shape[1] - 1, -1, -1):
        suffix &= correctness[:, index]
        stable_sufficient[suffix] = index
    non_monotone = np.any(
        correctness[:, :-1] & (~correctness[:, 1:]),
        axis=1,
    )
    never_correct = ~np.any(correctness, axis=1)
    unstable_correct = np.any(correctness, axis=1) & (
        stable_sufficient < 0
    )
    long_share = np.full(correctness.shape[0], np.nan, dtype=np.float64)
    denominator = (
        np.asarray(values["H64"], dtype=np.float64)
        - np.asarray(values["HFULL"], dtype=np.float64)
    )
    numerator = (
        np.asarray(values["H64"], dtype=np.float64)
        - np.asarray(values["H384"], dtype=np.float64)
    )
    defined = denominator > VALUE_SPARSITY_EPSILON
    long_share[defined] = numerator[defined] / denominator[defined]
    return {
        "correctness": correctness,
        "first_correct": first_correct,
        "stable_sufficient": stable_sufficient,
        "non_monotone": non_monotone,
        "never_correct": never_correct,
        "unstable_correct": unstable_correct,
        "long_history_share": long_share,
    }


def _mean_defined(values: np.ndarray, mask: np.ndarray) -> float | None:
    local = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    local = local[np.isfinite(local)]
    return float(np.mean(local)) if local.size else None


def _f1_for_mask(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    local_labels = np.asarray(labels, dtype=np.int64)[selected]
    if local_labels.size == 0 or np.unique(local_labels).size <= 1:
        return None
    local_predictions = (
        np.asarray(scores, dtype=np.float64)[selected]
        >= DECISION_THRESHOLD
    )
    return float(f1_score(local_labels, local_predictions, zero_division=0))


def _car2_history_rows(
    data: CarAnalysisData,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        for group in _condition_groups(data.reference):
            mask = _mask_for_group(data.reference, group)
            losses = data.losses[seed]
            for horizon in HORIZONS:
                key = _horizon_key(horizon)
                score = data.scores[seed][key]
                loss = losses[key]
                value = losses["H64"] - loss
                rows.append(
                    {
                        "scope": "seed",
                        "seed": seed,
                        "group": group,
                        "horizon": key,
                        "horizon_index": HORIZONS.index(horizon),
                        "frames": int(np.count_nonzero(mask)),
                        "loss": _point_mean(loss, mask),
                        "frame_error": float(
                            np.mean(
                                (
                                    (score >= DECISION_THRESHOLD)
                                    != (
                                        np.asarray(
                                            data.reference.labels,
                                            dtype=np.int64,
                                        )
                                        == 1
                                    )
                                )[mask]
                            )
                        )
                        if np.any(mask)
                        else None,
                        "f1": _f1_for_mask(
                            data.reference.labels,
                            score,
                            mask,
                        ),
                        "signed_incremental_value": _point_mean(
                            value,
                            mask,
                        ),
                    }
                )
    for group in _condition_groups(data.reference):
        mask = _mask_for_group(data.reference, group)
        for horizon in HORIZONS:
            key = _horizon_key(horizon)
            seed_rows = [
                row
                for row in rows
                if row["group"] == group and row["horizon"] == key
            ]
            rows.append(
                {
                    "scope": "aggregate",
                    "seed": "aggregate",
                    "group": group,
                    "horizon": key,
                    "horizon_index": HORIZONS.index(horizon),
                    "frames": int(np.count_nonzero(mask)),
                    "loss": _mean_or_none(
                        [row["loss"] for row in seed_rows]
                    ),
                    "frame_error": _mean_or_none(
                        [row["frame_error"] for row in seed_rows]
                    ),
                    "f1": _mean_or_none([row["f1"] for row in seed_rows]),
                    "signed_incremental_value": _point_mean(
                        _aggregate_values(
                            {
                                seed: (
                                    data.losses[seed]["H64"]
                                    - data.losses[seed][key]
                                )
                                for seed in MODEL_SEEDS
                            }
                        ),
                        mask,
                    ),
                }
            )
    return rows


def _nonmonotonicity_row(
    data: CarAnalysisData,
    *,
    scope: str,
    seed: int | str,
    group: str,
    values: Mapping[str, np.ndarray],
    correctness: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mask = _mask_for_group(data.reference, group)
    diagnostics = _horizon_diagnostics(correctness, values)
    not_short = (~diagnostics["correctness"][:, 0]) & mask
    first_defined = diagnostics["first_correct"] >= 0
    stable_defined = diagnostics["stable_sufficient"] >= 0
    stable_long = diagnostics["stable_sufficient"] >= 3
    point = {
        "scope": scope,
        "seed": seed,
        "group": group,
        "frames": int(np.count_nonzero(mask)),
        "non_monotone_rate": (
            float(
                np.count_nonzero(
                    diagnostics["non_monotone"] & not_short
                )
                / np.count_nonzero(not_short)
            )
            if np.any(not_short)
            else None
        ),
        "never_correct_rate": (
            float(
                np.count_nonzero(
                    diagnostics["never_correct"] & mask
                )
                / np.count_nonzero(mask)
            )
            if np.any(mask)
            else None
        ),
        "unstable_correct_rate": (
            float(
                np.count_nonzero(
                    diagnostics["unstable_correct"] & mask
                )
                / np.count_nonzero(mask)
            )
            if np.any(mask)
            else None
        ),
        "first_correct_defined_rate": (
            float(np.count_nonzero(first_defined & mask) / np.count_nonzero(mask))
            if np.any(mask)
            else None
        ),
        "stable_sufficient_defined_rate": (
            float(
                np.count_nonzero(stable_defined & mask)
                / np.count_nonzero(mask)
            )
            if np.any(mask)
            else None
        ),
        "stable_sufficient_long_rate": (
            float(
                np.count_nonzero(stable_long & stable_defined & mask)
                / np.count_nonzero(stable_defined & mask)
            )
            if np.any(stable_defined & mask)
            else None
        ),
        "mean_first_correct_index": _mean_defined(
            diagnostics["first_correct"].astype(np.float64),
            first_defined & mask,
        ),
        "mean_stable_sufficient_index": _mean_defined(
            diagnostics["stable_sufficient"].astype(np.float64),
            stable_defined & mask,
        ),
        "long_history_share": _mean_defined(
            diagnostics["long_history_share"],
            mask,
        ),
    }
    for index, horizon in enumerate(HORIZONS):
        point[f"first_correct_{_horizon_key(horizon)}"] = int(
            np.count_nonzero((diagnostics["first_correct"] == index) & mask)
        )
        point[f"stable_sufficient_{_horizon_key(horizon)}"] = int(
            np.count_nonzero(
                (diagnostics["stable_sufficient"] == index) & mask
            )
        )
    mean_values = {
        key: _point_mean(np.asarray(values[key], dtype=np.float64), mask)
        for key in HORIZON_LABELS.values()
    }
    for index in range(1, len(HORIZONS)):
        previous = HORIZONS[index - 1]
        current = HORIZONS[index]
        previous_value = mean_values[_horizon_key(previous)]
        current_value = mean_values[_horizon_key(current)]
        point[
            "delta_v_"
            f"{_horizon_key(previous).lower()}_"
            f"{_horizon_key(current).lower()}"
        ] = (
            None
            if previous_value is None or current_value is None
            else float(current_value - previous_value)
        )
    denominators = {
        "non_monotone_rate": not_short,
        "never_correct_rate": mask,
        "unstable_correct_rate": mask,
        "first_correct_defined_rate": mask,
        "stable_sufficient_defined_rate": mask,
        "stable_sufficient_long_rate": stable_defined & mask,
        "mean_first_correct_index": first_defined & mask,
        "mean_stable_sufficient_index": stable_defined & mask,
        "long_history_share": np.isfinite(
            diagnostics["long_history_share"]
        )
        & mask,
    }
    ratio_numerators = {
        "non_monotone_rate": diagnostics["non_monotone"] & not_short,
        "never_correct_rate": diagnostics["never_correct"] & mask,
        "unstable_correct_rate": diagnostics["unstable_correct"] & mask,
        "first_correct_defined_rate": first_defined & mask,
        "stable_sufficient_defined_rate": stable_defined & mask,
        "stable_sufficient_long_rate": stable_long & stable_defined & mask,
    }
    mean_metrics = {
        "mean_first_correct_index": (
            diagnostics["first_correct"].astype(np.float64)
        ),
        "mean_stable_sufficient_index": (
            diagnostics["stable_sufficient"].astype(np.float64)
        ),
        "long_history_share": diagnostics["long_history_share"],
    }
    mean_masks = {
        "mean_first_correct_index": first_defined & mask,
        "mean_stable_sufficient_index": stable_defined & mask,
        "long_history_share": (
            np.isfinite(diagnostics["long_history_share"]) & mask
        ),
    }
    bootstrap = _bootstrap_ratio_metrics(
        data=data,
        group=group,
        numerators=ratio_numerators,
        denominators=denominators,
        unit="source_cluster",
    )
    bootstrap.update(
        _bootstrap_mean_metrics(
            data=data,
            group=group,
            mask=mask,
            metrics=mean_metrics,
            unit="source_cluster",
            masks=mean_masks,
        )
    )
    for metric in (*ratio_numerators, *mean_metrics):
        point[f"source_ci95_low_{metric}"] = bootstrap[metric]["ci95_low"]
        point[f"source_ci95_high_{metric}"] = bootstrap[metric][
            "ci95_high"
        ]
    return point, denominators


def run_car2(
    data: CarAnalysisData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute horizon demand and non-monotonicity diagnostics."""
    history_rows = _car2_history_rows(data)
    non_monotone_rows: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        correctness = _horizon_correctness(
            data.scores[seed],
            data.reference.labels,
        )
        for group in _condition_groups(data.reference):
            row, _ = _nonmonotonicity_row(
                data,
                scope="seed",
                seed=seed,
                group=group,
                values=data.losses[seed],
                correctness=correctness,
            )
            non_monotone_rows.append(row)
    aggregate_values = {
        key: _aggregate_values(
            {
                seed: data.losses[seed][key]
                for seed in MODEL_SEEDS
            }
        )
        for key in HORIZON_LABELS.values()
    }
    aggregate_correctness = (
        np.mean(
            np.stack(
                [
                    _horizon_correctness(
                        data.scores[seed],
                        data.reference.labels,
                    ).astype(np.float64)
                    for seed in MODEL_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
        >= DECISION_THRESHOLD
    )
    for group in _condition_groups(data.reference):
        row, _ = _nonmonotonicity_row(
            data,
            scope="aggregate",
            seed="aggregate",
            group=group,
            values=aggregate_values,
            correctness=aggregate_correctness,
        )
        non_monotone_rows.append(row)
    _write_csv(output_dir / "car2_history_demand.csv", history_rows)
    _write_csv(
        output_dir / "car2_nonmonotonicity.csv",
        non_monotone_rows,
    )
    return {
        "history": history_rows,
        "non_monotonicity": non_monotone_rows,
        "aggregate": next(
            row
            for row in non_monotone_rows
            if row["scope"] == "aggregate" and row["group"] == "all"
        ),
    }


def _feature_bins(feature: str) -> tuple[str, ...]:
    return (
        DISTANCE_BINS
        if feature
        in {
            "onset_distance",
            "offset_distance",
            "posterior_transition_distance",
        }
        else DURATION_BINS
    )


def _event_valid_mask(
    data: CarAnalysisData,
    *,
    feature: str,
    group: str,
) -> np.ndarray:
    mask = _mask_for_group(data.reference, group)
    if feature == "uncertainty_persistence":
        mask = mask & data.reference.selected
    return mask


def _marble_event_lookup() -> dict[tuple[str, str, str], dict[str, str]]:
    rows = _read_csv(AE3_FEATURE_BINS_PATH)
    return {
        (row["feature"], row["group"], row["bin"]): row for row in rows
    }


def _event_structure_rows(
    data: CarAnalysisData,
) -> list[dict[str, Any]]:
    marble_rows = _marble_event_lookup()
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    aggregate_values = _aggregate_values(data.values)
    scopes: list[tuple[str, int | str, Mapping[str, np.ndarray], np.ndarray]] = [
        ("seed", seed, data.taxonomy[seed], data.values[seed])
        for seed in MODEL_SEEDS
    ]
    scopes.append(
        ("aggregate", "aggregate", aggregate_taxonomy, aggregate_values)
    )
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        values = data.reference.event_features[feature]
        for group in CELL_ORDER:
            valid = _event_valid_mask(data, feature=feature, group=group)
            for bin_label in _feature_bins(feature):
                bin_mask = valid & _feature_bin_mask(
                    feature,
                    values,
                    bin_label,
                )
                marble = marble_rows.get((feature, group, bin_label), {})
                marble_p_r = _float_or_none(
                    marble.get("refinable_rate")
                )
                marble_p_i = _float_or_none(
                    marble.get("irreducible_rate")
                )
                marble_p_h = _float_or_none(
                    marble.get("refinement_harm_rate")
                )
                marble_frames = _float_or_none(marble.get("frames"))
                if marble_frames is None or marble_frames <= 0.0:
                    marble_frames = 0
                marble_p_ss = (
                    1.0 - marble_p_r - marble_p_i - marble_p_h
                    if (
                        marble_p_r is not None
                        and marble_p_i is not None
                        and marble_p_h is not None
                    )
                    else None
                )
                rows.append(
                    {
                        "architecture": "MarbleNet",
                        "scope": "reference",
                        "seed": "reference",
                        "feature": feature,
                        "bin": bin_label,
                        "group": group,
                        "frames": int(marble_frames),
                        "mean_feature": _point_mean(values, bin_mask),
                        "mean_value": _point_mean(
                            data.marble_value,
                            bin_mask,
                        ),
                        "p_SS": marble_p_ss,
                        "p_R": marble_p_r,
                        "p_I": marble_p_i,
                        "p_H": marble_p_h,
                        "availability_net": _float_or_none(
                            marble.get("availability_net")
                        ),
                    }
                )
                for scope, seed, taxonomy, scope_values in scopes:
                    point = _taxonomy_metrics(
                        taxonomy,
                        scope_values,
                        bin_mask,
                    )
                    rows.append(
                        {
                            "architecture": "TinyGRU",
                            "scope": scope,
                            "seed": seed,
                            "feature": feature,
                            "bin": bin_label,
                            "group": group,
                            "frames": point["frames"],
                            "mean_feature": _point_mean(values, bin_mask),
                            "mean_value": point["mean_value"],
                            "p_SS": point["p_SS"],
                            "p_R": point["p_R"],
                            "p_I": point["p_I"],
                            "p_H": point["p_H"],
                            "availability_net": point["availability_net"],
                        }
                    )
    return rows


def _event_comparison_rows(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tiny_lookup = {
        (row["feature"], row["group"], row["bin"]): row
        for row in event_rows
        if row["architecture"] == "TinyGRU"
        and row["scope"] == "aggregate"
    }
    marble_lookup = {
        (row["feature"], row["group"], row["bin"]): row
        for row in event_rows
        if row["architecture"] == "MarbleNet"
    }
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        for group in CELL_ORDER:
            pairs: list[tuple[float, float, str]] = []
            for bin_label in _feature_bins(feature):
                tiny = tiny_lookup.get((feature, group, bin_label), {})
                marble = marble_lookup.get((feature, group, bin_label), {})
                tiny_value = tiny.get("availability_net")
                marble_value = marble.get("availability_net")
                if tiny_value is None or marble_value is None:
                    continue
                pairs.append(
                    (
                        float(tiny_value),
                        float(marble_value),
                        bin_label,
                    )
                )
            if not pairs:
                rows.append(
                    {
                        "feature": feature,
                        "group": group,
                        "bins_compared": 0,
                        "bin_sign_agreement": None,
                        "bin_spearman_rank_correlation": None,
                        "qualitative_peak_location_agreement": None,
                    }
                )
                continue
            tiny_values = np.asarray([item[0] for item in pairs])
            marble_values = np.asarray([item[1] for item in pairs])
            sign_agreement = float(
                np.mean(np.sign(tiny_values) == np.sign(marble_values))
            )
            if (
                tiny_values.size >= 3
                and np.unique(tiny_values).size > 1
                and np.unique(marble_values).size > 1
            ):
                rank = spearmanr(tiny_values, marble_values).statistic
                rank_value = (
                    float(rank) if np.isfinite(rank) else None
                )
            else:
                rank_value = None
            tiny_peak = max(pairs, key=lambda item: item[0])[2]
            marble_peak = max(pairs, key=lambda item: item[1])[2]
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "bins_compared": len(pairs),
                    "bin_sign_agreement": sign_agreement,
                    "bin_spearman_rank_correlation": rank_value,
                    "qualitative_peak_location_agreement": bool(
                        tiny_peak == marble_peak
                    ),
                    "tiny_peak_bin": tiny_peak,
                    "marble_peak_bin": marble_peak,
                }
            )
    return rows


def _profile_rows(
    data: CarAnalysisData,
    *,
    feature: str,
) -> list[dict[str, Any]]:
    values = data.reference.event_features[feature]
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    aggregate_values = _aggregate_values(data.values)
    scopes: list[tuple[str, int | str, Mapping[str, np.ndarray], np.ndarray]] = [
        ("seed", seed, data.taxonomy[seed], data.values[seed])
        for seed in MODEL_SEEDS
    ]
    scopes.append(
        ("aggregate", "aggregate", aggregate_taxonomy, aggregate_values)
    )
    rows: list[dict[str, Any]] = []
    regions = ("before", "at", "after")
    if feature == "posterior_transition_distance":
        regions = ("absolute",)
    for scope, seed, taxonomy, scope_values in scopes:
        for region in regions:
            if region == "before":
                region_mask = values < 0
            elif region == "at":
                region_mask = values == 0
            elif region == "after":
                region_mask = values > 0
            else:
                region_mask = np.ones(values.size, dtype=bool)
            for bin_label in _feature_bins(feature):
                mask = (
                    data.reference.test_mask
                    & region_mask
                    & _feature_bin_mask(feature, values, bin_label)
                )
                point = _taxonomy_metrics(
                    taxonomy,
                    scope_values,
                    mask,
                )
                rows.append(
                    {
                        "profile": (
                            "transition"
                            if feature == "posterior_transition_distance"
                            else (
                                "onset"
                                if feature == "onset_distance"
                                else "offset"
                            )
                        ),
                        "scope": scope,
                        "seed": seed,
                        "region": region,
                        "distance_bin": bin_label,
                        "frames": point["frames"],
                        "mean_value": point["mean_value"],
                        "p_R": point["p_R"],
                        "p_I": point["p_I"],
                        "p_H": point["p_H"],
                        "availability_net": point["availability_net"],
                    }
                )
    return rows


def run_car3(
    data: CarAnalysisData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute event-bin replication and centered value profiles."""
    rows = _event_structure_rows(data)
    comparisons = _event_comparison_rows(rows)
    _write_csv(output_dir / "car3_event_structure.csv", rows)
    _write_csv(
        output_dir / "car3_event_comparison.csv",
        comparisons,
    )
    profile_dir = output_dir / "car3_event_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        "transition": _profile_rows(
            data,
            feature="posterior_transition_distance",
        ),
        "onset": _profile_rows(data, feature="onset_distance"),
        "offset": _profile_rows(data, feature="offset_distance"),
    }
    for name, profile_rows in profiles.items():
        _write_csv(
            profile_dir / f"{name}_value_profile.csv",
            profile_rows,
        )
    return {
        "rows": rows,
        "comparisons": comparisons,
        "profiles": profiles,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _condition_group_type(group: str) -> str:
    if group in {"all", "clean", "seen", "unseen"}:
        return group
    if group.startswith("noise/"):
        return "noise"
    if group in CONDITION_CELLS:
        return "seen_unseen_snr"
    return "other"


def _condition_group_parts(
    group: str,
) -> tuple[str | None, str | None]:
    if "/" not in group:
        return None, None
    domain, snr = group.split("/", 1)
    if domain not in {"seen", "unseen"}:
        return None, None
    return domain, snr


def _marble_taxonomy_row(
    data: CarAnalysisData,
    *,
    group: str,
) -> dict[str, Any] | None:
    """Aggregate frozen AE3 posterior-transition rows by frame count."""
    rows = _read_csv(AE3_FEATURE_BINS_PATH)
    selected = [
        row
        for row in rows
        if row["feature"] == "posterior_transition_distance"
        and (
            row["group"] == group
            or (
                group in {"seen", "unseen"}
                and row["group"].startswith(f"{group}/")
            )
        )
    ]
    if not selected:
        return None
    valid_rows: list[tuple[Mapping[str, str], float]] = []
    for row in selected:
        frames = _float_or_none(row.get("frames"))
        if frames is None or frames <= 0.0:
            continue
        valid_rows.append((row, frames))
    total_frames = sum(frames for _, frames in valid_rows)
    if total_frames <= 0:
        return None

    def weighted(name: str) -> float | None:
        weighted_sum = 0.0
        weight_total = 0.0
        for row, frames in valid_rows:
            value = _float_or_none(row.get(name))
            if value is None:
                continue
            weighted_sum += frames * value
            weight_total += frames
        return (
            weighted_sum / weight_total
            if weight_total > 0.0
            else None
        )

    p_r = weighted("refinable_rate")
    p_i = weighted("irreducible_rate")
    p_h = weighted("refinement_harm_rate")
    if p_r is None or p_i is None or p_h is None:
        return None
    mask = _mask_for_group(data.reference, group)
    return {
        "architecture": "MarbleNet",
        "scope": "reference",
        "seed": "reference",
        "group": group,
        "group_type": _condition_group_type(group),
        "frames": int(np.count_nonzero(mask)),
        "p_SS": float(1.0 - p_r - p_i - p_h),
        "p_R": float(p_r),
        "p_I": float(p_i),
        "p_H": float(p_h),
        "availability_net": float(p_r - p_h),
        "mean_value": _point_mean(data.marble_value, mask),
    }


def _car4_condition_dependence_rows(
    data: CarAnalysisData,
) -> list[dict[str, Any]]:
    """Build Tiny-GRU and frozen-MarbleNet condition rows."""
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    aggregate_values = _aggregate_values(data.values)
    scopes: list[
        tuple[str, int | str, Mapping[str, np.ndarray], np.ndarray]
    ] = [
        ("seed", seed, data.taxonomy[seed], data.values[seed])
        for seed in MODEL_SEEDS
    ]
    scopes.append(
        ("aggregate", "aggregate", aggregate_taxonomy, aggregate_values)
    )
    rows: list[dict[str, Any]] = []
    for scope, seed, taxonomy, values in scopes:
        for group in _condition_groups(data.reference):
            row, _ = _taxonomy_output_row(
                data,
                scope=scope,
                seed=seed,
                group=group,
                taxonomy=taxonomy,
                values=values,
            )
            row["architecture"] = "TinyGRU"
            row["group_type"] = _condition_group_type(group)
            rows.append(row)

    for group in (
        "all",
        "clean",
        "seen",
        "unseen",
        *CONDITION_CELLS,
    ):
        marble = _marble_taxonomy_row(data, group=group)
        if marble is not None:
            rows.append(marble)
    return rows


def _marble_horizon_lookup() -> dict[str, dict[str, str]]:
    return {
        row["group"]: row
        for row in _read_csv(AE2_HORIZON_SUMMARY_PATH)
        if str(row.get("group", "")).strip()
    }


def _marble_horizon_metrics(
    row: Mapping[str, str] | None,
) -> dict[str, float | None]:
    if row is None:
        return {
            "marble_non_monotone_rate": None,
            "marble_stable_sufficient_long_rate": None,
            "marble_mean_stable_sufficient_span": None,
            "marble_oracle_net_utility_rf64": None,
            "marble_oracle_net_utility_rf128": None,
            "marble_oracle_net_utility_rf256": None,
            "marble_oracle_net_utility_rf384": None,
            "marble_oracle_net_utility_rf512": None,
            "marble_long_history_share": None,
        }
    utility_384 = _float_or_none(row.get("oracle_net_utility_rf384"))
    utility_512 = _float_or_none(row.get("oracle_net_utility_rf512"))
    long_share = (
        utility_384 / utility_512
        if (
            utility_384 is not None
            and utility_512 is not None
            and utility_512 > VALUE_SPARSITY_EPSILON
        )
        else None
    )
    return {
        "marble_non_monotone_rate": _float_or_none(
            row.get("non_monotone_rate")
        ),
        "marble_stable_sufficient_long_rate": _float_or_none(
            row.get("stable_sufficient_long_rate")
        ),
        "marble_mean_stable_sufficient_span": _float_or_none(
            row.get("mean_stable_sufficient_span")
        ),
        "marble_oracle_net_utility_rf64": _float_or_none(
            row.get("oracle_net_utility_rf64")
        ),
        "marble_oracle_net_utility_rf128": _float_or_none(
            row.get("oracle_net_utility_rf128")
        ),
        "marble_oracle_net_utility_rf256": _float_or_none(
            row.get("oracle_net_utility_rf256")
        ),
        "marble_oracle_net_utility_rf384": utility_384,
        "marble_oracle_net_utility_rf512": utility_512,
        "marble_long_history_share": long_share,
    }


def _aggregate_horizon_arrays(
    data: CarAnalysisData,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    losses = {
        key: _aggregate_values(
            {
                seed: data.losses[seed][key]
                for seed in MODEL_SEEDS
            }
        )
        for key in HORIZON_LABELS.values()
    }
    correctness = (
        np.mean(
            np.stack(
                [
                    _horizon_correctness(
                        data.scores[seed],
                        data.reference.labels,
                    ).astype(np.float64)
                    for seed in MODEL_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
        >= DECISION_THRESHOLD
    )
    return losses, correctness


def _safe_spearman(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.size < 3
        or np.unique(left_array).size <= 1
        or np.unique(right_array).size <= 1
    ):
        return None
    value = spearmanr(left_array, right_array).statistic
    return float(value) if math.isfinite(float(value)) else None


def _mean_for_cells(
    cell_rows: Mapping[str, Mapping[str, Any]],
    *,
    domain: str,
    key: str,
) -> float | None:
    values = [
        _float_or_none(row.get(key))
        for group, row in cell_rows.items()
        if group.startswith(f"{domain}/")
    ]
    return _mean_or_none(values)


def _trend_for_cells(
    cell_rows: Mapping[str, Mapping[str, Any]],
    *,
    domain: str,
    key: str,
) -> float | None:
    pairs = [
        (float(snr), _float_or_none(row.get(key)))
        for group, row in cell_rows.items()
        if group.startswith(f"{domain}/")
        for snr in [group.split("/", 1)[1]]
    ]
    pairs = [
        (snr, value)
        for snr, value in pairs
        if value is not None
    ]
    if len(pairs) < 3:
        return None
    return _safe_spearman(
        [item[0] for item in pairs],
        [float(item[1]) for item in pairs],
    )


def _car4_cross_architecture(
    data: CarAnalysisData,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    aggregate_values = _aggregate_values(data.values)
    aggregate_losses, aggregate_correctness = _aggregate_horizon_arrays(data)
    horizon_lookup = _marble_horizon_lookup()
    cell_rows: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    for group in CONDITION_CELLS:
        mask = _mask_for_group(data.reference, group)
        tiny_point, _ = _taxonomy_output_row(
            data,
            scope="aggregate",
            seed="aggregate",
            group=group,
            taxonomy=aggregate_taxonomy,
            values=aggregate_values,
        )
        tiny_horizon, _ = _nonmonotonicity_row(
            data,
            scope="aggregate",
            seed="aggregate",
            group=group,
            values=aggregate_losses,
            correctness=aggregate_correctness,
        )
        marble_taxonomy = _marble_taxonomy_row(data, group=group)
        marble_horizon = _marble_horizon_metrics(
            horizon_lookup.get(group)
        )
        domain, snr = _condition_group_parts(group)
        row: dict[str, Any] = {
            "row_type": "cell",
            "group": group,
            "domain": domain,
            "snr": snr,
            "frames": int(np.count_nonzero(mask)),
            "tiny_frames": int(np.count_nonzero(mask)),
            "marble_frames": (
                marble_taxonomy.get("frames")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_mean_v": tiny_point.get("mean_value"),
            "marble_mean_v": (
                marble_taxonomy.get("mean_value")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_availability_net": tiny_point.get(
                "availability_net"
            ),
            "marble_availability_net": (
                marble_taxonomy.get("availability_net")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_p_R": tiny_point.get("p_R"),
            "marble_p_R": (
                marble_taxonomy.get("p_R")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_p_I": tiny_point.get("p_I"),
            "marble_p_I": (
                marble_taxonomy.get("p_I")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_p_H": tiny_point.get("p_H"),
            "marble_p_H": (
                marble_taxonomy.get("p_H")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_p_SS": tiny_point.get("p_SS"),
            "marble_p_SS": (
                marble_taxonomy.get("p_SS")
                if marble_taxonomy is not None
                else None
            ),
            "tiny_non_monotone_rate": tiny_horizon.get(
                "non_monotone_rate"
            ),
            "tiny_long_history_share": tiny_horizon.get(
                "long_history_share"
            ),
            **marble_horizon,
            "tiny_source_ci95_low_availability_net": tiny_point.get(
                "source_ci95_low_availability_net"
            ),
            "tiny_source_ci95_high_availability_net": tiny_point.get(
                "source_ci95_high_availability_net"
            ),
            "tiny_source_ci95_low_mean_value": tiny_point.get(
                "source_ci95_low_mean_value"
            ),
            "tiny_source_ci95_high_mean_value": tiny_point.get(
                "source_ci95_high_mean_value"
            ),
        }
        cell_rows[group] = row
        output_rows.append(row)

    tiny_values = [
        _float_or_none(cell_rows[group].get("tiny_availability_net"))
        for group in CONDITION_CELLS
    ]
    marble_values = [
        _float_or_none(cell_rows[group].get("marble_availability_net"))
        for group in CONDITION_CELLS
    ]
    valid_pairs = [
        (float(tiny), float(marble))
        for tiny, marble in zip(tiny_values, marble_values)
        if tiny is not None and marble is not None
    ]
    sign_agreement = (
        float(
            np.mean(
                [
                    np.sign(tiny) == np.sign(marble)
                    for tiny, marble in valid_pairs
                ]
            )
        )
        if valid_pairs
        else None
    )
    rank_correlation = _safe_spearman(
        [item[0] for item in valid_pairs],
        [item[1] for item in valid_pairs],
    )
    tiny_seen = _mean_for_cells(
        cell_rows,
        domain="seen",
        key="tiny_availability_net",
    )
    tiny_unseen = _mean_for_cells(
        cell_rows,
        domain="unseen",
        key="tiny_availability_net",
    )
    marble_seen = _mean_for_cells(
        cell_rows,
        domain="seen",
        key="marble_availability_net",
    )
    marble_unseen = _mean_for_cells(
        cell_rows,
        domain="unseen",
        key="marble_availability_net",
    )
    tiny_direction = (
        None
        if tiny_seen is None or tiny_unseen is None
        else float(tiny_unseen - tiny_seen)
    )
    marble_direction = (
        None
        if marble_seen is None or marble_unseen is None
        else float(marble_unseen - marble_seen)
    )
    direction_agreement = (
        None
        if tiny_direction is None or marble_direction is None
        else bool(np.sign(tiny_direction) == np.sign(marble_direction))
    )
    tiny_trend_seen = _trend_for_cells(
        cell_rows,
        domain="seen",
        key="tiny_availability_net",
    )
    tiny_trend_unseen = _trend_for_cells(
        cell_rows,
        domain="unseen",
        key="tiny_availability_net",
    )
    marble_trend_seen = _trend_for_cells(
        cell_rows,
        domain="seen",
        key="marble_availability_net",
    )
    marble_trend_unseen = _trend_for_cells(
        cell_rows,
        domain="unseen",
        key="marble_availability_net",
    )
    trend_agreement = (
        None
        if (
            tiny_trend_seen is None
            or tiny_trend_unseen is None
            or marble_trend_seen is None
            or marble_trend_unseen is None
        )
        else bool(
            np.sign(tiny_trend_seen) == np.sign(marble_trend_seen)
            and np.sign(tiny_trend_unseen) == np.sign(marble_trend_unseen)
        )
    )
    tiny_range = (
        None
        if not any(value is not None for value in tiny_values)
        else float(
            max(value for value in tiny_values if value is not None)
            - min(value for value in tiny_values if value is not None)
        )
    )
    marble_range = (
        None
        if not any(value is not None for value in marble_values)
        else float(
            max(value for value in marble_values if value is not None)
            - min(value for value in marble_values if value is not None)
        )
    )
    summary = {
        "row_type": "summary",
        "group": "condition_cells",
        "cells_compared": len(valid_pairs),
        "cell_sign_agreement": sign_agreement,
        "cell_spearman_rank_correlation": rank_correlation,
        "seen_unseen_direction_agreement": direction_agreement,
        "snr_trend_agreement": trend_agreement,
        "tiny_seen_mean_availability_net": tiny_seen,
        "tiny_unseen_mean_availability_net": tiny_unseen,
        "marble_seen_mean_availability_net": marble_seen,
        "marble_unseen_mean_availability_net": marble_unseen,
        "tiny_seen_minus_unseen": tiny_direction,
        "marble_seen_minus_unseen": marble_direction,
        "tiny_snr_trend_seen": tiny_trend_seen,
        "tiny_snr_trend_unseen": tiny_trend_unseen,
        "marble_snr_trend_seen": marble_trend_seen,
        "marble_snr_trend_unseen": marble_trend_unseen,
        "tiny_cell_range": tiny_range,
        "marble_cell_range": marble_range,
    }
    output_rows.append(summary)
    return output_rows, summary


def _car4_gate(
    data: CarAnalysisData,
) -> dict[str, Any]:
    aggregate_values = _aggregate_values(data.values)
    masks = [
        _mask_for_group(data.reference, group)
        for group in CONDITION_CELLS
    ]
    bootstrap = _bootstrap_range(
        data=data,
        group="car4_condition_cells",
        masks=masks,
        values=aggregate_values,
    )
    point_range = bootstrap.get("range")
    lower = bootstrap.get("ci95_low")
    passed = bool(
        point_range is not None
        and point_range >= R4_RANGE_THRESHOLD
        and lower is not None
        and lower > 0.0
    )
    return {
        "statistic": "max_cell_mean_v_minus_min_cell_mean_v",
        "cells": list(CONDITION_CELLS),
        "point_range": point_range,
        "source_ci95_low": lower,
        "source_ci95_high": bootstrap.get("ci95_high"),
        "threshold": R4_RANGE_THRESHOLD,
        "pass": passed,
    }


def run_car4(
    data: CarAnalysisData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute condition dependence and cross-architecture diagnostics."""
    condition_rows = _car4_condition_dependence_rows(data)
    cross_rows, summary = _car4_cross_architecture(data)
    gate = _car4_gate(data)
    _write_csv(
        output_dir / "car4_condition_dependence.csv",
        condition_rows,
    )
    _write_csv(
        output_dir / "car4_cross_architecture.csv",
        cross_rows,
    )
    return {
        "condition_dependence": condition_rows,
        "cross_architecture": cross_rows,
        "cross_architecture_summary": summary,
        "gate": gate,
    }


def _tiny_r3_contrast_rows(
    data: CarAnalysisData,
    *,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregate_taxonomy = _aggregate_taxonomy(data.taxonomy)
    availability = (
        np.asarray(aggregate_taxonomy["R"], dtype=np.float64)
        - np.asarray(aggregate_taxonomy["H"], dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    for name, definition in protocol["r3_contrasts"].items():
        feature = str(definition["feature"])
        values = np.asarray(
            data.reference.event_features[feature],
            dtype=np.float64,
        )
        high_mask = _mask_for_group(data.reference, "all")
        reference_mask = _mask_for_group(data.reference, "all")
        if bool(definition.get("selected_only", False)):
            high_mask = high_mask & data.reference.selected
            reference_mask = reference_mask & data.reference.selected
        high_bins = tuple(str(item) for item in definition["high_bins"])
        reference_bins = tuple(
            str(item) for item in definition["reference_bins"]
        )
        high_mask = high_mask & np.logical_or.reduce(
            [
                _feature_bin_mask(feature, values, bin_label)
                for bin_label in high_bins
            ]
        )
        reference_mask = reference_mask & np.logical_or.reduce(
            [
                _feature_bin_mask(feature, values, bin_label)
                for bin_label in reference_bins
            ]
        )
        point_high = _point_mean(availability, high_mask)
        point_reference = _point_mean(availability, reference_mask)
        point_contrast = (
            None
            if point_high is None or point_reference is None
            else float(point_high - point_reference)
        )
        bootstrap = _bootstrap_difference(
            data=data,
            group=f"r3/{name}",
            focal_mask=high_mask,
            reference_mask=reference_mask,
            values=availability,
        )
        marble_contrast = _float_or_none(
            definition.get("marble_net_contrast")
        )
        lower = bootstrap.get("ci95_low")
        passed = bool(
            point_contrast is not None
            and point_contrast > 0.0
            and marble_contrast is not None
            and marble_contrast > 0.0
            and lower is not None
            and lower > 0.0
        )
        rows.append(
            {
                "structure": name,
                "feature": feature,
                "high_bins": "|".join(high_bins),
                "reference_bins": "|".join(reference_bins),
                "selected_only": bool(
                    definition.get("selected_only", False)
                ),
                "tiny_high_frames": int(np.count_nonzero(high_mask)),
                "tiny_reference_frames": int(
                    np.count_nonzero(reference_mask)
                ),
                "tiny_high_availability": point_high,
                "tiny_reference_availability": point_reference,
                "tiny_contrast": point_contrast,
                "tiny_source_ci95_low": bootstrap.get("ci95_low"),
                "tiny_source_ci95_high": bootstrap.get("ci95_high"),
                "marble_high_availability": _float_or_none(
                    definition.get("marble_net_high_availability")
                ),
                "marble_reference_availability": _float_or_none(
                    definition.get("marble_net_reference_availability")
                ),
                "marble_contrast": marble_contrast,
                "direction_agreement": (
                    None
                    if point_contrast is None or marble_contrast is None
                    else bool(
                        np.sign(point_contrast) == np.sign(marble_contrast)
                    )
                ),
                "pass": passed,
            }
        )
    return rows


def run_car3_gate_support(
    data: CarAnalysisData,
    *,
    protocol: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write the frozen transition/persistence/onset gate support table."""
    rows = _tiny_r3_contrast_rows(data, protocol=protocol)
    _write_csv(output_dir / "car3_gate_support.csv", rows)
    passing = [row["structure"] for row in rows if row["pass"]]
    return {
        "rows": rows,
        "passing_structures": passing,
        "passed_count": len(passing),
        "pass": len(passing) >= R3_MIN_PASSING_STRUCTURES,
    }


def _taxonomy_codes(taxonomy: Mapping[str, np.ndarray]) -> np.ndarray:
    categories = ("SS", "R", "I", "H")
    codes = np.full(
        np.asarray(taxonomy["SS"]).shape,
        len(categories),
        dtype=np.int8,
    )
    for index, category in enumerate(categories):
        codes[np.asarray(taxonomy[category], dtype=bool)] = index
    if np.any(codes == len(categories)):
        raise ValueError("taxonomy categories are not exhaustive")
    return codes


def _car5_rows(data: CarAnalysisData) -> list[dict[str, Any]]:
    mask = np.asarray(data.reference.test_mask, dtype=bool)
    sign_values = {
        seed: np.sign(np.asarray(data.values[seed], dtype=np.float64))
        for seed in MODEL_SEEDS
    }
    taxonomy_codes = {
        seed: _taxonomy_codes(data.taxonomy[seed])
        for seed in MODEL_SEEDS
    }
    refinable = {
        seed: np.asarray(data.taxonomy[seed]["R"], dtype=bool)
        for seed in MODEL_SEEDS
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_SEEDS, 2):
        left_sign = sign_values[left][mask]
        right_sign = sign_values[right][mask]
        left_r = refinable[left][mask]
        right_r = refinable[right][mask]
        union = np.count_nonzero(left_r | right_r)
        intersection = np.count_nonzero(left_r & right_r)
        rows.append(
            {
                "row_type": "pair",
                "left_seed": left,
                "right_seed": right,
                "frames": int(np.count_nonzero(mask)),
                "sign_agreement": float(
                    np.mean(left_sign == right_sign)
                ),
                "r_jaccard": (
                    float(intersection / union)
                    if union > 0
                    else None
                ),
                "taxonomy_agreement": float(
                    np.mean(
                        taxonomy_codes[left][mask]
                        == taxonomy_codes[right][mask]
                    )
                ),
                "r_intersection_frames": int(intersection),
                "r_union_frames": int(union),
            }
        )

    positive_counts = {
        seed: int(
            np.count_nonzero(
                (np.asarray(data.values[seed], dtype=np.float64) > 0.0)
                & mask
            )
        )
        for seed in MODEL_SEEDS
    }
    positive_total = int(sum(positive_counts.values()))
    all_values = _aggregate_values(data.values)
    for seed in MODEL_SEEDS:
        taxonomy = data.taxonomy[seed]
        values = np.asarray(data.values[seed], dtype=np.float64)
        point = _taxonomy_metrics(taxonomy, values, mask)
        loso_values = np.mean(
            np.stack(
                [
                    np.asarray(data.values[other], dtype=np.float64)
                    for other in MODEL_SEEDS
                    if other != seed
                ],
                axis=0,
            ),
            axis=0,
        )
        rows.append(
            {
                "row_type": "seed",
                "seed": seed,
                "frames": int(np.count_nonzero(mask)),
                "availability_net": point["availability_net"],
                "mean_v_seed": point["mean_value"],
                "mean_v_full": _point_mean(all_values, mask),
                "mean_v_loso": _point_mean(loso_values, mask),
                "positive_frame_count": positive_counts[seed],
                "positive_frame_share": (
                    float(positive_counts[seed] / positive_total)
                    if positive_total > 0
                    else None
                ),
                "p_R": point["p_R"],
                "p_I": point["p_I"],
                "p_H": point["p_H"],
            }
        )
    return rows


def run_car5(
    data: CarAnalysisData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute pairwise within-GRU reproducibility diagnostics."""
    rows = _car5_rows(data)
    _write_csv(output_dir / "car5_seed_agreement.csv", rows)
    pairs = [row for row in rows if row["row_type"] == "pair"]
    seeds = [row for row in rows if row["row_type"] == "seed"]
    return {
        "rows": rows,
        "pairs": pairs,
        "seeds": seeds,
    }


def _car5_gate(
    car5: Mapping[str, Any],
) -> dict[str, Any]:
    seed_rows = list(car5["seeds"])
    full_values = [
        _float_or_none(row.get("mean_v_full"))
        for row in seed_rows
    ]
    loso_values = [
        _float_or_none(row.get("mean_v_loso"))
        for row in seed_rows
    ]
    shares = [
        _float_or_none(row.get("positive_frame_share"))
        for row in seed_rows
    ]
    full_positive = bool(
        full_values
        and all(value is not None and value > 0.0 for value in full_values)
    )
    loso_positive = bool(
        loso_values
        and all(value is not None and value > 0.0 for value in loso_values)
    )
    share_pass = bool(
        shares
        and all(
            value is not None
            and value <= R5_MAX_SEED_POSITIVE_SHARE
            for value in shares
        )
    )
    return {
        "full_mean_v": _mean_or_none(full_values),
        "loso_mean_v": {
            row["seed"]: value
            for row, value in zip(seed_rows, loso_values)
        },
        "positive_frame_shares": {
            row["seed"]: value
            for row, value in zip(seed_rows, shares)
        },
        "maximum_positive_share": (
            max(value for value in shares if value is not None)
            if any(value is not None for value in shares)
            else None
        ),
        "full_mean_v_positive": full_positive,
        "all_loso_mean_v_positive": loso_positive,
        "no_seed_exceeds_positive_share": share_pass,
        "pass": bool(full_positive and loso_positive and share_pass),
    }


def _car_gate_results(
    *,
    car1: Mapping[str, Any],
    car2: Mapping[str, Any],
    car3_gate: Mapping[str, Any],
    car4: Mapping[str, Any],
    car5: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = car1["aggregate"]
    seed_rows = [
        row
        for row in car1["by_seed"]
        if row["group"] == "all"
    ]
    r1 = bool(
        len(seed_rows) == len(MODEL_SEEDS)
        and all(
            _float_or_none(row.get("availability_net")) is not None
            and float(row["availability_net"]) > 0.0
            for row in seed_rows
        )
    )
    p_r_lower = _float_or_none(
        aggregate.get("source_ci95_low_p_R")
    )
    p_r = _float_or_none(aggregate.get("p_R"))
    p_i = _float_or_none(aggregate.get("p_I"))
    r2 = bool(
        p_r is not None
        and p_r > 0.0
        and p_r_lower is not None
        and p_r_lower > 0.0
        and p_i is not None
        and p_i > p_r
    )
    r3 = bool(car3_gate["pass"])
    r4 = bool(car4["gate"]["pass"])
    car5_gate = _car5_gate(car5)
    r5 = bool(car5_gate["pass"])
    gates = {
        "R1": {
            "pass": r1,
            "rule": "all five seed aggregate NetRefinability > 0",
            "seed_net_refinability": {
                int(row["seed"]): row.get("availability_net")
                for row in seed_rows
            },
        },
        "R2": {
            "pass": r2,
            "rule": "aggregate P(R)>0 with source lower bound >0 and P(I)>P(R)",
            "p_R": p_r,
            "p_R_source_ci95_low": p_r_lower,
            "p_I": p_i,
        },
        "R3": {
            "pass": r3,
            "rule": "at least two transition/persistence/onset structures pass",
            "passing_structures": car3_gate["passing_structures"],
            "passed_count": car3_gate["passed_count"],
        },
        "R4": {
            "pass": r4,
            "rule": "12-cell mean-v range meets point and source-support rules",
            **car4["gate"],
        },
        "R5": {
            "pass": r5,
            "rule": "full and LOSO means positive and no seed dominates",
            **car5_gate,
        },
    }
    passed_count = sum(int(item["pass"]) for item in gates.values())
    if passed_count == 5:
        status = "STRONG_CROSS_ARCH_REPLICATION"
    elif passed_count == 4:
        status = "CROSS_ARCH_CONDITIONAL"
    else:
        status = "CROSS_ARCH_NO_GO"
    return {
        "gates": gates,
        "passed_count": passed_count,
        "status": status,
        "car5_gate": car5_gate,
    }


def _pair_classification(
    left: bool | None,
    right: bool | None,
) -> str:
    if left is None or right is None:
        return "INCOMPARABLE"
    if left and right:
        return "REPLICATED"
    return "NOT_REPLICATED"


def _build_cross_architecture_summary(
    *,
    protocol: Mapping[str, Any],
    data: CarAnalysisData,
    car1: Mapping[str, Any],
    car2: Mapping[str, Any],
    car3_gate: Mapping[str, Any],
    car4: Mapping[str, Any],
    car5: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregate = car1["aggregate"]
    tiny_p_r = _float_or_none(aggregate.get("p_R"))
    tiny_p_i = _float_or_none(aggregate.get("p_I"))
    tiny_p_h = _float_or_none(aggregate.get("p_H"))
    tiny_p_ss = _float_or_none(aggregate.get("p_SS"))
    tiny_net = _float_or_none(aggregate.get("availability_net"))
    tiny_net_lower = _float_or_none(
        aggregate.get("source_ci95_low_availability_net")
    )

    ae1_rows = _read_csv(AE1_CONDITION_AVAILABILITY_PATH)
    ae1_all = next(
        (row for row in ae1_rows if row["group"] == "all"),
        None,
    )
    marble_p_r = _float_or_none(
        ae1_all.get("state_rate_refinable") if ae1_all else None
    )
    marble_p_i = _float_or_none(
        ae1_all.get("state_rate_rf384_irreducible")
        if ae1_all
        else None
    )
    marble_p_h = _float_or_none(
        ae1_all.get("state_rate_refinement_harm")
        if ae1_all
        else None
    )
    marble_p_ss = (
        None
        if (
            marble_p_r is None
            or marble_p_i is None
            or marble_p_h is None
        )
        else float(1.0 - marble_p_r - marble_p_i - marble_p_h)
    )
    marble_net = _float_or_none(
        ae1_all.get("availability_net") if ae1_all else None
    )
    marble_net_lower = _float_or_none(
        ae1_all.get("source_ci95_low_availability_net")
        if ae1_all
        else None
    )
    tiny_ratio = (
        None
        if tiny_p_r is None or tiny_p_ss is None or tiny_p_ss <= 0.0
        else float(tiny_p_r / tiny_p_ss)
    )
    marble_ratio = (
        None
        if marble_p_r is None
        or marble_p_ss is None
        or marble_p_ss <= 0.0
        else float(marble_p_r / marble_p_ss)
    )
    tiny_sparse = (
        None
        if tiny_ratio is None
        else bool(tiny_ratio <= SPARSE_PREVALENCE_RATIO_THRESHOLD)
    )
    marble_sparse = (
        None
        if marble_ratio is None
        else bool(marble_ratio <= SPARSE_PREVALENCE_RATIO_THRESHOLD)
    )
    tiny_irreducible_dominant = (
        None
        if tiny_p_i is None or tiny_p_r is None
        else bool(tiny_p_i > tiny_p_r)
    )
    marble_irreducible_dominant = (
        None
        if marble_p_i is None or marble_p_r is None
        else bool(marble_p_i > marble_p_r)
    )
    tiny_harm_lower = (
        None
        if tiny_p_h is None or tiny_p_r is None
        else bool(tiny_p_h < tiny_p_r)
    )
    marble_harm_lower = (
        None
        if marble_p_h is None or marble_p_r is None
        else bool(marble_p_h < marble_p_r)
    )
    tiny_net_supported = bool(
        tiny_net is not None
        and tiny_net > 0.0
        and tiny_net_lower is not None
        and tiny_net_lower > 0.0
    )
    marble_net_supported = bool(
        marble_net is not None
        and marble_net > 0.0
        and marble_net_lower is not None
        and marble_net_lower > 0.0
    )
    tiny_net_positive = bool(tiny_net is not None and tiny_net > 0.0)
    marble_net_positive = bool(
        marble_net is not None and marble_net > 0.0
    )
    tiny_meaningful = float(
        np.mean(
            np.abs(_aggregate_values(data.values))
            > VALUE_SPARSITY_EPSILON
        )
    )
    marble_meaningful = float(
        np.mean(
            np.abs(np.asarray(data.marble_value, dtype=np.float64))
            > VALUE_SPARSITY_EPSILON
        )
    )
    tiny_sparse_value = bool(
        tiny_meaningful <= MEANINGFUL_VALUE_RATE_THRESHOLD
    )
    marble_sparse_value = bool(
        marble_meaningful <= MEANINGFUL_VALUE_RATE_THRESHOLD
    )

    r3_rows = {
        row["structure"]: row for row in car3_gate["rows"]
    }
    tiny_range = _float_or_none(
        car4["cross_architecture_summary"].get("tiny_cell_range")
    )
    marble_range = _float_or_none(
        car4["cross_architecture_summary"].get("marble_cell_range")
    )
    tiny_heterogeneous = bool(
        tiny_range is not None and tiny_range >= R4_RANGE_THRESHOLD
    )
    marble_heterogeneous = bool(
        marble_range is not None
        and marble_range >= R4_RANGE_THRESHOLD
    )
    tiny_long_share = _float_or_none(
        car2["aggregate"].get("long_history_share")
    )
    horizon_rows = _read_csv(AE2_HORIZON_SUMMARY_PATH)
    ae2_all = next(
        (row for row in horizon_rows if row["group"] == "all"),
        None,
    )
    marble_horizon = _marble_horizon_metrics(ae2_all)
    marble_long_share = _float_or_none(
        marble_horizon.get("marble_long_history_share")
    )
    tiny_non_monotone = _float_or_none(
        car2["aggregate"].get("non_monotone_rate")
    )
    marble_non_monotone = _float_or_none(
        marble_horizon.get("marble_non_monotone_rate")
    )
    tiny_long_not_dominant = (
        None
        if tiny_long_share is None
        else bool(tiny_long_share >= LONG_HISTORY_SHARE_THRESHOLD)
    )
    marble_long_not_dominant = (
        None
        if marble_long_share is None
        else bool(marble_long_share >= LONG_HISTORY_SHARE_THRESHOLD)
    )
    tiny_non_monotone_meaningful = (
        None
        if tiny_non_monotone is None
        else bool(tiny_non_monotone >= NON_MONOTONE_MIN_RATE)
    )
    marble_non_monotone_meaningful = (
        None
        if marble_non_monotone is None
        else bool(marble_non_monotone >= NON_MONOTONE_MIN_RATE)
    )

    def r3_classification(structure: str) -> tuple[str, Any, Any]:
        row = r3_rows.get(structure, {})
        tiny_value = _float_or_none(row.get("tiny_contrast"))
        marble_value = _float_or_none(row.get("marble_contrast"))
        if row.get("pass"):
            return "REPLICATED", tiny_value, marble_value
        if (
            tiny_value is not None
            and marble_value is not None
            and tiny_value > 0.0
            and marble_value > 0.0
        ):
            return "PARTIALLY_REPLICATED", tiny_value, marble_value
        if tiny_value is None or marble_value is None:
            return "INCOMPARABLE", tiny_value, marble_value
        return "NOT_REPLICATED", tiny_value, marble_value

    transition_class, transition_tiny, transition_marble = (
        r3_classification("transition")
    )
    persistence_class, persistence_tiny, persistence_marble = (
        r3_classification("persistence")
    )
    rows = [
        {
            "item": "A. Refinable prevalence",
            "metric": "P(R)/P(SS)",
            "classification": _pair_classification(
                tiny_sparse,
                marble_sparse,
            ),
            "tiny_value": tiny_ratio,
            "marble_value": marble_ratio,
            "tiny_support": (
                f"P(R)={tiny_p_r}; P(SS)={tiny_p_ss}"
            ),
            "marble_support": (
                f"P(R)={marble_p_r}; P(SS)={marble_p_ss}"
            ),
            "rule": (
                "both architectures classified sparse when the ratio "
                f"is <= {SPARSE_PREVALENCE_RATIO_THRESHOLD}"
            ),
        },
        {
            "item": "B. Irreducible prevalence",
            "metric": "P(I)>P(R)",
            "classification": _pair_classification(
                tiny_irreducible_dominant,
                marble_irreducible_dominant,
            ),
            "tiny_value": tiny_p_i,
            "marble_value": marble_p_i,
            "tiny_support": (
                f"P(I)={tiny_p_i}; P(R)={tiny_p_r}"
            ),
            "marble_support": (
                f"P(I)={marble_p_i}; P(R)={marble_p_r}"
            ),
            "rule": "both architectures show P(I)>P(R)",
        },
        {
            "item": "C. Harm prevalence",
            "metric": "P(H)<P(R)",
            "classification": _pair_classification(
                tiny_harm_lower,
                marble_harm_lower,
            ),
            "tiny_value": tiny_p_h,
            "marble_value": marble_p_h,
            "tiny_support": (
                f"P(H)={tiny_p_h}; P(R)={tiny_p_r}"
            ),
            "marble_support": (
                f"P(H)={marble_p_h}; P(R)={marble_p_r}"
            ),
            "rule": "both architectures show P(H)<P(R)",
        },
        {
            "item": "D. Net refinability",
            "metric": "P(R)-P(H)",
            "classification": (
                "REPLICATED"
                if tiny_net_supported and marble_net_supported
                else (
                    "PARTIALLY_REPLICATED"
                    if tiny_net_positive and marble_net_positive
                    else "NOT_REPLICATED"
                )
            ),
            "tiny_value": tiny_net,
            "marble_value": marble_net,
            "tiny_support": (
                f"source_ci95_low={tiny_net_lower}"
            ),
            "marble_support": (
                f"source_ci95_low={marble_net_lower}"
            ),
            "rule": (
                "both point values positive; full replication requires "
                "both source lower bounds positive"
            ),
        },
        {
            "item": "E. Value sparsity",
            "metric": "P(|v|>epsilon)",
            "classification": _pair_classification(
                tiny_sparse_value,
                marble_sparse_value,
            ),
            "tiny_value": tiny_meaningful,
            "marble_value": marble_meaningful,
            "tiny_support": (
                f"epsilon={VALUE_SPARSITY_EPSILON}; "
                f"threshold={MEANINGFUL_VALUE_RATE_THRESHOLD}"
            ),
            "marble_support": (
                f"epsilon={VALUE_SPARSITY_EPSILON}; "
                f"threshold={MEANINGFUL_VALUE_RATE_THRESHOLD}"
            ),
            "rule": (
                "both architectures classified sparse when meaningful "
                "value rate is below the frozen threshold"
            ),
        },
        {
            "item": "F. Transition enrichment",
            "metric": "transition-bin availability contrast",
            "classification": transition_class,
            "tiny_value": transition_tiny,
            "marble_value": transition_marble,
            "tiny_support": (
                "source lower bound > 0 required for replication"
            ),
            "marble_support": "frozen AE3 contrast",
            "rule": "positive contrast with source support in both",
        },
        {
            "item": "G. Uncertainty-persistence enrichment",
            "metric": "persistence-bin availability contrast",
            "classification": persistence_class,
            "tiny_value": persistence_tiny,
            "marble_value": persistence_marble,
            "tiny_support": (
                "selected frames only; source lower bound > 0 required"
            ),
            "marble_support": "frozen AE3 contrast",
            "rule": "positive contrast with source support in both",
        },
        {
            "item": "H. Condition heterogeneity",
            "metric": "12-cell availability range",
            "classification": _pair_classification(
                tiny_heterogeneous,
                marble_heterogeneous,
            ),
            "tiny_value": tiny_range,
            "marble_value": marble_range,
            "tiny_support": (
                f"point threshold={R4_RANGE_THRESHOLD}; "
                "source lower bound in R4"
            ),
            "marble_support": (
                f"point threshold={R4_RANGE_THRESHOLD}"
            ),
            "rule": (
                "both architectures show a condition-cell range above "
                "the frozen point threshold"
            ),
        },
        {
            "item": "I. Long-history demand",
            "metric": "H384 share of H64-to-full benefit",
            "classification": _pair_classification(
                tiny_long_not_dominant,
                marble_long_not_dominant,
            ),
            "tiny_value": tiny_long_share,
            "marble_value": marble_long_share,
            "tiny_support": "CAR2 aggregate long_history_share",
            "marble_support": (
                "AE2 RF384/RF512 oracle utility ratio; direction-only "
                "and not numerically interchangeable with Tiny-GRU"
            ),
            "rule": (
                "both architectures classify long history as not "
                f"dominant when share >= {LONG_HISTORY_SHARE_THRESHOLD}; "
                "raw shares use architecture-specific units"
            ),
        },
        {
            "item": "J. Non-monotonicity",
            "metric": "non-monotone prediction rate",
            "classification": _pair_classification(
                tiny_non_monotone_meaningful,
                marble_non_monotone_meaningful,
            ),
            "tiny_value": tiny_non_monotone,
            "marble_value": marble_non_monotone,
            "tiny_support": (
                f"minimum rate={NON_MONOTONE_MIN_RATE}"
            ),
            "marble_support": (
                f"minimum rate={NON_MONOTONE_MIN_RATE}"
            ),
            "rule": (
                "both architectures show non-monotone prediction rate "
                "above the frozen minimum"
            ),
        },
    ]
    return rows


def _stage_marker(output_dir: Path, stage: str) -> Path:
    return output_dir / f"{stage}.stage.json"


def _refuse_overwrite(paths: Sequence[Path], *, stage: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"{stage} is already complete; refusing to overwrite: {existing}"
        )


def _write_stage_marker(
    output_dir: Path,
    *,
    stage: str,
    artifacts: Sequence[Path],
) -> Path:
    marker = _stage_marker(output_dir, stage)
    _write_json(
        marker,
        {
            "stage": stage,
            "completed_at": _utc_now(),
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": file_sha256(path),
                    "bytes": int(path.stat().st_size),
                }
                for path in artifacts
            ],
        },
    )
    return marker


def _artifact_hash_rows(
    output_dir: Path,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        rows.append(
            {
                "path": str(resolved.relative_to(output_dir)),
                "sha256": file_sha256(resolved),
                "bytes": int(resolved.stat().st_size),
            }
        )
    return rows


def _files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _plot_car1_taxonomy(
    car1: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    seed_rows = [
        row for row in car1["by_seed"] if row.get("group") == "all"
    ]
    rows = [*seed_rows, car1["aggregate"]]
    labels = [str(int(row["seed"])) for row in seed_rows] + ["aggregate"]
    categories = ("SS", "R", "I", "H")
    matrix = np.asarray(
        [
            [
                float(row.get(f"p_{category}") or 0.0)
                for category in categories
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    colors = ("#4C78A8", "#F58518", "#E45756", "#72B7B2")
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(matrix.shape[0], dtype=np.float64)
    x = np.arange(matrix.shape[0])
    for index, category in enumerate(categories):
        axis.bar(
            x,
            matrix[:, index],
            bottom=bottom,
            label=category,
            color=colors[index],
        )
        bottom += matrix[:, index]
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Frame share")
    axis.set_title("CAR1 taxonomy by seed")
    axis.legend(ncol=4, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car2_horizon_demand(
    car2: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in car2["history"]
        if row.get("scope") == "aggregate" and row.get("group") == "all"
    ]
    rows.sort(key=lambda row: int(row["horizon_index"]))
    labels = [str(row["horizon"]) for row in rows]
    loss = np.asarray(
        [_float_or_none(row.get("loss")) for row in rows],
        dtype=np.float64,
    )
    error = np.asarray(
        [_float_or_none(row.get("frame_error")) for row in rows],
        dtype=np.float64,
    )
    value = np.asarray(
        [
            _float_or_none(row.get("signed_incremental_value"))
            for row in rows
        ],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    axes[0].plot(labels, loss, marker="o", label="BCE loss")
    axes[0].plot(labels, error, marker="s", label="Frame error")
    axes[0].set_title("Horizon demand")
    axes[0].set_ylabel("Mean")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="#555555", linewidth=1.0)
    axes[1].plot(labels, value, marker="o", color="#54A24B")
    axes[1].set_title("Signed incremental value")
    axes[1].set_ylabel("Loss(H64) - loss(horizon)")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car2_nonmonotonicity(
    car2: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in car2["non_monotonicity"]
        if row.get("scope") == "seed" and row.get("group") == "all"
    ]
    labels = [str(int(row["seed"])) for row in rows]
    values = [
        float(row.get("non_monotone_rate") or 0.0) for row in rows
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.bar(labels, values, color="#B279A2")
    axis.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.15))
    axis.set_ylabel("Non-monotone rate")
    axis.set_title("CAR2 non-monotonicity by seed")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car3_event_profiles(
    car3: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    panel_specs = (
        ("transition", "absolute"),
        ("onset", "at"),
        ("offset", "at"),
    )
    for axis, (name, region) in zip(axes, panel_specs):
        rows = [
            row
            for row in car3["profiles"][name]
            if row.get("scope") == "aggregate"
            and row.get("region") == region
        ]
        bin_order = _feature_bins(
            {
                "transition": "posterior_transition_distance",
                "onset": "onset_distance",
                "offset": "offset_distance",
            }[name]
        )
        rows.sort(
            key=lambda row: bin_order.index(str(row["distance_bin"]))
        )
        labels = [str(row["distance_bin"]) for row in rows]
        values = [
            _float_or_none(row.get("mean_value")) for row in rows
        ]
        axis.axhline(0.0, color="#555555", linewidth=1.0)
        axis.plot(labels, values, marker="o", color="#4C78A8")
        axis.set_title(f"{name} ({region})")
        axis.set_ylabel("Mean v")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car4_condition_heatmap(
    car4: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = {
        str(row["group"]): row
        for row in car4["condition_dependence"]
        if row.get("architecture") == "TinyGRU"
        and row.get("scope") == "aggregate"
        and row.get("group") in CONDITION_CELLS
    }
    domains = ("seen", "unseen")
    matrices = {
        "mean_v": np.full((len(domains), len(SNR_ORDER)), np.nan),
        "p_R": np.full((len(domains), len(SNR_ORDER)), np.nan),
    }
    for row_index, domain in enumerate(domains):
        for column_index, snr in enumerate(SNR_ORDER):
            row = rows.get(f"{domain}/{snr}", {})
            matrices["mean_v"][row_index, column_index] = (
                _float_or_none(row.get("mean_value"))
                if row
                else np.nan
            )
            matrices["p_R"][row_index, column_index] = (
                _float_or_none(row.get("p_R")) if row else np.nan
            )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, metric in zip(axes, ("mean_v", "p_R")):
        image = axis.imshow(
            matrices[metric],
            aspect="auto",
            cmap="viridis" if metric == "mean_v" else "magma",
        )
        axis.set_xticks(np.arange(len(SNR_ORDER)), SNR_ORDER)
        axis.set_yticks(np.arange(len(domains)), domains)
        axis.set_title(metric)
        figure.colorbar(image, ax=axis, shrink=0.85)
        for row_index in range(len(domains)):
            for column_index in range(len(SNR_ORDER)):
                value = matrices[metric][row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car5_seed_agreement(
    car5: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = list(car5["pairs"])
    labels = [
        f"{int(row['left_seed'])}-{int(row['right_seed'])}"
        for row in rows
    ]
    sign = [
        float(row.get("sign_agreement") or 0.0) for row in rows
    ]
    taxonomy = [
        float(row.get("taxonomy_agreement") or 0.0) for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    axis.bar(
        x - width / 2,
        sign,
        width,
        label="Sign agreement",
        color="#4C78A8",
    )
    axis.bar(
        x + width / 2,
        taxonomy,
        width,
        label="Taxonomy agreement",
        color="#F58518",
    )
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Agreement")
    axis.set_title("CAR5 within-GRU reproducibility")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_car_gate_status(
    gate_results: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    names = [f"R{index}" for index in range(1, 6)]
    values = [
        1.0 if gate_results["gates"][name]["pass"] else 0.0
        for name in names
    ]
    colors = ["#54A24B" if value > 0.0 else "#E45756" for value in values]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.bar(names, values, color=colors)
    axis.set_ylim(0.0, 1.15)
    axis.set_yticks([0.0, 1.0], ["fail", "pass"])
    axis.set_title(
        f"{gate_results['status']} ({gate_results['passed_count']}/5)"
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_car_figures(
    *,
    output_dir: Path,
    car1: Mapping[str, Any],
    car2: Mapping[str, Any],
    car3: Mapping[str, Any],
    car4: Mapping[str, Any],
    car5: Mapping[str, Any],
    gate_results: Mapping[str, Any],
) -> list[Path]:
    figure_dir = output_dir / "car_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "car1_taxonomy.png": lambda path: _plot_car1_taxonomy(car1, path),
        "car2_horizon_demand.png": lambda path: _plot_car2_horizon_demand(
            car2, path
        ),
        "car2_nonmonotonicity.png": (
            lambda path: _plot_car2_nonmonotonicity(car2, path)
        ),
        "car3_event_profiles.png": (
            lambda path: _plot_car3_event_profiles(car3, path)
        ),
        "car4_condition_heatmap.png": (
            lambda path: _plot_car4_condition_heatmap(car4, path)
        ),
        "car5_seed_agreement.png": (
            lambda path: _plot_car5_seed_agreement(car5, path)
        ),
        "car_gate_status.png": (
            lambda path: _plot_car_gate_status(gate_results, path)
        ),
    }
    written: list[Path] = []
    for name in CAR_FIGURE_FILES:
        path = figure_dir / name
        paths[name](path)
        written.append(path)
    return written


def _format_car_number(value: Any, digits: int = 6) -> str:
    numeric = _float_or_none(value)
    return "NA" if numeric is None else f"{numeric:.{digits}f}"


def _build_car_final_summary(
    *,
    protocol: Mapping[str, Any],
    output_dir: Path,
    data: CarAnalysisData,
    car1: Mapping[str, Any],
    car2: Mapping[str, Any],
    car3: Mapping[str, Any],
    car3_gate: Mapping[str, Any],
    car4: Mapping[str, Any],
    car5: Mapping[str, Any],
    gate_results: Mapping[str, Any],
    cross_architecture: Sequence[Mapping[str, Any]],
    figures: Sequence[Path],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "completed_at": _utc_now(),
        "status": gate_results["status"],
        "passed_count": int(gate_results["passed_count"]),
        "gates": gate_results["gates"],
        "car1_aggregate": dict(car1["aggregate"]),
        "car2_aggregate": dict(car2["aggregate"]),
        "car3_gate": {
            "pass": bool(car3_gate["pass"]),
            "passing_structures": list(
                car3_gate["passing_structures"]
            ),
            "passed_count": int(car3_gate["passed_count"]),
        },
        "car4_gate": dict(car4["gate"]),
        "car5_gate": dict(gate_results["car5_gate"]),
        "cross_architecture_summary": [
            dict(row) for row in cross_architecture
        ],
        "scope": {
            "test_frames": int(
                np.count_nonzero(data.reference.test_mask)
            ),
            "source_clusters": len(data.reference.source_order),
            "speaker_clusters": len(data.reference.speaker_order),
            "model_seeds": list(MODEL_SEEDS),
            "horizons": list(HORIZON_LABELS.values()),
            "primary_short": "H64",
            "primary_long": "HFULL",
            "secondary_long": "H384",
        },
        "figures": [
            str(path.relative_to(output_dir)) for path in figures
        ],
        "NEXT_METHOD_SEARCH_AUTHORIZED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
    }


def _car_final_report_lines(
    summary: Mapping[str, Any],
) -> list[str]:
    gates = summary["gates"]
    car1 = summary["car1_aggregate"]
    car2 = summary["car2_aggregate"]
    car3 = summary["car3_gate"]
    car4 = summary["car4_gate"]
    car5 = summary["car5_gate"]
    lines = [
        "# A-v2 Cross-Architecture Replication Final Report",
        "",
        f"- Protocol: `{summary['protocol_sha256']}`",
        f"- Status: **{summary['status']}**",
        f"- Gates passed: `{summary['passed_count']}/5`",
        "- `NEXT_METHOD_SEARCH_AUTHORIZED=false`",
        "- `NEW_FINAL_OOD_TOUCHED=false`",
        "",
        "## Gate Results",
        "",
        "| Gate | Pass | Result |",
        "|---|---:|---|",
    ]
    for name in ("R1", "R2", "R3", "R4", "R5"):
        lines.append(
            f"| {name} | {str(bool(gates[name]['pass'])).lower()} | "
            f"{gates[name]['rule']} |"
        )
    lines.extend(
        [
            "",
            "## CAR1-CAR2",
            "",
            (
                "- Aggregate taxonomy: "
                f"P(SS)={_format_car_number(car1.get('p_SS'))}, "
                f"P(R)={_format_car_number(car1.get('p_R'))}, "
                f"P(I)={_format_car_number(car1.get('p_I'))}, "
                f"P(H)={_format_car_number(car1.get('p_H'))}."
            ),
            (
                "- Aggregate NetRefinability: "
                f"{_format_car_number(car1.get('availability_net'))}; "
                "source lower bound: "
                f"{_format_car_number(car1.get('source_ci95_low_availability_net'))}."
            ),
            (
                "- Aggregate mean v: "
                f"{_format_car_number(car1.get('mean_value'))}."
            ),
            (
                "- CAR2 non-monotone rate: "
                f"{_format_car_number(car2.get('non_monotone_rate'))}; "
                "long-history share: "
                f"{_format_car_number(car2.get('long_history_share'))}."
            ),
            "",
            "## CAR3-CAR5",
            "",
            (
                "- CAR3 passing structures: "
                f"`{', '.join(car3['passing_structures']) or 'none'}`."
            ),
            (
                "- CAR4 12-cell mean-v range: "
                f"{_format_car_number(car4.get('point_range'))}; "
                "source lower bound: "
                f"{_format_car_number(car4.get('source_ci95_low'))}."
            ),
            (
                "- CAR5 full mean v: "
                f"{_format_car_number(car5.get('full_mean_v'))}; "
                "maximum seed positive share: "
                f"{_format_car_number(car5.get('maximum_positive_share'))}."
            ),
            "",
            "## Cross-Architecture Diagnostics",
            "",
            "| Item | Classification | Tiny-GRU | MarbleNet |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["cross_architecture_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["item"]),
                    str(row["classification"]),
                    _format_car_number(row.get("tiny_value")),
                    _format_car_number(row.get("marble_value")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            (
                "The result is limited to the frozen LibriVAD A9 test "
                "evaluation and the tested Tiny-GRU recurrent backbone."
            ),
            (
                "No router, method, architecture, or hyperparameter "
                "selection follows from these outcomes."
            ),
            (
                "The sealed NEW_FINAL_OOD split was not loaded, inspected, "
                "calibrated, or evaluated."
            ),
            "",
        ]
    )
    return lines


def _car_claim_freeze_lines(
    summary: Mapping[str, Any],
) -> list[str]:
    status = str(summary["status"])
    car1 = summary["car1_aggregate"]
    car2 = summary["car2_aggregate"]
    car3 = summary["car3_gate"]
    car4 = summary["car4_gate"]
    car5 = summary["car5_gate"]
    if status == "STRONG_CROSS_ARCH_REPLICATION":
        supported = (
            "The sparse and condition-dependent structure of temporal "
            "refinement value observed with the convolutional backbone "
            "was reproduced in the tested recurrent backbone."
        )
    elif status == "CROSS_ARCH_CONDITIONAL":
        supported = (
            "Only the gate-supported components are conditionally "
            "consistent across the tested backbones; the full "
            "cross-architecture replication claim is not supported."
        )
    else:
        supported = (
            "The frozen gates do not support a cross-architecture "
            "replication claim for the tested recurrent backbone."
        )
    return [
        "# CAR Claim Freeze",
        "",
        "## OBSERVATIONS",
        "",
        f"- Final frozen status: `{status}`.",
        f"- Gates passed: `{summary['passed_count']}/5`.",
        (
            "- Aggregate P(R), P(I), P(H), and NetRefinability: "
            f"`{_format_car_number(car1.get('p_R'))}`, "
            f"`{_format_car_number(car1.get('p_I'))}`, "
            f"`{_format_car_number(car1.get('p_H'))}`, "
            f"`{_format_car_number(car1.get('availability_net'))}`."
        ),
        (
            "- Aggregate non-monotone rate and long-history share: "
            f"`{_format_car_number(car2.get('non_monotone_rate'))}`, "
            f"`{_format_car_number(car2.get('long_history_share'))}`."
        ),
        (
            "- Passing CAR3 structures: "
            f"`{', '.join(car3['passing_structures']) or 'none'}`."
        ),
        (
            "- CAR4 condition-cell mean-v range: "
            f"`{_format_car_number(car4.get('point_range'))}`."
        ),
        (
            "- CAR5 maximum seed positive-frame share: "
            f"`{_format_car_number(car5.get('maximum_positive_share'))}`."
        ),
        "",
        "## SUPPORTED INTERPRETATIONS",
        "",
        f"- {supported}",
        (
            "- The comparison is restricted to the frozen A9 evaluation "
            "set and the five preregistered Tiny-GRU seeds."
        ),
        "",
        "## UNSUPPORTED CLAIMS",
        "",
        "- Universal across neural VAD architectures.",
        "- Architecture-independent temporal value.",
        "- Tiny GRU proves the MarbleNet mechanism.",
        "- Long context universally improves VAD.",
        "- Refinability-aware routing is superior.",
        "- Deployment-level generalization.",
        (
            "- Any claim based on the sealed NEW_FINAL_OOD split or on "
            "post-outcome method, architecture, or seed selection."
        ),
        "",
        "- `NEXT_METHOD_SEARCH_AUTHORIZED=false`",
        "- `NEW_FINAL_OOD_TOUCHED=false`",
        "",
    ]


def _collect_car_artifacts(output_dir: Path) -> list[Path]:
    paths = [
        PROTOCOL_PATH,
        PROTOCOL_HASH_PATH,
        output_dir / "car_gru_training_runs.csv",
        output_dir / "car_causality_audit.json",
        output_dir / "car1_taxonomy_by_seed.csv",
        output_dir / "car1_taxonomy_by_condition.csv",
        output_dir / "car1_bootstrap.csv",
        output_dir / "car2_history_demand.csv",
        output_dir / "car2_nonmonotonicity.csv",
        output_dir / "car3_event_structure.csv",
        output_dir / "car3_event_comparison.csv",
        output_dir / "car3_gate_support.csv",
        output_dir / "car4_condition_dependence.csv",
        output_dir / "car4_cross_architecture.csv",
        output_dir / "car5_seed_agreement.csv",
        output_dir / "car_cross_architecture_summary.csv",
        output_dir / "car_final_summary.json",
        output_dir / "car_final_report.md",
        output_dir / "car_claim_freeze.md",
    ]
    for directory in (
        "car3_event_profiles",
        "car_figures",
        "checkpoints",
        "evaluations",
    ):
        paths.extend(_files_under(output_dir / directory))
    return paths


def run_freeze_stage(*, results_dir: Path) -> Path:
    marker = _stage_marker(results_dir, "freeze")
    _refuse_overwrite(
        [marker, PROTOCOL_PATH, PROTOCOL_HASH_PATH],
        stage="CAR protocol freeze",
    )
    protocol_path = write_protocol_freeze()
    _write_stage_marker(
        results_dir,
        stage="freeze",
        artifacts=[PROTOCOL_PATH, PROTOCOL_HASH_PATH],
    )
    return protocol_path


def run_train_stage(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    marker = _stage_marker(results_dir, "train")
    complete_paths = [
        results_dir / "checkpoints" / f"seed{seed}" / "history.json"
        for seed in MODEL_SEEDS
    ] + [
        results_dir / "checkpoints" / f"seed{seed}" / "best.pt"
        for seed in MODEL_SEEDS
    ]
    if marker.exists() or all(path.is_file() for path in complete_paths):
        raise FileExistsError(
            "CAR training is already complete; refusing to overwrite "
            f"{marker}"
        )
    rows = train_all(
        protocol=protocol,
        results_dir=results_dir,
        device=device,
        resume=True,
    )
    artifacts = [
        results_dir / "car_gru_training_runs.csv",
        *complete_paths,
    ]
    _write_stage_marker(
        results_dir,
        stage="train",
        artifacts=artifacts,
    )
    return rows


def run_audit_stage(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    marker = _stage_marker(results_dir, "audit")
    audit_path = results_dir / "car_causality_audit.json"
    _refuse_overwrite([marker, audit_path], stage="CAR causality audit")
    result = run_causality_audit(
        protocol=protocol,
        results_dir=results_dir,
        device=device,
    )
    _write_stage_marker(
        results_dir,
        stage="audit",
        artifacts=[audit_path],
    )
    return result


def run_evaluate_stage(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
    device: torch.device,
) -> None:
    marker = _stage_marker(results_dir, "evaluate")
    evaluation_paths = [
        path
        for seed in MODEL_SEEDS
        for path in (
            results_dir / "evaluations" / f"seed{seed}" / "scores.npz",
            results_dir / "evaluations" / f"seed{seed}" / "evaluation.json",
        )
    ]
    if marker.exists() or all(path.is_file() for path in evaluation_paths):
        raise FileExistsError(
            "CAR evaluation is already complete; refusing to overwrite "
            f"{marker}"
        )
    evaluate_all(
        protocol=protocol,
        results_dir=results_dir,
        device=device,
    )
    _write_stage_marker(
        results_dir,
        stage="evaluate",
        artifacts=evaluation_paths,
    )


def run_analyze_stage(
    *,
    protocol: Mapping[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    marker = _stage_marker(results_dir, "analyze")
    final_paths = [
        results_dir / "car_final_summary.json",
        results_dir / "car_final_report.md",
        results_dir / "car_claim_freeze.md",
        results_dir / "car_execution_manifest.json",
    ]
    table_paths = [
        results_dir / name
        for name in protocol["outputs"]["tables"]
    ]
    _refuse_overwrite(
        [
            marker,
            *final_paths,
            *table_paths,
            results_dir / "car3_event_profiles",
            results_dir / "car_figures",
        ],
        stage="CAR analysis",
    )
    data = load_car_analysis_data(protocol, results_dir=results_dir)
    car1 = run_car1(data, output_dir=results_dir)
    car2 = run_car2(data, output_dir=results_dir)
    car3 = run_car3(data, output_dir=results_dir)
    car3_gate = run_car3_gate_support(
        data,
        protocol=protocol,
        output_dir=results_dir,
    )
    car4 = run_car4(data, output_dir=results_dir)
    car5 = run_car5(data, output_dir=results_dir)
    gate_results = _car_gate_results(
        car1=car1,
        car2=car2,
        car3_gate=car3_gate,
        car4=car4,
        car5=car5,
    )
    cross_architecture = _build_cross_architecture_summary(
        protocol=protocol,
        data=data,
        car1=car1,
        car2=car2,
        car3_gate=car3_gate,
        car4=car4,
        car5=car5,
    )
    _write_csv(
        results_dir / "car_cross_architecture_summary.csv",
        cross_architecture,
    )
    figures = _write_car_figures(
        output_dir=results_dir,
        car1=car1,
        car2=car2,
        car3=car3,
        car4=car4,
        car5=car5,
        gate_results=gate_results,
    )
    summary = _build_car_final_summary(
        protocol=protocol,
        output_dir=results_dir,
        data=data,
        car1=car1,
        car2=car2,
        car3=car3,
        car3_gate=car3_gate,
        car4=car4,
        car5=car5,
        gate_results=gate_results,
        cross_architecture=cross_architecture,
        figures=figures,
    )
    _write_json(results_dir / "car_final_summary.json", summary)
    (results_dir / "car_final_report.md").write_text(
        "\n".join(_car_final_report_lines(summary)),
        encoding="utf-8",
    )
    (results_dir / "car_claim_freeze.md").write_text(
        "\n".join(_car_claim_freeze_lines(summary)),
        encoding="utf-8",
    )
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "completed_at": _utc_now(),
        "status": gate_results["status"],
        "passed_count": int(gate_results["passed_count"]),
        "gates": {
            name: bool(gate["pass"])
            for name, gate in gate_results["gates"].items()
        },
        "scope_note": (
            "All CAR evaluation uses the frozen A9 test manifest and "
            "frozen RF384 reference. Frame counts are not pooled across "
            "architectures."
        ),
        "NEXT_METHOD_SEARCH_AUTHORIZED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "artifacts": _artifact_hash_rows(
            results_dir,
            _collect_car_artifacts(results_dir),
        ),
    }
    _write_json(results_dir / "car_execution_manifest.json", manifest)
    _write_stage_marker(
        results_dir,
        stage="analyze",
        artifacts=[
            *_collect_car_artifacts(results_dir),
            results_dir / "car_execution_manifest.json",
        ],
    )
    print(f"CAR_STATUS={gate_results['status']}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen A-v2 cross-architecture study."
    )
    parser.add_argument(
        "--stage",
        choices=("freeze", "train", "audit", "evaluate", "analyze", "all"),
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir).resolve()
    if results_dir != DEFAULT_RESULTS_DIR.resolve():
        raise ValueError(
            "CAR paths are frozen to "
            f"{DEFAULT_RESULTS_DIR}; got {results_dir}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.stage == "freeze":
        run_freeze_stage(results_dir=results_dir)
        return 0
    if args.stage == "all" and not (
        PROTOCOL_PATH.is_file() and PROTOCOL_HASH_PATH.is_file()
    ):
        run_freeze_stage(results_dir=results_dir)

    protocol = load_car_protocol(verify_hashes=True)
    if args.stage in ("train", "all"):
        run_train_stage(
            protocol=protocol,
            results_dir=results_dir,
            device=device,
        )
    if args.stage in ("audit", "all"):
        run_audit_stage(
            protocol=protocol,
            results_dir=results_dir,
            device=device,
        )
    if args.stage in ("evaluate", "all"):
        run_evaluate_stage(
            protocol=protocol,
            results_dir=results_dir,
            device=device,
        )
    if args.stage in ("analyze", "all"):
        run_analyze_stage(
            protocol=protocol,
            results_dir=results_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
