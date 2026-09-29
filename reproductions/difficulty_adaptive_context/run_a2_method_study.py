# -*- coding: utf-8 -*-
"""A-v2 frozen method study: M0/M1/M2 under a fixed refinement budget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    deterministic_speaker_split,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
)
from reproductions.difficulty_adaptive_context.run_ax1_value_predictability import (
    encode_short_stream,
    feature_blocks,
)
from reproductions.difficulty_adaptive_context.run_ax3_event_structure import (
    build_event_structure,
)
from reproductions.difficulty_adaptive_context.run_ae_mechanism import (
    _predict_checkpoint_on_a9,
)
from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_AX_PROTOCOL = (
    REPO_ROOT / "reproductions" / "difficulty_adaptive_context" / "ax_protocol.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "difficulty_adaptive_context" / "a2_method_study"
)
DEFAULT_PROTOCOL_PATH = DEFAULT_OUTPUT_DIR / "a2_method_protocol_freeze.json"
DEFAULT_PROTOCOL_HASH_PATH = DEFAULT_OUTPUT_DIR / "a2_protocol_sha256.txt"

PROTOCOL_ID = "A-v2-METHOD-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_TRAINING"
TRAINING_SEEDS = (17, 23, 41, 59, 71)
BUDGETS = (0.01, 0.02, 0.05, 0.10)
PRIMARY_BUDGET = 0.05
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_BASE_SEED = 20260921
SPLIT_SEED = 20260920
SPLIT_COUNTS = (24, 8, 8)
DECISION_THRESHOLD = 0.5
AX_WINDOW_FRAMES = 25
M1_HIDDEN_WIDTH = 64
M2_HIDDEN_WIDTH = 64
TRAIN_BATCH_SIZE = 4096
TRAIN_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
SMOOTH_L1_BETA = 0.5
MIN_CLASS_WEIGHT = 0.25
MAX_CLASS_WEIGHT = 8.0
PRIMARY_POS_WEIGHT_CAP = 10.0
CATastrophic_CELL_DELTA = -0.005
MAX_UNCONTROLLED_ACTIVATION = 0.15
PRIMARY_ACTIVATION_LOWER = 0.03
PRIMARY_ACTIVATION_UPPER = 0.075
COMPUTE_ADVANTAGE_MIN_MS = 0.20
COMPUTE_ADVANTAGE_MIN_FRACTION = 0.10
INDEPENDENT_SEEDS = (73, 79, 83, 89, 97)
AX_UNCERTAINTY_THRESHOLD = 0.13
SMOKE_MARKER_NAME = "a2_smoke_attempt.json"
INTERNAL_RUN_STARTED_MARKER_NAME = "a2_internal_test_run_started.json"
INTERNAL_RUN_COMPLETE_MARKER_NAME = "a2_internal_test_run_complete.json"

TAXONOMY_ORDER = ("R", "I", "H", "SS")
DOMAIN_ORDER = ("seen", "unseen")
SNR_ORDER = ("-5", "0", "5", "10", "15", "20")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def taxonomy_from_scores(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_correct = (short_scores >= DECISION_THRESHOLD).astype(np.int8) == labels
    refined_correct = (refined_scores >= DECISION_THRESHOLD).astype(np.int8) == labels
    value = refined_correct.astype(np.int8) - short_correct.astype(np.int8)
    tax = np.full(labels.size, "SS", dtype="U2")
    tax[(~short_correct) & refined_correct] = "R"
    tax[(~short_correct) & (~refined_correct)] = "I"
    tax[short_correct & (~refined_correct)] = "H"
    if np.any((value != 0) & (~np.isin(tax, ("R", "H")))):
        raise RuntimeError("signed value and taxonomy disagree")
    if np.any((value == 1) & (tax != "R")):
        raise RuntimeError("positive value is not exactly refinable R")
    if np.any((value == -1) & (tax != "H")):
        raise RuntimeError("negative value is not exactly harmful H")
    return value.astype(np.int8), tax


def split_fingerprint(
    speaker_ids: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> str:
    speakers = np.asarray(speaker_ids, dtype=str)
    role_by_speaker: dict[str, str] = {}
    for speaker in sorted(set(speakers.tolist())):
        roles = [
            role
            for role, mask in masks.items()
            if np.any(np.asarray(mask, dtype=bool) & (speakers == speaker))
        ]
        if len(roles) != 1:
            raise ValueError(f"speaker {speaker} does not occupy exactly one role")
        role_by_speaker[speaker] = roles[0]
    encoded = json.dumps(
        role_by_speaker,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_new_final_ood_untouched(payload: Mapping[str, Any]) -> None:
    section = payload.get("new_final_ood")
    if not isinstance(section, Mapping):
        raise ValueError("protocol is missing the NEW_FINAL_OOD declaration")
    if section.get("touched") is not False:
        raise ValueError("NEW_FINAL_OOD is not sealed")
    if section.get("result_generation") != "forbidden":
        raise ValueError("NEW_FINAL_OOD result generation is not forbidden")


def load_new_final_ood(*_: Any, **__: Any) -> None:
    raise RuntimeError(
        "NEW_FINAL_OOD is untouched and cannot be loaded by the A-v2 "
        "development study"
    )


def claim_once(path: Path, *, command: str, protocol_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; the A-v2 {command} run is closed"
        )
    write_json(
        path,
        {
            "command": command,
            "claimed_at_utc": utc_now(),
            "protocol_sha256": protocol_sha256,
        },
    )


def three_way_speaker_split(
    speaker_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    names = np.asarray(sorted(set(str(value) for value in speaker_ids)))
    if names.size != sum(SPLIT_COUNTS):
        raise ValueError(
            f"expected {sum(SPLIT_COUNTS)} speakers, found {names.size}"
        )
    shuffled = names[np.random.default_rng(SPLIT_SEED).permutation(names.size)]
    train_names = sorted(str(value) for value in shuffled[: SPLIT_COUNTS[0]])
    dev_names = sorted(
        str(value)
        for value in shuffled[SPLIT_COUNTS[0] : SPLIT_COUNTS[0] + SPLIT_COUNTS[1]]
    )
    test_names = sorted(
        str(value)
        for value in shuffled[SPLIT_COUNTS[0] + SPLIT_COUNTS[1] :]
    )
    if (
        set(train_names) & set(dev_names)
        or set(train_names) & set(test_names)
        or set(dev_names) & set(test_names)
    ):
        raise RuntimeError("speaker cohorts overlap")
    speakers = np.asarray([str(value) for value in speaker_ids])
    masks = {
        "TRAIN": np.isin(speakers, train_names),
        "DEV": np.isin(speakers, dev_names),
        "INTERNAL_TEST": np.isin(speakers, test_names),
    }
    if np.any(masks["TRAIN"] & masks["DEV"]) or np.any(masks["TRAIN"] & masks["INTERNAL_TEST"]) or np.any(masks["DEV"] & masks["INTERNAL_TEST"]):
        raise RuntimeError("speaker masks overlap")
    if not np.all(masks["TRAIN"] | masks["DEV"] | masks["INTERNAL_TEST"]):
        raise RuntimeError("speaker split does not cover all frames")
    return masks["TRAIN"], masks["DEV"], masks["INTERNAL_TEST"], {
        "TRAIN": train_names,
        "DEV": dev_names,
        "INTERNAL_TEST": test_names,
    }


@dataclass(frozen=True)
class FrameData:
    labels: np.ndarray
    short_scores: np.ndarray
    refined_scores: np.ndarray
    speaker_ids: np.ndarray
    source_key: np.ndarray
    noise_name: np.ndarray
    condition: np.ndarray
    value: np.ndarray
    taxonomy: np.ndarray
    train_mask: np.ndarray
    dev_mask: np.ndarray
    test_mask: np.ndarray
    speakers: dict[str, list[str]]
    hidden: np.ndarray
    mfcc: np.ndarray

    def selected_features(self) -> dict[str, np.ndarray]:
        segments = utterance_segments(
            self.source_key,
            self.noise_name,
            self.condition,
        )
        return feature_blocks(
            self.short_scores,
            self.hidden,
            self.mfcc,
            segments,
            window_frames=AX_WINDOW_FRAMES,
        )


def load_frame_data(
    *,
    frame_bundle_path: Path,
    feature_path: Path,
) -> FrameData:
    with np.load(frame_bundle_path, allow_pickle=False) as payload:
        required = {
            "labels",
            "short_scores",
            "full_adaptive_scores",
            "speaker_ids",
            "source_key",
            "noise_name",
            "condition",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"frame bundle is missing {missing}")
        labels = np.asarray(payload["labels"], dtype=np.int64)
        short_scores = np.asarray(payload["short_scores"], dtype=np.float64)
        refined_scores = np.asarray(
            payload["full_adaptive_scores"], dtype=np.float64
        )
        speaker_ids = np.asarray(payload["speaker_ids"], dtype=str)
        source_key = np.asarray(payload["source_key"], dtype=str)
        noise_name = np.asarray(payload["noise_name"], dtype=str)
        condition = np.asarray(payload["condition"], dtype=str)
    with np.load(feature_path, allow_pickle=False) as payload:
        hidden = np.asarray(payload["hidden"], dtype=np.float32)
        mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
    size = int(labels.size)
    for name, values in (
        ("short_scores", short_scores),
        ("refined_scores", refined_scores),
        ("speaker_ids", speaker_ids),
        ("source_key", source_key),
        ("noise_name", noise_name),
        ("condition", condition),
        ("hidden", hidden),
        ("mfcc", mfcc),
    ):
        if int(values.shape[0]) != size:
            raise ValueError(f"{name} has {values.shape[0]} rows, expected {size}")
    value, taxonomy = taxonomy_from_scores(labels, short_scores, refined_scores)
    train_mask, dev_mask, test_mask, speakers = three_way_speaker_split(
        speaker_ids
    )
    for source in sorted(set(source_key.tolist())):
        source_speakers = set(speaker_ids[source_key == source].tolist())
        if len(source_speakers) != 1:
            raise ValueError(f"source {source} spans multiple speakers")
    masks = {
        "TRAIN": train_mask,
        "DEV": dev_mask,
        "INTERNAL_TEST": test_mask,
    }
    for left, right in (
        ("TRAIN", "DEV"),
        ("TRAIN", "INTERNAL_TEST"),
        ("DEV", "INTERNAL_TEST"),
    ):
        left_sources = set(source_key[masks[left]].tolist())
        right_sources = set(source_key[masks[right]].tolist())
        if left_sources & right_sources:
            raise ValueError(f"{left} and {right} source cohorts overlap")
    return FrameData(
        labels=labels,
        short_scores=short_scores,
        refined_scores=refined_scores,
        speaker_ids=speaker_ids,
        source_key=source_key,
        noise_name=noise_name,
        condition=condition,
        value=value,
        taxonomy=taxonomy,
        train_mask=train_mask,
        dev_mask=dev_mask,
        test_mask=test_mask,
        speakers=speakers,
        hidden=hidden,
        mfcc=mfcc,
    )


def utterance_segments(
    source_key: np.ndarray,
    noise_name: np.ndarray,
    condition: np.ndarray,
) -> list[tuple[int, int]]:
    composite = np.char.add(
        np.char.add(np.asarray(source_key, dtype=str), "|"),
        np.char.add(
            np.asarray(noise_name, dtype=str),
            np.char.add("|", np.asarray(condition, dtype=str)),
        ),
    )
    starts = np.r_[0, np.flatnonzero(composite[1:] != composite[:-1]) + 1]
    ends = np.r_[starts[1:], composite.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


@dataclass(frozen=True)
class FrozenProtocol:
    path: Path
    payload: dict[str, Any]
    sha256: str


def read_compute_reference() -> dict[str, float]:
    a11_path = (
        REPO_ROOT
        / "results"
        / "difficulty_adaptive_context"
        / "a11_conditional_compute"
        / "a11_conditional_compute.json"
    )
    a11 = read_json(a11_path)
    rows_by_name = {str(row["name"]): row for row in a11["rows"]}
    return {
        "short_mean_ms_per_frame": float(
            rows_by_name["short"]["mean_ms_per_frame"]
        ),
        "short_core_ms_per_frame": float(
            rows_by_name["short_core"]["mean_ms_per_frame"]
        ),
        "always_refine_mean_ms_per_frame": float(
            rows_by_name["fixed_rf384"]["mean_ms_per_frame"]
        ),
        "refinement_kernel_mean_ms_per_frame": float(
            rows_by_name["fixed_rf384"]["refinement_kernel"][
                "mean_ms_per_frame"
            ]
        ),
        "m0_router_overhead_ms_per_frame": float(
            rows_by_name["adaptive_5pct"][
                "router_overhead_ms_per_frame"
            ]
        ),
        "scheduling_overhead_ms_per_frame": float(
            rows_by_name["adaptive_5pct"][
                "scheduling_overhead_ms_per_frame"
            ]
        ),
    }


def freeze_protocol(
    *,
    output_dir: Path,
    frame_bundle_path: Path,
    feature_path: Path,
    ax_protocol_path: Path,
) -> FrozenProtocol:
    frame_data = load_frame_data(
        frame_bundle_path=frame_bundle_path,
        feature_path=feature_path,
    )
    ax_payload = read_json(ax_protocol_path)
    checkpoint_path = resolve_path(ax_payload["data"]["adaptive_checkpoint"]["path"])
    compute_reference = read_compute_reference()
    split_masks = {
        "TRAIN": frame_data.train_mask,
        "DEV": frame_data.dev_mask,
        "INTERNAL_TEST": frame_data.test_mask,
    }
    fingerprint = split_fingerprint(frame_data.speaker_ids, split_masks)
    split_summary = {
        role: {
            "speakers": frame_data.speakers[role],
            "frames": int(mask.sum()),
            "sources": int(len(set(frame_data.source_key[mask].tolist()))),
            "taxonomy_counts": {
                name: int(np.count_nonzero(frame_data.taxonomy[mask] == name))
                for name in TAXONOMY_ORDER
            },
            "signed_value_mean": float(np.mean(frame_data.value[mask])),
        }
        for role, mask in (
            ("TRAIN", frame_data.train_mask),
            ("DEV", frame_data.dev_mask),
            ("INTERNAL_TEST", frame_data.test_mask),
        )
    }
    locks = {
        "frame_bundle": {
            "path": relpath(frame_bundle_path),
            "sha256": sha256_file(frame_bundle_path),
        },
        "feature_cache": {
            "path": relpath(feature_path),
            "sha256": sha256_file(feature_path),
        },
        "ax_protocol": {
            "path": relpath(ax_protocol_path),
            "sha256": sha256_file(ax_protocol_path),
        },
        "short_rf384_checkpoint": {
            "path": relpath(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "runner": {
            "path": relpath(RUNNER_PATH),
            "sha256": sha256_file(RUNNER_PATH),
        },
    }
    ae4_locks: dict[str, dict[str, str]] = {}
    for seed in INDEPENDENT_SEEDS:
        prediction = (
            REPO_ROOT
            / "results"
            / "difficulty_adaptive_context"
            / "ae_mechanism"
            / "ae4_replication_predictions"
            / f"seed{seed}"
            / "frame_predictions.npz"
        )
        checkpoint = (
            REPO_ROOT
            / "results"
            / "difficulty_adaptive_context"
            / "ae_mechanism"
            / "ae4_replication_training"
            / f"seed{seed}"
            / "refiner"
            / "best.pt"
        )
        if not prediction.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"independent replication inputs missing for seed {seed}"
            )
        ae4_locks[f"seed{seed}_predictions"] = {
            "path": relpath(prediction),
            "sha256": sha256_file(prediction),
        }
        ae4_locks[f"seed{seed}_checkpoint"] = {
            "path": relpath(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    actual_ax_protocol = read_json(ax_protocol_path)
    expected_bundle = resolve_path(
        actual_ax_protocol["data"]["prediction_bundle"]["path"]
    )
    if expected_bundle.resolve() != frame_bundle_path.resolve():
        raise ValueError("AX protocol and frame bundle disagree")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "frozen_at_utc": utc_now(),
        "locks": locks,
        "study": {
            "a_v1": "COMPLETE / CONDITIONAL GO",
            "a14": "FROZEN DEVELOPMENT EVIDENCE",
            "a15": "FORBIDDEN",
            "ae": "CLAIM FREEZE / DEVELOPMENT EVIDENCE",
            "purpose": (
                "Test whether learned refinability or signed-value routing "
                "beats frozen uncertainty routing at a fixed RF384 budget."
            ),
            "new_final_ood": "MANIFEST_ONLY / UNTOUCHED",
        },
        "data": {
            "frame_bundle": locks["frame_bundle"],
            "feature_cache": locks["feature_cache"],
            "frames": int(frame_data.labels.size),
            "speakers": int(len(set(frame_data.speaker_ids.tolist()))),
            "sources": int(len(set(frame_data.source_key.tolist()))),
            "split": {
                "method": "fresh_deterministic_speaker_partition",
                "seed": SPLIT_SEED,
                "counts": dict(zip(("TRAIN", "DEV", "INTERNAL_TEST"), SPLIT_COUNTS)),
                "fingerprint_sha256": fingerprint,
                "roles": split_summary,
                "old_a9_roles": (
                    "not inherited; all new roles are reconstructed from "
                    "the frozen frame bundle"
                ),
            },
        },
        "targets": {
            "decision_threshold": DECISION_THRESHOLD,
            "short_error": "1[short_prediction != label]",
            "refined_error": "1[RF384_prediction != label]",
            "signed_value": "v = short_error - refined_error",
            "taxonomy": {
                "R": "Short wrong, RF384 correct",
                "I": "Short wrong, RF384 wrong",
                "H": "Short correct, RF384 wrong",
                "SS": "Short correct, RF384 correct",
            },
        },
        "features": {
            "definition_source": relpath(
                REPO_ROOT
                / "reproductions"
                / "difficulty_adaptive_context"
                / "run_ax1_value_predictability.py"
            ),
            "input": "X3 exactly as feature_blocks(...)[\"X3\"]",
            "dimension": 330,
            "causal_window_frames": AX_WINDOW_FRAMES,
            "normalization": (
                "TRAIN-only mean/variance; variance uses ddof=0; zero "
                "variance channels are scaled by 1"
            ),
            "forbidden_inputs": [
                "future frames",
                "RF384 output",
                "label-derived features",
            ],
        },
        "methods": {
            "M0": {
                "name": "frozen uncertainty",
                "training": "none",
                "score": "-abs(short_score - 0.5)",
            },
            "M1": {
                "name": "signed value regressor",
                "input": "X3",
                "architecture": (
                    "Linear(330,64) -> GELU -> Dropout(0.1) -> "
                    "Linear(64,64) -> GELU -> Dropout(0.1) -> Linear(64,1)"
                ),
                "loss": (
                    "SmoothL1(beta=0.5, reduction=none) times TRAIN class "
                    "weights; class weights n/(3*n_class), clipped [0.25,8], "
                    "renormalized to mean 1"
                ),
                "score": "predicted signed value",
            },
            "M2": {
                "name": "refinability vs harmful/irreducible multi-head model",
                "input": "X3",
                "architecture": (
                    "shared Linear(330,64) -> GELU -> Dropout(0.1), then "
                    "three scalar heads: R-vs-(I+H+SS), R-vs-H, R-vs-I"
                ),
                "loss": (
                    "primary BCEWithLogits with clipped inverse-frequency "
                    "pos_weight plus 0.25*(R-vs-H BCE on R/H + R-vs-I BCE "
                    "on R/I)"
                ),
                "score": "sigmoid(primary R head)",
            },
        },
        "training": {
            "seeds": list(TRAINING_SEEDS),
            "batch_size": TRAIN_BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping": "none",
            "augmentation": "none",
            "dropout": DROPOUT,
            "device": "CUDA if available, otherwise CPU",
        },
        "selection": {
            "budgets": list(BUDGETS),
            "primary_budget": PRIMARY_BUDGET,
            "calibration_role": "DEV",
            "rule": (
                "select ceil(B * DEV frames) highest DEV scores; threshold "
                "is the minimum selected score; apply once to INTERNAL_TEST"
            ),
            "recalibration_on_internal_test": "forbidden",
            "primary_candidate_rule": (
                "if any learned candidate passes all gates, select the "
                "passing candidate with the largest primary-budget delta U; "
                "otherwise select the candidate with the largest primary-"
                "budget delta U; M1 wins exact ties as the simpler model"
            ),
        },
        "metrics": {
            "primary": "U(B)=mean(selected * v)",
            "secondary": [
                "F1",
                "frame error",
                "correction rate",
                "harm rate",
                "activation rate",
                "value captured",
                "oracle value recovery",
            ],
        },
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "primary_unit": "source cluster (source_key)",
            "sensitivity_unit": "speaker cluster",
            "paired": "same resampled clusters for method and M0",
            "seed": BOOTSTRAP_BASE_SEED,
            "aggregation": (
                "per-seed paired utility differences averaged over the five "
                "predeclared training seeds before cluster bootstrap"
            ),
        },
        "decision_rules": {
            "G1": "delta U > 0 and source-cluster 95% CI lower > 0",
            "G2": "at least 4/5 training seeds have delta U > 0",
            "G3": (
                "unseen aggregate delta U >= 0 and no cell with delta U <= "
                f"{CATastrophic_CELL_DELTA:.3f} among cells with >= 1000 frames"
            ),
            "G4": (
                "primary-budget activation in [0.03,0.075] and all budget "
                "activation <= 0.15"
            ),
            "G5": (
                "estimated 5% method latency is at least 0.20 ms/frame "
                "below AlwaysRefine and at least 10% below AlwaysRefine"
            ),
            "STATUS": [
                "METHOD_GO if a learned candidate passes G1-G5",
                "CONDITIONAL_GO if point delta U > 0 but CI/stability/domain "
                "criteria are incomplete",
                "METHOD_NO_GO if neither learned method has positive utility "
                "or activation/compute is uncontrolled",
            ],
            "post_result_model_search": "forbidden",
        },
        "mechanism_diagnostics": {
            "required": [
                "R/I/H/SS score distributions",
                "uncertainty persistence strata",
                "transition proxy strata",
                "onset/offset strata",
                "seen/unseen",
                "SNR",
                "AE0 availability/observability cells",
            ],
            "top5_selection": [
                "P(R|selected)",
                "P(I|selected)",
                "P(H|selected)",
                "P(SS|selected)",
                "NetOpportunity=P(R|selected)-P(H|selected)",
            ],
        },
        "independent_replication": {
            "seeds": list(INDEPENDENT_SEEDS),
            "role": "backbone sensitivity, not primary confirmation",
            "training_seed_rule": (
                "one training seed equal to the independent replicate seed"
            ),
            "recipe": "same X3, architectures, losses, epochs and budget rule",
            "inputs": ae4_locks,
        },
        "new_final_ood": {
            "status": "NO_UNUSED_CONFIRMATORY_DATASET_AVAILABLE",
            "touched": False,
            "result_generation": "forbidden",
            "description": (
                "The available LibriSpeech test speakers were already used "
                "for A-v1 evaluation. No new untouched confirmatory dataset "
                "is present, so no final-OOD result may be fabricated."
            ),
        },
        "compute": {
            "source": relpath(
                REPO_ROOT
                / "results"
                / "difficulty_adaptive_context"
                / "a11_conditional_compute"
                / "a11_conditional_compute.json"
            ),
            "additive_short_term": "short_core",
            "short_mean_ms_per_frame": compute_reference[
                "short_mean_ms_per_frame"
            ],
            "short_core_ms_per_frame": compute_reference[
                "short_core_ms_per_frame"
            ],
            "always_refine_mean_ms_per_frame": compute_reference[
                "always_refine_mean_ms_per_frame"
            ],
            "refinement_kernel_mean_ms_per_frame": compute_reference[
                "refinement_kernel_mean_ms_per_frame"
            ],
            "m0_router_overhead_ms_per_frame": compute_reference[
                "m0_router_overhead_ms_per_frame"
            ],
            "scheduling_overhead_ms_per_frame": compute_reference[
                "scheduling_overhead_ms_per_frame"
            ],
            "method": (
                "engineering diagnostic; measure CPU router forward overhead "
                "in the frozen environment and combine with the A11 "
                "short_core/refinement kernel references"
            ),
        },
        "outputs": {
            "directory": relpath(output_dir),
            "required": [
                "a2_method_protocol_freeze.json",
                "a2_protocol_sha256.txt",
                "a2_training_manifest.json",
                "a2_all_runs.csv",
                "a2_budget_results.csv",
                "a2_cluster_bootstrap.csv",
                "a2_cell_results.csv",
                "a2_selection_taxonomy.csv",
                "a2_compute_results.csv",
                "a2_independent_replication.csv",
                "a2_mechanism_diagnostics.csv",
                "a2_figures/",
                "a2_final_report.md",
                "a2_final_summary.json",
                "a2_claim_freeze.md",
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "a2_method_protocol_freeze.json"
    write_json(protocol_path, payload)
    protocol_hash = sha256_file(protocol_path)
    hash_path = output_dir / "a2_protocol_sha256.txt"
    hash_path.write_text(
        f"{protocol_hash}  {protocol_path.name}\n",
        encoding="ascii",
    )
    new_ood_manifest = output_dir / "a2_new_final_ood_manifest.json"
    write_json(
        new_ood_manifest,
        {
            "status": "NO_UNUSED_CONFIRMATORY_DATASET_AVAILABLE",
            "touched": False,
            "protocol": relpath(protocol_path),
            "protocol_sha256": protocol_hash,
            "candidate_dataset": None,
            "reason": payload["new_final_ood"]["description"],
            "loader": (
                "load_new_final_ood() refuses by design; a genuinely unused "
                "dataset and a new protocol are required"
            ),
        },
    )
    return FrozenProtocol(
        path=protocol_path,
        payload=payload,
        sha256=protocol_hash,
    )


def load_frozen_protocol(
    protocol_path: Path,
    *,
    hash_path: Path,
) -> FrozenProtocol:
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not hash_path.is_file():
        raise FileNotFoundError(hash_path)
    expected = hash_path.read_text(encoding="ascii").split()[0].strip()
    actual = sha256_file(protocol_path)
    if expected != actual:
        raise ValueError(
            f"protocol hash mismatch: expected {expected}, got {actual}"
        )
    payload = read_json(protocol_path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected A-v2 protocol id")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("A-v2 protocol is not frozen before training")
    assert_new_final_ood_untouched(payload)
    locks = payload.get("locks")
    if not isinstance(locks, Mapping):
        raise ValueError("A-v2 protocol has no lock section")
    runner = locks.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("A-v2 protocol has no runner lock")
    runner_path = resolve_path(str(runner["path"]))
    if sha256_file(runner_path) != runner["sha256"]:
        raise ValueError("A-v2 runner hash changed after protocol freeze")
    return FrozenProtocol(path=protocol_path, payload=payload, sha256=actual)


def verify_protocol_locks(payload: Mapping[str, Any]) -> None:
    locks = payload.get("locks")
    if not isinstance(locks, Mapping):
        raise ValueError("protocol has no lock section")
    for name in (
        "frame_bundle",
        "feature_cache",
        "ax_protocol",
        "short_rf384_checkpoint",
        "runner",
    ):
        entry = locks.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"protocol lock is missing: {name}")
        path = resolve_path(str(entry["path"]))
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{name} hash changed after protocol freeze")
    independent = payload.get("independent_replication", {}).get("inputs", {})
    if not isinstance(independent, Mapping) or not independent:
        raise ValueError("protocol has no independent replication locks")
    for name, entry in independent.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"malformed independent lock: {name}")
        path = resolve_path(str(entry["path"]))
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{name} hash changed after protocol freeze")


class RunningNormalizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, features: np.ndarray, mask: np.ndarray) -> "RunningNormalizer":
        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        count = 0
        total = np.zeros(features.shape[1], dtype=np.float64)
        total_sq = np.zeros(features.shape[1], dtype=np.float64)
        indices = np.flatnonzero(mask)
        for start in range(0, indices.size, 10_000):
            batch = features[indices[start : start + 10_000]].astype(np.float64)
            count += batch.shape[0]
            total += batch.sum(axis=0)
            total_sq += np.square(batch).sum(axis=0)
        if count == 0:
            raise ValueError("cannot fit normalizer on an empty split")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale <= 1e-12] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        result = np.empty(values.shape, dtype=np.float32)
        for start in range(0, values.shape[0], 20_000):
            result[start : start + 20_000] = (
                values[start : start + 20_000] - self.mean
            ) / self.scale
        return result


class M1Regressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, M1_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(M1_HIDDEN_WIDTH, M1_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(M1_HIDDEN_WIDTH, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class M2MultiHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, M2_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.primary = nn.Linear(M2_HIDDEN_WIDTH, 1)
        self.r_vs_h = nn.Linear(M2_HIDDEN_WIDTH, 1)
        self.r_vs_i = nn.Linear(M2_HIDDEN_WIDTH, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "primary": self.primary(hidden).squeeze(-1),
            "r_vs_h": self.r_vs_h(hidden).squeeze(-1),
            "r_vs_i": self.r_vs_i(hidden).squeeze(-1),
        }


def build_model(method: str, input_dim: int) -> nn.Module:
    if method == "M1":
        return M1Regressor(input_dim)
    if method == "M2":
        return M2MultiHead(input_dim)
    raise ValueError(f"unsupported trainable method: {method}")


def m1_class_weights(taxonomy: np.ndarray, value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.int64) + 1
    counts = np.bincount(values, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("TRAIN signed-value classes are incomplete")
    weights = values.size / (3.0 * counts)
    weights = np.clip(weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)
    weights /= weights.mean()
    return weights[values].astype(np.float32)


def train_method(
    method: str,
    features: np.ndarray,
    value: np.ndarray,
    taxonomy: np.ndarray,
    train_mask: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    set_seed(seed)
    x_train = np.asarray(features[train_mask], dtype=np.float32)
    y_train = np.asarray(value[train_mask], dtype=np.float32)
    tax_train = np.asarray(taxonomy[train_mask], dtype="U2")
    model = build_model(method, x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    if method == "M1":
        sample_weights = torch.from_numpy(
            m1_class_weights(tax_train, y_train)
        )
        criterion = nn.SmoothL1Loss(reduction="none", beta=SMOOTH_L1_BETA)
    else:
        primary_positive = tax_train == "R"
        primary_neg = tax_train != "R"
        primary_pos_weight = float(
            np.clip(
                np.count_nonzero(primary_neg)
                / max(np.count_nonzero(primary_positive), 1),
                1.0,
                PRIMARY_POS_WEIGHT_CAP,
            )
        )
        h_mask = np.isin(tax_train, ("R", "H"))
        h_positive = tax_train == "R"
        h_pos_weight = float(
            np.clip(
                np.count_nonzero(h_mask & ~h_positive)
                / max(np.count_nonzero(h_mask & h_positive), 1),
                1.0,
                PRIMARY_POS_WEIGHT_CAP,
            )
        )
        i_mask = np.isin(tax_train, ("R", "I"))
        i_positive = tax_train == "R"
        i_pos_weight = float(
            np.clip(
                np.count_nonzero(i_mask & ~i_positive)
                / max(np.count_nonzero(i_mask & i_positive), 1),
                1.0,
                PRIMARY_POS_WEIGHT_CAP,
            )
        )
        primary_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(primary_pos_weight, device=device)
        )
        h_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(h_pos_weight, device=device)
        )
        i_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(i_pos_weight, device=device)
        )
    for epoch in range(1, TRAIN_EPOCHS + 1):
        order = rng.permutation(x_train.shape[0])
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, order.size, TRAIN_BATCH_SIZE):
            indices = order[start : start + TRAIN_BATCH_SIZE]
            x_batch = torch.from_numpy(x_train[indices]).to(
                device, non_blocking=False
            )
            y_batch = torch.from_numpy(y_train[indices]).to(
                device, non_blocking=False
            )
            tax_batch = tax_train[indices]
            optimizer.zero_grad(set_to_none=True)
            if method == "M1":
                prediction = model(x_batch)
                per_sample = criterion(prediction, y_batch)
                weights = sample_weights[
                    torch.from_numpy(indices)
                ].to(device, non_blocking=False)
                loss = (per_sample * weights).mean()
            else:
                heads = model(x_batch)
                primary_target = torch.from_numpy(
                    (tax_batch == "R").astype(np.float32)
                ).to(device)
                loss_primary = primary_criterion(
                    heads["primary"], primary_target
                )
                h_keep = np.isin(tax_batch, ("R", "H"))
                h_indices = np.flatnonzero(h_keep)
                if h_indices.size:
                    loss_h = h_criterion(
                        heads["r_vs_h"][torch.from_numpy(h_indices).to(device)],
                        torch.from_numpy(
                            (tax_batch[h_indices] == "R").astype(np.float32)
                        ).to(device),
                    )
                else:
                    loss_h = torch.zeros((), device=device)
                i_keep = np.isin(tax_batch, ("R", "I"))
                i_indices = np.flatnonzero(i_keep)
                if i_indices.size:
                    loss_i = i_criterion(
                        heads["r_vs_i"][torch.from_numpy(i_indices).to(device)],
                        torch.from_numpy(
                            (tax_batch[i_indices] == "R").astype(np.float32)
                        ).to(device),
                    )
                else:
                    loss_i = torch.zeros((), device=device)
                loss = loss_primary + 0.25 * (loss_h + loss_i)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(batches, 1),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "seed": int(seed),
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "history": history,
            "protocol_id": PROTOCOL_ID,
        },
        output_path,
    )
    return {
        "method": method,
        "seed": int(seed),
        "model_path": relpath(output_path),
        "model_sha256": sha256_file(output_path),
        "final_train_loss": history[-1]["loss"],
        "train_frames": int(x_train.shape[0]),
        "history": history,
    }


@torch.inference_mode()
def predict_method(
    model: nn.Module,
    method: str,
    features: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 8192,
) -> np.ndarray:
    model.eval()
    result = np.empty(features.shape[0], dtype=np.float64)
    for start in range(0, features.shape[0], chunk_size):
        x = torch.from_numpy(
            np.asarray(features[start : start + chunk_size], dtype=np.float32)
        ).to(device)
        if method == "M1":
            score = model(x)
        else:
            score = torch.sigmoid(model(x)["primary"])
        result[start : start + score.numel()] = (
            score.detach().cpu().numpy().astype(np.float64)
        )
    return result


def uncertainty_score(short_scores: np.ndarray) -> np.ndarray:
    return -np.abs(np.asarray(short_scores, dtype=np.float64) - 0.5)


def calibrate_threshold(
    score: np.ndarray,
    calibration_mask: np.ndarray,
    budget: float,
) -> float:
    values = np.asarray(score, dtype=np.float64)[calibration_mask]
    if values.size == 0:
        raise ValueError("cannot calibrate on an empty split")
    target = int(math.ceil(float(budget) * values.size))
    target = min(max(target, 1), values.size)
    order = np.argsort(-values, kind="stable")
    return float(np.min(values[order[:target]]))


def apply_threshold(
    score: np.ndarray,
    evaluation_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    selected = np.zeros(score.size, dtype=bool)
    selected[evaluation_mask] = score[evaluation_mask] >= float(threshold)
    return selected


def method_metrics(
    data: FrameData,
    selected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool) & np.asarray(mask, dtype=bool)
    labels = data.labels[mask]
    short_pred = data.short_scores[mask] >= DECISION_THRESHOLD
    refined_pred = data.refined_scores[mask] >= DECISION_THRESHOLD
    selected_local = selected[mask]
    adaptive_pred = np.where(selected_local, refined_pred, short_pred)
    value = data.value[mask].astype(np.float64)
    frame_delta = selected_local.astype(np.float64) * value
    short_wrong = short_pred != labels
    refined_wrong = refined_pred != labels
    correction = selected_local & short_wrong & (~refined_wrong)
    harm = selected_local & (~short_wrong) & refined_wrong
    positive_oracle = float(np.maximum(value, 0.0).sum())
    selected_value = float(value[selected_local].sum())
    frames = int(mask.sum())
    return {
        "frames": frames,
        "selected": int(selected_local.sum()),
        "activation_rate": (
            float(selected_local.sum() / frames) if frames else None
        ),
        "utility": float(frame_delta.mean()) if frames else None,
        "f1": (
            float(f1_score(labels, adaptive_pred, zero_division=0))
            if frames
            else None
        ),
        "frame_error": float(np.mean(adaptive_pred != labels)) if frames else None,
        "correction": int(correction.sum()),
        "harm": int(harm.sum()),
        "correction_rate": (
            float(correction.sum() / selected_local.sum())
            if selected_local.sum()
            else None
        ),
        "harm_rate": (
            float(harm.sum() / selected_local.sum())
            if selected_local.sum()
            else None
        ),
        "selected_net_value": selected_value,
        "oracle_positive_value": positive_oracle,
        "value_captured": (
            float(selected_value / positive_oracle)
            if positive_oracle > 0.0
            else None
        ),
    }


def evaluate_replicate_scores(
    replicate_data: FrameData,
    scores: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method in ("M0", "M1", "M2"):
        score = np.asarray(scores[method], dtype=np.float64)
        threshold = calibrate_threshold(
            score,
            replicate_data.dev_mask,
            PRIMARY_BUDGET,
        )
        selected = apply_threshold(
            score,
            replicate_data.test_mask,
            threshold,
        )
        rows[method] = {
            "threshold": float(threshold),
            **method_metrics(
                replicate_data,
                selected,
                replicate_data.test_mask,
            ),
        }
    return rows


def paired_cluster_bootstrap(
    frame_delta: np.ndarray,
    clusters: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    frame_delta = np.asarray(frame_delta, dtype=np.float64).reshape(-1)
    clusters = np.asarray(clusters, dtype=str).reshape(-1)
    if frame_delta.size != clusters.size:
        raise ValueError("delta and cluster arrays must have equal lengths")
    names, inverse = np.unique(clusters, return_inverse=True)
    cluster_counts = np.bincount(inverse, minlength=names.size).astype(np.float64)
    cluster_sums = np.bincount(
        inverse,
        weights=frame_delta,
        minlength=names.size,
    ).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(
        0,
        names.size,
        size=(int(repeats), names.size),
        dtype=np.int64,
    )
    sampled_counts = cluster_counts[samples].sum(axis=1)
    sampled_sums = cluster_sums[samples].sum(axis=1)
    estimates = sampled_sums / np.maximum(sampled_counts, 1.0)
    point = float(frame_delta.mean()) if frame_delta.size else None
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {
        "clusters": int(names.size),
        "repeats": int(repeats),
        "point": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _domain_masks(data: FrameData, mask: np.ndarray) -> dict[str, np.ndarray]:
    unseen = np.isin(data.noise_name, list(UNSEEN_NOISE))
    noisy = data.condition != "clean"
    return {
        "all": np.asarray(mask, dtype=bool),
        "seen": np.asarray(mask, dtype=bool) & noisy & (~unseen),
        "unseen": np.asarray(mask, dtype=bool) & noisy & unseen,
        "clean": np.asarray(mask, dtype=bool) & (~noisy),
    }


def _cell_masks(data: FrameData, mask: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    unseen = np.isin(data.noise_name, list(UNSEEN_NOISE))
    for domain in DOMAIN_ORDER:
        domain_mask = np.asarray(mask, dtype=bool) & (
            unseen if domain == "unseen" else (~unseen)
        )
        for snr in SNR_ORDER:
            result[(domain, snr)] = domain_mask & (data.condition == snr)
    return result


def selection_taxonomy(
    data: FrameData,
    selected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_local = np.asarray(selected, dtype=bool)[mask]
    tax = data.taxonomy[mask]
    selected_count = int(selected_local.sum())
    counts = {
        name: int(np.count_nonzero(selected_local & (tax == name)))
        for name in TAXONOMY_ORDER
    }
    probabilities = {
        name: (
            float(counts[name] / selected_count)
            if selected_count
            else None
        )
        for name in TAXONOMY_ORDER
    }
    if probabilities["R"] is None or probabilities["H"] is None:
        net_opportunity = None
    else:
        net_opportunity = probabilities["R"] - probabilities["H"]
    return {
        "selected_count": selected_count,
        "counts": counts,
        "probabilities": probabilities,
        "net_opportunity": net_opportunity,
    }


def binary_ranking_metrics(
    score: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[float | None, float | None]:
    values = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    negative = np.asarray(negative, dtype=bool)
    if positive.size != values.size or negative.size != values.size:
        raise ValueError("ranking inputs must align")
    keep = positive | negative
    labels = positive[keep].astype(np.int8)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None, None
    ranking = values[keep]
    if not np.all(np.isfinite(ranking)):
        raise ValueError("ranking scores must be finite")
    return (
        float(roc_auc_score(labels, ranking)),
        float(average_precision_score(labels, ranking)),
    )


def build_feature_arrays(
    data: FrameData,
) -> tuple[np.ndarray, RunningNormalizer]:
    features = data.selected_features()
    x3 = np.asarray(features["X3"], dtype=np.float32)
    normalizer = RunningNormalizer.fit(x3, data.train_mask)
    return normalizer.transform(x3), normalizer


def train_primary_models(
    data: FrameData,
    x3: np.ndarray,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], np.ndarray]]:
    x3 = np.asarray(x3, dtype=np.float32)
    model_dir = output_dir / "a2_models"
    manifest: list[dict[str, Any]] = []
    scores: dict[tuple[str, int], np.ndarray] = {}
    for method in ("M1", "M2"):
        for seed in TRAINING_SEEDS:
            output_path = model_dir / f"{method.lower()}_seed{seed}.pt"
            print(f"[A2] training {method} seed {seed}", flush=True)
            row = train_method(
                method,
                x3,
                data.value,
                data.taxonomy,
                data.train_mask,
                seed=seed,
                device=device,
                output_path=output_path,
            )
            payload = torch.load(output_path, map_location=device, weights_only=False)
            model = build_model(method, int(payload["input_dim"])).to(device)
            model.load_state_dict(payload["state_dict"])
            scores[(method, seed)] = predict_method(
                model, method, x3, device=device
            )
            row["protocol_id"] = PROTOCOL_ID
            manifest.append(row)
    return manifest, scores


def evaluate_primary(
    data: FrameData,
    scores: Mapping[tuple[str, int], np.ndarray],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    m0_score = uncertainty_score(data.short_scores)
    budget_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, int, float], np.ndarray] = {}

    method_scores: dict[tuple[str, int], np.ndarray] = {
        ("M0", seed): m0_score for seed in TRAINING_SEEDS
    }
    method_scores.update(scores)
    for method in ("M0", "M1", "M2"):
        for seed in TRAINING_SEEDS:
            score = method_scores[(method, seed)]
            for budget in BUDGETS:
                threshold = calibrate_threshold(
                    score, data.dev_mask, budget
                )
                selected = apply_threshold(
                    score, data.test_mask, threshold
                )
                selections[(method, seed, budget)] = selected
                overall = method_metrics(data, selected, data.test_mask)
                domains = _domain_masks(data, data.test_mask)
                for domain, domain_mask in domains.items():
                    metrics = method_metrics(data, selected, domain_mask)
                    budget_rows.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "budget": float(budget),
                            "domain": domain,
                            "threshold": float(threshold),
                            **metrics,
                        }
                    )
                taxonomy = selection_taxonomy(
                    data, selected, data.test_mask
                )
                test_taxonomy = data.taxonomy[data.test_mask]
                test_score = score[data.test_mask]
                auroc_rest, auprc_rest = binary_ranking_metrics(
                    test_score,
                    test_taxonomy == "R",
                    test_taxonomy != "R",
                )
                auroc_h, auprc_h = binary_ranking_metrics(
                    test_score,
                    test_taxonomy == "R",
                    test_taxonomy == "H",
                )
                auroc_i, auprc_i = binary_ranking_metrics(
                    test_score,
                    test_taxonomy == "R",
                    test_taxonomy == "I",
                )
                taxonomy_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "budget": float(budget),
                        "threshold": float(threshold),
                        "selected": taxonomy["selected_count"],
                        "R": taxonomy["counts"]["R"],
                        "I": taxonomy["counts"]["I"],
                        "H": taxonomy["counts"]["H"],
                        "SS": taxonomy["counts"]["SS"],
                        "P_R_selected": taxonomy["probabilities"]["R"],
                        "P_I_selected": taxonomy["probabilities"]["I"],
                        "P_H_selected": taxonomy["probabilities"]["H"],
                        "P_SS_selected": taxonomy["probabilities"]["SS"],
                        "net_opportunity": taxonomy["net_opportunity"],
                        "auroc_R_vs_rest": auroc_rest,
                        "auprc_R_vs_rest": auprc_rest,
                        "auroc_R_vs_H": auroc_h,
                        "auprc_R_vs_H": auprc_h,
                        "auroc_R_vs_I": auroc_i,
                        "auprc_R_vs_I": auprc_i,
                    }
                )

    for method in ("M1", "M2"):
        for budget_index, budget in enumerate(BUDGETS):
            m0_deltas: list[np.ndarray] = []
            for seed_index, seed in enumerate(TRAINING_SEEDS):
                selected = selections[(method, seed, budget)]
                selected_m0 = selections[("M0", seed, budget)]
                method_delta = selected.astype(np.float64) * data.value
                m0_delta = selected_m0.astype(np.float64) * data.value
                m0_deltas.append(method_delta - m0_delta)
                test_delta = (method_delta - m0_delta)[data.test_mask]
                bootstrap = paired_cluster_bootstrap(
                    test_delta,
                    data.source_key[data.test_mask],
                    repeats=BOOTSTRAP_REPEATS,
                    seed=(
                        BOOTSTRAP_BASE_SEED
                        + 100_003 * budget_index
                        + 1009 * seed_index
                        + (0 if method == "M1" else 500_000)
                    ),
                )
                cluster_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "budget": float(budget),
                        "unit": "source_cluster",
                        **bootstrap,
                    }
                )
            averaged_delta = np.mean(np.stack(m0_deltas, axis=0), axis=0)
            aggregate_bootstrap = paired_cluster_bootstrap(
                averaged_delta[data.test_mask],
                data.source_key[data.test_mask],
                repeats=BOOTSTRAP_REPEATS,
                seed=BOOTSTRAP_BASE_SEED + 10_000 + budget_index,
            )
            cluster_rows.append(
                {
                    "method": method,
                    "seed": "aggregate_5_seeds",
                    "budget": float(budget),
                    "unit": "source_cluster",
                    **aggregate_bootstrap,
                }
            )
            speaker_bootstrap = paired_cluster_bootstrap(
                averaged_delta[data.test_mask],
                data.speaker_ids[data.test_mask],
                repeats=BOOTSTRAP_REPEATS,
                seed=BOOTSTRAP_BASE_SEED + 20_000 + budget_index,
            )
            cluster_rows.append(
                {
                    "method": method,
                    "seed": "aggregate_5_seeds",
                    "budget": float(budget),
                    "unit": "speaker_cluster",
                    **speaker_bootstrap,
                }
            )
            for domain, domain_mask in _domain_masks(data, data.test_mask).items():
                domain_deltas: list[np.ndarray] = []
                for seed in TRAINING_SEEDS:
                    method_delta = (
                        selections[(method, seed, budget)].astype(np.float64)
                        * data.value
                    )
                    baseline_delta = (
                        selections[("M0", seed, budget)].astype(np.float64)
                        * data.value
                    )
                    domain_deltas.append(method_delta - baseline_delta)
                averaged_domain_delta = np.mean(
                    np.stack(domain_deltas, axis=0), axis=0
                )
                domain_bootstrap = paired_cluster_bootstrap(
                    averaged_domain_delta[domain_mask],
                    data.source_key[domain_mask],
                    repeats=BOOTSTRAP_REPEATS,
                    seed=(
                        BOOTSTRAP_BASE_SEED
                        + 30_000
                        + 100 * budget_index
                        + DOMAIN_ORDER.index(domain)
                        + (0 if method == "M1" else 500_000)
                    ),
                )
                cluster_rows.append(
                    {
                        "method": method,
                        "seed": "aggregate_5_seeds",
                        "budget": float(budget),
                        "unit": "source_cluster",
                        "domain": domain,
                        **domain_bootstrap,
                    }
                )

    for method in ("M0", "M1", "M2"):
        aggregate_selected = np.mean(
            np.stack(
                [
                    selections[(method, seed, PRIMARY_BUDGET)]
                    for seed in TRAINING_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
        for (domain, snr), cell_mask in _cell_masks(data, data.test_mask).items():
            method_cell = method_metrics(data, aggregate_selected, cell_mask)
            m0_cell = method_metrics(
                data,
                np.mean(
                    np.stack(
                        [
                            selections[("M0", seed, PRIMARY_BUDGET)]
                            for seed in TRAINING_SEEDS
                        ],
                        axis=0,
                    ),
                    axis=0,
                ),
                cell_mask,
            )
            delta = (
                method_cell["utility"] - m0_cell["utility"]
                if method_cell["utility"] is not None
                and m0_cell["utility"] is not None
                else None
            )
            cell_rows.append(
                {
                    "method": method,
                    "budget": PRIMARY_BUDGET,
                    "domain": domain,
                    "snr": snr,
                    "frames": method_cell["frames"],
                    "method_utility": method_cell["utility"],
                    "uncertainty_utility": m0_cell["utility"],
                    "delta_utility": delta,
                    "method_activation": method_cell["activation_rate"],
                    "method_f1": method_cell["f1"],
                    "evidence_role": (
                        "failure_analysis_not_significance_selection"
                    ),
                    "catastrophic": (
                        method_cell["frames"] >= 1000
                        and delta is not None
                        and delta <= CATastrophic_CELL_DELTA
                    ),
                }
            )
    return {
        "budget_rows": budget_rows,
        "cluster_rows": cluster_rows,
        "taxonomy_rows": taxonomy_rows,
        "cell_rows": cell_rows,
        "selections": selections,
        "m0_score": m0_score,
        "method_scores": method_scores,
    }


def mechanism_diagnostics(
    data: FrameData,
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selections = evaluation["selections"]
    method_scores = evaluation["method_scores"]
    methods = ("M0", "M1", "M2")

    def averaged_scores(method: str) -> np.ndarray:
        return np.mean(
            np.stack(
                [
                    np.asarray(method_scores[(method, seed)], dtype=np.float64)
                    for seed in TRAINING_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )

    def selection_probability(method: str) -> np.ndarray:
        return np.mean(
            np.stack(
                [
                    selections[(method, seed, PRIMARY_BUDGET)].astype(
                        np.float64
                    )
                    for seed in TRAINING_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )

    score_by_method = {
        method: averaged_scores(method) for method in methods
    }
    selection_by_method = {
        method: selection_probability(method) for method in methods
    }
    rows: list[dict[str, Any]] = []
    test_mask = data.test_mask

    for method in methods:
        score = score_by_method[method]
        for tax_name in TAXONOMY_ORDER:
            mask = test_mask & (data.taxonomy == tax_name)
            values = score[mask]
            if values.size == 0:
                continue
            rows.append(
                {
                    "diagnostic_type": "score_distribution",
                    "dimension": "taxonomy",
                    "stratum": tax_name,
                    "method": method,
                    "frames": int(mask.sum()),
                    "score_mean": float(np.mean(values)),
                    "score_q25": float(np.quantile(values, 0.25)),
                    "score_median": float(np.median(values)),
                    "score_q75": float(np.quantile(values, 0.75)),
                }
            )

    segments = utterance_segments(
        data.source_key,
        data.noise_name,
        data.condition,
    )
    event = build_event_structure(
        data.labels,
        data.short_scores,
        segments,
        decision_threshold=DECISION_THRESHOLD,
        uncertainty_threshold=AX_UNCERTAINTY_THRESHOLD,
    )
    dimensions = (
        "recent_uncertainty_duration",
        "recent_posterior_transition",
        "onset_distance",
        "offset_distance",
    )
    for dimension in dimensions:
        values = event[dimension]
        for stratum in sorted(set(values[test_mask].tolist())):
            stratum_mask = test_mask & (values == stratum)
            if not np.any(stratum_mask):
                continue
            tax = data.taxonomy[stratum_mask]
            value = data.value[stratum_mask].astype(np.float64)
            frames = int(stratum_mask.sum())
            for method in methods:
                selection = selection_by_method[method][stratum_mask]
                utility = float(np.mean(selection * value))
                auroc, _ = binary_ranking_metrics(
                    score_by_method[method][stratum_mask],
                    tax == "R",
                    tax != "R",
                )
                selected_total = float(selection.sum())
                probabilities = {
                    name: (
                        float(
                            np.sum(selection * (tax == name))
                            / selected_total
                        )
                        if selected_total > 0.0
                        else None
                    )
                    for name in TAXONOMY_ORDER
                }
                rows.append(
                    {
                        "diagnostic_type": "stratum",
                        "dimension": dimension,
                        "stratum": str(stratum),
                        "method": method,
                        "frames": frames,
                        "activation_rate": float(np.mean(selection)),
                        "utility": utility,
                        "availability_mean_positive_value": float(
                            np.mean(np.maximum(value, 0.0))
                        ),
                        "availability_positive_rate": float(
                            np.mean(value > 0.0)
                        ),
                        "observability_auroc_R_vs_rest": auroc,
                        "actionability_utility": utility,
                        "P_R": probabilities["R"],
                        "P_I": probabilities["I"],
                        "P_H": probabilities["H"],
                        "P_SS": probabilities["SS"],
                    }
                )

    domain_masks = _domain_masks(data, test_mask)
    for domain, domain_mask in domain_masks.items():
        tax = data.taxonomy[domain_mask]
        value = data.value[domain_mask].astype(np.float64)
        frames = int(domain_mask.sum())
        if frames == 0:
            continue
        for method in methods:
            selection = selection_by_method[method][domain_mask]
            utility = float(np.mean(selection * value))
            auroc, auprc = binary_ranking_metrics(
                score_by_method[method][domain_mask],
                tax == "R",
                tax != "R",
            )
            rows.append(
                {
                    "diagnostic_type": "ae0_availability_observability",
                    "dimension": "domain",
                    "stratum": domain,
                    "method": method,
                    "frames": frames,
                    "activation_rate": float(np.mean(selection)),
                    "availability_mean_positive_value": float(
                        np.mean(np.maximum(value, 0.0))
                    ),
                    "availability_positive_rate": float(
                        np.mean(value > 0.0)
                    ),
                    "observability_auroc_R_vs_rest": auroc,
                    "observability_auprc_R_vs_rest": auprc,
                    "actionability_utility": utility,
                }
            )
    for snr in SNR_ORDER:
        snr_mask = test_mask & (data.condition == snr)
        frames = int(snr_mask.sum())
        if frames == 0:
            continue
        tax = data.taxonomy[snr_mask]
        value = data.value[snr_mask].astype(np.float64)
        for method in methods:
            selection = selection_by_method[method][snr_mask]
            utility = float(np.mean(selection * value))
            auroc, auprc = binary_ranking_metrics(
                score_by_method[method][snr_mask],
                tax == "R",
                tax != "R",
            )
            rows.append(
                {
                    "diagnostic_type": "ae0_availability_observability",
                    "dimension": "snr",
                    "stratum": snr,
                    "method": method,
                    "frames": frames,
                    "activation_rate": float(np.mean(selection)),
                    "availability_mean_positive_value": float(
                        np.mean(np.maximum(value, 0.0))
                    ),
                    "availability_positive_rate": float(
                        np.mean(value > 0.0)
                    ),
                    "observability_auroc_R_vs_rest": auroc,
                    "observability_auprc_R_vs_rest": auprc,
                    "actionability_utility": utility,
                }
            )
    for (domain, snr), cell_mask in _cell_masks(data, test_mask).items():
        frames = int(cell_mask.sum())
        if frames == 0:
            continue
        tax = data.taxonomy[cell_mask]
        value = data.value[cell_mask].astype(np.float64)
        for method in methods:
            selection = selection_by_method[method][cell_mask]
            utility = float(np.mean(selection * value))
            auroc, auprc = binary_ranking_metrics(
                score_by_method[method][cell_mask],
                tax == "R",
                tax != "R",
            )
            rows.append(
                {
                    "diagnostic_type": "ae0_availability_observability",
                    "dimension": "domain_x_snr",
                    "stratum": f"{domain}|{snr}",
                    "method": method,
                    "frames": frames,
                    "activation_rate": float(np.mean(selection)),
                    "availability_mean_positive_value": float(
                        np.mean(np.maximum(value, 0.0))
                    ),
                    "availability_positive_rate": float(
                        np.mean(value > 0.0)
                    ),
                    "observability_auroc_R_vs_rest": auroc,
                    "observability_auprc_R_vs_rest": auprc,
                    "actionability_utility": utility,
                }
            )
    return rows


def benchmark_router_overhead(
    model: nn.Module | None,
    method: str,
    input_dim: int,
    *,
    device: torch.device,
    warmup: int = 30,
    repeats: int = 200,
) -> float:
    cpu = torch.device("cpu")
    rng = np.random.default_rng(20260921)
    batch = torch.from_numpy(
        rng.standard_normal((4096, input_dim), dtype=np.float32)
    )
    if model is None:
        def forward() -> torch.Tensor:
            return -torch.abs(batch[:, 0] - 0.5)
    else:
        model = model.to(cpu)
        model.eval()

        def forward() -> torch.Tensor:
            with torch.inference_mode():
                if method == "M1":
                    return model(batch)
                return model(batch)["primary"]

    for _ in range(warmup):
        forward()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            forward()
    elapsed = time.perf_counter() - start
    return float(1000.0 * elapsed / (repeats * batch.shape[0]))


def compute_estimates(
    data: FrameData,
    scores: Mapping[tuple[str, int], np.ndarray],
    *,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    a11_path = (
        REPO_ROOT
        / "results"
        / "difficulty_adaptive_context"
        / "a11_conditional_compute"
        / "a11_conditional_compute.json"
    )
    a11 = read_json(a11_path)
    rows_by_name = {str(row["name"]): row for row in a11["rows"]}
    c_short_core = float(rows_by_name["short_core"]["mean_ms_per_frame"])
    c_always = float(rows_by_name["fixed_rf384"]["mean_ms_per_frame"])
    c_refine = float(
        rows_by_name["fixed_rf384"]["refinement_kernel"]["mean_ms_per_frame"]
    )
    m0_overhead = float(
        rows_by_name["adaptive_5pct"]["router_overhead_ms_per_frame"]
    )
    scheduling = float(
        rows_by_name["adaptive_5pct"]["scheduling_overhead_ms_per_frame"]
    )
    results: list[dict[str, Any]] = []
    for method in ("M0", "M1", "M2"):
        if method == "M0":
            overhead = m0_overhead
        else:
            model_hashes = sorted(
                path
                for path in (output_dir / "a2_models").glob(
                    f"{method.lower()}_seed*.pt"
                )
            )
            if not model_hashes:
                raise FileNotFoundError(f"no trained models for {method}")
            payload = torch.load(
                model_hashes[0], map_location="cpu", weights_only=False
            )
            model = build_model(method, int(payload["input_dim"]))
            model.load_state_dict(payload["state_dict"])
            overhead = benchmark_router_overhead(
                model, method, int(payload["input_dim"]), device=device
            )
        for budget in BUDGETS:
            activations = []
            for seed in TRAINING_SEEDS:
                score = (
                    uncertainty_score(data.short_scores)
                    if method == "M0"
                    else scores[(method, seed)]
                )
                threshold = calibrate_threshold(
                    score, data.dev_mask, budget
                )
                selected = apply_threshold(score, data.test_mask, threshold)
                activations.append(float(selected[data.test_mask].mean()))
            activation = float(np.mean(activations))
            estimated = (
                c_short_core
                + overhead
                + activation * c_refine
                + scheduling
            )
            advantage = c_always - estimated
            results.append(
                {
                    "method": method,
                    "budget": float(budget),
                    "activation": activation,
                    "short_mean_ms_per_frame": c_short_core,
                    "router_overhead_ms_per_frame": overhead,
                    "refinement_kernel_ms_per_frame": c_refine,
                    "scheduling_overhead_ms_per_frame": scheduling,
                    "estimated_mean_ms_per_frame": estimated,
                    "always_refine_mean_ms_per_frame": c_always,
                    "conditional_advantage_ms_per_frame": advantage,
                    "conditional_advantage_fraction": advantage / c_always,
                    "measurement_role": (
                        "engineering diagnostic; A11 kernels plus frozen "
                        "router-forward measurement"
                    ),
                }
            )
    return results


def assess_primary(
    *,
    budget_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def budget_point(method: str) -> float:
        values = [
            float(row["utility"])
            for row in budget_rows
            if row["method"] == method
            and float(row["budget"]) == PRIMARY_BUDGET
            and row["domain"] == "all"
            and row["utility"] is not None
        ]
        return float(np.mean(values))

    def domain_point(method: str, domain: str) -> float:
        values = [
            float(row["utility"])
            for row in budget_rows
            if row["method"] == method
            and float(row["budget"]) == PRIMARY_BUDGET
            and row["domain"] == domain
            and row["utility"] is not None
        ]
        return float(np.mean(values))

    m0_point = budget_point("M0")
    candidates: dict[str, dict[str, Any]] = {}
    for method in ("M1", "M2"):
        aggregate = next(
            row
            for row in cluster_rows
            if row["method"] == method
            and row["seed"] == "aggregate_5_seeds"
            and float(row["budget"]) == PRIMARY_BUDGET
            and row["unit"] == "source_cluster"
            and "domain" not in row
        )
        per_seed = [
            float(row["utility"])
            for row in budget_rows
            if row["method"] == method
            and float(row["budget"]) == PRIMARY_BUDGET
            and row["domain"] == "all"
        ]
        m0_per_seed = [
            float(row["utility"])
            for row in budget_rows
            if row["method"] == "M0"
            and float(row["budget"]) == PRIMARY_BUDGET
            and row["domain"] == "all"
        ]
        seed_deltas = [
            learned - baseline
            for learned, baseline in zip(per_seed, m0_per_seed)
        ]
        seed_positive = int(sum(value > 0.0 for value in seed_deltas))
        seen_delta = domain_point(method, "seen") - domain_point("M0", "seen")
        unseen_delta = domain_point(method, "unseen") - domain_point("M0", "unseen")
        catastrophic = [
            row
            for row in cell_rows
            if row["method"] == method and bool(row["catastrophic"])
        ]
        primary_compute = next(
            row
            for row in compute_rows
            if row["method"] == method
            and float(row["budget"]) == PRIMARY_BUDGET
        )
        activation_rows = [
            row
            for row in compute_rows
            if row["method"] == method
        ]
        max_activation = max(
            float(row["activation"]) for row in activation_rows
        )
        primary_activation = float(primary_compute["activation"])
        g1 = (
            float(aggregate["point"]) > 0.0
            and float(aggregate["ci95_low"]) > 0.0
        )
        g2 = seed_positive >= 4
        g3 = unseen_delta >= 0.0 and not catastrophic
        g4 = (
            PRIMARY_ACTIVATION_LOWER
            <= primary_activation
            <= PRIMARY_ACTIVATION_UPPER
            and max_activation <= MAX_UNCONTROLLED_ACTIVATION
        )
        g5 = (
            float(primary_compute["conditional_advantage_ms_per_frame"])
            >= COMPUTE_ADVANTAGE_MIN_MS
            and float(primary_compute["conditional_advantage_fraction"])
            >= COMPUTE_ADVANTAGE_MIN_FRACTION
        )
        candidates[method] = {
            "method": method,
            "point_delta_utility": float(aggregate["point"]),
            "ci95_low": float(aggregate["ci95_low"]),
            "ci95_high": float(aggregate["ci95_high"]),
            "seed_positive": int(seed_positive),
            "seed_deltas": seed_deltas,
            "seen_delta": float(seen_delta),
            "unseen_delta": float(unseen_delta),
            "catastrophic_cells": catastrophic,
            "primary_activation": primary_activation,
            "max_activation": max_activation,
            "compute_advantage_ms": float(
                primary_compute["conditional_advantage_ms_per_frame"]
            ),
            "compute_advantage_fraction": float(
                primary_compute["conditional_advantage_fraction"]
            ),
            "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5},
            "passes_all": bool(g1 and g2 and g3 and g4 and g5),
        }
    passing_methods = [
        method for method in ("M1", "M2") if candidates[method]["passes_all"]
    ]
    candidate_pool = passing_methods or ["M1", "M2"]
    primary_method = sorted(
        candidate_pool,
        key=lambda method: (
            -float(candidates[method]["point_delta_utility"]),
            0 if method == "M1" else 1,
        ),
    )[0]
    selected = candidates[primary_method]
    if selected["passes_all"]:
        status = "METHOD_GO"
        failure_mode = None
    elif (
        selected["point_delta_utility"] > 0.0
        and selected["max_activation"] <= MAX_UNCONTROLLED_ACTIVATION
        and selected["compute_advantage_ms"] > 0.0
        and selected["compute_advantage_fraction"] > 0.0
    ):
        status = "CONDITIONAL_GO"
        failed_gates = sorted(
            gate for gate, passed in selected["gates"].items() if not passed
        )
        failure_mode = (
            "POSITIVE_POINT_WITH_INCOMPLETE_GATES_"
            + "_".join(failed_gates)
        )
    elif (
        selected["max_activation"] > MAX_UNCONTROLLED_ACTIVATION
        or (
            selected["compute_advantage_ms"] <= 0.0
            or selected["compute_advantage_fraction"] <= 0.0
        )
    ):
        status = "METHOD_NO_GO"
        failure_mode = "ACTIVATION_OR_COMPUTE_CONTROL_FAILURE"
    else:
        status = "METHOD_NO_GO"
        failure_mode = "NO_LEARNED_METHOD_EXCEEDS_UNCERTAINTY"
    return {
        "status": status,
        "primary_method": primary_method,
        "primary_budget": PRIMARY_BUDGET,
        "m0_utility": m0_point,
        "candidates": candidates,
        "failure_mode": failure_mode,
    }


@torch.inference_mode()
def extract_independent_hidden(
    *,
    checkpoint_path: Path,
    output_path: Path,
    ax_protocol_path: Path,
    device: torch.device,
    progress_every: int,
) -> str:
    if output_path.is_file():
        return sha256_file(output_path)
    ax = read_json(ax_protocol_path)
    data = ax["data"]
    model, _, _ = load_adaptive_model(checkpoint_path, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    items = build_evaluation_items(
        resolve_path(data["manifest"]["path"]),
        generated_root=resolve_path(data["generated_root"]),
        label_root=resolve_path(data["label_root"]),
        librispeech_root=resolve_path(data["librispeech_root"]),
        row_sample=int(data["row_sample"]),
        seed=int(data["manifest_seed"]),
        include_clean=bool(data["include_clean"]),
    )
    valid_start = int(data["valid_start_frame"])
    score_chunk_frames = int(data["score_chunk_frames"])
    hidden_parts: list[np.ndarray] = []
    for index, item in enumerate(items):
        waveform = read_int16_audio(item.audio_path)
        sample_labels = np.load(item.label_path)
        if sample_labels.size != waveform.size:
            raise ValueError(f"label/audio mismatch: {item.audio_path}")
        n_frames = waveform.size // FRAME_HOP + 1
        if n_frames <= valid_start:
            continue
        tensor = torch.from_numpy(
            waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        features = frontend(tensor.to(device, non_blocking=True))
        encoded, _ = encode_short_stream(
            model, features, chunk_frames=score_chunk_frames
        )
        hidden_parts.append(
            encoded[0, :, valid_start:]
            .transpose(0, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if progress_every > 0 and (
            (index + 1) % progress_every == 0 or index + 1 == len(items)
        ):
            print(
                f"[A2 independent] {checkpoint_path.parent.parent.name}: "
                f"{index + 1}/{len(items)}",
                flush=True,
            )
    hidden = np.concatenate(hidden_parts, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, hidden=hidden)
    return sha256_file(output_path)


def independent_replication(
    *,
    data: FrameData,
    output_dir: Path,
    ax_protocol_path: Path,
    primary_method: str,
    primary_delta_utility: float,
    device: torch.device,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hidden_dir = output_dir / "a2_independent_hidden"
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for seed in INDEPENDENT_SEEDS:
        prediction_path = (
            REPO_ROOT
            / "results"
            / "difficulty_adaptive_context"
            / "ae_mechanism"
            / "ae4_replication_predictions"
            / f"seed{seed}"
            / "frame_predictions.npz"
        )
        checkpoint_path = (
            REPO_ROOT
            / "results"
            / "difficulty_adaptive_context"
            / "ae_mechanism"
            / "ae4_replication_training"
            / f"seed{seed}"
            / "refiner"
            / "best.pt"
        )
        hidden_path = hidden_dir / f"seed{seed}.npz"
        hidden_hash = extract_independent_hidden(
            checkpoint_path=checkpoint_path,
            output_path=hidden_path,
            ax_protocol_path=ax_protocol_path,
            device=device,
            progress_every=progress_every,
        )
        with np.load(prediction_path, allow_pickle=False) as payload:
            short = np.asarray(payload["short_scores"], dtype=np.float64)
            refined = np.asarray(payload["refined_scores"], dtype=np.float64)
            labels = np.asarray(payload["labels"], dtype=np.int64)
        if not np.array_equal(labels, data.labels):
            raise ValueError(f"independent replicate {seed} label mismatch")
        value, tax = taxonomy_from_scores(labels, short, refined)
        with np.load(hidden_path, allow_pickle=False) as payload:
            hidden = np.asarray(payload["hidden"], dtype=np.float32)
        replicate_data = replace(
            data,
            short_scores=short,
            refined_scores=refined,
            value=value,
            taxonomy=tax,
            hidden=hidden,
        )
        segments = utterance_segments(
            data.source_key, data.noise_name, data.condition
        )
        x3 = feature_blocks(
            short,
            hidden,
            data.mfcc,
            segments,
            window_frames=AX_WINDOW_FRAMES,
        )["X3"].astype(np.float32)
        normalizer = RunningNormalizer.fit(x3, data.train_mask)
        x3 = normalizer.transform(x3)
        train_manifest: dict[str, Any] = {}
        scores: dict[str, np.ndarray] = {
            "M0": uncertainty_score(short),
        }
        for method in ("M1", "M2"):
            model_path = (
                output_dir
                / "a2_independent_models"
                / f"seed{seed}_{method.lower()}.pt"
            )
            train_manifest[method] = train_method(
                method,
                x3,
                value,
                tax,
                data.train_mask,
                seed=seed,
                device=device,
                output_path=model_path,
            )
            training_rows.append(
                {
                    "replicate_seed": int(seed),
                    **train_manifest[method],
                }
            )
            payload = torch.load(
                model_path, map_location=device, weights_only=False
            )
            model = build_model(method, int(payload["input_dim"])).to(device)
            model.load_state_dict(payload["state_dict"])
            scores[method] = predict_method(model, method, x3, device=device)
        replicate_metrics = evaluate_replicate_scores(replicate_data, scores)
        for method in ("M0", "M1", "M2"):
            rows.append(
                {
                    "replicate_seed": int(seed),
                    "method": method,
                    "prediction_sha256": sha256_file(prediction_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "hidden_sha256": hidden_hash,
                    **replicate_metrics[method],
                }
            )
    lookup = {
        (int(row["replicate_seed"]), str(row["method"])): row for row in rows
    }
    deltas = []
    for seed in INDEPENDENT_SEEDS:
        learned = lookup[(seed, primary_method)]
        baseline = lookup[(seed, "M0")]
        deltas.append(
            float(learned["utility"]) - float(baseline["utility"])
        )
    summary = {
        "primary_method": primary_method,
        "replicate_delta_utilities": deltas,
        "positive_replicates": int(sum(value > 0.0 for value in deltas)),
        "mean_delta_utility": float(np.mean(deltas)),
        "sign_matches_primary": bool(
            np.sign(np.mean(deltas)) == np.sign(float(primary_delta_utility))
        ),
        "primary_delta_utility": float(primary_delta_utility),
    }
    return rows, training_rows, summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_primary_figures(
    *,
    output_dir: Path,
    primary_method: str,
    budget_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
    score_distributions: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "a2_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = ("M0", "M1", "M2")
    colors = {"M0": "#4c78a8", "M1": "#f58518", "M2": "#54a24b"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        values = []
        for budget in BUDGETS:
            utilities = [
                float(row["utility"])
                for row in budget_rows
                if row["method"] == method
                and float(row["budget"]) == budget
                and row["domain"] == "all"
                and row["utility"] is not None
            ]
            values.append(float(np.mean(utilities)))
        ax.plot(
            [100.0 * budget for budget in BUDGETS],
            values,
            marker="o",
            color=colors[method],
            label=method,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Budget (%)")
    ax.set_ylabel("OOD utility")
    ax.set_title("A-v2 utility by budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_budget_utility.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    offsets = {"M1": -0.06, "M2": 0.06}
    for method in ("M1", "M2"):
        for index, budget in enumerate(BUDGETS):
            row = next(
                item
                for item in cluster_rows
                if item["method"] == method
                and item["seed"] == "aggregate_5_seeds"
                and float(item["budget"]) == budget
                and item["unit"] == "source_cluster"
                and "domain" not in item
            )
            point = float(row["point"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            position = 100.0 * budget + offsets[method]
            ax.errorbar(
                position,
                point,
                yerr=np.asarray([[point - low], [high - point]]),
                fmt="o",
                color=colors[method],
                capsize=4,
                label=method if index == 0 else None,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Budget (%)")
    ax.set_ylabel("Delta utility vs M0")
    ax.set_title("Paired source-cluster bootstrap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_delta_utility_ci.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        values = [
            float(row["activation"])
            for row in compute_rows
            if row["method"] == method
        ]
        ax.plot(
            [100.0 * budget for budget in BUDGETS],
            values,
            marker="o",
            color=colors[method],
            label=method,
        )
    ax.axhline(0.15, color="#d62728", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Budget (%)")
    ax.set_ylabel("Observed activation")
    ax.set_title("A-v2 activation control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_activation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.24
    positions = np.arange(len(methods))
    for tax_index, tax_name in enumerate(TAXONOMY_ORDER):
        values = []
        for method in methods:
            matching = [
                float(row[f"P_{tax_name}_selected"])
                for row in taxonomy_rows
                if row["method"] == method
                and float(row["budget"]) == PRIMARY_BUDGET
                and row[f"P_{tax_name}_selected"] is not None
            ]
            values.append(float(np.mean(matching)))
        ax.bar(
            positions + (tax_index - 1.5) * width,
            values,
            width=width,
            label=tax_name,
        )
    ax.set_xticks(positions, methods)
    ax.set_ylabel("P(taxonomy | selected)")
    ax.set_title("Top-5% selection composition")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_selection_taxonomy.png", dpi=180)
    plt.close(fig)

    cell_matrix = np.full(
        (len(DOMAIN_ORDER), len(SNR_ORDER)), np.nan, dtype=np.float64
    )
    for row in cell_rows:
        if row["method"] != primary_method:
            continue
        if row["delta_utility"] is None:
            continue
        cell_matrix[
            DOMAIN_ORDER.index(str(row["domain"])),
            SNR_ORDER.index(str(row["snr"])),
        ] = float(row["delta_utility"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    image = ax.imshow(cell_matrix, cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    ax.set_xticks(np.arange(len(SNR_ORDER)), SNR_ORDER)
    ax.set_yticks(np.arange(len(DOMAIN_ORDER)), DOMAIN_ORDER)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Domain")
    ax.set_title(f"{primary_method} minus M0 utility at 5%")
    fig.colorbar(image, ax=ax, label="Delta utility")
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_cell_delta_utility.png", dpi=180)
    plt.close(fig)

    distribution_methods = tuple(score_distributions)
    fig, axes = plt.subplots(
        1,
        len(distribution_methods),
        figsize=(5.5 * len(distribution_methods), 4.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for ax, method in zip(axes, distribution_methods):
        distributions = score_distributions[method]
        for tax_name in TAXONOMY_ORDER:
            ax.hist(
                distributions[tax_name],
                bins=80,
                density=True,
                histtype="step",
                linewidth=1.4,
                label=tax_name,
            )
        ax.set_title(method)
        ax.set_xlabel("Router score")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("A-v2 score distributions by frozen taxonomy")
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_score_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        values = [
            float(row["estimated_mean_ms_per_frame"])
            for row in compute_rows
            if row["method"] == method
        ]
        ax.plot(
            [100.0 * budget for budget in BUDGETS],
            values,
            marker="o",
            color=colors[method],
            label=method,
        )
    always = float(compute_rows[0]["always_refine_mean_ms_per_frame"])
    ax.axhline(always, color="black", linestyle="--", label="AlwaysRefine")
    ax.set_xlabel("Budget (%)")
    ax.set_ylabel("Estimated ms/frame")
    ax.set_title("A-v2 compute-aware diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "a2_compute_diagnostic.png", dpi=180)
    plt.close(fig)


def write_final_report(
    *,
    output_dir: Path,
    assessment: Mapping[str, Any],
    budget_rows: Sequence[Mapping[str, Any]],
    compute_rows: Sequence[Mapping[str, Any]],
    independent_summary: Mapping[str, Any],
) -> None:
    selected = assessment["candidates"][assessment["primary_method"]]
    lines = [
        "# A-v2 Method Study",
        "",
        "## Frozen Protocol",
        "",
        f"- Status: **{assessment['status']}**",
        f"- Primary method: **{assessment['primary_method']}**",
        f"- Primary budget: **{100.0 * float(assessment['primary_budget']):.0f}%**",
        f"- Delta utility vs uncertainty: "
        f"**{_fmt(selected['point_delta_utility'])}**",
        f"- Source-cluster 95% CI: "
        f"**[{_fmt(selected['ci95_low'])}, {_fmt(selected['ci95_high'])}]**",
        f"- Seed-positive count: **{selected['seed_positive']}/5**",
        f"- Unseen delta utility: **{_fmt(selected['unseen_delta'])}**",
        f"- Primary activation: "
        f"**{100.0 * float(selected['primary_activation']):.3f}%**",
        f"- Maximum activation: "
        f"**{100.0 * float(selected['max_activation']):.3f}%**",
        "",
        "## Primary Gates",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
    ]
    for gate, passed in selected["gates"].items():
        lines.append(f"| {gate} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Budget Results",
            "",
            "| Method | Budget | Activation | Utility |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method in ("M0", "M1", "M2"):
        for budget in BUDGETS:
            matching = [
                row
                for row in budget_rows
                if row["method"] == method
                and float(row["budget"]) == budget
                and row["domain"] == "all"
            ]
            utilities = [float(row["utility"]) for row in matching]
            activations = [float(row["activation_rate"]) for row in matching]
            lines.append(
                f"| {method} | {100.0 * budget:.0f}% | "
                f"{100.0 * float(np.mean(activations)):.3f}% | "
                f"{_fmt(float(np.mean(utilities)))} |"
            )
    lines.extend(
        [
            "",
            "## Compute Diagnostic",
            "",
            "| Method | Budget | Estimated ms/frame | AlwaysRefine advantage |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in compute_rows:
        lines.append(
            f"| {row['method']} | {100.0 * float(row['budget']):.0f}% | "
            f"{_fmt(row['estimated_mean_ms_per_frame'])} | "
            f"{_fmt(row['conditional_advantage_ms_per_frame'])} ms "
            f"({100.0 * float(row['conditional_advantage_fraction']):.2f}%) |"
        )
    lines.extend(
        [
            "",
            "## Independent-Backbone Sensitivity",
            "",
            f"- Primary method replicated: "
            f"`{independent_summary['primary_method']}`",
            f"- Positive independent replicates: "
            f"`{independent_summary['positive_replicates']}/5`",
            f"- Mean delta utility: "
            f"`{_fmt(independent_summary['mean_delta_utility'])}`",
            "",
            "## Interpretation Boundary",
            "",
            "- OBSERVATION: The reported utility uses actual Short-to-RF384 "
            "selection outcomes, not classifier accuracy alone.",
            "- SUPPORTED INTERPRETATION: Differences are limited to the frozen "
            "TRAIN/DEV/INTERNAL_TEST split and the tested observables.",
            "- UNSUPPORTED CLAIM: This experiment does not establish "
            "deployment-wide generalization or justify further router search.",
            "",
            "`NEW_FINAL_OOD_TOUCHED=false`",
            "",
            "`NEXT_SEARCH_AUTHORIZED=false`",
            "",
        ]
    )
    (output_dir / "a2_final_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8f}"


def run_protocol(
    *,
    protocol: FrozenProtocol,
    output_dir: Path,
    device_name: str,
    progress_every: int,
    skip_independent: bool,
) -> dict[str, Any]:
    payload = protocol.payload
    verify_protocol_locks(payload)
    frame_bundle = resolve_path(payload["data"]["frame_bundle"]["path"])
    feature_cache = resolve_path(payload["data"]["feature_cache"]["path"])
    data = load_frame_data(
        frame_bundle_path=frame_bundle,
        feature_path=feature_cache,
    )
    normalized_x3, normalizer = build_feature_arrays(data)
    np.savez_compressed(
        output_dir / "a2_train_normalizer.npz",
        mean=normalizer.mean,
        scale=normalizer.scale,
    )
    device = resolve_device(device_name)
    print(f"[A2] device={device}", flush=True)
    training_manifest, scores = train_primary_models(
        data,
        normalized_x3,
        device=device,
        output_dir=output_dir,
    )
    evaluation = evaluate_primary(data, scores, output_dir=output_dir)
    mechanism_rows = mechanism_diagnostics(data, evaluation)
    compute_rows = compute_estimates(
        data,
        scores,
        output_dir=output_dir,
        device=device,
    )
    assessment = assess_primary(
        budget_rows=evaluation["budget_rows"],
        cluster_rows=evaluation["cluster_rows"],
        cell_rows=evaluation["cell_rows"],
        compute_rows=compute_rows,
    )
    selected = assessment["candidates"][assessment["primary_method"]]
    if skip_independent:
        independent_rows: list[dict[str, Any]] = []
        independent_training_rows: list[dict[str, Any]] = []
        independent_summary = {
            "primary_method": assessment["primary_method"],
            "replicate_delta_utilities": [],
            "positive_replicates": 0,
            "mean_delta_utility": None,
            "sign_matches_primary": None,
            "skipped": True,
        }
    else:
        (
            independent_rows,
            independent_training_rows,
            independent_summary,
        ) = independent_replication(
            data=data,
            output_dir=output_dir,
            ax_protocol_path=DEFAULT_AX_PROTOCOL,
            primary_method=str(assessment["primary_method"]),
            primary_delta_utility=float(selected["point_delta_utility"]),
            device=device,
            progress_every=progress_every,
        )
    write_csv(
        output_dir / "a2_all_runs.csv",
        [*training_manifest, *independent_training_rows],
    )
    write_csv(output_dir / "a2_budget_results.csv", evaluation["budget_rows"])
    write_csv(output_dir / "a2_cluster_bootstrap.csv", evaluation["cluster_rows"])
    write_csv(output_dir / "a2_cell_results.csv", evaluation["cell_rows"])
    write_csv(
        output_dir / "a2_selection_taxonomy.csv",
        evaluation["taxonomy_rows"],
    )
    write_csv(output_dir / "a2_compute_results.csv", compute_rows)
    write_csv(
        output_dir / "a2_independent_replication.csv",
        independent_rows,
    )
    write_csv(
        output_dir / "a2_mechanism_diagnostics.csv",
        mechanism_rows,
    )
    write_json(
        output_dir / "a2_training_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol.sha256,
            "seeds": list(TRAINING_SEEDS),
            "training": training_manifest,
            "independent_sensitivity": independent_training_rows,
            "normalizer_path": relpath(
                output_dir / "a2_train_normalizer.npz"
            ),
            "normalizer_sha256": sha256_file(
                output_dir / "a2_train_normalizer.npz"
            ),
        },
    )
    score_distributions = {
        method: {
            tax: evaluation["method_scores"][(method, TRAINING_SEEDS[0])][
                data.test_mask
            ][
                data.taxonomy[data.test_mask] == tax
            ]
            for tax in TAXONOMY_ORDER
        }
        for method in ("M0", "M1", "M2")
    }
    write_primary_figures(
        output_dir=output_dir,
        primary_method=str(assessment["primary_method"]),
        budget_rows=evaluation["budget_rows"],
        cluster_rows=evaluation["cluster_rows"],
        taxonomy_rows=evaluation["taxonomy_rows"],
        cell_rows=evaluation["cell_rows"],
        compute_rows=compute_rows,
        score_distributions=score_distributions,
    )
    write_final_report(
        output_dir=output_dir,
        assessment=assessment,
        budget_rows=evaluation["budget_rows"],
        compute_rows=compute_rows,
        independent_summary=independent_summary,
    )
    final_summary = {
        "STATUS": assessment["status"],
        "PRIMARY_METHOD": assessment["primary_method"],
        "PRIMARY_BUDGET": float(assessment["primary_budget"]),
        "DELTA_UTILITY_VS_UNCERTAINTY": selected["point_delta_utility"],
        "PRIMARY_CI": {
            "lower": selected["ci95_low"],
            "upper": selected["ci95_high"],
            "unit": "source_cluster",
            "repeats": BOOTSTRAP_REPEATS,
        },
        "SEED_CONSISTENCY": {
            "positive": selected["seed_positive"],
            "total": len(TRAINING_SEEDS),
            "per_seed_delta": selected["seed_deltas"],
        },
        "SEEN_RESULT": assessment["candidates"][
            assessment["primary_method"]
        ].get("seen_delta"),
        "UNSEEN_RESULT": selected["unseen_delta"],
        "MAX_ACTIVATION": selected["max_activation"],
        "COMPUTE_RESULT": {
            "advantage_ms_per_frame": selected["compute_advantage_ms"],
            "advantage_fraction": selected["compute_advantage_fraction"],
            "role": "engineering_diagnostic",
        },
        "INDEPENDENT_REPLICATION": independent_summary,
        "FAILURE_MODE": assessment["failure_mode"],
        "PROTOCOL_DEVIATIONS": [],
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_SEARCH_AUTHORIZED": False,
        "OBSERVATION": (
            "Utility is evaluated from actual Short and RF384 predictions "
            "on the frozen INTERNAL_TEST split."
        ),
        "SUPPORTED_INTERPRETATION": (
            "Any learned-router improvement is conditional on the frozen "
            "features, split, seeds, and budget rule."
        ),
        "UNSUPPORTED_CLAIM": (
            "The study does not establish a new deployed router or "
            "generalization beyond the tested data."
        ),
    }
    write_json(output_dir / "a2_final_summary.json", final_summary)
    claim_lines = [
        "# A-v2 Claim Freeze",
        "",
        f"- Final status: **{assessment['status']}**",
        f"- Primary method: **{assessment['primary_method']}**",
        f"- Primary budget: **{100.0 * float(assessment['primary_budget']):.0f}%**",
        f"- Delta utility: **{_fmt(selected['point_delta_utility'])}**",
        f"- 95% source-cluster CI: "
        f"**[{_fmt(selected['ci95_low'])}, {_fmt(selected['ci95_high'])}]**",
        "",
        "No post-result model, threshold, loss, feature, budget, or seed search "
        "is authorized. The next step is human review.",
        "",
        "`NEXT_SEARCH_AUTHORIZED=false`",
        "",
    ]
    (output_dir / "a2_claim_freeze.md").write_text(
        "\n".join(claim_lines), encoding="utf-8"
    )
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "run", "smoke"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--protocol-hash", type=Path, default=None)
    parser.add_argument("--frame-bundle", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--ax-protocol", type=Path, default=DEFAULT_AX_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-independent", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    ax_payload = read_json(args.ax_protocol)
    frame_bundle = (
        args.frame_bundle.resolve()
        if args.frame_bundle is not None
        else resolve_path(ax_payload["data"]["prediction_bundle"]["path"])
    )
    feature_cache = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else REPO_ROOT
        / "results"
        / "difficulty_adaptive_context"
        / "ax1_value_predictability"
        / "ax1_features.npz"
    )
    if args.command == "freeze":
        protocol = freeze_protocol(
            output_dir=output_dir,
            frame_bundle_path=frame_bundle,
            feature_path=feature_cache,
            ax_protocol_path=args.ax_protocol,
        )
        print(
            json.dumps(
                {
                    "protocol": str(protocol.path),
                    "sha256": protocol.sha256,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    protocol_path = (
        args.protocol.resolve()
        if args.protocol is not None
        else output_dir / "a2_method_protocol_freeze.json"
    )
    hash_path = (
        args.protocol_hash.resolve()
        if args.protocol_hash is not None
        else output_dir / "a2_protocol_sha256.txt"
    )
    protocol = load_frozen_protocol(
        protocol_path,
        hash_path=hash_path,
    )
    if args.command == "smoke":
        verify_protocol_locks(protocol.payload)
        claim_once(
            output_dir / SMOKE_MARKER_NAME,
            command="smoke",
            protocol_sha256=protocol.sha256,
        )
        locks = protocol.payload["locks"]
        frame_bundle = resolve_path(locks["frame_bundle"]["path"])
        feature_cache = resolve_path(locks["feature_cache"]["path"])
        data = load_frame_data(
            frame_bundle_path=frame_bundle,
            feature_path=feature_cache,
        )
        result = {
            "protocol_sha256": protocol.sha256,
            "frames": int(data.labels.size),
            "train_frames": int(data.train_mask.sum()),
            "dev_frames": int(data.dev_mask.sum()),
            "internal_test_frames": int(data.test_mask.sum()),
            "taxonomy": {
                role: {
                    name: int(
                        np.count_nonzero(
                            data.taxonomy[mask] == name
                        )
                    )
                    for name in TAXONOMY_ORDER
                }
                for role, mask in (
                    ("TRAIN", data.train_mask),
                    ("DEV", data.dev_mask),
                    ("INTERNAL_TEST", data.test_mask),
                )
            },
        }
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.skip_independent:
        raise ValueError(
            "the frozen A-v2 internal-test run must include independent "
            "backbone sensitivity; --skip-independent is forbidden"
        )
    claim_once(
        output_dir / INTERNAL_RUN_STARTED_MARKER_NAME,
        command="internal-test",
        protocol_sha256=protocol.sha256,
    )
    result = run_protocol(
        protocol=protocol,
        output_dir=output_dir,
        device_name=args.device,
        progress_every=args.progress_every,
        skip_independent=False,
    )
    claim_once(
        output_dir / INTERNAL_RUN_COMPLETE_MARKER_NAME,
        command="internal-test-complete",
        protocol_sha256=protocol.sha256,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
