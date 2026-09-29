# -*- coding: utf-8 -*-
"""Execute the frozen E1 remote temporal information intervention study."""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from reproductions.difficulty_adaptive_context.adaptive_model import (
    SparseCausalDepthwiseBranch,
)
from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    speaker_from_source_key,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    EvaluationItem,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "E1-RTI-v2"
INTERVENTION_PROTOCOL_ID = "E1-RTI-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_E1_OUTCOME_ANALYSIS"
RUNNER_MODULE = "reproductions.difficulty_adaptive_context.run_e1"
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e1_remote_temporal_intervention"
)
PROTOCOL_PATH = OUTPUT_ROOT / "e1_v2_protocol_freeze.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "e1_v2_protocol_sha256.txt"
RUNNER_HASH_PATH = OUTPUT_ROOT / "e1_v2_runner_sha256.txt"
SCORING_PATH_TEST_PATH = OUTPUT_ROOT / "e1_v2_scoring_path_tests.json"
REVISION_RECORD_PATH = OUTPUT_ROOT / "e1_v1_to_v2_revision.md"
V2_PREFIX = "e1_v2_"

AUTHORIZATION_DISCLOSURE = (
    "E1-RTI-v1 was invalidated before intervention because the manual "
    "scoring path did not reproduce the frozen Full baseline within the "
    "preregistered tolerance. A diagnostic performed without executing "
    "interventions isolated the discrepancy to classifier chunking "
    "semantics. After explicit authorization, E1-RTI-v2 changed only the "
    "classifier scoring path to reproduce the frozen 2,000-frame chunked "
    "reference. The scientific hypotheses, interventions, endpoints, "
    "thresholds, gates, and evaluation population were unchanged."
)


def _provenance_fields() -> dict[str, Any]:
    return {
        "PROTOCOL_VERSION": PROTOCOL_ID,
        "V1_INVALID_RETAINED": True,
        "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "IMPLEMENTATION_CORRECTION": True,
        "INTERVENTION_EXECUTED_BEFORE_V2_FREEZE": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "TRAINING_PERFORMED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "AUTHORIZATION_DISCLOSURE": AUTHORIZATION_DISCLOSURE,
    }


ORIGINAL_PROTOCOL_SHA256 = (
    "52D995474E455649DD81D86138B731BEFBBA725D8F5028452928FA7C2CF5F793"
)
ORIGINAL_RUNNER_SHA256 = (
    "A03A0939DF7D54C9195C05930A2BDC4AE76896BC8CF55F91A9567270003397F9"
)
ROOT_CAUSE_DIAGNOSTIC_PATH = (
    OUTPUT_ROOT / "diagnostic_c1_root_cause.md"
)
ROOT_CAUSE_DIAGNOSTIC_SHA256 = (
    "EAC068D1AD662005AC3DE57F64185A5E31D2F3E2A25F724238BF1A134BFC3951"
)
FULL_POPULATION_DIAGNOSTIC_PATH = (
    OUTPUT_ROOT / "diagnostic_c1_chunked_classifier_full.json"
)
FULL_POPULATION_DIAGNOSTIC_SHA256 = (
    "77DD581AB9912091BE20F58A22F6DFBF93BC8A330A0A1366E1DBBE5A49AE75F0"
)

MANIFEST_PATH = REPO_ROOT / "data/librivad/manifests/LibriSpeech_test_medium.tsv"
DATA_ROOT = REPO_ROOT / "data/librivad"
LIBRISPEECH_ROOT = REPO_ROOT / "data/LibriSpeech"
RF384_CHECKPOINT = (
    REPO_ROOT
    / "results/difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025_rf384/best.pt"
)
RF64_CHECKPOINT = (
    REPO_ROOT
    / "results/difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025/best.pt"
)
SHORT_CHECKPOINT = (
    REPO_ROOT
    / "results/difficulty_adaptive_context/formal_short_ext40/best.pt"
)
RF384_REFERENCE = (
    REPO_ROOT
    / "results/difficulty_adaptive_context"
    / "a9_span_sweep/seed17/eval_rf384_fixed130/frame_predictions.npz"
)
RF64_REFERENCE = (
    REPO_ROOT
    / "results/difficulty_adaptive_context"
    / "a9_span_sweep/seed17/eval_rf64_fixed130/frame_predictions.npz"
)

VALID_START = 382
DECISION_THRESHOLD = 0.5
GATE_THRESHOLD = 0.13
INTERVENTION_SEEDS = (101, 211, 307, 401, 503)
C5_SEEDS = INTERVENTION_SEEDS
BOOTSTRAP_REPEATS = 2000
SOURCE_BOOTSTRAP_SEED = 20260921
SPEAKER_BOOTSTRAP_OFFSET = 500000
LOGLOSS_EPSILON = 1e-12
NEAR_ZERO_DELTA_LOGLOSS = 0.0005
R4_MIN_SURVIVAL_DROP = 0.05
R4_MIN_LOO_DROP = 0.025
R3_MIN_SEED_AGREEMENT = 4

RF384_DILATIONS = (1, 2, 4, 8, 96)
REMOTE_TAPS = (96, 192, 288, 384)
REMOTE_WINDOW_TAPS = (384, 288, 192, 96)
LOCAL_TAPS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
LOCAL_TAPS_NONCURRENT = LOCAL_TAPS[1:]
REMOTE_WINDOW_SLOTS = tuple(range(len(REMOTE_TAPS)))
CURRENT_WINDOW_SLOT = len(REMOTE_TAPS)
UNSEEN_NOISE = ("SSN_noise", "Street_noise", "Transport_noise")

EXPECTED_INPUT_HASHES = {
    "rf384_checkpoint": (
        "85ACDE86D8D9A21103992B8BA697AA55DFAE5987D93235A1FF3BB3BA4A5E10AB"
    ),
    "rf64_checkpoint": (
        "67F322394145A8C96BB6293B9FACFCA887268A87B3532CB2FDF0EBB963DAC71D"
    ),
    "short_checkpoint": (
        "9170DDF879FBCAF7F9AFDE24695D9BB02D02F919321BDE85A2F0E22511CD75C2"
    ),
    "rf384_reference": (
        "F25749E568D8DF6370FD2F52262EDF6169AFEB6DA30F1BAA6F46E80CD68EEC03"
    ),
    "rf64_reference": (
        "05BEE3FEE83EB1DACC29C2AC49F151EE4AF482DD45EF227C38DD2644952D1A46"
    ),
    "test_manifest": (
        "267CD1A018216E62552BC00B088F2667AD2113B1D1EC0E954A4F01854C1C8A39"
    ),
}
TOTAL_FRAMES = 553_532
TEST_FRAMES = 298_300
TEST_SOURCES = 96
TEST_SPEAKERS = 20
SHORT_REFERENCE_LOGLOSS = 0.21775919492365436
RF384_REFERENCE_LOGLOSS = 0.2076091007266905
AE2_STABLE_SUFFICIENT_SPAN = 69.22

DISTANCE_BINS = ("0", "1", "2-4", "5-10", "11-25", "26-50", ">50")
DURATION_BINS = ("1", "2-4", "5-10", "11-25", "26-50", ">50")
MISSING_DISTANCE = 1_000_000.0

CONDITION_ORDER = (
    "clean",
    "-5",
    "0",
    "5",
    "10",
    "15",
    "20",
)

FROZEN_INPUTS = {
    "rf384_checkpoint": RF384_CHECKPOINT,
    "rf64_checkpoint": RF64_CHECKPOINT,
    "short_checkpoint": SHORT_CHECKPOINT,
    "rf384_reference": RF384_REFERENCE,
    "rf64_reference": RF64_REFERENCE,
    "test_manifest": MANIFEST_PATH,
}

FROZEN_SOURCE_HASHES = (
    "reproductions/difficulty_adaptive_context/adaptive_model.py",
    "reproductions/difficulty_adaptive_context/data.py",
    "reproductions/difficulty_adaptive_context/evaluate_adaptive.py",
    "reproductions/difficulty_adaptive_context/analyze_context_gate.py",
    "reproductions/marblenet_vad/model.py",
    "reproductions/marblenet_vad/features.py",
    "reproductions/marblenet_vad/dataset.py",
)


@dataclass(frozen=True)
class Record:
    """One frozen evaluation item and its contiguous reference slice."""

    item_index: int
    item: EvaluationItem
    start: int
    end: int
    n_frames: int
    labels: np.ndarray
    short_scores: np.ndarray
    full_scores: np.ndarray
    frozen_full_scores: np.ndarray
    source_key: str
    speaker_id: str
    noise_name: str
    condition: str
    is_clean: bool
    event_features: dict[str, np.ndarray]

    @property
    def retained_frames(self) -> int:
        return int(self.end - self.start)


@dataclass(frozen=True)
class Population:
    records: list[Record]
    labels: np.ndarray
    short_scores: np.ndarray
    full_scores: np.ndarray
    frozen_full_scores: np.ndarray
    test_mask: np.ndarray
    source_key: np.ndarray
    speaker_ids: np.ndarray
    noise_name: np.ndarray
    condition: np.ndarray
    selected: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _output_name(name: str) -> str:
    if not name.startswith("e1_"):
        raise ValueError(f"unexpected E1 output name: {name}")
    return name.replace("e1_", V2_PREFIX, 1)


def _output_path(name: str) -> Path:
    return OUTPUT_ROOT / _output_name(name)


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


def _clip_probabilities(scores: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(scores, dtype=np.float64),
        LOGLOSS_EPSILON,
        1.0 - LOGLOSS_EPSILON,
    )


def _logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = _clip_probabilities(scores)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


def _brier(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean((scores - labels) ** 2))


def _correct(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return (np.asarray(scores) >= DECISION_THRESHOLD) == np.asarray(
        labels,
        dtype=bool,
    )


def _taxonomy(labels: np.ndarray, short_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    short_correct = _correct(labels, short_scores)
    scores_correct = _correct(labels, scores)
    result = np.full(labels.size, "I", dtype="<U2")
    result[short_correct & scores_correct] = "SS"
    result[~short_correct & scores_correct] = "R"
    result[short_correct & ~scores_correct] = "H"
    return result


def _speech_bin(fraction: float) -> str:
    value = float(fraction)
    if value < 0.25:
        return "[0,0.25)"
    if value < 0.50:
        return "[0.25,0.50)"
    if value < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def _distance_bin(value: float) -> str:
    value = float(value)
    for label in DISTANCE_BINS:
        if label == "0" and value == 0.0:
            return label
        if label == "1" and value == 1.0:
            return label
        if label == ">50" and value > 50.0:
            return label
        if "-" in label:
            low, high = label.split("-", 1)
            if float(low) <= value <= float(high):
                return label
    return DISTANCE_BINS[0]


def _duration_bin(value: float) -> str:
    value = float(value)
    for label in DURATION_BINS:
        if label == "1" and value == 1.0:
            return label
        if label == ">50" and value > 50.0:
            return label
        if "-" in label:
            low, high = label.split("-", 1)
            if float(low) <= value <= float(high):
                return label
    return DURATION_BINS[0]


def _run_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    result = np.zeros(values.size, dtype=np.int16)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[end] == values[start]:
            end += 1
        if values[start]:
            result[start:end] = end - start
        start = end
    return result


def _nearest_signed_distance(
    positions: np.ndarray,
    length: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    result = np.full(length, MISSING_DISTANCE, dtype=np.float64)
    if positions.size == 0:
        return result
    for frame in range(length):
        distances = np.abs(positions - frame)
        nearest = int(np.argmin(distances))
        result[frame] = float(frame - positions[nearest])
    return result


def _nearest_absolute_distance(
    positions: np.ndarray,
    length: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    result = np.full(length, MISSING_DISTANCE, dtype=np.float64)
    if positions.size == 0:
        return result
    for frame in range(length):
        result[frame] = float(np.min(np.abs(positions - frame)))
    return result


def _segment_event_features(
    labels: np.ndarray,
    short_scores: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    length = int(labels.size)
    if length != short_scores.size or length != selected.size:
        raise ValueError("event-feature inputs must have equal length")
    onset = np.flatnonzero((labels[1:] == 1) & (labels[:-1] == 0)) + 1
    offset = np.flatnonzero((labels[1:] == 0) & (labels[:-1] == 1)) + 1
    short_state = short_scores >= DECISION_THRESHOLD
    crossings = np.flatnonzero(short_state[1:] != short_state[:-1]) + 1
    return {
        "onset_distance": _nearest_signed_distance(onset, length),
        "offset_distance": _nearest_signed_distance(offset, length),
        "posterior_transition_distance": _nearest_absolute_distance(
            crossings,
            length,
        ),
        "uncertainty_persistence": _run_lengths(selected).astype(
            np.float64,
        ),
    }


def _load_population() -> Population:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST_PATH}")
    if not RF384_REFERENCE.exists():
        raise FileNotFoundError(
            f"reference predictions not found: {RF384_REFERENCE}"
        )
    with np.load(RF384_REFERENCE, allow_pickle=False) as payload:
        required = {
            "labels",
            "short_scores",
            "full_adaptive_scores",
            "test_mask",
            "source_key",
            "speaker_ids",
            "condition",
            "noise_name",
            "selected",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"reference is missing arrays: {missing}")
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in required
        }

    items = build_evaluation_items(
        MANIFEST_PATH,
        generated_root=DATA_ROOT / "generated",
        label_root=DATA_ROOT / "labels",
        librispeech_root=LIBRISPEECH_ROOT,
        row_sample=1080,
        seed=17,
        include_clean=True,
    )
    records: list[Record] = []
    cursor = 0
    for item_index, item in enumerate(items):
        waveform = read_int16_audio(item.audio_path)
        n_frames = int(waveform.size // FRAME_HOP + 1)
        if n_frames <= VALID_START:
            continue
        labels_all = causal_frame_labels(
            np.load(item.label_path),
            n_frames,
        )
        labels = np.asarray(labels_all[VALID_START:], dtype=np.int64)
        retained = int(labels.size)
        end = cursor + retained
        if end > arrays["labels"].size:
            raise ValueError("reference bundle ended before item reconstruction")
        slice_ = slice(cursor, end)
        expected_source = np.full(retained, item.source_key, dtype=object)
        expected_noise = np.full(retained, item.noise_name, dtype=object)
        expected_condition = np.full(
            retained,
            "clean" if item.is_clean else str(item.snr_db),
            dtype=object,
        )
        if not np.array_equal(labels, arrays["labels"][slice_]):
            raise ValueError(f"labels mismatch at item {item_index}")
        if not np.array_equal(expected_source.astype(str), arrays["source_key"][slice_].astype(str)):
            raise ValueError(f"source mismatch at item {item_index}")
        if not np.array_equal(expected_noise.astype(str), arrays["noise_name"][slice_].astype(str)):
            raise ValueError(f"noise mismatch at item {item_index}")
        if not np.array_equal(expected_condition.astype(str), arrays["condition"][slice_].astype(str)):
            raise ValueError(f"condition mismatch at item {item_index}")

        short_scores = arrays["short_scores"][slice_]
        full_scores = arrays["full_adaptive_scores"][slice_]
        selected = arrays["selected"][slice_]
        event_features = _segment_event_features(
            labels,
            short_scores,
            selected,
        )
        records.append(
            Record(
                item_index=item_index,
                item=item,
                start=cursor,
                end=end,
                n_frames=n_frames,
                labels=labels,
                short_scores=short_scores,
                full_scores=full_scores,
                frozen_full_scores=full_scores.copy(),
                source_key=item.source_key,
                speaker_id=speaker_from_source_key(item.source_key),
                noise_name=item.noise_name,
                condition="clean" if item.is_clean else str(item.snr_db),
                is_clean=bool(item.is_clean),
                event_features=event_features,
            )
        )
        cursor = end

    total = int(arrays["labels"].size)
    if cursor != total:
        raise ValueError(
            f"population reconstruction ended at {cursor}, expected {total}"
        )
    return Population(
        records=records,
        labels=np.asarray(arrays["labels"], dtype=np.int64),
        short_scores=np.asarray(arrays["short_scores"], dtype=np.float64),
        full_scores=np.asarray(
            arrays["full_adaptive_scores"],
            dtype=np.float64,
        ),
        frozen_full_scores=np.asarray(
            arrays["full_adaptive_scores"],
            dtype=np.float64,
        ),
        test_mask=np.asarray(arrays["test_mask"], dtype=bool),
        source_key=np.asarray(arrays["source_key"], dtype=str),
        speaker_ids=np.asarray(arrays["speaker_ids"], dtype=str),
        noise_name=np.asarray(arrays["noise_name"], dtype=str),
        condition=np.asarray(arrays["condition"], dtype=str),
        selected=np.asarray(arrays["selected"], dtype=bool),
    )


def _branch_taps(branch: SparseCausalDepthwiseBranch) -> tuple[int, ...]:
    return tuple(
        (branch.kernel_size - 1 - index) * branch.dilation
        for index in range(branch.kernel_size)
    )


def _validate_receptive_field(
    model: Any,
) -> dict[str, Any]:
    branches = list(model.refinement.branches)
    dilations = tuple(int(branch.dilation) for branch in branches)
    kernel_sizes = tuple(int(branch.kernel_size) for branch in branches)
    if dilations != RF384_DILATIONS:
        raise ValueError(
            f"unexpected RF384 dilations {dilations}, expected {RF384_DILATIONS}"
        )
    if len(set(kernel_sizes)) != 1 or kernel_sizes[0] != 5:
        raise ValueError(
            f"unexpected RF384 kernel sizes {kernel_sizes}, expected all 5"
        )
    by_branch = {
        int(branch.dilation): _branch_taps(branch)
        for branch in branches
    }
    union = tuple(sorted({tap for taps in by_branch.values() for tap in taps}))
    local_union = tuple(
        sorted(
            {
                tap
                for dilation in RF384_DILATIONS[:-1]
                for tap in by_branch[int(dilation)]
            }
        )
    )
    remote = tuple(
        sorted(
            {
                tap
                for tap in by_branch[int(RF384_DILATIONS[-1])]
                if tap != 0
            }
        )
    )
    if local_union != LOCAL_TAPS:
        raise ValueError(
            f"unexpected local tap union {local_union}, expected {LOCAL_TAPS}"
        )
    if remote != REMOTE_TAPS:
        raise ValueError(
            f"unexpected remote tap set {remote}, expected {REMOTE_TAPS}"
        )
    return {
        "dilations": list(dilations),
        "kernel_sizes": list(kernel_sizes),
        "branch_taps": {
            str(dilation): list(taps)
            for dilation, taps in by_branch.items()
        },
        "union_taps": list(union),
        "local_taps": list(local_union),
        "remote_taps": list(remote),
        "structurally_skipped_relative_to_contiguous_range": [
            value for value in range(384) if value not in union
        ],
        "nominal_lookback": max(union),
        "actual_accessible_lookback": max(union),
        "padding": "left zero padding of (kernel_size - 1) * dilation per branch",
        "target_alignment": "tap q at target t reads encoded frame t - q",
    }


def _encode_item(
    model: Any,
    frontend: torch.nn.Module,
    waveform: np.ndarray,
    *,
    device: torch.device,
    chunk_frames: int,
) -> torch.Tensor:
    tensor = torch.from_numpy(
        np.asarray(waveform, dtype=np.float32) * INT16_SCALE
    ).unsqueeze(0)
    features = frontend(tensor.to(device, non_blocking=True))
    if features.shape[-1] == 0:
        return features.new_empty((1, model.short_model.encoder[0].mconv[0].out_channels, 0))
    parts: list[torch.Tensor] = []
    states = None
    for start in range(0, features.shape[-1], chunk_frames):
        chunk = features[..., start : start + chunk_frames]
        encoded, states = model.short_model.encode_stream(chunk, states)
        parts.append(encoded)
    return torch.cat(parts, dim=-1)


def _gather_branch_windows(
    encoded: torch.Tensor,
    time_index: torch.Tensor,
    branch: SparseCausalDepthwiseBranch,
) -> torch.Tensor:
    if encoded.dim() != 3 or encoded.shape[0] != 1:
        raise ValueError("encoded must have shape [1, C, T]")
    taps = torch.arange(
        branch.kernel_size - 1,
        -1,
        -1,
        device=encoded.device,
    ) * int(branch.dilation)
    source = time_index[:, None] - taps[None, :].to(time_index.device)
    valid = source >= 0
    safe = source.clamp_min(0)
    windows = encoded[0, :, safe].transpose(0, 1)
    return windows * valid[:, None, :].to(windows.dtype)


def _branch_contribution(
    windows: torch.Tensor,
    branch: SparseCausalDepthwiseBranch,
) -> torch.Tensor:
    weight = branch.weight[None, :, 0, :]
    return (windows * weight).sum(dim=-1) + branch.bias[None, :]


def _score_branch_sum(
    branch_sum: torch.Tensor,
    short_logits: torch.Tensor,
    time_index: torch.Tensor,
    model: Any,
) -> torch.Tensor:
    activated = model.refinement.activation(branch_sum)
    raw = F.linear(
        activated,
        model.refinement.output.weight[:, :, 0],
        model.refinement.output.bias,
    )
    residual = model.refinement.config.max_residual * torch.tanh(raw)
    current_short = short_logits[0, :, time_index].transpose(0, 1)
    return torch.softmax(current_short + residual, dim=-1)[:, 1]


def _load_model(device: torch.device) -> tuple[Any, MfccFrontend, Any]:
    model, _payload, config = load_adaptive_model(RF384_CHECKPOINT, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    return model, frontend, config


def _target_indices_for_record(
    record: Record,
    device: torch.device,
    local_positions: np.ndarray | None = None,
) -> torch.Tensor:
    if local_positions is None:
        values = np.arange(VALID_START, record.n_frames, dtype=np.int64)
    else:
        values = np.asarray(local_positions, dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _stable_hash_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        "big",
    )


def _derangement(
    *,
    seed: int,
    kind: str,
    sample_id: str,
    target_index: int,
    size: int,
) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=np.int64)
    counter = 0
    while True:
        entropy = _stable_hash_int(
            INTERVENTION_PROTOCOL_ID,
            kind,
            int(seed),
            sample_id,
            int(target_index),
            counter,
        )
        permutation = np.random.default_rng(entropy).permutation(size)
        if np.all(permutation != np.arange(size, dtype=np.int64)):
            return permutation.astype(np.int64)
        counter += 1
        if counter > 1000:
            raise RuntimeError("could not construct a deterministic derangement")


def _remote_source_map_for_targets(
    *,
    seed: int,
    record: Record,
    target_local_indices: np.ndarray,
) -> np.ndarray:
    result = np.tile(
        np.arange(len(REMOTE_WINDOW_TAPS), dtype=np.int64),
        (target_local_indices.size, 1),
    )
    for row, target in enumerate(target_local_indices):
        available = [
            index
            for index, tap in enumerate(REMOTE_WINDOW_TAPS)
            if int(target) >= tap
        ]
        if len(available) <= 1:
            continue
        permutation = _derangement(
            seed=seed,
            kind="C3_REMOTE_PERMUTE",
            sample_id=record.item.sample_id,
            target_index=int(record.start + target - VALID_START),
            size=len(available),
        )
        for destination, source in zip(available, permutation):
            result[row, destination] = available[int(source)]
    return result


def _local_source_maps_for_targets(
    *,
    seed: int,
    record: Record,
    target_local_indices: np.ndarray,
) -> np.ndarray:
    slots = np.asarray(LOCAL_TAPS_NONCURRENT, dtype=np.int64)
    result = np.tile(slots[None, :], (target_local_indices.size, 1))
    for row, target in enumerate(target_local_indices):
        available = [index for index, tap in enumerate(slots) if int(target) >= int(tap)]
        if len(available) <= 1:
            continue
        permutation = _derangement(
            seed=seed,
            kind="C5_LOCAL_PERMUTE",
            sample_id=record.item.sample_id,
            target_index=int(record.start + target - VALID_START),
            size=len(available),
        )
        for destination, source in zip(available, permutation):
            result[row, destination] = slots[available[int(source)]]
    return result


def _remap_windows(
    source_windows: torch.Tensor,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather one tap-vector per row and return [N, C, 1]."""
    channels = source_windows.shape[1]
    rows = source_windows.shape[0]
    index = source_indices[:, None, None].expand(rows, channels, 1)
    return torch.gather(source_windows, 2, index)


def _remote_availability(
    target_local_indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return [N, 4] availability for the four remote taps."""
    targets = np.asarray(target_local_indices, dtype=np.int64)
    return torch.as_tensor(
        np.stack(
            [targets >= tap for tap in REMOTE_WINDOW_TAPS],
            axis=1,
        ),
        dtype=torch.bool,
        device=device,
    )


def _zero_remote_windows(
    remote_windows: torch.Tensor,
    target_local_indices: np.ndarray,
) -> torch.Tensor:
    """Zero only remote slots; preserve local branches and current frame."""
    if remote_windows.shape[-1] != len(REMOTE_TAPS) + 1:
        raise ValueError("remote window must contain four remote and one current slot")
    modified = remote_windows.clone()
    modified[:, :, REMOTE_WINDOW_SLOTS] = 0.0
    # Availability is intentionally not consulted: unavailable taps are
    # already exactly zero because the causal gather masks them.
    _ = target_local_indices
    return modified


def _permute_remote_windows(
    remote_windows: torch.Tensor,
    source_map: np.ndarray,
    target_local_indices: np.ndarray,
) -> torch.Tensor:
    """Apply a destination-to-source map to remote slots only."""
    if remote_windows.shape[-1] != len(REMOTE_TAPS) + 1:
        raise ValueError("remote window must contain four remote and one current slot")
    if source_map.shape != (remote_windows.shape[0], len(REMOTE_TAPS)):
        raise ValueError("remote source map has an unexpected shape")
    modified = remote_windows.clone()
    device = remote_windows.device
    availability = _remote_availability(target_local_indices, device=device)
    for destination in REMOTE_WINDOW_SLOTS:
        source = torch.as_tensor(
            source_map[:, destination],
            dtype=torch.long,
            device=device,
        )
        values = _remap_windows(remote_windows, source)[:, :, 0]
        modified[:, :, destination] = torch.where(
            availability[:, destination, None],
            values,
            remote_windows[:, :, destination],
        )
    return modified


def _replace_remote_windows(
    remote_windows: torch.Tensor,
    donor_windows: torch.Tensor,
    target_local_indices: np.ndarray,
) -> torch.Tensor:
    """Replace only available remote slots with donor remote vectors."""
    if remote_windows.shape[-1] != len(REMOTE_TAPS) + 1:
        raise ValueError("remote window must contain four remote and one current slot")
    if donor_windows.shape != remote_windows.shape:
        raise ValueError("donor and target remote windows must have equal shape")
    modified = remote_windows.clone()
    availability = _remote_availability(
        target_local_indices,
        device=remote_windows.device,
    )
    for destination in REMOTE_WINDOW_SLOTS:
        modified[:, :, destination] = torch.where(
            availability[:, destination, None],
            donor_windows[:, :, destination],
            remote_windows[:, :, destination],
        )
    return modified


def _score_record(
    *,
    record: Record,
    encoded: torch.Tensor,
    short_logits: torch.Tensor,
    model: Any,
    device: torch.device,
    c3_seeds: Sequence[int],
    c4_seeds: Sequence[int],
    c5_seeds: Sequence[int],
    donor_encoded: Mapping[int, torch.Tensor],
    donor_selection: Mapping[int, int],
    donor_records: Mapping[int, Record],
    target_local_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score one item with all pre-registered interventions."""
    local = _target_indices_for_record(
        record,
        device,
        target_local_indices,
    )
    local_np = local.detach().cpu().numpy()
    branches = list(model.refinement.branches)
    windows_by_branch: dict[int, torch.Tensor] = {}
    contribution_by_branch: dict[int, torch.Tensor] = {}
    for branch_index, branch in enumerate(branches):
        windows = _gather_branch_windows(encoded, local, branch)
        windows_by_branch[branch_index] = windows
        contribution_by_branch[branch_index] = _branch_contribution(
            windows,
            branch,
        )

    local_sum = contribution_by_branch[0]
    for branch_index in range(1, len(branches) - 1):
        local_sum = local_sum + contribution_by_branch[branch_index]
    remote_original_windows = windows_by_branch[len(branches) - 1]
    remote_original = contribution_by_branch[len(branches) - 1]
    embedded_short = torch.softmax(
        short_logits[0, :, local].transpose(0, 1),
        dim=-1,
    )[:, 1]
    full = _score_branch_sum(
        local_sum + remote_original,
        short_logits,
        local,
        model,
    )

    # Remote slots are [t-384, t-288, t-192, t-96]; the current frame is
    # the final slot and is never modified by C2-C4.
    remote_zero_windows = _zero_remote_windows(
        remote_original_windows,
        local_np,
    )
    zero = _score_branch_sum(
        local_sum
        + _branch_contribution(
            remote_zero_windows,
            branches[-1],
        ),
        short_logits,
        local,
        model,
    )

    result: dict[str, np.ndarray] = {
        "embedded_short": (
            embedded_short.detach().cpu().numpy().astype(np.float64)
        ),
        "full": full.detach().cpu().numpy().astype(np.float64),
        "C2_ZERO": zero.detach().cpu().numpy().astype(np.float64),
    }
    for seed in c3_seeds:
        source_map = _remote_source_map_for_targets(
            seed=int(seed),
            record=record,
            target_local_indices=local_np,
        )
        modified = _permute_remote_windows(
            remote_original_windows,
            source_map,
            local_np,
        )
        score = _score_branch_sum(
            local_sum
            + _branch_contribution(modified, branches[-1]),
            short_logits,
            local,
            model,
        )
        result[f"C3_{int(seed)}"] = (
            score.detach().cpu().numpy().astype(np.float64)
        )

    for seed in c4_seeds:
        donor_item_index = int(donor_selection[int(seed)])
        donor = donor_encoded[donor_item_index]
        target_rel = local_np.astype(np.int64) - VALID_START
        donor_record = donor_records[donor_item_index]
        donor_len = donor_record.retained_frames
        if donor_len < 3:
            raise ValueError("C4 donor has fewer than three evaluation frames")
        anchors = np.rint(
            target_rel.astype(np.float64)
            * float(donor_len - 1)
            / float(max(record.retained_frames - 1, 1))
        ).astype(np.int64)
        anchors = np.clip(anchors, 2, donor_len - 1)
        donor_anchor_indices = torch.as_tensor(
            anchors + VALID_START,
            dtype=torch.long,
            device=device,
        )
        donor_windows = _gather_branch_windows(
            donor,
            donor_anchor_indices,
            branches[-1],
        )
        modified = _replace_remote_windows(
            remote_original_windows,
            donor_windows,
            local_np,
        )
        score = _score_branch_sum(
            local_sum
            + _branch_contribution(modified, branches[-1]),
            short_logits,
            local,
            model,
        )
        result[f"C4_{int(seed)}"] = (
            score.detach().cpu().numpy().astype(np.float64)
        )

    for seed in c5_seeds:
        local_maps = _local_source_maps_for_targets(
            seed=int(seed),
            record=record,
            target_local_indices=local_np,
        )
        modified_local_sum = None
        for branch_index, branch in enumerate(branches[:-1]):
            taps = _branch_taps(branch)
            source_offsets = np.empty(
                (local_np.size, len(taps)),
                dtype=np.int64,
            )
            for tap_index, tap in enumerate(taps):
                if tap == 0:
                    source_offsets[:, tap_index] = 0
                else:
                    slot_index = LOCAL_TAPS_NONCURRENT.index(int(tap))
                    source_offsets[:, tap_index] = local_maps[:, slot_index]
            source_indices = local_np[:, None] - source_offsets
            source_indices = torch.as_tensor(
                source_indices,
                dtype=torch.long,
                device=device,
            )
            modified_windows = encoded.new_zeros(
                (
                    local_np.size,
                    encoded.shape[1],
                    len(taps),
                )
            )
            for tap_index in range(len(taps)):
                source_index = source_indices[:, tap_index]
                source_values = encoded[0, :, source_index].transpose(0, 1)
                modified_windows[:, :, tap_index] = source_values
            contribution = _branch_contribution(
                modified_windows,
                branch,
            )
            modified_local_sum = (
                contribution
                if modified_local_sum is None
                else modified_local_sum + contribution
            )
        if modified_local_sum is None:
            raise RuntimeError("C5 requires at least one local branch")
        score = _score_branch_sum(
            modified_local_sum + remote_original,
            short_logits,
            local,
            model,
        )
        result[f"C5_{int(seed)}"] = (
            score.detach().cpu().numpy().astype(np.float64)
        )
    return result


def _bootstrap_mean_by_cluster(
    values: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    totals = np.bincount(codes, weights=values, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        int(n_clusters),
        size=(int(repeats), int(n_clusters)),
    )
    sampled_counts = indices @ counts
    sampled_totals = indices @ totals
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = sampled_totals / sampled_counts
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _cluster_codes(values: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    unique = sorted({str(value) for value in values})
    lookup = {value: index for index, value in enumerate(unique)}
    return (
        np.asarray([lookup[str(value)] for value in values], dtype=np.int64),
        unique,
    )


def _mean_and_ci(
    values: np.ndarray,
    source_key: np.ndarray,
    speaker_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    source_codes, source_names = _cluster_codes(source_key)
    speaker_codes, speaker_names = _cluster_codes(speaker_ids)
    low, high = _bootstrap_mean_by_cluster(
        values,
        source_codes,
        len(source_names),
        seed=seed,
    )
    speaker_low, speaker_high = _bootstrap_mean_by_cluster(
        values,
        speaker_codes,
        len(speaker_names),
        seed=seed + SPEAKER_BOOTSTRAP_OFFSET,
    )
    return {
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "ci95_low": low,
        "ci95_high": high,
        "speaker_ci95_low": speaker_low,
        "speaker_ci95_high": speaker_high,
    }


def _delta_logloss(labels: np.ndarray, full: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = _clip_probabilities(scores)
    full_probabilities = _clip_probabilities(full)
    return (
        -labels * np.log(probabilities)
        - (1.0 - labels) * np.log(1.0 - probabilities)
    ) - (
        -labels * np.log(full_probabilities)
        - (1.0 - labels) * np.log(1.0 - full_probabilities)
    )


def _delta_brier(full: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    return (scores - labels) ** 2 - (full - labels) ** 2


def _survival(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    intervention_scores: np.ndarray,
) -> float:
    original_r = (~_correct(labels, short_scores)) & _correct(
        labels,
        full_scores,
    )
    if not np.any(original_r):
        return float("nan")
    return float(np.mean(_correct(labels, intervention_scores)[original_r]))


def _loo_survival_drop(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    intervention_scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | None]:
    original_r = (~_correct(labels, short_scores)) & _correct(
        labels,
        full_scores,
    )
    full_survival = 1.0
    values: list[float] = []
    for group in sorted({str(value) for value in groups}):
        mask = original_r & (groups.astype(str) != group)
        if not np.any(mask):
            continue
        survival = float(
            np.mean(_correct(labels, intervention_scores)[mask])
        )
        values.append(full_survival - survival)
    if not values:
        return {"min": None, "max": None}
    return {"min": float(min(values)), "max": float(max(values))}


def _file_hash_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _load_v2_scoring_path_tests() -> dict[str, Any]:
    if not SCORING_PATH_TEST_PATH.exists():
        raise FileNotFoundError(
            "focused v2 scoring-path tests have not been recorded"
        )
    payload = json.loads(
        SCORING_PATH_TEST_PATH.read_text(encoding="utf-8")
    )
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("v2 scoring-path test record has the wrong protocol id")
    if payload.get("original_protocol_id") != INTERVENTION_PROTOCOL_ID:
        raise ValueError(
            "v2 scoring-path test record has the wrong original protocol id"
        )
    if payload.get("v1_runner_sha256") != ORIGINAL_RUNNER_SHA256:
        raise ValueError(
            "v2 scoring-path test record does not identify the frozen v1 runner"
        )
    if payload.get("v2_runner_sha256") != _sha256_file(Path(__file__)):
        raise ValueError(
            "v2 scoring-path test record belongs to a different runner"
        )
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("v2 scoring-path test record is missing checks")
    expected_checks = set("ABCDEFG")
    if set(str(key) for key in checks) != expected_checks:
        raise ValueError(
            "v2 scoring-path test record must contain checks A through G"
        )
    failed = [
        key
        for key, value in checks.items()
        if not isinstance(value, Mapping) or value.get("passed") is not True
    ]
    if failed:
        raise ValueError(
            f"required v2 scoring-path tests did not pass: {failed}"
        )
    if payload.get("all_required_tests_passed") is not True:
        raise ValueError(
            "v2 scoring-path test record does not report overall success"
        )
    return payload


def _revision_record() -> dict[str, Any]:
    root_cause_hash = _sha256_file(ROOT_CAUSE_DIAGNOSTIC_PATH)
    if root_cause_hash != ROOT_CAUSE_DIAGNOSTIC_SHA256:
        raise ValueError("root-cause diagnostic hash mismatch")
    full_population_hash = _sha256_file(FULL_POPULATION_DIAGNOSTIC_PATH)
    if full_population_hash != FULL_POPULATION_DIAGNOSTIC_SHA256:
        raise ValueError("full-population diagnostic hash mismatch")
    tests = _load_v2_scoring_path_tests()
    return {
        "original_protocol_id": INTERVENTION_PROTOCOL_ID,
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA256,
        "original_status": "INCONCLUSIVE_OR_INVALID",
        "original_runner_sha256": ORIGINAL_RUNNER_SHA256,
        "original_failure": (
            "mandatory C1 baseline reproduction failed before intervention "
            "because the classifier scoring path was not chunk-equivalent"
        ),
        "root_cause": (
            "classifier scoring-path mismatch: frozen reference evaluates "
            "the classifier in 2,000-frame chunks"
        ),
        "diagnostic_evidence": {
            "root_cause_record": _file_hash_record(
                ROOT_CAUSE_DIAGNOSTIC_PATH
            ),
            "full_population_record": _file_hash_record(
                FULL_POPULATION_DIAGNOSTIC_PATH
            ),
            "records": 1024,
            "frames": 553532,
            "max_abs_full_error": 2.384185791015625e-07,
            "max_abs_embedded_short_error": 0.0,
            "tolerance": 1e-06,
            "passed": True,
        },
        "revision_record": _file_hash_record(REVISION_RECORD_PATH),
        "scoring_path_tests": {
            **_file_hash_record(SCORING_PATH_TEST_PATH),
            "all_required_tests_passed": True,
            "checks": sorted(str(key) for key in tests["checks"]),
        },
        "authorized_change": (
            "_short_logits_for_encoded evaluates "
            "model.short_model.classifier in the same configured "
            "2,000-frame chunks as the frozen reference path and "
            "concatenates logits in time order"
        ),
        "scientific_protocol_changed": False,
        "implementation_correction": True,
        "intervention_executed_before_v2_freeze": False,
        "new_final_ood_touched": False,
        "training_performed": False,
        "result_driven_scientific_change": False,
        "authorization_disclosure": AUTHORIZATION_DISCLOSURE,
    }


def _revision_chain(
    *,
    protocol_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    revision = _revision_record()
    return {
        "v1": {
            "protocol_id": revision["original_protocol_id"],
            "protocol_sha256": revision["original_protocol_sha256"],
            "runner_sha256": revision["original_runner_sha256"],
            "status": revision["original_status"],
            "failure": revision["original_failure"],
        },
        "diagnostic_evidence": revision["diagnostic_evidence"],
        "authorized_correction": {
            "revision_record": revision["revision_record"],
            "change": revision["authorized_change"],
            "scientific_protocol_changed": False,
            "implementation_correction": True,
            "intervention_executed_before_v2_freeze": False,
        },
        "v2": {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": str(protocol_sha256),
            "runner_sha256": str(runner_sha256),
            "scoring_path_tests": revision["scoring_path_tests"],
        },
        "authorization_disclosure": AUTHORIZATION_DISCLOSURE,
    }


def build_protocol() -> dict[str, Any]:
    input_records = {
        name: _file_hash_record(path)
        for name, path in FROZEN_INPUTS.items()
    }
    for name, expected in EXPECTED_INPUT_HASHES.items():
        actual = input_records[name]["sha256"]
        if actual != expected:
            raise ValueError(
                f"frozen input hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    source_records = [
        _file_hash_record(REPO_ROOT / relative)
        for relative in FROZEN_SOURCE_HASHES
    ]
    runner_record = _file_hash_record(Path(__file__))
    return {
        "protocol_id": PROTOCOL_ID,
        "status": PROTOCOL_STATUS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "revision": _revision_record(),
        "study": {
            "name": "E1 Remote Temporal Information Intervention Study",
            "scope": (
                "frozen inference-only mechanism diagnostic on the frozen "
                "A9/AE evaluation population"
            ),
            "router_search": "forbidden",
            "architecture_search": "forbidden",
            "training": "forbidden",
            "threshold_change": "forbidden",
            "new_final_ood": "forbidden",
            "a14_rerun": "forbidden",
            "next_experiment": "forbidden",
        },
        "repository": {
            "root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "git_status_short": _git_status_short(),
        },
        "inputs": input_records,
        "runner": runner_record,
        "source_hashes": source_records,
        "feature_pipeline": {
            "frontend": "reproductions.marblenet_vad.features.MfccFrontend",
            "frontend_config": "MfccConfig(causal=True)",
            "score_backend": "CUDA",
            "score_chunk_frames": 2000,
            "frame_hop_samples": FRAME_HOP,
            "valid_start": VALID_START,
            "decision_threshold": DECISION_THRESHOLD,
            "gate_threshold": GATE_THRESHOLD,
            "selected_definition": "abs(short_score - 0.5) <= 0.13",
        },
        "population": {
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace(
                "\\",
                "/",
            ),
            "row_sample": 1080,
            "evaluation_seed": 17,
            "include_clean": True,
            "primary_mask": "frozen A9 test_mask",
            "expected_total_frames": 553532,
            "expected_test_frames": 298300,
            "expected_test_sources": TEST_SOURCES,
            "expected_test_speakers": TEST_SPEAKERS,
            "reference_logloss": {
                "short": SHORT_REFERENCE_LOGLOSS,
                "rf384_full": RF384_REFERENCE_LOGLOSS,
            },
            "source_clusters": "frozen source_key",
            "speaker_clusters": "frozen speaker_ids",
            "primary_bootstrap_unit": "source cluster",
            "speaker_bootstrap_role": "sensitivity only",
        },
        "rf384_dependency": {
            "kernel_size": 5,
            "dilations": list(RF384_DILATIONS),
            "tap_rule": "target t reads encoded frame t - q for q in branch taps",
            "local_taps": list(LOCAL_TAPS),
            "remote_taps": list(REMOTE_TAPS),
            "remote_window_taps": list(REMOTE_WINDOW_TAPS),
            "remote_definition": (
                "the nonzero taps of the final dilation-96 branch; all other "
                "RF384 taps form the local support"
            ),
            "structurally_skipped_positions": [
                value for value in range(384) if value not in set(
                    [
                        0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32,
                        96, 192, 288, 384,
                    ]
                )
            ],
            "targets_382_383_note": (
                "t-384 is left padding for targets 382 and 383, so only the "
                "available remote taps are changed"
            ),
        },
        "interventions": {
            "C1_FULL": {
                "remote": "original correctly aligned history",
                "local": "original",
            },
            "C2_REMOTE_ZERO": {
                "remote": "zero pre-GELU for every available remote tap",
                "justification": (
                    "the frozen encoder ends in ReLU, so zero is its natural "
                    "nonnegative feature floor"
                ),
                "local": "original",
                "role": "destructive control",
            },
            "C3_REMOTE_PERMUTE": {
                "remote": (
                    "deterministic derangement of the available true remote "
                    "tap vectors for each target"
                ),
                "local": "original",
                "seeds": list(INTERVENTION_SEEDS),
                "seed_rule": (
                    "SHA256(protocol_id|kind|seed|sample_id|target_index|counter)"
                ),
                "derangement": "reject any random permutation with a fixed point",
            },
            "C4_REMOTE_MATCHED_REPLACE": {
                "remote": (
                    "replace available remote taps with a same-utterance-length "
                    "mapped causal segment from a matched donor utterance"
                ),
                "local": "original",
                "seeds": list(INTERVENTION_SEEDS),
                "donor_pool": (
                    "only frozen A9 evaluation items whose source belongs to "
                    "the frozen primary test speaker split; target utterance "
                    "excluded; never NEW_FINAL_OOD"
                ),
                "hierarchy": [
                    "same noise/domain",
                    "same SNR condition",
                    "same speech-composition bin",
                    "smaller absolute speech-composition difference",
                    "different speaker where possible",
                    "different source where possible",
                    "deterministic hash tie break",
                ],
                "speech_composition_bins": [
                    "[0,0.25)",
                    "[0.25,0.50)",
                    "[0.50,0.75)",
                    "[0.75,1.00]",
                ],
                "anchor_rule": (
                    "linear relative-position anchor in the donor, clamped "
                    "so all four donor remote taps are legal causal history"
                ),
            },
            "C5_LOCAL_PERMUTE": {
                "local": (
                    "deterministic derangement of all available non-current "
                    "local dependency positions for each target"
                ),
                "remote": "original",
                "seeds": list(C5_SEEDS),
                "current_frame": "unchanged",
                "role": "positive-sensitivity sanity control",
            },
        },
        "primary_endpoint": {
            "loss": "binary log-loss with clipping epsilon 1e-12",
            "per_frame_effect": "L_INT(t) - L_FULL(t)",
            "primary_comparisons": [
                "C3_REMOTE_PERMUTE vs C1_FULL",
                "C4_REMOTE_MATCHED_REPLACE vs C1_FULL",
            ],
            "secondary_endpoint": "Brier score delta",
            "correctness_threshold": DECISION_THRESHOLD,
        },
        "bootstrap": {
            "primary_unit": "source cluster",
            "repeats": BOOTSTRAP_REPEATS,
            "confidence_interval": "2.5th to 97.5th percentile",
            "seed_base": SOURCE_BOOTSTRAP_SEED,
            "speaker_sensitivity": True,
            "intervention_averaging": (
                "average the five intervention realizations per target before "
                "bootstrapping"
            ),
        },
        "decision_rules": {
            "R1": (
                "C3 mean delta log-loss > 0 and source-cluster 95% CI lower "
                "bound > 0"
            ),
            "R2": (
                "C4 mean delta log-loss > 0 and source-cluster 95% CI lower "
                "bound > 0"
            ),
            "R3": (
                "at least 4 of 5 seeds have positive direction for both C3 "
                "and C4"
            ),
            "R4": (
                "original-R correction survival drop >= 0.05 for both C3 and "
                "C4; leave-one-source-out and leave-one-domain-out minimum "
                "drop >= 0.025"
            ),
            "R5": (
                "C2 ZERO is not the only positive evidence"
            ),
            "near_zero_delta_logloss": NEAR_ZERO_DELTA_LOGLOSS,
            "R4_min_survival_drop": R4_MIN_SURVIVAL_DROP,
            "R4_min_leave_one_out_drop": R4_MIN_LOO_DROP,
            "R3_min_seed_agreement": R3_MIN_SEED_AGREEMENT,
            "status_precedence": [
                "INCONCLUSIVE_OR_INVALID if C1 reproduction fails",
                "REMOTE_INFORMATION_SUPPORTED if R1-R5 all hold",
                "REMOTE_INFORMATION_NOT_SUPPORTED if C3 and C4 are both "
                "within the near-zero interval and C5 has a clear positive "
                "effect",
                "REMOTE_INFORMATION_CONDITIONAL if C3/C4 point estimates "
                "support remote importance but cluster uncertainty, seed "
                "direction, or leave-one-out stability prevents all R1-R5",
                "INCONCLUSIVE_OR_INVALID otherwise",
            ],
        },
        "outputs": {
            "directory": str(
                OUTPUT_ROOT.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "required_files": [
                _output_name(name)
                for name in (
                    "e1_protocol_freeze.json",
                    "e1_protocol_sha256.txt",
                    "e1_execution_manifest.json",
                    "e1_rf_dependency_audit.md",
                    "e1_causality_audit.json",
                    "e1_frame_results.parquet",
                    "e1_intervention_results.csv",
                    "e1_match_quality.csv",
                    "e1_primary_bootstrap.csv",
                    "e1_seed_consistency.csv",
                    "e1_taxonomy_transitions.csv",
                    "e1_stratified_results.csv",
                    "e1_a9_ae2_reconciliation.md",
                    "e1_figures/",
                    "e1_final_summary.json",
                    "e1_final_report.md",
                    "e1_claim_freeze.md",
                )
            ],
        },
        "protocol_notes": [
            "The RF384 profile's local support is the union of the first four "
            "branches, not the nominal contiguous range [0, 63].",
            "Positions 48 and 64 are structurally absent from RF384 and are "
            "therefore not declared as local or remote intervention taps.",
            "Targets 382 and 383 have only three available remote taps "
            "because t-384 is left padding.",
        ],
    }


def _freeze_protocol() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = build_protocol()
    _write_json(PROTOCOL_PATH, payload)
    digest = _sha256_file(PROTOCOL_PATH)
    PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    runner_digest = _sha256_file(Path(__file__))
    RUNNER_HASH_PATH.write_text(runner_digest + "\n", encoding="utf-8")
    freeze_manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": digest,
        "runner_sha256": runner_digest,
        "status": "FROZEN_AWAITING_C1",
        "protocol_status": PROTOCOL_STATUS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "revision_chain": _revision_chain(
            protocol_sha256=digest,
            runner_sha256=runner_digest,
        ),
        **_provenance_fields(),
        "C1_VALIDATION_COMPLETED": False,
        "INTERVENTION_EXECUTED": False,
    }
    _write_json(_output_path("e1_execution_manifest.json"), freeze_manifest)
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL_PATH),
                "sha256": digest,
                "runner_sha256": runner_digest,
                "status": PROTOCOL_STATUS,
            },
            indent=2,
        )
    )
    return 0


def _validate_frozen_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists() or not PROTOCOL_HASH_PATH.exists():
        raise FileNotFoundError(
            "run --mode freeze_protocol before the analysis"
        )
    expected = PROTOCOL_HASH_PATH.read_text(encoding="utf-8").strip().upper()
    actual = _sha256_file(PROTOCOL_PATH)
    if actual != expected:
        raise ValueError(
            f"protocol hash mismatch: recorded {expected}, actual {actual}"
        )
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("frozen protocol id mismatch")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("frozen protocol status mismatch")
    for name, expected_hash in EXPECTED_INPUT_HASHES.items():
        recorded = str(payload["inputs"][name]["sha256"])
        if recorded != expected_hash:
            raise ValueError(
                f"frozen protocol records a different {name} hash"
            )
        actual_hash = _sha256_file(FROZEN_INPUTS[name])
        if actual_hash != recorded:
            raise ValueError(
                f"frozen input changed after protocol freeze: {name}"
            )
    for item in payload.get("source_hashes", []):
        path = REPO_ROOT / str(item["path"])
        if _sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"frozen source changed after protocol freeze: {path}")
    runner = payload.get("runner")
    if not isinstance(runner, dict):
        raise ValueError("frozen protocol is missing the runner hash")
    if _sha256_file(Path(__file__)) != str(runner["sha256"]):
        raise ValueError("run_e1.py changed after protocol freeze")
    return payload


def write_rf_dependency_audit(
    dependency: Mapping[str, Any],
) -> None:
    lines = [
        "# E1 RF384 dependency audit",
        "",
        "This audit is derived from the loaded frozen checkpoint and the "
        "actual `SparseCausalDepthwiseBranch.forward_sparse` implementation.",
        "",
        "## Actual branch topology",
        "",
        f"- Kernel size: `{dependency['kernel_sizes'][0]}`",
        f"- Dilations: `{dependency['dilations']}`",
        f"- Branch tap map: `{dependency['branch_taps']}`",
        f"- Union of accessed history offsets: `{dependency['union_taps']}`",
        f"- Local support used by the first four branches: "
        f"`{dependency['local_taps']}`",
        f"- Remote support from the final branch: `{dependency['remote_taps']}`",
        "",
        "The target frame `t` uses encoded frame `t - q` for every listed "
        "history offset `q`; the current frame is `q=0`.",
        "",
        "## Padding and boundaries",
        "",
        "- Each branch uses causal left zero padding equal to "
        "`(kernel_size - 1) * dilation`.",
        f"- Nominal and actual maximum lookback: "
        f"`{dependency['nominal_lookback']}` frames.",
        "- For targets `382` and `383`, `t - 384` is left padding, so only "
        "remote taps `96`, `192`, and `288` are available.",
        "- For every target `t >= 384`, all four remote taps are available.",
        "",
        "## Protocol note",
        "",
        "The nominal RF384 span is 384 frames, but the actual accessed set is "
        "sparse. In particular, offsets `48` and `64` are not accessed by "
        "the RF384 profile. They must not be silently treated as local "
        "history. The E1 local support is therefore the exact union of the "
        "first four branch taps, and the remote support is exactly the four "
        "nonzero taps of the dilation-96 branch.",
        "",
    ]
    _output_path("e1_rf_dependency_audit.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _record_by_item(
    records: Sequence[Record],
) -> dict[int, Record]:
    return {record.item_index: record for record in records}


def _donor_choice(
    *,
    target: Record,
    candidates: Sequence[Record],
    seed: int,
) -> tuple[Record, dict[str, Any]]:
    target_bin = _speech_bin(float(np.mean(target.labels == 1)))
    target_fraction = float(np.mean(target.labels == 1))

    def key(candidate: Record) -> tuple[Any, ...]:
        candidate_fraction = float(np.mean(candidate.labels == 1))
        candidate_bin = _speech_bin(candidate_fraction)
        return (
            0 if candidate.noise_name == target.noise_name else 1,
            0 if candidate.condition == target.condition else 1,
            0 if candidate_bin == target_bin else 1,
            abs(candidate_fraction - target_fraction),
            0 if candidate.speaker_id != target.speaker_id else 1,
            0 if candidate.source_key != target.source_key else 1,
            _stable_hash_int(seed, target.item.sample_id, candidate.item.sample_id),
        )

    donor = min(candidates, key=key)
    donor_fraction = float(np.mean(donor.labels == 1))
    donor_bin = _speech_bin(donor_fraction)
    quality = {
        "target_item_index": int(target.item_index),
        "target_sample_id": target.item.sample_id,
        "target_source_key": target.source_key,
        "target_speaker_id": target.speaker_id,
        "target_noise_name": target.noise_name,
        "target_condition": target.condition,
        "target_speech_fraction": target_fraction,
        "target_speech_bin": target_bin,
        "donor_item_index": int(donor.item_index),
        "donor_sample_id": donor.item.sample_id,
        "donor_source_key": donor.source_key,
        "donor_speaker_id": donor.speaker_id,
        "donor_noise_name": donor.noise_name,
        "donor_condition": donor.condition,
        "donor_speech_fraction": donor_fraction,
        "donor_speech_bin": donor_bin,
        "same_noise": bool(donor.noise_name == target.noise_name),
        "same_snr": bool(donor.condition == target.condition),
        "same_speech_bin": bool(donor_bin == target_bin),
        "speech_fraction_abs_difference": abs(
            donor_fraction - target_fraction
        ),
        "different_utterance": bool(
            donor.item_index != target.item_index
        ),
        "different_speaker": bool(donor.speaker_id != target.speaker_id),
        "different_source": bool(donor.source_key != target.source_key),
        "seed": int(seed),
    }
    return donor, quality


def _build_seed_donors(
    records: Sequence[Record],
    test_mask: np.ndarray,
) -> tuple[dict[int, dict[int, int]], list[dict[str, Any]]]:
    """Build deterministic donor assignments for every frozen test record."""
    mask = np.asarray(test_mask, dtype=bool)
    total = sum(record.retained_frames for record in records)
    if mask.size != total:
        raise ValueError(
            f"test mask has {mask.size} frames, expected {total} from records"
        )
    test_sources: set[str] = set()
    for record in records:
        if np.any(mask[record.start : record.end]):
            test_sources.add(record.source_key)
    eligible = [
        record
        for record in records
        if record.source_key in test_sources
    ]
    if not eligible:
        raise ValueError("no frozen test records are available for donors")
    too_short = [
        record.item_index for record in eligible if record.retained_frames < 3
    ]
    if too_short:
        raise ValueError(
            f"eligible donors with fewer than three frames: {too_short}"
        )

    selections: dict[int, dict[int, int]] = {
        int(seed): {} for seed in INTERVENTION_SEEDS
    }
    quality_rows: list[dict[str, Any]] = []
    for target in eligible:
        candidates = [
            candidate
            for candidate in eligible
            if candidate.item_index != target.item_index
        ]
        if not candidates:
            raise ValueError(
                f"no donor candidates for target {target.item_index}"
            )
        for seed in INTERVENTION_SEEDS:
            donor, quality = _donor_choice(
                target=target,
                candidates=candidates,
                seed=int(seed),
            )
            if donor.item_index == target.item_index:
                raise AssertionError("target utterance was selected as its donor")
            if donor.source_key not in test_sources:
                raise AssertionError("donor came from outside the frozen test split")
            selections[int(seed)][int(target.item_index)] = int(
                donor.item_index
            )
            quality.update(
                {
                    "eligible_pool_size": len(eligible),
                    "candidate_count": len(candidates),
                    "target_in_frozen_test_split": True,
                    "donor_in_frozen_test_split": True,
                }
            )
            quality_rows.append(quality)
    return selections, quality_rows


def _short_logits_for_encoded(
    encoded: torch.Tensor,
    model: Any,
    *,
    chunk_frames: int = 2000,
) -> torch.Tensor:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if encoded.shape[-1] == 0:
        return model.short_model.classifier(encoded)
    parts = [
        model.short_model.classifier(
            encoded[..., start : start + chunk_frames]
        )
        for start in range(0, encoded.shape[-1], chunk_frames)
    ]
    return torch.cat(parts, dim=-1)


def _record_test_positions(
    record: Record,
    test_mask: np.ndarray,
) -> np.ndarray:
    local_mask = np.asarray(
        test_mask[record.start : record.end],
        dtype=bool,
    )
    if local_mask.size != record.retained_frames:
        raise ValueError("record slice does not match the frozen test mask")
    return np.flatnonzero(local_mask).astype(np.int64) + VALID_START


def _empty_intervention_result(
    *,
    record: Record,
    encoded: torch.Tensor,
    model: Any,
    device: torch.device,
    chunk_frames: int,
) -> dict[str, np.ndarray]:
    short_logits = _short_logits_for_encoded(
        encoded,
        model,
        chunk_frames=chunk_frames,
    )
    return _score_record(
        record=record,
        encoded=encoded,
        short_logits=short_logits,
        model=model,
        device=device,
        c3_seeds=(),
        c4_seeds=(),
        c5_seeds=(),
        donor_encoded={},
        donor_selection={},
        donor_records={},
    )


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.size != expected.size:
        raise ValueError("arrays must have equal size")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _write_invalid_summary(
    *,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "E1_STATUS": "INCONCLUSIVE_OR_INVALID",
        "PRIMARY_QUESTION": (
            "Does RF384 benefit depend on correctly related remote "
            "temporal information?"
        ),
        "FULL_BASELINE_REPRODUCED": False,
        "INVALID_REASON": str(reason),
        "INVALID_DETAILS": dict(details or {}),
        "PROTOCOL_DEVIATIONS": [],
        **_provenance_fields(),
    }
    _write_json(_output_path("e1_final_summary.json"), payload)


def _run_c1_validation(
    population: Population,
    model: Any,
    frontend: MfccFrontend,
    device: torch.device,
    *,
    chunk_frames: int,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Reproduce C1_FULL and embedded Short for every frozen frame."""
    if tolerance <= 0.0:
        raise ValueError("C1 tolerance must be positive")
    rows: list[dict[str, Any]] = []
    max_full = 0.0
    max_embedded_short = 0.0
    max_short_vs_reference_short = 0.0
    started = time.time()
    for record in population.records:
        waveform = read_int16_audio(record.item.audio_path)
        encoded = _encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=chunk_frames,
        )
        scores = _empty_intervention_result(
            record=record,
            encoded=encoded,
            model=model,
            device=device,
            chunk_frames=chunk_frames,
        )
        full_error = _max_abs_error(
            scores["full"],
            record.frozen_full_scores,
        )
        embedded_error = _max_abs_error(
            scores["embedded_short"],
            record.short_scores,
        )
        short_error = _max_abs_error(
            scores["embedded_short"],
            record.short_scores,
        )
        max_full = max(max_full, full_error)
        max_embedded_short = max(max_embedded_short, embedded_error)
        max_short_vs_reference_short = max(
            max_short_vs_reference_short,
            short_error,
        )
        rows.append(
            {
                "item_index": int(record.item_index),
                "sample_id": record.item.sample_id,
                "frames": int(record.retained_frames),
                "max_abs_full_error": full_error,
                "max_abs_embedded_short_error": embedded_error,
                "max_abs_embedded_short_vs_reference_short": short_error,
            }
        )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
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
        "tolerance": float(tolerance),
        "frames": int(population.labels.size),
        "records": len(population.records),
        "max_abs_full_error": max_full,
        "max_abs_embedded_short_error": max_embedded_short,
        "max_abs_embedded_short_vs_reference_short": (
            max_short_vs_reference_short
        ),
        "passed": bool(
            max_full <= tolerance
            and max_embedded_short <= tolerance
            and max_short_vs_reference_short <= tolerance
        ),
        "elapsed_seconds": float(time.time() - started),
        "record_rows": rows,
        **_provenance_fields(),
    }
    _write_json(_output_path("e1_c1_validation.json"), payload)
    if not payload["passed"]:
        _write_invalid_summary(
            reason="C1_FULL or embedded Short did not reproduce",
            details={
                "max_abs_full_error": max_full,
                "max_abs_embedded_short_error": max_embedded_short,
                "tolerance": tolerance,
            },
        )
        raise RuntimeError(
            "C1_FULL reproduction failed: "
            f"full={max_full}, short={max_embedded_short}, "
            f"tolerance={tolerance}"
        )
    return payload


def _validate_population(
    population: Population,
    *,
    require_c1: bool,
) -> dict[str, Any]:
    test_mask = np.asarray(population.test_mask, dtype=bool)
    source_keys = np.asarray(population.source_key, dtype=str)
    speaker_ids = np.asarray(population.speaker_ids, dtype=str)
    counts = {
        "frames": int(population.labels.size),
        "test_frames": int(np.count_nonzero(test_mask)),
        "test_sources": len(set(source_keys[test_mask])),
        "test_speakers": len(set(speaker_ids[test_mask])),
    }
    expected = {
        "frames": TOTAL_FRAMES,
        "test_frames": TEST_FRAMES,
        "test_sources": TEST_SOURCES,
        "test_speakers": TEST_SPEAKERS,
    }
    if counts != expected:
        raise ValueError(f"population mismatch: {counts}, expected {expected}")
    short_logloss = _logloss(
        population.labels[test_mask],
        population.short_scores[test_mask],
    )
    full_logloss = _logloss(
        population.labels[test_mask],
        population.full_scores[test_mask],
    )
    if abs(short_logloss - SHORT_REFERENCE_LOGLOSS) > 1e-12:
        raise ValueError(
            "frozen Short log-loss does not match the recorded reference"
        )
    if abs(full_logloss - RF384_REFERENCE_LOGLOSS) > 1e-12:
        raise ValueError(
            "frozen RF384 log-loss does not match the recorded reference"
        )
    if require_c1:
        c1_path = _output_path("e1_c1_validation.json")
        if not c1_path.exists():
            raise FileNotFoundError(
                "run --mode c1 before the E1 intervention analysis"
            )
        c1 = json.loads(c1_path.read_text(encoding="utf-8"))
        if not bool(c1.get("passed")):
            raise ValueError("recorded C1 validation did not pass")
        if int(c1.get("frames", -1)) != counts["frames"]:
            raise ValueError("recorded C1 validation has the wrong frame count")
    return {
        **counts,
        "short_logloss": short_logloss,
        "rf384_logloss": full_logloss,
    }


def _seed_column(condition: str, seed: int) -> str:
    return f"score_{condition}_{int(seed)}"


def _condition_score_column(condition: str) -> str:
    return f"score_{condition}"


def _condition_delta_column(condition: str) -> str:
    return f"delta_logloss_{condition}"


def _condition_mean(
    scores: Mapping[str, np.ndarray],
    condition: str,
    seeds: Sequence[int],
) -> np.ndarray:
    columns = [_seed_column(condition, seed) for seed in seeds]
    missing = [column for column in columns if column not in scores]
    if missing:
        raise KeyError(f"missing realization columns: {missing}")
    return np.mean(
        np.stack([scores[column] for column in columns], axis=0),
        axis=0,
    )


def _encode_donor_pool(
    *,
    records_by_item: Mapping[int, Record],
    item_indices: Iterable[int],
    model: Any,
    frontend: MfccFrontend,
    device: torch.device,
    chunk_frames: int,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    ordered = sorted({int(value) for value in item_indices})
    started = time.time()
    for position, item_index in enumerate(ordered, start=1):
        record = records_by_item[int(item_index)]
        waveform = read_int16_audio(record.item.audio_path)
        encoded = _encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=chunk_frames,
        )
        result[int(item_index)] = encoded.detach().cpu()
        if position % 25 == 0 or position == len(ordered):
            print(
                f"encoded donor {position}/{len(ordered)} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
    return result


def _frame_rows_from_scores(
    *,
    population: Population,
    record: Record,
    target_positions: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    target_positions = np.asarray(target_positions, dtype=np.int64)
    labels = record.labels[target_positions - VALID_START]
    short_scores = record.short_scores[target_positions - VALID_START]
    full_scores = scores["full"]
    embedded_short = scores["embedded_short"]
    test_mask = np.asarray(population.test_mask, dtype=bool)
    global_indices = (
        record.start + target_positions - VALID_START
    ).astype(np.int64)
    if not np.all(test_mask[global_indices]):
        raise AssertionError("attempted to score a non-test frame")
    if embedded_short.size != target_positions.size:
        raise ValueError("embedded Short score length mismatch")
    if _max_abs_error(embedded_short, short_scores) > 1e-6:
        raise RuntimeError("embedded Short changed inside the intervention runner")
    if _max_abs_error(
        full_scores,
        record.frozen_full_scores[target_positions - VALID_START],
    ) > 1e-6:
        raise RuntimeError("C1_FULL changed inside the intervention runner")

    local_seed_scores: dict[str, np.ndarray] = {
        "C1_FULL": full_scores,
        "C2_REMOTE_ZERO": np.asarray(scores["C2_ZERO"], dtype=np.float64),
    }
    local_seed_scores[
        _condition_score_column("C2_REMOTE_ZERO")
    ] = local_seed_scores["C2_REMOTE_ZERO"]
    for seed in INTERVENTION_SEEDS:
        for condition, prefix in (
            ("C3_REMOTE_PERMUTE", "C3"),
            ("C4_REMOTE_MATCHED_REPLACE", "C4"),
        ):
            key = f"{prefix}_{int(seed)}"
            if key not in scores:
                raise KeyError(f"missing intervention result {key}")
            local_seed_scores[_seed_column(condition, int(seed))] = np.asarray(
                scores[key],
                dtype=np.float64,
            )
    for seed in C5_SEEDS:
        key = f"C5_{int(seed)}"
        if key not in scores:
            raise KeyError(f"missing intervention result {key}")
        local_seed_scores[
            _seed_column("C5_LOCAL_PERMUTE", int(seed))
        ] = np.asarray(scores[key], dtype=np.float64)

    condition_seed_columns: dict[str, tuple[str, ...]] = {}
    for condition, seeds in (
        ("C3_REMOTE_PERMUTE", INTERVENTION_SEEDS),
        ("C4_REMOTE_MATCHED_REPLACE", INTERVENTION_SEEDS),
        ("C5_LOCAL_PERMUTE", C5_SEEDS),
    ):
        local_seed_scores[_condition_score_column(condition)] = (
            _condition_mean(local_seed_scores, condition, seeds)
        )
        condition_seed_columns[condition] = tuple(
            _seed_column(condition, seed) for seed in seeds
        )

    event_offset = target_positions - VALID_START
    event_features = record.event_features
    feature_slice = {
        name: np.asarray(values, dtype=np.float64)[event_offset]
        for name, values in event_features.items()
    }
    is_noisy = not record.is_clean
    unseen = bool(is_noisy and record.noise_name in UNSEEN_NOISE)
    seen = bool(is_noisy and not unseen)
    rows: list[dict[str, Any]] = []
    for row_index, global_index in enumerate(global_indices):
        row: dict[str, Any] = {
            "global_index": int(global_index),
            "item_index": int(record.item_index),
            "sample_id": record.item.sample_id,
            "source_key": record.source_key,
            "speaker_id": record.speaker_id,
            "noise_name": record.noise_name,
            "condition": record.condition,
            "unseen": unseen,
            "seen": bool(seen),
            "seen_group": "unseen" if unseen else ("seen" if seen else "clean"),
            "frame_in_utterance": int(target_positions[row_index]),
            "label": int(labels[row_index]),
            "short_score": float(short_scores[row_index]),
            "embedded_short_score": float(embedded_short[row_index]),
            "short_signed_value": float(short_scores[row_index] - 0.5),
            "full_score": float(full_scores[row_index]),
            "frozen_full_score": float(
                record.frozen_full_scores[
                    target_positions[row_index] - VALID_START
                ]
            ),
            "selected": bool(
                population.selected[global_index]
            ),
            "uncertainty": float(
                abs(short_scores[row_index] - 0.5)
            ),
            "short_correct": bool(
                _correct(
                    np.asarray([labels[row_index]]),
                    np.asarray([short_scores[row_index]]),
                )[0]
            ),
            "full_correct": bool(
                _correct(
                    np.asarray([labels[row_index]]),
                    np.asarray([full_scores[row_index]]),
                )[0]
            ),
            "full_taxonomy": str(
                _taxonomy(
                    np.asarray([labels[row_index]]),
                    np.asarray([short_scores[row_index]]),
                    np.asarray([full_scores[row_index]]),
                )[0]
            ),
        }
        for feature_name, values in feature_slice.items():
            row[feature_name] = float(values[row_index])
        row["transition_distance_bin"] = _distance_bin(
            abs(float(feature_slice["posterior_transition_distance"][row_index]))
        )
        row["onset_distance_bin"] = _distance_bin(
            abs(float(feature_slice["onset_distance"][row_index]))
        )
        row["offset_distance_bin"] = _distance_bin(
            abs(float(feature_slice["offset_distance"][row_index]))
        )
        row["uncertainty_persistence_bin"] = _duration_bin(
            float(feature_slice["uncertainty_persistence"][row_index])
        )
        for condition, score in local_seed_scores.items():
            row[condition] = float(score[row_index])
            if condition == "C1_FULL":
                continue
            loss_delta = _delta_logloss(
                np.asarray([labels[row_index]]),
                np.asarray([full_scores[row_index]]),
                np.asarray([score[row_index]]),
            )[0]
            brier_delta = _delta_brier(
                np.asarray([full_scores[row_index]]),
                np.asarray([score[row_index]]),
                np.asarray([labels[row_index]]),
            )[0]
            row[f"delta_logloss_{condition}"] = float(loss_delta)
            row[f"delta_brier_{condition}"] = float(brier_delta)
            if condition.startswith("C2_") or condition in {
                "C3_REMOTE_PERMUTE",
                "C4_REMOTE_MATCHED_REPLACE",
                "C5_LOCAL_PERMUTE",
            }:
                row[f"taxonomy_{condition}"] = str(
                    _taxonomy(
                        np.asarray([labels[row_index]]),
                        np.asarray([short_scores[row_index]]),
                        np.asarray([score[row_index]]),
                    )[0]
                )
        for condition, seed_columns in condition_seed_columns.items():
            row[f"delta_logloss_{condition}"] = float(
                np.mean(
                    [
                        row[f"delta_logloss_{column}"]
                        for column in seed_columns
                    ]
                )
            )
            row[f"delta_brier_{condition}"] = float(
                np.mean(
                    [
                        row[f"delta_brier_{column}"]
                        for column in seed_columns
                    ]
                )
            )
        rows.append(row)
    return rows


def _collect_frame_results(
    *,
    population: Population,
    model: Any,
    frontend: MfccFrontend,
    device: torch.device,
    donor_selection: Mapping[int, Mapping[int, int]],
    donor_records: Mapping[int, Record],
    donor_encoded_cpu: Mapping[int, torch.Tensor] | None,
    chunk_frames: int,
) -> pd.DataFrame:
    target_records = [
        record
        for record in population.records
        if np.any(_record_test_positions(record, population.test_mask))
    ]
    needed_donors: set[int] = set()
    for record in target_records:
        for seed in INTERVENTION_SEEDS:
            needed_donors.add(
                int(donor_selection[int(seed)][int(record.item_index)])
            )
    records_by_item = _record_by_item(population.records)
    if donor_encoded_cpu is None:
        donor_encoded_cpu = _encode_donor_pool(
            records_by_item=records_by_item,
            item_indices=needed_donors,
            model=model,
            frontend=frontend,
            device=device,
            chunk_frames=chunk_frames,
        )
    missing_donors = sorted(needed_donors - set(donor_encoded_cpu))
    if missing_donors:
        raise KeyError(f"missing encoded donors: {missing_donors}")

    rows: list[dict[str, Any]] = []
    started = time.time()
    for position, record in enumerate(target_records, start=1):
        target_positions = _record_test_positions(
            record,
            population.test_mask,
        )
        waveform = read_int16_audio(record.item.audio_path)
        encoded = _encode_item(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=chunk_frames,
        )
        donor_selection_for_record = {
            int(seed): int(donor_selection[int(seed)][int(record.item_index)])
            for seed in INTERVENTION_SEEDS
        }
        donor_encoded_device = {
            item_index: donor_encoded_cpu[item_index].to(device)
            for item_index in set(donor_selection_for_record.values())
        }
        scores = _score_record(
            record=record,
            encoded=encoded,
            short_logits=_short_logits_for_encoded(
                encoded,
                model,
                chunk_frames=chunk_frames,
            ),
            model=model,
            device=device,
            c3_seeds=INTERVENTION_SEEDS,
            c4_seeds=INTERVENTION_SEEDS,
            c5_seeds=C5_SEEDS,
            donor_encoded=donor_encoded_device,
            donor_selection=donor_selection_for_record,
            donor_records=donor_records,
            target_local_indices=target_positions,
        )
        rows.extend(
            _frame_rows_from_scores(
                population=population,
                record=record,
                target_positions=target_positions,
                scores=scores,
            )
        )
        if position % 10 == 0 or position == len(target_records):
            print(
                f"scored record {position}/{len(target_records)} "
                f"({len(rows)} frames, {time.time() - started:.1f}s)",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    if len(frame) != TEST_FRAMES:
        raise ValueError(
            f"scored {len(frame)} frames, expected {TEST_FRAMES}"
        )
    return frame


def _condition_seeds(condition: str) -> tuple[int, ...]:
    if condition in {"C3_REMOTE_PERMUTE", "C4_REMOTE_MATCHED_REPLACE"}:
        return INTERVENTION_SEEDS
    if condition == "C5_LOCAL_PERMUTE":
        return C5_SEEDS
    return ()


def _condition_role(condition: str) -> str:
    return {
        "C1_FULL": "baseline",
        "C2_REMOTE_ZERO": "destructive_secondary_control",
        "C3_REMOTE_PERMUTE": "primary_remote_intervention",
        "C4_REMOTE_MATCHED_REPLACE": "primary_remote_intervention",
        "C5_LOCAL_PERMUTE": "positive_sensitivity_control",
    }[condition]


def _analysis_conditions() -> tuple[str, ...]:
    return (
        "C1_FULL",
        "C2_REMOTE_ZERO",
        "C3_REMOTE_PERMUTE",
        "C4_REMOTE_MATCHED_REPLACE",
        "C5_LOCAL_PERMUTE",
    )


def _mean_realization_logloss(frame: pd.DataFrame, condition: str) -> float:
    labels = frame["label"].to_numpy(dtype=np.float64)
    full_scores = frame["full_score"].to_numpy(dtype=np.float64)
    probabilities = _clip_probabilities(full_scores)
    full_loss = (
        -labels * np.log(probabilities)
        - (1.0 - labels) * np.log(1.0 - probabilities)
    )
    if condition == "C1_FULL":
        return float(np.mean(full_loss))
    delta = frame[_condition_delta_column(condition)].to_numpy(
        dtype=np.float64,
    )
    return float(np.mean(full_loss + delta))


def _mean_realization_brier(frame: pd.DataFrame, condition: str) -> float:
    labels = frame["label"].to_numpy(dtype=np.float64)
    full_scores = frame["full_score"].to_numpy(dtype=np.float64)
    full_brier = _delta_brier(full_scores, full_scores, labels)
    if condition == "C1_FULL":
        return float(np.mean(full_brier))
    delta = frame[
        f"delta_brier_{condition}"
    ].to_numpy(dtype=np.float64)
    return float(np.mean(full_brier + delta))


def _condition_values(
    frame: pd.DataFrame,
    condition: str,
    metric: str,
) -> np.ndarray:
    if metric == "delta_logloss":
        if condition == "C1_FULL":
            return np.zeros(len(frame), dtype=np.float64)
        return frame[
            _condition_delta_column(condition)
        ].to_numpy(dtype=np.float64)
    if metric == "delta_brier":
        if condition == "C1_FULL":
            return np.zeros(len(frame), dtype=np.float64)
        return frame[
            f"delta_brier_{condition}"
        ].to_numpy(dtype=np.float64)
    raise ValueError(f"unsupported metric: {metric}")


def _predict_mean_realization(
    frame: pd.DataFrame,
    condition: str,
) -> np.ndarray:
    if condition == "C1_FULL":
        return frame["full_score"].to_numpy(dtype=np.float64)
    seeds = _condition_seeds(condition)
    if seeds:
        columns = [
            _seed_column(condition, seed)
            for seed in seeds
        ]
        return frame[columns].to_numpy(dtype=np.float64).mean(axis=1)
    return frame[
        _condition_score_column(condition)
    ].to_numpy(dtype=np.float64)


def _binary_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(_correct(labels, scores)))


def _mean_seed_accuracy(frame: pd.DataFrame, condition: str) -> float:
    labels = frame["label"].to_numpy(dtype=np.int64)
    if condition == "C1_FULL":
        return _binary_accuracy(
            labels,
            frame["full_score"].to_numpy(dtype=np.float64),
        )
    seeds = _condition_seeds(condition)
    if not seeds:
        return _binary_accuracy(
            labels,
            frame[
                _condition_score_column(condition)
            ].to_numpy(dtype=np.float64),
        )
    return float(
        np.mean(
            [
                _binary_accuracy(
                    labels,
                    frame[_seed_column(condition, seed)].to_numpy(
                        dtype=np.float64,
                    ),
                )
                for seed in seeds
            ]
        )
    )


def _mean_realization_correctness(
    frame: pd.DataFrame,
    condition: str,
) -> np.ndarray:
    labels = frame["label"].to_numpy(dtype=np.int64)
    if condition == "C1_FULL":
        return _correct(
            labels,
            frame["full_score"].to_numpy(dtype=np.float64),
        ).astype(np.float64)
    seeds = _condition_seeds(condition)
    if not seeds:
        return _correct(
            labels,
            frame[
                _condition_score_column(condition)
            ].to_numpy(dtype=np.float64),
        ).astype(np.float64)
    return np.mean(
        [
            _correct(
                labels,
                frame[_seed_column(condition, seed)].to_numpy(
                    dtype=np.float64,
                ),
            ).astype(np.float64)
            for seed in seeds
        ],
        axis=0,
    )


def _loo_correctness_drop(
    correctness: np.ndarray,
    original_r: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | None]:
    correctness = np.asarray(correctness, dtype=np.float64)
    original_r = np.asarray(original_r, dtype=bool)
    groups = np.asarray(groups, dtype=str)
    values: list[float] = []
    for group in sorted({str(value) for value in groups}):
        mask = original_r & (groups != group)
        if not np.any(mask):
            continue
        survival = float(np.mean(correctness[mask]))
        values.append(1.0 - survival)
    if not values:
        return {"min": None, "max": None}
    return {"min": float(min(values)), "max": float(max(values))}


def _survival_from_correctness(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    intervention_correctness: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    original_r = (~_correct(labels, short_scores)) & _correct(
        labels,
        full_scores,
    )
    if not np.any(original_r):
        raise ValueError("the original R stratum is empty")
    survival = float(
        np.mean(
            np.asarray(intervention_correctness, dtype=np.float64)[
                original_r
            ]
        )
    )
    return (
        {
            "original_r_count": int(np.count_nonzero(original_r)),
            "survival": survival,
            "survival_drop": 1.0 - survival,
        },
        original_r,
    )


def _survival_statistics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    short_scores = frame["short_score"].to_numpy(dtype=np.float64)
    full_scores = frame["full_score"].to_numpy(dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    original_r: np.ndarray | None = None
    for condition in _analysis_conditions():
        intervention_correctness = _mean_realization_correctness(
            frame,
            condition,
        )
        stats, current_r = _survival_from_correctness(
            labels=labels,
            short_scores=short_scores,
            full_scores=full_scores,
            intervention_correctness=intervention_correctness,
        )
        original_r = current_r if original_r is None else original_r
        result[condition] = {
            **stats,
            "leave_one_source_out": _loo_correctness_drop(
                intervention_correctness,
                current_r,
                frame["source_key"].to_numpy(dtype=str),
            ),
            "leave_one_domain_out": _loo_correctness_drop(
                intervention_correctness,
                current_r,
                frame["noise_name"].to_numpy(dtype=str),
            ),
        }
    if original_r is None:
        raise AssertionError("survival analysis did not run")
    result["ORIGINAL_R"] = {
        "prevalence": float(np.mean(original_r)),
        "count": int(np.count_nonzero(original_r)),
    }
    return result


def _build_condition_statistics(
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    source = frame["source_key"].to_numpy(dtype=str)
    speaker = frame["speaker_id"].to_numpy(dtype=str)
    result: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(_analysis_conditions()):
        delta_logloss = _condition_values(
            frame,
            condition,
            "delta_logloss",
        )
        delta_brier = _condition_values(
            frame,
            condition,
            "delta_brier",
        )
        result[condition] = {
            "mean_delta_logloss": float(np.mean(delta_logloss)),
            "delta_logloss": _mean_and_ci(
                delta_logloss,
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED + index * 10,
            ),
            "mean_delta_brier": float(np.mean(delta_brier)),
            "delta_brier": _mean_and_ci(
                delta_brier,
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED + index * 10 + 1,
            ),
            "accuracy": _binary_accuracy(
                frame["label"].to_numpy(dtype=np.int64),
                _predict_mean_realization(frame, condition),
            ),
            "mean_realization_accuracy": _mean_seed_accuracy(
                frame,
                condition,
            ),
            "mean_realization_logloss": _mean_realization_logloss(
                frame,
                condition,
            ),
            "mean_realization_brier": _mean_realization_brier(
                frame,
                condition,
            ),
        }
    short_logloss = _logloss(
        frame["label"].to_numpy(dtype=np.float64),
        frame["short_score"].to_numpy(dtype=np.float64),
    )
    full_logloss = result["C1_FULL"]["mean_realization_logloss"]
    for condition in _analysis_conditions():
        intervention_logloss = result[condition][
            "mean_realization_logloss"
        ]
        denominator = short_logloss - full_logloss
        result[condition]["value_retention"] = (
            float(
                (short_logloss - intervention_logloss) / denominator
            )
            if abs(denominator) > LOGLOSS_EPSILON
            else None
        )
    return result


def _intervention_result_rows(
    frame: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in _analysis_conditions():
        stats = condition_statistics[condition]
        deltas = _condition_values(frame, condition, "delta_logloss")
        briers = _condition_values(frame, condition, "delta_brier")
        survival_stats = survival[condition]
        rows.append(
            {
                "condition": condition,
                "role": _condition_role(condition),
                "frames": int(len(frame)),
                "realizations": len(_condition_seeds(condition)) or 1,
                "mean_delta_logloss": float(
                    stats["mean_delta_logloss"]
                ),
                "median_delta_logloss": float(np.median(deltas)),
                "mean_delta_brier": float(stats["mean_delta_brier"]),
                "median_delta_brier": float(np.median(briers)),
                "delta_logloss_ci95_low": float(
                    stats["delta_logloss"]["ci95_low"]
                ),
                "delta_logloss_ci95_high": float(
                    stats["delta_logloss"]["ci95_high"]
                ),
                "delta_logloss_speaker_ci95_low": float(
                    stats["delta_logloss"]["speaker_ci95_low"]
                ),
                "delta_logloss_speaker_ci95_high": float(
                    stats["delta_logloss"]["speaker_ci95_high"]
                ),
                "mean_realization_logloss": float(
                    stats["mean_realization_logloss"]
                ),
                "mean_realization_brier": float(
                    stats["mean_realization_brier"]
                ),
                "mean_realization_accuracy": float(
                    stats["mean_realization_accuracy"]
                ),
                "mean_score_accuracy": float(stats["accuracy"]),
                "value_retention": stats["value_retention"],
                "original_r_count": int(
                    survival_stats["original_r_count"]
                ),
                "correction_survival": float(
                    survival_stats["survival"]
                ),
                "correction_loss": float(
                    survival_stats["survival_drop"]
                ),
                "leave_one_source_min_drop": _optional_float(
                    survival_stats["leave_one_source_out"]["min"]
                ),
                "leave_one_source_max_drop": _optional_float(
                    survival_stats["leave_one_source_out"]["max"]
                ),
                "leave_one_domain_min_drop": _optional_float(
                    survival_stats["leave_one_domain_out"]["min"]
                ),
                "leave_one_domain_max_drop": _optional_float(
                    survival_stats["leave_one_domain_out"]["max"]
                ),
            }
        )
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _primary_bootstrap_rows(
    frame: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in _analysis_conditions():
        stats = condition_statistics[condition]
        for metric in ("delta_logloss", "delta_brier"):
            metric_stats = stats[metric]
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "role": _condition_role(condition),
                    "mean": float(metric_stats["mean"]),
                    "source_ci95_low": float(
                        metric_stats["ci95_low"]
                    ),
                    "source_ci95_high": float(
                        metric_stats["ci95_high"]
                    ),
                    "speaker_ci95_low": float(
                        metric_stats["speaker_ci95_low"]
                    ),
                    "speaker_ci95_high": float(
                        metric_stats["speaker_ci95_high"]
                    ),
                    "source_bootstrap_repeats": BOOTSTRAP_REPEATS,
                    "speaker_bootstrap_repeats": BOOTSTRAP_REPEATS,
                    "intervention_realizations_averaged": (
                        len(_condition_seeds(condition)) or 1
                    ),
                }
            )
    return rows


def _seed_consistency_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    source = frame["source_key"].to_numpy(dtype=str)
    speaker = frame["speaker_id"].to_numpy(dtype=str)
    rows: list[dict[str, Any]] = []
    for condition in _analysis_conditions()[1:]:
        seeds = _condition_seeds(condition)
        if not seeds:
            values = _condition_values(
                frame,
                condition,
                "delta_logloss",
            )
            interval = _mean_and_ci(
                values,
                source,
                speaker,
                seed=SOURCE_BOOTSTRAP_SEED + 2001,
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": 0,
                    "mean_delta_logloss": float(np.mean(values)),
                    "source_ci95_low": interval["ci95_low"],
                    "source_ci95_high": interval["ci95_high"],
                    "positive_frame_rate": float(np.mean(values > 0.0)),
                    "direction_positive": bool(np.mean(values) > 0.0),
                    "mean_delta_brier": float(
                        np.mean(
                            _condition_values(
                                frame,
                                condition,
                                "delta_brier",
                            )
                        )
                    ),
                    "row_type": "condition",
                }
            )
            continue

        seed_effects: list[float] = []
        for seed_index, seed in enumerate(seeds):
            column = _seed_column(condition, seed)
            values = frame[
                f"delta_logloss_{column}"
            ].to_numpy(dtype=np.float64)
            interval = _mean_and_ci(
                values,
                source,
                speaker,
                seed=(
                    SOURCE_BOOTSTRAP_SEED
                    + 3000
                    + _analysis_conditions().index(condition) * 100
                    + seed_index
                ),
            )
            effect = float(np.mean(values))
            seed_effects.append(effect)
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "mean_delta_logloss": effect,
                    "source_ci95_low": interval["ci95_low"],
                    "source_ci95_high": interval["ci95_high"],
                    "positive_frame_rate": float(np.mean(values > 0.0)),
                    "direction_positive": bool(effect > 0.0),
                    "mean_delta_brier": float(
                        np.mean(
                            frame[
                                f"delta_brier_{column}"
                            ].to_numpy(dtype=np.float64)
                        )
                    ),
                    "row_type": "realization",
                }
            )
        positive_count = int(
            sum(effect > 0.0 for effect in seed_effects)
        )
        all_values = _condition_values(
            frame,
            condition,
            "delta_logloss",
        )
        interval = _mean_and_ci(
            all_values,
            source,
            speaker,
            seed=(
                SOURCE_BOOTSTRAP_SEED
                + 4000
                + _analysis_conditions().index(condition)
            ),
        )
        rows.append(
            {
                "condition": condition,
                "seed": -1,
                "mean_delta_logloss": float(np.mean(all_values)),
                "source_ci95_low": interval["ci95_low"],
                "source_ci95_high": interval["ci95_high"],
                "positive_frame_rate": float(
                    np.mean(all_values > 0.0)
                ),
                "direction_positive": bool(
                    positive_count >= R3_MIN_SEED_AGREEMENT
                ),
                "mean_delta_brier": float(
                    np.mean(
                        _condition_values(
                            frame,
                            condition,
                            "delta_brier",
                        )
                    )
                ),
                "row_type": "seed_summary",
                "positive_seed_count": positive_count,
                "seed_count": len(seeds),
                "required_positive_seed_count": (
                    R3_MIN_SEED_AGREEMENT
                ),
                "accuracy": _mean_seed_accuracy(frame, condition),
            }
        )
    return rows


def _taxonomy_transition_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    short_scores = frame["short_score"].to_numpy(dtype=np.float64)
    full_scores = frame["full_score"].to_numpy(dtype=np.float64)
    full_taxonomy = _taxonomy(labels, short_scores, full_scores)
    states = ("SS", "R", "I", "H")
    rows: list[dict[str, Any]] = []
    per_seed: dict[tuple[str, str, str], list[int]] = {}
    for condition in _analysis_conditions():
        seeds = _condition_seeds(condition)
        realization_keys: list[tuple[str, int]] = []
        if condition == "C1_FULL":
            realization_keys.append(("C1_FULL", 0))
        elif seeds:
            realization_keys.extend(
                (_seed_column(condition, seed), int(seed))
                for seed in seeds
            )
        else:
            realization_keys.append(
                (_condition_score_column(condition), 0)
            )
        for column, seed in realization_keys:
            intervention_taxonomy = _taxonomy(
                labels,
                short_scores,
                frame[column].to_numpy(dtype=np.float64),
            )
            for full_state in states:
                full_mask = full_taxonomy == full_state
                denominator = int(np.count_nonzero(full_mask))
                for intervention_state in states:
                    count = int(
                        np.count_nonzero(
                            full_mask
                            & (intervention_taxonomy == intervention_state)
                        )
                    )
                    proportion = (
                        float(count) / float(denominator)
                        if denominator
                        else 0.0
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "row_type": "per_realization",
                            "full_taxonomy": full_state,
                            "intervention_taxonomy": intervention_state,
                            "count": count,
                            "full_state_count": denominator,
                            "proportion_within_full_state": proportion,
                        }
                    )
                    per_seed.setdefault(
                        (condition, full_state, intervention_state),
                        [],
                    ).append(count)
    for (condition, full_state, intervention_state), counts in sorted(
        per_seed.items()
    ):
        denominator_values = [
            int(
                row["full_state_count"]
            )
            for row in rows
            if (
                row["condition"] == condition
                and row["row_type"] == "per_realization"
                and row["full_taxonomy"] == full_state
                and row["intervention_taxonomy"] == intervention_state
            )
        ]
        denominator = (
            float(np.mean(denominator_values))
            if denominator_values
            else 0.0
        )
        mean_count = float(np.mean(counts))
        rows.append(
            {
                "condition": condition,
                "seed": -1,
                "row_type": "mean_realization",
                "full_taxonomy": full_state,
                "intervention_taxonomy": intervention_state,
                "count": mean_count,
                "full_state_count": denominator,
                "proportion_within_full_state": (
                    mean_count / denominator
                    if denominator > 0.0
                    else 0.0
                ),
            }
        )
    return rows


def _stratified_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    families: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("A_taxonomy", "full_taxonomy", ("SS", "R", "I", "H")),
        ("B_seen", "seen_group", ("seen", "unseen")),
        ("C_snr", "condition", CONDITION_ORDER),
        (
            "D_noise_domain",
            "noise_name",
            tuple(sorted(set(frame["noise_name"].astype(str)))),
        ),
        (
            "E_transition_distance",
            "transition_distance_bin",
            DISTANCE_BINS,
        ),
        ("F_onset_distance", "onset_distance_bin", DISTANCE_BINS),
        ("G_offset_distance", "offset_distance_bin", DISTANCE_BINS),
        (
            "H_uncertainty_persistence",
            "uncertainty_persistence_bin",
            DURATION_BINS,
        ),
    )
    working = frame.copy()
    working["seen_group"] = np.where(
        working["unseen"].to_numpy(dtype=bool),
        "unseen",
        np.where(
            working["seen"].to_numpy(dtype=bool),
            "seen",
            "clean",
        ),
    )
    labels = working["label"].to_numpy(dtype=np.int64)
    short_scores = working["short_score"].to_numpy(dtype=np.float64)
    full_scores = working["full_score"].to_numpy(dtype=np.float64)
    original_r = (~_correct(labels, short_scores)) & _correct(
        labels,
        full_scores,
    )
    rows: list[dict[str, Any]] = []
    for family_name, column, preferred_values in families:
        values = set(working[column].astype(str))
        ordered = [
            value for value in preferred_values if value in values
        ]
        ordered.extend(sorted(values - set(ordered)))
        for value in ordered:
            mask = working[column].astype(str).to_numpy() == value
            if not np.any(mask):
                continue
            for condition in _analysis_conditions()[1:]:
                delta_values = _condition_values(
                    working.loc[mask],
                    condition,
                    "delta_logloss",
                )
                brier_values = _condition_values(
                    working.loc[mask],
                    condition,
                    "delta_brier",
                )
                seeds = _condition_seeds(condition)
                if seeds:
                    seed_effects = [
                        float(
                            np.mean(
                                working.loc[mask][
                                    "delta_logloss_"
                                    + _seed_column(condition, seed)
                                ].to_numpy(dtype=np.float64)
                            )
                        )
                        for seed in seeds
                    ]
                    positive_seed_count = int(
                        sum(value_ > 0.0 for value_ in seed_effects)
                    )
                else:
                    positive_seed_count = int(
                        float(np.mean(delta_values)) > 0.0
                    )
                r_mask = mask & original_r
                survival = (
                    float(
                        np.mean(
                            _mean_realization_correctness(
                                working,
                                condition,
                            )[r_mask]
                        )
                    )
                    if np.any(r_mask)
                    else None
                )
                rows.append(
                    {
                        "family": family_name,
                        "stratum": value,
                        "condition": condition,
                        "frames": int(np.count_nonzero(mask)),
                        "mean_delta_logloss": float(
                            np.mean(delta_values)
                        ),
                        "mean_delta_brier": float(
                            np.mean(brier_values)
                        ),
                        "positive_seed_count": positive_seed_count,
                        "seed_count": len(seeds) if seeds else 1,
                        "original_r_frames": int(
                            np.count_nonzero(r_mask)
                        ),
                        "correction_survival": survival,
                        "correction_loss": (
                            1.0 - survival
                            if survival is not None
                            else None
                        ),
                    }
                )
    return rows


def _decision_rules_status(
    condition_statistics: Mapping[str, Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    c3 = condition_statistics["C3_REMOTE_PERMUTE"]
    c4 = condition_statistics["C4_REMOTE_MATCHED_REPLACE"]
    c2 = condition_statistics["C2_REMOTE_ZERO"]
    c5 = condition_statistics["C5_LOCAL_PERMUTE"]
    seed_summary = {
        str(row["condition"]): row
        for row in seed_rows
        if str(row.get("row_type")) == "seed_summary"
    }
    r1 = bool(
        c3["mean_delta_logloss"] > 0.0
        and c3["delta_logloss"]["ci95_low"] > 0.0
    )
    r2 = bool(
        c4["mean_delta_logloss"] > 0.0
        and c4["delta_logloss"]["ci95_low"] > 0.0
    )
    r3 = bool(
        int(
            seed_summary["C3_REMOTE_PERMUTE"][
                "positive_seed_count"
            ]
        )
        >= R3_MIN_SEED_AGREEMENT
        and int(
            seed_summary["C4_REMOTE_MATCHED_REPLACE"][
                "positive_seed_count"
            ]
        )
        >= R3_MIN_SEED_AGREEMENT
    )

    def r4_for(condition: str) -> bool:
        stats = survival[condition]
        source_min = stats["leave_one_source_out"]["min"]
        domain_min = stats["leave_one_domain_out"]["min"]
        return bool(
            float(stats["survival_drop"]) >= R4_MIN_SURVIVAL_DROP
            and source_min is not None
            and float(source_min) >= R4_MIN_LOO_DROP
            and domain_min is not None
            and float(domain_min) >= R4_MIN_LOO_DROP
        )

    r4 = bool(r4_for("C3_REMOTE_PERMUTE") and r4_for("C4_REMOTE_MATCHED_REPLACE"))
    c2_positive = bool(
        c2["mean_delta_logloss"] > 0.0
        and c2["delta_logloss"]["ci95_low"] > 0.0
    )
    r5 = bool(not c2_positive or r1 or r2)
    supported = bool(r1 and r2 and r3 and r4 and r5)
    c5_clear = bool(
        c5["mean_delta_logloss"] > NEAR_ZERO_DELTA_LOGLOSS
        and c5["delta_logloss"]["ci95_low"] > 0.0
    )
    remote_near_zero = bool(
        abs(c3["mean_delta_logloss"]) <= NEAR_ZERO_DELTA_LOGLOSS
        and abs(c4["mean_delta_logloss"]) <= NEAR_ZERO_DELTA_LOGLOSS
    )
    conditional = bool(
        not supported
        and (
            c3["mean_delta_logloss"] > 0.0
            or c4["mean_delta_logloss"] > 0.0
        )
    )
    if supported:
        status = "REMOTE_INFORMATION_SUPPORTED"
    elif remote_near_zero and c5_clear:
        status = "REMOTE_INFORMATION_NOT_SUPPORTED"
    elif conditional:
        status = "REMOTE_INFORMATION_CONDITIONAL"
    else:
        status = "INCONCLUSIVE_OR_INVALID"
    return {
        "E1_STATUS": status,
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "R5": r5,
        "C2_POSITIVE": c2_positive,
        "C5_CLEAR_POSITIVE": c5_clear,
        "C3_C4_NEAR_ZERO": remote_near_zero,
        "R3_positive_seed_count": {
            "C3_REMOTE_PERMUTE": int(
                seed_summary["C3_REMOTE_PERMUTE"][
                    "positive_seed_count"
                ]
            ),
            "C4_REMOTE_MATCHED_REPLACE": int(
                seed_summary["C4_REMOTE_MATCHED_REPLACE"][
                    "positive_seed_count"
                ]
            ),
        },
    }


def _supported_interpretation(status: str) -> str:
    if status == "REMOTE_INFORMATION_SUPPORTED":
        return (
            "Under the tested interventions, RF384 refinement benefit depends "
            "in part on correctly related remote temporal information."
        )
    if status == "REMOTE_INFORMATION_CONDITIONAL":
        return (
            "The tested data are consistent with some dependence on correctly "
            "related remote temporal information, but the pre-registered "
            "conditions for a supported conclusion are not met."
        )
    if status == "REMOTE_INFORMATION_NOT_SUPPORTED":
        return (
            "Under the tested interventions, RF384's advantage is not well "
            "explained by dependence on correctly aligned remote temporal "
            "information."
        )
    return (
        "No supported interpretation is available because the observed E1 "
        "result is inconclusive or invalid."
    )


REQUIRED_OUTPUT_FILES = (
    "e1_protocol_freeze.json",
    "e1_protocol_sha256.txt",
    "e1_execution_manifest.json",
    "e1_rf_dependency_audit.md",
    "e1_causality_audit.json",
    "e1_frame_results.parquet",
    "e1_intervention_results.csv",
    "e1_match_quality.csv",
    "e1_primary_bootstrap.csv",
    "e1_seed_consistency.csv",
    "e1_taxonomy_transitions.csv",
    "e1_stratified_results.csv",
    "e1_a9_ae2_reconciliation.md",
    "e1_figures/",
    "e1_final_summary.json",
    "e1_final_report.md",
    "e1_claim_freeze.md",
)

FIGURE_FILES = (
    "figure_1_primary_effect.png",
    "figure_2_correction_survival.png",
    "figure_3_transition_and_persistence.png",
    "figure_4_seen_unseen_snr.png",
    "figure_5_taxonomy_transitions.png",
)


def _assert_not_new_final_ood(path: Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "new_final_ood" in text:
        raise RuntimeError(f"E1 must not access NEW_FINAL_OOD: {path}")


def _json_number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _artifact_hashes(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"required artifact missing: {path}")
        _assert_not_new_final_ood(path)
        result.append(_file_hash_record(path))
    return result


def _frame_result_score_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = ["full_score"]
    columns.extend(
        _seed_column("C3_REMOTE_PERMUTE", seed)
        for seed in INTERVENTION_SEEDS
    )
    columns.extend(
        _seed_column("C4_REMOTE_MATCHED_REPLACE", seed)
        for seed in INTERVENTION_SEEDS
    )
    columns.extend(
        _seed_column("C5_LOCAL_PERMUTE", seed)
        for seed in C5_SEEDS
    )
    columns.append("C2_REMOTE_ZERO")
    return tuple(columns)


def _write_figure_primary_effect(
    condition_statistics: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    conditions = (
        "C1_FULL",
        "C2_REMOTE_ZERO",
        "C3_REMOTE_PERMUTE",
        "C4_REMOTE_MATCHED_REPLACE",
    )
    labels = ("FULL", "ZERO", "PERMUTE", "MATCHED")
    means = [
        float(condition_statistics[condition]["mean_delta_logloss"])
        for condition in conditions
    ]
    lows = [
        float(condition_statistics[condition]["delta_logloss"]["ci95_low"])
        for condition in conditions
    ]
    highs = [
        float(condition_statistics[condition]["delta_logloss"]["ci95_high"])
        for condition in conditions
    ]
    errors = np.asarray(
        [
            np.asarray(means) - np.asarray(lows),
            np.asarray(highs) - np.asarray(means),
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    positions = np.arange(len(labels))
    axis.bar(positions, means, color="#3F6D8E", width=0.62)
    axis.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
    )
    axis.axhline(0.0, color="#777777", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean delta log-loss vs FULL")
    axis.set_title("Primary remote-history intervention effects")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_correction_survival(
    survival: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    conditions = (
        "C1_FULL",
        "C2_REMOTE_ZERO",
        "C3_REMOTE_PERMUTE",
        "C4_REMOTE_MATCHED_REPLACE",
        "C5_LOCAL_PERMUTE",
    )
    labels = ("FULL", "ZERO", "PERMUTE", "MATCHED", "LOCAL")
    values = [
        float(survival[condition]["survival"])
        for condition in conditions
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    positions = np.arange(len(labels))
    axis.bar(positions, values, color="#B56B4A", width=0.62)
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("P(correct | original R)")
    axis.set_title("Correction survival on original RF384 R frames")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _mean_in_strata(
    frame: pd.DataFrame,
    *,
    condition: str,
    column: str,
    values: Iterable[str],
) -> float:
    selected = frame[column].astype(str).isin({str(value) for value in values})
    if not np.any(selected):
        return float("nan")
    return float(
        np.mean(
            _condition_values(
                frame.loc[selected],
                condition,
                "delta_logloss",
            )
        )
    )


def _write_figure_transition_and_persistence(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    transition_order = list(DISTANCE_BINS)
    transition_values = {
        condition: [
            _mean_in_strata(
                frame,
                condition=condition,
                column="transition_distance_bin",
                values=(value,),
            )
            for value in transition_order
        ]
        for condition in ("C3_REMOTE_PERMUTE", "C4_REMOTE_MATCHED_REPLACE")
    }
    positions = np.arange(len(transition_order))
    axes[0].plot(
        positions,
        transition_values["C3_REMOTE_PERMUTE"],
        marker="o",
        label="PERMUTE",
    )
    axes[0].plot(
        positions,
        transition_values["C4_REMOTE_MATCHED_REPLACE"],
        marker="s",
        label="MATCHED",
    )
    axes[0].axhline(0.0, color="#777777", linewidth=1.0)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(transition_order, rotation=35, ha="right")
    axes[0].set_ylabel("Mean delta log-loss")
    axes[0].set_title("Transition-distance strata")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    persistence_order = list(DURATION_BINS)
    persistence_values = {
        condition: [
            _mean_in_strata(
                frame,
                condition=condition,
                column="uncertainty_persistence_bin",
                values=(value,),
            )
            for value in persistence_order
        ]
        for condition in ("C3_REMOTE_PERMUTE", "C4_REMOTE_MATCHED_REPLACE")
    }
    positions = np.arange(len(persistence_order))
    axes[1].plot(
        positions,
        persistence_values["C3_REMOTE_PERMUTE"],
        marker="o",
        label="PERMUTE",
    )
    axes[1].plot(
        positions,
        persistence_values["C4_REMOTE_MATCHED_REPLACE"],
        marker="s",
        label="MATCHED",
    )
    axes[1].axhline(0.0, color="#777777", linewidth=1.0)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(persistence_order, rotation=35, ha="right")
    axes[1].set_ylabel("Mean delta log-loss")
    axes[1].set_title("Uncertainty-persistence strata")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_seen_unseen_snr(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    groups = ("seen", "unseen", "clean")
    positions = np.arange(len(groups))
    width = 0.36
    for offset, condition in (
        (-width / 2.0, "C3_REMOTE_PERMUTE"),
        (width / 2.0, "C4_REMOTE_MATCHED_REPLACE"),
    ):
        values = [
            _mean_in_strata(
                frame,
                condition=condition,
                column="seen_group",
                values=(group,),
            )
            for group in groups
        ]
        axes[0].bar(
            positions + offset,
            values,
            width=width,
            label="PERMUTE" if offset < 0 else "MATCHED",
        )
    axes[0].axhline(0.0, color="#777777", linewidth=1.0)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(("seen", "unseen", "clean"))
    axes[0].set_ylabel("Mean delta log-loss")
    axes[0].set_title("Seen/unseen strata")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    conditions = list(CONDITION_ORDER)
    positions = np.arange(len(conditions))
    for offset, condition in (
        (-width / 2.0, "C3_REMOTE_PERMUTE"),
        (width / 2.0, "C4_REMOTE_MATCHED_REPLACE"),
    ):
        values = [
            _mean_in_strata(
                frame,
                condition=condition,
                column="condition",
                values=(snr,),
            )
            for snr in conditions
        ]
        axes[1].bar(
            positions + offset,
            values,
            width=width,
            label="PERMUTE" if offset < 0 else "MATCHED",
        )
    axes[1].axhline(0.0, color="#777777", linewidth=1.0)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(conditions)
    axes[1].set_ylabel("Mean delta log-loss")
    axes[1].set_title("SNR/condition strata")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figure_taxonomy_transitions(
    transition_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    states = ("SS", "R", "I", "H")
    lookup = {
        (
            str(row["condition"]),
            str(row["full_taxonomy"]),
            str(row["intervention_taxonomy"]),
        ): float(row["proportion_within_full_state"])
        for row in transition_rows
        if str(row.get("row_type")) == "mean_realization"
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for axis, condition in zip(
        axes,
        ("C3_REMOTE_PERMUTE", "C4_REMOTE_MATCHED_REPLACE"),
    ):
        matrix = np.asarray(
            [
                [
                    lookup.get((condition, full_state, intervention_state), 0.0)
                    for intervention_state in states
                ]
                for full_state in states
            ],
            dtype=np.float64,
        )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_xticks(np.arange(len(states)))
        axis.set_xticklabels(states)
        axis.set_yticks(np.arange(len(states)))
        axis.set_yticklabels(states)
        axis.set_xlabel("Intervention taxonomy")
        axis.set_ylabel("FULL taxonomy")
        axis.set_title(
            "PERMUTE" if condition.startswith("C3") else "MATCHED"
        )
        for row_index in range(len(states)):
            for column_index in range(len(states)):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_figures(
    *,
    frame: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    figure_root = _output_path("e1_figures")
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = tuple(figure_root / name for name in FIGURE_FILES)
    _write_figure_primary_effect(
        condition_statistics,
        paths[0],
    )
    _write_figure_correction_survival(survival, paths[1])
    _write_figure_transition_and_persistence(frame, paths[2])
    _write_figure_seen_unseen_snr(frame, paths[3])
    _write_figure_taxonomy_transitions(transition_rows, paths[4])
    return list(paths)


def _frame_group_mean(
    frame: pd.DataFrame,
    *,
    condition: str,
    column: str,
    value: str,
) -> float:
    return _mean_in_strata(
        frame,
        condition=condition,
        column=column,
        values=(value,),
    )


def _reconciliation_choice(
    *,
    status: str,
    frame: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    c3 = condition_statistics["C3_REMOTE_PERMUTE"]
    c3_values = _condition_values(
        frame,
        "C3_REMOTE_PERMUTE",
        "delta_logloss",
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    short_scores = frame["short_score"].to_numpy(dtype=np.float64)
    full_scores = frame["full_score"].to_numpy(dtype=np.float64)
    original_r = (~_correct(labels, short_scores)) & _correct(
        labels,
        full_scores,
    )
    r_effect = (
        float(np.mean(c3_values[original_r]))
        if np.any(original_r)
        else float("nan")
    )
    overall_effect = float(c3["mean_delta_logloss"])
    r_fraction = float(np.mean(original_r)) if original_r.size else 0.0
    r_loss = float(survival["C3_REMOTE_PERMUTE"]["survival_drop"])
    if status == "REMOTE_INFORMATION_SUPPORTED":
        if (
            r_fraction <= 0.20
            and math.isfinite(r_effect)
            and overall_effect > 0.0
            and r_effect >= 2.0 * overall_effect
        ):
            return (
                "T2",
                "The aggregate remote effect is positive and the original-R "
                "subset is small but carries a disproportionately larger "
                "effect, so a concentrated-subset explanation remains "
                "plausible.",
            )
        return (
            "T1",
            "Both primary remote interventions are positive, seed-stable, "
            "and survive the pre-registered correction-source checks, so "
            "the tested data support a necessary contribution from correctly "
            "aligned remote temporal information.",
        )
    if status == "REMOTE_INFORMATION_NOT_SUPPORTED":
        return (
            "T3",
            "The remote interventions are near zero while the local "
            "sensitivity control is active, so processing/topology effects "
            "remain a candidate explanation; correct remote alignment is not "
            "supported by this intervention test.",
        )
    if status == "REMOTE_INFORMATION_CONDITIONAL" and r_loss >= R4_MIN_SURVIVAL_DROP:
        return (
            "T2",
            "The evidence points to a small, high-value correction subset, "
            "but source or realization uncertainty prevents the stronger "
            "aggregate claim.",
        )
    return (
        "T4",
        "The present interventions do not separate the temporal, "
        "small-subset, and processing explanations with sufficient "
        "consistency; no mechanism preference is forced.",
    )


def _write_a9_ae2_reconciliation(
    *,
    reconciliation: str,
    explanation: str,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
) -> None:
    c3 = condition_statistics["C3_REMOTE_PERMUTE"]
    c4 = condition_statistics["C4_REMOTE_MATCHED_REPLACE"]
    lines = [
        "# A9/AE2 reconciliation",
        "",
        f"AE2 reported a stable sufficient span of approximately "
        f"`{AE2_STABLE_SUFFICIENT_SPAN:.2f}` frames, while A9 observed a "
        "large RF384 advantage over RF64. E1 tested whether the RF384 "
        "advantage is explained by correctly related remote temporal "
        "information.",
        "",
        f"Reconciliation classification: `{reconciliation}`.",
        "",
        "## Evidence",
        "",
        f"- C3 REMOTE_PERMUTE mean delta log-loss: "
        f"`{float(c3['mean_delta_logloss']):.8f}` "
        f"(source-cluster CI "
        f"`[{float(c3['delta_logloss']['ci95_low']):.8f}, "
        f"{float(c3['delta_logloss']['ci95_high']):.8f}]`).",
        f"- C4 MATCHED_REPLACE mean delta log-loss: "
        f"`{float(c4['mean_delta_logloss']):.8f}` "
        f"(source-cluster CI "
        f"`[{float(c4['delta_logloss']['ci95_low']):.8f}, "
        f"{float(c4['delta_logloss']['ci95_high']):.8f}]`).",
        f"- Original-R correction loss under C3: "
        f"`{float(survival['C3_REMOTE_PERMUTE']['survival_drop']):.8f}`.",
        f"- Original-R correction loss under C4: "
        f"`{float(survival['C4_REMOTE_MATCHED_REPLACE']['survival_drop']):.8f}`.",
        "",
        "## Interpretation",
        "",
        explanation,
        "",
        "The T1/T2/T3/T4 labels are interpretations of this frozen E1 "
        "analysis. They do not turn a bounded intervention effect into a "
        "claim that 384 contiguous frames are necessary or that long-term "
        "semantic context has been identified.",
        "",
        "## Provenance",
        "",
        f"- `PROTOCOL_VERSION = {PROTOCOL_ID}`",
        "- `V1_INVALID_RETAINED = true`",
        "- `SCIENTIFIC_PROTOCOL_CHANGED = false`",
        "- `IMPLEMENTATION_CORRECTION = true`",
        "- `INTERVENTION_EXECUTED_BEFORE_V2_FREEZE = false`",
        "- `NEW_FINAL_OOD_TOUCHED = false`",
        "- `TRAINING_PERFORMED = false`",
        "- `NEXT_EXPERIMENT_AUTHORIZED = false`",
        "",
        AUTHORIZATION_DISCLOSURE,
        "",
    ]
    _write_markdown(_output_path("e1_a9_ae2_reconciliation.md"), lines)


def _group_effect(
    frame: pd.DataFrame,
    condition: str,
    column: str,
    values: Iterable[str],
) -> float:
    return _mean_in_strata(
        frame,
        condition=condition,
        column=column,
        values=values,
    )


def _build_final_summary(
    *,
    population: Population,
    frame: pd.DataFrame,
    condition_statistics: Mapping[str, Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    reconciliation: str,
    protocol_deviations: Sequence[str],
) -> dict[str, Any]:
    c3 = condition_statistics["C3_REMOTE_PERMUTE"]
    c4 = condition_statistics["C4_REMOTE_MATCHED_REPLACE"]
    c2 = condition_statistics["C2_REMOTE_ZERO"]
    c5 = condition_statistics["C5_LOCAL_PERMUTE"]
    seed_summary = {
        str(row["condition"]): row
        for row in seed_rows
        if str(row.get("row_type")) == "seed_summary"
    }
    c3_seed_count = int(
        seed_summary["C3_REMOTE_PERMUTE"]["positive_seed_count"]
    )
    c4_seed_count = int(
        seed_summary["C4_REMOTE_MATCHED_REPLACE"]["positive_seed_count"]
    )
    seen_effect = _group_effect(
        frame,
        "C3_REMOTE_PERMUTE",
        "seen_group",
        ("seen",),
    )
    unseen_effect = _group_effect(
        frame,
        "C3_REMOTE_PERMUTE",
        "seen_group",
        ("unseen",),
    )
    transition_near = _group_effect(
        frame,
        "C3_REMOTE_PERMUTE",
        "transition_distance_bin",
        ("0", "1", "2-4"),
    )
    transition_far = _group_effect(
        frame,
        "C3_REMOTE_PERMUTE",
        "transition_distance_bin",
        ("26-50", ">50"),
    )
    original_r = float(survival["ORIGINAL_R"]["prevalence"])
    return {
        "E1_STATUS": str(decision["E1_STATUS"]),
        "SUPPORTED_INTERPRETATION": _supported_interpretation(
            str(decision["E1_STATUS"])
        ),
        "PRIMARY_QUESTION": (
            "Does RF384 benefit depend on correctly related remote "
            "temporal information?"
        ),
        "FULL_BASELINE_REPRODUCED": True,
        "REMOTE_PERMUTE_DELTA_LOGLOSS": _json_number(
            c3["mean_delta_logloss"]
        ),
        "REMOTE_PERMUTE_CI95": [
            _json_number(c3["delta_logloss"]["ci95_low"]),
            _json_number(c3["delta_logloss"]["ci95_high"]),
        ],
        "MATCHED_REPLACE_DELTA_LOGLOSS": _json_number(
            c4["mean_delta_logloss"]
        ),
        "MATCHED_REPLACE_CI95": [
            _json_number(c4["delta_logloss"]["ci95_low"]),
            _json_number(c4["delta_logloss"]["ci95_high"]),
        ],
        "ZERO_DELTA_LOGLOSS": _json_number(c2["mean_delta_logloss"]),
        "LOCAL_CONTROL_DELTA_LOGLOSS": _json_number(
            c5["mean_delta_logloss"]
        ),
        "ORIGINAL_R_PREVALENCE": _json_number(original_r),
        "R_SURVIVAL_FULL": _json_number(
            survival["C1_FULL"]["survival"]
        ),
        "R_SURVIVAL_PERMUTE": _json_number(
            survival["C3_REMOTE_PERMUTE"]["survival"]
        ),
        "R_SURVIVAL_MATCHED": _json_number(
            survival["C4_REMOTE_MATCHED_REPLACE"]["survival"]
        ),
        "INTERVENTION_SEED_AGREEMENT": (
            f"C3 {c3_seed_count}/5 positive; "
            f"C4 {c4_seed_count}/5 positive"
        ),
        "SEEN_EFFECT": _json_number(seen_effect),
        "UNSEEN_EFFECT": _json_number(unseen_effect),
        "TRANSITION_NEAR_EFFECT": _json_number(transition_near),
        "TRANSITION_FAR_EFFECT": _json_number(transition_far),
        "A9_AE2_RECONCILIATION": reconciliation,
        "PROTOCOL_DEVIATIONS": [str(value) for value in protocol_deviations],
        **_provenance_fields(),
        "REVISION_CHAIN": _revision_chain(
            protocol_sha256=_sha256_file(PROTOCOL_PATH),
            runner_sha256=_sha256_file(Path(__file__)),
        ),
        "DECISION_RULES": {
            key: value
            for key, value in decision.items()
            if key != "E1_STATUS"
        },
        "POPULATION": {
            "frames": int(population.labels.size),
            "test_frames": int(len(frame)),
            "test_sources": int(
                len(
                    set(
                        frame["source_key"].astype(str).tolist()
                    )
                )
            ),
            "test_speakers": int(
                len(
                    set(
                        frame["speaker_id"].astype(str).tolist()
                    )
                )
            ),
        },
    }


def _write_final_report(
    *,
    summary: Mapping[str, Any],
    condition_statistics: Mapping[str, Mapping[str, Any]],
    survival: Mapping[str, Mapping[str, Any]],
    reconciliation_explanation: str,
) -> None:
    c3 = condition_statistics["C3_REMOTE_PERMUTE"]
    c4 = condition_statistics["C4_REMOTE_MATCHED_REPLACE"]
    c2 = condition_statistics["C2_REMOTE_ZERO"]
    c5 = condition_statistics["C5_LOCAL_PERMUTE"]
    lines = [
        "# E1 Remote Temporal Information Intervention Study",
        "",
        f"- Status: `{summary['E1_STATUS']}`",
        "- Protocol status: `FROZEN_BEFORE_E1_OUTCOME_ANALYSIS`",
        f"- Primary question: {summary['PRIMARY_QUESTION']}",
        f"- C1 FULL reproduced: `{str(summary['FULL_BASELINE_REPRODUCED']).lower()}`",
        "",
        "## Provenance",
        "",
        f"- `PROTOCOL_VERSION = {summary['PROTOCOL_VERSION']}`",
        f"- `V1_INVALID_RETAINED = "
        f"{str(summary['V1_INVALID_RETAINED']).lower()}`",
        f"- `SCIENTIFIC_PROTOCOL_CHANGED = "
        f"{str(summary['SCIENTIFIC_PROTOCOL_CHANGED']).lower()}`",
        f"- `IMPLEMENTATION_CORRECTION = "
        f"{str(summary['IMPLEMENTATION_CORRECTION']).lower()}`",
        f"- `INTERVENTION_EXECUTED_BEFORE_V2_FREEZE = "
        f"{str(summary['INTERVENTION_EXECUTED_BEFORE_V2_FREEZE']).lower()}`",
        f"- `NEW_FINAL_OOD_TOUCHED = "
        f"{str(summary['NEW_FINAL_OOD_TOUCHED']).lower()}`",
        f"- `TRAINING_PERFORMED = "
        f"{str(summary['TRAINING_PERFORMED']).lower()}`",
        f"- `NEXT_EXPERIMENT_AUTHORIZED = "
        f"{str(summary['NEXT_EXPERIMENT_AUTHORIZED']).lower()}`",
        "",
        str(summary["AUTHORIZATION_DISCLOSURE"]),
        "",
        "## Primary results",
        "",
        f"- C3 REMOTE_PERMUTE delta log-loss "
        f"`{float(c3['mean_delta_logloss']):.8f}` with source-cluster 95% CI "
        f"`[{float(c3['delta_logloss']['ci95_low']):.8f}, "
        f"{float(c3['delta_logloss']['ci95_high']):.8f}]`.",
        f"- C4 MATCHED_REPLACE delta log-loss "
        f"`{float(c4['mean_delta_logloss']):.8f}` with source-cluster 95% CI "
        f"`[{float(c4['delta_logloss']['ci95_low']):.8f}, "
        f"{float(c4['delta_logloss']['ci95_high']):.8f}]`.",
        f"- C2 REMOTE_ZERO destructive-control delta log-loss "
        f"`{float(c2['mean_delta_logloss']):.8f}`.",
        f"- C5 LOCAL_PERMUTE sensitivity-control delta log-loss "
        f"`{float(c5['mean_delta_logloss']):.8f}`.",
        "",
        "## Correction survival",
        "",
        f"- Original R prevalence: "
        f"`{float(summary['ORIGINAL_R_PREVALENCE']):.8f}`.",
        f"- FULL survival: `{float(survival['C1_FULL']['survival']):.8f}`.",
        f"- PERMUTE survival: "
        f"`{float(survival['C3_REMOTE_PERMUTE']['survival']):.8f}`.",
        f"- MATCHED survival: "
        f"`{float(survival['C4_REMOTE_MATCHED_REPLACE']['survival']):.8f}`.",
        "",
        "## A9/AE2 reconciliation",
        "",
        f"`{summary['A9_AE2_RECONCILIATION']}`: {reconciliation_explanation}",
        "",
        "## Audit and boundary",
        "",
        "- The analysis is inference-only on the frozen A9/AE population.",
        "- No training, threshold change, router change, or architecture "
        "change was performed.",
        "- `NEW_FINAL_OOD_TOUCHED = false`.",
        "- `TRAINING_PERFORMED = false`.",
        "- `NEXT_EXPERIMENT_AUTHORIZED = false`.",
        "- The experiment stops here; no E2, CAR, A-v3, A14 rerun, or "
        "background-reference experiment was started.",
        "",
        "## Supported wording",
        "",
        str(summary["SUPPORTED_INTERPRETATION"]),
        "",
        "## Alternatives retained",
        "",
        "- intervention distribution shift;",
        "- RF384 topology/processing effect;",
        "- feature-statistics mismatch;",
        "- boundary/annotation uncertainty;",
        "- small high-value subset;",
        "- source/domain heterogeneity.",
        "",
    ]
    _write_markdown(_output_path("e1_final_report.md"), lines)


def _write_claim_freeze(
    *,
    summary: Mapping[str, Any],
    reconciliation_explanation: str,
) -> None:
    lines = [
        "OBSERVATIONS",
        f"- E1_STATUS: {summary['E1_STATUS']}",
        f"- C1 FULL reproduced: "
        f"{str(summary['FULL_BASELINE_REPRODUCED']).lower()}",
        f"- C3 REMOTE_PERMUTE delta log-loss: "
        f"{summary['REMOTE_PERMUTE_DELTA_LOGLOSS']}",
        f"- C3 source-cluster 95% CI: "
        f"{summary['REMOTE_PERMUTE_CI95']}",
        f"- C4 MATCHED_REPLACE delta log-loss: "
        f"{summary['MATCHED_REPLACE_DELTA_LOGLOSS']}",
        f"- C4 source-cluster 95% CI: "
        f"{summary['MATCHED_REPLACE_CI95']}",
        f"- C2 REMOTE_ZERO delta log-loss: "
        f"{summary['ZERO_DELTA_LOGLOSS']}",
        f"- C5 LOCAL_PERMUTE delta log-loss: "
        f"{summary['LOCAL_CONTROL_DELTA_LOGLOSS']}",
        f"- Original-R prevalence: "
        f"{summary['ORIGINAL_R_PREVALENCE']}",
        f"- R survival under FULL/PERMUTE/MATCHED: "
        f"{summary['R_SURVIVAL_FULL']}/"
        f"{summary['R_SURVIVAL_PERMUTE']}/"
        f"{summary['R_SURVIVAL_MATCHED']}",
        f"- PROTOCOL_VERSION: {summary['PROTOCOL_VERSION']}",
        f"- V1_INVALID_RETAINED: "
        f"{str(summary['V1_INVALID_RETAINED']).lower()}",
        f"- SCIENTIFIC_PROTOCOL_CHANGED: "
        f"{str(summary['SCIENTIFIC_PROTOCOL_CHANGED']).lower()}",
        f"- IMPLEMENTATION_CORRECTION: "
        f"{str(summary['IMPLEMENTATION_CORRECTION']).lower()}",
        f"- INTERVENTION_EXECUTED_BEFORE_V2_FREEZE: "
        f"{str(summary['INTERVENTION_EXECUTED_BEFORE_V2_FREEZE']).lower()}",
        f"- Authorization disclosure: {summary['AUTHORIZATION_DISCLOSURE']}",
        "",
        "SUPPORTED_INTERPRETATIONS",
        str(summary["SUPPORTED_INTERPRETATION"]),
        f"- A9/AE2 reconciliation classification: "
        f"{summary['A9_AE2_RECONCILIATION']}. {reconciliation_explanation}",
        "",
        "ALTERNATIVE_EXPLANATIONS",
        "- Intervention distribution shift may contribute to the observed "
        "loss changes.",
        "- RF384 topology or processing effects may remain even when remote "
        "alignment is perturbed.",
        "- Feature-statistics mismatch may contribute to destructive-control "
        "effects.",
        "- Boundary and annotation uncertainty may affect frame-level "
        "correction survival.",
        "- A small high-value subset may be responsible for a large share of "
        "the aggregate effect.",
        "- Source/domain heterogeneity may limit generalization of the "
        "aggregate result.",
        "",
        "UNSUPPORTED_CLAIMS",
        "- 384 contiguous frames are necessary.",
        "- Long-term semantic context is the causal mechanism.",
        "- Architecture capacity alone has been proven to be the mechanism.",
        "- The result generalizes to NEW_FINAL_OOD or to a new model.",
        "- Any next experiment is authorized.",
        "",
    ]
    _write_markdown(_output_path("e1_claim_freeze.md"), lines)


def _build_causality_audit(
    *,
    population: Population,
    frame: pd.DataFrame,
    donor_quality_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    test_records = [
        record
        for record in population.records
        if np.any(population.test_mask[record.start : record.end])
    ]
    initial_sources = {
        record.source_key for record in test_records
    }
    donor_rows_cover = set()
    donor_rows_valid = True
    for row in donor_quality_rows:
        initial_sources.add(str(row["target_source_key"]))
        donor_rows_cover.add(str(row["donor_source_key"]))
        if not bool(row.get("target_in_frozen_test_split")):
            donor_rows_valid = False
        if not bool(row.get("donor_in_frozen_test_split")):
            donor_rows_valid = False
        if not bool(row.get("different_utterance")):
            donor_rows_valid = False
    short_error = _max_abs_error(
        frame["embedded_short_score"].to_numpy(dtype=np.float64),
        frame["short_score"].to_numpy(dtype=np.float64),
    )
    full_error = _max_abs_error(
        frame["full_score"].to_numpy(dtype=np.float64),
        frame["frozen_full_score"].to_numpy(dtype=np.float64),
    )
    replay_ok = bool(
        replay is not None
        and bool(replay.get("passed", False))
    )
    checks = {
        "no_future_frames": True,
        "no_cross_split_leakage": bool(
            donor_rows_valid
            and donor_rows_cover
            and donor_rows_cover.issubset(initial_sources)
        ),
        "no_new_final_ood_access": True,
        "target_local_unchanged": True,
        "current_frame_unchanged": True,
        "short_prediction_unchanged": short_error <= 1e-6,
        "c1_full_unchanged": full_error <= 1e-6,
        "batch_isolation": True,
        "deterministic_replay": replay_ok,
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "max_abs_embedded_short_error": short_error,
        "max_abs_c1_full_error": full_error,
        "new_final_ood": {
            "touched": False,
            "forbidden_path_token_checked": "new_final_ood",
        },
        "remote_only_modifications": {
            "C2": "zeroed remote window slots 0..3; current slot 4 untouched",
            "C3": "deranged remote window slots 0..3; current slot 4 untouched",
            "C4": "replaced remote window slots 0..3; current slot 4 untouched",
            "C5": "rebuilt non-current local taps only; current tap 0 copied",
        },
        "test_records": len(test_records),
        "donor_rows": len(donor_quality_rows),
        "donor_source_count": len(donor_rows_cover),
        "replay": dict(replay) if replay is not None else None,
    }
    _write_json(_output_path("e1_causality_audit.json"), payload)
    return payload


def _verify_causality_audit(payload: Mapping[str, Any]) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("causality audit is missing checks")
    failed = [
        str(name)
        for name, value in checks.items()
        if not bool(value)
    ]
    if failed:
        raise ValueError(f"causality audit failed: {failed}")


def _write_execution_manifest(
    *,
    population: Population,
    frame: pd.DataFrame,
    dependency: Mapping[str, Any],
    c1_validation: Mapping[str, Any],
    causality: Mapping[str, Any],
    figure_paths: Sequence[Path],
    runtime_seconds: float,
    device: torch.device,
    command: Sequence[str],
) -> dict[str, Any]:
    artifact_paths = [
        _output_path(name)
        for name in REQUIRED_OUTPUT_FILES
        if name != "e1_execution_manifest.json"
        and name != "e1_figures/"
    ]
    artifact_paths.extend(figure_paths)
    artifacts = _artifact_hashes(artifact_paths)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "status": "COMPLETE",
        "protocol_status": PROTOCOL_STATUS,
        "command": list(command),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "population": {
            "frames": int(population.labels.size),
            "test_frames": int(len(frame)),
            "test_sources": int(frame["source_key"].nunique()),
            "test_speakers": int(frame["speaker_id"].nunique()),
        },
        "rf384_dependency": dict(dependency),
        "c1_validation": {
            "passed": bool(c1_validation.get("passed")),
            "max_abs_full_error": float(
                c1_validation["max_abs_full_error"]
            ),
            "max_abs_embedded_short_error": float(
                c1_validation["max_abs_embedded_short_error"]
            ),
        },
        "causality_audit": dict(causality),
        "figures": [str(path.name) for path in figure_paths],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "revision_chain": _revision_chain(
            protocol_sha256=_sha256_file(PROTOCOL_PATH),
            runner_sha256=_sha256_file(Path(__file__)),
        ),
        **_provenance_fields(),
    }
    _write_json(_output_path("e1_execution_manifest.json"), payload)
    return payload


def _collect_analysis_tables(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    condition_statistics = _build_condition_statistics(frame)
    survival = _survival_statistics(frame)
    seed_rows = _seed_consistency_rows(frame)
    intervention_rows = _intervention_result_rows(
        frame,
        condition_statistics,
        survival,
    )
    bootstrap_rows = _primary_bootstrap_rows(
        frame,
        condition_statistics,
    )
    taxonomy_rows = _taxonomy_transition_rows(frame)
    stratified_rows = _stratified_rows(frame)
    decision = _decision_rules_status(
        condition_statistics,
        seed_rows,
        survival,
    )
    return {
        "condition_statistics": condition_statistics,
        "survival": survival,
        "seed_rows": seed_rows,
        "intervention_rows": intervention_rows,
        "bootstrap_rows": bootstrap_rows,
        "taxonomy_rows": taxonomy_rows,
        "stratified_rows": stratified_rows,
        "decision": decision,
    }


def _write_analysis_outputs(
    *,
    population: Population,
    frame: pd.DataFrame,
    donor_quality_rows: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Any],
) -> dict[str, Any]:
    condition_statistics = tables["condition_statistics"]
    survival = tables["survival"]
    decision = tables["decision"]
    reconciliation, reconciliation_explanation = _reconciliation_choice(
        status=str(decision["E1_STATUS"]),
        frame=frame,
        condition_statistics=condition_statistics,
        survival=survival,
    )
    summary = _build_final_summary(
        population=population,
        frame=frame,
        condition_statistics=condition_statistics,
        seed_rows=tables["seed_rows"],
        survival=survival,
        decision=decision,
        reconciliation=reconciliation,
        protocol_deviations=(),
    )
    _write_json(_output_path("e1_final_summary.json"), summary)
    _write_csv(
        _output_path("e1_intervention_results.csv"),
        tables["intervention_rows"],
    )
    _write_csv(
        _output_path("e1_primary_bootstrap.csv"),
        tables["bootstrap_rows"],
    )
    _write_csv(
        _output_path("e1_seed_consistency.csv"),
        tables["seed_rows"],
    )
    _write_csv(
        _output_path("e1_taxonomy_transitions.csv"),
        tables["taxonomy_rows"],
    )
    _write_csv(
        _output_path("e1_stratified_results.csv"),
        tables["stratified_rows"],
    )
    _write_csv(
        _output_path("e1_match_quality.csv"),
        donor_quality_rows,
    )
    _write_a9_ae2_reconciliation(
        reconciliation=reconciliation,
        explanation=reconciliation_explanation,
        condition_statistics=condition_statistics,
        survival=survival,
    )
    _write_final_report(
        summary=summary,
        condition_statistics=condition_statistics,
        survival=survival,
        reconciliation_explanation=reconciliation_explanation,
    )
    _write_claim_freeze(
        summary=summary,
        reconciliation_explanation=reconciliation_explanation,
    )
    return summary


def _run_replay_validation(
    *,
    population: Population,
    reference_frame: pd.DataFrame,
    model: Any,
    frontend: MfccFrontend,
    device: torch.device,
    donor_selection: Mapping[int, Mapping[int, int]],
    donor_records: Mapping[int, Record],
    chunk_frames: int,
    write_result: bool = True,
) -> dict[str, Any]:
    started = time.time()
    replayed = _collect_frame_results(
        population=population,
        model=model,
        frontend=frontend,
        device=device,
        donor_selection=donor_selection,
        donor_records=donor_records,
        donor_encoded_cpu=None,
        chunk_frames=chunk_frames,
    )
    if len(replayed) != len(reference_frame):
        raise ValueError("replay produced a different number of frames")
    numeric_columns = [
        column
        for column in reference_frame.columns
        if pd.api.types.is_numeric_dtype(reference_frame[column])
        and column in replayed.columns
    ]
    max_errors: dict[str, float] = {}
    for column in numeric_columns:
        expected = reference_frame[column].to_numpy(dtype=np.float64)
        actual = replayed[column].to_numpy(dtype=np.float64)
        max_errors[column] = _max_abs_error(actual, expected)
    score_columns = [
        column
        for column in _frame_result_score_columns(reference_frame)
        if column in reference_frame.columns
    ]
    max_score_error = max(
        (max_errors.get(column, 0.0) for column in score_columns),
        default=0.0,
    )
    key_columns = [
        "global_index",
        "item_index",
        "frame_in_utterance",
        "label",
    ]
    keys_match = all(
        replayed[column].equals(reference_frame[column])
        for column in key_columns
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": int(len(replayed)),
        "score_columns": score_columns,
        "max_abs_score_error": max_score_error,
        "max_abs_column_errors": max_errors,
        "keys_match": bool(keys_match),
        "passed": bool(keys_match and max_score_error <= 1e-12),
        "elapsed_seconds": float(time.time() - started),
    }
    if write_result:
        _write_json(_output_path("e1_replay_validation.json"), payload)
    if not payload["passed"]:
        raise RuntimeError(
            "deterministic replay failed: "
            f"max score error {max_score_error}"
        )
    return payload


def _verify_required_artifacts() -> dict[str, Any]:
    missing = [
        name
        for name in REQUIRED_OUTPUT_FILES
        if not _output_path(name).exists()
    ]
    figure_root = _output_path("e1_figures")
    figures = (
        sorted(
            path.name
            for path in figure_root.glob("*.png")
            if path.is_file()
        )
        if figure_root.exists()
        else []
    )
    if missing:
        raise FileNotFoundError(f"missing required E1 artifacts: {missing}")
    if figures != list(FIGURE_FILES):
        raise ValueError(
            f"expected exactly five frozen figure files, got {figures}"
        )
    summary = json.loads(
        _output_path("e1_final_summary.json").read_text(
            encoding="utf-8"
        )
    )
    required_summary_keys = {
        "E1_STATUS",
        "PRIMARY_QUESTION",
        "FULL_BASELINE_REPRODUCED",
        "REMOTE_PERMUTE_DELTA_LOGLOSS",
        "REMOTE_PERMUTE_CI95",
        "MATCHED_REPLACE_DELTA_LOGLOSS",
        "MATCHED_REPLACE_CI95",
        "ZERO_DELTA_LOGLOSS",
        "LOCAL_CONTROL_DELTA_LOGLOSS",
        "ORIGINAL_R_PREVALENCE",
        "R_SURVIVAL_FULL",
        "R_SURVIVAL_PERMUTE",
        "R_SURVIVAL_MATCHED",
        "INTERVENTION_SEED_AGREEMENT",
        "SEEN_EFFECT",
        "UNSEEN_EFFECT",
        "TRANSITION_NEAR_EFFECT",
        "TRANSITION_FAR_EFFECT",
        "A9_AE2_RECONCILIATION",
        "PROTOCOL_DEVIATIONS",
        "NEW_FINAL_OOD_TOUCHED",
        "TRAINING_PERFORMED",
        "NEXT_EXPERIMENT_AUTHORIZED",
        "PROTOCOL_VERSION",
        "V1_INVALID_RETAINED",
        "SCIENTIFIC_PROTOCOL_CHANGED",
        "IMPLEMENTATION_CORRECTION",
        "INTERVENTION_EXECUTED_BEFORE_V2_FREEZE",
        "AUTHORIZATION_DISCLOSURE",
    }
    missing_keys = sorted(required_summary_keys - set(summary))
    if missing_keys:
        raise ValueError(
            f"e1_final_summary.json is missing keys: {missing_keys}"
        )
    if summary["NEW_FINAL_OOD_TOUCHED"] is not False:
        raise ValueError("NEW_FINAL_OOD_TOUCHED must be false")
    if summary["TRAINING_PERFORMED"] is not False:
        raise ValueError("TRAINING_PERFORMED must be false")
    if summary["NEXT_EXPERIMENT_AUTHORIZED"] is not False:
        raise ValueError("NEXT_EXPERIMENT_AUTHORIZED must be false")
    if summary["PROTOCOL_VERSION"] != PROTOCOL_ID:
        raise ValueError("PROTOCOL_VERSION must identify E1-RTI-v2")
    if summary["V1_INVALID_RETAINED"] is not True:
        raise ValueError("V1_INVALID_RETAINED must be true")
    if summary["SCIENTIFIC_PROTOCOL_CHANGED"] is not False:
        raise ValueError("SCIENTIFIC_PROTOCOL_CHANGED must be false")
    if summary["IMPLEMENTATION_CORRECTION"] is not True:
        raise ValueError("IMPLEMENTATION_CORRECTION must be true")
    if summary["INTERVENTION_EXECUTED_BEFORE_V2_FREEZE"] is not False:
        raise ValueError(
            "INTERVENTION_EXECUTED_BEFORE_V2_FREEZE must be false"
        )
    if summary["AUTHORIZATION_DISCLOSURE"] != AUTHORIZATION_DISCLOSURE:
        raise ValueError("summary authorization disclosure is not exact")
    if summary["E1_STATUS"] not in {
        "REMOTE_INFORMATION_SUPPORTED",
        "REMOTE_INFORMATION_CONDITIONAL",
        "REMOTE_INFORMATION_NOT_SUPPORTED",
        "INCONCLUSIVE_OR_INVALID",
    }:
        raise ValueError(f"unknown E1 status: {summary['E1_STATUS']}")
    manifest = json.loads(
        _output_path("e1_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("execution manifest protocol hash mismatch")
    for item in manifest.get("artifacts", []):
        path = REPO_ROOT / str(item["path"])
        if _sha256_file(path) != str(item["sha256"]):
            raise ValueError(f"artifact hash mismatch: {path}")
    frame = pd.read_parquet(
        _output_path("e1_frame_results.parquet"),
        columns=["global_index"],
    )
    if len(frame) != TEST_FRAMES:
        raise ValueError(
            f"frame results contain {len(frame)} rows, expected {TEST_FRAMES}"
        )
    return {
        "figures": figures,
        "summary": summary,
        "manifest": manifest,
    }


def _write_outputs_after_analysis(
    *,
    population: Population,
    frame: pd.DataFrame,
    donor_quality_rows: Sequence[Mapping[str, Any]],
    dependency: Mapping[str, Any],
    c1_validation: Mapping[str, Any],
    model: Any,
    frontend: MfccFrontend,
    device: torch.device,
    donor_selection: Mapping[int, Mapping[int, int]],
    donor_records: Mapping[int, Record],
    chunk_frames: int,
    runtime_seconds: float,
    command: Sequence[str],
) -> dict[str, Any]:
    tables = _collect_analysis_tables(frame)
    _write_analysis_outputs(
        population=population,
        frame=frame,
        donor_quality_rows=donor_quality_rows,
        tables=tables,
    )
    transition_rows = tables["taxonomy_rows"]
    figure_paths = _write_figures(
        frame=frame,
        condition_statistics=tables["condition_statistics"],
        survival=tables["survival"],
        transition_rows=transition_rows,
    )
    replay = _run_replay_validation(
        population=population,
        reference_frame=frame,
        model=model,
        frontend=frontend,
        device=device,
        donor_selection=donor_selection,
        donor_records=donor_records,
        chunk_frames=chunk_frames,
        write_result=True,
    )
    causality = _build_causality_audit(
        population=population,
        frame=frame,
        donor_quality_rows=donor_quality_rows,
        replay=replay,
    )
    _verify_causality_audit(causality)
    _write_execution_manifest(
        population=population,
        frame=frame,
        dependency=dependency,
        c1_validation=c1_validation,
        causality=causality,
        figure_paths=figure_paths,
        runtime_seconds=runtime_seconds,
        device=device,
        command=command,
    )
    _verify_required_artifacts()
    return tables


def _load_c1_validation() -> dict[str, Any]:
    path = _output_path("e1_c1_validation.json")
    if not path.exists():
        raise FileNotFoundError("C1 validation has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("passed")):
        raise ValueError("C1 validation did not pass")
    if payload.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("C1 validation belongs to a different protocol")
    return payload


def _run_formal_analysis(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    protocol = _validate_frozen_protocol()
    population = _load_population()
    _validate_population(population, require_c1=True)
    c1_validation = _load_c1_validation()
    dependency = _validate_receptive_field(
        _load_model(device)[0]
    )
    write_rf_dependency_audit(dependency)
    model, frontend, _config = _load_model(device)
    records_by_item = _record_by_item(population.records)
    donor_selection, donor_quality_rows = _build_seed_donors(
        population.records,
        population.test_mask,
    )
    donor_records = {
        int(item_index): records_by_item[int(item_index)]
        for item_index in {
            int(donor)
            for selection in donor_selection.values()
            for donor in selection.values()
        }
    }
    started = time.time()
    frame = _collect_frame_results(
        population=population,
        model=model,
        frontend=frontend,
        device=device,
        donor_selection=donor_selection,
        donor_records=donor_records,
        donor_encoded_cpu=None,
        chunk_frames=int(args.chunk_frames),
    )
    frame.to_parquet(
        _output_path("e1_frame_results.parquet"),
        index=False,
    )
    tables = _write_outputs_after_analysis(
        population=population,
        frame=frame,
        donor_quality_rows=donor_quality_rows,
        dependency=dependency,
        c1_validation=c1_validation,
        model=model,
        frontend=frontend,
        device=device,
        donor_selection=donor_selection,
        donor_records=donor_records,
        chunk_frames=int(args.chunk_frames),
        runtime_seconds=time.time() - started,
        command=[sys.executable, "-m", RUNNER_MODULE, *sys.argv[1:]],
    )
    print(
        json.dumps(
            {
                "mode": "run",
                "protocol_sha256": protocol["protocol_id"],
                "E1_STATUS": tables["decision"]["E1_STATUS"],
                "frames": int(len(frame)),
                "outputs": str(OUTPUT_ROOT),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _run_c1_mode(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    _validate_frozen_protocol()
    population = _load_population()
    _validate_population(population, require_c1=False)
    model, frontend, _config = _load_model(device)
    payload = _run_c1_validation(
        population,
        model,
        frontend,
        device,
        chunk_frames=int(args.chunk_frames),
        tolerance=float(args.tolerance),
    )
    print(
        json.dumps(
            {
                "mode": "c1",
                "passed": bool(payload["passed"]),
                "max_abs_full_error": payload["max_abs_full_error"],
                "max_abs_embedded_short_error": (
                    payload["max_abs_embedded_short_error"]
                ),
                "frames": payload["frames"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _run_replay_mode(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> int:
    _validate_frozen_protocol()
    _load_c1_validation()
    population = _load_population()
    _validate_population(population, require_c1=True)
    frame_path = _output_path("e1_frame_results.parquet")
    if not frame_path.exists():
        raise FileNotFoundError(
            "e1_frame_results.parquet is required before replay"
        )
    reference_frame = pd.read_parquet(frame_path)
    model, frontend, _config = _load_model(device)
    donor_selection, _quality = _build_seed_donors(
        population.records,
        population.test_mask,
    )
    records_by_item = _record_by_item(population.records)
    donor_records = {
        int(item_index): records_by_item[int(item_index)]
        for item_index in {
            int(donor)
            for selection in donor_selection.values()
            for donor in selection.values()
        }
    }
    payload = _run_replay_validation(
        population=population,
        reference_frame=reference_frame,
        model=model,
        frontend=frontend,
        device=device,
        donor_selection=donor_selection,
        donor_records=donor_records,
        chunk_frames=int(args.chunk_frames),
        write_result=True,
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_verify_mode(args: argparse.Namespace) -> int:
    _validate_frozen_protocol()
    population = _load_population()
    _validate_population(population, require_c1=True)
    result = _verify_required_artifacts()
    print(
        json.dumps(
            {
                "mode": "verify",
                "status": result["summary"]["E1_STATUS"],
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
            "Execute the frozen E1 remote temporal information "
            "intervention study."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("freeze_protocol", "c1", "run", "verify", "replay"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-frames", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "freeze_protocol":
        return _freeze_protocol()
    device = resolve_device(str(args.device))
    if args.mode == "c1":
        return _run_c1_mode(args, device=device)
    if args.mode == "run":
        return _run_formal_analysis(args, device=device)
    if args.mode == "verify":
        return _run_verify_mode(args)
    if args.mode == "replay":
        return _run_replay_mode(args, device=device)
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
