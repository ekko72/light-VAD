# -*- coding: utf-8 -*-
"""Execute the frozen A-v2 / AE mechanism study.

The runner is deliberately stage-oriented.  AE1-AE3 use only frozen
predictions.  AE4 performs checkpoint agreement analysis and the
pre-registered five-pair replication training.  AE5 joins the frozen
upstream outputs without changing any metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ae_protocol_freeze.json"
)
FREEZE_MANIFEST_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ae_freeze_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae_mechanism"
)
AE0_RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae0_value_decomposition"
    / "ae0_value_decomposition.json"
)
DEFAULT_TEST_MANIFEST = (
    REPO_ROOT
    / "data"
    / "librivad"
    / "manifests"
    / "LibriSpeech_test_medium.tsv"
)
DEFAULT_TRAIN_MANIFEST = (
    REPO_ROOT
    / "data"
    / "librivad"
    / "manifests"
    / "LibriSpeech_train_medium.tsv"
)
DEFAULT_VAL_MANIFEST = (
    REPO_ROOT
    / "data"
    / "librivad"
    / "manifests"
    / "LibriSpeech_val_medium.tsv"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
DEFAULT_LONG_TEACHER = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_long_ext40"
    / "best.pt"
)

PROTOCOL_ID = "A-v2-AE-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_AE1_OUTCOME_ANALYSIS"
PROTOCOL_SHA256 = (
    "28218C73CC8CEDFE14DC9CDB5622B50C2D927627744AB9B9679FDFEE7F6B3842"
)
VALID_START = 382
DECISION_THRESHOLD = 0.5
GATE_THRESHOLD = 0.13
SPAN_ORDER = (64, 128, 256, 384, 512)
BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED_BASE = 20_260_920
SPEAKER_SEED_OFFSET = 500_000
UNSEEN_NOISE = ("SSN_noise", "Street_noise", "Transport_noise")
SNR_ORDER = ("-5", "0", "5", "10", "15", "20")
DOMAIN_ORDER = ("seen", "unseen")
CELL_ORDER = (
    "all",
    "clean",
    *tuple(f"{domain}/{snr}" for domain in DOMAIN_ORDER for snr in SNR_ORDER),
)
AE0_CELL_ORDER = tuple(
    f"{domain}/{snr}" for domain in DOMAIN_ORDER for snr in SNR_ORDER
)
FEATURE_ORDER = (
    "onset_distance",
    "offset_distance",
    "speech_run_length",
    "silence_run_length",
    "posterior_transition_distance",
    "uncertainty_persistence",
)
DISTANCE_BINS = ("0", "1", "2-4", "5-10", "11-25", "26-50", ">50")
DURATION_BINS = ("1", "2-4", "5-10", "11-25", "26-50", ">50")
MISSING_DISTANCE = 51.0
AE4_SEEDS = (73, 79, 83, 89, 97)
AE4_EXISTING_SEEDS = (17, 18, 19, 23)
REPLICATION_WORDING = (
    "five independent initializations of Short/RF384 replicates"
)


@dataclass(frozen=True)
class FrozenProtocol:
    path: Path
    payload: dict[str, Any]
    sha256: str
    manifest: dict[str, Any]


@dataclass
class AeData:
    protocol: FrozenProtocol
    spans: dict[int, dict[str, np.ndarray]]
    labels: np.ndarray
    short_scores: np.ndarray
    selected: np.ndarray
    test_mask: np.ndarray
    calibration_mask: np.ndarray
    source_key: np.ndarray
    speaker_ids: np.ndarray
    condition: np.ndarray
    noise_name: np.ndarray
    source_order: tuple[str, ...]
    speaker_order: tuple[str, ...]
    cell_labels: np.ndarray
    group_masks: dict[str, np.ndarray]
    segments: list[tuple[int, int]]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _verify_hash(path: Path, expected: str, *, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != str(expected).upper():
        raise ValueError(f"{role} hash mismatch: {actual} != {expected}")
    return actual


def load_frozen_protocol(
    protocol_path: str | Path = PROTOCOL_PATH,
    *,
    verify_inputs: bool = True,
) -> FrozenProtocol:
    """Validate the manifest hash and, by default, all frozen inputs."""
    manifest_path = FREEZE_MANIFEST_PATH
    manifest = _load_json(manifest_path)
    expected_protocol = _resolve_path(str(manifest.get("protocol_path", "")))
    if expected_protocol.resolve() != Path(protocol_path).resolve():
        raise ValueError("freeze manifest points to a different protocol")
    if str(manifest.get("protocol_sha256", "")).upper() != PROTOCOL_SHA256:
        raise ValueError("freeze manifest protocol hash differs from the runner")
    _verify_hash(Path(protocol_path), PROTOCOL_SHA256, role="AE protocol")
    payload = _load_json(Path(protocol_path))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported AE protocol id")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("AE protocol is not frozen before outcome analysis")
    if tuple(payload.get("study", {}).get("sequence", [])) != (
        "AE0",
        "AE1",
        "AE2",
        "AE3",
        "AE4",
        "AE5",
    ):
        raise ValueError("AE study sequence differs from the freeze")
    if payload.get("study", {}).get("automatic_a_v2_method_experiment") != "forbidden":
        raise ValueError("automatic A-v2 method experiments must be forbidden")
    protocol = FrozenProtocol(
        path=Path(protocol_path),
        payload=payload,
        sha256=PROTOCOL_SHA256,
        manifest=manifest,
    )
    if verify_inputs:
        verify_locked_inputs(protocol)
    return protocol


def verify_locked_inputs(protocol: FrozenProtocol) -> None:
    """Verify all locked result, protocol, and checkpoint hashes."""
    locks = protocol.payload["locks"]
    for name in (
        "a14_protocol",
        "a14_result",
        "a14_per_cell",
        "a14_per_seed",
        "a14_runner",
        "ae0_protocol",
        "ae0_result",
        "short_checkpoint",
        "long_checkpoint",
    ):
        record = locks[name]
        _verify_hash(
            _resolve_path(str(record["path"])),
            str(record["sha256"]),
            role=name,
        )
    for record in locks["a14_checkpoints"]:
        _verify_hash(
            _resolve_path(str(record["path"])),
            str(record["sha256"]),
            role=f"A14 checkpoint seed {record['seed']}",
        )
    for record in locks["a9_predictions"]["bundles"]:
        _verify_hash(
            _resolve_path(str(record["path"])),
            str(record["sha256"]),
            role=f"A9 RF{record['span']} bundle",
        )


def _cell_for_frame(
    condition: str,
    noise_name: str,
) -> str:
    if str(condition) == "clean":
        return "clean"
    domain = "unseen" if str(noise_name) in set(UNSEEN_NOISE) else "seen"
    return f"{domain}/{str(condition)}"


def _build_group_masks(
    *,
    test_mask: np.ndarray,
    condition: np.ndarray,
    noise_name: np.ndarray,
) -> dict[str, np.ndarray]:
    test_mask = np.asarray(test_mask, dtype=bool)
    condition = np.asarray(condition, dtype=str)
    noise_name = np.asarray(noise_name, dtype=str)
    masks: dict[str, np.ndarray] = {"all": test_mask.copy()}
    masks["clean"] = test_mask & (condition == "clean")
    noisy = test_mask & (condition != "clean")
    unseen = noisy & np.isin(noise_name, list(UNSEEN_NOISE))
    seen = noisy & ~unseen
    masks["seen"] = seen
    masks["unseen"] = unseen
    for domain, mask in (("seen", seen), ("unseen", unseen)):
        for snr in SNR_ORDER:
            masks[f"{domain}/{snr}"] = mask & (condition == snr)
    return masks


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def load_a9_data(
    protocol: FrozenProtocol,
    *,
    test_manifest: str | Path = DEFAULT_TEST_MANIFEST,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    librispeech_root: str | Path = DEFAULT_LIBRISPEECH_ROOT,
) -> AeData:
    """Load the five frozen A9 bundles and reconstruct utterance boundaries."""
    from reproductions.difficulty_adaptive_context.data import (
        FRAME_HOP,
        audio_frame_count,
        build_evaluation_items,
    )

    a9 = protocol.payload["locks"]["a9_predictions"]
    span_records = {int(record["span"]): record for record in a9["bundles"]}
    if tuple(sorted(span_records)) != SPAN_ORDER:
        raise ValueError("A9 span order differs from the frozen protocol")
    spans: dict[int, dict[str, np.ndarray]] = {}
    for span in SPAN_ORDER:
        path = _resolve_path(str(span_records[span]["path"]))
        arrays = _load_npz_arrays(path)
        required = {
            "labels",
            "short_scores",
            "embedded_short_scores",
            "full_adaptive_scores",
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
            raise ValueError(f"A9 RF{span} bundle is missing {missing}")
        spans[span] = arrays

    reference = spans[384]
    labels = np.asarray(reference["labels"], dtype=np.int64)
    short_scores = np.asarray(reference["embedded_short_scores"], dtype=np.float64)
    selected = np.asarray(reference["selected"], dtype=bool)
    test_mask = np.asarray(reference["test_mask"], dtype=bool)
    calibration_mask = np.asarray(reference["calibration_mask"], dtype=bool)
    source_key = np.asarray(reference["source_key"], dtype=str)
    speaker_ids = np.asarray(reference["speaker_ids"], dtype=str)
    condition = np.asarray(reference["condition"], dtype=str)
    noise_name = np.asarray(reference["noise_name"], dtype=str)
    for span, arrays in spans.items():
        for name, expected in (
            ("labels", labels),
            ("embedded_short_scores", short_scores),
            ("selected", selected),
            ("test_mask", test_mask),
            ("calibration_mask", calibration_mask),
            ("source_key", source_key),
            ("speaker_ids", speaker_ids),
            ("condition", condition),
            ("noise_name", noise_name),
        ):
            actual = np.asarray(arrays[name])
            if not np.array_equal(actual, expected):
                raise ValueError(f"A9 RF{span} {name} differs across spans")
        if not np.allclose(
            np.asarray(arrays["short_scores"], dtype=np.float64),
            short_scores,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"A9 RF{span} short_scores differ from embedded short")

    items = build_evaluation_items(
        test_manifest,
        generated_root=Path(data_root) / "generated",
        label_root=Path(data_root) / "labels",
        librispeech_root=librispeech_root,
        row_sample=int(a9["row_sample"]),
        seed=int(a9["evaluation_seed"]),
        include_clean=True,
    )
    segments: list[tuple[int, int]] = []
    cursor = 0
    for item in items:
        n_frames = audio_frame_count(item.audio_path) // FRAME_HOP + 1
        if n_frames <= VALID_START:
            continue
        length = n_frames - VALID_START
        segments.append((cursor, cursor + length))
        cursor += length
    if cursor != int(labels.size):
        raise ValueError(
            f"reconstructed utterance frames {cursor} != frozen {labels.size}"
        )

    source_order = tuple(sorted({str(value) for value in source_key}))
    speaker_order = tuple(sorted({str(value) for value in speaker_ids}))
    cell_labels = np.asarray(
        [_cell_for_frame(c, n) for c, n in zip(condition, noise_name)],
        dtype=object,
    )
    group_masks = _build_group_masks(
        test_mask=test_mask,
        condition=condition,
        noise_name=noise_name,
    )
    return AeData(
        protocol=protocol,
        spans=spans,
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
        cell_labels=cell_labels,
        group_masks=group_masks,
        segments=segments,
    )


def _short_correct(data: AeData) -> np.ndarray:
    return (data.short_scores >= DECISION_THRESHOLD) == data.labels.astype(bool)


def _span_correct(data: AeData, span: int) -> np.ndarray:
    scores = np.asarray(data.spans[span]["full_adaptive_scores"], dtype=np.float64)
    return (scores >= DECISION_THRESHOLD) == data.labels.astype(bool)


def _value_sign(short_correct: np.ndarray, refined_correct: np.ndarray) -> np.ndarray:
    correction = (~short_correct) & refined_correct
    harm = short_correct & (~refined_correct)
    return np.where(correction, 1, np.where(harm, -1, 0)).astype(np.int8)


def _horizon_arrays(data: AeData) -> dict[str, np.ndarray]:
    short_correct = _short_correct(data)
    correctness = np.stack(
        [_span_correct(data, span) for span in SPAN_ORDER],
        axis=1,
    )
    first = np.full(data.labels.size, -1, dtype=np.int16)
    stable = np.full(data.labels.size, -1, dtype=np.int16)
    for index, span in enumerate(SPAN_ORDER):
        first[(first < 0) & correctness[:, index]] = span
        suffix_correct = np.all(correctness[:, index:], axis=1)
        stable[(stable < 0) & suffix_correct] = span
    never = ~np.any(correctness, axis=1)
    unstable = np.any(correctness, axis=1) & (stable < 0)
    non_monotone = np.zeros(data.labels.size, dtype=bool)
    for index in range(1, len(SPAN_ORDER)):
        non_monotone |= correctness[:, index - 1] & (~correctness[:, index])
    return {
        "short_correct": short_correct,
        "correctness": correctness,
        "first_correct": first,
        "stable_sufficient": stable,
        "never_correct": never,
        "unstable_correct": unstable,
        "non_monotone": non_monotone,
    }


def _bin_value(value: float, bins: Sequence[str]) -> str:
    value = float(value)
    for label in bins:
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
    # Values below the first duration bin (notably zero for a non-selected
    # frame) are deliberately assigned to the smallest declared bin.
    return bins[0]


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


def _segment_event_features(
    labels: np.ndarray,
    short_scores: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    length = labels.size
    if not (length == short_scores.size == selected.size):
        raise ValueError("event feature arrays must have equal length")
    onset = np.flatnonzero((labels[1:] == 1) & (labels[:-1] == 0)) + 1
    offset = np.flatnonzero((labels[1:] == 0) & (labels[:-1] == 1)) + 1
    speech_run = _run_lengths(labels == 1)
    silence_run = _run_lengths(labels == 0)
    short_state = short_scores >= DECISION_THRESHOLD
    crossings = np.flatnonzero(short_state[1:] != short_state[:-1]) + 1
    return {
        "onset_distance": _nearest_signed_distance(onset, length),
        "offset_distance": _nearest_signed_distance(offset, length),
        "speech_run_length": speech_run.astype(np.float64),
        "silence_run_length": silence_run.astype(np.float64),
        "posterior_transition_distance": _nearest_absolute_distance(
            crossings,
            length,
        ),
        "uncertainty_persistence": _run_lengths(selected).astype(np.float64),
    }


def _build_event_features(data: AeData) -> dict[str, np.ndarray]:
    features = {
        name: np.zeros(data.labels.size, dtype=np.float64)
        for name in FEATURE_ORDER
    }
    for start, end in data.segments:
        local = _segment_event_features(
            data.labels[start:end],
            data.short_scores[start:end],
            data.selected[start:end],
        )
        for name, values in local.items():
            features[name][start:end] = values
    return features


def _group_mask(data: AeData, group: str) -> np.ndarray:
    if group not in data.group_masks:
        raise ValueError(f"unknown AE group: {group}")
    return np.asarray(data.group_masks[group], dtype=bool)


def _source_codes(data: AeData) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(data.source_order)}
    return np.asarray([lookup[str(value)] for value in data.source_key], dtype=np.int64)


def _speaker_codes(data: AeData) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(data.speaker_order)}
    return np.asarray(
        [lookup[str(value)] for value in data.speaker_ids],
        dtype=np.int64,
    )


def _group_seed(
    group: str,
    *,
    group_index: int | None = None,
    feature_bin: int | None = None,
) -> int:
    if group == "all":
        return BOOTSTRAP_SEED_BASE
    if group == "clean":
        return BOOTSTRAP_SEED_BASE + 1
    if group in DOMAIN_ORDER:
        return BOOTSTRAP_SEED_BASE + 10 + DOMAIN_ORDER.index(group)
    if group in AE0_CELL_ORDER:
        return BOOTSTRAP_SEED_BASE + 100 + AE0_CELL_ORDER.index(group)
    if group_index is None:
        raise ValueError("group_index is required for derived feature groups")
    if feature_bin is None:
        return BOOTSTRAP_SEED_BASE + 2_000 + int(group_index)
    return (
        BOOTSTRAP_SEED_BASE
        + 3_000
        + int(group_index) * 100
        + int(feature_bin)
    )


def _bootstrap_indices(
    n_clusters: int,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_clusters), size=(int(repeats), int(n_clusters)))


def _ci(samples: Sequence[float]) -> tuple[float | None, float | None]:
    values = np.asarray([value for value in samples if np.isfinite(value)], dtype=np.float64)
    if values.size == 0:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _metric_with_ci(
    values: Sequence[float | None],
    *,
    name: str,
) -> dict[str, Any]:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return {
            name: None,
            f"{name}_ci95_low": None,
            f"{name}_ci95_high": None,
        }
    low, high = np.quantile(valid, [0.025, 0.975])
    return {
        name: float(np.mean(valid)),
        f"{name}_ci95_low": float(low),
        f"{name}_ci95_high": float(high),
    }


def _weighted_rate(
    counts: np.ndarray,
    denominator: np.ndarray,
    multiplicity: np.ndarray,
) -> float | None:
    num = float(multiplicity @ counts)
    den = float(multiplicity @ denominator)
    return num / den if den > 0.0 else None


def _bootstrap_rate_metrics(
    *,
    group: str,
    metric_counts: Mapping[str, np.ndarray],
    metric_denominators: Mapping[str, np.ndarray],
    source_codes: np.ndarray,
    speaker_codes: np.ndarray,
    n_sources: int,
    n_speakers: int,
    repeats: int = BOOTSTRAP_REPEATS,
    group_index: int | None = None,
    feature_bin: int | None = None,
) -> dict[str, Any]:
    """Bootstrap all supplied ratios with one shared resample per unit."""
    seed = _group_seed(
        group,
        group_index=group_index,
        feature_bin=feature_bin,
    )
    source_indices = _bootstrap_indices(
        n_sources,
        repeats=repeats,
        seed=seed,
    )
    speaker_indices = _bootstrap_indices(
        n_speakers,
        repeats=repeats,
        seed=seed + SPEAKER_SEED_OFFSET,
    )
    result: dict[str, Any] = {"source_cluster": {}, "speaker_sensitivity": {}}
    for unit_name, codes, indices in (
        ("source_cluster", source_codes, source_indices),
        ("speaker_sensitivity", speaker_codes, speaker_indices),
    ):
        n_clusters = n_sources if unit_name == "source_cluster" else n_speakers
        metric_names = list(metric_counts)
        count_matrix = np.stack(
            [
                np.bincount(
                    codes,
                    weights=np.asarray(metric_counts[name], dtype=np.float64),
                    minlength=n_clusters,
                )
                for name in metric_names
            ],
            axis=0,
        )
        denominator_matrix = np.stack(
            [
                np.bincount(
                    codes,
                    weights=np.asarray(
                        metric_denominators[name],
                        dtype=np.float64,
                    ),
                    minlength=n_clusters,
                )
                for name in metric_names
            ],
            axis=0,
        )
        sampled_counts = indices @ count_matrix.T
        sampled_denominators = indices @ denominator_matrix.T
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = sampled_counts / sampled_denominators
        for index, name in enumerate(metric_names):
            values = samples[:, index]
            values = values[np.isfinite(values)]
            low, high = _ci(values)
            result[unit_name][f"{name}_ci95_low"] = low
            result[unit_name][f"{name}_ci95_high"] = high
    return result


def _ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float | None:
    numerator = np.asarray(numerator, dtype=bool)
    denominator = np.asarray(denominator, dtype=bool)
    total = int(np.count_nonzero(denominator))
    if total == 0:
        return None
    return float(np.count_nonzero(numerator & denominator) / total)


def _safe_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _histogram(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    categories: Sequence[int],
) -> dict[str, int]:
    local = np.asarray(values, dtype=np.int64)[np.asarray(mask, dtype=bool)]
    return {
        str(category): int(np.count_nonzero(local == int(category)))
        for category in categories
    }


def _format_float(value: Any, digits: int = 8) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _group_label(group: str) -> str:
    return str(group)


def _group_test_mask(data: AeData, group: str) -> np.ndarray:
    return np.asarray(_group_mask(data, group), dtype=bool) & data.test_mask


def _state_arrays(data: AeData) -> dict[str, np.ndarray]:
    short_correct = _short_correct(data)
    refined_correct = _span_correct(data, 384)
    refinable = (~short_correct) & refined_correct
    irreducible = (~short_correct) & (~refined_correct)
    harm = short_correct & (~refined_correct)
    short_sufficient = short_correct & refined_correct
    return {
        "short_correct": short_correct,
        "refined_correct": refined_correct,
        "short_sufficient": short_sufficient,
        "refinable": refinable,
        "irreducible": irreducible,
        "harm": harm,
    }


def _point_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float | None:
    return _ratio(numerator, denominator)


def _point_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    local = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if local.size == 0:
        return None
    return float(np.mean(local))


def _ci_columns(
    prefix: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    source = bootstrap["source_cluster"]
    speaker = bootstrap["speaker_sensitivity"]
    return {
        f"source_ci95_low_{prefix}": source.get(f"{prefix}_ci95_low"),
        f"source_ci95_high_{prefix}": source.get(f"{prefix}_ci95_high"),
        f"speaker_ci95_low_{prefix}": speaker.get(f"{prefix}_ci95_low"),
        f"speaker_ci95_high_{prefix}": speaker.get(f"{prefix}_ci95_high"),
    }


def _bootstrap_for_metrics(
    *,
    data: AeData,
    group: str,
    metric_counts: Mapping[str, np.ndarray],
    metric_denominators: Mapping[str, np.ndarray],
    group_index: int | None = None,
    feature_bin: int | None = None,
) -> dict[str, Any]:
    return _bootstrap_rate_metrics(
        group=group,
        metric_counts=metric_counts,
        metric_denominators=metric_denominators,
        source_codes=_source_codes(data),
        speaker_codes=_speaker_codes(data),
        n_sources=len(data.source_order),
        n_speakers=len(data.speaker_order),
        repeats=BOOTSTRAP_REPEATS,
        group_index=group_index,
        feature_bin=feature_bin,
    )


def _ae1_group_row(data: AeData, group: str) -> dict[str, Any]:
    states = _state_arrays(data)
    mask = _group_test_mask(data, group)
    selected = data.selected & mask
    frames = int(np.count_nonzero(mask))
    counts = {
        "frames": frames,
        "short_sufficient": int(
            np.count_nonzero(states["short_sufficient"] & mask)
        ),
        "refinable": int(np.count_nonzero(states["refinable"] & mask)),
        "rf384_irreducible": int(
            np.count_nonzero(states["irreducible"] & mask)
        ),
        "refinement_harm": int(np.count_nonzero(states["harm"] & mask)),
        "selected": int(np.count_nonzero(selected)),
        "selected_refinable": int(
            np.count_nonzero(selected & states["refinable"])
        ),
        "selected_harm": int(
            np.count_nonzero(selected & states["harm"])
        ),
    }
    point = {
        **counts,
        "state_rate_short_sufficient": _point_ratio(
            states["short_sufficient"] & mask,
            mask,
        ),
        "state_rate_refinable": _point_ratio(
            states["refinable"] & mask,
            mask,
        ),
        "state_rate_rf384_irreducible": _point_ratio(
            states["irreducible"] & mask,
            mask,
        ),
        "state_rate_refinement_harm": _point_ratio(
            states["harm"] & mask,
            mask,
        ),
        "availability_gross": _point_ratio(states["refinable"] & mask, mask),
        "availability_net": (
            None
            if frames == 0
            else float(
                (
                    np.count_nonzero(states["refinable"] & mask)
                    - np.count_nonzero(states["harm"] & mask)
                )
                / frames
            )
        ),
        "gate_utility": (
            None
            if frames == 0
            else float(
                (
                    np.count_nonzero(selected & states["refinable"])
                    - np.count_nonzero(selected & states["harm"])
                )
                / frames
            )
        ),
        "activation_rate": _point_ratio(selected, mask),
        "selected_refinable_rate": _point_ratio(
            selected & states["refinable"],
            selected,
        ),
        "selected_harm_rate": _point_ratio(
            selected & states["harm"],
            selected,
        ),
    }
    metric_counts = {
        "state_rate_short_sufficient": states["short_sufficient"] & mask,
        "state_rate_refinable": states["refinable"] & mask,
        "state_rate_rf384_irreducible": states["irreducible"] & mask,
        "state_rate_refinement_harm": states["harm"] & mask,
        "availability_gross": states["refinable"] & mask,
        "availability_net": (
            states["refinable"].astype(np.int16)
            - states["harm"].astype(np.int16)
        )
        * mask,
        "gate_utility": (
            selected & states["refinable"]
        ).astype(np.int16)
        - (selected & states["harm"]).astype(np.int16),
        "activation_rate": selected,
        "selected_refinable_rate": selected & states["refinable"],
        "selected_harm_rate": selected & states["harm"],
    }
    denominators = {
        "state_rate_short_sufficient": mask,
        "state_rate_refinable": mask,
        "state_rate_rf384_irreducible": mask,
        "state_rate_refinement_harm": mask,
        "availability_gross": mask,
        "availability_net": mask,
        "gate_utility": mask,
        "activation_rate": mask,
        "selected_refinable_rate": selected,
        "selected_harm_rate": selected,
    }
    bootstrap = _bootstrap_for_metrics(
        data=data,
        group=group,
        metric_counts=metric_counts,
        metric_denominators=denominators,
    )
    row = {"group": _group_label(group), **point}
    for metric in metric_counts:
        row.update(_ci_columns(metric, bootstrap))
    return row


def run_ae1(
    data: AeData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute the frozen AE1 state decomposition."""
    rows = [_ae1_group_row(data, group) for group in CELL_ORDER]
    summary = rows[0]
    cell_rows = [
        {
            "domain": group.split("/", 1)[0],
            "snr_db": group.split("/", 1)[1],
            **row,
        }
        for group, row in zip(CELL_ORDER[2:], rows[2:])
    ]
    condition_rows = [
        row
        for group, row in zip(CELL_ORDER[:2], rows[:2])
    ]
    condition_rows.extend(
        _ae1_group_row(data, group) for group in ("seen", "unseen")
    )
    _write_csv(output_dir / "ae1_state_summary.csv", rows)
    _write_csv(output_dir / "ae1_state_by_cell.csv", cell_rows)
    _write_csv(
        output_dir / "ae1_condition_availability.csv",
        condition_rows,
    )
    return {
        "rows": rows,
        "summary": summary,
        "cell_rows": cell_rows,
        "condition_rows": condition_rows,
    }


def _ae2_group_row(data: AeData, group: str) -> dict[str, Any]:
    horizons = _horizon_arrays(data)
    mask = _group_test_mask(data, group)
    not_short = (~horizons["short_correct"]) & mask
    stable_defined = horizons["stable_sufficient"] >= 0
    frames = int(np.count_nonzero(mask))
    first_hist = _histogram(
        horizons["first_correct"],
        mask=mask,
        categories=(-1, *SPAN_ORDER),
    )
    stable_hist = _histogram(
        horizons["stable_sufficient"],
        mask=mask,
        categories=(-1, *SPAN_ORDER),
    )
    first_defined = horizons["first_correct"] >= 0
    stable_defined_mask = stable_defined & mask
    long_stable = (horizons["stable_sufficient"] >= 256) & mask
    point: dict[str, Any] = {
        "frames": frames,
        "short_sufficient_rate": _point_ratio(
            horizons["short_correct"] & mask,
            mask,
        ),
        "never_correct_rate": _point_ratio(
            horizons["never_correct"] & mask,
            mask,
        ),
        "unstable_correct_rate": _point_ratio(
            horizons["unstable_correct"] & mask,
            mask,
        ),
        "non_monotone_rate": _point_ratio(
            horizons["non_monotone"] & mask,
            not_short,
        ),
        "first_correct_defined_rate": _point_ratio(first_defined & mask, mask),
        "stable_sufficient_defined_rate": _point_ratio(
            stable_defined & mask,
            mask,
        ),
        "stable_sufficient_long_rate": _point_ratio(
            long_stable,
            stable_defined_mask,
        ),
        "mean_first_correct_span": _point_mean(
            horizons["first_correct"].astype(np.float64),
            first_defined & mask,
        ),
        "mean_stable_sufficient_span": _point_mean(
            horizons["stable_sufficient"].astype(np.float64),
            stable_defined_mask,
        ),
    }
    for category, count in first_hist.items():
        point[f"first_correct_{category}"] = count
    for category, count in stable_hist.items():
        point[f"stable_sufficient_{category}"] = count
    correctness = horizons["correctness"]
    for index, span in enumerate(SPAN_ORDER):
        point[f"oracle_net_utility_rf{span}"] = (
            None
            if frames == 0
            else float(
                (
                    np.count_nonzero(correctness[:, index] & mask)
                    - np.count_nonzero(horizons["short_correct"] & mask)
                )
                / frames
            )
        )
    metric_counts: dict[str, np.ndarray] = {
        "short_sufficient_rate": horizons["short_correct"] & mask,
        "never_correct_rate": horizons["never_correct"] & mask,
        "unstable_correct_rate": horizons["unstable_correct"] & mask,
        "non_monotone_rate": horizons["non_monotone"] & not_short,
        "first_correct_defined_rate": first_defined & mask,
        "stable_sufficient_defined_rate": stable_defined & mask,
        "stable_sufficient_long_rate": long_stable,
        "mean_first_correct_span": (
            horizons["first_correct"].astype(np.float64) * (first_defined & mask)
        ),
        "mean_stable_sufficient_span": (
            horizons["stable_sufficient"].astype(np.float64)
            * stable_defined_mask
        ),
    }
    denominators: dict[str, np.ndarray] = {
        "short_sufficient_rate": mask,
        "never_correct_rate": mask,
        "unstable_correct_rate": mask,
        "non_monotone_rate": not_short,
        "first_correct_defined_rate": mask,
        "stable_sufficient_defined_rate": mask,
        "stable_sufficient_long_rate": stable_defined_mask,
        "mean_first_correct_span": first_defined & mask,
        "mean_stable_sufficient_span": stable_defined_mask,
    }
    for index, span in enumerate(SPAN_ORDER):
        name = f"oracle_net_utility_rf{span}"
        metric_counts[name] = correctness[:, index] & mask
        denominators[name] = mask
    bootstrap = _bootstrap_for_metrics(
        data=data,
        group=group,
        metric_counts=metric_counts,
        metric_denominators=denominators,
    )
    row = {"group": _group_label(group), **point}
    for metric in metric_counts:
        row.update(_ci_columns(metric, bootstrap))
    return row


def run_ae2(
    data: AeData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute the frozen AE2 horizon and non-monotonicity analysis."""
    rows = [_ae2_group_row(data, group) for group in CELL_ORDER]
    cell_rows = [
        {
            "domain": group.split("/", 1)[0],
            "snr_db": group.split("/", 1)[1],
            **row,
        }
        for group, row in zip(CELL_ORDER[2:], rows[2:])
    ]
    _write_csv(output_dir / "ae2_horizon_summary.csv", rows)
    _write_csv(output_dir / "ae2_horizon_by_cell.csv", cell_rows)
    _write_csv(
        output_dir / "ae2_nonmonotonicity.csv",
        [
            {
                "group": row["group"],
                "frames": row["frames"],
                "non_monotone_rate": row["non_monotone_rate"],
                "non_monotone_rate_source_ci95_low": row[
                    "source_ci95_low_non_monotone_rate"
                ],
                "non_monotone_rate_source_ci95_high": row[
                    "source_ci95_high_non_monotone_rate"
                ],
                "never_correct_rate": row["never_correct_rate"],
                "unstable_correct_rate": row["unstable_correct_rate"],
                "stable_sufficient_long_rate": row[
                    "stable_sufficient_long_rate"
                ],
            }
            for row in rows
        ],
    )
    return {"rows": rows, "cell_rows": cell_rows}


def _feature_values(data: AeData) -> dict[str, np.ndarray]:
    return _build_event_features(data)


def _feature_bin_mask(
    feature: str,
    values: np.ndarray,
    bin_label: str,
) -> np.ndarray:
    bins = DISTANCE_BINS if feature in {
        "onset_distance",
        "offset_distance",
        "posterior_transition_distance",
    } else DURATION_BINS
    values = np.asarray(values, dtype=np.float64)
    if feature in {
        "onset_distance",
        "offset_distance",
        "posterior_transition_distance",
    }:
        values = np.abs(values)
    assigned = np.full(values.size, bins[0], dtype=object)
    for label in bins[1:]:
        if label == "1":
            assigned[values == 1.0] = label
        elif label == ">50":
            assigned[values > 50.0] = label
        elif "-" in label:
            low, high = label.split("-", 1)
            assigned[
                (values >= float(low)) & (values <= float(high))
            ] = label
        else:
            assigned[values == float(label)] = label
    return assigned == bin_label


def _ae3_metric_payload(
    data: AeData,
    *,
    group: str,
    mask: np.ndarray,
    horizons: Mapping[str, np.ndarray],
    states: Mapping[str, np.ndarray],
    group_index: int | None = None,
    feature_bin: int | None = None,
) -> dict[str, Any]:
    frames = int(np.count_nonzero(mask))
    selected = data.selected & mask
    not_short = (~horizons["short_correct"]) & mask
    stable_defined = horizons["stable_sufficient"] >= 0
    stable_mask = stable_defined & mask
    point = {
        "frames": frames,
        "refinable_rate": _point_ratio(states["refinable"] & mask, mask),
        "irreducible_rate": _point_ratio(states["irreducible"] & mask, mask),
        "refinement_harm_rate": _point_ratio(states["harm"] & mask, mask),
        "availability_net": (
            None
            if frames == 0
            else float(
                (
                    np.count_nonzero(states["refinable"] & mask)
                    - np.count_nonzero(states["harm"] & mask)
                )
                / frames
            )
        ),
        "mean_first_correct_span": _point_mean(
            horizons["first_correct"].astype(np.float64),
            (horizons["first_correct"] >= 0) & mask,
        ),
        "mean_stable_sufficient_span": _point_mean(
            horizons["stable_sufficient"].astype(np.float64),
            stable_mask,
        ),
        "non_monotone_rate": _point_ratio(
            horizons["non_monotone"] & mask,
            not_short,
        ),
    }
    metric_counts = {
        "refinable_rate": states["refinable"] & mask,
        "irreducible_rate": states["irreducible"] & mask,
        "refinement_harm_rate": states["harm"] & mask,
        "availability_net": (
            states["refinable"].astype(np.int16)
            - states["harm"].astype(np.int16)
        )
        * mask,
        "mean_first_correct_span": (
            horizons["first_correct"].astype(np.float64)
            * ((horizons["first_correct"] >= 0) & mask)
        ),
        "mean_stable_sufficient_span": (
            horizons["stable_sufficient"].astype(np.float64) * stable_mask
        ),
        "non_monotone_rate": horizons["non_monotone"] & not_short,
    }
    denominators = {
        "refinable_rate": mask,
        "irreducible_rate": mask,
        "refinement_harm_rate": mask,
        "availability_net": mask,
        "mean_first_correct_span": (horizons["first_correct"] >= 0) & mask,
        "mean_stable_sufficient_span": stable_mask,
        "non_monotone_rate": not_short,
    }
    bootstrap = _bootstrap_for_metrics(
        data=data,
        group=group,
        metric_counts=metric_counts,
        metric_denominators=denominators,
        group_index=group_index,
        feature_bin=feature_bin,
    )
    row = dict(point)
    for metric in metric_counts:
        row.update(_ci_columns(metric, bootstrap))
    return row


def run_ae3(
    data: AeData,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute event-feature gradients and condition-level feature maps."""
    features = _feature_values(data)
    horizons = _horizon_arrays(data)
    states = _state_arrays(data)
    feature_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for feature in FEATURE_ORDER:
        values = features[feature]
        bins = DISTANCE_BINS if feature in {
            "onset_distance",
            "offset_distance",
            "posterior_transition_distance",
        } else DURATION_BINS
        for group in CELL_ORDER:
            group_index = (
                FEATURE_ORDER.index(feature) * len(CELL_ORDER)
                + CELL_ORDER.index(group)
            )
            group_mask = _group_test_mask(data, group)
            if feature == "uncertainty_persistence":
                valid = group_mask & data.selected
            else:
                valid = group_mask
            summary_payload = _ae3_metric_payload(
                data,
                group=group,
                mask=valid,
                horizons=horizons,
                states=states,
                group_index=group_index,
            )
            feature_rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "mean_feature": _point_mean(values, valid),
                    **summary_payload,
                }
            )
            for bin_index, bin_label in enumerate(bins):
                bin_mask = valid & _feature_bin_mask(
                    feature,
                    values,
                    bin_label,
                )
                payload = _ae3_metric_payload(
                    data,
                    group=group,
                    mask=bin_mask,
                    horizons=horizons,
                    states=states,
                    group_index=group_index,
                    feature_bin=bin_index,
                )
                bin_rows.append(
                    {
                        "feature": feature,
                        "bin": bin_label,
                        "group": group,
                        "mean_feature": _point_mean(values, bin_mask),
                        **payload,
                    }
                )
            map_rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "mean_feature": _point_mean(values, valid),
                    "refinable_rate": summary_payload["refinable_rate"],
                    "irreducible_rate": summary_payload["irreducible_rate"],
                    "refinement_harm_rate": summary_payload[
                        "refinement_harm_rate"
                    ],
                    "availability_net": summary_payload["availability_net"],
                    "mean_stable_sufficient_span": summary_payload[
                        "mean_stable_sufficient_span"
                    ],
                    "non_monotone_rate": summary_payload[
                        "non_monotone_rate"
                    ],
                }
            )
    _write_csv(output_dir / "ae3_event_feature_summary.csv", feature_rows)
    _write_csv(output_dir / "ae3_feature_bins.csv", bin_rows)
    _write_csv(output_dir / "ae3_condition_feature_map.csv", map_rows)
    return {
        "features": feature_rows,
        "bins": bin_rows,
        "map": map_rows,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": file_sha256(path),
                }
                for path in artifacts
            ],
        },
    )
    return marker


def _load_a9_items(
    protocol: FrozenProtocol,
    *,
    test_manifest: str | Path = DEFAULT_TEST_MANIFEST,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    librispeech_root: str | Path = DEFAULT_LIBRISPEECH_ROOT,
) -> list[Any]:
    from reproductions.difficulty_adaptive_context.data import (
        build_evaluation_items,
    )

    a9 = protocol.payload["locks"]["a9_predictions"]
    items = build_evaluation_items(
        test_manifest,
        generated_root=Path(data_root) / "generated",
        label_root=Path(data_root) / "labels",
        librispeech_root=librispeech_root,
        row_sample=int(a9["row_sample"]),
        seed=int(a9["evaluation_seed"]),
        include_clean=True,
    )
    selected_rows = sum(not bool(item.is_clean) for item in items)
    if selected_rows != int(a9["expected_rows"]):
        raise RuntimeError(
            f"A9 inference selected {selected_rows} manifest rows, "
            f"expected {a9['expected_rows']}"
        )
    return items


def _predict_checkpoint_on_a9(
    data: AeData,
    *,
    checkpoint_path: Path,
    protocol: FrozenProtocol,
    output_path: Path,
    device: str,
    chunk_frames: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    from reproductions.difficulty_adaptive_context.data import (
        FRAME_HOP,
        causal_frame_labels,
        read_int16_audio,
    )
    from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
        load_adaptive_model,
        predict_full_adaptive_frames,
    )
    from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
    from reproductions.marblenet_vad.train import resolve_device

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    resolved_device = resolve_device(device)
    model, _, _ = load_adaptive_model(checkpoint_path, resolved_device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(resolved_device)
    items = _load_a9_items(protocol)
    labels_parts: list[np.ndarray] = []
    short_parts: list[np.ndarray] = []
    refined_parts: list[np.ndarray] = []
    for index, item in enumerate(items):
        waveform = read_int16_audio(item.audio_path)
        sample_labels = np.load(item.label_path)
        if sample_labels.size != waveform.size:
            raise ValueError(
                f"label/audio mismatch for {item.audio_path}: "
                f"{sample_labels.size} != {waveform.size}"
            )
        n_frames = waveform.size // FRAME_HOP + 1
        if n_frames <= VALID_START:
            continue
        frame_labels = causal_frame_labels(sample_labels, n_frames)
        refined, embedded_short, _, _ = predict_full_adaptive_frames(
            model,
            frontend,
            waveform,
            device=resolved_device,
            chunk_frames=chunk_frames,
        )
        if refined.size != n_frames or embedded_short.size != n_frames:
            raise RuntimeError("adaptive output frame count is inconsistent")
        labels_parts.append(frame_labels[VALID_START:])
        short_parts.append(embedded_short[VALID_START:])
        refined_parts.append(refined[VALID_START:])
        if progress_every > 0 and (
            (index + 1) % progress_every == 0 or index + 1 == len(items)
        ):
            print(
                f"AE4 inference {checkpoint_path.name}: "
                f"{index + 1}/{len(items)} rows",
                flush=True,
            )
    if not labels_parts:
        raise RuntimeError("AE4 inference produced no frames")
    labels = np.concatenate(labels_parts)
    short_scores = np.concatenate(short_parts)
    refined_scores = np.concatenate(refined_parts)
    if not np.array_equal(labels, data.labels):
        raise RuntimeError("AE4 inference labels differ from the frozen A9 rows")
    if labels.size != data.labels.size:
        raise RuntimeError(
            f"AE4 inference frames {labels.size} != frozen {data.labels.size}"
        )
    selected = np.abs(short_scores - DECISION_THRESHOLD) <= GATE_THRESHOLD
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        labels=labels.astype(np.int64),
        short_scores=short_scores.astype(np.float64),
        refined_scores=refined_scores.astype(np.float64),
        selected=selected.astype(bool),
        checkpoint=str(checkpoint_path),
    )
    return {
        "labels": labels,
        "short_scores": short_scores,
        "refined_scores": refined_scores,
        "selected": selected,
    }


def _value_metrics_for_bundle(
    *,
    data: AeData,
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    short_correct = (short_scores >= DECISION_THRESHOLD) == labels.astype(bool)
    refined_correct = (
        (refined_scores >= DECISION_THRESHOLD) == labels.astype(bool)
    )
    value = _value_sign(short_correct, refined_correct)
    return {
        "short_correct": short_correct,
        "refined_correct": refined_correct,
        "value": value,
        "refinable": (~short_correct) & refined_correct,
        "harm": short_correct & (~refined_correct),
    }


def _bundle_condition_directions(
    data: AeData,
    *,
    value: np.ndarray,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for group in CELL_ORDER:
        mask = _group_test_mask(data, group)
        result[group] = _point_mean(value.astype(np.float64), mask)
    return result


def _sign_agreement(
    left: np.ndarray,
    right: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        left = left[keep]
        right = right[keep]
    if left.size == 0:
        return None
    return float(np.count_nonzero(left == right) / left.size)


def _direction_agreement(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
) -> tuple[float | None, int]:
    compared = 0
    matched = 0
    for group in CELL_ORDER:
        left_value = left.get(group)
        right_value = right.get(group)
        if left_value is None or right_value is None:
            continue
        compared += 1
        left_sign = 1 if left_value > 0 else -1 if left_value < 0 else 0
        right_sign = 1 if right_value > 0 else -1 if right_value < 0 else 0
        matched += int(left_sign == right_sign)
    return (None if compared == 0 else matched / compared), compared


def _bootstrap_mean_by_cluster(
    values: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    totals = np.bincount(codes, weights=values, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    indices = _bootstrap_indices(n_clusters, repeats=repeats, seed=seed)
    sampled_counts = indices @ counts
    sampled_totals = indices @ totals
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = sampled_totals / sampled_counts
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"ci95_low": None, "ci95_high": None}
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"ci95_low": float(low), "ci95_high": float(high)}


def _ae4_seed_metric_rows(
    data: AeData,
    *,
    seed: int,
    role: str,
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> list[dict[str, Any]]:
    metrics = _value_metrics_for_bundle(
        data=data,
        labels=labels,
        short_scores=short_scores,
        refined_scores=refined_scores,
    )
    value = metrics["value"]
    source_codes = _source_codes(data)
    rows: list[dict[str, Any]] = []
    for group in CELL_ORDER:
        mask = _group_test_mask(data, group)
        frames = int(np.count_nonzero(mask))
        local_value = value[mask]
        availability_net = _point_mean(value.astype(np.float64), mask)
        ci = _bootstrap_mean_by_cluster(
            local_value.astype(np.float64),
            source_codes[mask],
            len(data.source_order),
            seed=_group_seed(
                group,
                group_index=CELL_ORDER.index(group),
            )
            + 100_000
            + int(seed),
        )
        rows.append(
            {
                "seed": int(seed),
                "role": role,
                "group": group,
                "frames": frames,
                "p_value_plus": (
                    None
                    if frames == 0
                    else float(np.count_nonzero(local_value == 1) / frames)
                ),
                "p_value_minus": (
                    None
                    if frames == 0
                    else float(np.count_nonzero(local_value == -1) / frames)
                ),
                "mean_value": availability_net,
                "availability_net": availability_net,
                "refinable_rate": _point_ratio(
                    metrics["refinable"] & mask,
                    mask,
                ),
                "refinement_harm_rate": _point_ratio(
                    metrics["harm"] & mask,
                    mask,
                ),
                "source_ci95_low_availability_net": ci["ci95_low"],
                "source_ci95_high_availability_net": ci["ci95_high"],
            }
        )
    return rows


def _ae4_pair_rows(
    data: AeData,
    *,
    seed_values: Mapping[int, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = list(seed_values)
    directions = {
        seed: _bundle_condition_directions(
            data,
            value=seed_values[seed]["value"],
        )
        for seed in seeds
    }
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            mask = data.test_mask
            sign_agreement = _sign_agreement(
                seed_values[left_seed]["value"],
                seed_values[right_seed]["value"],
                mask=mask,
            )
            value_correlation = None
            left_value = seed_values[left_seed]["value"][mask].astype(
                np.float64
            )
            right_value = seed_values[right_seed]["value"][mask].astype(
                np.float64
            )
            if left_value.size and np.std(left_value) > 0 and np.std(right_value) > 0:
                value_correlation = float(
                    np.corrcoef(left_value, right_value)[0, 1]
                )
            direction_agreement, compared = _direction_agreement(
                directions[left_seed],
                directions[right_seed],
            )
            if {int(left_seed), int(right_seed)} <= {17, 18, 19}:
                comparison_role = "shared_short_encoder_checkpoint_variation"
            elif 23 in {int(left_seed), int(right_seed)}:
                comparison_role = "independent_short_encoder_sensitivity"
            else:
                comparison_role = "replication_pair"
            rows.append(
                {
                    "left_seed": int(left_seed),
                    "right_seed": int(right_seed),
                    "comparison_role": comparison_role,
                    "frames": int(np.count_nonzero(mask)),
                    "sign_agreement": sign_agreement,
                    "pairwise_value_agreement": sign_agreement,
                    "value_correlation": value_correlation,
                    "condition_level_direction_agreement": direction_agreement,
                    "condition_level_groups_compared": compared,
                }
            )
    return rows


def run_ae4_existing(
    data: AeData,
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
    device: str = "auto",
    chunk_frames: int = 2_000,
    progress_every: int = 100,
) -> dict[str, Any]:
    prediction_dir = output_dir / "ae4_existing_predictions"
    marker = _stage_marker(output_dir, "ae4_existing")
    _refuse_overwrite(
        [
            marker,
            output_dir / "ae4_checkpoint_agreement.csv",
            prediction_dir / "seed17" / "frame_predictions.npz",
            prediction_dir / "seed18" / "frame_predictions.npz",
            prediction_dir / "seed19" / "frame_predictions.npz",
            prediction_dir / "seed23" / "frame_predictions.npz",
        ],
        stage="AE4 existing-checkpoint analysis",
    )
    checkpoint_by_seed = {
        int(record["seed"]): _resolve_path(str(record["path"]))
        for record in protocol.payload["locks"]["a14_checkpoints"]
    }
    seed_metrics: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for seed in AE4_EXISTING_SEEDS:
        role = (
            "shared_short_encoder_checkpoint"
            if seed in (17, 18, 19)
            else "independent_short_encoder_checkpoint"
        )
        path = prediction_dir / f"seed{seed}" / "frame_predictions.npz"
        arrays = _predict_checkpoint_on_a9(
            data,
            checkpoint_path=checkpoint_by_seed[seed],
            protocol=protocol,
            output_path=path,
            device=device,
            chunk_frames=chunk_frames,
            progress_every=progress_every,
        )
        metrics = _value_metrics_for_bundle(
            data=data,
            labels=arrays["labels"],
            short_scores=arrays["short_scores"],
            refined_scores=arrays["refined_scores"],
        )
        seed_metrics[seed] = metrics
        rows.extend(
            _ae4_seed_metric_rows(
                data,
                seed=seed,
                role=role,
                labels=arrays["labels"],
                short_scores=arrays["short_scores"],
                refined_scores=arrays["refined_scores"],
            )
        )
    pair_rows = _ae4_pair_rows(data, seed_values=seed_metrics)
    _write_csv(output_dir / "ae4_checkpoint_agreement.csv", rows + pair_rows)
    artifacts = [
        output_dir / "ae4_checkpoint_agreement.csv",
        *[
            prediction_dir / f"seed{seed}" / "frame_predictions.npz"
            for seed in AE4_EXISTING_SEEDS
        ],
    ]
    _write_stage_marker(
        output_dir,
        stage="ae4_existing",
        artifacts=artifacts,
    )
    return {
        "rows": rows,
        "pairs": pair_rows,
        "seed_metrics": seed_metrics,
    }


def _run_logged_command(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path = REPO_ROOT,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _ae4_training_commands(
    *,
    seed: int,
    seed_root: Path,
    train_manifest: Path,
    val_manifest: Path,
    data_root: Path,
    long_teacher: Path,
) -> dict[str, tuple[list[str], Path]]:
    stage_a = seed_root / "stage_a"
    stage_b = seed_root / "stage_b"
    refiner = seed_root / "refiner"
    common = [
        sys.executable,
        "-m",
    ]
    stage_a_command = [
        *common,
        "reproductions.difficulty_adaptive_context.train_context",
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--data-root",
        str(data_root),
        "--results-dir",
        str(stage_a),
        "--rf-profile",
        "short",
        "--train-rows",
        "8640",
        "--val-rows",
        "432",
        "--context-seconds",
        "4.0",
        "--target-seconds",
        "4.0",
        "--batch-size",
        "128",
        "--num-workers",
        "0",
        "--epochs",
        "30",
        "--seed",
        str(seed),
        "--max-lr",
        "0.01",
        "--min-lr",
        "0.001",
        "--warmup-ratio",
        "0.03",
        "--hold-ratio",
        "0.25",
        "--decay-power",
        "2.0",
        "--momentum",
        "0.9",
        "--weight-decay",
        "0.001",
    ]
    stage_b_command = [
        *common,
        "reproductions.difficulty_adaptive_context.train_context",
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--data-root",
        str(data_root),
        "--results-dir",
        str(stage_b),
        "--rf-profile",
        "short",
        "--train-rows",
        "8640",
        "--val-rows",
        "432",
        "--context-seconds",
        "4.0",
        "--target-seconds",
        "4.0",
        "--batch-size",
        "128",
        "--num-workers",
        "0",
        "--epochs",
        "40",
        "--seed",
        str(seed),
        "--max-lr",
        "0.001",
        "--min-lr",
        "0.001",
        "--warmup-ratio",
        "0.0",
        "--hold-ratio",
        "0.0",
        "--decay-power",
        "2.0",
        "--momentum",
        "0.9",
        "--weight-decay",
        "0.001",
        "--resume",
        str(stage_a / "last.pt"),
    ]
    refiner_command = [
        *common,
        "reproductions.difficulty_adaptive_context.train_adaptive",
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--data-root",
        str(data_root),
        "--short-checkpoint",
        str(stage_b / "last.pt"),
        "--long-checkpoint",
        str(long_teacher),
        "--results-dir",
        str(refiner),
        "--train-rows",
        "8640",
        "--val-rows",
        "432",
        "--context-seconds",
        "4.0",
        "--target-seconds",
        "4.0",
        "--batch-size",
        "128",
        "--num-workers",
        "0",
        "--epochs",
        "20",
        "--seed",
        str(seed),
        "--rf-span",
        "384",
        "--training-threshold",
        "0.13",
        "--activation-threshold",
        "0.13",
        "--label-weight",
        "2.0",
        "--distill-weight",
        "0.25",
        "--max-lr",
        "0.001",
        "--min-lr",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--max-residual",
        "2.0",
    ]
    return {
        "stage_a": (stage_a_command, stage_a / "last.pt"),
        "stage_b": (stage_b_command, stage_b / "last.pt"),
        "refiner": (refiner_command, refiner / "best.pt"),
    }


def run_ae4_training(
    data: AeData,
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
    device: str = "auto",
    chunk_frames: int = 2_000,
    progress_every: int = 100,
    train_manifest: str | Path = DEFAULT_TRAIN_MANIFEST,
    val_manifest: str | Path = DEFAULT_VAL_MANIFEST,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    long_teacher: str | Path = DEFAULT_LONG_TEACHER,
) -> dict[str, Any]:
    training_dir = output_dir / "ae4_replication_training"
    prediction_dir = output_dir / "ae4_replication_predictions"
    marker = _stage_marker(output_dir, "ae4_training")
    figure_path = output_dir / "ae4_model_relative_value.png"
    _refuse_overwrite(
        [
            marker,
            output_dir / "ae4_replication_summary.csv",
            figure_path,
            *[
                prediction_dir / f"seed{seed}" / "frame_predictions.npz"
                for seed in AE4_SEEDS
            ],
        ],
        stage="AE4 replication training",
    )
    train_manifest = Path(train_manifest)
    val_manifest = Path(val_manifest)
    data_root = Path(data_root)
    long_teacher = Path(long_teacher)
    seed_metrics: dict[int, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    for seed in AE4_SEEDS:
        seed_root = training_dir / f"seed{seed}"
        commands = _ae4_training_commands(
            seed=seed,
            seed_root=seed_root,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            data_root=data_root,
            long_teacher=long_teacher,
        )
        for stage_name, (command, expected_checkpoint) in commands.items():
            log_path = seed_root / f"{stage_name}.log"
            if expected_checkpoint.exists():
                raise FileExistsError(
                    f"refusing to overwrite completed {stage_name} "
                    f"for seed{seed}: {expected_checkpoint}"
                )
            print(
                f"AE4 seed{seed} {stage_name}: starting",
                flush=True,
            )
            _run_logged_command(command, log_path=log_path)
            if not expected_checkpoint.exists():
                raise RuntimeError(
                    f"{stage_name} did not produce {expected_checkpoint}"
                )
            command_rows.append(
                {
                    "seed": int(seed),
                    "stage": stage_name,
                    "command": " ".join(command),
                    "checkpoint": str(expected_checkpoint),
                    "checkpoint_sha256": file_sha256(expected_checkpoint),
                    "log": str(log_path),
                }
            )
        prediction_path = (
            prediction_dir / f"seed{seed}" / "frame_predictions.npz"
        )
        arrays = _predict_checkpoint_on_a9(
            data,
            checkpoint_path=commands["refiner"][1],
            protocol=protocol,
            output_path=prediction_path,
            device=device,
            chunk_frames=chunk_frames,
            progress_every=progress_every,
        )
        metrics = _value_metrics_for_bundle(
            data=data,
            labels=arrays["labels"],
            short_scores=arrays["short_scores"],
            refined_scores=arrays["refined_scores"],
        )
        seed_metrics[seed] = metrics
        metric_rows.extend(
            _ae4_seed_metric_rows(
                data,
                seed=seed,
                role="independent_initialization_replicate",
                labels=arrays["labels"],
                short_scores=arrays["short_scores"],
                refined_scores=arrays["refined_scores"],
            )
        )
    pair_rows = _ae4_pair_rows(data, seed_values=seed_metrics)
    _write_csv(
        output_dir / "ae4_replication_summary.csv",
        metric_rows + pair_rows,
    )
    _write_csv(
        output_dir / "ae4_training_manifest.csv",
        command_rows,
    )
    _write_ae4_figure(output_dir)
    artifacts = [
        output_dir / "ae4_replication_summary.csv",
        output_dir / "ae4_training_manifest.csv",
        figure_path,
        *[
            prediction_dir / f"seed{seed}" / "frame_predictions.npz"
            for seed in AE4_SEEDS
        ],
    ]
    _write_stage_marker(
        output_dir,
        stage="ae4_training",
        artifacts=artifacts,
    )
    return {
        "rows": metric_rows,
        "pairs": pair_rows,
        "seed_metrics": seed_metrics,
        "commands": command_rows,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _ae0_group_payload() -> dict[str, Any]:
    payload = _load_json(AE0_RESULT_PATH)
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("AE0 result is missing groups")
    return payload


def _ae4_agreement_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    lookup: dict[str, dict[str, float | None]] = {}
    for row in rows:
        if row.get("left_seed") and row.get("right_seed"):
            continue
        group = str(row.get("group", ""))
        if not group:
            continue
        key = "checkpoint" if "shared_short" in str(row.get("role", "")) or "independent_short" in str(row.get("role", "")) else "replication"
        entry = lookup.setdefault(
            group,
            {
                "checkpoint_value_agreement": None,
                "replication_value_agreement": None,
                "checkpoint_availability_net": None,
                "replication_availability_net": None,
            },
        )
        if key == "checkpoint":
            entry["checkpoint_availability_net"] = _as_float(
                row.get("availability_net")
            )
        else:
            entry["replication_availability_net"] = _as_float(
                row.get("availability_net")
            )
    return lookup


def _pair_agreement_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {
        "checkpoint_value_agreement": {},
        "replication_value_agreement": {},
    }
    for row in rows:
        if "left_seed" not in row or "right_seed" not in row:
            continue
        role = str(row.get("comparison_role", ""))
        agreement = _as_float(row.get("sign_agreement"))
        direction = _as_float(
            row.get("condition_level_direction_agreement")
        )
        if "shared_short_encoder" in role or "independent_short_encoder" in role:
            bucket = result["checkpoint_value_agreement"]
        else:
            bucket = result["replication_value_agreement"]
        for key, value in (
            ("sign_agreement", agreement),
            ("condition_level_direction_agreement", direction),
        ):
            if value is None:
                continue
            bucket.setdefault(key, []).append(value)
    return result


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def _classify_ae5(
    *,
    availability_net: float | None,
    observability_auroc: float | None,
    gate_utility: float | None,
    stable_long_rate: float | None,
    checkpoint_agreement: float | None,
    replication_agreement: float | None,
) -> tuple[str, str]:
    """Apply the frozen AE5 precedence with a deterministic agreement rule."""
    if availability_net is None:
        return "VALUE_SCARCITY", "availability_net is undefined"
    if availability_net <= 0.0:
        return "VALUE_SCARCITY", "availability_net <= 0"
    if observability_auroc is None or observability_auroc <= 0.5:
        return (
            "OBSERVABILITY_LIMITED",
            "availability is positive but observability AUROC <= 0.5",
        )
    if gate_utility is not None and gate_utility <= 0.0:
        return (
            "ACTIONABILITY_LIMITED",
            "availability and observability are positive but gate utility <= 0",
        )
    if stable_long_rate is not None and stable_long_rate >= 0.5:
        return (
            "LONG_HORIZON_DEMAND",
            "at least half of defined stable horizons are RF256-RF512",
        )
    agreements = [
        value
        for value in (checkpoint_agreement, replication_agreement)
        if value is not None
    ]
    low_agreement = bool(agreements) and min(agreements) < 0.5
    if low_agreement:
        return (
            "MODEL_RELATIVE_VALUE",
            "availability is positive but a measured sign agreement is below 0.5",
        )
    if agreements and min(agreements) > 0.5:
        return (
            "REPLICATED_VALUE",
            "availability is positive and measured sign agreement is above chance",
        )
    return (
        "TEMPORAL_SPAN_NOT_LIMITING",
        "no long-horizon demand or agreement-based model-relative signal",
    )


def _build_ae5_rows(
    *,
    ae0_payload: Mapping[str, Any],
    ae1_rows: Sequence[Mapping[str, Any]],
    ae2_rows: Sequence[Mapping[str, Any]],
    ae3_rows: Sequence[Mapping[str, Any]],
    ae4_rows: Sequence[Mapping[str, Any]],
    ae4_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ae0_groups = ae0_payload["groups"]
    ae1_by_group = {str(row["group"]): row for row in ae1_rows}
    ae2_by_group = {str(row["group"]): row for row in ae2_rows}
    ae3_by_group = {
        (str(row["feature"]), str(row["group"])): row
        for row in ae3_rows
    }
    seed_availability: dict[str, dict[str, list[float]]] = {}
    for row in ae4_rows:
        if row.get("left_seed") and row.get("right_seed"):
            continue
        group = str(row.get("group", ""))
        value = _as_float(row.get("availability_net"))
        if group and value is not None:
            seed_availability.setdefault(group, {}).setdefault(
                str(row.get("role", "")),
                [],
            ).append(value)
    pair_lookup = _pair_agreement_lookup(ae4_pairs)
    rows: list[dict[str, Any]] = []
    for group in CELL_ORDER:
        ae0_group = ae0_groups.get(group)
        if not isinstance(ae0_group, Mapping):
            continue
        ae0_point = ae0_group.get("point", {})
        ae1 = ae1_by_group.get(group, {})
        ae2 = ae2_by_group.get(group, {})
        ae3 = ae3_by_group.get(("uncertainty_persistence", group), {})
        availability_net = _as_float(ae0_point.get("availability_net"))
        if availability_net is None:
            availability_net = _as_float(ae1.get("availability_net"))
        observability = _as_float(
            ae0_point.get("observability_auroc")
        )
        gate_utility = _as_float(ae0_point.get("gate_utility"))
        if gate_utility is None:
            gate_utility = _as_float(ae1.get("gate_utility"))
        activation = _as_float(ae0_point.get("activation_rate"))
        if activation is None:
            activation = _as_float(ae1.get("activation_rate"))
        stable_long_rate = _as_float(
            ae2.get("stable_sufficient_long_rate")
        )
        checkpoint_agreement = _mean_or_none(
            pair_lookup["checkpoint_value_agreement"].get(
                "sign_agreement",
                [],
            )
        )
        replication_agreement = _mean_or_none(
            pair_lookup["replication_value_agreement"].get(
                "sign_agreement",
                [],
            )
        )
        classification, reason = _classify_ae5(
            availability_net=availability_net,
            observability_auroc=observability,
            gate_utility=gate_utility,
            stable_long_rate=stable_long_rate,
            checkpoint_agreement=checkpoint_agreement,
            replication_agreement=replication_agreement,
        )
        rows.append(
            {
                "group": group,
                "domain": (
                    None
                    if group in ("all", "clean")
                    else group.split("/", 1)[0]
                ),
                "snr_db": (
                    None
                    if group in ("all", "clean")
                    else group.split("/", 1)[1]
                ),
                "classification": classification,
                "classification_reason": reason,
                "availability_net": availability_net,
                "observability_auroc": observability,
                "gate_utility": gate_utility,
                "activation_rate": activation,
                "refinable_rate": _as_float(ae1.get("state_rate_refinable")),
                "refinement_harm_rate": _as_float(
                    ae1.get("state_rate_refinement_harm")
                ),
                "rf384_irreducible_rate": _as_float(
                    ae1.get("state_rate_rf384_irreducible")
                ),
                "mean_stable_sufficient_span": _as_float(
                    ae2.get("mean_stable_sufficient_span")
                ),
                "stable_sufficient_long_rate": stable_long_rate,
                "non_monotone_rate": _as_float(
                    ae2.get("non_monotone_rate")
                ),
                "uncertainty_persistence_availability_net": _as_float(
                    ae3.get("availability_net")
                ),
                "checkpoint_value_agreement": checkpoint_agreement,
                "replication_value_agreement": replication_agreement,
                "checkpoint_availability_net": _mean_or_none(
                    seed_availability.get(group, {}).get(
                        "shared_short_encoder_checkpoint",
                        [],
                    )
                    + seed_availability.get(group, {}).get(
                        "independent_short_encoder_checkpoint",
                        [],
                    )
                ),
                "replication_availability_net": _mean_or_none(
                    seed_availability.get(group, {}).get(
                        "independent_initialization_replicate",
                        [],
                    )
                ),
            }
        )
    return rows


def _plot_ae1(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["group"]) for row in rows]
    values = [
        [
            _as_float(row.get("state_rate_short_sufficient")) or 0.0,
            _as_float(row.get("state_rate_refinable")) or 0.0,
            _as_float(row.get("state_rate_rf384_irreducible")) or 0.0,
            _as_float(row.get("state_rate_refinement_harm")) or 0.0,
        ]
        for row in rows
    ]
    array = np.asarray(values, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(array.shape[0])
    names = (
        "Short-sufficient",
        "Refinable",
        "RF384-Irreducible",
        "Refinement-Harm",
    )
    colors = ("#4C78A8", "#54A24B", "#E45756", "#F58518")
    x = np.arange(array.shape[0])
    for index, name in enumerate(names):
        axis.bar(x, array[:, index], bottom=bottom, label=name, color=colors[index])
        bottom += array[:, index]
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Frame state rate")
    axis.set_title("AE1: frozen refinement-value states")
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ae2(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = [str(row["group"]) for row in rows]
    spans = list(SPAN_ORDER)
    matrix = np.asarray(
        [
            [
                _as_float(row.get(f"oracle_net_utility_rf{span}")) or 0.0
                for span in spans
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(len(spans)))
    axis.set_xticklabels([f"RF{span}" for span in spans])
    axis.set_yticks(np.arange(len(groups)))
    axis.set_yticklabels(groups, fontsize=8)
    axis.set_title("AE2: oracle refinement utility by temporal span")
    fig.colorbar(image, ax=axis, label="Oracle net utility")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ae3(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    features = list(FEATURE_ORDER)
    fig, axes = plt.subplots(
        len(features),
        1,
        figsize=(10, 2.0 * len(features)),
        sharex=False,
    )
    for axis, feature in zip(np.atleast_1d(axes), features):
        local = [row for row in rows if str(row["feature"]) == feature]
        labels = [str(row["bin"]) for row in local]
        values = [
            _as_float(row.get("availability_net")) or 0.0 for row in local
        ]
        axis.bar(np.arange(len(labels)), values, color="#4C78A8")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylabel(feature, fontsize=7)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    fig.suptitle("AE3: feature-bin availability gradients")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ae4(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    seed_rows = [row for row in rows if "seed" in row and "group" in row]
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    matrix = np.asarray(
        [
            [
                _as_float(row.get("mean_value")) or 0.0
                for row in seed_rows
                if int(row["seed"]) == seed
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    if matrix.size == 0:
        matrix = np.zeros((1, 1))
    fig, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(matrix.shape[1]))
    groups = [
        str(row["group"]) for row in seed_rows if int(row["seed"]) == seeds[0]
    ]
    axis.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(seeds)))
    axis.set_yticklabels([f"seed{seed}" for seed in seeds])
    axis.set_title("AE4: refinement-value direction by checkpoint/replicate")
    fig.colorbar(image, ax=axis, label="Mean value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_ae4_figure(output_dir: Path) -> Path:
    checkpoint_rows = _read_csv_rows(
        output_dir / "ae4_checkpoint_agreement.csv"
    )
    replication_rows = _read_csv_rows(
        output_dir / "ae4_replication_summary.csv"
    )
    metric_rows = [
        row
        for row in [*checkpoint_rows, *replication_rows]
        if not (row.get("left_seed") and row.get("right_seed"))
    ]
    pair_rows = [
        row
        for row in [*checkpoint_rows, *replication_rows]
        if row.get("left_seed") and row.get("right_seed")
    ]
    path = output_dir / "ae4_model_relative_value.png"
    _plot_ae4(metric_rows, pair_rows, path)
    return path


def _plot_ae5(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["group"]) for row in rows]
    values = [_as_float(row.get("availability_net")) or 0.0 for row in rows]
    colors = [
        {
            "VALUE_SCARCITY": "#E45756",
            "OBSERVABILITY_LIMITED": "#F58518",
            "ACTIONABILITY_LIMITED": "#B279A2",
            "LONG_HORIZON_DEMAND": "#4C78A8",
            "TEMPORAL_SPAN_NOT_LIMITING": "#72B7B2",
            "MODEL_RELATIVE_VALUE": "#9D755D",
            "REPLICATED_VALUE": "#54A24B",
        }.get(str(row["classification"]), "#888888")
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(np.arange(len(rows)), values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(rows)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Availability net")
    axis.set_title("AE5: condition-level mechanism map")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_ae5(
    *,
    output_dir: Path,
) -> dict[str, Any]:
    marker = _stage_marker(output_dir, "ae5")
    _refuse_overwrite(
        [
            marker,
            output_dir / "ae5_condition_mechanism_map.csv",
            output_dir / "ae5_mechanism_summary.json",
            output_dir / "ae5_mechanism_map.png",
        ],
        stage="AE5 integration",
    )
    ae0_payload = _ae0_group_payload()
    ae1_rows = _read_csv_rows(output_dir / "ae1_state_summary.csv")
    ae2_rows = _read_csv_rows(output_dir / "ae2_horizon_summary.csv")
    ae3_rows = _read_csv_rows(output_dir / "ae3_feature_bins.csv")
    ae4_rows = _read_csv_rows(output_dir / "ae4_checkpoint_agreement.csv")
    ae4_pairs = [
        row
        for row in ae4_rows
        if row.get("left_seed") and row.get("right_seed")
    ]
    replication_rows = _read_csv_rows(
        output_dir / "ae4_replication_summary.csv"
    )
    replication_pairs = [
        row
        for row in replication_rows
        if row.get("left_seed") and row.get("right_seed")
    ]
    ae4_rows = [
        row for row in ae4_rows if not (row.get("left_seed") and row.get("right_seed"))
    ]
    rows = _build_ae5_rows(
        ae0_payload=ae0_payload,
        ae1_rows=ae1_rows,
        ae2_rows=ae2_rows,
        ae3_rows=ae3_rows,
        ae4_rows=ae4_rows + [
            row
            for row in replication_rows
            if not (row.get("left_seed") and row.get("right_seed"))
        ],
        ae4_pairs=ae4_pairs + replication_pairs,
    )
    _write_csv(output_dir / "ae5_condition_mechanism_map.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["classification"])] = counts.get(
            str(row["classification"]),
            0,
        ) + 1
    summary = {
        "classification_counts": counts,
        "agreement_rule": (
            "low = below-chance sign agreement (<0.5); replicated = "
            "above-chance agreement (>0.5) and positive availability"
        ),
        "rows": rows,
        "stop_rule": ae0_payload.get("classification", {}),
    }
    _write_json(output_dir / "ae5_mechanism_summary.json", summary)
    _plot_ae5(rows, output_dir / "ae5_mechanism_map.png")
    artifacts = [
        output_dir / "ae5_condition_mechanism_map.csv",
        output_dir / "ae5_mechanism_summary.json",
        output_dir / "ae5_mechanism_map.png",
    ]
    _write_stage_marker(output_dir, stage="ae5", artifacts=artifacts)
    return summary


def _artifact_hash_rows(
    output_dir: Path,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return rows


def _find_ae4_pair_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    role_fragment: str,
    key: str,
) -> float | None:
    values: list[float] = []
    for row in rows:
        if not (row.get("left_seed") and row.get("right_seed")):
            continue
        if role_fragment not in str(row.get("comparison_role", "")):
            continue
        value = _as_float(row.get(key))
        if value is not None:
            values.append(value)
    return _mean_or_none(values)


def _build_final_summary(
    *,
    protocol: FrozenProtocol,
    output_dir: Path,
) -> dict[str, Any]:
    ae1_rows = _read_csv_rows(output_dir / "ae1_state_summary.csv")
    ae2_rows = _read_csv_rows(output_dir / "ae2_horizon_summary.csv")
    ae3_rows = _read_csv_rows(output_dir / "ae3_feature_bins.csv")
    checkpoint_rows = _read_csv_rows(
        output_dir / "ae4_checkpoint_agreement.csv"
    )
    replication_rows = _read_csv_rows(
        output_dir / "ae4_replication_summary.csv"
    )
    ae5_payload = _load_json(output_dir / "ae5_mechanism_summary.json")
    ae0_payload = _ae0_group_payload()
    all_ae1 = next(row for row in ae1_rows if row["group"] == "all")
    all_ae2 = next(row for row in ae2_rows if row["group"] == "all")
    checkpoint_pairs = [
        row
        for row in checkpoint_rows
        if row.get("left_seed") and row.get("right_seed")
    ]
    replication_pairs = [
        row
        for row in replication_rows
        if row.get("left_seed") and row.get("right_seed")
    ]
    shared_agreement = _find_ae4_pair_metric(
        checkpoint_pairs,
        role_fragment="shared_short_encoder",
        key="sign_agreement",
    )
    independent_agreement = _find_ae4_pair_metric(
        checkpoint_pairs,
        role_fragment="independent_short_encoder",
        key="sign_agreement",
    )
    replication_agreement = _find_ae4_pair_metric(
        replication_pairs,
        role_fragment="replication_pair",
        key="sign_agreement",
    )
    classification_counts = dict(
        ae5_payload.get("classification_counts", {})
    )
    stop_triggered = bool(
        ae0_payload.get("classification", {}).get("stop_rule_triggered")
    )
    summary = {
        "version": PROTOCOL_ID,
        "date": "2026-09-20",
        "protocol": {
            "path": str(protocol.path),
            "sha256": protocol.sha256,
            "status": protocol.payload["status"],
        },
        "scope": {
            "a14": "opened once; post-hoc A-v2 development data",
            "a_v1": "COMPLETE_CONDITIONAL_GO",
            "a_v1_method_search": "CLOSED",
            "a15": "FORBIDDEN",
            "router_search": "forbidden",
            "gate_tuning": "forbidden",
            "model_selection_from_ae_outcomes": "forbidden",
        },
        "ae1": {
            "frames": int(float(all_ae1["frames"])),
            "availability_net": _as_float(all_ae1["availability_net"]),
            "gate_utility": _as_float(all_ae1["gate_utility"]),
            "activation_rate": _as_float(all_ae1["activation_rate"]),
            "refinable_rate": _as_float(
                all_ae1["state_rate_refinable"]
            ),
            "rf384_irreducible_rate": _as_float(
                all_ae1["state_rate_rf384_irreducible"]
            ),
            "refinement_harm_rate": _as_float(
                all_ae1["state_rate_refinement_harm"]
            ),
        },
        "ae2": {
            "non_monotone_rate": _as_float(all_ae2["non_monotone_rate"]),
            "mean_stable_sufficient_span": _as_float(
                all_ae2["mean_stable_sufficient_span"]
            ),
            "stable_sufficient_long_rate": _as_float(
                all_ae2["stable_sufficient_long_rate"]
            ),
            "never_correct_rate": _as_float(all_ae2["never_correct_rate"]),
            "unstable_correct_rate": _as_float(
                all_ae2["unstable_correct_rate"]
            ),
        },
        "ae3": {
            "feature_bin_rows": len(ae3_rows),
            "interpretation": (
                "feature-bin gradients are descriptive mechanism diagnostics; "
                "no post-outcome feature selection was performed"
            ),
        },
        "ae4": {
            "checkpoint_shared_short_sign_agreement": shared_agreement,
            "checkpoint_independent_short_sign_agreement": independent_agreement,
            "replication_sign_agreement": replication_agreement,
            "required_wording": REPLICATION_WORDING,
            "a14_relation_wording": (
                "The five training replicates are not independent encoders "
                "relative to the A14 checkpoints."
            ),
        },
        "ae5": {
            "classification_counts": classification_counts,
            "classification_rule": ae5_payload.get("agreement_rule"),
        },
        "ae0_stop_rule_triggered": stop_triggered,
        "claim_freeze": "A-v2 AE CLAIM FREEZE",
        "allowed_next_step": (
            "Only a separately authorized A-v2 method study with a new "
            "untouched OOD set."
        ),
        "forbidden_next_steps": protocol.payload["claim_freeze"][
            "forbidden_next_steps"
        ],
    }
    return summary


def _final_report_lines(
    *,
    summary: Mapping[str, Any],
    ae5_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    counts = summary["ae5"]["classification_counts"]
    lines = [
        "# A-v2 / AE Final Mechanism Report",
        "",
        f"- Protocol: `{summary['protocol']['sha256']}`",
        f"- Status: **{summary['claim_freeze']}**",
        "- A-v1: `COMPLETE — CONDITIONAL GO`",
        "- A15: `FORBIDDEN`",
        "",
        "## Scope",
        "",
        (
            "AE0 uses the already opened A14 development data. "
            "AE1-AE4 use the frozen A9 test manifest and frozen predictions; "
            "AE4 adds post-hoc checkpoint inference and the pre-registered "
            "five paired Short/RF384 replication runs. The A14 and A9 frame "
            "counts are different by design and are not pooled."
        ),
        "",
        "## AE1-AE2",
        "",
        (
            f"- AE1 aggregate availability net: "
            f"`{_format_float(summary['ae1']['availability_net'])}`"
        ),
        (
            f"- AE1 aggregate gate utility: "
            f"`{_format_float(summary['ae1']['gate_utility'])}`"
        ),
        (
            f"- AE1 aggregate activation rate: "
            f"`{_format_float(summary['ae1']['activation_rate'])}`"
        ),
        (
            f"- AE2 non-monotone rate: "
            f"`{_format_float(summary['ae2']['non_monotone_rate'])}`"
        ),
        (
            f"- AE2 mean stable-sufficient span: "
            f"`{_format_float(summary['ae2']['mean_stable_sufficient_span'])}`"
        ),
        "",
        "## AE4",
        "",
        (
            f"- Shared-Short checkpoint sign agreement: "
            f"`{_format_float(summary['ae4']['checkpoint_shared_short_sign_agreement'])}`"
        ),
        (
            f"- Independent-Short seed23 sensitivity agreement: "
            f"`{_format_float(summary['ae4']['checkpoint_independent_short_sign_agreement'])}`"
        ),
        (
            f"- Replication sign agreement: "
            f"`{_format_float(summary['ae4']['replication_sign_agreement'])}`"
        ),
        "",
        summary["ae4"]["required_wording"] + ".",
        "",
        summary["ae4"]["a14_relation_wording"],
        "",
        "## AE5 Mechanism Map",
        "",
    ]
    for name in (
        "VALUE_SCARCITY",
        "OBSERVABILITY_LIMITED",
        "ACTIONABILITY_LIMITED",
        "LONG_HORIZON_DEMAND",
        "TEMPORAL_SPAN_NOT_LIMITING",
        "MODEL_RELATIVE_VALUE",
        "REPLICATED_VALUE",
    ):
        lines.append(f"- `{name}`: `{counts.get(name, 0)}`")
    lines.extend(
        [
            "",
            "| Group | Classification | Availability | Observability | Gate utility |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ae5_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["classification"]),
                    _format_float(row.get("availability_net")),
                    _format_float(row.get("observability_auroc")),
                    _format_float(row.get("gate_utility")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Claim Freeze",
            "",
            "- `A-v2 AE CLAIM FREEZE`",
            "- No router, gate, architecture, or model selection follows from this report.",
            (
                "- Any later method study requires a separately authorized "
                "candidate and a new untouched OOD set."
            ),
            "",
        ]
    )
    return lines


def finalize_ae(
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    marker = _stage_marker(output_dir, "final")
    final_paths = [
        output_dir / "ae_summary.json",
        output_dir / "ae_final_report.md",
        output_dir / "ae_protocol_hash.json",
        output_dir / "ae_execution_manifest.json",
        output_dir / "ae_claim_freeze.json",
    ]
    _refuse_overwrite(
        [marker, *final_paths],
        stage="AE final claim freeze",
    )
    summary = _build_final_summary(
        protocol=protocol,
        output_dir=output_dir,
    )
    _write_json(output_dir / "ae_summary.json", summary)
    ae5_rows = _read_csv_rows(
        output_dir / "ae5_condition_mechanism_map.csv"
    )
    report = "\n".join(
        _final_report_lines(summary=summary, ae5_rows=ae5_rows)
    )
    (output_dir / "ae_final_report.md").write_text(
        report,
        encoding="utf-8",
    )
    _write_json(
        output_dir / "ae_protocol_hash.json",
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_path": str(protocol.path),
            "protocol_sha256": protocol.sha256,
            "freeze_manifest_path": str(FREEZE_MANIFEST_PATH),
            "freeze_manifest_sha256": file_sha256(FREEZE_MANIFEST_PATH),
            "status": protocol.payload["status"],
        },
    )
    table_paths = [
        output_dir / name
        for name in protocol.payload["outputs"]["tables"]
    ]
    figure_paths = [
        output_dir / name
        for name in protocol.payload["outputs"]["figures"]
    ]
    core_paths = [
        output_dir / "ae_summary.json",
        output_dir / "ae_final_report.md",
        output_dir / "ae_protocol_hash.json",
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol.sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "study_sequence": protocol.payload["study"]["sequence"],
        "scope_note": (
            "AE0 uses opened A14 development data; AE1-AE4 use the frozen "
            "A9 test manifest. Frame counts are not pooled."
        ),
        "artifacts": _artifact_hash_rows(
            output_dir,
            [*core_paths, *table_paths, *figure_paths],
        ),
    }
    _write_json(output_dir / "ae_execution_manifest.json", manifest)
    _write_json(
        output_dir / "ae_claim_freeze.json",
        {
            "status": "A-v2 AE CLAIM FREEZE",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol.sha256,
            "summary_sha256": file_sha256(output_dir / "ae_summary.json"),
            "report_sha256": file_sha256(output_dir / "ae_final_report.md"),
            "execution_manifest_sha256": file_sha256(
                output_dir / "ae_execution_manifest.json"
            ),
            "a15": "FORBIDDEN",
            "router_or_gate_tuning": "FORBIDDEN",
            "automatic_method_experiment": "FORBIDDEN",
            "allowed_next_step": (
                "Only a separately authorized A-v2 method study with a new "
                "untouched OOD set."
            ),
        },
    )
    _write_stage_marker(
        output_dir,
        stage="final",
        artifacts=[*final_paths, *table_paths, *figure_paths],
    )
    return summary


def _run_ae1_stage(
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
    test_manifest: str | Path,
    data_root: str | Path,
    librispeech_root: str | Path,
) -> dict[str, Any]:
    marker = _stage_marker(output_dir, "ae1")
    artifacts = [
        output_dir / "ae1_state_summary.csv",
        output_dir / "ae1_state_by_cell.csv",
        output_dir / "ae1_condition_availability.csv",
        output_dir / "ae1_state_rates.png",
    ]
    _refuse_overwrite([marker, *artifacts], stage="AE1")
    data = load_a9_data(
        protocol,
        test_manifest=test_manifest,
        data_root=data_root,
        librispeech_root=librispeech_root,
    )
    result = run_ae1(data, output_dir=output_dir)
    _plot_ae1(
        result["rows"],
        output_dir / "ae1_state_rates.png",
    )
    _write_stage_marker(
        output_dir,
        stage="ae1",
        artifacts=artifacts,
    )
    return result


def _run_ae2_stage(
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
    test_manifest: str | Path,
    data_root: str | Path,
    librispeech_root: str | Path,
) -> dict[str, Any]:
    marker = _stage_marker(output_dir, "ae2")
    table_paths = [
        output_dir / "ae2_horizon_summary.csv",
        output_dir / "ae2_horizon_by_cell.csv",
        output_dir / "ae2_nonmonotonicity.csv",
    ]
    figure_path = output_dir / "ae2_horizon_saturation.png"
    _refuse_overwrite([marker, figure_path], stage="AE2")
    existing_tables = [path for path in table_paths if path.exists()]
    if len(existing_tables) == len(table_paths):
        rows = _read_csv_rows(table_paths[0])
        _plot_ae2(rows, figure_path)
        _write_stage_marker(
            output_dir,
            stage="ae2",
            artifacts=[*table_paths, figure_path],
        )
        return {"rows": rows, "resumed_from_existing_tables": True}
    if existing_tables:
        raise FileExistsError(
            "AE2 has partial tabular output; refusing to overwrite: "
            f"{[str(path) for path in existing_tables]}"
        )
    data = load_a9_data(
        protocol,
        test_manifest=test_manifest,
        data_root=data_root,
        librispeech_root=librispeech_root,
    )
    result = run_ae2(data, output_dir=output_dir)
    _plot_ae2(
        result["rows"],
        figure_path,
    )
    _write_stage_marker(
        output_dir,
        stage="ae2",
        artifacts=[*table_paths, figure_path],
    )
    return result


def _run_ae3_stage(
    protocol: FrozenProtocol,
    *,
    output_dir: Path,
    test_manifest: str | Path,
    data_root: str | Path,
    librispeech_root: str | Path,
) -> dict[str, Any]:
    marker = _stage_marker(output_dir, "ae3")
    artifacts = [
        output_dir / "ae3_event_feature_summary.csv",
        output_dir / "ae3_feature_bins.csv",
        output_dir / "ae3_condition_feature_map.csv",
        output_dir / "ae3_feature_gradients.png",
    ]
    _refuse_overwrite([marker, *artifacts], stage="AE3")
    data = load_a9_data(
        protocol,
        test_manifest=test_manifest,
        data_root=data_root,
        librispeech_root=librispeech_root,
    )
    result = run_ae3(data, output_dir=output_dir)
    all_bins = [
        row
        for row in result["bins"]
        if str(row["group"]) == "all"
    ]
    _plot_ae3(
        all_bins,
        output_dir / "ae3_feature_gradients.png",
    )
    _write_stage_marker(
        output_dir,
        stage="ae3",
        artifacts=artifacts,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen A-v2 / AE mechanism study."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "ae1",
            "ae2",
            "ae3",
            "ae4-existing",
            "ae4-train",
            "ae4",
            "ae5",
            "finalize",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_TEST_MANIFEST,
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=DEFAULT_VAL_MANIFEST,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=DEFAULT_LIBRISPEECH_ROOT,
    )
    parser.add_argument(
        "--long-teacher",
        type=Path,
        default=DEFAULT_LONG_TEACHER,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-chunk-frames", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    protocol = load_frozen_protocol()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("ae1", "all"):
        _run_ae1_stage(
            protocol,
            output_dir=output_dir,
            test_manifest=args.test_manifest,
            data_root=args.data_root,
            librispeech_root=args.librispeech_root,
        )
    if args.stage in ("ae2", "all"):
        _run_ae2_stage(
            protocol,
            output_dir=output_dir,
            test_manifest=args.test_manifest,
            data_root=args.data_root,
            librispeech_root=args.librispeech_root,
        )
    if args.stage in ("ae3", "all"):
        _run_ae3_stage(
            protocol,
            output_dir=output_dir,
            test_manifest=args.test_manifest,
            data_root=args.data_root,
            librispeech_root=args.librispeech_root,
        )
    if args.stage in ("ae4-existing", "ae4", "all"):
        data = load_a9_data(
            protocol,
            test_manifest=args.test_manifest,
            data_root=args.data_root,
            librispeech_root=args.librispeech_root,
        )
        run_ae4_existing(
            data,
            protocol,
            output_dir=output_dir,
            device=args.device,
            chunk_frames=args.score_chunk_frames,
            progress_every=args.progress_every,
        )
    if args.stage in ("ae4-train", "ae4", "all"):
        data = load_a9_data(
            protocol,
            test_manifest=args.test_manifest,
            data_root=args.data_root,
            librispeech_root=args.librispeech_root,
        )
        run_ae4_training(
            data,
            protocol,
            output_dir=output_dir,
            device=args.device,
            chunk_frames=args.score_chunk_frames,
            progress_every=args.progress_every,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            data_root=args.data_root,
            long_teacher=args.long_teacher,
        )
    if args.stage in ("ae5", "all"):
        run_ae5(output_dir=output_dir)
    if args.stage in ("finalize", "all"):
        finalize_ae(protocol, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
