# -*- coding: utf-8 -*-
"""Execute the frozen E3 boundary and value-robustness audit.

E3 is an analysis-only audit of already frozen predictions.  It does not
train, fine-tune, change checkpoints, alter interventions, or inspect the
new final OOD set.
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
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproductions.difficulty_adaptive_context.run_a13_streaming_audit import (
    SegmentProtocol,
    Utterance,
    evaluate_segment_scores,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "E3-BVRA-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_E3_OUTCOME_ANALYSIS"
RUNNER_MODULE = "reproductions.difficulty_adaptive_context.run_e3"
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e3_boundary_value_robustness"
)
PROTOCOL_PATH = OUTPUT_ROOT / "e3_protocol_freeze.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "e3_protocol_sha256.txt"
RUNNER_HASH_PATH = OUTPUT_ROOT / "e3_runner_sha256.txt"
BASELINE_PATH = OUTPUT_ROOT / "e3_baseline_reproduction.json"
EXECUTION_MANIFEST_PATH = OUTPUT_ROOT / "e3_execution_manifest.json"

E1_RESULTS_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e1_remote_temporal_intervention"
)
E1_FRAME_RESULTS_PATH = E1_RESULTS_ROOT / "e1_v2_frame_results.parquet"
E1_FINAL_SUMMARY_PATH = E1_RESULTS_ROOT / "e1_v2_final_summary.json"
E1_PROTOCOL_PATH = E1_RESULTS_ROOT / "e1_v2_protocol_freeze.json"
E1_RUNNER_PATH = (
    REPO_ROOT / "reproductions" / "difficulty_adaptive_context" / "run_e1.py"
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
TINY_GRU_SCORE_PATHS = tuple(
    REPO_ROOT
    / "results"
    / "cross_architecture_replication"
    / "evaluations"
    / f"seed{seed}"
    / "scores.npz"
    for seed in (41, 59, 71, 83, 97)
)
TINY_GRU_SEED_VALUES = (41, 59, 71, 83, 97)

THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
EXCLUSION_WIDTHS_MS = (0, 10, 20, 30, 50, 100)
JITTER_WIDTHS_MS = (10, 20, 30, 50)
JITTER_SEEDS = (113, 227, 349, 463, 587)
FRAME_DURATION_MS = 10.0
FRAME_DURATION_SECONDS = 0.01
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260921
LOGLOSS_EPSILON = 1e-12
MISSING_DISTANCE = 1_000_000.0
DECISION_THRESHOLD = 0.50
D1_SMALL_MINORITY_MAX = 0.05
EVENT_BINS = ("0", "1-2", "3-5", "6-10", "11-25", "26+")
EVENT_DIMENSIONS = ("onset_distance", "posterior_transition_distance", "offset_distance")
SOURCE_LOSO_SIGN_TOLERANCE = 0.0
MIN_SEED_POSITIVE_COUNT = 4
SEGMENT_PROTOCOL = SegmentProtocol(
    decision_threshold=DECISION_THRESHOLD,
    short_speech_max_frames=20,
    onset_tolerance_frames=20,
)

EXPECTED_TEST_FRAMES = 298_300
EXPECTED_TEST_SOURCES = 96
EXPECTED_TEST_SPEAKERS = 20
BASELINE_TOLERANCE = 1e-6

REQUIRED_OUTPUT_FILES = (
    "e3_protocol_freeze.json",
    "e3_protocol_sha256.txt",
    "e3_runner_sha256.txt",
    "e3_execution_manifest.json",
    "e3_baseline_reproduction.json",
    "e3_threshold_robustness.csv",
    "e3_boundary_exclusion.csv",
    "e3_boundary_bootstrap.csv",
    "e3_label_jitter.csv",
    "e3_label_jitter_bootstrap.csv",
    "e3_event_profiles.csv",
    "e3_event_contrasts.csv",
    "e3_proper_scoring_value.csv",
    "e3_decision_vs_probabilistic_value.csv",
    "e3_value_concentration.csv",
    "e3_segment_metrics.csv",
    "e3_condition_robustness.csv",
    "e3_source_influence.csv",
    "e3_final_summary.json",
    "e3_final_report.md",
    "e3_claim_freeze.md",
)

FIGURE_FILES = (
    "figure1_threshold_taxonomy.png",
    "figure2_boundary_robustness.png",
    "figure3_event_profiles.png",
    "figure4_label_jitter.png",
    "figure5_decision_vs_probabilistic_value.png",
)
OPTIONAL_FIGURE_FILE = "figure6_cross_architecture.png"


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    source_key: str
    condition: str
    noise_name: str


@dataclass(frozen=True)
class SourceIndex:
    names: np.ndarray
    codes: np.ndarray
    count: int


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
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
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


def _stable_hash_int(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:12], 16)


def _clip_probabilities(scores: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(scores, dtype=np.float64),
        LOGLOSS_EPSILON,
        1.0 - LOGLOSS_EPSILON,
    )


def _logloss_frame(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = _clip_probabilities(scores)
    return -(
        labels * np.log(probabilities)
        + (1.0 - labels) * np.log(1.0 - probabilities)
    )


def _brier_frame(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    return (scores - labels) ** 2


def _correct_at(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.asarray(scores, dtype=np.float64) >= float(threshold)
    ) == np.asarray(labels, dtype=bool)


def _taxonomy_at(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    short_correct = _correct_at(labels, short_scores, threshold)
    full_correct = _correct_at(labels, full_scores, threshold)
    ss = short_correct & full_correct
    r = (~short_correct) & full_correct
    i = (~short_correct) & (~full_correct)
    h = short_correct & (~full_correct)
    return ss, r, i, h


def _f1(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(scores, dtype=np.float64) >= float(threshold)
    tp = int(np.count_nonzero(predictions & labels))
    fp = int(np.count_nonzero(predictions & ~labels))
    fn = int(np.count_nonzero(~predictions & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _event_bin(value: float) -> str:
    value = abs(float(value))
    if value == 0.0:
        return "0"
    if value <= 2.0:
        return "1-2"
    if value <= 5.0:
        return "3-5"
    if value <= 10.0:
        return "6-10"
    if value <= 25.0:
        return "11-25"
    return "26+"


def _find_runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _nearest_signed_distance(
    positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    length = int(end - start)
    result = np.full(length, MISSING_DISTANCE, dtype=np.float64)
    if positions.size == 0:
        return result
    frames = np.arange(start, end, dtype=np.int64)
    insertion = np.searchsorted(positions, frames)
    left = np.clip(insertion - 1, 0, positions.size - 1)
    right = np.clip(insertion, 0, positions.size - 1)
    left_distance = np.abs(frames - positions[left])
    right_distance = np.abs(frames - positions[right])
    choose_right = right_distance < left_distance
    nearest = np.where(choose_right, positions[right], positions[left])
    result[:] = frames - nearest
    return result


def _nearest_absolute_distance(
    positions: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    signed = _nearest_signed_distance(positions, start, end)
    return np.abs(signed)


def _event_distances(
    labels: np.ndarray,
    short_scores: np.ndarray,
    segments: Sequence[Segment],
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    onset = np.full(labels.size, MISSING_DISTANCE, dtype=np.float64)
    offset = np.full(labels.size, MISSING_DISTANCE, dtype=np.float64)
    posterior = np.full(labels.size, MISSING_DISTANCE, dtype=np.float64)
    for segment in segments:
        start, end = int(segment.start), int(segment.end)
        local_labels = labels[start:end]
        local_state = short_scores[start:end] >= float(threshold)
        onset_positions = (
            np.flatnonzero(
                (local_labels[1:] == 1) & (local_labels[:-1] == 0)
            )
            + 1
        )
        offset_positions = (
            np.flatnonzero(
                (local_labels[1:] == 0) & (local_labels[:-1] == 1)
            )
            + 1
        )
        transition_positions = (
            np.flatnonzero(local_state[1:] != local_state[:-1]) + 1
        )
        onset[start:end] = _nearest_signed_distance(
            onset_positions + start,
            start,
            end,
        )
        offset[start:end] = _nearest_signed_distance(
            offset_positions + start,
            start,
            end,
        )
        posterior[start:end] = _nearest_absolute_distance(
            transition_positions + start,
            start,
            end,
        )
    return {
        "onset_distance": onset,
        "offset_distance": offset,
        "posterior_transition_distance": posterior,
    }


def _build_segments(frame: pd.DataFrame) -> list[Segment]:
    source = frame["source_key"].astype(str).to_numpy()
    condition = frame["condition"].astype(str).to_numpy()
    noise = frame["noise_name"].astype(str).to_numpy()
    if not (source.size == condition.size == noise.size):
        raise ValueError("segment metadata lengths disagree")
    starts = [0]
    for index in range(1, source.size):
        if (
            source[index] != source[index - 1]
            or condition[index] != condition[index - 1]
            or noise[index] != noise[index - 1]
        ):
            starts.append(index)
    ends = starts[1:] + [source.size]
    segments = [
        Segment(
            start=int(start),
            end=int(end),
            source_key=str(source[start]),
            condition=str(condition[start]),
            noise_name=str(noise[start]),
        )
        for start, end in zip(starts, ends)
    ]
    if len(segments) == 0:
        raise ValueError("no test segments were found")
    return segments


def _source_index(source_keys: np.ndarray) -> SourceIndex:
    names, codes = np.unique(
        np.asarray(source_keys, dtype=str),
        return_inverse=True,
    )
    return SourceIndex(
        names=names,
        codes=np.asarray(codes, dtype=np.int64),
        count=int(names.size),
    )


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


def _bootstrap_mean_from_sums(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(sums))
    denominator = float(np.sum(counts))
    estimate = total / denominator if denominator else float("nan")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sums.size, size=sums.size)
        selected_counts = float(np.sum(counts[selected]))
        estimates[index] = (
            float(np.sum(sums[selected])) / selected_counts
            if selected_counts
            else float("nan")
        )
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_difference_from_sums(
    left_sums: np.ndarray,
    left_counts: np.ndarray,
    right_sums: np.ndarray,
    right_counts: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    left_denominator = float(np.sum(left_counts))
    right_denominator = float(np.sum(right_counts))
    estimate = (
        float(np.sum(left_sums)) / left_denominator
        - float(np.sum(right_sums)) / right_denominator
        if left_denominator and right_denominator
        else float("nan")
    )
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, left_sums.size, size=left_sums.size)
        left_count = float(np.sum(left_counts[selected]))
        right_count = float(np.sum(right_counts[selected]))
        estimates[index] = (
            float(np.sum(left_sums[selected])) / left_count
            - float(np.sum(right_sums[selected])) / right_count
            if left_count and right_count
            else float("nan")
        )
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_mean_replicates(
    replicate_values: np.ndarray,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    replicate_values = np.asarray(replicate_values, dtype=np.float64)
    replicate_sums = []
    replicate_counts = []
    for values in replicate_values:
        sums, counts = _source_sums(values, None, sources)
        replicate_sums.append(sums)
        replicate_counts.append(counts)
    sums = np.asarray(replicate_sums, dtype=np.float64)
    counts = np.asarray(replicate_counts, dtype=np.float64)
    estimate = float(np.mean(np.sum(sums, axis=1) / np.sum(counts, axis=1)))
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sources.count, size=sources.count)
        selected_counts = np.sum(counts[:, selected], axis=1)
        selected_sums = np.sum(sums[:, selected], axis=1)
        estimates[index] = float(
            np.mean(
                np.divide(
                    selected_sums,
                    selected_counts,
                    out=np.full_like(selected_sums, np.nan),
                    where=selected_counts > 0,
                )
            )
        )
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _bootstrap_difference_replicates(
    left_values: np.ndarray,
    right_values: np.ndarray,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    left_values = np.asarray(left_values, dtype=np.float64)
    right_values = np.asarray(right_values, dtype=np.float64)
    left_sums = []
    left_counts = []
    right_sums = []
    right_counts = []
    for values in left_values:
        sums, counts = _source_sums(values, None, sources)
        left_sums.append(sums)
        left_counts.append(counts)
    for values in right_values:
        sums, counts = _source_sums(values, None, sources)
        right_sums.append(sums)
        right_counts.append(counts)
    left_sums = np.asarray(left_sums, dtype=np.float64)
    left_counts = np.asarray(left_counts, dtype=np.float64)
    right_sums = np.asarray(right_sums, dtype=np.float64)
    right_counts = np.asarray(right_counts, dtype=np.float64)
    estimate = float(
        np.mean(
            np.sum(left_sums, axis=1) / np.sum(left_counts, axis=1)
            - np.sum(right_sums, axis=1) / np.sum(right_counts, axis=1)
        )
    )
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sources.count, size=sources.count)
        selected_left_counts = np.sum(left_counts[:, selected], axis=1)
        selected_right_counts = np.sum(right_counts[:, selected], axis=1)
        selected_left_sums = np.sum(left_sums[:, selected], axis=1)
        selected_right_sums = np.sum(right_sums[:, selected], axis=1)
        left_means = np.divide(
            selected_left_sums,
            selected_left_counts,
            out=np.full_like(selected_left_sums, np.nan),
            where=selected_left_counts > 0,
        )
        right_means = np.divide(
            selected_right_sums,
            selected_right_counts,
            out=np.full_like(selected_right_sums, np.nan),
            where=selected_right_counts > 0,
        )
        estimates[index] = float(np.nanmean(left_means - right_means))
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _summary_metrics(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    *,
    threshold: float = DECISION_THRESHOLD,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    full_scores = np.asarray(full_scores, dtype=np.float64)
    if valid is None:
        valid = np.ones(labels.size, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return {
            "frames": 0.0,
            "P_SS": float("nan"),
            "P_R": float("nan"),
            "P_I": float("nan"),
            "P_H": float("nan"),
            "NetRefinability": float("nan"),
            "mean_v_log": float("nan"),
            "mean_v_brier": float("nan"),
            "P_v_log_positive": float("nan"),
            "P_v_brier_positive": float("nan"),
            "short_error": float("nan"),
            "full_error": float("nan"),
            "short_f1": float("nan"),
            "full_f1": float("nan"),
        }
    selected_labels = labels[valid]
    selected_short = short_scores[valid]
    selected_full = full_scores[valid]
    ss, r, i, h = _taxonomy_at(
        selected_labels,
        selected_short,
        selected_full,
        threshold,
    )
    v_log = _logloss_frame(selected_labels, selected_short) - _logloss_frame(
        selected_labels,
        selected_full,
    )
    v_brier = _brier_frame(selected_labels, selected_short) - _brier_frame(
        selected_labels,
        selected_full,
    )
    short_correct = _correct_at(selected_labels, selected_short, threshold)
    full_correct = _correct_at(selected_labels, selected_full, threshold)
    return {
        "frames": float(selected_labels.size),
        "P_SS": float(np.mean(ss)),
        "P_R": float(np.mean(r)),
        "P_I": float(np.mean(i)),
        "P_H": float(np.mean(h)),
        "NetRefinability": float(np.mean(r) - np.mean(h)),
        "mean_v_log": float(np.mean(v_log)),
        "mean_v_brier": float(np.mean(v_brier)),
        "P_v_log_positive": float(np.mean(v_log > 0.0)),
        "P_v_brier_positive": float(np.mean(v_brier > 0.0)),
        "short_error": float(1.0 - np.mean(short_correct)),
        "full_error": float(1.0 - np.mean(full_correct)),
        "short_f1": float(_f1(selected_labels, selected_short, threshold)),
        "full_f1": float(_f1(selected_labels, selected_full, threshold)),
    }


def _proper_scoring_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    sources: SourceIndex,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    full_scores = np.asarray(full_scores, dtype=np.float64)
    v_log = _logloss_frame(labels, short_scores) - _logloss_frame(
        labels,
        full_scores,
    )
    v_brier = _brier_frame(labels, short_scores) - _brier_frame(
        labels,
        full_scores,
    )
    rows: list[dict[str, Any]] = []
    for name, values in (("v_log", v_log), ("v_brier", v_brier)):
        sums, counts = _source_sums(values, None, sources)
        estimate, low, high = _bootstrap_mean_from_sums(
            sums,
            counts,
            seed=BOOTSTRAP_SEED + _stable_hash_int("proper", name),
        )
        quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append(
            {
                "metric": name,
                "frames": int(values.size),
                "mean": float(estimate),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_share": float(np.mean(values > 0.0)),
                "negative_share": float(np.mean(values < 0.0)),
                "zero_share": float(np.mean(values == 0.0)),
                "q01": float(quantiles[0]),
                "q05": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "q50": float(quantiles[3]),
                "q75": float(quantiles[4]),
                "q95": float(quantiles[5]),
                "q99": float(quantiles[6]),
            }
        )
    return rows


def _decision_vs_probabilistic_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    full_scores = np.asarray(full_scores, dtype=np.float64)
    v_log = _logloss_frame(labels, short_scores) - _logloss_frame(
        labels,
        full_scores,
    )
    _, r, _, _ = _taxonomy_at(
        labels,
        short_scores,
        full_scores,
        threshold,
    )
    short_correct = _correct_at(labels, short_scores, threshold)
    full_correct = _correct_at(labels, full_scores, threshold)
    no_flip = short_correct == full_correct
    plus_no_flip = (v_log > 0.0) & no_flip
    positive = v_log > 0.0
    total_positive = float(np.sum(np.maximum(v_log, 0.0)))
    rows: list[dict[str, Any]] = []
    for name, mask in (
        ("R_decision_changing", r),
        ("P_PLUS_NO_FLIP", plus_no_flip),
        ("P_PLUS_ANY", positive),
    ):
        selected = v_log[mask]
        total_vlog = float(np.sum(selected))
        positive_vlog = float(np.sum(np.maximum(selected, 0.0)))
        rows.append(
            {
                "group": name,
                "frames": int(np.count_nonzero(mask)),
                "frame_share": float(np.mean(mask)),
                "mean_v_log": float(np.mean(selected)) if selected.size else float("nan"),
                "total_v_log": total_vlog,
                "total_positive_v_log": positive_vlog,
                "positive_value_contribution_share": (
                    positive_vlog / total_positive
                    if total_positive > 0.0
                    else float("nan")
                ),
            }
        )
    return rows


def _value_concentration_rows(v_log: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(v_log, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)
    total_positive = float(np.sum(positive))
    total_negative = float(np.sum(negative))
    rows: list[dict[str, Any]] = []
    for fraction in (0.01, 0.02, 0.05, 0.10, 0.20):
        top_count = max(1, int(math.ceil(fraction * values.size)))
        top_positive = float(np.sum(np.sort(positive)[-top_count:]))
        top_negative = float(np.sum(np.sort(negative)[-top_count:]))
        rows.append(
            {
                "top_fraction": fraction,
                "top_frames": top_count,
                "positive_value_share": (
                    top_positive / total_positive
                    if total_positive > 0.0
                    else float("nan")
                ),
                "negative_value_share": (
                    top_negative / total_negative
                    if total_negative > 0.0
                    else float("nan")
                ),
                "total_positive_v_log": total_positive,
                "total_negative_v_log_abs": total_negative,
            }
        )
    return rows


def _event_bin_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    absolute = np.abs(np.asarray(distance, dtype=np.float64))
    present = absolute < MISSING_DISTANCE / 2.0
    return {
        "0": present & (absolute == 0.0),
        "1-2": present & (absolute >= 1.0) & (absolute <= 2.0),
        "3-5": present & (absolute >= 3.0) & (absolute <= 5.0),
        "6-10": present & (absolute >= 6.0) & (absolute <= 10.0),
        "11-25": present & (absolute >= 11.0) & (absolute <= 25.0),
        "26+": present & (absolute >= 26.0),
    }


def _event_contrast_masks(distance: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    bins = _event_bin_masks(distance)
    exact = bins["0"]
    near = bins["0"] | bins["1-2"]
    beyond_one = bins["1-2"] | bins["3-5"] | bins["6-10"] | bins["11-25"] | bins["26+"]
    beyond_three = bins["3-5"] | bins["6-10"] | bins["11-25"] | bins["26+"]
    return {
        "exact_0_minus_beyond_1_plus": (exact, beyond_one),
        "near_0_2_minus_far_3_plus": (near, beyond_three),
        "beyond_1_plus_mean": (beyond_one, None),
        "beyond_3_plus_mean": (beyond_three, None),
    }


def _event_profile_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    distances: Mapping[str, np.ndarray],
    *,
    scenario: str,
    valid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if valid is None:
        valid = np.ones(np.asarray(labels).size, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    rows: list[dict[str, Any]] = []
    for dimension in EVENT_DIMENSIONS:
        masks = _event_bin_masks(distances[dimension])
        for event_bin in EVENT_BINS:
            selected = valid & masks[event_bin]
            metrics = _summary_metrics(
                labels,
                short_scores,
                full_scores,
                valid=selected,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "dimension": dimension,
                    "event_bin": event_bin,
                    **metrics,
                }
            )
    return rows


def _event_contrast_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    distances: Mapping[str, np.ndarray],
    sources: SourceIndex,
    *,
    scenario: str,
    valid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if valid is None:
        valid = np.ones(np.asarray(labels).size, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    v_log = _logloss_frame(labels, short_scores) - _logloss_frame(
        labels,
        full_scores,
    )
    rows: list[dict[str, Any]] = []
    for dimension in EVENT_DIMENSIONS:
        for name, (left, right) in _event_contrast_masks(
            distances[dimension]
        ).items():
            left_mask = valid & left
            if right is None:
                sums, counts = _source_sums(v_log, left_mask, sources)
                estimate, low, high = _bootstrap_mean_from_sums(
                    sums,
                    counts,
                    seed=BOOTSTRAP_SEED
                    + _stable_hash_int("event", scenario, dimension, name),
                )
            else:
                right_mask = valid & right
                left_sums, left_counts = _source_sums(
                    v_log,
                    left_mask,
                    sources,
                )
                right_sums, right_counts = _source_sums(
                    v_log,
                    right_mask,
                    sources,
                )
                estimate, low, high = _bootstrap_difference_from_sums(
                    left_sums,
                    left_counts,
                    right_sums,
                    right_counts,
                    seed=BOOTSTRAP_SEED
                    + _stable_hash_int("event", scenario, dimension, name),
                )
            rows.append(
                {
                    "scenario": scenario,
                    "dimension": dimension,
                    "contrast": name,
                    "estimate": float(estimate),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "left_frames": int(np.count_nonzero(left_mask)),
                    "right_frames": (
                        0
                        if right is None
                        else int(np.count_nonzero(valid & right))
                    ),
                }
            )
    return rows


@dataclass(frozen=True)
class E3Population:
    frame: pd.DataFrame
    labels: np.ndarray
    short_scores: np.ndarray
    full_scores: np.ndarray
    global_index: np.ndarray
    source_keys: np.ndarray
    speaker_ids: np.ndarray
    noise_names: np.ndarray
    conditions: np.ndarray
    seen: np.ndarray
    segments: tuple[Segment, ...]
    sources: SourceIndex


def _load_rf384_reference() -> dict[str, np.ndarray]:
    if not RF384_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"RF384 reference is missing: {RF384_REFERENCE_PATH}"
        )
    with np.load(RF384_REFERENCE_PATH, allow_pickle=False) as payload:
        required = {
            "labels",
            "short_scores",
            "full_adaptive_scores",
            "embedded_short_scores",
            "test_mask",
            "source_key",
            "condition",
            "noise_name",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(
                "RF384 reference is missing fields: " + ", ".join(missing)
            )
        return {
            name: np.asarray(payload[name]).copy()
            for name in required
        }


def _load_population() -> E3Population:
    if not E1_FRAME_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"E1 frame results are missing: {E1_FRAME_RESULTS_PATH}"
        )
    columns = (
        "global_index",
        "source_key",
        "speaker_id",
        "noise_name",
        "condition",
        "seen",
        "label",
        "short_score",
        "embedded_short_score",
        "full_score",
    )
    frame = pd.read_parquet(E1_FRAME_RESULTS_PATH, columns=list(columns))
    if len(frame) != EXPECTED_TEST_FRAMES:
        raise ValueError(
            f"E1 frame population has {len(frame)} rows, "
            f"expected {EXPECTED_TEST_FRAMES}"
        )
    global_index = frame["global_index"].to_numpy(dtype=np.int64)
    if np.unique(global_index).size != global_index.size:
        raise ValueError("E1 global_index contains duplicates")
    reference = _load_rf384_reference()
    size = int(reference["labels"].size)
    if np.any((global_index < 0) | (global_index >= size)):
        raise ValueError("E1 global_index is outside the RF384 reference")
    test_mask = np.asarray(reference["test_mask"], dtype=bool)
    if not np.all(test_mask[global_index]):
        raise ValueError("E1 frame population contains non-test frames")

    labels = frame["label"].to_numpy(dtype=np.int64)
    reference_labels = np.asarray(
        reference["labels"][global_index],
        dtype=np.int64,
    )
    if not np.array_equal(labels, reference_labels):
        raise ValueError("E1 labels do not match the RF384 reference")
    short_scores = np.asarray(
        reference["short_scores"][global_index],
        dtype=np.float64,
    )
    full_scores = np.asarray(
        reference["full_adaptive_scores"][global_index],
        dtype=np.float64,
    )
    for name, values in (
        ("short_scores", short_scores),
        ("full_scores", full_scores),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} lies outside [0, 1]")

    source_keys = frame["source_key"].astype(str).to_numpy()
    speaker_ids = frame["speaker_id"].astype(str).to_numpy()
    noise_names = frame["noise_name"].astype(str).to_numpy()
    conditions = frame["condition"].astype(str).to_numpy()
    seen = frame["seen"].to_numpy(dtype=bool)
    source_count = int(np.unique(source_keys).size)
    speaker_count = int(np.unique(speaker_ids).size)
    if source_count != EXPECTED_TEST_SOURCES:
        raise ValueError(
            f"population has {source_count} sources, "
            f"expected {EXPECTED_TEST_SOURCES}"
        )
    if speaker_count != EXPECTED_TEST_SPEAKERS:
        raise ValueError(
            f"population has {speaker_count} speakers, "
            f"expected {EXPECTED_TEST_SPEAKERS}"
        )
    segments = tuple(_build_segments(frame))
    return E3Population(
        frame=frame,
        labels=labels,
        short_scores=short_scores,
        full_scores=full_scores,
        global_index=global_index,
        source_keys=source_keys,
        speaker_ids=speaker_ids,
        noise_names=noise_names,
        conditions=conditions,
        seen=seen,
        segments=segments,
        sources=_source_index(source_keys),
    )


def _baseline_reproduction(population: E3Population) -> dict[str, Any]:
    reference = _load_rf384_reference()
    indices = population.global_index
    frame = population.frame
    full_error = np.abs(
        frame["full_score"].to_numpy(dtype=np.float64)
        - np.asarray(
            reference["full_adaptive_scores"][indices],
            dtype=np.float64,
        )
    )
    embedded_short_error = np.abs(
        frame["embedded_short_score"].to_numpy(dtype=np.float64)
        - np.asarray(
            reference["embedded_short_scores"][indices],
            dtype=np.float64,
        )
    )
    short_error = np.abs(
        frame["short_score"].to_numpy(dtype=np.float64)
        - np.asarray(reference["short_scores"][indices], dtype=np.float64)
    )
    metadata_match = {
        "source_key": bool(
            np.array_equal(
                population.source_keys,
                np.asarray(reference["source_key"][indices], dtype=str),
            )
        ),
        "condition": bool(
            np.array_equal(
                population.conditions,
                np.asarray(reference["condition"][indices], dtype=str),
            )
        ),
        "noise_name": bool(
            np.array_equal(
                population.noise_names,
                np.asarray(reference["noise_name"][indices], dtype=str),
            )
        ),
    }
    max_full_error = float(np.max(full_error)) if full_error.size else 0.0
    max_embedded_short_error = (
        float(np.max(embedded_short_error))
        if embedded_short_error.size
        else 0.0
    )
    max_short_error = (
        float(np.max(short_error)) if short_error.size else 0.0
    )
    passed = bool(
        max_full_error <= BASELINE_TOLERANCE
        and max_embedded_short_error <= BASELINE_TOLERANCE
        and all(metadata_match.values())
    )
    return {
        "BASELINE_REPRODUCED": passed,
        "tolerance": BASELINE_TOLERANCE,
        "max_abs_full_error": max_full_error,
        "max_abs_embedded_short_error": max_embedded_short_error,
        "max_abs_short_error": max_short_error,
        "metadata_match": metadata_match,
        "population": {
            "test_frames": int(population.labels.size),
            "test_sources": int(population.sources.count),
            "test_speakers": int(np.unique(population.speaker_ids).size),
        },
        "inputs": {
            "e1_frame_results": _file_record(E1_FRAME_RESULTS_PATH),
            "rf384_reference": _file_record(RF384_REFERENCE_PATH),
        },
        "status": (
            "PASSED"
            if passed
            else "FAILED"
        ),
    }


def _threshold_rows(population: E3Population) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        metrics = _summary_metrics(
            population.labels,
            population.short_scores,
            population.full_scores,
            threshold=threshold,
        )
        p_r = float(metrics["P_R"])
        p_i = float(metrics["P_I"])
        rows.append(
            {
                "threshold": float(threshold),
                "deployed_threshold": bool(
                    abs(float(threshold) - DECISION_THRESHOLD) < 1e-12
                ),
                **metrics,
                "I_over_R": (
                    p_i / p_r
                    if p_r > 0.0
                    else float("nan")
                ),
            }
        )
    return rows


def _any_boundary_distance(
    distances: Mapping[str, np.ndarray],
) -> np.ndarray:
    onset = np.abs(
        np.asarray(distances["onset_distance"], dtype=np.float64)
    )
    offset = np.abs(
        np.asarray(distances["offset_distance"], dtype=np.float64)
    )
    return np.minimum(onset, offset)


def _boundary_valid(
    distances: Mapping[str, np.ndarray],
    width_ms: float,
) -> np.ndarray:
    width_frames = float(width_ms) / FRAME_DURATION_MS
    return _any_boundary_distance(distances) > width_frames


def _boundary_exclusion_rows(
    population: E3Population,
    distances: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for width_ms in EXCLUSION_WIDTHS_MS:
        valid = _boundary_valid(distances, float(width_ms))
        metrics = _summary_metrics(
            population.labels,
            population.short_scores,
            population.full_scores,
            valid=valid,
        )
        rows.append(
            {
                "boundary_exclusion_ms": float(width_ms),
                "frames_retained": int(np.count_nonzero(valid)),
                "frames_removed": int(valid.size - np.count_nonzero(valid)),
                **metrics,
            }
        )
        v_log = _logloss_frame(
            population.labels,
            population.short_scores,
        ) - _logloss_frame(
            population.labels,
            population.full_scores,
        )
        _, r, i, h = _taxonomy_at(
            population.labels,
            population.short_scores,
            population.full_scores,
            DECISION_THRESHOLD,
        )
        net_values = r.astype(np.float64) - h.astype(np.float64)
        for statistic, values in (
            ("mean_v_log", v_log),
            ("net_refinability", net_values),
            ("P_I_minus_P_R", i.astype(np.float64) - r.astype(np.float64)),
        ):
            sums, counts = _source_sums(values, valid, population.sources)
            estimate, low, high = _bootstrap_mean_from_sums(
                sums,
                counts,
                seed=BOOTSTRAP_SEED
                + _stable_hash_int("boundary", width_ms, statistic),
            )
            bootstrap_rows.append(
                {
                    "boundary_exclusion_ms": float(width_ms),
                    "statistic": statistic,
                    "estimate": float(estimate),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "frames": int(np.count_nonzero(valid)),
                }
            )
    return rows, bootstrap_rows


def _proper_scoring_vs_decision(
    population: E3Population,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proper_rows = _proper_scoring_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        population.sources,
    )
    decision_rows = _decision_vs_probabilistic_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
    )
    v_log = _logloss_frame(
        population.labels,
        population.short_scores,
    ) - _logloss_frame(
        population.labels,
        population.full_scores,
    )
    concentration_rows = _value_concentration_rows(v_log)
    return proper_rows, decision_rows, concentration_rows


def _jitter_labels(
    labels: np.ndarray,
    segments: Sequence[Segment],
    *,
    width_frames: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if width_frames < 0:
        raise ValueError("jitter width must be non-negative")
    result = np.zeros(labels.size, dtype=np.int64)
    for segment_index, segment in enumerate(segments):
        start, end = int(segment.start), int(segment.end)
        if not 0 <= start < end <= labels.size:
            raise ValueError("invalid segment bounds during jitter")
        local_labels = labels[start:end]
        for run_start, run_end in _find_runs(local_labels == 1):
            rng = np.random.default_rng(
                _stable_hash_int(
                    "jitter",
                    int(seed),
                    int(width_frames),
                    int(segment_index),
                    int(run_start),
                    int(run_end),
                )
            )
            delta_start = int(
                rng.integers(-width_frames, width_frames + 1)
            )
            delta_end = int(
                rng.integers(-width_frames, width_frames + 1)
            )
            new_start = int(run_start + delta_start)
            new_end = int(run_end + delta_end)
            new_start = max(0, min(new_start, int(local_labels.size) - 1))
            new_end = max(1, min(new_end, int(local_labels.size)))
            if new_end <= new_start:
                if new_start + 1 <= int(local_labels.size):
                    new_end = new_start + 1
                else:
                    new_start = new_end - 1
            result[start + new_start : start + new_end] = 1
    if not np.all(np.isin(result, (0, 1))):
        raise ValueError("jitter generated non-binary labels")
    return result


def _bootstrap_event_contrast_replicates(
    v_log_replicates: np.ndarray,
    left_masks: np.ndarray,
    right_masks: np.ndarray | None,
    sources: SourceIndex,
    *,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(v_log_replicates, dtype=np.float64)
    left = np.asarray(left_masks, dtype=bool)
    if values.shape != left.shape:
        raise ValueError("v_log replicates and masks must have equal shape")
    if right_masks is not None:
        right = np.asarray(right_masks, dtype=bool)
        if right.shape != values.shape:
            raise ValueError("left and right masks must have equal shape")
    else:
        right = None

    left_sums = []
    left_counts = []
    right_sums = []
    right_counts = []
    for index in range(values.shape[0]):
        sums, counts = _source_sums(
            values[index],
            left[index],
            sources,
        )
        left_sums.append(sums)
        left_counts.append(counts)
        if right is not None:
            sums, counts = _source_sums(
                values[index],
                right[index],
                sources,
            )
            right_sums.append(sums)
            right_counts.append(counts)
    left_sums = np.asarray(left_sums, dtype=np.float64)
    left_counts = np.asarray(left_counts, dtype=np.float64)
    if right is None:
        right_sums = None
        right_counts = None
    else:
        right_sums = np.asarray(right_sums, dtype=np.float64)
        right_counts = np.asarray(right_counts, dtype=np.float64)

    def replicate_estimate(
        selected: np.ndarray,
    ) -> float:
        left_means = []
        for index in range(left_sums.shape[0]):
            count = float(np.sum(left_counts[index, selected]))
            if count:
                left_means.append(
                    float(np.sum(left_sums[index, selected])) / count
                )
        if not left_means:
            return float("nan")
        if right_sums is None or right_counts is None:
            return float(np.mean(left_means))
        right_means = []
        for index in range(right_sums.shape[0]):
            count = float(np.sum(right_counts[index, selected]))
            if count:
                right_means.append(
                    float(np.sum(right_sums[index, selected])) / count
                )
        if len(right_means) != len(left_means):
            return float("nan")
        return float(np.mean(np.asarray(left_means) - np.asarray(right_means)))

    all_sources = np.arange(sources.count, dtype=np.int64)
    estimate = replicate_estimate(all_sources)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, sources.count, size=sources.count)
        estimates[index] = replicate_estimate(selected)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return estimate, float(low), float(high)


def _event_contrast_replicates(
    v_log_replicates: np.ndarray,
    distance_replicates: Sequence[Mapping[str, np.ndarray]],
    sources: SourceIndex,
    *,
    scenario: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in EVENT_DIMENSIONS:
        mask_sets = [
            _event_contrast_masks(distances[dimension])
            for distances in distance_replicates
        ]
        for name in _event_contrast_masks(
            distance_replicates[0][dimension]
        ):
            left_masks = np.stack(
                [masks[name][0] for masks in mask_sets]
            )
            right_values = [masks[name][1] for masks in mask_sets]
            if right_values[0] is None:
                right_masks = None
            else:
                right_masks = np.stack(right_values)
            estimate, low, high = _bootstrap_event_contrast_replicates(
                v_log_replicates,
                left_masks,
                right_masks,
                sources,
                seed=BOOTSTRAP_SEED
                + _stable_hash_int(
                    "jitter_event",
                    scenario,
                    dimension,
                    name,
                ),
            )
            rows.append(
                {
                    "scenario": scenario,
                    "dimension": dimension,
                    "contrast": name,
                    "estimate": float(estimate),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "left_frames_mean": float(
                        np.mean(np.sum(left_masks, axis=1))
                    ),
                    "right_frames_mean": (
                        float("nan")
                        if right_masks is None
                        else float(np.mean(np.sum(right_masks, axis=1)))
                    ),
                }
            )
    return rows


def _aggregate_event_profile_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    group_columns = ["scenario", "dimension", "event_bin"]
    numeric_columns = [
        column
        for column in frame.columns
        if column not in group_columns
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = (
        frame.groupby(group_columns, dropna=False, as_index=False)[
            numeric_columns
        ]
        .mean()
        .to_dict(orient="records")
    )
    return [
        {key: value for key, value in row.items()}
        for row in grouped
    ]


def _segment_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    segments: Sequence[Segment],
    *,
    scenario: str,
) -> list[dict[str, Any]]:
    utterances = [
        Utterance(
            start=int(segment.start),
            end=int(segment.end),
            source_key=str(segment.source_key),
            condition=str(segment.condition),
            noise_name=str(segment.noise_name),
        )
        for segment in segments
    ]
    rows: list[dict[str, Any]] = []
    for method, scores in (
        ("short", short_scores),
        ("full", full_scores),
    ):
        metrics = evaluate_segment_scores(
            labels,
            scores,
            utterances,
            SEGMENT_PROTOCOL,
        )
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                **metrics,
            }
        )
    return rows


def _group_robustness_rows(
    population: E3Population,
    distances: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, np.ndarray]] = []
    for value in sorted(set(population.conditions)):
        groups.append(
            ("condition", str(value), population.conditions == value)
        )
    seen_group = np.where(population.seen, "seen", "unseen")
    for value in sorted(set(seen_group)):
        groups.append(
            ("seen_unseen", str(value), seen_group == value)
        )
    for value in sorted(set(population.noise_names)):
        groups.append(
            ("noise_name", str(value), population.noise_names == value)
        )

    v_log = _logloss_frame(
        population.labels,
        population.short_scores,
    ) - _logloss_frame(
        population.labels,
        population.full_scores,
    )
    _, r, i, h = _taxonomy_at(
        population.labels,
        population.short_scores,
        population.full_scores,
        DECISION_THRESHOLD,
    )
    net_values = r.astype(np.float64) - h.astype(np.float64)
    rows: list[dict[str, Any]] = []
    for width_ms in EXCLUSION_WIDTHS_MS:
        boundary_valid = _boundary_valid(distances, float(width_ms))
        for group_type, group_name, group_mask in groups:
            selected = boundary_valid & group_mask
            if not np.any(selected):
                continue
            metrics = _summary_metrics(
                population.labels,
                population.short_scores,
                population.full_scores,
                valid=selected,
            )
            v_sums, v_counts = _source_sums(
                v_log,
                selected,
                population.sources,
            )
            v_estimate, v_low, v_high = _bootstrap_mean_from_sums(
                v_sums,
                v_counts,
                seed=BOOTSTRAP_SEED
                + _stable_hash_int(
                    "group",
                    group_type,
                    group_name,
                    width_ms,
                    "v_log",
                ),
            )
            net_sums, net_counts = _source_sums(
                net_values,
                selected,
                population.sources,
            )
            net_estimate, net_low, net_high = _bootstrap_mean_from_sums(
                net_sums,
                net_counts,
                seed=BOOTSTRAP_SEED
                + _stable_hash_int(
                    "group",
                    group_type,
                    group_name,
                    width_ms,
                    "net",
                ),
            )
            rows.append(
                {
                    "boundary_exclusion_ms": float(width_ms),
                    "group_type": group_type,
                    "group": group_name,
                    "frames": int(np.count_nonzero(selected)),
                    **metrics,
                    "mean_v_log_ci95_low": float(v_low),
                    "mean_v_log_ci95_high": float(v_high),
                    "net_refinability_ci95_low": float(net_low),
                    "net_refinability_ci95_high": float(net_high),
                }
            )
    return rows


def _loso_rows(
    population: E3Population,
    distances: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    v_log = _logloss_frame(
        population.labels,
        population.short_scores,
    ) - _logloss_frame(
        population.labels,
        population.full_scores,
    )
    _, r, _, h = _taxonomy_at(
        population.labels,
        population.short_scores,
        population.full_scores,
        DECISION_THRESHOLD,
    )
    statistics: list[
        tuple[str, np.ndarray, np.ndarray | None]
    ] = [
        ("mean_v_log", v_log, None),
        (
            "net_refinability",
            r.astype(np.float64) - h.astype(np.float64),
            None,
        ),
    ]
    for dimension in EVENT_DIMENSIONS:
        for contrast_name, (left, right) in _event_contrast_masks(
            distances[dimension]
        ).items():
            if right is None:
                continue
            left_values = np.where(left, v_log, np.nan)
            right_values = np.where(right, v_log, np.nan)
            statistics.append(
                (
                    f"{dimension}:{contrast_name}",
                    left_values,
                    right_values,
                )
            )
    rows: list[dict[str, Any]] = []
    for name, left_values, right_values in statistics:
        if right_values is None:
            full_estimate = float(
                np.nanmean(left_values)
                if np.any(np.isfinite(left_values))
                else float("nan")
            )
        else:
            full_estimate = float(
                np.nanmean(left_values) - np.nanmean(right_values)
                if np.any(np.isfinite(left_values))
                and np.any(np.isfinite(right_values))
                else float("nan")
            )
        loso: list[float] = []
        for source_index in range(population.sources.count):
            keep = population.sources.codes != source_index
            left_selected = left_values[keep]
            if right_values is None:
                if not np.any(np.isfinite(left_selected)):
                    continue
                loso.append(float(np.nanmean(left_selected)))
            else:
                right_selected = right_values[keep]
                if (
                    not np.any(np.isfinite(left_selected))
                    or not np.any(np.isfinite(right_selected))
                ):
                    continue
                loso.append(
                    float(
                        np.nanmean(left_selected)
                        - np.nanmean(right_selected)
                    )
                )
        loso_array = np.asarray(loso, dtype=np.float64)
        rows.append(
            {
                "statistic": name,
                "full_estimate": full_estimate,
                "loso_count": int(loso_array.size),
                "loso_min": (
                    float(np.min(loso_array))
                    if loso_array.size
                    else float("nan")
                ),
                "loso_max": (
                    float(np.max(loso_array))
                    if loso_array.size
                    else float("nan")
                ),
                "sign_consistent": bool(
                    loso_array.size > 0
                    and (
                        np.all(loso_array > 0.0)
                        or np.all(loso_array < 0.0)
                    )
                ),
            }
        )
    return rows


def _jitter_analysis(
    population: E3Population,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    jitter_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for width_ms in JITTER_WIDTHS_MS:
        width_frames = int(round(width_ms / FRAME_DURATION_MS))
        v_log_replicates: list[np.ndarray] = []
        net_replicates: list[np.ndarray] = []
        seed_metric_rows: list[dict[str, Any]] = []
        seed_contrast_rows: list[list[dict[str, Any]]] = []
        seed_profile_rows: list[list[dict[str, Any]]] = []
        seed_segment_rows: list[list[dict[str, Any]]] = []
        distance_replicates: list[Mapping[str, np.ndarray]] = []
        for seed in JITTER_SEEDS:
            labels_jittered = _jitter_labels(
                population.labels,
                population.segments,
                width_frames=width_frames,
                seed=int(seed),
            )
            distances_jittered = _event_distances(
                labels_jittered,
                population.short_scores,
                population.segments,
            )
            metrics = _summary_metrics(
                labels_jittered,
                population.short_scores,
                population.full_scores,
            )
            seed_metric_rows.append(
                {
                    "jitter_ms": float(width_ms),
                    "jitter_seed": int(seed),
                    "frames": int(population.labels.size),
                    **metrics,
                }
            )
            v_log = _logloss_frame(
                labels_jittered,
                population.short_scores,
            ) - _logloss_frame(
                labels_jittered,
                population.full_scores,
            )
            _, r, i, h = _taxonomy_at(
                labels_jittered,
                population.short_scores,
                population.full_scores,
                DECISION_THRESHOLD,
            )
            v_log_replicates.append(v_log)
            net_replicates.append(
                r.astype(np.float64) - h.astype(np.float64)
            )
            distance_replicates.append(distances_jittered)
            seed_profile_rows.append(
                _event_profile_rows(
                    labels_jittered,
                    population.short_scores,
                    population.full_scores,
                    distances_jittered,
                    scenario=f"jitter_{int(width_ms)}ms",
                )
            )
            seed_contrast_rows.append(
                _event_contrast_rows(
                    labels_jittered,
                    population.short_scores,
                    population.full_scores,
                    distances_jittered,
                    population.sources,
                    scenario=f"jitter_{int(width_ms)}ms",
                )
            )
            seed_segment_rows.append(
                _segment_rows(
                    labels_jittered,
                    population.short_scores,
                    population.full_scores,
                    population.segments,
                    scenario=f"jitter_{int(width_ms)}ms",
                )
            )

        v_log_replicates_array = np.stack(v_log_replicates, axis=0)
        net_replicates_array = np.stack(net_replicates, axis=0)
        mean_v_log, v_low, v_high = _bootstrap_mean_replicates(
            v_log_replicates_array,
            population.sources,
            seed=BOOTSTRAP_SEED
            + _stable_hash_int("jitter", width_ms, "mean_v_log"),
        )
        mean_net, net_low, net_high = _bootstrap_mean_replicates(
            net_replicates_array,
            population.sources,
            seed=BOOTSTRAP_SEED
            + _stable_hash_int("jitter", width_ms, "net_refinability"),
        )
        p_i_minus_p_r_replicates = np.stack(
            [
                (
                    _taxonomy_at(
                        labels_jittered,
                        population.short_scores,
                        population.full_scores,
                        DECISION_THRESHOLD,
                    )[2].astype(np.float64)
                    - _taxonomy_at(
                        labels_jittered,
                        population.short_scores,
                        population.full_scores,
                        DECISION_THRESHOLD,
                    )[1].astype(np.float64)
                )
                for labels_jittered in [
                    _jitter_labels(
                        population.labels,
                        population.segments,
                        width_frames=width_frames,
                        seed=int(seed),
                    )
                    for seed in JITTER_SEEDS
                ]
            ],
            axis=0,
        )
        p_i_minus_p_r, p_low, p_high = _bootstrap_mean_replicates(
            p_i_minus_p_r_replicates,
            population.sources,
            seed=BOOTSTRAP_SEED
            + _stable_hash_int("jitter", width_ms, "P_I_minus_P_R"),
        )
        for statistic, estimate, low, high in (
            ("mean_v_log", mean_v_log, v_low, v_high),
            ("net_refinability", mean_net, net_low, net_high),
            ("P_I_minus_P_R", p_i_minus_p_r, p_low, p_high),
        ):
            bootstrap_rows.append(
                {
                    "jitter_ms": float(width_ms),
                    "statistic": statistic,
                    "estimate": float(estimate),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "jitter_seed_count": len(JITTER_SEEDS),
                }
            )

        metric_frame = pd.DataFrame(seed_metric_rows)
        aggregate_metrics = {
            key: float(value)
            for key, value in metric_frame.select_dtypes(
                include=[np.number]
            ).mean().to_dict().items()
            if key not in {"jitter_seed"}
        }
        positive_seed_count = int(
            np.count_nonzero(
                metric_frame["mean_v_log"].to_numpy(dtype=np.float64) > 0.0
            )
        )
        aggregate_metrics.update(
            {
                "jitter_ms": float(width_ms),
                "jitter_seed": -1,
                "jitter_seed_count": len(JITTER_SEEDS),
                "positive_seed_count": positive_seed_count,
                "mean_v_log_ci95_low": float(v_low),
                "mean_v_log_ci95_high": float(v_high),
                "net_refinability_ci95_low": float(net_low),
                "net_refinability_ci95_high": float(net_high),
            }
        )
        jitter_rows.extend(seed_metric_rows)
        jitter_rows.append(aggregate_metrics)
        profile_rows.extend(
            _aggregate_event_profile_rows(
                [row for rows in seed_profile_rows for row in rows]
            )
        )
        contrast_rows.extend(
            _event_contrast_replicates(
                v_log_replicates_array,
                distance_replicates,
                population.sources,
                scenario=f"jitter_{int(width_ms)}ms",
            )
        )
        segment_frame = pd.DataFrame(
            [row for rows in seed_segment_rows for row in rows]
        )
        if not segment_frame.empty:
            group_columns = ["scenario", "method"]
            numeric_columns = [
                column
                for column in segment_frame.columns
                if column not in group_columns
                and pd.api.types.is_numeric_dtype(segment_frame[column])
            ]
            segment_rows.extend(
                segment_frame.groupby(
                    group_columns,
                    dropna=False,
                    as_index=False,
                )[numeric_columns]
                .mean()
                .to_dict(orient="records")
            )
    return (
        jitter_rows,
        bootstrap_rows,
        profile_rows,
        contrast_rows,
        segment_rows,
    )


def _decision_gate(
    threshold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(threshold_rows)
    d1 = bool(
        rows
        and all(
            float(row["P_R"]) <= D1_SMALL_MINORITY_MAX
            for row in rows
        )
    )
    relation_count = int(
        sum(
            float(row["P_I"]) > float(row["P_R"])
            for row in rows
        )
    )
    net_count = int(
        sum(float(row["NetRefinability"]) > 0.0 for row in rows)
    )
    d2 = relation_count >= 8
    d3 = net_count >= 8
    deployed = [
        row
        for row in rows
        if bool(row.get("deployed_threshold", False))
    ]
    d4 = bool(
        deployed
        and float(deployed[0]["P_I"]) > float(deployed[0]["P_R"])
        and float(deployed[0]["NetRefinability"]) > 0.0
        and relation_count >= 8
        and net_count >= 8
    )
    return {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
        "relation_count": relation_count,
        "net_positive_count": net_count,
        "threshold_count": len(rows),
        "passed": bool(d1 and d2 and d3 and d4),
    }


def _proper_gate(
    proper_rows: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    v_log = next(
        (
            row
            for row in proper_rows
            if str(row.get("metric")) == "v_log"
        ),
        None,
    )
    v_brier = next(
        (
            row
            for row in proper_rows
            if str(row.get("metric")) == "v_brier"
        ),
        None,
    )
    loso = next(
        (
            row
            for row in loso_rows
            if str(row.get("statistic")) == "mean_v_log"
        ),
        None,
    )
    aggregate_positive = bool(
        v_log is not None
        and float(v_log["mean"]) > 0.0
        and float(v_log["ci95_low"]) > 0.0
    )
    brier_agrees = bool(
        v_brier is not None and float(v_brier["mean"]) > 0.0
    )
    loso_stable = bool(
        loso is not None
        and bool(loso.get("sign_consistent", False))
        and float(loso.get("loso_min", float("nan"))) > 0.0
    )
    return {
        "aggregate_positive": aggregate_positive,
        "brier_agrees": brier_agrees,
        "loso_not_single_source": loso_stable,
        "passed": bool(aggregate_positive and brier_agrees and loso_stable),
        "mean_v_log": (
            float(v_log["mean"]) if v_log is not None else float("nan")
        ),
        "mean_v_log_ci95_low": (
            float(v_log["ci95_low"])
            if v_log is not None
            else float("nan")
        ),
        "mean_v_brier": (
            float(v_brier["mean"])
            if v_brier is not None
            else float("nan")
        ),
        "loso_min": (
            float(loso["loso_min"])
            if loso is not None
            else float("nan")
        ),
        "loso_max": (
            float(loso["loso_max"])
            if loso is not None
            else float("nan")
        ),
    }


def _event_signal_at(
    event_contrasts: Sequence[Mapping[str, Any]],
    scenario: str,
    dimensions: Sequence[str] = (
        "onset_distance",
        "posterior_transition_distance",
    ),
) -> bool:
    candidates = [
        row
        for row in event_contrasts
        if str(row.get("scenario")) == scenario
        and str(row.get("dimension")) in dimensions
        and str(row.get("contrast"))
        in {
            "near_0_2_minus_far_3_plus",
            "beyond_1_plus_mean",
        }
    ]
    return any(
        float(row.get("estimate", float("nan"))) > 0.0
        and float(row.get("ci95_low", float("nan"))) > 0.0
        for row in candidates
    )


def _boundary_gate(
    boundary_rows: Sequence[Mapping[str, Any]],
    boundary_bootstrap: Sequence[Mapping[str, Any]],
    event_contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row_30 = next(
        (
            row
            for row in boundary_rows
            if abs(float(row["boundary_exclusion_ms"]) - 30.0) < 1e-12
        ),
        None,
    )
    mean_row = next(
        (
            row
            for row in boundary_bootstrap
            if abs(float(row["boundary_exclusion_ms"]) - 30.0) < 1e-12
            and str(row["statistic"]) == "mean_v_log"
        ),
        None,
    )
    value_positive = bool(
        row_30 is not None and float(row_30["mean_v_log"]) > 0.0
    )
    ci_not_reversed = bool(
        mean_row is not None
        and float(mean_row["ci95_high"]) > 0.0
    )
    relation = bool(
        row_30 is not None
        and float(row_30["P_I"]) > float(row_30["P_R"])
    )
    event_signal = _event_signal_at(
        event_contrasts,
        "exclusion_30ms",
    )
    return {
        "value_positive_at_30ms": value_positive,
        "ci_not_reversed_at_30ms": ci_not_reversed,
        "P_I_gt_P_R_at_30ms": relation,
        "event_signal_beyond_exact_at_30ms": event_signal,
        "passed": bool(
            value_positive and ci_not_reversed and relation and event_signal
        ),
    }


def _jitter_gate(
    jitter_rows: Sequence[Mapping[str, Any]],
    jitter_bootstrap: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    all_widths_passed = True
    for width_ms in JITTER_WIDTHS_MS:
        aggregate = next(
            (
                row
                for row in jitter_rows
                if abs(float(row["jitter_ms"]) - float(width_ms)) < 1e-12
                and int(row.get("jitter_seed", -1)) == -1
            ),
            None,
        )
        seed_rows = [
            row
            for row in jitter_rows
            if abs(float(row["jitter_ms"]) - float(width_ms)) < 1e-12
            and int(row.get("jitter_seed", -1)) >= 0
        ]
        mean_row = next(
            (
                row
                for row in jitter_bootstrap
                if abs(float(row["jitter_ms"]) - float(width_ms)) < 1e-12
                and str(row["statistic"]) == "mean_v_log"
            ),
            None,
        )
        positive_value = bool(
            aggregate is not None
            and float(aggregate["mean_v_log"]) > 0.0
        )
        net_not_reversed = bool(
            aggregate is not None
            and float(aggregate["NetRefinability"]) >= 0.0
        )
        relation = bool(
            aggregate is not None
            and float(aggregate["P_I"]) > float(aggregate["P_R"])
        )
        seed_positive_count = int(
            sum(
                float(row["mean_v_log"]) > 0.0
                for row in seed_rows
            )
        )
        seed_net_positive_count = int(
            sum(
                float(row["NetRefinability"]) > 0.0
                for row in seed_rows
            )
        )
        seed_independent = bool(
            seed_positive_count >= MIN_SEED_POSITIVE_COUNT
            and seed_net_positive_count >= MIN_SEED_POSITIVE_COUNT
        )
        width_passed = bool(
            positive_value
            and net_not_reversed
            and relation
            and seed_independent
            and mean_row is not None
            and float(mean_row["ci95_high"]) > 0.0
        )
        details[str(width_ms)] = {
            "positive_value": positive_value,
            "net_not_reversed": net_not_reversed,
            "P_I_gt_P_R": relation,
            "positive_seed_count": seed_positive_count,
            "positive_net_seed_count": seed_net_positive_count,
            "ci_high": (
                float(mean_row["ci95_high"])
                if mean_row is not None
                else float("nan")
            ),
            "passed": width_passed,
        }
        all_widths_passed = all_widths_passed and width_passed
    through_30ms_passed = bool(
        details.get("10", {}).get("passed", False)
        and details.get("20", {}).get("passed", False)
        and details.get("30", {}).get("passed", False)
    )
    return {
        "through_30ms_passed": through_30ms_passed,
        "all_preregistered_widths_passed": all_widths_passed,
        "details": details,
        # G4 is defined through the preregistered +/-30 ms scale.
        "passed": through_30ms_passed,
    }


def _tiny_gru_populations(
    population: E3Population,
) -> tuple[list[E3Population], list[dict[str, Any]]]:
    populations: list[E3Population] = []
    records: list[dict[str, Any]] = []
    indices = population.global_index
    for path in TINY_GRU_SCORE_PATHS:
        if not path.exists():
            return [], [
                {
                    "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "available": False,
                    "reason": "score file missing",
                }
            ]
        with np.load(path, allow_pickle=False) as payload:
            required = {"H64", "HFULL", "test_mask", "labels"}
            missing = sorted(required - set(payload.files))
            if missing:
                return [], [
                    {
                        "path": str(path.relative_to(REPO_ROOT)).replace(
                            "\\",
                            "/",
                        ),
                        "available": False,
                        "reason": "missing fields: " + ", ".join(missing),
                    }
                ]
            labels = np.asarray(payload["labels"], dtype=np.int64)
            test_mask = np.asarray(payload["test_mask"], dtype=bool)
            short_scores = np.asarray(
                payload["H64"][indices],
                dtype=np.float64,
            )
            full_scores = np.asarray(
                payload["HFULL"][indices],
                dtype=np.float64,
            )
        if not np.array_equal(labels[indices], population.labels):
            return [], [
                {
                    "path": str(path.relative_to(REPO_ROOT)).replace(
                        "\\",
                        "/",
                    ),
                    "available": False,
                    "reason": "labels do not align with E1 population",
                }
            ]
        if not np.all(test_mask[indices]):
            return [], [
                {
                    "path": str(path.relative_to(REPO_ROOT)).replace(
                        "\\",
                        "/",
                    ),
                    "available": False,
                    "reason": "selected frames are outside test_mask",
                }
            ]
        if not np.all(np.isfinite(short_scores)) or not np.all(
            np.isfinite(full_scores)
        ):
            return [], [
                {
                    "path": str(path.relative_to(REPO_ROOT)).replace(
                        "\\",
                        "/",
                    ),
                    "available": False,
                    "reason": "non-finite scores",
                }
            ]
        populations.append(
            E3Population(
                frame=population.frame,
                labels=population.labels,
                short_scores=short_scores,
                full_scores=full_scores,
                global_index=population.global_index,
                source_keys=population.source_keys,
                speaker_ids=population.speaker_ids,
                noise_names=population.noise_names,
                conditions=population.conditions,
                seen=population.seen,
                segments=population.segments,
                sources=population.sources,
            )
        )
        records.append(
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "available": True,
                "frames": int(indices.size),
            }
        )
    return populations, records


def _aggregate_cross_arch_rows(
    rows: Sequence[Mapping[str, Any]],
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in set(group_columns)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = (
        frame.groupby(list(group_columns), dropna=False, as_index=False)[
            numeric_columns
        ]
        .mean()
        .to_dict(orient="records")
    )
    return [
        {
            **{column: row[column] for column in group_columns},
            **{
                column: row[column]
                for column in numeric_columns
            },
            "architecture_seed": -1,
        }
        for row in grouped
    ]


def _cross_architecture_rows(
    population: E3Population,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    tiny_populations, availability = _tiny_gru_populations(population)
    if not tiny_populations:
        return [], availability, False
    threshold_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    boundary_bootstrap_rows: list[dict[str, Any]] = []
    proper_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    event_profile_rows: list[dict[str, Any]] = []
    event_contrast_rows: list[dict[str, Any]] = []
    distance_replicates: list[Mapping[str, np.ndarray]] = []
    v_log_replicates: list[np.ndarray] = []
    for seed, tiny in zip(TINY_GRU_SEED_VALUES, tiny_populations):
        tiny_distances = _event_distances(
            tiny.labels,
            tiny.short_scores,
            tiny.segments,
        )
        for row in _threshold_rows(tiny):
            threshold_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        boundary, boundary_bootstrap = _boundary_exclusion_rows(
            tiny,
            tiny_distances,
        )
        for row in boundary:
            boundary_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        for row in boundary_bootstrap:
            boundary_bootstrap_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        for row in _proper_scoring_rows(
            tiny.labels,
            tiny.short_scores,
            tiny.full_scores,
            tiny.sources,
        ):
            proper_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        for row in _decision_vs_probabilistic_rows(
            tiny.labels,
            tiny.short_scores,
            tiny.full_scores,
        ):
            decision_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        for row in _event_profile_rows(
            tiny.labels,
            tiny.short_scores,
            tiny.full_scores,
            tiny_distances,
            scenario="baseline",
        ):
            event_profile_rows.append(
                {
                    "architecture": "TinyGRU",
                    "architecture_seed": int(seed),
                    **row,
                }
            )
        distance_replicates.append(tiny_distances)
        v_log_replicates.append(
            _logloss_frame(tiny.labels, tiny.short_scores)
            - _logloss_frame(tiny.labels, tiny.full_scores)
        )
    threshold_aggregate = _aggregate_cross_arch_rows(
        threshold_rows,
        ("architecture", "threshold"),
    )
    boundary_aggregate = _aggregate_cross_arch_rows(
        boundary_rows,
        ("architecture", "boundary_exclusion_ms"),
    )
    proper_aggregate = _aggregate_cross_arch_rows(
        proper_rows,
        ("architecture", "metric"),
    )
    decision_aggregate = _aggregate_cross_arch_rows(
        decision_rows,
        ("architecture", "group"),
    )
    event_profile_aggregate = _aggregate_cross_arch_rows(
        event_profile_rows,
        ("architecture", "scenario", "dimension", "event_bin"),
    )
    event_contrast_rows.extend(
        _event_contrast_replicates(
            np.stack(v_log_replicates, axis=0),
            distance_replicates,
            population.sources,
            scenario="baseline",
        )
    )
    for row in event_contrast_rows:
        row["architecture"] = "TinyGRU"
        row["architecture_seed"] = -1

    marble_threshold = _threshold_rows(population)
    marble_boundary, marble_boundary_bootstrap = _boundary_exclusion_rows(
        population,
        _event_distances(
            population.labels,
            population.short_scores,
            population.segments,
        ),
    )
    marble_proper = _proper_scoring_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        population.sources,
    )
    marble_decision = _decision_vs_probabilistic_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
    )
    marble_event_contrasts = _event_contrast_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        _event_distances(
            population.labels,
            population.short_scores,
            population.segments,
        ),
        population.sources,
        scenario="baseline",
    )
    comparison_rows: list[dict[str, Any]] = []
    for architecture, threshold_values, boundary_values, proper_values, decision_values, event_values in (
        (
            "MarbleNet",
            marble_threshold,
            marble_boundary,
            marble_proper,
            marble_decision,
            marble_event_contrasts,
        ),
        (
            "TinyGRU",
            threshold_aggregate,
            boundary_aggregate,
            proper_aggregate,
            decision_aggregate,
            event_contrast_rows,
        ),
    ):
        gate = _decision_gate(threshold_values)
        proper_gate = _proper_gate(proper_values, [])
        event_signal = _event_signal_at(event_values, "baseline")
        comparison_rows.append(
            {
                "architecture": architecture,
                "decision_sparsity_robust": bool(gate["passed"]),
                "proper_scoring_positive": bool(
                    proper_gate["aggregate_positive"]
                ),
                "event_signal_baseline": bool(event_signal),
                "boundary_exclusion_30ms_value_positive": bool(
                    next(
                        (
                            float(row["mean_v_log"]) > 0.0
                            for row in boundary_values
                            if abs(
                                float(row["boundary_exclusion_ms"]) - 30.0
                            )
                            < 1e-12
                        ),
                        False,
                    )
                ),
                "status": (
                    "REPLICATED"
                    if (
                        gate["passed"]
                        and proper_gate["aggregate_positive"]
                        and event_signal
                    )
                    else (
                        "PARTIALLY_REPLICATED"
                        if (
                            gate["passed"]
                            or proper_gate["aggregate_positive"]
                            or event_signal
                        )
                        else "NOT_REPLICATED"
                    )
                ),
            }
        )
    return comparison_rows, availability, True


def _runner_hash() -> str:
    return _sha256_file(Path(__file__).resolve())


def _protocol_payload() -> dict[str, Any]:
    frozen_inputs = {
        "e1_frame_results": E1_FRAME_RESULTS_PATH,
        "e1_final_summary": E1_FINAL_SUMMARY_PATH,
        "e1_protocol": E1_PROTOCOL_PATH,
        "e1_runner": E1_RUNNER_PATH,
        "rf384_reference": RF384_REFERENCE_PATH,
    }
    for index, path in enumerate(TINY_GRU_SCORE_PATHS):
        frozen_inputs[f"tiny_gru_seed{TINY_GRU_SEED_VALUES[index]}"] = path
    return {
        "PROTOCOL_ID": PROTOCOL_ID,
        "PROTOCOL_STATUS": PROTOCOL_STATUS,
        "CREATED_UTC": datetime.now(timezone.utc).isoformat(),
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "ANALYSIS_ONLY": True,
        "PRIMARY_MODEL": "RF384_SHORT_vs_FULL",
        "PRIMARY_ARCHITECTURE": "MarbleNet",
        "SECONDARY_ARCHITECTURE": "TinyGRU",
        "POPULATION": {
            "test_frames": EXPECTED_TEST_FRAMES,
            "test_sources": EXPECTED_TEST_SOURCES,
            "test_speakers": EXPECTED_TEST_SPEAKERS,
            "source": str(E1_FRAME_RESULTS_PATH.relative_to(REPO_ROOT)).replace(
                "\\",
                "/",
            ),
        },
        "THRESHOLDS": list(THRESHOLDS),
        "DECISION_THRESHOLD": DECISION_THRESHOLD,
        "D1_SMALL_MINORITY_MAX": D1_SMALL_MINORITY_MAX,
        "EXCLUSION_WIDTHS_MS": list(EXCLUSION_WIDTHS_MS),
        "JITTER_WIDTHS_MS": list(JITTER_WIDTHS_MS),
        "JITTER_SEEDS": list(JITTER_SEEDS),
        "JITTER_DISTRIBUTION": (
            "independent discrete-uniform onset and offset offsets in "
            "frame units within the corresponding +/- width; clamp to "
            "utterance bounds; force duration >= 1 frame; rasterize union "
            "to avoid contradictory labels"
        ),
        "FRAME_DURATION_MS": FRAME_DURATION_MS,
        "BOOTSTRAP": {
            "method": "source-cluster percentile bootstrap",
            "repeats": BOOTSTRAP_REPEATS,
            "confidence": 0.95,
            "seed": BOOTSTRAP_SEED,
        },
        "VALUE_LAYERS": {
            "A_decision": "SS/R/I/H with NetRefinability=P(R)-P(H)",
            "B_probabilistic": (
                "v_log=LogLoss(Short)-LogLoss(Full); "
                "v_brier=Brier(Short)-Brier(Full)"
            ),
            "C_segment_event": (
                "existing A13 segment protocol and frozen CAR/AE event bins"
            ),
            "logloss_epsilon": LOGLOSS_EPSILON,
        },
        "EVENT_BINS": list(EVENT_BINS),
        "EVENT_DIMENSIONS": list(EVENT_DIMENSIONS),
        "SEGMENT_PROTOCOL": SEGMENT_PROTOCOL.to_dict(),
        "PRIMARY_CONTRASTS": {
            "exact_boundary": "distance == 0 versus distance >= 1",
            "near_boundary": "distance <= 2 versus distance >= 3",
            "stable_reference": (
                "not declared; no favorable stable control is invented"
            ),
        },
        "FROZEN_INPUTS": {
            name: _file_record(path)
            for name, path in frozen_inputs.items()
        },
        "RUNNER": {
            "module": RUNNER_MODULE,
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace(
                "\\",
                "/",
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
            f"E3 protocol is not frozen: {PROTOCOL_PATH}"
        )
    if not PROTOCOL_HASH_PATH.exists() or not RUNNER_HASH_PATH.exists():
        raise FileNotFoundError("E3 freeze hash files are missing")
    protocol_hash = _sha256_file(PROTOCOL_PATH)
    recorded_protocol_hash = PROTOCOL_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    runner_hash = _runner_hash()
    recorded_runner_hash = RUNNER_HASH_PATH.read_text(
        encoding="ascii"
    ).strip().upper()
    if protocol_hash != recorded_protocol_hash:
        raise ValueError("E3 protocol hash does not match its freeze file")
    if runner_hash != recorded_runner_hash:
        raise ValueError("E3 runner hash does not match its freeze file")
    return {
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner_hash,
    }


def _read_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            "E3 baseline reproduction must be run before analysis"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _event_status(
    event_contrasts: Sequence[Mapping[str, Any]],
    dimension: str,
    *,
    onset_style: bool = False,
) -> str:
    def signal(scenario: str) -> bool:
        candidates = [
            row
            for row in event_contrasts
            if str(row.get("scenario")) == scenario
            and str(row.get("dimension")) == dimension
            and str(row.get("contrast"))
            in {
                "exact_0_minus_beyond_1_plus",
                "near_0_2_minus_far_3_plus",
                "beyond_1_plus_mean",
            }
        ]
        return any(
            float(row.get("estimate", float("nan"))) > 0.0
            and float(row.get("ci95_low", float("nan"))) > 0.0
            for row in candidates
        )

    baseline = signal("baseline")
    at_30 = signal("exclusion_30ms")
    at_100 = signal("exclusion_100ms")
    if onset_style:
        if baseline and at_30:
            return "SUPPORTED"
        if baseline:
            return "BOUNDARY_SENSITIVE"
        return "NOT_SUPPORTED"
    if baseline and at_30:
        return "SUPPORTED"
    if baseline:
        return "CONDITIONAL"
    if at_100:
        return "CONDITIONAL"
    return "NOT_SUPPORTED"


def _claim_b(
    proper_gate: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
    p_r_at_05: float,
) -> str:
    if not bool(proper_gate.get("aggregate_positive", False)):
        return "NOT_SUPPORTED"
    no_flip = next(
        (
            float(row["frame_share"])
            for row in decision_rows
            if str(row.get("group")) == "P_PLUS_NO_FLIP"
        ),
        0.0,
    )
    positive = next(
        (
            float(row["frame_share"])
            for row in decision_rows
            if str(row.get("group")) == "P_PLUS_ANY"
        ),
        0.0,
    )
    if no_flip > max(0.02, 2.0 * p_r_at_05) or positive > 0.20:
        return "TOO_STRONG"
    return "SUPPORTED"


def _claim_c(
    p_r_at_05: float,
    p_i_at_05: float,
) -> str:
    if p_r_at_05 > D1_SMALL_MINORITY_MAX:
        return "NOT_SUPPORTED"
    hard = p_r_at_05 + p_i_at_05
    if hard <= 0.0:
        return "CONDITIONAL"
    if p_r_at_05 / hard <= 0.20:
        return "SUPPORTED"
    return "CONDITIONAL"


def _claim_f(
    posterior_status: str,
    boundary_rows: Sequence[Mapping[str, Any]],
) -> str:
    value_100 = next(
        (
            float(row["mean_v_log"])
            for row in boundary_rows
            if abs(float(row["boundary_exclusion_ms"]) - 100.0) < 1e-12
        ),
        float("nan"),
    )
    if posterior_status in {"SUPPORTED", "CONDITIONAL"} and value_100 > 0.0:
        return "NOT_SUPPORTED"
    if posterior_status == "NOT_SUPPORTED":
        return "UNRESOLVED"
    return "NOT_SUPPORTED"


def _select_e3_status(
    gates: Mapping[str, Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    p_r_at_05: float,
) -> str:
    if not bool(gates["G1"]["passed"]):
        return "VALUE_STRUCTURE_CONDITIONAL"
    if not bool(gates["G2"]["passed"]):
        no_flip = next(
            (
                float(row["frame_share"])
                for row in decision_rows
                if str(row.get("group")) == "P_PLUS_NO_FLIP"
            ),
            0.0,
        )
        if no_flip > max(0.02, 2.0 * p_r_at_05):
            return "DECISION_ONLY_SPARSITY"
        return "VALUE_STRUCTURE_CONDITIONAL"
    if bool(gates["G3"]["passed"]) and bool(gates["G4"]["passed"]):
        return "ROBUST_VALUE_STRUCTURE"
    if not bool(gates["G3"]["passed"]) and not bool(gates["G4"]["passed"]):
        return "BOUNDARY_SENSITIVE_VALUE_STRUCTURE"
    return "VALUE_STRUCTURE_CONDITIONAL"


def _write_figures(
    threshold_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    event_profile_rows: Sequence[Mapping[str, Any]],
    jitter_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    cross_arch_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    threshold_frame = pd.DataFrame(threshold_rows)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        threshold_frame["threshold"],
        threshold_frame["P_R"],
        marker="o",
        label="P(R)",
    )
    axis.plot(
        threshold_frame["threshold"],
        threshold_frame["P_I"],
        marker="o",
        label="P(I)",
    )
    axis.plot(
        threshold_frame["threshold"],
        threshold_frame["P_H"],
        marker="o",
        label="P(H)",
    )
    axis.plot(
        threshold_frame["threshold"],
        threshold_frame["NetRefinability"],
        marker="o",
        label="NetRefinability",
    )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Frame probability")
    axis.set_title("Decision taxonomy across thresholds")
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[0]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    boundary_frame = pd.DataFrame(boundary_rows)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        boundary_frame["boundary_exclusion_ms"],
        boundary_frame["mean_v_log"],
        marker="o",
        label="mean v_log",
    )
    axis.plot(
        boundary_frame["boundary_exclusion_ms"],
        boundary_frame["NetRefinability"],
        marker="o",
        label="NetRefinability",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("GT boundary exclusion width (ms)")
    axis.set_ylabel("Value")
    axis.set_title("Boundary exclusion robustness")
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[1]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    event_frame = pd.DataFrame(event_profile_rows)
    baseline_events = event_frame[
        event_frame["scenario"].astype(str) == "baseline"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, dimension in zip(axes, EVENT_DIMENSIONS):
        subset = baseline_events[
            baseline_events["dimension"].astype(str) == dimension
        ]
        subset = subset.set_index("event_bin").reindex(EVENT_BINS)
        axis.bar(
            np.arange(len(EVENT_BINS)),
            subset["mean_v_log"].to_numpy(dtype=np.float64),
        )
        axis.set_xticks(np.arange(len(EVENT_BINS)))
        axis.set_xticklabels(EVENT_BINS, rotation=45, ha="right")
        axis.set_title(dimension)
        axis.set_xlabel("distance bin")
    axes[0].set_ylabel("mean v_log")
    fig.suptitle("Event-bin temporal value")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[2]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    jitter_frame = pd.DataFrame(jitter_rows)
    jitter_frame = jitter_frame[
        jitter_frame["jitter_seed"].astype(int) == -1
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        jitter_frame["jitter_ms"],
        jitter_frame["mean_v_log"],
        marker="o",
        label="mean v_log",
    )
    axis.plot(
        jitter_frame["jitter_ms"],
        jitter_frame["NetRefinability"],
        marker="o",
        label="NetRefinability",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Symmetric label-jitter width (ms)")
    axis.set_ylabel("Value")
    axis.set_title("Label-jitter robustness")
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[3]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    decision_frame = pd.DataFrame(decision_rows)
    fig, axis = plt.subplots(figsize=(8, 5))
    labels = decision_frame["group"].astype(str).tolist()
    values = decision_frame["total_positive_v_log"].to_numpy(
        dtype=np.float64
    )
    axis.bar(np.arange(len(labels)), values)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Total positive v_log")
    axis.set_title("Decision-changing versus broader positive value")
    fig.tight_layout()
    path = OUTPUT_ROOT / FIGURE_FILES[4]
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    if cross_arch_rows:
        cross_frame = pd.DataFrame(cross_arch_rows)
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(
            cross_frame["architecture"],
            cross_frame["status"].astype(str).map(
                {
                    "NOT_REPLICATED": 0,
                    "PARTIALLY_REPLICATED": 1,
                    "REPLICATED": 2,
                }
            ),
        )
        axis.set_yticks([0, 1, 2])
        axis.set_yticklabels(
            ["NOT_REPLICATED", "PARTIALLY_REPLICATED", "REPLICATED"]
        )
        axis.set_title("Cross-architecture pattern status")
        fig.tight_layout()
        path = OUTPUT_ROOT / OPTIONAL_FIGURE_FILE
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)
    return figure_paths


def _write_report(
    summary: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    jitter_rows: Sequence[Mapping[str, Any]],
    proper_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> None:
    deployed = next(
        (
            row
            for row in threshold_rows
            if bool(row.get("deployed_threshold", False))
        ),
        {},
    )
    lines = [
        "# E3 Boundary and Value-Robustness Audit",
        "",
        f"- E3 status: `{summary['E3_STATUS']}`",
        f"- Baseline reproduced: `{summary['BASELINE_REPRODUCED']}`",
        f"- G1 decision taxonomy: `{summary['G1_DECISION_TAXONOMY']['passed']}`",
        f"- G2 proper scoring: `{summary['G2_PROPER_SCORING']['passed']}`",
        f"- G3 boundary exclusion: `{summary['G3_BOUNDARY_EXCLUSION']['passed']}`",
        f"- G4 label jitter: `{summary['G4_LABEL_JITTER']['passed']}`",
        "",
        "## Decision Layer",
        "",
        (
            f"At threshold 0.50: P(R)={float(deployed.get('P_R', float('nan'))):.8f}, "
            f"P(I)={float(deployed.get('P_I', float('nan'))):.8f}, "
            f"P(H)={float(deployed.get('P_H', float('nan'))):.8f}, "
            f"NetRefinability={float(deployed.get('NetRefinability', float('nan'))):.8f}."
        ),
        "",
        "## Probabilistic Layer",
        "",
    ]
    for row in proper_rows:
        lines.append(
            f"- `{row['metric']}` mean={float(row['mean']):.8f}, "
            f"CI95=[{float(row['ci95_low']):.8f}, "
            f"{float(row['ci95_high']):.8f}], "
            f"positive share={float(row['positive_share']):.8f}."
        )
    lines.extend(
        [
            "",
            "## Boundary Exclusion",
            "",
            "| width ms | frames | mean v_log | P(R) | P(I) | NetRefinability |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in boundary_rows:
        lines.append(
            f"| {float(row['boundary_exclusion_ms']):.0f} | "
            f"{int(row['frames_retained'])} | "
            f"{float(row['mean_v_log']):.8f} | "
            f"{float(row['P_R']):.8f} | "
            f"{float(row['P_I']):.8f} | "
            f"{float(row['NetRefinability']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Label Jitter",
            "",
            "| width ms | seed | mean v_log | P(R) | P(I) | NetRefinability |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in jitter_rows:
        if int(row.get("jitter_seed", -1)) != -1:
            continue
        lines.append(
            f"| {float(row['jitter_ms']):.0f} | aggregate | "
            f"{float(row['mean_v_log']):.8f} | "
            f"{float(row['P_R']):.8f} | "
            f"{float(row['P_I']):.8f} | "
            f"{float(row['NetRefinability']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Decision versus Probabilistic Value",
            "",
            "| group | frame share | mean v_log | total positive v_log |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['group']} | {float(row['frame_share']):.8f} | "
            f"{float(row['mean_v_log']):.8f} | "
            f"{float(row['total_positive_v_log']):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Claims",
            "",
            f"- Claim A: `{summary['CLAIM_A_DECISION_SPARSITY']}`",
            f"- Claim B: `{summary['CLAIM_B_TEMPORAL_VALUE_SPARSITY']}`",
            f"- Claim C: `{summary['CLAIM_C_HARD_NOT_REFINABLE']}`",
            f"- Claim D: `{summary['CLAIM_D_ONSET']}`",
            f"- Claim E: `{summary['CLAIM_E_POSTERIOR_TRANSITION']}`",
            f"- Claim F: `{summary['CLAIM_F_BOUNDARY_ARTIFACT']}`",
            "",
            "E2 remains `INCONCLUSIVE_OR_INVALID`; E3 does not rescue or "
            "upgrade E2.",
            "",
            "No training, checkpoint change, OOD access, or next experiment "
            "was performed.",
        ]
    )
    _write_markdown(OUTPUT_ROOT / "e3_final_report.md", lines)
    claim_lines = [
        "# E3 Claim Freeze",
        "",
        f"E3_STATUS={summary['E3_STATUS']}",
        f"CLAIM_A_DECISION_SPARSITY={summary['CLAIM_A_DECISION_SPARSITY']}",
        f"CLAIM_B_TEMPORAL_VALUE_SPARSITY={summary['CLAIM_B_TEMPORAL_VALUE_SPARSITY']}",
        f"CLAIM_C_HARD_NOT_REFINABLE={summary['CLAIM_C_HARD_NOT_REFINABLE']}",
        f"CLAIM_D_ONSET={summary['CLAIM_D_ONSET']}",
        f"CLAIM_E_POSTERIOR_TRANSITION={summary['CLAIM_E_POSTERIOR_TRANSITION']}",
        f"CLAIM_F_BOUNDARY_ARTIFACT={summary['CLAIM_F_BOUNDARY_ARTIFACT']}",
        "",
        "These labels are frozen after the pre-registered E3 analysis.",
    ]
    _write_markdown(OUTPUT_ROOT / "e3_claim_freeze.md", claim_lines)


def _write_execution_manifest(
    *,
    protocol_hash: str,
    runner_hash: str,
    baseline: Mapping[str, Any],
    runtime_seconds: float,
    figure_paths: Sequence[Path],
) -> None:
    output_paths = [
        OUTPUT_ROOT / name
        for name in REQUIRED_OUTPUT_FILES
        if name != EXECUTION_MANIFEST_PATH.name
    ]
    output_paths.extend(figure_paths)
    output_paths.extend(
        path
        for path in (OUTPUT_ROOT / "e3_cross_arch_replication.csv",)
        if path.exists()
    )
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
        "BASELINE_STATUS": baseline.get("status"),
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
            "E3 required artifacts are missing: " + ", ".join(missing)
        )
    figure_paths = [OUTPUT_ROOT / name for name in FIGURE_FILES]
    missing_figures = [path.name for path in figure_paths if not path.exists()]
    if missing_figures:
        raise FileNotFoundError(
            "E3 core figures are missing: " + ", ".join(missing_figures)
        )
    manifest = json.loads(
        EXECUTION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    for record in manifest.get("ARTIFACTS", []):
        path = REPO_ROOT / str(record["path"])
        if not path.exists():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        if _sha256_file(path) != str(record["sha256"]).upper():
            raise ValueError(f"manifest artifact hash mismatch: {path}")
    summary = json.loads(
        (OUTPUT_ROOT / "e3_final_summary.json").read_text(encoding="utf-8")
    )
    return {
        "summary": summary,
        "figures": [path.name for path in figure_paths],
    }


def _build_final_summary(
    *,
    population: E3Population,
    baseline: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    proper_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    concentration_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    boundary_bootstrap: Sequence[Mapping[str, Any]],
    event_contrasts: Sequence[Mapping[str, Any]],
    jitter_rows: Sequence[Mapping[str, Any]],
    jitter_bootstrap: Sequence[Mapping[str, Any]],
    loso_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    cross_arch_rows: Sequence[Mapping[str, Any]],
    cross_arch_available: bool,
) -> dict[str, Any]:
    decision_gate = _decision_gate(threshold_rows)
    proper_gate = _proper_gate(proper_rows, loso_rows)
    boundary_gate = _boundary_gate(
        boundary_rows,
        boundary_bootstrap,
        event_contrasts,
    )
    jitter_gate = _jitter_gate(jitter_rows, jitter_bootstrap)
    gates = {
        "G1": {"passed": bool(decision_gate["passed"]), **decision_gate},
        "G2": {"passed": bool(proper_gate["passed"]), **proper_gate},
        "G3": {"passed": bool(boundary_gate["passed"]), **boundary_gate},
        "G4": {"passed": bool(jitter_gate["passed"]), **jitter_gate},
    }
    deployed = next(
        (
            row
            for row in threshold_rows
            if bool(row.get("deployed_threshold", False))
        ),
        {},
    )
    p_r = float(deployed.get("P_R", float("nan")))
    p_i = float(deployed.get("P_I", float("nan")))
    p_h = float(deployed.get("P_H", float("nan")))
    net = float(deployed.get("NetRefinability", float("nan")))
    proper_log = next(
        (
            row
            for row in proper_rows
            if str(row.get("metric")) == "v_log"
        ),
        {},
    )
    proper_brier = next(
        (
            row
            for row in proper_rows
            if str(row.get("metric")) == "v_brier"
        ),
        {},
    )
    decision_by_group = {
        str(row["group"]): row for row in decision_rows
    }
    positive_share = float(
        proper_log.get("positive_share", float("nan"))
    )
    no_flip_share = float(
        decision_by_group.get("P_PLUS_NO_FLIP", {}).get(
            "frame_share",
            float("nan"),
        )
    )
    onset_status = _event_status(
        event_contrasts,
        "onset_distance",
        onset_style=True,
    )
    posterior_status = _event_status(
        event_contrasts,
        "posterior_transition_distance",
    )
    offset_status = _event_status(
        event_contrasts,
        "offset_distance",
    )
    claim_b = _claim_b(proper_gate, decision_rows, p_r)
    claim_c = _claim_c(p_r, p_i)
    claim_f = _claim_f(posterior_status, boundary_rows)
    e3_status = _select_e3_status(gates, decision_rows, p_r)
    if not bool(baseline.get("BASELINE_REPRODUCED", False)):
        e3_status = "INCONCLUSIVE_OR_INVALID"
    cross_arch_result = "NOT_AVAILABLE"
    if cross_arch_available:
        tiny_row = next(
            (
                row
                for row in cross_arch_rows
                if str(row.get("architecture")) == "TinyGRU"
            ),
            None,
        )
        if tiny_row is not None:
            cross_arch_result = str(
                tiny_row.get("status", "NOT_AVAILABLE")
            )
    summary = {
        "E3_STATUS": e3_status,
        "BASELINE_REPRODUCED": bool(
            baseline.get("BASELINE_REPRODUCED", False)
        ),
        "G1_DECISION_TAXONOMY": gates["G1"],
        "G2_PROPER_SCORING": gates["G2"],
        "G3_BOUNDARY_EXCLUSION": gates["G3"],
        "G4_LABEL_JITTER": gates["G4"],
        "THRESHOLD_RESULTS": list(threshold_rows),
        "BOUNDARY_EXCLUSION_RESULTS": list(boundary_rows),
        "LABEL_JITTER_RESULTS": list(jitter_rows),
        "AGGREGATE_LOGLOSS_VALUE": float(
            proper_log.get("mean", float("nan"))
        ),
        "AGGREGATE_LOGLOSS_CI": [
            float(proper_log.get("ci95_low", float("nan"))),
            float(proper_log.get("ci95_high", float("nan"))),
        ],
        "AGGREGATE_BRIER_VALUE": float(
            proper_brier.get("mean", float("nan"))
        ),
        "P_R_AT_05": p_r,
        "P_I_AT_05": p_i,
        "P_H_AT_05": p_h,
        "NET_REFINABILITY_AT_05": net,
        "POSITIVE_VALUE_FRAME_SHARE": positive_share,
        "POSITIVE_VALUE_NO_FLIP_SHARE": no_flip_share,
        "POSITIVE_VALUE_CONCENTRATION": list(concentration_rows),
        "ONSET_ROBUSTNESS": onset_status,
        "POSTERIOR_TRANSITION_ROBUSTNESS": posterior_status,
        "OFFSET_ROBUSTNESS": offset_status,
        "SEEN_UNSEEN_ROBUSTNESS": [
            row
            for row in condition_rows
            if str(row.get("group_type")) == "seen_unseen"
        ],
        "SOURCE_LOSO_STABILITY": list(loso_rows),
        "CROSS_ARCH_AVAILABLE": bool(cross_arch_available),
        "CROSS_ARCH_RESULT": cross_arch_result,
        "CLAIM_A_DECISION_SPARSITY": (
            "SUPPORTED"
            if decision_gate["passed"]
            else (
                "CONDITIONAL"
                if p_r <= D1_SMALL_MINORITY_MAX
                else "NOT_SUPPORTED"
            )
        ),
        "CLAIM_B_TEMPORAL_VALUE_SPARSITY": claim_b,
        "CLAIM_C_HARD_NOT_REFINABLE": claim_c,
        "CLAIM_D_ONSET": onset_status,
        "CLAIM_E_POSTERIOR_TRANSITION": posterior_status,
        "CLAIM_F_BOUNDARY_ARTIFACT": claim_f,
        "PROTOCOL_DEVIATIONS": [],
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "E2_STATUS": "INCONCLUSIVE_OR_INVALID",
        "POPULATION": {
            "test_frames": int(population.labels.size),
            "test_sources": int(population.sources.count),
            "test_speakers": int(np.unique(population.speaker_ids).size),
        },
    }
    return summary


def _run_baseline_mode() -> int:
    _validate_frozen_protocol()
    population = _load_population()
    baseline = _baseline_reproduction(population)
    _write_json(BASELINE_PATH, baseline)
    print(json.dumps(baseline, indent=2), flush=True)
    return 0 if baseline["BASELINE_REPRODUCED"] else 2


def _run_formal_analysis() -> int:
    started = time.time()
    freeze = _validate_frozen_protocol()
    baseline = _read_baseline()
    population = _load_population()
    if not bool(baseline.get("BASELINE_REPRODUCED", False)):
        summary = {
            "E3_STATUS": "INCONCLUSIVE_OR_INVALID",
            "BASELINE_REPRODUCED": False,
            "TRAINING_PERFORMED": False,
            "NEW_FINAL_OOD_TOUCHED": False,
            "NEXT_EXPERIMENT_AUTHORIZED": False,
            "PROTOCOL_DEVIATIONS": [],
        }
        _write_json(OUTPUT_ROOT / "e3_final_summary.json", summary)
        _write_markdown(
            OUTPUT_ROOT / "e3_final_report.md",
            [
                "# E3 Boundary and Value-Robustness Audit",
                "",
                "Baseline reproduction failed. E3 is INCONCLUSIVE_OR_INVALID "
                "and the analysis stopped before outcome interpretation.",
            ],
        )
        return 2

    distances = _event_distances(
        population.labels,
        population.short_scores,
        population.segments,
    )
    threshold_rows = _threshold_rows(population)
    proper_rows, decision_rows, concentration_rows = (
        _proper_scoring_vs_decision(population)
    )
    boundary_rows, boundary_bootstrap = _boundary_exclusion_rows(
        population,
        distances,
    )
    event_profiles = _event_profile_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        distances,
        scenario="baseline",
    )
    event_contrasts = _event_contrast_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        distances,
        population.sources,
        scenario="baseline",
    )
    for width_ms in EXCLUSION_WIDTHS_MS:
        valid = _boundary_valid(distances, float(width_ms))
        scenario = f"exclusion_{int(width_ms)}ms"
        event_profiles.extend(
            _event_profile_rows(
                population.labels,
                population.short_scores,
                population.full_scores,
                distances,
                scenario=scenario,
                valid=valid,
            )
        )
        width_contrasts = _event_contrast_rows(
            population.labels,
            population.short_scores,
            population.full_scores,
            distances,
            population.sources,
            scenario=scenario,
            valid=valid,
        )
        event_contrasts.extend(width_contrasts)
        for row in width_contrasts:
            boundary_bootstrap.append(
                {
                    "boundary_exclusion_ms": float(width_ms),
                    "statistic": (
                        f"{row['dimension']}:{row['contrast']}"
                    ),
                    "estimate": float(row["estimate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "frames": int(row["left_frames"]),
                }
            )

    (
        jitter_rows,
        jitter_bootstrap,
        jitter_profiles,
        jitter_contrasts,
        jitter_segment_rows,
    ) = _jitter_analysis(population)
    event_profiles.extend(jitter_profiles)
    event_contrasts.extend(jitter_contrasts)
    segment_rows = _segment_rows(
        population.labels,
        population.short_scores,
        population.full_scores,
        population.segments,
        scenario="baseline",
    )
    segment_rows.extend(jitter_segment_rows)
    condition_rows = _group_robustness_rows(population, distances)
    loso_rows = _loso_rows(population, distances)
    cross_arch_rows, cross_arch_availability, cross_arch_available = (
        _cross_architecture_rows(population)
    )
    summary = _build_final_summary(
        population=population,
        baseline=baseline,
        threshold_rows=threshold_rows,
        proper_rows=proper_rows,
        decision_rows=decision_rows,
        concentration_rows=concentration_rows,
        boundary_rows=boundary_rows,
        boundary_bootstrap=boundary_bootstrap,
        event_contrasts=event_contrasts,
        jitter_rows=jitter_rows,
        jitter_bootstrap=jitter_bootstrap,
        loso_rows=loso_rows,
        condition_rows=condition_rows,
        cross_arch_rows=cross_arch_rows,
        cross_arch_available=cross_arch_available,
    )
    summary["CROSS_ARCH_INPUTS"] = cross_arch_availability
    _write_csv(OUTPUT_ROOT / "e3_threshold_robustness.csv", threshold_rows)
    _write_csv(
        OUTPUT_ROOT / "e3_boundary_exclusion.csv",
        boundary_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e3_boundary_bootstrap.csv",
        boundary_bootstrap,
    )
    _write_csv(OUTPUT_ROOT / "e3_label_jitter.csv", jitter_rows)
    _write_csv(
        OUTPUT_ROOT / "e3_label_jitter_bootstrap.csv",
        jitter_bootstrap,
    )
    _write_csv(OUTPUT_ROOT / "e3_event_profiles.csv", event_profiles)
    _write_csv(OUTPUT_ROOT / "e3_event_contrasts.csv", event_contrasts)
    _write_csv(OUTPUT_ROOT / "e3_proper_scoring_value.csv", proper_rows)
    _write_csv(
        OUTPUT_ROOT / "e3_decision_vs_probabilistic_value.csv",
        decision_rows,
    )
    _write_csv(
        OUTPUT_ROOT / "e3_value_concentration.csv",
        concentration_rows,
    )
    _write_csv(OUTPUT_ROOT / "e3_segment_metrics.csv", segment_rows)
    _write_csv(
        OUTPUT_ROOT / "e3_condition_robustness.csv",
        condition_rows,
    )
    _write_csv(OUTPUT_ROOT / "e3_source_influence.csv", loso_rows)
    if cross_arch_available:
        _write_csv(
            OUTPUT_ROOT / "e3_cross_arch_replication.csv",
            cross_arch_rows,
        )
    _write_json(OUTPUT_ROOT / "e3_final_summary.json", summary)
    _write_report(
        summary,
        threshold_rows,
        boundary_rows,
        jitter_rows,
        proper_rows,
        decision_rows,
    )
    figure_paths = _write_figures(
        threshold_rows,
        boundary_rows,
        event_profiles,
        jitter_rows,
        decision_rows,
        cross_arch_rows,
    )
    _write_execution_manifest(
        protocol_hash=str(freeze["protocol_sha256"]),
        runner_hash=str(freeze["runner_sha256"]),
        baseline=baseline,
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
                "status": result["summary"]["E3_STATUS"],
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
            "Execute the frozen E3 boundary and value-robustness audit."
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
