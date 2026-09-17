# -*- coding: utf-8 -*-
"""Phase A2 confirmation for uncertainty-gated long-context routing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOOTSTRAP_REPEATS = 5_000
DEFAULT_RANDOM_REPEATS = 1_000
DEFAULT_SPEAKER_SPLIT_SEED = 20260917
COUNT_COLUMNS = (
    "frames",
    "selected",
    "correction",
    "harm",
    "short_error",
    "long_error",
    "gated_error",
)


def speaker_from_source_key(source_key: str) -> str:
    """Return the LibriSpeech speaker id from a split/speaker/... key."""
    normalized = str(source_key).strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise ValueError("source_key does not contain a speaker id")
    # evaluate_context stores ``<split_dir>/<source_relative_path>``.  The
    # LibriSpeech speaker is therefore the second component.
    return parts[1] if len(parts) >= 2 else parts[0]


def deterministic_speaker_split(
    speaker_ids: Sequence[str],
    *,
    calibration_fraction: float = 0.5,
    seed: int = DEFAULT_SPEAKER_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Split speakers into disjoint calibration and test cohorts."""
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    unique = np.asarray(sorted(set(str(value) for value in speaker_ids)))
    if unique.size < 2:
        raise ValueError("speaker split requires at least two speakers")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.size)]
    calibration_count = int(
        round(float(calibration_fraction) * float(unique.size))
    )
    calibration_count = min(max(calibration_count, 1), unique.size - 1)
    calibration_speakers = sorted(str(value) for value in shuffled[:calibration_count])
    test_speakers = sorted(str(value) for value in shuffled[calibration_count:])
    calibration_set = set(calibration_speakers)
    speaker_array = np.asarray([str(value) for value in speaker_ids])
    calibration_mask = np.isin(speaker_array, calibration_speakers)
    test_mask = np.isin(speaker_array, test_speakers)
    if np.any(calibration_mask & test_mask):
        raise RuntimeError("calibration and test speaker cohorts overlap")
    if not np.all(calibration_mask | test_mask):
        raise RuntimeError("speaker split did not cover every frame")
    if set(calibration_speakers) & set(test_speakers):
        raise RuntimeError("calibration and test speaker sets overlap")
    if calibration_set != set(calibration_speakers):
        raise RuntimeError("calibration speaker set is inconsistent")
    return calibration_mask, test_mask, calibration_speakers, test_speakers


def choose_confidence_threshold(
    short_scores: np.ndarray,
    *,
    activation_rate: float,
) -> tuple[float, np.ndarray]:
    """Choose a low-confidence threshold using calibration frames only."""
    if not 0.0 < float(activation_rate) <= 1.0:
        raise ValueError("activation_rate must be in (0, 1]")
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("cannot choose a threshold from an empty score array")
    confidence = np.abs(scores - 0.5)
    target = int(math.ceil(float(activation_rate) * scores.size))
    target = min(max(target, 1), scores.size)
    order = np.argsort(confidence, kind="stable")
    selected_indices = order[:target]
    selected = np.zeros(scores.size, dtype=bool)
    selected[selected_indices] = True
    threshold = float(np.max(confidence[selected]))
    return threshold, selected


def count_frames_by_cluster(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    selected: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate the signed-utility confusion counts by speaker cluster."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    long_scores = np.asarray(long_scores, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    cluster_ids = np.asarray(cluster_ids, dtype=str).reshape(-1)
    if not (
        labels.size
        == short_scores.size
        == long_scores.size
        == selected.size
        == cluster_ids.size
    ):
        raise ValueError("frame arrays must have identical lengths")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != labels.size:
            raise ValueError("mask must match the frame arrays")
        labels = labels[mask]
        short_scores = short_scores[mask]
        long_scores = long_scores[mask]
        selected = selected[mask]
        cluster_ids = cluster_ids[mask]
    if labels.size == 0:
        raise ValueError("cannot aggregate an empty frame subset")

    cluster_names, inverse = np.unique(cluster_ids, return_inverse=True)
    short_predictions = short_scores >= 0.5
    long_predictions = long_scores >= 0.5
    short_wrong = short_predictions != labels
    long_wrong = long_predictions != labels
    values = {
        "frames": np.ones(labels.size, dtype=np.int64),
        "selected": selected.astype(np.int64),
        "correction": (short_wrong & ~long_wrong).astype(np.int64),
        "harm": (~short_wrong & long_wrong).astype(np.int64),
        "short_error": short_wrong.astype(np.int64),
        "long_error": long_wrong.astype(np.int64),
        "gated_error": np.where(selected, long_wrong, short_wrong).astype(
            np.int64
        ),
    }
    columns = [
        np.bincount(inverse, weights=values[name], minlength=cluster_names.size)
        for name in COUNT_COLUMNS
    ]
    return cluster_names, np.stack(columns, axis=1).astype(np.int64)


def metrics_from_counts(counts: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(counts, dtype=np.int64).reshape(-1)
    if values.size != len(COUNT_COLUMNS):
        raise ValueError("count vector has an unexpected size")
    row = dict(zip(COUNT_COLUMNS, (int(value) for value in values)))
    frames = row["frames"]
    selected = row["selected"]
    signed_utility = row["correction"] - row["harm"]
    return {
        **row,
        "activation_rate": (
            float(selected / frames) if frames else None
        ),
        "signed_utility": int(signed_utility),
        "net_utility_per_frame": (
            float(signed_utility / frames) if frames else None
        ),
        "net_utility_per_selected": (
            float(signed_utility / selected) if selected else None
        ),
        "correction_rate_selected": (
            float(row["correction"] / selected) if selected else None
        ),
        "harm_rate_selected": (
            float(row["harm"] / selected) if selected else None
        ),
        "short_error_rate": (
            float(row["short_error"] / frames) if frames else None
        ),
        "long_error_rate": (
            float(row["long_error"] / frames) if frames else None
        ),
        "gated_error_rate": (
            float(row["gated_error"] / frames) if frames else None
        ),
    }


def _aggregate_run_counts(
    run_counts: Sequence[np.ndarray],
    cluster_indices: np.ndarray,
) -> dict[str, float | int | None]:
    run_metrics = [
        metrics_from_counts(counts[cluster_indices].sum(axis=0))
        for counts in run_counts
    ]
    fields = (
        "activation_rate",
        "net_utility_per_frame",
        "net_utility_per_selected",
        "correction_rate_selected",
        "harm_rate_selected",
    )
    result: dict[str, float | int | None] = {}
    for field in fields:
        values = [
            float(row[field])
            for row in run_metrics
            if row[field] is not None
        ]
        result[field] = float(np.mean(values)) if values else None
    result["runs"] = len(run_metrics)
    return result


def speaker_cluster_bootstrap(
    run_counts: Sequence[np.ndarray],
    *,
    cluster_indices: np.ndarray | None = None,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap speakers and average fixed-seed gate metrics across runs."""
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    if not run_counts:
        raise ValueError("at least one run is required")
    n_clusters = int(run_counts[0].shape[0])
    if any(counts.shape[0] != n_clusters for counts in run_counts):
        raise ValueError("all runs must have the same cluster dimension")
    if cluster_indices is None:
        cluster_indices = np.arange(n_clusters, dtype=np.int64)
    cluster_indices = np.asarray(cluster_indices, dtype=np.int64).reshape(-1)
    if cluster_indices.size == 0:
        raise ValueError("bootstrap requires at least one non-empty cluster")

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        "activation_rate": [],
        "net_utility_per_frame": [],
        "net_utility_per_selected": [],
    }
    for _ in range(int(repeats)):
        sampled = rng.choice(cluster_indices, size=cluster_indices.size, replace=True)
        row = _aggregate_run_counts(run_counts, sampled)
        for name in samples:
            value = row[name]
            if value is not None:
                samples[name].append(float(value))
    point = _aggregate_run_counts(run_counts, cluster_indices)
    result: dict[str, Any] = {
        "point": point,
        "bootstrap_repeats": int(repeats),
    }
    for name, values in samples.items():
        if values:
            low, high = np.quantile(values, [0.025, 0.975])
            result[f"{name}_ci95_low"] = float(low)
            result[f"{name}_ci95_high"] = float(high)
        else:
            result[f"{name}_ci95_low"] = None
            result[f"{name}_ci95_high"] = None
    return result


def _random_cluster_values(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    selected: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_names: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    labels = np.asarray(labels, dtype=np.int64)
    short_predictions = np.asarray(short_scores) >= 0.5
    long_predictions = np.asarray(long_scores) >= 0.5
    short_wrong = short_predictions != labels
    long_wrong = long_predictions != labels
    values = (
        (short_wrong & ~long_wrong).astype(np.int8)
        - (~short_wrong & long_wrong).astype(np.int8)
    )
    result: list[tuple[np.ndarray, int]] = []
    for name in cluster_names:
        mask = cluster_ids == name
        result.append(
            (
                np.ascontiguousarray(values[mask], dtype=np.int8),
                int(np.count_nonzero(selected[mask])),
            )
        )
    return result


def random_gate_comparison(
    run_random_values: Sequence[Sequence[tuple[np.ndarray, int]]],
    observed_net_per_selected: float,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Compare confidence gating with speaker-matched random activation."""
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    if not run_random_values:
        raise ValueError("at least one run is required")
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        run_utilities: list[float] = []
        for cluster_values in run_random_values:
            total_selected = 0
            total_value = 0
            for values, selected_count in cluster_values:
                if selected_count <= 0:
                    continue
                if selected_count >= values.size:
                    chosen = values
                else:
                    chosen = rng.choice(
                        values, size=selected_count, replace=False
                    )
                total_selected += int(selected_count)
                total_value += int(chosen.sum())
            if total_selected:
                run_utilities.append(total_value / total_selected)
        distribution[repeat] = float(np.mean(run_utilities))
    observed = float(observed_net_per_selected)
    exceedances = int(np.count_nonzero(distribution >= observed))
    return {
        "observed_net_per_selected": observed,
        "random_mean": float(np.mean(distribution)),
        "random_ci95_low": float(np.quantile(distribution, 0.025)),
        "random_ci95_high": float(np.quantile(distribution, 0.975)),
        "random_p95": float(np.quantile(distribution, 0.95)),
        "one_sided_p_value": float(
            (1 + exceedances) / (int(repeats) + 1)
        ),
        "repeats": int(repeats),
    }


def _load_run(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(
            f"prediction specification must be LABEL=PATH, got: {spec}"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise ValueError("prediction label must not be empty")
    if not path.exists():
        raise FileNotFoundError(f"prediction file not found: {path}")
    with np.load(path) as payload:
        required = {
            "labels",
            "short_scores",
            "long_scores",
            "condition",
            "noise_name",
            "source_key",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in required
        }
    labels = arrays["labels"].astype(np.int64)
    short_scores = arrays["short_scores"].astype(np.float64)
    long_scores = arrays["long_scores"].astype(np.float64)
    condition = arrays["condition"].astype(str)
    noise_name = arrays["noise_name"].astype(str)
    source_key = arrays["source_key"].astype(str)
    if not (
        labels.size
        == short_scores.size
        == long_scores.size
        == condition.size
        == noise_name.size
        == source_key.size
    ):
        raise ValueError(f"prediction arrays have inconsistent lengths: {path}")
    speaker_ids = np.asarray(
        [speaker_from_source_key(value) for value in source_key],
        dtype=str,
    )
    return {
        "label": label,
        "path": str(path),
        "labels": labels,
        "short_scores": short_scores,
        "long_scores": long_scores,
        "condition": condition,
        "noise_name": noise_name,
        "source_key": source_key,
        "speaker_ids": speaker_ids,
    }


def _validate_run_compatibility(runs: Sequence[dict[str, Any]]) -> None:
    if not runs:
        raise ValueError("at least one prediction run is required")
    first = runs[0]
    for run in runs[1:]:
        for field in ("labels", "condition", "noise_name", "source_key"):
            if not np.array_equal(first[field], run[field]):
                raise ValueError(
                    f"prediction runs differ in frame-aligned field `{field}`: "
                    f"{first['label']} vs {run['label']}"
                )


def _run_fixed_gate(
    run: dict[str, Any],
    *,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    activation_rate: float,
) -> dict[str, Any]:
    threshold, calibration_selected = choose_confidence_threshold(
        run["short_scores"][calibration_mask],
        activation_rate=activation_rate,
    )
    test_confidence = np.abs(run["short_scores"][test_mask] - 0.5)
    test_selected = test_confidence <= threshold
    calibration_labels = run["labels"][calibration_mask]
    test_labels = run["labels"][test_mask]
    calibration_short = run["short_scores"][calibration_mask]
    calibration_long = run["long_scores"][calibration_mask]
    test_short = run["short_scores"][test_mask]
    test_long = run["long_scores"][test_mask]

    _, calibration_counts = count_frames_by_cluster(
        calibration_labels,
        calibration_short,
        calibration_long,
        calibration_selected,
        np.full(calibration_labels.size, "calibration", dtype=str),
    )
    _, test_counts = count_frames_by_cluster(
        test_labels,
        test_short,
        test_long,
        test_selected,
        np.full(test_labels.size, "test", dtype=str),
    )
    return {
        "threshold": threshold,
        "calibration_activation_rate": float(np.mean(calibration_selected)),
        "calibration": metrics_from_counts(calibration_counts.sum(axis=0)),
        "test_selected": test_selected,
        "test": metrics_from_counts(test_counts.sum(axis=0)),
    }


def _fixed_gate_counts(
    run: dict[str, Any],
    test_mask: np.ndarray,
    test_selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_names, counts = count_frames_by_cluster(
        run["labels"][test_mask],
        run["short_scores"][test_mask],
        run["long_scores"][test_mask],
        test_selected,
        run["speaker_ids"][test_mask],
    )
    return cluster_names, counts


def _nonempty_cluster_indices(
    run_counts: Sequence[np.ndarray],
) -> np.ndarray:
    frames = np.zeros(run_counts[0].shape[0], dtype=np.int64)
    for counts in run_counts:
        frames += counts[:, COUNT_COLUMNS.index("frames")]
    return np.flatnonzero(frames > 0)


def _parse_run_specs(values: Sequence[str]) -> list[str]:
    specs = [str(value).strip() for value in values if str(value).strip()]
    if not specs:
        raise ValueError("at least one --prediction LABEL=PATH is required")
    labels = [spec.split("=", 1)[0].strip() for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("prediction labels must be unique")
    return specs


def _markdown_report(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    lines = [
        "# Phase A2: Statistical Confirmation",
        "",
        "## Fixed Protocol",
        "",
        f"- Runs: {', '.join(protocol['run_labels'])}",
        f"- Calibration speakers: {protocol['calibration_speaker_count']}",
        f"- Final-test speakers: {protocol['test_speaker_count']}",
        f"- Speaker split seed: {protocol['speaker_split_seed']}",
        f"- Primary calibration activation: "
        f"{100.0 * protocol['primary_calibration_activation']:.2f}%",
        f"- Speaker-cluster bootstrap repeats: {protocol['bootstrap_repeats']}",
        f"- Random-gate repeats: {protocol['random_repeats']}",
        "",
        "The threshold is fixed on the calibration speakers. The final-test "
        "activation rate is reported as observed, not forced to the calibration "
        "budget.",
        "",
        "## Per-Seed Fixed Gate",
        "",
        "| Run | Threshold | Calibration activation | Test activation | "
        "Net / selected | Net / frame | Correction | Harm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in summary["per_run"].items():
        test = row["test"]
        lines.append(
            f"| {label} | {row['threshold']:.6f} | "
            f"{100.0 * row['calibration_activation_rate']:.3f}% | "
            f"{100.0 * test['activation_rate']:.3f}% | "
            f"{100.0 * test['net_utility_per_selected']:.3f}% | "
            f"{100.0 * test['net_utility_per_frame']:.3f}% | "
            f"{100.0 * test['correction_rate_selected']:.3f}% | "
            f"{100.0 * test['harm_rate_selected']:.3f}% |"
        )

    aggregate = summary["aggregate"]["test"]
    lines.extend(
        [
            "",
            "## Seed-Averaged Test Result",
            "",
            f"- Test activation: "
            f"{100.0 * aggregate['point']['activation_rate']:.3f}% "
            f"(95% CI "
            f"[{100.0 * aggregate['activation_rate_ci95_low']:.3f}, "
            f"{100.0 * aggregate['activation_rate_ci95_high']:.3f}]%).",
            f"- Net utility per selected frame: "
            f"{100.0 * aggregate['point']['net_utility_per_selected']:.3f}% "
            f"(95% CI "
            f"[{100.0 * aggregate['net_utility_per_selected_ci95_low']:.3f}, "
            f"{100.0 * aggregate['net_utility_per_selected_ci95_high']:.3f}]%).",
            f"- Net utility per all frames: "
            f"{100.0 * aggregate['point']['net_utility_per_frame']:.3f}% "
            f"(95% CI "
            f"[{100.0 * aggregate['net_utility_per_frame_ci95_low']:.3f}, "
            f"{100.0 * aggregate['net_utility_per_frame_ci95_high']:.3f}]%).",
        ]
    )

    unseen = summary["aggregate"]["unseen"]
    lines.extend(
        [
            "",
            "## Unseen-Noise Check",
            "",
            f"- Activation: "
            f"{100.0 * unseen['point']['activation_rate']:.3f}%.",
            f"- Net utility per selected frame: "
            f"{100.0 * unseen['point']['net_utility_per_selected']:.3f}% "
            f"(95% CI "
            f"[{100.0 * unseen['net_utility_per_selected_ci95_low']:.3f}, "
            f"{100.0 * unseen['net_utility_per_selected_ci95_high']:.3f}]%).",
        ]
    )

    random_gate = summary["random_gate"]
    lines.extend(
        [
            "",
            "## Confidence vs Random Activation",
            "",
            f"- Confidence gate: "
            f"{100.0 * random_gate['observed_net_per_selected']:.3f}% net "
            "per selected frame.",
            f"- Speaker-matched random gate: "
            f"{100.0 * random_gate['random_mean']:.3f}% mean, "
            f"95th percentile "
            f"{100.0 * random_gate['random_p95']:.3f}%.",
            f"- One-sided Monte Carlo p-value: "
            f"{random_gate['one_sided_p_value']:.5f}.",
            "",
            "## Activation-Budget Curve",
            "",
            "| Calibration activation | Mean test activation | "
            "Net / selected | Net / frame |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for budget, row in summary["activation_budgets"].items():
        aggregate = row["aggregate"]
        lines.append(
            f"| {100.0 * float(budget):.2f}% | "
            f"{100.0 * aggregate['point']['activation_rate']:.3f}% | "
            f"{100.0 * aggregate['point']['net_utility_per_selected']:.3f}% | "
            f"{100.0 * aggregate['point']['net_utility_per_frame']:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Formal GO Criteria",
            "",
            f"- Overall status: **{summary['formal_status']}**.",
        ]
    )
    labels = {
        "minimum_seed_count": "At least the required number of seeds.",
        "all_seed_net_positive": "Every seed has positive test net utility.",
        "cluster_ci_excludes_zero": (
            "Speaker-cluster 95% CI for net utility excludes zero."
        ),
        "confidence_beats_random": (
            "Confidence gating beats matched random activation."
        ),
        "unseen_ci_excludes_zero": (
            "Unseen-noise speaker-cluster 95% CI excludes zero."
        ),
    }
    for name, description in labels.items():
        mark = "PASS" if summary["criteria"][name] else "FAIL"
        lines.append(f"- {mark}: {description}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm fixed uncertainty gating with isolated calibration/test "
            "speakers and speaker-cluster bootstrap."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction run as LABEL=PATH; repeat for multiple seeds.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--speaker-split-seed",
        type=int,
        default=DEFAULT_SPEAKER_SPLIT_SEED,
    )
    parser.add_argument(
        "--primary-calibration-activation",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--activation-budgets",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=DEFAULT_RANDOM_REPEATS,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--random-seed", type=int, default=20260918)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument(
        "--unseen-noise",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be in (0, 1)")
    if not 0.0 < args.primary_calibration_activation <= 1.0:
        parser.error(
            "--primary-calibration-activation must be in (0, 1]"
        )
    if not args.activation_budgets:
        parser.error("--activation-budgets must not be empty")
    if any(not 0.0 < value <= 1.0 for value in args.activation_budgets):
        parser.error("--activation-budgets values must be in (0, 1]")
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")
    if args.random_repeats <= 0:
        parser.error("--random-repeats must be positive")
    if args.min_seeds <= 0:
        parser.error("--min-seeds must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        specs = _parse_run_specs(args.prediction)
        runs = [_load_run(spec) for spec in specs]
        _validate_run_compatibility(runs)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    all_speakers = runs[0]["speaker_ids"]
    (
        calibration_mask,
        test_mask,
        calibration_speakers,
        test_speakers,
    ) = deterministic_speaker_split(
        all_speakers,
        calibration_fraction=args.calibration_fraction,
        seed=args.speaker_split_seed,
    )

    primary_rate = float(args.primary_calibration_activation)
    per_run: dict[str, Any] = {}
    primary_counts: list[np.ndarray] = []
    primary_cluster_names: np.ndarray | None = None
    primary_random_values: list[list[tuple[np.ndarray, int]]] = []
    test_selected_by_run: list[np.ndarray] = []
    for run in runs:
        fixed = _run_fixed_gate(
            run,
            calibration_mask=calibration_mask,
            test_mask=test_mask,
            activation_rate=primary_rate,
        )
        cluster_names, counts = _fixed_gate_counts(
            run,
            test_mask,
            fixed["test_selected"],
        )
        if primary_cluster_names is None:
            primary_cluster_names = cluster_names
        elif not np.array_equal(primary_cluster_names, cluster_names):
            raise RuntimeError("speaker clusters differ across prediction runs")
        primary_counts.append(counts)
        primary_random_values.append(
            _random_cluster_values(
                run["labels"][test_mask],
                run["short_scores"][test_mask],
                run["long_scores"][test_mask],
                fixed["test_selected"],
                run["speaker_ids"][test_mask],
                cluster_names,
            )
        )
        test_selected_by_run.append(fixed["test_selected"])
        per_run[run["label"]] = {
            "path": run["path"],
            "threshold": fixed["threshold"],
            "calibration_activation_rate": fixed[
                "calibration_activation_rate"
            ],
            "calibration": fixed["calibration"],
            "test": fixed["test"],
        }

    if primary_cluster_names is None:
        raise RuntimeError("no test clusters were available")
    all_cluster_indices = np.arange(primary_cluster_names.size, dtype=np.int64)
    test_bootstrap = speaker_cluster_bootstrap(
        primary_counts,
        cluster_indices=all_cluster_indices,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    observed_net = float(
        test_bootstrap["point"]["net_utility_per_selected"]
    )
    random_gate = random_gate_comparison(
        primary_random_values,
        observed_net,
        repeats=args.random_repeats,
        seed=args.random_seed,
    )

    noise_name = runs[0]["noise_name"]
    unseen_names = {str(value) for value in args.unseen_noise}
    unseen_mask = test_mask & (runs[0]["condition"] != "clean") & np.isin(
        noise_name, sorted(unseen_names)
    )
    unseen_counts: list[np.ndarray] = []
    for run, selected in zip(runs, test_selected_by_run):
        _, counts = count_frames_by_cluster(
            run["labels"][test_mask],
            run["short_scores"][test_mask],
            run["long_scores"][test_mask],
            selected,
            run["speaker_ids"][test_mask],
            mask=unseen_mask[test_mask],
        )
        unseen_counts.append(counts)
    unseen_cluster_indices = _nonempty_cluster_indices(unseen_counts)
    unseen_bootstrap = speaker_cluster_bootstrap(
        unseen_counts,
        cluster_indices=unseen_cluster_indices,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed + 1,
    )

    activation_budgets: dict[str, Any] = {}
    for budget_index, budget in enumerate(
        sorted(set(float(value) for value in args.activation_budgets))
    ):
        run_rows: dict[str, Any] = {}
        run_counts: list[np.ndarray] = []
        for run in runs:
            fixed = _run_fixed_gate(
                run,
                calibration_mask=calibration_mask,
                test_mask=test_mask,
                activation_rate=budget,
            )
            _, counts = _fixed_gate_counts(
                run,
                test_mask,
                fixed["test_selected"],
            )
            run_counts.append(counts)
            run_rows[run["label"]] = {
                "threshold": fixed["threshold"],
                "calibration_activation_rate": fixed[
                    "calibration_activation_rate"
                ],
                "test": fixed["test"],
            }
        activation_budgets[f"{budget:.6f}"] = {
            "calibration_activation": budget,
            "per_run": run_rows,
            "aggregate": speaker_cluster_bootstrap(
                run_counts,
                cluster_indices=all_cluster_indices,
                repeats=args.bootstrap_repeats,
                seed=args.bootstrap_seed + 10 + budget_index,
            ),
        }

    criteria = {
        "minimum_seed_count": len(runs) >= args.min_seeds,
        "all_seed_net_positive": all(
            float(row["test"]["net_utility_per_selected"]) > 0.0
            for row in per_run.values()
            if row["test"]["net_utility_per_selected"] is not None
        ),
        "cluster_ci_excludes_zero": bool(
            test_bootstrap["net_utility_per_selected_ci95_low"] is not None
            and test_bootstrap["net_utility_per_selected_ci95_low"] > 0.0
        ),
        "confidence_beats_random": bool(
            random_gate["one_sided_p_value"] is not None
            and random_gate["one_sided_p_value"] < 0.05
        ),
        "unseen_ci_excludes_zero": bool(
            unseen_bootstrap["net_utility_per_selected_ci95_low"] is not None
            and unseen_bootstrap["net_utility_per_selected_ci95_low"] > 0.0
        ),
    }
    if not criteria["minimum_seed_count"]:
        formal_status = "INCOMPLETE"
    elif all(criteria.values()):
        formal_status = "FORMAL_GO"
    else:
        formal_status = "NOT_CONFIRMED"

    summary = {
        "protocol": {
            "run_labels": [run["label"] for run in runs],
            "prediction_paths": {
                run["label"]: run["path"] for run in runs
            },
            "frames": int(all_speakers.size),
            "speaker_count": int(np.unique(all_speakers).size),
            "calibration_speaker_count": len(calibration_speakers),
            "test_speaker_count": len(test_speakers),
            "calibration_speakers": calibration_speakers,
            "test_speakers": test_speakers,
            "calibration_fraction": float(args.calibration_fraction),
            "speaker_split_seed": int(args.speaker_split_seed),
            "primary_calibration_activation": primary_rate,
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "random_repeats": int(args.random_repeats),
            "bootstrap_seed": int(args.bootstrap_seed),
            "random_seed": int(args.random_seed),
            "unseen_noise": sorted(unseen_names),
        },
        "per_run": per_run,
        "aggregate": {
            "test": test_bootstrap,
            "unseen": unseen_bootstrap,
        },
        "random_gate": random_gate,
        "activation_budgets": activation_budgets,
        "criteria": criteria,
        "formal_status": formal_status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "phase_a2_confirmatory.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        args.output_dir / "phase_a2_confirmatory.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(_markdown_report(summary))
    print(f"Phase A2 status: {formal_status}")
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
