# -*- coding: utf-8 -*-
"""Analyze A10 temporal-value predictability from frozen predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    COUNT_COLUMNS,
    metrics_from_counts,
)
from reproductions.difficulty_adaptive_context.data import FRAME_HOP
from reproductions.difficulty_adaptive_context.evaluate_context import (
    binary_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
DEFAULT_RESULTS_JSON = DEFAULT_PREDICTIONS.with_name("results.json")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a10_value_predictability"
)
FEATURE_NAMES = (
    "posterior_entropy",
    "posterior_change",
    "short_term_variance",
    "prediction_switch_rate",
)
GATE_NAMES = ("uncertainty", "value", "random")


def _require_1d(
    payload: Any,
    path: Path,
    name: str,
) -> np.ndarray:
    if name not in payload.files:
        raise ValueError(f"{path} is missing array `{name}`")
    array = np.asarray(payload[name])
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional, got {array.shape}")
    return array


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the frozen frame-level prediction bundle."""
    if not path.exists():
        raise FileNotFoundError(f"predictions not found: {path}")
    with np.load(path) as payload:
        required = {
            "labels",
            "short_scores",
            "full_adaptive_scores",
            "test_mask",
            "calibration_mask",
            "speaker_ids",
            "condition",
            "noise_name",
            "source_key",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        arrays = {
            name: _require_1d(payload, path, name).copy()
            for name in required
        }

    lengths = {name: int(array.size) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays have inconsistent lengths: {lengths}")
    if arrays["labels"].size == 0:
        raise ValueError("prediction arrays must not be empty")

    labels = arrays["labels"].astype(np.int64)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    for name in ("short_scores", "full_adaptive_scores"):
        values = arrays[name].astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"`{name}` contains non-finite values")
        arrays[name] = values

    calibration_mask = arrays["calibration_mask"].astype(bool)
    test_mask = arrays["test_mask"].astype(bool)
    if np.any(calibration_mask & test_mask):
        raise ValueError("calibration and test masks must be disjoint")
    if not np.all(calibration_mask | test_mask):
        raise ValueError("calibration and test masks must cover every frame")
    arrays["labels"] = labels
    arrays["calibration_mask"] = calibration_mask
    arrays["test_mask"] = test_mask
    for name in ("speaker_ids", "condition", "noise_name", "source_key"):
        arrays[name] = arrays[name].astype(str)
        if np.any(np.char.str_len(arrays[name]) == 0):
            raise ValueError(f"`{name}` contains an empty value")
    return arrays


def utterance_segments(
    source_key: np.ndarray,
    noise_name: np.ndarray,
    condition: np.ndarray,
) -> list[tuple[int, int]]:
    """Return contiguous frame ranges for every evaluated utterance variant."""
    if not (
        source_key.size == noise_name.size == condition.size
    ):
        raise ValueError("utterance metadata arrays must have equal lengths")
    composite = np.char.add(
        np.char.add(
            np.char.add(source_key.astype(str), "|"),
            noise_name.astype(str),
        ),
        np.char.add("|", condition.astype(str)),
    )
    starts = np.r_[
        0,
        np.flatnonzero(composite[1:] != composite[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], composite.size]
    segments = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end > start
    ]
    unique, unique_counts = np.unique(composite, return_counts=True)
    segment_counts = np.asarray(
        [end - start for start, end in segments],
        dtype=np.int64,
    )
    if unique.size != len(segments) or not np.array_equal(
        np.sort(unique_counts),
        np.sort(segment_counts),
    ):
        raise ValueError(
            "source/noise/condition identities are not unique contiguous "
            "utterance variants; frame-level temporal features require "
            "utterance ids in the prediction bundle"
        )
    return segments


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return binary posterior entropy in natural-log units."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, indices - window + 1)
    counts = indices - starts + 1
    return (
        cumulative[indices + 1] - cumulative[starts]
    ) / counts.astype(np.float64)


def _trailing_variance(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    cumulative_sq = np.concatenate(
        ([0.0], np.cumsum(values * values, dtype=np.float64))
    )
    indices = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, indices - window + 1)
    counts = (indices - starts + 1).astype(np.float64)
    sums = cumulative[indices + 1] - cumulative[starts]
    sums_sq = cumulative_sq[indices + 1] - cumulative_sq[starts]
    variance = sums_sq / counts - (sums / counts) ** 2
    return np.maximum(variance, 0.0)


def extract_short_only_features(
    short_scores: np.ndarray,
    segments: list[tuple[int, int]],
    *,
    window_frames: int,
) -> np.ndarray:
    """Extract strictly causal score-only features inside utterance bounds."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if not segments:
        raise ValueError("at least one utterance segment is required")
    features = np.empty((scores.size, len(FEATURE_NAMES)), dtype=np.float64)

    for start, end in segments:
        if not 0 <= start < end <= scores.size:
            raise ValueError("utterance segment falls outside the score array")
        local_scores = scores[start:end]
        local_size = local_scores.size

        local_change = np.zeros(local_size, dtype=np.float64)
        if local_size > 1:
            local_change[1:] = np.abs(np.diff(local_scores))

        local_predictions = (local_scores >= 0.5).astype(np.float64)
        local_switch = np.zeros(local_size, dtype=np.float64)
        if local_size > 1:
            local_switch[1:] = np.abs(np.diff(local_predictions))

        features[start:end, 0] = binary_entropy(local_scores)
        features[start:end, 1] = local_change
        features[start:end, 2] = _trailing_variance(
            local_scores,
            window_frames,
        )
        features[start:end, 3] = _trailing_mean(
            local_switch,
            window_frames,
        )

    if np.any(~np.isfinite(features)):
        raise RuntimeError("feature extraction produced non-finite values")
    return features


def signed_temporal_value(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> np.ndarray:
    """Return 1[refiner correct] - 1[short correct]."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    refined_scores = np.asarray(refined_scores, dtype=np.float64).reshape(-1)
    if not (labels.size == short_scores.size == refined_scores.size):
        raise ValueError("signed-value arrays must have equal lengths")
    short_correct = (short_scores >= 0.5).astype(np.int64) == labels
    refined_correct = (refined_scores >= 0.5).astype(np.int64) == labels
    return refined_correct.astype(np.int8) - short_correct.astype(np.int8)


def select_top_fraction(
    scores: np.ndarray,
    activation_rate: float,
) -> tuple[float, np.ndarray]:
    """Select the highest-scoring calibration frames without test leakage."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("cannot threshold an empty score array")
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not 0.0 < float(activation_rate) <= 1.0:
        raise ValueError("activation_rate must be in (0, 1]")
    target = int(math.ceil(float(activation_rate) * scores.size))
    target = min(max(target, 1), scores.size)
    order = np.argsort(-scores, kind="stable")
    selected = np.zeros(scores.size, dtype=bool)
    selected[order[:target]] = True
    threshold = float(np.min(scores[selected]))
    return threshold, selected


def fit_value_score(
    features: np.ndarray,
    values: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit multinomial logistic regression and score expected signed value."""
    features = np.asarray(features, dtype=np.float64)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    calibration_mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if features.ndim != 2 or features.shape[0] != values.size:
        raise ValueError("features and values have incompatible shapes")
    if calibration_mask.size != values.size:
        raise ValueError("calibration mask does not match the frames")
    calibration_values = values[calibration_mask]
    classes = np.unique(calibration_values)
    if classes.size < 2:
        raise ValueError("calibration frames contain fewer than two value classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=2_000,
            random_state=int(random_seed),
        ),
    )
    model.fit(features[calibration_mask], calibration_values)
    probabilities = model.predict_proba(features)
    fitted_classes = model.named_steps["logisticregression"].classes_.astype(
        np.int64
    )
    expected_value = probabilities @ fitted_classes.astype(np.float64)
    classifier = model.named_steps["logisticregression"]
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "classes": fitted_classes.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "iterations": [
            int(value) for value in np.atleast_1d(classifier.n_iter_)
        ],
    }
    return expected_value.astype(np.float64), metadata


def _cluster_count_matrix(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    refined_scores = np.asarray(refined_scores, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (
        labels.size
        == short_scores.size
        == refined_scores.size
        == selected.size
        == cluster_codes.size
    ):
        raise ValueError("cluster-count arrays must have equal lengths")
    short_wrong = (short_scores >= 0.5).astype(np.int64) != labels
    refined_wrong = (refined_scores >= 0.5).astype(np.int64) != labels
    values = {
        "frames": np.ones(labels.size, dtype=np.int64),
        "selected": selected.astype(np.int64),
        "correction": (short_wrong & ~refined_wrong).astype(np.int64),
        "harm": (~short_wrong & refined_wrong).astype(np.int64),
        "short_error": short_wrong.astype(np.int64),
        "long_error": refined_wrong.astype(np.int64),
        "gated_error": np.where(
            selected,
            refined_wrong,
            short_wrong,
        ).astype(np.int64),
    }
    return np.column_stack(
        [
            np.bincount(
                cluster_codes,
                weights=values[name],
                minlength=n_clusters,
            )
            for name in COUNT_COLUMNS
        ]
    ).astype(np.int64)


def _gate_metrics(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    counts = _cluster_count_matrix(
        labels,
        short_scores,
        refined_scores,
        selected,
        cluster_codes,
        n_clusters,
    )
    utility = metrics_from_counts(counts.sum(axis=0))
    adaptive_scores = np.where(selected, refined_scores, short_scores)
    accuracy = binary_metrics(labels, adaptive_scores)
    return utility, accuracy, counts


def paired_speaker_cluster_bootstrap(
    first_counts: np.ndarray,
    second_counts: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap speakers and estimate first-minus-second net utility."""
    first_counts = np.asarray(first_counts, dtype=np.int64)
    second_counts = np.asarray(second_counts, dtype=np.int64)
    if first_counts.shape != second_counts.shape:
        raise ValueError("paired count matrices must have equal shapes")
    if first_counts.ndim != 2 or first_counts.shape[0] == 0:
        raise ValueError("paired count matrices must be non-empty and 2D")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")

    selected_column = COUNT_COLUMNS.index("selected")
    correction_column = COUNT_COLUMNS.index("correction")
    harm_column = COUNT_COLUMNS.index("harm")

    def utility(counts: np.ndarray) -> tuple[float, float]:
        selected = int(counts[:, selected_column].sum())
        net = int(
            counts[:, correction_column].sum()
            - counts[:, harm_column].sum()
        )
        net_per_frame = float(net / counts[:, 0].sum())
        net_per_selected = float(net / selected) if selected else float("nan")
        return net_per_selected, net_per_frame

    first_point, first_frame_point = utility(first_counts)
    second_point, second_frame_point = utility(second_counts)
    rng = np.random.default_rng(int(seed))
    first_values: list[float] = []
    second_values: list[float] = []
    first_frame_values: list[float] = []
    second_frame_values: list[float] = []
    differences: list[float] = []
    frame_differences: list[float] = []
    for _ in range(int(repeats)):
        sampled = rng.integers(
            0,
            first_counts.shape[0],
            size=first_counts.shape[0],
        )
        first_value, first_frame_value = utility(first_counts[sampled])
        second_value, second_frame_value = utility(second_counts[sampled])
        if np.isfinite(first_value) and np.isfinite(second_value):
            first_values.append(first_value)
            second_values.append(second_value)
            differences.append(first_value - second_value)
        first_frame_values.append(first_frame_value)
        second_frame_values.append(second_frame_value)
        frame_differences.append(first_frame_value - second_frame_value)

    def interval(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    first_low, first_high = interval(first_values)
    second_low, second_high = interval(second_values)
    difference_low, difference_high = interval(differences)
    first_frame_low, first_frame_high = interval(first_frame_values)
    second_frame_low, second_frame_high = interval(second_frame_values)
    frame_difference_low, frame_difference_high = interval(
        frame_differences
    )
    return {
        "repeats": int(repeats),
        "first_net_utility_per_selected": first_point,
        "first_net_utility_per_selected_ci95_low": first_low,
        "first_net_utility_per_selected_ci95_high": first_high,
        "second_net_utility_per_selected": second_point,
        "second_net_utility_per_selected_ci95_low": second_low,
        "second_net_utility_per_selected_ci95_high": second_high,
        "difference_net_utility_per_selected": (
            first_point - second_point
        ),
        "difference_net_utility_per_selected_ci95_low": difference_low,
        "difference_net_utility_per_selected_ci95_high": difference_high,
        "first_net_utility_per_frame": first_frame_point,
        "first_net_utility_per_frame_ci95_low": first_frame_low,
        "first_net_utility_per_frame_ci95_high": first_frame_high,
        "second_net_utility_per_frame": second_frame_point,
        "second_net_utility_per_frame_ci95_low": second_frame_low,
        "second_net_utility_per_frame_ci95_high": second_frame_high,
        "difference_net_utility_per_frame": (
            first_frame_point - second_frame_point
        ),
        "difference_net_utility_per_frame_ci95_low": (
            frame_difference_low
        ),
        "difference_net_utility_per_frame_ci95_high": (
            frame_difference_high
        ),
    }


def _evaluate_gate(
    name: str,
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    activation_rate: float,
) -> dict[str, Any]:
    threshold, calibration_selected = select_top_fraction(
        scores[calibration_mask],
        activation_rate,
    )
    test_selected = scores[test_mask] >= threshold
    test_labels = labels[test_mask]
    test_short = short_scores[test_mask]
    test_refined = refined_scores[test_mask]
    utility, accuracy, counts = _gate_metrics(
        test_labels,
        test_short,
        test_refined,
        test_selected,
        cluster_codes,
        n_clusters,
    )
    return {
        "name": name,
        "calibration_activation_target": float(activation_rate),
        "threshold": threshold,
        "calibration_activation_rate": float(
            np.mean(calibration_selected)
        ),
        "test_activation_rate": float(np.mean(test_selected)),
        "test_correction": int(utility["correction"]),
        "test_harm": int(utility["harm"]),
        "test_net_utility": int(utility["signed_utility"]),
        "test_net_utility_per_selected": utility[
            "net_utility_per_selected"
        ],
        "test_net_utility_per_frame": utility["net_utility_per_frame"],
        "test_correction_rate_selected": utility[
            "correction_rate_selected"
        ],
        "test_harm_rate_selected": utility["harm_rate_selected"],
        "test_f1": accuracy["f1"],
        "test_error": accuracy["error"],
        "cluster_counts": counts,
    }


def _random_gate_summary(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    activation_rate: float,
    repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(random_seed))
    selected_values: list[float] = []
    net_values: list[float] = []
    net_frame_values: list[float] = []
    f1_values: list[float] = []
    correction_values: list[float] = []
    harm_values: list[float] = []
    test_labels = labels[test_mask]
    test_short = short_scores[test_mask]
    test_refined = refined_scores[test_mask]
    for _ in range(int(repeats)):
        random_scores = rng.random(labels.size)
        threshold, calibration_selected = select_top_fraction(
            random_scores[calibration_mask],
            activation_rate,
        )
        test_selected = random_scores[test_mask] >= threshold
        adaptive_scores = np.where(
            test_selected,
            test_refined,
            test_short,
        )
        accuracy = binary_metrics(test_labels, adaptive_scores)
        correction = int(
            np.count_nonzero(
                ((test_short >= 0.5).astype(np.int64) != test_labels)
                & ((test_refined >= 0.5).astype(np.int64) == test_labels)
                & test_selected
            )
        )
        harm = int(
            np.count_nonzero(
                ((test_short >= 0.5).astype(np.int64) == test_labels)
                & ((test_refined >= 0.5).astype(np.int64) != test_labels)
                & test_selected
            )
        )
        selected = int(np.count_nonzero(test_selected))
        selected_values.append(float(np.mean(test_selected)))
        correction_values.append(
            float(correction / selected) if selected else float("nan")
        )
        harm_values.append(
            float(harm / selected) if selected else float("nan")
        )
        net_values.append(
            float((correction - harm) / selected)
            if selected
            else float("nan")
        )
        net_frame_values.append(
            float((correction - harm) / test_selected.size)
        )
        f1_values.append(float(accuracy["f1"]))
    calibration_activation = float(
        math.ceil(float(activation_rate) * np.count_nonzero(calibration_mask))
        / np.count_nonzero(calibration_mask)
    )

    def summary(values: list[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return {"mean": None, "ci95_low": None, "ci95_high": None}
        low, high = np.quantile(array, [0.025, 0.975])
        return {
            "mean": float(np.mean(array)),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }

    return {
        "name": "random",
        "calibration_activation_target": float(activation_rate),
        "calibration_activation_rate": calibration_activation,
        "test_activation_rate": summary(selected_values),
        "test_net_utility_per_selected": summary(net_values),
        "test_net_utility_per_frame": summary(net_frame_values),
        "test_correction_rate_selected": summary(correction_values),
        "test_harm_rate_selected": summary(harm_values),
        "test_f1": summary(f1_values),
        "repeats": int(repeats),
    }


def temporal_value_map(
    features: np.ndarray,
    values: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    bins: int,
) -> dict[str, Any]:
    """Aggregate E[v] over calibration entropy/variance quantile bins."""
    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    features = np.asarray(features, dtype=np.float64)
    values = np.asarray(values, dtype=np.int8).reshape(-1)
    calibration_mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if features.ndim != 2 or features.shape[0] != values.size:
        raise ValueError("feature/value shapes are incompatible")
    entropy = features[calibration_mask, 0]
    variance = features[calibration_mask, 2]
    local_values = values[calibration_mask]
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    entropy_edges = np.unique(np.quantile(entropy, quantiles))
    variance_edges = np.unique(np.quantile(variance, quantiles))
    if entropy_edges.size < 2 or variance_edges.size < 2:
        raise ValueError("calibration features cannot form two-dimensional bins")
    entropy_edges[0] = -np.inf
    entropy_edges[-1] = np.inf
    variance_edges[0] = -np.inf
    variance_edges[-1] = np.inf
    entropy_bins = np.clip(
        np.digitize(entropy, entropy_edges[1:-1], right=False),
        0,
        entropy_edges.size - 2,
    )
    variance_bins = np.clip(
        np.digitize(variance, variance_edges[1:-1], right=False),
        0,
        variance_edges.size - 2,
    )
    count = np.zeros(
        (variance_edges.size - 1, entropy_edges.size - 1),
        dtype=np.int64,
    )
    mean_value = np.full(count.shape, np.nan, dtype=np.float64)
    correction_rate = np.full(count.shape, np.nan, dtype=np.float64)
    harm_rate = np.full(count.shape, np.nan, dtype=np.float64)
    for variance_bin in range(count.shape[0]):
        for entropy_bin in range(count.shape[1]):
            mask = (
                (variance_bins == variance_bin)
                & (entropy_bins == entropy_bin)
            )
            size = int(np.count_nonzero(mask))
            count[variance_bin, entropy_bin] = size
            if size:
                cell_values = local_values[mask]
                mean_value[variance_bin, entropy_bin] = float(
                    np.mean(cell_values)
                )
                correction_rate[variance_bin, entropy_bin] = float(
                    np.mean(cell_values == 1)
                )
                harm_rate[variance_bin, entropy_bin] = float(
                    np.mean(cell_values == -1)
                )
    return {
        "x_feature": FEATURE_NAMES[0],
        "y_feature": FEATURE_NAMES[2],
        "x_edges": entropy_edges.tolist(),
        "y_edges": variance_edges.tolist(),
        "count": count.tolist(),
        "mean_value": mean_value.tolist(),
        "correction_rate": correction_rate.tolist(),
        "harm_rate": harm_rate.tolist(),
        "max_mean_value": float(np.nanmax(mean_value)),
        "max_mean_value_count": int(
            count.reshape(-1)[int(np.nanargmax(mean_value))]
        ),
    }


def _write_value_map_csv(map_summary: dict[str, Any], path: Path) -> None:
    x_edges = np.asarray(map_summary["x_edges"], dtype=np.float64)
    y_edges = np.asarray(map_summary["y_edges"], dtype=np.float64)
    count = np.asarray(map_summary["count"], dtype=np.int64)
    mean_value = np.asarray(map_summary["mean_value"], dtype=np.float64)
    correction = np.asarray(
        map_summary["correction_rate"],
        dtype=np.float64,
    )
    harm = np.asarray(map_summary["harm_rate"], dtype=np.float64)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "entropy_bin",
                "variance_bin",
                "entropy_low",
                "entropy_high",
                "variance_low",
                "variance_high",
                "frames",
                "mean_signed_value",
                "correction_rate",
                "harm_rate",
            ],
        )
        writer.writeheader()
        for y_index in range(count.shape[0]):
            for x_index in range(count.shape[1]):
                writer.writerow(
                    {
                        "entropy_bin": x_index,
                        "variance_bin": y_index,
                        "entropy_low": x_edges[x_index],
                        "entropy_high": x_edges[x_index + 1],
                        "variance_low": y_edges[y_index],
                        "variance_high": y_edges[y_index + 1],
                        "frames": int(count[y_index, x_index]),
                        "mean_signed_value": mean_value[y_index, x_index],
                        "correction_rate": correction[y_index, x_index],
                        "harm_rate": harm[y_index, x_index],
                    }
                )


def _write_value_map_plot(
    map_summary: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_value = np.asarray(map_summary["mean_value"], dtype=np.float64)
    count = np.asarray(map_summary["count"], dtype=np.int64)
    x_edges = np.asarray(map_summary["x_edges"], dtype=np.float64)
    y_edges = np.asarray(map_summary["y_edges"], dtype=np.float64)
    masked = np.ma.masked_where(count == 0, mean_value)
    absolute_limit = max(
        0.01,
        float(np.nanmax(np.abs(mean_value[np.isfinite(mean_value)]))),
    )
    _, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-absolute_limit,
        vmax=absolute_limit,
    )
    axes[0].set_title("Calibration E[signed temporal value]")
    axes[0].set_xlabel("Posterior entropy quantile bin")
    axes[0].set_ylabel("Short-term variance quantile bin")
    plt.colorbar(image, ax=axes[0], label="E[v]")
    count_image = axes[1].imshow(
        count,
        origin="lower",
        aspect="auto",
        cmap="Blues",
    )
    axes[1].set_title("Calibration frame count")
    axes[1].set_xlabel("Posterior entropy quantile bin")
    axes[1].set_ylabel("Short-term variance quantile bin")
    plt.colorbar(count_image, ax=axes[1], label="Frames")
    for axis in axes:
        axis.set_xticks(np.arange(x_edges.size - 1))
        axis.set_yticks(np.arange(y_edges.size - 1))
        axis.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_gate_plot(
    budget_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [float(row["calibration_activation_target"]) for row in budget_rows]
    _, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, label in (
        ("uncertainty", "Uncertainty"),
        ("value", "Value logistic"),
    ):
        rows = [row[name] for row in budget_rows]
        axes[0].plot(
            budgets,
            [float(row["test_net_utility_per_selected"]) for row in rows],
            marker="o",
            label=label,
        )
        axes[1].plot(
            budgets,
            [float(row["test_f1"]) for row in rows],
            marker="o",
            label=label,
        )
    random_rows = [row["random"] for row in budget_rows]
    axes[0].errorbar(
        budgets,
        [
            float(row["test_net_utility_per_selected"]["mean"])
            for row in random_rows
        ],
        yerr=np.asarray(
            [
                [
                    float(row["test_net_utility_per_selected"]["mean"])
                    - float(
                        row["test_net_utility_per_selected"]["ci95_low"]
                    )
                    for row in random_rows
                ],
                [
                    float(
                        row["test_net_utility_per_selected"]["ci95_high"]
                    )
                    - float(row["test_net_utility_per_selected"]["mean"])
                    for row in random_rows
                ],
            ]
        ),
        marker="s",
        linestyle="--",
        label="Random mean and 95% interval",
    )
    axes[1].plot(
        budgets,
        [float(row["test_f1"]["mean"]) for row in random_rows],
        marker="s",
        linestyle="--",
        label="Random mean",
    )
    for axis in axes:
        axis.set_xlabel("Calibration activation target")
        axis.set_xticks(budgets)
        axis.set_xticklabels([f"{100.0 * value:.0f}%" for value in budgets])
        axis.grid(alpha=0.25)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Net utility / selected")
    axes[0].set_title("Signed refinement value")
    axes[1].set_ylabel("Test F1")
    axes[1].set_title("Adaptive F1")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _md_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _md_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _write_markdown(
    summary: dict[str, Any],
    path: Path,
) -> None:
    protocol = summary["protocol"]
    signed = summary["signed_value"]
    primary = summary["primary"]
    map_summary = summary["temporal_value_map"]
    lines = [
        "# A10 Temporal Value Predictability",
        "",
        "## Frozen Protocol",
        "",
        f"- Predictions: `{protocol['predictions']}`",
        f"- Adaptive checkpoint: `{protocol['adaptive_checkpoint']}`",
        f"- Refinement span: {protocol['lookback_frames']} frames "
        f"({protocol['lookback_seconds']:.2f} s)",
        f"- Temporal feature window: {protocol['window_frames']} frames "
        f"({protocol['window_seconds'] * 1000.0:.0f} ms)",
        f"- Calibration/test speakers: "
        f"{protocol['calibration_speaker_count']}/"
        f"{protocol['test_speaker_count']}",
        f"- Primary calibration activation: "
        f"{100.0 * protocol['primary_activation']:.2f}%",
        "- Features: "
        + ", ".join(f"`{name}`" for name in protocol["feature_names"]),
        "- Short hidden-embedding delta was not present in the frozen "
        "prediction bundle, so it was not used.",
        "",
        "All router fitting and thresholds use calibration speakers only. "
        "Test activation is observed, not forced to the calibration target. "
        "The random baseline repeats a calibration-thresholded uniform gate.",
        "",
        "## Signed Temporal Value",
        "",
        "| Split | Frames | +1 Correction | 0 No effect | -1 Harm | "
        "E[v] | Correction/frame | Harm/frame |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("calibration", "test"):
        row = signed[name]
        lines.append(
            f"| {name} | {row['frames']:,} | {row['positive']:,} | "
            f"{row['zero']:,} | {row['negative']:,} | "
            f"{row['mean_value']:.6f} | "
            f"{row['correction_rate'] * 100.0:.3f}% | "
            f"{row['harm_rate'] * 100.0:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Primary Gate Comparison",
            "",
            "| Gate | Test activation | Correction/selected | "
            "Harm/selected | Net/selected | Net/frame | F1 | "
            "Cluster 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("uncertainty", "value"):
        row = primary[name]
        bootstrap = row["cluster_bootstrap"]
        bootstrap_prefix = "first" if name == "value" else "second"
        lines.append(
            f"| {name} | "
            f"{100.0 * row['test_activation_rate']:.3f}% | "
            f"{100.0 * row['test_correction_rate_selected']:.3f}% | "
            f"{100.0 * row['test_harm_rate_selected']:.3f}% | "
            f"{100.0 * row['test_net_utility_per_selected']:.3f}% | "
            f"{100.0 * row['test_net_utility_per_frame']:.3f}% | "
            f"{_md_number(row['test_f1'])} | "
            f"[{100.0 * bootstrap[f'{bootstrap_prefix}_net_utility_per_selected_ci95_low']:.3f}%, "
            f"{100.0 * bootstrap[f'{bootstrap_prefix}_net_utility_per_selected_ci95_high']:.3f}%] |"
        )
    random_row = primary["random"]
    random_net = random_row["test_net_utility_per_selected"]
    random_f1 = random_row["test_f1"]
    lines.append(
        f"| random mean | "
        f"{100.0 * random_row['test_activation_rate']['mean']:.3f}% | "
        f"{100.0 * random_row['test_correction_rate_selected']['mean']:.3f}% | "
        f"{100.0 * random_row['test_harm_rate_selected']['mean']:.3f}% | "
        f"{100.0 * random_net['mean']:.3f}% | "
        f"{100.0 * random_row['test_net_utility_per_frame']['mean']:.3f}% | "
        f"{_md_number(random_f1['mean'])} | "
        f"[{100.0 * random_net['ci95_low']:.3f}%, "
        f"{100.0 * random_net['ci95_high']:.3f}%] |"
    )
    paired = primary["value_minus_uncertainty"]
    lines.extend(
        [
            "",
            f"- Value minus uncertainty net/selected: "
            f"{_md_percent(paired['difference_net_utility_per_selected'])} "
            f"(paired speaker-cluster 95% CI "
            f"[{_md_percent(paired['difference_net_utility_per_selected_ci95_low'])}, "
            f"{_md_percent(paired['difference_net_utility_per_selected_ci95_high'])}]).",
            "",
            "## Activation-Budget Comparison",
            "",
            "| Calibration target | Gate | Mean test activation | "
            "Net/selected | Net/frame | F1 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["activation_budget_rows"]:
        budget = float(row["calibration_activation_target"])
        for name in ("uncertainty", "value"):
            gate = row[name]
            lines.append(
                f"| {100.0 * budget:.0f}% | {name} | "
                f"{100.0 * gate['test_activation_rate']:.3f}% | "
                f"{100.0 * gate['test_net_utility_per_selected']:.3f}% | "
                f"{100.0 * gate['test_net_utility_per_frame']:.3f}% | "
                f"{_md_number(gate['test_f1'])} |"
            )
        gate = row["random"]
        lines.append(
            f"| {100.0 * budget:.0f}% | random mean | "
            f"{100.0 * gate['test_activation_rate']['mean']:.3f}% | "
            f"{100.0 * gate['test_net_utility_per_selected']['mean'] * 100.0:.3f}% | "
            f"{100.0 * gate['test_net_utility_per_frame']['mean']:.3f}% | "
            f"{_md_number(gate['test_f1']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Temporal Context Value Map",
            "",
            "- Coordinates: posterior entropy (x) and short-term posterior "
            "variance (y), each split into calibration quantile bins.",
            f"- Highest finite-bin E[v]: "
            f"{map_summary['max_mean_value']:.6f} with "
            f"{map_summary['max_mean_value_count']:,} calibration frames.",
            "- Full bins: `a10_temporal_value_map.csv`.",
            "- Plot: `a10_temporal_value_map.png`.",
            "",
            "## GO Assessment",
            "",
            f"- Status: **{summary['go_assessment']['status']}**.",
        ]
    )
    for name, description in (
        (
            "value_net_utility_above_uncertainty",
            "Value gate has higher test net/selected than uncertainty gate.",
        ),
        (
            "paired_cluster_ci_excludes_zero",
            "Paired speaker-cluster 95% CI for the difference excludes zero.",
        ),
    ):
        mark = (
            "PASS"
            if summary["go_assessment"]["criteria"][name]
            else "FAIL"
        )
        lines.append(f"- {mark}: {description}")
    lines.extend(
        [
            f"- Interpretation: {summary['go_assessment']['interpretation']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _signed_value_summary(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    local = values[mask]
    if local.size == 0:
        raise ValueError("signed-value summary subset is empty")
    positive = int(np.count_nonzero(local == 1))
    zero = int(np.count_nonzero(local == 0))
    negative = int(np.count_nonzero(local == -1))
    return {
        "frames": int(local.size),
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "mean_value": float(np.mean(local)),
        "correction_rate": float(positive / local.size),
        "harm_rate": float(negative / local.size),
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload.get("protocol", {}))


def run(args: argparse.Namespace) -> int:
    predictions = load_predictions(args.predictions)
    labels = predictions["labels"]
    short_scores = predictions["short_scores"]
    refined_scores = predictions["full_adaptive_scores"]
    calibration_mask = predictions["calibration_mask"]
    test_mask = predictions["test_mask"]
    speaker_ids = predictions["speaker_ids"]
    segments = utterance_segments(
        predictions["source_key"],
        predictions["noise_name"],
        predictions["condition"],
    )
    features = extract_short_only_features(
        short_scores,
        segments,
        window_frames=args.window_frames,
    )
    values = signed_temporal_value(
        labels,
        short_scores,
        refined_scores,
    )
    value_score, model_metadata = fit_value_score(
        features,
        values,
        calibration_mask,
        random_seed=args.random_seed,
    )
    uncertainty_score = features[:, 0]

    test_cluster_names, test_cluster_codes = np.unique(
        speaker_ids[test_mask],
        return_inverse=True,
    )
    n_test_clusters = int(test_cluster_names.size)
    gate_budget_rows: list[dict[str, Any]] = []
    primary_gates: dict[str, Any] | None = None
    for budget_index, budget in enumerate(
        sorted(set(float(value) for value in args.activation_budgets))
    ):
        gates: dict[str, Any] = {}
        for gate_index, (name, score) in enumerate(
            (
                ("uncertainty", uncertainty_score),
                ("value", value_score),
            )
        ):
            gates[name] = _evaluate_gate(
                name,
                score,
                labels=labels,
                short_scores=short_scores,
                refined_scores=refined_scores,
                calibration_mask=calibration_mask,
                test_mask=test_mask,
                cluster_codes=test_cluster_codes,
                n_clusters=n_test_clusters,
                activation_rate=budget,
            )
        gates["random"] = _random_gate_summary(
            labels=labels,
            short_scores=short_scores,
            refined_scores=refined_scores,
            calibration_mask=calibration_mask,
            test_mask=test_mask,
            activation_rate=budget,
            repeats=args.random_repeats,
            random_seed=args.random_seed + 10_000 + budget_index,
        )
        gate_budget_rows.append(
            {
                "calibration_activation_target": budget,
                **gates,
            }
        )
        if math.isclose(
            budget,
            float(args.primary_activation),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            primary_gates = gates
    if primary_gates is None:
        raise RuntimeError("primary activation was not evaluated")

    paired = paired_speaker_cluster_bootstrap(
        primary_gates["value"]["cluster_counts"],
        primary_gates["uncertainty"]["cluster_counts"],
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    for name in ("uncertainty", "value"):
        primary_gates[name]["cluster_bootstrap"] = paired
    primary_gates["value_minus_uncertainty"] = paired
    if "random" in primary_gates:
        primary_gates.pop("random")
        primary_gates["random"] = gate_budget_rows[
            [
                float(row["calibration_activation_target"])
                for row in gate_budget_rows
            ].index(float(args.primary_activation))
        ]["random"]

    value_delta = float(
        paired["difference_net_utility_per_selected"]
    )
    value_ci_low = paired[
        "difference_net_utility_per_selected_ci95_low"
    ]
    criteria = {
        "value_net_utility_above_uncertainty": value_delta > 0.0,
        "paired_cluster_ci_excludes_zero": bool(
            value_ci_low is not None and value_ci_low > 0.0
        ),
    }
    if all(criteria.values()):
        status = "GO"
        interpretation = (
            "Short-side observables predict additional signed temporal "
            "refinement value beyond uncertainty alone."
        )
    else:
        status = "UNCERTAINTY_PROXY"
        interpretation = (
            "The value score does not establish a significant advantage "
            "over uncertainty at the primary budget. Keep the simple "
            "uncertainty gate as the honest low-cost proxy; this is not an "
            "A-wide NO-GO."
        )

    map_summary = temporal_value_map(
        features,
        values,
        calibration_mask,
        bins=args.map_bins,
    )
    protocol = _load_protocol(args.results_json)
    lookback_frames = int(
        protocol.get("lookback_frames", args.lookback_frames)
    )
    summary: dict[str, Any] = {
        "protocol": {
            "predictions": str(args.predictions),
            "results_json": str(args.results_json),
            "adaptive_checkpoint": protocol.get("adaptive_checkpoint"),
            "lookback_frames": lookback_frames,
            "lookback_seconds": (
                lookback_frames * FRAME_HOP / 16_000.0
            ),
            "window_frames": int(args.window_frames),
            "window_seconds": (
                int(args.window_frames) * FRAME_HOP / 16_000.0
            ),
            "feature_names": list(FEATURE_NAMES),
            "calibration_speaker_count": int(
                np.unique(speaker_ids[calibration_mask]).size
            ),
            "test_speaker_count": int(
                np.unique(speaker_ids[test_mask]).size
            ),
            "primary_activation": float(args.primary_activation),
            "activation_budgets": [
                float(row["calibration_activation_target"])
                for row in gate_budget_rows
            ],
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "random_repeats": int(args.random_repeats),
        },
        "signed_value": {
            "calibration": _signed_value_summary(
                values,
                calibration_mask,
            ),
            "test": _signed_value_summary(values, test_mask),
        },
        "value_model": model_metadata,
        "primary": primary_gates,
        "activation_budget_rows": gate_budget_rows,
        "temporal_value_map": map_summary,
        "go_assessment": {
            "status": status,
            "criteria": criteria,
            "interpretation": interpretation,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "a10_value_predictability.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )
    _write_value_map_csv(
        map_summary,
        args.output_dir / "a10_temporal_value_map.csv",
    )
    _write_value_map_plot(
        map_summary,
        args.output_dir / "a10_temporal_value_map.png",
    )
    _write_gate_plot(
        gate_budget_rows,
        args.output_dir / "a10_gate_comparison.png",
    )
    _write_markdown(
        summary,
        args.output_dir / "a10_value_predictability.md",
    )
    print(f"A10 artifacts written to {args.output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze A10 temporal-value predictability."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--activation-budgets",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--primary-activation", type=float, default=0.05)
    parser.add_argument(
        "--window-frames",
        type=int,
        default=25,
        help="Causal frame window for short-term variance/switch rate.",
    )
    parser.add_argument("--map-bins", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--random-seed", type=int, default=20260920)
    parser.add_argument(
        "--lookback-frames",
        type=int,
        default=384,
        help="Fallback refinement span when results.json is unavailable.",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not 0.0 < float(args.primary_activation) <= 1.0:
        parser.error("--primary-activation must be in (0, 1]")
    if not args.activation_budgets or any(
        not 0.0 < float(value) <= 1.0
        for value in args.activation_budgets
    ):
        parser.error("--activation-budgets values must be in (0, 1]")
    if not any(
        math.isclose(
            float(value),
            float(args.primary_activation),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value in args.activation_budgets
    ):
        parser.error(
            "--primary-activation must also appear in --activation-budgets"
        )
    if args.window_frames <= 0:
        parser.error("--window-frames must be positive")
    if args.map_bins < 2:
        parser.error("--map-bins must be at least 2")
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")
    if args.random_repeats <= 0:
        parser.error("--random-repeats must be positive")
    if args.lookback_frames <= 0:
        parser.error("--lookback-frames must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
