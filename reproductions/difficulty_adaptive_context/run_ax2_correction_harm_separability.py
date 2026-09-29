# -*- coding: utf-8 -*-
"""Measure correction-versus-harm separability with the frozen AX feature ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from reproductions.difficulty_adaptive_context.analyze_a10_value_predictability import (
    signed_temporal_value,
    utterance_segments,
)
from reproductions.difficulty_adaptive_context.run_ax1_value_predictability import (
    Ax1Config,
    _validate_split,
    feature_blocks,
    load_frozen_bundle,
    validate_frozen_inputs,
)
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ax2_protocol.json"
)
INFORMATION_LEVELS = ("X0", "X1", "X2", "X3")
TARGET_NAMES = ("correction", "harm")
PRIMARY_MODEL = "X3_logistic"
BASELINE_MODEL = "X0_logistic"


@dataclass(frozen=True)
class Ax2Config:
    protocol_path: Path
    payload: Mapping[str, Any]

    def resolve(self, relative_path: str) -> Path:
        return REPO_ROOT / relative_path

    @property
    def output_dir(self) -> Path:
        return self.resolve(str(self.payload["outputs"]["directory"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_PROTOCOL) -> Ax2Config:
    if not path.exists():
        raise FileNotFoundError(f"AX2 protocol not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("protocol_id")) != "A-AX2-v1":
        raise ValueError("unexpected AX2 protocol id")
    return Ax2Config(protocol_path=path, payload=payload)


def verify_hash(
    path: Path,
    expected_sha256: str,
    *,
    verify: bool,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"frozen input not found: {path}")
    if not verify:
        return "not_verified"
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise ValueError(
            f"hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def load_ax2_context(
    config: Ax2Config,
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    inputs = config.payload["inputs"]
    ax1_protocol_path = config.resolve(
        str(inputs["ax1_protocol"]["path"])
    )
    ax1_protocol_sha = verify_hash(
        ax1_protocol_path,
        str(inputs["ax1_protocol"]["sha256"]),
        verify=verify_hashes,
    )
    with open(ax1_protocol_path, "r", encoding="utf-8") as handle:
        ax1_payload = json.load(handle)
    if str(ax1_payload.get("protocol_id")) != "A-AX1-v1":
        raise ValueError("unexpected AX1 protocol id")
    ax1_config = Ax1Config(
        protocol_path=ax1_protocol_path,
        payload=ax1_payload,
    )
    nested_hashes = validate_frozen_inputs(
        ax1_config,
        verify_hashes=verify_hashes,
    )

    feature_path = config.resolve(
        str(inputs["ax1_feature_cache"]["path"])
    )
    feature_sha = verify_hash(
        feature_path,
        str(inputs["ax1_feature_cache"]["sha256"]),
        verify=verify_hashes,
    )
    summary_path = config.resolve(str(inputs["ax1_summary"]["path"]))
    summary_sha = verify_hash(
        summary_path,
        str(inputs["ax1_summary"]["sha256"]),
        verify=verify_hashes,
    )

    bundle = load_frozen_bundle(ax1_config)
    calibration_mask, test_mask, calibration_speakers, test_speakers = (
        _validate_split(bundle, ax1_config)
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        required = {"hidden", "mfcc"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"AX1 feature cache is missing: {missing}")
        hidden = np.asarray(payload["hidden"], dtype=np.float32)
        mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
    if hidden.shape[0] != bundle["labels"].size:
        raise ValueError("AX1 feature cache does not match the frozen bundle")

    segments = utterance_segments(
        bundle["source_key"],
        bundle["noise_name"],
        bundle["condition"],
    )
    features = feature_blocks(
        bundle["short_scores"],
        hidden,
        mfcc,
        segments,
        window_frames=int(
            ax1_payload["feature_ladder"]["causal_window_frames"]
        ),
    )
    values = signed_temporal_value(
        bundle["labels"],
        bundle["short_scores"],
        bundle["full_adaptive_scores"],
    ).astype(np.int64)
    targets = binary_targets(values)
    cluster_names, cluster_codes = np.unique(
        bundle["speaker_ids"][test_mask],
        return_inverse=True,
    )
    return {
        "bundle": bundle,
        "features": features,
        "values": values,
        "targets": targets,
        "calibration_mask": calibration_mask,
        "test_mask": test_mask,
        "calibration_speakers": calibration_speakers,
        "test_speakers": test_speakers,
        "cluster_names": cluster_names,
        "cluster_codes": cluster_codes,
        "hashes": {
            "ax1_protocol": ax1_protocol_sha,
            "ax1_feature_cache": feature_sha,
            "ax1_summary": summary_sha,
            **nested_hashes,
        },
    }


def binary_targets(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if not np.all(np.isin(values, (-1, 0, 1))):
        raise ValueError("signed values must be in {-1, 0, 1}")
    return {
        "correction": values == 1,
        "harm": values == -1,
    }


def binary_classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=bool).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if target.size != probabilities.size:
        raise ValueError("target and probabilities must have equal length")
    if target.size == 0:
        raise ValueError("binary metrics require at least one frame")
    positive = int(np.count_nonzero(target))
    negative = int(target.size - positive)
    if positive == 0 or negative == 0:
        raise ValueError("binary metrics require both target classes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    prevalence = float(positive / target.size)
    auprc = float(average_precision_score(target, probabilities))
    return {
        "positive": positive,
        "negative": negative,
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(target, probabilities)),
        "auprc": auprc,
        "normalized_auprc_lift": float(
            (auprc - prevalence) / (1.0 - prevalence)
        ),
    }


def fit_binary_logistic_probability(
    features: np.ndarray,
    target: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    features = np.asarray(features)
    target = np.asarray(target, dtype=bool).reshape(-1)
    calibration_mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if features.shape[0] != target.size:
        raise ValueError("feature and target row counts differ")
    if calibration_mask.size != target.size:
        raise ValueError("calibration mask size differs from target")
    if not np.any(calibration_mask):
        raise ValueError("calibration mask is empty")
    calibration_target = target[calibration_mask]
    if np.unique(calibration_target).size != 2:
        raise ValueError("calibration target must contain both classes")
    scaler = StandardScaler()
    scaled_calibration = scaler.fit_transform(
        features[calibration_mask].astype(np.float64)
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=int(max_iter),
        tol=float(tolerance),
        class_weight="balanced",
        random_state=int(seed),
    )
    classifier.fit(scaled_calibration, calibration_target)
    scaled_all = scaler.transform(features.astype(np.float64))
    return classifier.predict_proba(scaled_all)[:, 1].astype(np.float64)


class _BinaryMlp(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def fit_binary_mlp_probability(
    features: np.ndarray,
    target: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target, dtype=bool).reshape(-1)
    calibration_mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if features.shape[0] != target.size:
        raise ValueError("feature and target row counts differ")
    if calibration_mask.size != target.size:
        raise ValueError("calibration mask size differs from target")
    calibration_target = target[calibration_mask]
    if np.unique(calibration_target).size != 2:
        raise ValueError("calibration target must contain both classes")

    scaler = StandardScaler()
    scaled_calibration = scaler.fit_transform(
        features[calibration_mask].astype(np.float64)
    ).astype(np.float32)
    scaled_all = scaler.transform(features.astype(np.float64)).astype(
        np.float32
    )

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = _BinaryMlp(scaled_all.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    positive = int(np.count_nonzero(calibration_target))
    negative = int(calibration_target.size - positive)
    positive_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    calibration_features = torch.from_numpy(scaled_calibration)
    calibration_labels = torch.from_numpy(
        calibration_target.astype(np.float32)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for _ in range(int(epochs)):
        order = torch.randperm(
            calibration_features.shape[0],
            generator=generator,
        )
        for start in range(0, order.numel(), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            batch_features = calibration_features[indices].to(
                device,
                non_blocking=True,
            )
            batch_labels = calibration_labels[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    probabilities = np.empty(scaled_all.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, scaled_all.shape[0], int(batch_size)):
            batch = torch.from_numpy(
                scaled_all[start : start + int(batch_size)]
            ).to(device, non_blocking=True)
            probabilities[start : start + batch.shape[0]] = (
                torch.sigmoid(model(batch)).cpu().numpy().astype(np.float64)
            )
    return probabilities


def prepare_binary_curve(
    target: np.ndarray,
    probabilities: np.ndarray,
    cluster_codes: np.ndarray,
) -> dict[str, np.ndarray | int | dict[str, Any]]:
    target = np.asarray(target, dtype=bool).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (
        target.size == probabilities.size == cluster_codes.size
    ):
        raise ValueError("curve arrays must have equal lengths")
    if target.size == 0:
        raise ValueError("curve requires at least one frame")
    n_clusters = int(cluster_codes.max()) + 1
    order = np.argsort(-probabilities, kind="stable")
    target_sorted = target[order]
    scores_sorted = probabilities[order]
    cluster_sorted = cluster_codes[order]
    change = np.empty(scores_sorted.size, dtype=bool)
    change[0] = True
    if scores_sorted.size > 1:
        change[1:] = scores_sorted[1:] != scores_sorted[:-1]
    group_ids = (np.cumsum(change) - 1).astype(np.int64)
    n_groups = int(group_ids[-1]) + 1
    positive_flat = np.bincount(
        cluster_sorted[target_sorted] * n_groups
        + group_ids[target_sorted],
        minlength=n_clusters * n_groups,
    )
    negative_flat = np.bincount(
        cluster_sorted[~target_sorted] * n_groups
        + group_ids[~target_sorted],
        minlength=n_clusters * n_groups,
    )
    return {
        "positive_by_cluster": positive_flat.reshape(
            n_clusters,
            n_groups,
        ),
        "negative_by_cluster": negative_flat.reshape(
            n_clusters,
            n_groups,
        ),
        "n_clusters": n_clusters,
        "n_groups": n_groups,
        "point_metrics": binary_classification_metrics(
            target,
            probabilities,
        ),
    }


def bootstrap_cluster_counts(
    n_clusters: int,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(
        int(n_clusters),
        1.0 / float(n_clusters),
        dtype=np.float64,
    )
    return rng.multinomial(
        int(n_clusters),
        probabilities,
        size=int(repeats),
    ).astype(np.int64)


def bootstrap_curve_metrics(
    prepared: Mapping[str, Any],
    cluster_counts: np.ndarray,
    *,
    group_chunk_size: int = 8192,
    weight_chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    positive_by_cluster = np.asarray(
        prepared["positive_by_cluster"],
        dtype=np.float64,
    )
    negative_by_cluster = np.asarray(
        prepared["negative_by_cluster"],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(cluster_counts, dtype=np.float64)
    if positive_by_cluster.shape != negative_by_cluster.shape:
        raise ValueError("positive and negative curve shapes differ")
    if cluster_counts.ndim != 2:
        raise ValueError("cluster_counts must be two-dimensional")
    if cluster_counts.shape[1] != positive_by_cluster.shape[0]:
        raise ValueError("cluster_counts does not match curve clusters")

    n_repeats = cluster_counts.shape[0]
    n_groups = positive_by_cluster.shape[1]
    auroc = np.full(n_repeats, np.nan, dtype=np.float64)
    auprc = np.full(n_repeats, np.nan, dtype=np.float64)
    normalized_lift = np.full(n_repeats, np.nan, dtype=np.float64)
    for weight_start in range(0, n_repeats, int(weight_chunk_size)):
        weight_end = min(
            weight_start + int(weight_chunk_size),
            n_repeats,
        )
        weights = cluster_counts[weight_start:weight_end]
        batch_size = weights.shape[0]
        negative_total = weights @ np.sum(
            negative_by_cluster,
            axis=1,
        )
        prefix_positive = np.zeros(batch_size, dtype=np.float64)
        prefix_negative = np.zeros(batch_size, dtype=np.float64)
        average_precision_numerator = np.zeros(
            batch_size,
            dtype=np.float64,
        )
        rank_sum = np.zeros(batch_size, dtype=np.float64)
        for group_start in range(0, n_groups, int(group_chunk_size)):
            group_end = min(
                group_start + int(group_chunk_size),
                n_groups,
            )
            group_positive = (
                weights
                @ positive_by_cluster[:, group_start:group_end]
            )
            group_negative = (
                weights
                @ negative_by_cluster[:, group_start:group_end]
            )
            cumulative_positive = (
                np.cumsum(group_positive, axis=1)
                + prefix_positive[:, None]
            )
            cumulative_negative = (
                np.cumsum(group_negative, axis=1)
                + prefix_negative[:, None]
            )
            cumulative_total = cumulative_positive + cumulative_negative
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive),
                where=cumulative_total > 0.0,
            )
            average_precision_numerator += np.sum(
                group_positive * precision,
                axis=1,
            )
            group_total = group_positive + group_negative
            negative_after = (
                negative_total[:, None]
                - cumulative_negative
            )
            rank_sum += np.sum(
                group_positive
                * (negative_after + 0.5 * group_negative),
                axis=1,
            )
            prefix_positive = cumulative_positive[:, -1]
            prefix_negative = cumulative_negative[:, -1]

        positive_total = prefix_positive
        negative_total = prefix_negative
        valid = (positive_total > 0.0) & (negative_total > 0.0)
        batch_auroc = np.full(batch_size, np.nan, dtype=np.float64)
        batch_auroc[valid] = (
            rank_sum[valid]
            / (positive_total[valid] * negative_total[valid])
        )
        batch_auprc = np.full(batch_size, np.nan, dtype=np.float64)
        batch_auprc[valid] = (
            average_precision_numerator[valid] / positive_total[valid]
        )
        sample_prevalence = np.full(batch_size, np.nan, dtype=np.float64)
        sample_prevalence[valid] = (
            positive_total[valid]
            / (positive_total[valid] + negative_total[valid])
        )
        batch_lift = np.full(batch_size, np.nan, dtype=np.float64)
        batch_lift[valid] = (
            batch_auprc[valid] - sample_prevalence[valid]
        ) / (1.0 - sample_prevalence[valid])
        auroc[weight_start:weight_end] = batch_auroc
        auprc[weight_start:weight_end] = batch_auprc
        normalized_lift[weight_start:weight_end] = batch_lift
    return {
        "auroc": auroc,
        "auprc": auprc,
        "normalized_auprc_lift": normalized_lift,
    }


def percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_primary_metrics(
    prepared: Mapping[str, Any],
    cluster_counts: np.ndarray,
) -> dict[str, Any]:
    bootstrap = bootstrap_curve_metrics(prepared, cluster_counts)
    point = dict(prepared["point_metrics"])
    result: dict[str, Any] = {
        **point,
        "bootstrap_repeats": int(cluster_counts.shape[0]),
    }
    for name in ("auroc", "auprc", "normalized_auprc_lift"):
        low, high = percentile_interval(bootstrap[name])
        result[f"{name}_ci95_low"] = low
        result[f"{name}_ci95_high"] = high
        result[f"{name}_valid_bootstrap_repeats"] = int(
            np.count_nonzero(np.isfinite(bootstrap[name]))
        )
    return result


def paired_curve_difference(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    cluster_counts: np.ndarray,
) -> dict[str, Any]:
    first_bootstrap = bootstrap_curve_metrics(first, cluster_counts)
    second_bootstrap = bootstrap_curve_metrics(second, cluster_counts)
    first_point = first["point_metrics"]
    second_point = second["point_metrics"]
    result: dict[str, Any] = {
        "first_auroc": float(first_point["auroc"]),
        "second_auroc": float(second_point["auroc"]),
        "auroc_difference": float(
            first_point["auroc"] - second_point["auroc"]
        ),
        "first_normalized_auprc_lift": float(
            first_point["normalized_auprc_lift"]
        ),
        "second_normalized_auprc_lift": float(
            second_point["normalized_auprc_lift"]
        ),
        "normalized_auprc_lift_difference": float(
            first_point["normalized_auprc_lift"]
            - second_point["normalized_auprc_lift"]
        ),
    }
    for metric in ("auroc", "normalized_auprc_lift"):
        differences = (
            first_bootstrap[metric] - second_bootstrap[metric]
        )
        low, high = percentile_interval(differences)
        result[f"{metric}_difference_ci95_low"] = low
        result[f"{metric}_difference_ci95_high"] = high
        result[f"{metric}_valid_bootstrap_repeats"] = int(
            np.count_nonzero(np.isfinite(differences))
        )
    return result


def _primary_row(
    rows: Sequence[Mapping[str, Any]],
    target: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if str(row["target"]) == target
        and str(row["model"]) == PRIMARY_MODEL
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one primary row for target {target}")
    return matches[0]


def _robust_direction(row: Mapping[str, Any]) -> bool:
    auroc_low = row.get("auroc_ci95_low")
    lift_low = row.get("normalized_auprc_lift_ci95_low")
    return (
        auroc_low is not None
        and lift_low is not None
        and float(auroc_low) > 0.5
        and float(lift_low) > 0.0
    )


def assess_ax2(
    rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    correction = _primary_row(rows, "correction")
    harm = _primary_row(rows, "harm")
    correction_robust = _robust_direction(correction)
    harm_robust = _robust_direction(harm)
    direction = paired["correction_vs_harm"]
    direction_low = direction.get("auroc_difference_ci95_low")
    direction_high = direction.get("auroc_difference_ci95_high")
    lift_low = direction.get(
        "normalized_auprc_lift_difference_ci95_low"
    )
    lift_high = direction.get(
        "normalized_auprc_lift_difference_ci95_high"
    )
    correction_larger = (
        direction_low is not None
        and lift_low is not None
        and float(direction_low) > 0.0
        and float(lift_low) > 0.0
    )
    harm_larger = (
        direction_high is not None
        and lift_high is not None
        and float(direction_high) < 0.0
        and float(lift_high) < 0.0
    )

    if correction_robust and harm_robust:
        status = "BIDIRECTIONAL_SEPARABILITY"
        interpretation = (
            "Both correction and harm contain robust ranking signal. "
            "AX2 is diagnostic and does not authorize a new router."
        )
    elif correction_robust and correction_larger:
        status = "ASYMMETRIC_SEPARABILITY"
        interpretation = (
            "Correction is robustly separable while harm is not; the paired "
            "direction difference excludes zero with correction easier."
        )
    elif harm_robust and harm_larger:
        status = "ASYMMETRIC_SEPARABILITY"
        interpretation = (
            "Harm is robustly separable while correction is not; the paired "
            "direction difference excludes zero with harm easier."
        )
    elif correction_robust or harm_robust:
        status = "UNCERTAIN_ROBUST_ASYMMETRY"
        interpretation = (
            "Only one direction passes robust separability, but the paired "
            "direction difference does not establish asymmetry."
        )
    else:
        status = "NO_ROBUST_DIRECTION_SEPARABILITY"
        interpretation = (
            "Neither correction nor harm passes the frozen robust-direction "
            "rule. Preserve the diagnostic result and continue to AX3."
        )
    return {
        "status": status,
        "interpretation": interpretation,
        "primary_predictor": PRIMARY_MODEL,
        "criteria": {
            "correction_robustly_separable": correction_robust,
            "harm_robustly_separable": harm_robust,
            "correction_larger_with_paired_ci_excluding_zero": (
                correction_larger
            ),
            "harm_larger_with_paired_ci_excluding_zero": harm_larger,
        },
        "correction": dict(correction),
        "harm": dict(harm),
        "correction_vs_harm": dict(direction),
    }


def evaluate_ax2(
    config: Ax2Config,
    *,
    context: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    features = context["features"]
    targets = context["targets"]
    calibration_mask = np.asarray(context["calibration_mask"], dtype=bool)
    test_mask = np.asarray(context["test_mask"], dtype=bool)
    model_config = config.payload["models"]
    logistic_config = model_config["logistic"]
    mlp_config = model_config["mlp_ceiling_probe"]

    rows: list[dict[str, Any]] = []
    primary_probabilities: dict[str, dict[str, np.ndarray]] = {
        name: {}
        for name in TARGET_NAMES
    }
    for target_name in TARGET_NAMES:
        target = np.asarray(targets[target_name], dtype=bool)
        for level in INFORMATION_LEVELS:
            print(
                f"fitting {target_name} {level} logistic",
                flush=True,
            )
            logistic_probability = fit_binary_logistic_probability(
                features[level],
                target,
                calibration_mask,
                seed=int(logistic_config["random_state"]),
                max_iter=int(logistic_config["max_iter"]),
                tolerance=float(logistic_config["tol"]),
            )
            logistic_metrics = binary_classification_metrics(
                target[test_mask],
                logistic_probability[test_mask],
            )
            rows.append(
                {
                    "target": target_name,
                    "information_level": level,
                    "model_family": "logistic",
                    "model": f"{level}_logistic",
                    **logistic_metrics,
                }
            )
            if level in {"X0", "X3"}:
                primary_probabilities[target_name][
                    f"{level}_logistic"
                ] = logistic_probability

            print(
                f"fitting {target_name} {level} MLP",
                flush=True,
            )
            mlp_probability = fit_binary_mlp_probability(
                features[level],
                target,
                calibration_mask,
                seed=int(mlp_config["torch_seed"]),
                learning_rate=float(mlp_config["learning_rate"]),
                weight_decay=float(mlp_config["weight_decay"]),
                batch_size=int(mlp_config["batch_size"]),
                epochs=int(mlp_config["epochs"]),
                device=device,
            )
            mlp_metrics = binary_classification_metrics(
                target[test_mask],
                mlp_probability[test_mask],
            )
            rows.append(
                {
                    "target": target_name,
                    "information_level": level,
                    "model_family": "mlp",
                    "model": f"{level}_mlp",
                    **mlp_metrics,
                }
            )

    bootstrap_config = config.payload["bootstrap"]
    cluster_counts = bootstrap_cluster_counts(
        int(context["cluster_names"].size),
        repeats=int(bootstrap_config["repeats"]),
        seed=int(bootstrap_config["seed"]),
    )
    prepared: dict[str, dict[str, Mapping[str, Any]]] = {
        name: {}
        for name in TARGET_NAMES
    }
    primary_bootstrap: dict[str, Any] = {}
    for target_name in TARGET_NAMES:
        target = np.asarray(targets[target_name], dtype=bool)
        for model_name in (BASELINE_MODEL, PRIMARY_MODEL):
            prepared[target_name][model_name] = prepare_binary_curve(
                target[test_mask],
                primary_probabilities[target_name][model_name][test_mask],
                np.asarray(context["cluster_codes"], dtype=np.int64),
            )
        primary_bootstrap[target_name] = bootstrap_primary_metrics(
            prepared[target_name][PRIMARY_MODEL],
            cluster_counts,
        )
        row = _primary_row(rows, target_name)
        row.update(primary_bootstrap[target_name])

    paired = {
        target_name: paired_curve_difference(
            prepared[target_name][PRIMARY_MODEL],
            prepared[target_name][BASELINE_MODEL],
            cluster_counts,
        )
        for target_name in TARGET_NAMES
    }
    paired["correction_vs_harm"] = paired_curve_difference(
        prepared["correction"][PRIMARY_MODEL],
        prepared["harm"][PRIMARY_MODEL],
        cluster_counts,
    )
    assessment = assess_ax2(rows, paired)
    return {
        "protocol": {
            "path": str(config.protocol_path),
            "id": config.payload["protocol_id"],
            "primary_predictor": PRIMARY_MODEL,
        },
        "split": {
            "calibration_speakers": list(
                context["calibration_speakers"]
            ),
            "test_speakers": list(context["test_speakers"]),
            "calibration_frames": int(np.count_nonzero(calibration_mask)),
            "test_frames": int(np.count_nonzero(test_mask)),
            "speaker_clusters": int(context["cluster_names"].size),
        },
        "targets": {
            target_name: {
                "test_positive": int(
                    np.count_nonzero(targets[target_name][test_mask])
                ),
                "test_negative": int(
                    np.count_nonzero(
                        ~np.asarray(targets[target_name], dtype=bool)[
                            test_mask
                        ]
                    )
                ),
            }
            for target_name in TARGET_NAMES
        },
        "rows": rows,
        "bootstrap": {
            "repeats": int(cluster_counts.shape[0]),
            "seed": int(bootstrap_config["seed"]),
            "unit": "speaker_cluster",
            "primary": primary_bootstrap,
        },
        "paired": paired,
        "assessment": assessment,
    }


def write_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "information_level",
        "model_family",
        "model",
        "positive",
        "negative",
        "prevalence",
        "auroc",
        "auprc",
        "normalized_auprc_lift",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "normalized_auprc_lift_ci95_low",
        "normalized_auprc_lift_ci95_high",
        "bootstrap_repeats",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.3f}%"


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    rows = list(summary["rows"])
    lines = [
        "# AX2 Correction-versus-Harm Separability",
        "",
        "## Frozen Setup",
        "",
        f"- Calibration speakers/frames: "
        f"{len(summary['split']['calibration_speakers'])}/"
        f"{summary['split']['calibration_frames']:,}",
        f"- Test speakers/frames: "
        f"{len(summary['split']['test_speakers'])}/"
        f"{summary['split']['test_frames']:,}",
        f"- Primary predictor: **{summary['protocol']['primary_predictor']}**.",
        "- Correction target: signed value `+1` versus `{0, -1}`.",
        "- Harm target: signed value `-1` versus `{0, +1}`.",
        "- All model fitting uses calibration speakers only.",
        "",
        "## Point Predictability",
        "",
        "| Target | Model | Prevalence | AUROC | AUPRC | Normalized "
        "AUPRC lift |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for target_name in TARGET_NAMES:
        for row in rows:
            if str(row["target"]) != target_name:
                continue
            lines.append(
                f"| {target_name} | {row['model']} | "
                f"{_percent(row['prevalence'])} | "
                f"{_number(row['auroc'])} | "
                f"{_number(row['auprc'])} | "
                f"{_number(row['normalized_auprc_lift'])} |"
            )
    lines.extend(
        [
            "",
            "## Primary Speaker-Cluster Bootstrap",
            "",
            "| Target | AUROC 95% CI | Normalized AUPRC lift 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for target_name in TARGET_NAMES:
        row = summary["bootstrap"]["primary"][target_name]
        lines.append(
            f"| {target_name} | "
            f"[{_number(row['auroc_ci95_low'])}, "
            f"{_number(row['auroc_ci95_high'])}] | "
            f"[{_number(row['normalized_auprc_lift_ci95_low'])}, "
            f"{_number(row['normalized_auprc_lift_ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Paired Differences",
            "",
            "| Comparison | AUROC difference 95% CI | Normalized AUPRC "
            "lift difference 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for comparison, row in summary["paired"].items():
        lines.append(
            f"| {comparison} | "
            f"[{_number(row['auroc_difference_ci95_low'])}, "
            f"{_number(row['auroc_difference_ci95_high'])}] | "
            f"[{_number(row['normalized_auprc_lift_difference_ci95_low'])}, "
            f"{_number(row['normalized_auprc_lift_difference_ci95_high'])}] |"
        )
    assessment = summary["assessment"]
    lines.extend(
        [
            "",
            "## AX2 Assessment",
            "",
            f"- Status: **{assessment['status']}**.",
            f"- Interpretation: {assessment['interpretation']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.protocol)
    context = load_ax2_context(
        config,
        verify_hashes=not args.skip_hash_check,
    )
    device = resolve_device(args.device)
    summary = evaluate_ax2(
        config,
        context=context,
        device=device,
    )
    summary["inputs"] = context["hashes"]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config.output_dir / str(config.payload["outputs"]["summary"]),
        summary,
    )
    write_metrics_csv(
        summary["rows"],
        config.output_dir / str(config.payload["outputs"]["metrics_table"]),
    )
    write_report(
        summary,
        config.output_dir / str(config.payload["outputs"]["report"]),
    )
    print(json.dumps(summary["assessment"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
