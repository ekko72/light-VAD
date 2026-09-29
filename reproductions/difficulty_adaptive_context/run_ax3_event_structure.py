# -*- coding: utf-8 -*-
"""Characterize where signed long-context value occurs in event structure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reproductions.difficulty_adaptive_context.analyze_a10_value_predictability import (
    signed_temporal_value,
    utterance_segments,
)
from reproductions.difficulty_adaptive_context.run_a13_streaming_audit import (
    find_runs,
)
from reproductions.difficulty_adaptive_context.run_ax1_value_predictability import (
    Ax1Config,
    _validate_split,
    load_frozen_bundle,
    sha256_file,
    validate_frozen_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ax3_protocol.json"
)
METRICS = ("p_positive", "p_negative", "mean_value")


@dataclass(frozen=True)
class Ax3Config:
    protocol_path: Path
    payload: Mapping[str, Any]

    def resolve(self, relative_path: str) -> Path:
        return REPO_ROOT / relative_path

    @property
    def output_dir(self) -> Path:
        return self.resolve(str(self.payload["outputs"]["directory"]))


def load_config(path: Path = DEFAULT_PROTOCOL) -> Ax3Config:
    if not path.exists():
        raise FileNotFoundError(f"AX3 protocol not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("protocol_id")) != "A-AX3-v1":
        raise ValueError("unexpected AX3 protocol id")
    return Ax3Config(protocol_path=path, payload=payload)


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


def load_ax3_context(
    config: Ax3Config,
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
    bundle = load_frozen_bundle(ax1_config)
    calibration_mask, test_mask, calibration_speakers, test_speakers = (
        _validate_split(bundle, ax1_config)
    )
    segments = utterance_segments(
        bundle["source_key"],
        bundle["noise_name"],
        bundle["condition"],
    )
    values = signed_temporal_value(
        bundle["labels"],
        bundle["short_scores"],
        bundle["full_adaptive_scores"],
    ).astype(np.int64)
    return {
        "ax1_config": ax1_config,
        "bundle": bundle,
        "values": values,
        "segments": segments,
        "calibration_mask": calibration_mask,
        "test_mask": test_mask,
        "calibration_speakers": calibration_speakers,
        "test_speakers": test_speakers,
        "hashes": {
            "ax1_protocol": ax1_protocol_sha,
            **nested_hashes,
        },
    }


def _distance_bin(distance: int) -> str:
    if distance <= 0:
        return "0"
    if distance <= 2:
        return "1-2"
    if distance <= 5:
        return "3-5"
    if distance <= 10:
        return "6-10"
    if distance <= 20:
        return "11-20"
    if distance <= 50:
        return "21-50"
    return "51+"


def _run_length_bin(length: int) -> str:
    if length <= 2:
        return "1-2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    if length <= 50:
        return "21-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    return "201+"


def _transition_age_bin(age: int) -> str:
    if age <= 0:
        return "0"
    if age <= 2:
        return "1-2"
    if age <= 5:
        return "3-5"
    if age <= 10:
        return "6-10"
    if age <= 25:
        return "11-25"
    return "26+"


def _uncertainty_duration_bin(duration: int) -> str:
    if duration <= 0:
        return "0"
    if duration <= 2:
        return "1-2"
    if duration <= 5:
        return "3-5"
    if duration <= 10:
        return "6-10"
    if duration <= 25:
        return "11-25"
    return "26+"


def build_event_structure(
    labels: np.ndarray,
    short_scores: np.ndarray,
    segments: Sequence[tuple[int, int]],
    *,
    decision_threshold: float,
    uncertainty_threshold: float,
) -> dict[str, np.ndarray]:
    """Construct the six frozen event-structure dimensions."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    if labels.size != short_scores.size:
        raise ValueError("labels and short_scores must have equal length")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(short_scores)):
        raise ValueError("short_scores must be finite")
    if not 0.0 < float(decision_threshold) < 1.0:
        raise ValueError("decision_threshold must be in (0, 1)")
    if float(uncertainty_threshold) < 0.0:
        raise ValueError("uncertainty_threshold must be non-negative")
    if not segments:
        raise ValueError("at least one utterance segment is required")

    onset = np.full(labels.size, "not_in_speech", dtype=object)
    offset = np.full(labels.size, "pre_first_speech", dtype=object)
    speech_run_length = np.full(labels.size, "not_in_speech", dtype=object)
    silence_run_length = np.full(labels.size, "in_speech", dtype=object)
    transition = np.full(labels.size, "none", dtype=object)
    uncertainty_duration = np.empty(labels.size, dtype=object)

    for segment_start, segment_end in segments:
        if not 0 <= int(segment_start) < int(segment_end) <= labels.size:
            raise ValueError(f"invalid utterance segment: {segment_start}:{segment_end}")
        start = int(segment_start)
        end = int(segment_end)
        local_labels = labels[start:end]
        local_scores = short_scores[start:end]

        speech_runs = find_runs(local_labels == 1)
        for run_start, run_end in speech_runs:
            length = int(run_end - run_start)
            length_label = _run_length_bin(length)
            for local_index in range(run_start, run_end):
                onset[start + local_index] = _distance_bin(
                    local_index - run_start
                )
                speech_run_length[start + local_index] = length_label
                offset[start + local_index] = "in_speech"
            if run_end < local_labels.size:
                for local_index in range(run_end, local_labels.size):
                    if local_labels[local_index] == 1:
                        break
                    offset[start + local_index] = _distance_bin(
                        local_index - run_end
                    )

        for run_start, run_end in find_runs(local_labels == 0):
            length_label = _run_length_bin(int(run_end - run_start))
            silence_run_length[
                start + run_start : start + run_end
            ] = length_label

        predictions = local_scores >= float(decision_threshold)
        last_switch: int | None = None
        for local_index in range(local_scores.size):
            if (
                local_index > 0
                and predictions[local_index] != predictions[local_index - 1]
            ):
                last_switch = local_index
            if last_switch is not None:
                transition[start + local_index] = _transition_age_bin(
                    local_index - last_switch
                )

        uncertain = (
            np.abs(local_scores - 0.5)
            <= float(uncertainty_threshold) + 1e-12
        )
        streak = 0
        for local_index, is_uncertain in enumerate(uncertain):
            streak = streak + 1 if bool(is_uncertain) else 0
            uncertainty_duration[start + local_index] = (
                _uncertainty_duration_bin(streak)
            )

    result = {
        "onset_distance": onset.astype(str),
        "offset_distance": offset.astype(str),
        "speech_run_length": speech_run_length.astype(str),
        "silence_run_length": silence_run_length.astype(str),
        "recent_posterior_transition": transition.astype(str),
        "recent_uncertainty_duration": uncertainty_duration.astype(str),
    }
    if any(values.size != labels.size for values in result.values()):
        raise RuntimeError("event-structure outputs are misaligned")
    return result


def bootstrap_cluster_weights(
    n_clusters: int,
    *,
    repeats: int,
    seed: int,
) -> np.ndarray:
    """Return one shared cluster-resample weight matrix of shape C x R."""
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(int(seed))
    weights = np.empty((int(n_clusters), int(repeats)), dtype=np.int64)
    for repeat in range(int(repeats)):
        sampled = rng.integers(0, int(n_clusters), size=int(n_clusters))
        weights[:, repeat] = np.bincount(
            sampled,
            minlength=int(n_clusters),
        )
    return weights


def _cluster_value_counts(
    mask: np.ndarray,
    values: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> dict[str, np.ndarray]:
    selected_mask = np.asarray(mask, dtype=bool).reshape(-1)
    selected_values = np.asarray(values, dtype=np.int64).reshape(-1)
    selected_clusters = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (
        selected_mask.size
        == selected_values.size
        == selected_clusters.size
    ):
        raise ValueError("cluster value arrays must have equal lengths")
    return {
        "frames": np.bincount(
            selected_clusters[selected_mask],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "corrections": np.bincount(
            selected_clusters[selected_mask & (selected_values == 1)],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "harms": np.bincount(
            selected_clusters[selected_mask & (selected_values == -1)],
            minlength=int(n_clusters),
        ).astype(np.int64),
    }


def _metric_point(counts: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    frames = int(np.sum(counts["frames"]))
    corrections = int(np.sum(counts["corrections"]))
    harms = int(np.sum(counts["harms"]))
    if frames == 0:
        return {
            "p_positive": None,
            "p_negative": None,
            "mean_value": None,
        }
    return {
        "p_positive": float(corrections / frames),
        "p_negative": float(harms / frames),
        "mean_value": float((corrections - harms) / frames),
    }


def _percentile_interval(
    values: np.ndarray,
) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_metrics(
    counts: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, float | None], dict[str, float | None]]:
    frames = np.asarray(counts["frames"], dtype=np.int64)
    corrections = np.asarray(counts["corrections"], dtype=np.int64)
    harms = np.asarray(counts["harms"], dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != frames.size:
        raise ValueError("bootstrap weights do not match the cluster counts")
    sampled_frames = frames @ weights
    sampled_corrections = corrections @ weights
    sampled_harms = harms @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        p_positive = np.where(
            sampled_frames > 0,
            sampled_corrections / sampled_frames,
            np.nan,
        )
        p_negative = np.where(
            sampled_frames > 0,
            sampled_harms / sampled_frames,
            np.nan,
        )
        mean_value = np.where(
            sampled_frames > 0,
            (sampled_corrections - sampled_harms) / sampled_frames,
            np.nan,
        )

    point = _metric_point(counts)
    intervals: dict[str, float | None] = {}
    for name, replicates in (
        ("p_positive", p_positive),
        ("p_negative", p_negative),
        ("mean_value", mean_value),
    ):
        low, high = _percentile_interval(replicates)
        intervals[f"{name}_ci95_low"] = low
        intervals[f"{name}_ci95_high"] = high
    return point, intervals, {
        "p_positive_replicates": p_positive,
        "p_negative_replicates": p_negative,
        "mean_value_replicates": mean_value,
    }


def summarize_bins(
    dimension: str,
    bin_labels: np.ndarray,
    values: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    bins: Sequence[str],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(bin_labels, dtype=str).reshape(-1)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    clusters = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (labels.size == values.size == clusters.size):
        raise ValueError("bin labels, values, and clusters must align")
    n_clusters = int(np.max(clusters)) + 1 if clusters.size else 0
    total_frames = int(labels.size)
    total_corrections = int(np.count_nonzero(values == 1))
    total_harms = int(np.count_nonzero(values == -1))
    rows: list[dict[str, Any]] = []
    for bin_name in bins:
        mask = labels == str(bin_name)
        counts = _cluster_value_counts(
            mask,
            values,
            clusters,
            n_clusters,
        )
        point, intervals, _ = _bootstrap_metrics(counts, weights)
        frames = int(np.sum(counts["frames"]))
        corrections = int(np.sum(counts["corrections"]))
        harms = int(np.sum(counts["harms"]))
        rows.append(
            {
                "dimension": str(dimension),
                "bin": str(bin_name),
                "frames": frames,
                "speaker_clusters": int(
                    np.count_nonzero(counts["frames"] > 0)
                ),
                "corrections": corrections,
                "harms": harms,
                "p_positive": point["p_positive"],
                "p_negative": point["p_negative"],
                "mean_value": point["mean_value"],
                "frame_share": (
                    float(frames / total_frames) if total_frames else None
                ),
                "correction_share": (
                    float(corrections / total_corrections)
                    if total_corrections
                    else None
                ),
                "harm_share": (
                    float(harms / total_harms) if total_harms else None
                ),
                **intervals,
            }
        )
    return rows


def _point_metric_difference(
    focal: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, float | None]:
    focal_point = _metric_point(focal)
    reference_point = _metric_point(reference)
    return {
        name: (
            None
            if focal_point[name] is None or reference_point[name] is None
            else float(focal_point[name] - reference_point[name])
        )
        for name in METRICS
    }


def _paired_metric_differences(
    focal: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    _, _, focal_replicates = _bootstrap_metrics(focal, weights)
    _, _, reference_replicates = _bootstrap_metrics(reference, weights)
    differences: dict[str, float | None] = {}
    intervals: dict[str, float | None] = {}
    for name in METRICS:
        replicate_difference = (
            focal_replicates[f"{name}_replicates"]
            - reference_replicates[f"{name}_replicates"]
        )
        point = _point_metric_difference(focal, reference)[name]
        differences[name] = point
        low, high = _percentile_interval(replicate_difference)
        intervals[f"{name}_ci95_low"] = low
        intervals[f"{name}_ci95_high"] = high
    return differences, intervals


def _ci_excludes_zero(
    low: float | None,
    high: float | None,
) -> bool:
    return bool(
        low is not None
        and high is not None
        and (float(low) > 0.0 or float(high) < 0.0)
    )


def _sign(value: float | None) -> int:
    if value is None or float(value) == 0.0:
        return 0
    return 1 if float(value) > 0.0 else -1


def summarize_contrast(
    contrast: Mapping[str, Any],
    values: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    focal_mask: np.ndarray,
    reference_mask: np.ndarray,
    weights: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    clusters = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    focal_mask = np.asarray(focal_mask, dtype=bool).reshape(-1)
    reference_mask = np.asarray(reference_mask, dtype=bool).reshape(-1)
    if not (
        values.size
        == clusters.size
        == focal_mask.size
        == reference_mask.size
    ):
        raise ValueError("contrast arrays must align")
    if np.any(focal_mask & reference_mask):
        raise ValueError("focal and reference bins must be disjoint")
    n_clusters = int(np.max(clusters)) + 1 if clusters.size else 0
    focal = _cluster_value_counts(
        focal_mask,
        values,
        clusters,
        n_clusters,
    )
    reference = _cluster_value_counts(
        reference_mask,
        values,
        clusters,
        n_clusters,
    )
    differences, intervals = _paired_metric_differences(
        focal,
        reference,
        weights,
    )
    focal_point = _metric_point(focal)
    reference_point = _metric_point(reference)

    sensitivity: dict[str, dict[str, float | None]] = {}
    for population, population_mask in (
        ("calibration", np.asarray(calibration_mask, dtype=bool)),
        ("test", np.asarray(test_mask, dtype=bool)),
    ):
        population_mask = population_mask.reshape(-1)
        if population_mask.size != values.size:
            raise ValueError("population mask does not match values")
        sensitivity[population] = _point_metric_difference(
            _cluster_value_counts(
                focal_mask & population_mask,
                values,
                clusters,
                n_clusters,
            ),
            _cluster_value_counts(
                reference_mask & population_mask,
                values,
                clusters,
                n_clusters,
            ),
        )

    pooled_sign = _sign(differences["mean_value"])
    calibration_sign = _sign(sensitivity["calibration"]["mean_value"])
    test_sign = _sign(sensitivity["test"]["mean_value"])
    sign_consistent = (
        pooled_sign != 0
        and calibration_sign == pooled_sign
        and test_sign == pooled_sign
    )
    material = (
        differences["mean_value"] is not None
        and abs(float(differences["mean_value"])) >= float(
            contrast["_material_effect_threshold"]
        )
    )
    minimum_frames = int(contrast["_minimum_frames_each_side"])
    enough_frames = (
        int(np.sum(focal["frames"])) >= minimum_frames
        and int(np.sum(reference["frames"])) >= minimum_frames
    )
    net_ci_excludes_zero = _ci_excludes_zero(
        intervals["mean_value_ci95_low"],
        intervals["mean_value_ci95_high"],
    )
    return {
        "id": str(contrast["id"]),
        "dimension": str(contrast["dimension"]),
        "focal_bins": list(contrast["focal_bins"]),
        "reference_bins": list(contrast["reference_bins"]),
        "focal_frames": int(np.sum(focal["frames"])),
        "reference_frames": int(np.sum(reference["frames"])),
        "focal_speaker_clusters": int(
            np.count_nonzero(focal["frames"] > 0)
        ),
        "reference_speaker_clusters": int(
            np.count_nonzero(reference["frames"] > 0)
        ),
        "focal_p_positive": focal_point["p_positive"],
        "focal_p_negative": focal_point["p_negative"],
        "focal_mean_value": focal_point["mean_value"],
        "reference_p_positive": reference_point["p_positive"],
        "reference_p_negative": reference_point["p_negative"],
        "reference_mean_value": reference_point["mean_value"],
        "p_positive_difference": differences["p_positive"],
        "p_negative_difference": differences["p_negative"],
        "mean_value_difference": differences["mean_value"],
        **intervals,
        "calibration_p_positive_difference": sensitivity["calibration"][
            "p_positive"
        ],
        "calibration_p_negative_difference": sensitivity["calibration"][
            "p_negative"
        ],
        "calibration_mean_value_difference": sensitivity["calibration"][
            "mean_value"
        ],
        "test_p_positive_difference": sensitivity["test"]["p_positive"],
        "test_p_negative_difference": sensitivity["test"]["p_negative"],
        "test_mean_value_difference": sensitivity["test"]["mean_value"],
        "criteria": {
            "material_mean_value_difference": material,
            "minimum_frames_each_side": enough_frames,
            "paired_mean_value_ci_excludes_zero": net_ci_excludes_zero,
            "calibration_test_sign_consistent": sign_consistent,
        },
        "supported": bool(
            material
            and enough_frames
            and net_ci_excludes_zero
            and sign_consistent
        ),
    }


def assess_ax3(
    contrast_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supported = [
        str(row["id"]) for row in contrast_rows if bool(row["supported"])
    ]
    limited_by_metric: dict[str, list[str]] = {name: [] for name in METRICS}
    for row in contrast_rows:
        for metric in METRICS:
            if _ci_excludes_zero(
                row.get(f"{metric}_ci95_low"),
                row.get(f"{metric}_ci95_high"),
            ):
                limited_by_metric[metric].append(str(row["id"]))
    any_ci = any(values for values in limited_by_metric.values())
    if supported:
        status = "EVENT_STRUCTURE_SUPPORTED"
        interpretation = (
            "At least one predefined event-structure contrast has a "
            "material paired net-value difference with a speaker-cluster "
            "95% CI excluding zero and calibration/test sign consistency. "
            "This is descriptive and does not authorize a new router."
        )
    elif any_ci:
        status = "EVENT_STRUCTURE_LIMITED"
        interpretation = (
            "No contrast passes every supported rule, but at least one "
            "predefined paired contrast has a speaker-cluster 95% CI "
            "excluding zero. Event structure is suggestive but not stable "
            "under the frozen criteria."
        )
    else:
        status = "NO_STABLE_EVENT_STRUCTURE"
        interpretation = (
            "No predefined event-structure contrast has a paired "
            "speaker-cluster 95% CI excluding zero for net value, "
            "correction probability, or harm probability."
        )
    return {
        "status": status,
        "supported_contrasts": supported,
        "ci_excluding_zero_contrasts": limited_by_metric,
        "interpretation": interpretation,
        "interpretation_limit": (
            "AX3 is descriptive. No status authorizes a new router, "
            "changes the frozen gate, or changes A14."
        ),
    }


def evaluate_ax3(context: Mapping[str, Any]) -> dict[str, Any]:
    config = context["config"]
    bundle = context["bundle"]
    values = np.asarray(context["values"], dtype=np.int64)
    segments = context["segments"]
    calibration_mask = np.asarray(context["calibration_mask"], dtype=bool)
    test_mask = np.asarray(context["test_mask"], dtype=bool)
    event_payload = config.payload["event_definitions"]
    event_structure = build_event_structure(
        bundle["labels"],
        bundle["short_scores"],
        segments,
        decision_threshold=float(event_payload["decision_threshold"]),
        uncertainty_threshold=float(
            event_payload["uncertainty_threshold"]
        ),
    )

    bootstrap_payload = config.payload["bootstrap"]
    repeats = int(bootstrap_payload["repeats"])
    seed = int(bootstrap_payload["seed"])
    result: dict[str, Any] = {
        "protocol": {
            "id": str(config.payload["protocol_id"]),
            "path": str(config.protocol_path),
            "frozen_scope": config.payload["frozen_scope"],
            "event_definitions": event_payload,
            "contrasts": config.payload["contrasts"],
            "bootstrap": bootstrap_payload,
        },
        "split": {
            "all_frames": int(values.size),
            "calibration_frames": int(np.count_nonzero(calibration_mask)),
            "test_frames": int(np.count_nonzero(test_mask)),
            "calibration_speakers": list(context["calibration_speakers"]),
            "test_speakers": list(context["test_speakers"]),
        },
        "signed_value": {
            "corrections": int(np.count_nonzero(values == 1)),
            "zero": int(np.count_nonzero(values == 0)),
            "harms": int(np.count_nonzero(values == -1)),
            "mean_value": float(np.mean(values)),
        },
        "bins": {},
        "contrasts": [],
    }

    all_cluster_names, all_cluster_codes = np.unique(
        bundle["speaker_ids"],
        return_inverse=True,
    )
    primary_weights = bootstrap_cluster_weights(
        int(all_cluster_names.size),
        repeats=repeats,
        seed=seed,
    )
    for dimension_name, definition in event_payload["dimensions"].items():
        result["bins"][dimension_name] = summarize_bins(
            dimension_name,
            event_structure[dimension_name],
            values,
            all_cluster_codes,
            bins=definition["bins"],
            weights=primary_weights,
        )
    material_threshold = float(
        config.payload["assessment"]["material_effect_threshold"]
    )
    minimum_frames = int(
        config.payload["assessment"]["minimum_frames_each_contrast_side"]
    )
    for contrast in config.payload["contrasts"]:
        dimension = str(contrast["dimension"])
        labels = event_structure[dimension]
        focal_mask = np.isin(labels, list(contrast["focal_bins"]))
        reference_mask = np.isin(labels, list(contrast["reference_bins"]))
        enriched = {
            **contrast,
            "_material_effect_threshold": material_threshold,
            "_minimum_frames_each_side": minimum_frames,
        }
        result["contrasts"].append(
            summarize_contrast(
                enriched,
                values,
                all_cluster_codes,
                focal_mask=focal_mask,
                reference_mask=reference_mask,
                weights=primary_weights,
                calibration_mask=calibration_mask,
                test_mask=test_mask,
            )
        )

    result["assessment"] = assess_ax3(result["contrasts"])
    result["bootstrap"] = {
        "unit": str(bootstrap_payload["unit"]),
        "repeats": repeats,
        "seed": seed,
        "confidence_interval": str(
            bootstrap_payload["confidence_interval"]
        ),
        "shared_weights_per_population": True,
    }
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_metrics_csv(
    summary: Mapping[str, Any],
    path: Path,
) -> None:
    fieldnames = [
        "record_type",
        "dimension",
        "bin",
        "contrast_id",
        "frames",
        "speaker_clusters",
        "corrections",
        "harms",
        "p_positive",
        "p_positive_ci95_low",
        "p_positive_ci95_high",
        "p_negative",
        "p_negative_ci95_low",
        "p_negative_ci95_high",
        "mean_value",
        "mean_value_ci95_low",
        "mean_value_ci95_high",
        "frame_share",
        "correction_share",
        "harm_share",
        "focal_bins",
        "reference_bins",
        "focal_frames",
        "reference_frames",
        "focal_p_positive",
        "focal_p_negative",
        "focal_mean_value",
        "reference_p_positive",
        "reference_p_negative",
        "reference_mean_value",
        "p_positive_difference",
        "p_negative_difference",
        "mean_value_difference",
        "calibration_mean_value_difference",
        "test_mean_value_difference",
        "supported",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dimension_rows in summary["bins"].values():
            for row in dimension_rows:
                writer.writerow(
                    {
                        **{name: row.get(name) for name in fieldnames},
                        "record_type": "bin",
                    }
                )
        for row in summary["contrasts"]:
            values = {
                **{name: row.get(name) for name in fieldnames},
                "record_type": "contrast",
                "contrast_id": row["id"],
                "focal_bins": "|".join(row["focal_bins"]),
                "reference_bins": "|".join(row["reference_bins"]),
                "supported": row["supported"],
            }
            writer.writerow(values)


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.3f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# AX3 Temporal Location and Event Structure",
        "",
        "## Frozen Scope",
        "",
        f"- Protocol: `{summary['protocol']['id']}`.",
        f"- All development frames: {summary['split']['all_frames']:,}.",
        f"- Speaker clusters: "
        f"{len(summary['split']['calibration_speakers']) + len(summary['split']['test_speakers'])}.",
        f"- Signed value: +1 {summary['signed_value']['corrections']:,}, "
        f"0 {summary['signed_value']['zero']:,}, "
        f"-1 {summary['signed_value']['harms']:,}.",
        "- Ground-truth boundary and run-length dimensions are explanatory "
        "strata only.",
        "- Posterior transition and uncertainty-duration dimensions are "
        "causal diagnostics within each utterance.",
        "",
        "## Per-Bin Results",
        "",
        "| Dimension | Bin | Frames | Speakers | P(v=+1) | 95% CI | "
        "P(v=-1) | 95% CI | E[v] | 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dimension_rows in summary["bins"].values():
        for row in dimension_rows:
            lines.append(
                f"| {row['dimension']} | {row['bin']} | "
                f"{int(row['frames']):,} | "
                f"{int(row['speaker_clusters'])} | "
                f"{_percent(row.get('p_positive'))} | "
                f"[{_percent(row.get('p_positive_ci95_low'))}, "
                f"{_percent(row.get('p_positive_ci95_high'))}] | "
                f"{_percent(row.get('p_negative'))} | "
                f"[{_percent(row.get('p_negative_ci95_low'))}, "
                f"{_percent(row.get('p_negative_ci95_high'))}] | "
                f"{_number(row.get('mean_value'))} | "
                f"[{_number(row.get('mean_value_ci95_low'))}, "
                f"{_number(row.get('mean_value_ci95_high'))}] |"
            )

    lines.extend(
        [
            "",
            "## Predefined Contrasts",
            "",
            "| Contrast | Focal vs reference bins | Frames | "
            "Delta P(v=+1) | Delta P(v=-1) | Delta E[v] | 95% CI | "
            "Calibration/test Delta E[v] | Supported |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["contrasts"]:
        lines.append(
            f"| {row['id']} | "
            f"{','.join(row['focal_bins'])} vs "
            f"{','.join(row['reference_bins'])} | "
            f"{int(row['focal_frames']):,}/{int(row['reference_frames']):,} | "
            f"{_percent(row.get('p_positive_difference'))} | "
            f"{_percent(row.get('p_negative_difference'))} | "
            f"{_number(row.get('mean_value_difference'))} | "
            f"[{_number(row.get('mean_value_ci95_low'))}, "
            f"{_number(row.get('mean_value_ci95_high'))}] | "
            f"{_number(row.get('calibration_mean_value_difference'))}/"
            f"{_number(row.get('test_mean_value_difference'))} | "
            f"{'yes' if row['supported'] else 'no'} |"
        )

    assessment = summary["assessment"]
    lines.extend(
        [
            "",
            "## AX3 Assessment",
            "",
            f"- Status: **{assessment['status']}**.",
            f"- Supported contrasts: "
            + (
                ", ".join(assessment["supported_contrasts"])
                if assessment["supported_contrasts"]
                else "none"
            ),
            f"- Interpretation: {assessment['interpretation']}",
            f"- Limit: {assessment['interpretation_limit']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.protocol)
    context = load_ax3_context(
        config,
        verify_hashes=not bool(args.skip_hash_check),
    )
    context["config"] = config
    summary = evaluate_ax3(context)
    summary["inputs"] = {
        "hashes": context["hashes"],
        "prediction_bundle": str(
            config.resolve(
                str(
                    config.payload["inputs"]["prediction_bundle"]["path"]
                )
            )
        ),
        "adaptive_checkpoint": str(
            config.resolve(
                str(
                    config.payload["inputs"]["adaptive_checkpoint"]["path"]
                )
            )
        ),
        "manifest": str(
            config.resolve(str(config.payload["inputs"]["manifest"]["path"]))
        ),
    }
    output_dir = config.output_dir
    outputs = config.payload["outputs"]
    write_json(output_dir / str(outputs["summary"]), summary)
    write_metrics_csv(
        summary,
        output_dir / str(outputs["metrics_table"]),
    )
    write_report(summary, output_dir / str(outputs["report"]))
    print(
        json.dumps(summary["assessment"], indent=2, ensure_ascii=False),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
