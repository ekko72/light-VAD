# -*- coding: utf-8 -*-
"""Classify routing failures for the frozen A12 development cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reproductions.difficulty_adaptive_context.run_a12_routing_robustness import (
    _domain_mask,
    load_a12_bundle,
)
from reproductions.ssr_prior_shift.protocol import stratified_frame_resample


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ax4_protocol.json"
)
TAXONOMY_CLASSES = (
    "F1_VALUE_SCARCITY",
    "F2_RANKING_FAILURE",
    "F3_BUDGET_CALIBRATION_DRIFT",
    "NO_FAILURE",
)


@dataclass(frozen=True)
class Ax4Config:
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


def load_config(path: Path = DEFAULT_PROTOCOL) -> Ax4Config:
    if not path.exists():
        raise FileNotFoundError(f"AX4 protocol not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("protocol_id")) != "A-AX4-v1":
        raise ValueError("unexpected AX4 protocol id")
    return Ax4Config(protocol_path=path, payload=payload)


def verify_hash(
    path: Path,
    expected_sha256: str,
    *,
    verify: bool,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"frozen AX4 input not found: {path}")
    if not verify:
        return "not_verified"
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise ValueError(
            f"hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def load_ax4_bundle(
    path: str | Path,
) -> dict[str, Any]:
    """Load the A12 bundle plus the broader speaker-cluster IDs."""
    bundle = load_a12_bundle(path)
    with np.load(Path(path), allow_pickle=False) as payload:
        if "speaker_ids" not in payload.files:
            raise ValueError("prediction bundle is missing speaker_ids")
        speaker_ids = np.asarray(payload["speaker_ids"], dtype=str).copy()
    if speaker_ids.size != np.asarray(bundle["labels"]).size:
        raise ValueError("speaker_ids must match the prediction bundle")
    bundle["speaker_ids"] = speaker_ids
    return bundle


def _cell_key(
    *,
    section: str,
    domain: str,
    condition: str,
    target_ssr: float | None,
) -> tuple[str, str, str, float | None]:
    return (
        str(section),
        str(domain),
        str(condition),
        None if target_ssr is None else float(target_ssr),
    )


def build_cells(
    bundle: Mapping[str, Any],
    config: Ax4Config,
) -> list[dict[str, Any]]:
    """Recreate the frozen A12 SSR and acoustic development cells."""
    labels = np.asarray(bundle["labels"], dtype=np.int64).reshape(-1)
    short_scores = np.asarray(
        bundle["short_scores"],
        dtype=np.float64,
    ).reshape(-1)
    refined_scores = np.asarray(
        bundle["refined_scores"],
        dtype=np.float64,
    ).reshape(-1)
    source_key = np.asarray(bundle["source_key"], dtype=str).reshape(-1)
    speaker_ids = np.asarray(bundle["speaker_ids"], dtype=str).reshape(-1)
    condition = np.asarray(bundle["condition"], dtype=str).reshape(-1)
    noise_name = np.asarray(bundle["noise_name"], dtype=str).reshape(-1)
    test_mask = np.asarray(bundle["test_mask"], dtype=bool).reshape(-1)
    threshold = float(bundle["threshold"])
    arrays = (
        labels,
        short_scores,
        refined_scores,
        source_key,
        speaker_ids,
        condition,
        noise_name,
        test_mask,
    )
    if any(values.size != labels.size for values in arrays):
        raise ValueError("prediction bundle arrays must have equal lengths")

    cell_payload = config.payload["cells"]
    ssr_payload = cell_payload["ssr_sweep"]
    acoustic_payload = cell_payload["acoustic_cell"]
    domains = tuple(str(value) for value in ssr_payload["domains"])
    conditions = tuple(str(value) for value in ssr_payload["conditions"])
    unseen_noise = tuple(
        str(value) for value in ssr_payload["unseen_noise"]
    )
    levels = tuple(
        sorted(float(value) for value in ssr_payload["target_ssr"])
    )
    if not levels:
        raise ValueError("at least one SSR level is required")
    if tuple(str(value) for value in acoustic_payload["domains"]) != domains:
        raise ValueError("SSR and acoustic domains must match")
    if tuple(str(value) for value in acoustic_payload["conditions"]) != conditions:
        raise ValueError("SSR and acoustic conditions must match")
    if tuple(
        str(value) for value in acoustic_payload["unseen_noise"]
    ) != unseen_noise:
        raise ValueError("SSR and acoustic unseen-noise definitions must match")

    cells: list[dict[str, Any]] = []
    condition_mask = np.isin(condition, np.asarray(conditions, dtype=str))
    for domain_index, domain in enumerate(domains):
        domain_mask = _domain_mask(
            noise_name,
            domain=domain,
            unseen_noise=unseen_noise,
        )
        pool_mask = test_mask & condition_mask & domain_mask
        if not np.any(pool_mask):
            raise ValueError(f"no test frames remain for domain {domain}")
        pool_indices = np.flatnonzero(pool_mask)
        for ssr_index, target_ssr in enumerate(levels):
            sample = stratified_frame_resample(
                labels[pool_indices],
                short_scores[pool_indices],
                noise_name[pool_indices],
                condition[pool_indices],
                source_key[pool_indices],
                target_ssr=float(target_ssr),
                frames_per_stratum=int(
                    ssr_payload["frames_per_stratum"]
                ),
                seed=(
                    int(ssr_payload["resample_seed"])
                    + 7_919 * domain_index
                    + 1_003 * ssr_index
                ),
            )
            global_indices = pool_indices[sample["indices"]]
            cells.append(
                {
                    "section": "ssr_sweep",
                    "domain": str(domain),
                    "condition": "all_noisy",
                    "target_ssr": float(target_ssr),
                    "actual_ssr": float(sample["actual_ssr"]),
                    "indices": global_indices,
                    "labels": sample["labels"].astype(np.int64),
                    "short_scores": sample["scores"].astype(np.float64),
                    "refined_scores": refined_scores[global_indices],
                    "source_key": sample["source_key"].astype(str),
                    "speaker_ids": speaker_ids[global_indices],
                    "threshold": threshold,
                    "is_major_development_cell": True,
                }
            )

        for cell_name in conditions:
            mask = test_mask & domain_mask & (condition == str(cell_name))
            if not np.any(mask):
                raise ValueError(
                    f"no test frames remain for {domain}/{cell_name}"
                )
            indices = np.flatnonzero(mask)
            cells.append(
                {
                    "section": "acoustic_cell",
                    "domain": str(domain),
                    "condition": str(cell_name),
                    "target_ssr": None,
                    "actual_ssr": float(np.mean(labels[indices])),
                    "indices": indices,
                    "labels": labels[indices],
                    "short_scores": short_scores[indices],
                    "refined_scores": refined_scores[indices],
                    "source_key": source_key[indices],
                    "speaker_ids": speaker_ids[indices],
                    "threshold": threshold,
                    "is_major_development_cell": True,
                }
            )

    expected = int(cell_payload["major_cell_count"])
    if len(cells) != expected:
        raise RuntimeError(
            f"constructed {len(cells)} cells, expected {expected}"
        )
    return cells


def signed_values(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    refined_scores = np.asarray(refined_scores, dtype=np.float64).reshape(-1)
    if not (labels.size == short_scores.size == refined_scores.size):
        raise ValueError("value arrays must have equal lengths")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    short_wrong = (short_scores >= 0.5) != labels
    refined_wrong = (refined_scores >= 0.5) != labels
    return (
        (short_wrong & ~refined_wrong).astype(np.int64)
        - (~short_wrong & refined_wrong).astype(np.int64)
    )


def raw_gate_selection(
    short_scores: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    return np.abs(scores - 0.5) <= threshold


def bootstrap_cluster_weights(
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
    weights = np.empty((int(n_clusters), int(repeats)), dtype=np.int64)
    for repeat in range(int(repeats)):
        sampled = rng.integers(0, int(n_clusters), size=int(n_clusters))
        weights[:, repeat] = np.bincount(
            sampled,
            minlength=int(n_clusters),
        )
    return weights


def cluster_cell_counts(
    values: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    clusters = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (values.size == selected.size == clusters.size):
        raise ValueError("value, selection, and cluster arrays must align")
    return {
        "frames": np.bincount(
            clusters,
            minlength=int(n_clusters),
        ).astype(np.int64),
        "corrections": np.bincount(
            clusters[values == 1],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "harms": np.bincount(
            clusters[values == -1],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "selected": np.bincount(
            clusters[selected],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "selected_corrections": np.bincount(
            clusters[selected & (values == 1)],
            minlength=int(n_clusters),
        ).astype(np.int64),
        "selected_harms": np.bincount(
            clusters[selected & (values == -1)],
            minlength=int(n_clusters),
        ).astype(np.int64),
    }


def _safe_ratio(
    numerator: np.ndarray | int,
    denominator: np.ndarray | int,
) -> np.ndarray | float | None:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            denominator_array > 0.0,
            numerator_array / denominator_array,
            np.nan,
        )
    if ratio.ndim == 0:
        value = float(ratio)
        return value if np.isfinite(value) else None
    return ratio


def point_metrics(
    counts: Mapping[str, np.ndarray],
) -> dict[str, int | float | None]:
    frames = int(np.sum(counts["frames"]))
    corrections = int(np.sum(counts["corrections"]))
    harms = int(np.sum(counts["harms"]))
    selected = int(np.sum(counts["selected"]))
    selected_corrections = int(np.sum(counts["selected_corrections"]))
    selected_harms = int(np.sum(counts["selected_harms"]))
    return {
        "frames": frames,
        "corrections": corrections,
        "harms": harms,
        "p_positive": _safe_ratio(corrections, frames),
        "p_negative": _safe_ratio(harms, frames),
        "mean_value": _safe_ratio(corrections - harms, frames),
        "selected": selected,
        "selected_corrections": selected_corrections,
        "selected_harms": selected_harms,
        "activation_rate": _safe_ratio(selected, frames),
        "selected_p_positive": _safe_ratio(
            selected_corrections,
            selected,
        ),
        "selected_p_negative": _safe_ratio(selected_harms, selected),
        "selected_mean_value": _safe_ratio(
            selected_corrections - selected_harms,
            selected,
        ),
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


def bootstrap_metrics(
    counts: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, float | None]:
    weights = np.asarray(weights, dtype=np.int64)
    frames = np.asarray(counts["frames"], dtype=np.int64)
    if weights.ndim != 2 or weights.shape[0] != frames.size:
        raise ValueError("bootstrap weights do not match the cluster counts")
    sampled_frames = frames @ weights
    sampled_corrections = (
        np.asarray(counts["corrections"], dtype=np.int64) @ weights
    )
    sampled_harms = np.asarray(counts["harms"], dtype=np.int64) @ weights
    sampled_selected = (
        np.asarray(counts["selected"], dtype=np.int64) @ weights
    )
    sampled_selected_corrections = (
        np.asarray(counts["selected_corrections"], dtype=np.int64) @ weights
    )
    sampled_selected_harms = (
        np.asarray(counts["selected_harms"], dtype=np.int64) @ weights
    )
    replicates = {
        "p_positive": _safe_ratio(
            sampled_corrections,
            sampled_frames,
        ),
        "p_negative": _safe_ratio(sampled_harms, sampled_frames),
        "mean_value": _safe_ratio(
            sampled_corrections - sampled_harms,
            sampled_frames,
        ),
        "activation_rate": _safe_ratio(
            sampled_selected,
            sampled_frames,
        ),
        "selected_p_positive": _safe_ratio(
            sampled_selected_corrections,
            sampled_selected,
        ),
        "selected_p_negative": _safe_ratio(
            sampled_selected_harms,
            sampled_selected,
        ),
        "selected_mean_value": _safe_ratio(
            sampled_selected_corrections - sampled_selected_harms,
            sampled_selected,
        ),
    }
    intervals: dict[str, float | None] = {}
    for name, values in replicates.items():
        low, high = _percentile_interval(np.asarray(values))
        intervals[f"{name}_ci95_low"] = low
        intervals[f"{name}_ci95_high"] = high
    return intervals


def classify_failure(
    *,
    population_mean_value: float | None,
    selected_mean_value: float | None,
    activation_rate: float | None,
    activation_warning_threshold: float,
) -> str:
    if population_mean_value is None or activation_rate is None:
        raise ValueError("population value and activation must be defined")
    if selected_mean_value is None:
        raise ValueError("selected value is undefined with zero activation")
    population = float(population_mean_value)
    selected_value = float(selected_mean_value)
    activation = float(activation_rate)
    warning = float(activation_warning_threshold)
    if population <= 0.0 and selected_value <= 0.0:
        return "F1_VALUE_SCARCITY"
    if population > 0.0 and selected_value <= 0.0:
        return "F2_RANKING_FAILURE"
    if selected_value > 0.0 and activation > warning:
        return "F3_BUDGET_CALIBRATION_DRIFT"
    return "NO_FAILURE"


def summarize_cell(
    cell: Mapping[str, Any],
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    activation_warning_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(cell["labels"], dtype=np.int64)
    short_scores = np.asarray(cell["short_scores"], dtype=np.float64)
    refined_scores = np.asarray(
        cell["refined_scores"],
        dtype=np.float64,
    )
    source_key = np.asarray(cell["source_key"], dtype=str)
    speaker_ids = np.asarray(cell["speaker_ids"], dtype=str)
    values = signed_values(labels, short_scores, refined_scores)
    selected = raw_gate_selection(
        short_scores,
        threshold=float(cell["threshold"]),
    )

    source_names, source_codes = np.unique(source_key, return_inverse=True)
    speaker_names, speaker_codes = np.unique(
        speaker_ids,
        return_inverse=True,
    )
    source_counts = cluster_cell_counts(
        values,
        selected,
        source_codes,
        int(source_names.size),
    )
    speaker_counts = cluster_cell_counts(
        values,
        selected,
        speaker_codes,
        int(speaker_names.size),
    )
    point = point_metrics(source_counts)
    source_intervals = bootstrap_metrics(
        source_counts,
        bootstrap_cluster_weights(
            int(source_names.size),
            repeats=int(bootstrap_repeats),
            seed=int(bootstrap_seed),
        ),
    )
    speaker_intervals = bootstrap_metrics(
        speaker_counts,
        bootstrap_cluster_weights(
            int(speaker_names.size),
            repeats=int(bootstrap_repeats),
            seed=int(bootstrap_seed) + 1,
        ),
    )
    taxonomy = classify_failure(
        population_mean_value=point["mean_value"],
        selected_mean_value=point["selected_mean_value"],
        activation_rate=point["activation_rate"],
        activation_warning_threshold=activation_warning_threshold,
    )
    row = {
        "section": str(cell["section"]),
        "domain": str(cell["domain"]),
        "condition": str(cell["condition"]),
        "target_ssr": (
            None
            if cell["target_ssr"] is None
            else float(cell["target_ssr"])
        ),
        "actual_ssr": float(cell["actual_ssr"]),
        "source_clusters": int(source_names.size),
        "speaker_clusters": int(speaker_names.size),
        **point,
        **source_intervals,
        "speaker_sensitivity": {
            key: value for key, value in speaker_intervals.items()
        },
        "failure_class": taxonomy,
        "is_taxonomy_failure": taxonomy != "NO_FAILURE",
    }
    return row


def _focus_key_from_spec(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return _cell_key(
        section=str(spec["section"]),
        domain=str(spec["domain"]),
        condition=str(spec["condition"]),
        target_ssr=(
            None
            if spec.get("target_ssr") is None
            else float(spec["target_ssr"])
        ),
    )


def assess_ax4(
    rows: Sequence[Mapping[str, Any]],
    *,
    focus_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {
        _cell_key(
            section=str(row["section"]),
            domain=str(row["domain"]),
            condition=str(row["condition"]),
            target_ssr=(
                None
                if row.get("target_ssr") is None
                else float(row["target_ssr"])
            ),
        ): row
        for row in rows
    }
    focus_rows: list[Mapping[str, Any]] = []
    for spec in focus_cells:
        key = _focus_key_from_spec(spec)
        if key not in lookup:
            raise KeyError(f"focus cell not found: {key}")
        focus_rows.append(lookup[key])

    failure_rows = [
        row
        for row in rows
        if str(row["failure_class"]) != "NO_FAILURE"
    ]
    counts = {
        class_name: sum(
            str(row["failure_class"]) == class_name for row in rows
        )
        for class_name in TAXONOMY_CLASSES
    }
    f3_rows = [
        row
        for row in rows
        if str(row["failure_class"]) == "F3_BUDGET_CALIBRATION_DRIFT"
    ]
    ax5_triggered = bool(f3_rows)
    focus_classes = {
        str(row["failure_class"]) for row in focus_rows
    }
    if not failure_rows:
        status = "NO_RAW_FAILURE"
        interpretation = (
            "No major development cell has non-positive selected value or "
            "activation above the frozen warning threshold."
        )
    elif ax5_triggered:
        status = "F3_BUDGET_CALIBRATION_DRIFT_DETECTED"
        interpretation = (
            "At least one major development cell retains positive selected "
            "value but exceeds the frozen 15% activation warning threshold. "
            "This permits the separately gated AX5 budget-policy diagnostic."
        )
    elif len(focus_classes) == 1:
        only_class = next(iter(focus_classes))
        if only_class == "F1_VALUE_SCARCITY":
            status = "F1_VALUE_SCARCITY_DOMINANT"
            interpretation = (
                "The frozen A12 failure cells have non-positive population "
                "and selected signed value. Long-context value is scarce in "
                "these conditions."
            )
        elif only_class == "F2_RANKING_FAILURE":
            status = "F2_RANKING_FAILURE_DOMINANT"
            interpretation = (
                "The frozen A12 failure cells have positive population "
                "signed value but non-positive selected value. The gate "
                "selects the wrong frames in these conditions."
            )
        else:
            status = "MIXED_FAILURE"
            interpretation = (
                "The frozen A12 failure cells do not share a single failure "
                "class."
            )
    else:
        status = "MIXED_FAILURE"
        interpretation = (
            "The frozen A12 failure cells do not share a single failure "
            "class."
        )

    return {
        "status": status,
        "failure_class_counts": counts,
        "taxonomy_failure_count": len(failure_rows),
        "f3_cell_count": len(f3_rows),
        "ax5_triggered": ax5_triggered,
        "focus_cells": [
            {
                "section": row["section"],
                "domain": row["domain"],
                "condition": row["condition"],
                "target_ssr": row.get("target_ssr"),
                "population_mean_value": row.get("mean_value"),
                "selected_mean_value": row.get("selected_mean_value"),
                "activation_rate": row.get("activation_rate"),
                "failure_class": row["failure_class"],
            }
            for row in focus_rows
        ],
        "f3_keys": [
            {
                "section": row["section"],
                "domain": row["domain"],
                "condition": row["condition"],
                "target_ssr": row.get("target_ssr"),
            }
            for row in f3_rows
        ],
        "interpretation": interpretation,
        "interpretation_limit": (
            "AX4 is diagnostic. Its point taxonomy does not authorize gate "
            "tuning, a new router, or opening A14."
        ),
    }


def evaluate_ax4(context: Mapping[str, Any]) -> dict[str, Any]:
    config = context["config"]
    bundle = context["bundle"]
    cells = build_cells(bundle, config)
    bootstrap = config.payload["bootstrap"]
    warning = float(
        config.payload["taxonomy"]["activation_warning_threshold"]
    )
    rows = [
        summarize_cell(
            cell,
            bootstrap_repeats=int(bootstrap["repeats"]),
            bootstrap_seed=int(bootstrap["seed"]) + 10_007 * index,
            activation_warning_threshold=warning,
        )
        for index, cell in enumerate(cells)
    ]
    assessment = assess_ax4(
        rows,
        focus_cells=config.payload["taxonomy"]["focus_cells"],
    )
    return {
        "protocol": {
            "id": str(config.payload["protocol_id"]),
            "path": str(config.protocol_path),
            "frozen_scope": config.payload["frozen_scope"],
            "gate": config.payload["gate"],
            "cells": config.payload["cells"],
            "taxonomy": config.payload["taxonomy"],
            "bootstrap": bootstrap,
        },
        "bundle": {
            "path": str(bundle["path"]),
            "threshold": float(bundle["threshold"]),
            "test_frames": int(np.count_nonzero(bundle["test_mask"])),
            "source_clusters": int(
                np.unique(np.asarray(bundle["source_key"], dtype=str)).size
            ),
            "speaker_clusters": int(
                np.unique(
                    np.asarray(bundle["speaker_ids"], dtype=str)
                ).size
            ),
        },
        "rows": rows,
        "assessment": assessment,
    }


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(payload),
            handle,
            indent=2,
            ensure_ascii=False,
        )


def write_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fieldnames = [
        "section",
        "domain",
        "condition",
        "target_ssr",
        "actual_ssr",
        "frames",
        "source_clusters",
        "speaker_clusters",
        "corrections",
        "harms",
        "p_positive",
        "p_negative",
        "mean_value",
        "mean_value_ci95_low",
        "mean_value_ci95_high",
        "selected",
        "selected_corrections",
        "selected_harms",
        "activation_rate",
        "activation_rate_ci95_low",
        "activation_rate_ci95_high",
        "selected_p_positive",
        "selected_p_negative",
        "selected_mean_value",
        "selected_mean_value_ci95_low",
        "selected_mean_value_ci95_high",
        "speaker_mean_value_ci95_low",
        "speaker_mean_value_ci95_high",
        "speaker_selected_mean_value_ci95_low",
        "speaker_selected_mean_value_ci95_high",
        "failure_class",
        "is_taxonomy_failure",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sensitivity = row["speaker_sensitivity"]
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "speaker_mean_value_ci95_low": sensitivity.get(
                        "mean_value_ci95_low"
                    ),
                    "speaker_mean_value_ci95_high": sensitivity.get(
                        "mean_value_ci95_high"
                    ),
                    "speaker_selected_mean_value_ci95_low": sensitivity.get(
                        "selected_mean_value_ci95_low"
                    ),
                    "speaker_selected_mean_value_ci95_high": sensitivity.get(
                        "selected_mean_value_ci95_high"
                    ),
                }
            )


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.3f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _cell_label(row: Mapping[str, Any]) -> str:
    if row["section"] == "ssr_sweep":
        return f"SSR {100.0 * float(row['target_ssr']):.0f}%"
    return f"SNR {row['condition']} dB"


def write_report(
    summary: Mapping[str, Any],
    path: Path,
) -> None:
    lines = [
        "# AX4 Routing Failure Taxonomy",
        "",
        "## Frozen Scope",
        "",
        f"- Protocol: `{summary['protocol']['id']}`.",
        f"- Raw gate threshold: `{float(summary['bundle']['threshold']):.6f}`.",
        f"- Test frames: {int(summary['bundle']['test_frames']):,}.",
        f"- Source clusters: {int(summary['bundle']['source_clusters'])}.",
        f"- Speaker clusters: {int(summary['bundle']['speaker_clusters'])}.",
        "- The threshold and cells are reused from frozen A12 definitions.",
        "- The source-rank stabilizer is not evaluated or retuned.",
        "",
        "## Cell Taxonomy",
        "",
        "| Section | Domain | Cell | Activation | Population P(+1) | "
        "Population P(-1) | Population E[v] | Selected P(+1) | "
        "Selected P(-1) | Selected E[v] | Class |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['section']} | {row['domain']} | {_cell_label(row)} | "
            f"{_percent(row.get('activation_rate'))} | "
            f"{_percent(row.get('p_positive'))} | "
            f"{_percent(row.get('p_negative'))} | "
            f"{_number(row.get('mean_value'))} | "
            f"{_percent(row.get('selected_p_positive'))} | "
            f"{_percent(row.get('selected_p_negative'))} | "
            f"{_number(row.get('selected_mean_value'))} | "
            f"{row['failure_class']} |"
        )

    lines.extend(
        [
            "",
            "## Bootstrap Intervals",
            "",
            "| Section | Domain | Cell | Population E[v] 95% source CI | "
            "Selected E[v] 95% source CI | Selected E[v] 95% speaker CI |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        speaker = row["speaker_sensitivity"]
        lines.append(
            f"| {row['section']} | {row['domain']} | {_cell_label(row)} | "
            f"[{_number(row.get('mean_value_ci95_low'))}, "
            f"{_number(row.get('mean_value_ci95_high'))}] | "
            f"[{_number(row.get('selected_mean_value_ci95_low'))}, "
            f"{_number(row.get('selected_mean_value_ci95_high'))}] | "
            f"[{_number(speaker.get('selected_mean_value_ci95_low'))}, "
            f"{_number(speaker.get('selected_mean_value_ci95_high'))}] |"
        )

    assessment = summary["assessment"]
    lines.extend(
        [
            "",
            "## AX4 Assessment",
            "",
            f"- Status: **{assessment['status']}**.",
            f"- Taxonomy failures: {assessment['taxonomy_failure_count']}.",
            f"- F1 value scarcity cells: "
            f"{assessment['failure_class_counts']['F1_VALUE_SCARCITY']}.",
            f"- F2 ranking failure cells: "
            f"{assessment['failure_class_counts']['F2_RANKING_FAILURE']}.",
            f"- F3 budget/calibration drift cells: "
            f"{assessment['failure_class_counts']['F3_BUDGET_CALIBRATION_DRIFT']}.",
            f"- AX5 triggered: "
            f"{'yes' if assessment['ax5_triggered'] else 'no'}.",
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
    verify_hashes = not bool(args.skip_hash_check)
    hashes: dict[str, str] = {}
    for name, entry in config.payload["inputs"].items():
        path = config.resolve(str(entry["path"]))
        hashes[name] = verify_hash(
            path,
            str(entry["sha256"]),
            verify=verify_hashes,
        )
    bundle_path = config.resolve(
        str(config.payload["inputs"]["prediction_bundle"]["path"])
    )
    bundle = load_ax4_bundle(bundle_path)
    summary = evaluate_ax4({"config": config, "bundle": bundle})
    summary["inputs"] = {
        "hashes": hashes,
        "prediction_bundle": str(bundle_path),
        "a12_runner": str(
            config.resolve(str(config.payload["inputs"]["a12_runner"]["path"]))
        ),
        "a12_results": str(
            config.resolve(str(config.payload["inputs"]["a12_results"]["path"]))
        ),
    }
    output_dir = config.output_dir
    outputs = config.payload["outputs"]
    write_json(
        output_dir / str(outputs["summary"]),
        summary,
    )
    write_metrics_csv(
        summary["rows"],
        output_dir / str(outputs["metrics_table"]),
    )
    write_report(
        summary,
        output_dir / str(outputs["report"]),
    )
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
