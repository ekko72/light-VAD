# -*- coding: utf-8 -*-
"""Run the A12 fixed-gate routing robustness audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    count_frames_by_cluster,
    metrics_from_counts,
    speaker_cluster_bootstrap,
)
from reproductions.difficulty_adaptive_context.benchmark_a11_conditional_compute import (
    binary_detection_metrics,
)
from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE
from reproductions.ssr_prior_shift.protocol import stratified_frame_resample


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
DEFAULT_A11_RESULTS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a11_conditional_compute"
    / "a11_conditional_compute.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a12_routing_robustness"
)

DEFAULT_SSR_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
DEFAULT_CONDITIONS = ("20", "10", "5", "0", "-5")
DEFAULT_FRAMES_PER_STRATUM = 160
DEFAULT_RANK_ACTIVATION = 0.05
DEFAULT_NOMINAL_ACTIVATION = 0.05
DEFAULT_ACTIVATION_WARNING = 0.15
DEFAULT_ACTIVATION_FAILURE = 0.20

FRAME_FIELDS = (
    "labels",
    "short_scores",
    "full_adaptive_scores",
    "test_mask",
    "calibration_mask",
    "source_key",
    "condition",
    "noise_name",
)


def load_a12_bundle(path: str | Path) -> dict[str, Any]:
    """Load the frozen A11 bundle while preserving its scalar threshold."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"prediction bundle not found: {bundle_path}")
    with np.load(bundle_path, allow_pickle=False) as payload:
        required = set(FRAME_FIELDS) | {"threshold"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(
                f"prediction bundle is missing fields: {', '.join(missing)}"
            )
        frame_arrays = {
            name: np.asarray(payload[name]).copy() for name in FRAME_FIELDS
        }
        threshold = np.asarray(payload["threshold"]).reshape(-1)

    if threshold.size != 1:
        raise ValueError("threshold must be a scalar")
    threshold_value = float(threshold[0])
    if not 0.0 <= threshold_value <= 0.5:
        raise ValueError("threshold must be in [0, 0.5]")

    size = int(frame_arrays["labels"].size)
    for name, values in frame_arrays.items():
        if int(values.size) != size:
            raise ValueError(
                f"prediction field {name!r} has size {values.size}, "
                f"expected {size}"
            )

    labels = frame_arrays["labels"].astype(np.int64).reshape(-1)
    short_scores = frame_arrays["short_scores"].astype(np.float64).reshape(-1)
    refined_scores = (
        frame_arrays["full_adaptive_scores"].astype(np.float64).reshape(-1)
    )
    test_mask = frame_arrays["test_mask"].astype(bool).reshape(-1)
    calibration_mask = (
        frame_arrays["calibration_mask"].astype(bool).reshape(-1)
    )
    source_key = frame_arrays["source_key"].astype(str).reshape(-1)
    condition = frame_arrays["condition"].astype(str).reshape(-1)
    noise_name = frame_arrays["noise_name"].astype(str).reshape(-1)

    if size == 0:
        raise ValueError("prediction bundle is empty")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    for name, scores in (
        ("short_scores", short_scores),
        ("full_adaptive_scores", refined_scores),
    ):
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} must be finite")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
    if not np.any(test_mask):
        raise ValueError("test_mask is empty")
    if not np.any(calibration_mask):
        raise ValueError("calibration_mask is empty")
    if np.any(test_mask & calibration_mask):
        raise ValueError("test and calibration masks overlap")

    return {
        "labels": labels,
        "short_scores": short_scores,
        "refined_scores": refined_scores,
        "test_mask": test_mask,
        "calibration_mask": calibration_mask,
        "source_key": source_key,
        "condition": condition,
        "noise_name": noise_name,
        "threshold": threshold_value,
        "path": str(bundle_path),
    }


def source_rank_uncertainty(
    short_scores: np.ndarray,
    source_key: np.ndarray,
    *,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the within-source percentile rank of low-confidence scores."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    source = np.asarray(source_key, dtype=str).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not (scores.size == source.size == mask.size):
        raise ValueError("scores, source keys and mask must have equal length")
    if not np.any(mask):
        raise ValueError("rank gate requires at least one selected frame")

    ranks = np.full(scores.size, np.nan, dtype=np.float64)
    for name in np.unique(source[mask]):
        indices = np.flatnonzero(mask & (source == name))
        uncertainty = np.abs(scores[indices] - 0.5)
        order = np.argsort(uncertainty, kind="stable")
        ordered_ranks = (np.arange(indices.size) + 0.5) / indices.size
        ranks[indices[order]] = ordered_ranks
    if np.any(~np.isfinite(ranks[mask])):
        raise RuntimeError("source-rank gate produced missing ranks")
    return ranks


def source_rank_selection(
    ranks: np.ndarray,
    *,
    mask: np.ndarray,
    activation_rate: float,
) -> np.ndarray:
    """Select a fixed within-source uncertainty percentile."""
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if ranks.size != mask.size:
        raise ValueError("ranks and mask must have equal length")
    if not 0.0 < float(activation_rate) <= 1.0:
        raise ValueError("activation_rate must be in (0, 1]")
    selected = np.zeros(mask.size, dtype=bool)
    selected[mask] = ranks[mask] <= float(activation_rate)
    return selected


def load_a11_cost_model(path: str | Path) -> dict[str, Any]:
    """Read the measured A11 cost terms used by the A12 projection."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"A11 results not found: {model_path}")
    with open(model_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indexed = {str(row["name"]): row for row in payload["rows"]}
    required = {"short", "adaptive_5pct", "fixed_rf384"}
    missing = sorted(required - set(indexed))
    if missing:
        raise ValueError(f"A11 results are missing rows: {missing}")

    short = indexed["short"]
    adaptive = indexed["adaptive_5pct"]
    always_refine = indexed["fixed_rf384"]
    kernel = adaptive.get("refinement_kernel")
    if not kernel:
        raise ValueError("A11 adaptive row has no refinement kernel")
    adaptive_activation = float(adaptive["test_activation_rate"])
    adaptive_ms = float(adaptive["mean_ms_per_frame"])
    refinement_ms = float(kernel["mean_ms_per_frame"])
    if adaptive_activation <= 0.0 or refinement_ms <= 0.0:
        raise ValueError("A11 cost terms must be positive")
    base_ms = adaptive_ms - adaptive_activation * refinement_ms
    if base_ms <= 0.0:
        raise ValueError("A11-derived base latency must be positive")

    return {
        "source": str(model_path),
        "short_gate_ms_per_frame": float(short["mean_ms_per_frame"]),
        "adaptive_5pct_ms_per_frame": adaptive_ms,
        "adaptive_5pct_activation": adaptive_activation,
        "refinement_ms_per_selected_frame": refinement_ms,
        "projected_base_ms_per_frame": base_ms,
        "always_refine_ms_per_frame": float(
            always_refine["mean_ms_per_frame"]
        ),
        "short_macs_per_frame": int(short["analytical_macs_per_frame"]),
        "refinement_macs_per_selected_frame": int(
            short["refinement_macs_per_selected_frame"]
        ),
        "cache_bytes": int(always_refine["cache_bytes"]),
    }


def project_cost(
    activation_rate: float,
    cost_model: Mapping[str, Any],
) -> dict[str, float | int]:
    """Project A12 CPU/MAC cost from the A11 measured cost terms."""
    activation = float(activation_rate)
    if not 0.0 <= activation <= 1.0:
        raise ValueError("activation_rate must be in [0, 1]")
    base_ms = float(cost_model["projected_base_ms_per_frame"])
    refinement_ms = float(
        cost_model["refinement_ms_per_selected_frame"]
    )
    short_macs = int(cost_model["short_macs_per_frame"])
    refinement_macs = int(
        cost_model["refinement_macs_per_selected_frame"]
    )
    always_ms = float(cost_model["always_refine_ms_per_frame"])
    estimated_ms = base_ms + activation * refinement_ms
    return {
        "estimated_ms_per_frame": estimated_ms,
        "estimated_relative_to_always_refine": (
            estimated_ms / always_ms if always_ms > 0.0 else None
        ),
        "analytical_macs_per_frame": int(
            round(short_macs + activation * refinement_macs)
        ),
        "cache_bytes": int(cost_model["cache_bytes"]),
    }


def _evaluate_gate(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    source_key: np.ndarray,
    *,
    mask: np.ndarray,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    cluster_names, counts = count_frames_by_cluster(
        labels,
        short_scores,
        refined_scores,
        selected,
        source_key,
        mask=mask,
    )
    utility = metrics_from_counts(counts.sum(axis=0))
    adaptive_scores = np.where(selected, refined_scores, short_scores)
    detection = binary_detection_metrics(
        labels[mask],
        adaptive_scores[mask],
    )
    bootstrap = speaker_cluster_bootstrap(
        [counts],
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    return {
        **utility,
        "f1": detection["f1"],
        "far": detection["far"],
        "miss_rate": detection["miss_rate"],
        "precision": detection["precision"],
        "recall": detection["recall"],
        "activation_rate_ci95_low": bootstrap[
            "activation_rate_ci95_low"
        ],
        "activation_rate_ci95_high": bootstrap[
            "activation_rate_ci95_high"
        ],
        "net_utility_per_selected_ci95_low": bootstrap[
            "net_utility_per_selected_ci95_low"
        ],
        "net_utility_per_selected_ci95_high": bootstrap[
            "net_utility_per_selected_ci95_high"
        ],
        "net_utility_per_frame_ci95_low": bootstrap[
            "net_utility_per_frame_ci95_low"
        ],
        "net_utility_per_frame_ci95_high": bootstrap[
            "net_utility_per_frame_ci95_high"
        ],
        "cluster_count": int(cluster_names.size),
        "bootstrap_repeats": int(bootstrap_repeats),
    }


def _domain_mask(
    noise_name: np.ndarray,
    *,
    domain: str,
    unseen_noise: Sequence[str],
) -> np.ndarray:
    is_unseen = np.isin(
        np.asarray(noise_name, dtype=str),
        np.asarray(tuple(unseen_noise), dtype=str),
    )
    if domain == "all":
        return np.ones(is_unseen.size, dtype=bool)
    if domain == "seen":
        return ~is_unseen
    if domain == "unseen":
        return is_unseen
    raise ValueError(f"unknown domain: {domain}")


def build_ssr_rows(
    bundle: Mapping[str, Any],
    *,
    ssr_levels: Sequence[float],
    conditions: Sequence[str],
    unseen_noise: Sequence[str],
    frames_per_stratum: int,
    rank_selection: np.ndarray,
    cost_model: Mapping[str, Any],
    resample_seed: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Build the fixed-threshold SSR sweep for all/seen/unseen pools."""
    labels = np.asarray(bundle["labels"], dtype=np.int64)
    short_scores = np.asarray(bundle["short_scores"], dtype=np.float64)
    refined_scores = np.asarray(
        bundle["refined_scores"],
        dtype=np.float64,
    )
    source_key = np.asarray(bundle["source_key"], dtype=str)
    condition = np.asarray(bundle["condition"], dtype=str)
    noise_name = np.asarray(bundle["noise_name"], dtype=str)
    test_mask = np.asarray(bundle["test_mask"], dtype=bool)
    threshold = float(bundle["threshold"])
    rank_selection = np.asarray(rank_selection, dtype=bool).reshape(-1)
    if rank_selection.size != labels.size:
        raise ValueError("rank selection must match the bundle")

    condition_mask = np.isin(
        condition,
        np.asarray(tuple(conditions), dtype=str),
    )
    levels = tuple(sorted(set(float(value) for value in ssr_levels)))
    if not levels:
        raise ValueError("at least one SSR level is required")
    if any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("SSR levels must lie strictly inside (0, 1)")
    if int(frames_per_stratum) <= 0:
        raise ValueError("frames_per_stratum must be positive")

    rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(("all", "seen", "unseen")):
        pool_mask = (
            test_mask
            & condition_mask
            & _domain_mask(
                noise_name,
                domain=domain,
                unseen_noise=unseen_noise,
            )
        )
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
                target_ssr=target_ssr,
                frames_per_stratum=int(frames_per_stratum),
                seed=(
                    int(resample_seed)
                    + 7_919 * domain_index
                    + 1_003 * ssr_index
                ),
            )
            global_indices = pool_indices[sample["indices"]]
            gate_selections = {
                "raw": np.abs(sample["scores"] - 0.5) <= threshold,
                "source_rank": rank_selection[global_indices],
            }
            for gate_name, gate in gate_selections.items():
                metrics = _evaluate_gate(
                    sample["labels"],
                    sample["scores"],
                    refined_scores[global_indices],
                    gate,
                    sample["source_key"],
                    mask=np.ones(sample["frames"], dtype=bool),
                    bootstrap_repeats=bootstrap_repeats,
                    bootstrap_seed=(
                        int(bootstrap_seed)
                        + 10_007 * domain_index
                        + 101 * ssr_index
                        + (0 if gate_name == "raw" else 17)
                    ),
                )
                cost = project_cost(
                    float(metrics["activation_rate"]),
                    cost_model,
                )
                rows.append(
                    {
                        "section": "ssr_sweep",
                        "domain": domain,
                        "condition": "all_noisy",
                        "target_ssr": float(target_ssr),
                        "actual_ssr": float(sample["actual_ssr"]),
                        "gate": gate_name,
                        "gate_setting": (
                            threshold
                            if gate_name == "raw"
                            else float(DEFAULT_RANK_ACTIVATION)
                        ),
                        "frames_per_stratum": int(frames_per_stratum),
                        "strata": len(sample["strata"]),
                        "is_major_development_cell": True,
                        **metrics,
                        **cost,
                    }
                )
    return rows


def build_acoustic_rows(
    bundle: Mapping[str, Any],
    *,
    conditions: Sequence[str],
    unseen_noise: Sequence[str],
    rank_selection: np.ndarray,
    cost_model: Mapping[str, Any],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Audit the fixed gate on unmodified held-out acoustic cells."""
    labels = np.asarray(bundle["labels"], dtype=np.int64)
    short_scores = np.asarray(bundle["short_scores"], dtype=np.float64)
    refined_scores = np.asarray(
        bundle["refined_scores"],
        dtype=np.float64,
    )
    source_key = np.asarray(bundle["source_key"], dtype=str)
    condition = np.asarray(bundle["condition"], dtype=str)
    noise_name = np.asarray(bundle["noise_name"], dtype=str)
    test_mask = np.asarray(bundle["test_mask"], dtype=bool)
    threshold = float(bundle["threshold"])
    rank_selection = np.asarray(rank_selection, dtype=bool).reshape(-1)
    if rank_selection.size != labels.size:
        raise ValueError("rank selection must match the bundle")

    raw_selection = np.abs(short_scores - 0.5) <= threshold
    rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(("all", "seen", "unseen")):
        domain_mask = _domain_mask(
            noise_name,
            domain=domain,
            unseen_noise=unseen_noise,
        )
        cells: list[tuple[str, np.ndarray, bool]] = [
            (
                "overall",
                test_mask & domain_mask,
                False,
            )
        ]
        cells.extend(
            (
                str(value),
                test_mask & domain_mask & (condition == str(value)),
                True,
            )
            for value in conditions
        )
        for cell_index, (cell_name, mask, is_major) in enumerate(cells):
            if not np.any(mask):
                raise ValueError(
                    f"no test frames remain for {domain}/{cell_name}"
                )
            for gate_name, gate in (
                ("raw", raw_selection),
                ("source_rank", rank_selection),
            ):
                metrics = _evaluate_gate(
                    labels,
                    short_scores,
                    refined_scores,
                    gate,
                    source_key,
                    mask=mask,
                    bootstrap_repeats=bootstrap_repeats,
                    bootstrap_seed=(
                        int(bootstrap_seed)
                        + 20_011 * domain_index
                        + 211 * cell_index
                        + (0 if gate_name == "raw" else 29)
                    ),
                )
                cost = project_cost(
                    float(metrics["activation_rate"]),
                    cost_model,
                )
                rows.append(
                    {
                        "section": "acoustic_cell",
                        "domain": domain,
                        "condition": cell_name,
                        "target_ssr": None,
                        "actual_ssr": float(np.mean(labels[mask])),
                        "gate": gate_name,
                        "gate_setting": (
                            threshold
                            if gate_name == "raw"
                            else float(DEFAULT_RANK_ACTIVATION)
                        ),
                        "frames_per_stratum": None,
                        "strata": None,
                        "is_major_development_cell": bool(is_major),
                        **metrics,
                        **cost,
                    }
                )
    return rows


def _failure_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["section"]),
        str(row["domain"]),
        str(row["condition"]),
        (
            None
            if row.get("target_ssr") is None
            else float(row["target_ssr"])
        ),
    )


def _point_gate_failure(row: Mapping[str, Any]) -> bool:
    utility = row.get("net_utility_per_selected")
    activation = row.get("activation_rate")
    return bool(
        utility is None
        or float(utility) <= 0.0
        or activation is None
        or float(activation) > DEFAULT_ACTIVATION_FAILURE
    )


def assess_a12(
    ssr_rows: Sequence[Mapping[str, Any]],
    acoustic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the predeclared A12 utility and activation gates."""
    raw_rows = [
        row
        for row in (*ssr_rows, *acoustic_rows)
        if row["gate"] == "raw"
        and bool(row["is_major_development_cell"])
    ]
    rank_rows = [
        row
        for row in (*ssr_rows, *acoustic_rows)
        if row["gate"] == "source_rank"
        and bool(row["is_major_development_cell"])
    ]
    if not raw_rows:
        raise ValueError("A12 assessment requires major raw-gate rows")
    rank_lookup = {_failure_key(row): row for row in rank_rows}

    point_failures = [row for row in raw_rows if _point_gate_failure(row)]
    unresolved: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    resolved_point_only: list[dict[str, Any]] = []
    for row in point_failures:
        rank = rank_lookup.get(_failure_key(row))
        if rank is not None and not _point_gate_failure(rank):
            resolved.append(row)
            rank_ci_low = rank.get("net_utility_per_selected_ci95_low")
            if rank_ci_low is None or float(rank_ci_low) <= 0.0:
                resolved_point_only.append(row)
        else:
            unresolved.append(row)

    weak_ci = [
        row
        for row in raw_rows
        if row.get("net_utility_per_selected") is not None
        and float(row["net_utility_per_selected"]) > 0.0
        and (
            row.get("net_utility_per_selected_ci95_low") is None
            or float(row["net_utility_per_selected_ci95_low"]) <= 0.0
        )
    ]
    activation_warnings = [
        row
        for row in raw_rows
        if float(row["activation_rate"]) > DEFAULT_ACTIVATION_WARNING
    ]
    activations = [float(row["activation_rate"]) for row in raw_rows]
    max_activation = max(activations)
    min_activation = min(activations)
    if unresolved:
        status = "NO_GO"
        interpretation = (
            "The fixed posterior gate has negative-utility or uncontrolled "
            "activation cells, and the source-rank stabilizer does not "
            "resolve them."
        )
    elif point_failures:
        status = "GO_WITH_SOURCE_RANK_STABILIZER"
        if resolved_point_only:
            interpretation = (
                "The raw gate fails at least one major cell. The simple "
                "source-rank stabilizer is point-positive with bounded "
                "activation in every failed cell, but at least one repaired "
                "speaker-cluster interval still includes zero. Treat this as "
                "conditional router robustness, not a strong GO."
            )
        else:
            interpretation = (
                "The raw gate fails at least one major cell, but the simple "
                "source-rank stabilizer restores positive utility and bounded "
                "activation in every failed cell."
            )
    elif weak_ci:
        status = "PARTIAL_GO"
        interpretation = (
            "All major point utilities are positive, but at least one "
            "speaker-cluster interval includes zero."
        )
    else:
        status = "GO"
        interpretation = (
            "The fixed gate keeps positive utility and bounded activation "
            "across all major cells."
        )

    return {
        "status": status,
        "interpretation": interpretation,
        "major_cell_count": len(raw_rows),
        "point_failure_count": len(point_failures),
        "resolved_failure_count": len(resolved),
        "unresolved_failure_count": len(unresolved),
        "resolved_point_only_count": len(resolved_point_only),
        "resolved_point_only_keys": [
            _failure_key(row) for row in resolved_point_only
        ],
        "weak_ci_count": len(weak_ci),
        "activation_warning_count": len(activation_warnings),
        "activation_failure_count": sum(
            float(row["activation_rate"]) > DEFAULT_ACTIVATION_FAILURE
            for row in raw_rows
        ),
        "max_activation": max_activation,
        "min_activation": min_activation,
        "activation_range": max_activation - min_activation,
        "max_activation_ratio_to_nominal": (
            max_activation / DEFAULT_NOMINAL_ACTIVATION
        ),
        "activation_warning_threshold": DEFAULT_ACTIVATION_WARNING,
        "activation_failure_threshold": DEFAULT_ACTIVATION_FAILURE,
        "failure_keys": [_failure_key(row) for row in point_failures],
        "unresolved_failure_keys": [
            _failure_key(row) for row in unresolved
        ],
        "weak_ci_keys": [_failure_key(row) for row in weak_ci],
        "criteria": {
            "all_major_utility_positive": not point_failures,
            "all_major_utility_ci_low_positive": not weak_ci,
            "activation_below_failure_limit": not any(
                float(row["activation_rate"]) > DEFAULT_ACTIVATION_FAILURE
                for row in raw_rows
            ),
            "simple_stabilizer_resolves_all_failures": (
                bool(point_failures) and not unresolved
            ),
        },
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {str(key) for row in rows for key in row.keys()}
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(payload),
            handle,
            indent=2,
            ensure_ascii=False,
        )


def _pct(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _find_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    section: str,
    domain: str,
    condition: str,
    target_ssr: float | None,
    gate: str,
) -> Mapping[str, Any]:
    for row in rows:
        if (
            row["section"] == section
            and row["domain"] == domain
            and row["condition"] == condition
            and row["gate"] == gate
            and (
                row.get("target_ssr") is None
                if target_ssr is None
                else float(row["target_ssr"]) == float(target_ssr)
            )
        ):
            return row
    raise KeyError(
        f"missing row: {(section, domain, condition, target_ssr, gate)}"
    )


def build_markdown_report(
    *,
    protocol: Mapping[str, Any],
    cost_model: Mapping[str, Any],
    ssr_rows: Sequence[Mapping[str, Any]],
    acoustic_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "# A12 Routing Robustness",
        "",
        "## Protocol",
        "",
        f"- Frozen bundle: `{protocol['bundle']}`",
        f"- Frozen gate threshold: `{float(protocol['threshold']):.6f}`",
        f"- Refined score source: `{protocol['refined_score_field']}`",
        f"- SSR levels: {', '.join(f'{100.0 * float(value):.0f}%' for value in protocol['ssr_levels'])}",
        f"- Acoustic conditions: {', '.join(str(value) for value in protocol['conditions'])}",
        f"- Frames per `(noise_name, condition)` stratum: {protocol['frames_per_stratum']}",
        f"- Source-rank stabilizer percentile: {_pct(protocol['rank_activation'])}",
        "- The gate is never retuned on a seen/unseen or SNR test cell.",
        f"- CPU projection source: `{cost_model['source']}`",
        (
            "- CPU projection: `base_ms + activation * "
            "refinement_ms_per_selected_frame`, calibrated to the A11 5% "
            "measured endpoint; it is not a new hardware benchmark."
        ),
        "",
        "## A12.1 SSR Sweep",
        "",
        "| SSR | Pool | Activation | Net / selected | F1 | CPU ms/frame | CPU / AlwaysRefine |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target_ssr in protocol["ssr_levels"]:
        for domain in ("all", "seen", "unseen"):
            row = _find_row(
                ssr_rows,
                section="ssr_sweep",
                domain=domain,
                condition="all_noisy",
                target_ssr=float(target_ssr),
                gate="raw",
            )
            lines.append(
                f"| {100.0 * float(target_ssr):.0f}% | {domain} | "
                f"{_pct(row['activation_rate'])} | "
                f"{_pct(row['net_utility_per_selected'])} | "
                f"{_number(row['f1'], 5)} | "
                f"{_number(row['estimated_ms_per_frame'], 4)} | "
                f"{_pct(row['estimated_relative_to_always_refine'])} |"
            )

    lines.extend(
        [
            "",
            "## A12.2 Seen/Unseen x SNR",
            "",
            "| SNR | Domain | Raw activation | Raw net / selected | Raw net CI95 | Raw F1 | Rank activation | Rank net / selected |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in protocol["conditions"]:
        for domain in ("seen", "unseen"):
            raw = _find_row(
                acoustic_rows,
                section="acoustic_cell",
                domain=domain,
                condition=str(condition),
                target_ssr=None,
                gate="raw",
            )
            rank = _find_row(
                acoustic_rows,
                section="acoustic_cell",
                domain=domain,
                condition=str(condition),
                target_ssr=None,
                gate="source_rank",
            )
            ci = (
                f"[{_pct(raw['net_utility_per_selected_ci95_low'])}, "
                f"{_pct(raw['net_utility_per_selected_ci95_high'])}]"
            )
            lines.append(
                f"| {condition} | {domain} | "
                f"{_pct(raw['activation_rate'])} | "
                f"{_pct(raw['net_utility_per_selected'])} | {ci} | "
                f"{_number(raw['f1'], 5)} | "
                f"{_pct(rank['activation_rate'])} | "
                f"{_pct(rank['net_utility_per_selected'])} |"
            )

    lines.extend(
        [
            "",
            "## A12.3 Simple Stabilizer",
            "",
            f"- Triggered point failures: {decision['point_failure_count']}",
            f"- Resolved by source-rank gate: {decision['resolved_failure_count']}",
            f"- Unresolved failures: {decision['unresolved_failure_count']}",
            (
                "- Repaired intervals that still include zero: "
                f"{decision['resolved_point_only_count']}"
            ),
            f"- Weak speaker-cluster intervals: {decision['weak_ci_count']}",
            (
                f"- Maximum raw activation: "
                f"{_pct(decision['max_activation'])} "
                f"({_number(decision['max_activation_ratio_to_nominal'], 2)}x nominal)"
            ),
            (
                f"- Activation warning threshold: "
                f"{_pct(decision['activation_warning_threshold'])}; "
                f"failure threshold: "
                f"{_pct(decision['activation_failure_threshold'])}"
            ),
        ]
    )
    if decision["unresolved_failure_keys"]:
        lines.append("- Unresolved keys:")
        lines.extend(
            f"  - `{key}`" for key in decision["unresolved_failure_keys"]
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{decision['status']}**",
            "",
            str(decision["interpretation"]),
            "",
            (
                "The result distinguishes overall pooled utility from "
                "condition-level stability. A positive pooled value does not "
                "satisfy A12 if major development cells have non-positive "
                "selected utility."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def plot_a12(
    *,
    ssr_rows: Sequence[Mapping[str, Any]],
    acoustic_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"all": "#2f6f9f", "seen": "#3f8f62", "unseen": "#bd5b3d"}
    domains = ("all", "seen", "unseen")
    ssr_levels = sorted(
        {
            float(row["target_ssr"])
            for row in ssr_rows
            if row["section"] == "ssr_sweep"
        }
    )

    for domain in domains:
        raw = [
            _find_row(
                ssr_rows,
                section="ssr_sweep",
                domain=domain,
                condition="all_noisy",
                target_ssr=value,
                gate="raw",
            )
            for value in ssr_levels
        ]
        rank = [
            _find_row(
                ssr_rows,
                section="ssr_sweep",
                domain=domain,
                condition="all_noisy",
                target_ssr=value,
                gate="source_rank",
            )
            for value in ssr_levels
        ]
        x = np.asarray(ssr_levels) * 100.0
        axes[0, 0].plot(
            x,
            [100.0 * float(row["activation_rate"]) for row in raw],
            marker="o",
            color=colors[domain],
            label=f"{domain} raw",
        )
        axes[0, 0].plot(
            x,
            [100.0 * float(row["activation_rate"]) for row in rank],
            marker="x",
            linestyle="--",
            color=colors[domain],
            alpha=0.8,
            label=f"{domain} rank",
        )
        axes[0, 1].plot(
            x,
            [
                100.0 * float(row["net_utility_per_selected"])
                for row in raw
            ],
            marker="o",
            color=colors[domain],
            label=f"{domain} raw",
        )
        axes[0, 1].plot(
            x,
            [
                100.0 * float(row["net_utility_per_selected"])
                for row in rank
            ],
            marker="x",
            linestyle="--",
            color=colors[domain],
            alpha=0.8,
            label=f"{domain} rank",
        )

    axes[0, 0].axhline(
        100.0 * DEFAULT_ACTIVATION_FAILURE,
        color="#b33a3a",
        linestyle=":",
        label="20% failure",
    )
    axes[0, 0].set_title("Activation drift under SSR")
    axes[0, 0].set_xlabel("SSR (%)")
    axes[0, 0].set_ylabel("Activation (%)")
    axes[0, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0, 1].set_title("Selected utility under SSR")
    axes[0, 1].set_xlabel("SSR (%)")
    axes[0, 1].set_ylabel("Net utility / selected (%)")

    conditions = tuple(
        str(value)
        for value in sorted(
            {
                row["condition"]
                for row in acoustic_rows
                if row["section"] == "acoustic_cell"
                and row["condition"] != "overall"
                and row["gate"] == "raw"
            },
            key=lambda value: float(value),
        )
    )
    positions = np.arange(len(conditions), dtype=np.float64)
    width = 0.36
    for offset_index, domain in enumerate(("seen", "unseen")):
        raw_rows = [
            _find_row(
                acoustic_rows,
                section="acoustic_cell",
                domain=domain,
                condition=condition,
                target_ssr=None,
                gate="raw",
            )
            for condition in conditions
        ]
        axes[1, 0].bar(
            positions + (offset_index - 0.5) * width,
            [100.0 * float(row["activation_rate"]) for row in raw_rows],
            width=width,
            color=colors[domain],
            label=domain,
        )
        values = [
            100.0 * float(row["net_utility_per_selected"])
            for row in raw_rows
        ]
        axes[1, 1].bar(
            positions + (offset_index - 0.5) * width,
            values,
            width=width,
            color=[
                "#3f8f62" if value > 0.0 else "#b33a3a"
                for value in values
            ],
            alpha=0.9,
            label=domain,
        )

    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[1, 0].axhline(
        100.0 * DEFAULT_ACTIVATION_FAILURE,
        color="#b33a3a",
        linestyle=":",
    )
    axes[1, 0].set_xticks(positions, conditions)
    axes[1, 0].set_title("Fixed-gate activation by acoustic cell")
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[1, 0].set_ylabel("Activation (%)")
    axes[1, 0].legend(frameon=False, ncols=2)
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xticks(positions, conditions)
    axes[1, 1].set_title("Fixed-gate selected utility by acoustic cell")
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("Net utility / selected (%)")
    axes[0, 1].legend(frameon=False, ncols=2, fontsize=8)
    axes[0, 0].legend(frameon=False, ncols=2, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A12 fixed-gate routing robustness audit."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
    )
    parser.add_argument(
        "--a11-results",
        type=Path,
        default=DEFAULT_A11_RESULTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--ssr-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_SSR_LEVELS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument(
        "--unseen-noise",
        nargs="+",
        default=list(UNSEEN_NOISE),
    )
    parser.add_argument(
        "--frames-per-stratum",
        type=int,
        default=DEFAULT_FRAMES_PER_STRATUM,
    )
    parser.add_argument(
        "--rank-activation",
        type=float,
        default=DEFAULT_RANK_ACTIVATION,
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260919,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260920,
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    ssr_levels = tuple(float(value) for value in args.ssr_levels)
    if not ssr_levels or any(
        not 0.0 < value < 1.0 for value in ssr_levels
    ):
        parser.error("--ssr-levels values must lie strictly inside (0, 1)")
    if not args.conditions:
        parser.error("--conditions must not be empty")
    if not args.unseen_noise:
        parser.error("--unseen-noise must not be empty")
    if int(args.frames_per_stratum) <= 0:
        parser.error("--frames-per-stratum must be positive")
    if not 0.0 < float(args.rank_activation) <= 1.0:
        parser.error("--rank-activation must be in (0, 1]")
    if int(args.bootstrap_repeats) <= 0:
        parser.error("--bootstrap-repeats must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    bundle = load_a12_bundle(args.bundle)
    cost_model = load_a11_cost_model(args.a11_results)
    test_mask = np.asarray(bundle["test_mask"], dtype=bool)
    ranks = source_rank_uncertainty(
        bundle["short_scores"],
        bundle["source_key"],
        mask=test_mask,
    )
    rank_selection = source_rank_selection(
        ranks,
        mask=test_mask,
        activation_rate=float(args.rank_activation),
    )
    ssr_rows = build_ssr_rows(
        bundle,
        ssr_levels=tuple(float(value) for value in args.ssr_levels),
        conditions=tuple(str(value) for value in args.conditions),
        unseen_noise=tuple(str(value) for value in args.unseen_noise),
        frames_per_stratum=int(args.frames_per_stratum),
        rank_selection=rank_selection,
        cost_model=cost_model,
        resample_seed=int(args.seed),
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    acoustic_rows = build_acoustic_rows(
        bundle,
        conditions=tuple(str(value) for value in args.conditions),
        unseen_noise=tuple(str(value) for value in args.unseen_noise),
        rank_selection=rank_selection,
        cost_model=cost_model,
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    decision = assess_a12(ssr_rows, acoustic_rows)

    protocol = {
        "bundle": str(args.bundle),
        "threshold": float(bundle["threshold"]),
        "threshold_source": "frozen_bundle",
        "refined_score_field": "full_adaptive_scores",
        "ssr_levels": [float(value) for value in args.ssr_levels],
        "conditions": [str(value) for value in args.conditions],
        "unseen_noise": [str(value) for value in args.unseen_noise],
        "frames_per_stratum": int(args.frames_per_stratum),
        "rank_activation": float(args.rank_activation),
        "activation_warning_threshold": DEFAULT_ACTIVATION_WARNING,
        "activation_failure_threshold": DEFAULT_ACTIVATION_FAILURE,
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "seed": int(args.seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "test_frames": int(np.count_nonzero(bundle["test_mask"])),
        "rank_selected_frames": int(np.count_nonzero(rank_selection)),
        "rank_test_activation": float(
            np.count_nonzero(rank_selection)
            / np.count_nonzero(bundle["test_mask"])
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = [*ssr_rows, *acoustic_rows]
    csv_path = output_dir / "a12_routing_robustness.csv"
    json_path = output_dir / "a12_routing_robustness.json"
    markdown_path = output_dir / "a12_routing_robustness.md"
    plot_path = output_dir / "a12_routing_robustness.png"
    _write_csv(csv_path, combined_rows)
    _write_json(
        json_path,
        {
            "protocol": protocol,
            "cost_model": cost_model,
            "ssr_rows": ssr_rows,
            "acoustic_rows": acoustic_rows,
            "decision": decision,
        },
    )
    report = build_markdown_report(
        protocol=protocol,
        cost_model=cost_model,
        ssr_rows=ssr_rows,
        acoustic_rows=acoustic_rows,
        decision=decision,
    )
    markdown_path.write_text(report, encoding="utf-8")
    plot_a12(
        ssr_rows=ssr_rows,
        acoustic_rows=acoustic_rows,
        output_path=plot_path,
    )

    print(f"A12 status: {decision['status']}")
    print(
        f"Major failures: {decision['point_failure_count']}; "
        f"resolved by source-rank: {decision['resolved_failure_count']}; "
        f"unresolved: {decision['unresolved_failure_count']}"
    )
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
