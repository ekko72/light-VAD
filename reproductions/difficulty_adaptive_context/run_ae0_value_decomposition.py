# -*- coding: utf-8 -*-
"""AE0: decompose opened A14 OOD utility into value-failure layers.

AE0 is deliberately post-hoc.  It reuses the frozen A14 lock, sealed-row
loader, and inference interface, but it does not train, select, or tune any
model, router, threshold, or stabilizer.  The only inference pass is the
seed-17 reconstruction needed to recover all-frame correction/harm outcomes
that were not stored in the A14 result JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
    predict_full_adaptive_frames,
)
from reproductions.difficulty_adaptive_context.run_a14_final_ood import (
    CELL_ORDER,
    DEFAULT_DATA_ROOT,
    EXPECTED_FRAMES,
    EXPECTED_ROWS,
    EXPECTED_SOURCES,
    EXPECTED_SPEAKER_KEYS,
    PRIMARY_SEED,
    SCORE_CHUNK_FRAMES,
    ValidatedFinalOOD,
    _is_unseen_row,
    _raw_gate_selection,
    _row_paths,
    _snr,
    _source_key,
    _speaker_key,
    file_sha256,
    load_sealed_final_ood_rows,
    validate_final_ood_lock,
)
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "reproductions"
    / "difficulty_adaptive_context"
    / "ae0_protocol.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ae0_value_decomposition"
)

PROTOCOL_VERSION = "A-v2-AE0-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_AE0_OUTCOME_ANALYSIS"
PRIMARY_SEED_EXPECTED = 17
DECISION_THRESHOLD = 0.5
GATE_THRESHOLD = 0.13
BOOTSTRAP_REPEATS = 5_000
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_SEED_RULES = {
    "aggregate_source_cluster": 20260919,
    "speaker_cluster_sensitivity": "base + 1",
    "domain_source_cluster": "base + 10 + domain_index",
    "cell_source_cluster": "base + 100 + cell_index",
}

STAT_COLUMNS = (
    "frames",
    "correction_all",
    "harm_all",
    "selected",
    "selected_correction",
    "selected_harm",
)
METRIC_COLUMNS = (
    "activation_rate",
    "availability_gross",
    "availability_net",
    "gate_utility",
    "observability_auroc",
    "actionability_ratio",
    "harm_auroc",
    "selected_correction_recall",
    "selected_harm_rate",
)
DOMAIN_ORDER = ("seen", "unseen")
GROUP_ORDER: tuple[Any, ...] = (
    "all",
    "seen",
    "unseen",
    *((domain, snr) for domain in DOMAIN_ORDER for snr in (
        "-5",
        "0",
        "5",
        "10",
        "15",
        "20",
    )),
)


@dataclass(frozen=True)
class Ae0Config:
    """Validated AE0 protocol."""

    path: Path
    payload: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class FrameBundle:
    """Frame-level seed-17 reconstruction used by AE0."""

    labels: np.ndarray
    short_scores: np.ndarray
    refined_scores: np.ndarray
    selected: np.ndarray
    source_codes: np.ndarray
    speaker_codes: np.ndarray
    domain_codes: np.ndarray
    cell_codes: np.ndarray
    source_order: tuple[str, ...]
    speaker_order: tuple[str, ...]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _verify_hash(path: Path, expected: str, *, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path).upper()
    if actual != str(expected).upper():
        raise ValueError(
            f"AE0 {role} hash mismatch: {actual} != {expected}"
        )
    return actual


def load_config(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> Ae0Config:
    """Validate the frozen AE0 protocol and all A14 input hashes."""
    path = Path(protocol_path)
    payload = _load_json(path)
    if payload.get("protocol_id") != PROTOCOL_VERSION:
        raise ValueError("unsupported AE0 protocol version")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("AE0 protocol is not frozen before outcome analysis")

    study = payload.get("study", {})
    required_study = {
        "analysis_type": "post_hoc_scientific_analysis",
        "training": "none",
        "router_search": "forbidden",
        "gate_tuning": "forbidden",
        "threshold_change": "forbidden",
        "model_change": "forbidden",
        "a14_rerun": "forbidden",
        "a15": "forbidden",
        "seed_variation_role": (
            "excluded_from_AE0; primary AE0 inference uses seed 17 only"
        ),
    }
    for key, expected in required_study.items():
        if study.get(key) != expected:
            raise ValueError(f"AE0 study field {key!r} differs from freeze")

    reconstruction = payload.get("reconstruction", {})
    expected_reconstruction = {
        "required_call": "predict_full_adaptive_frames",
        "required_assertion": "np.all(stream_selected)",
        "decision_threshold": DECISION_THRESHOLD,
        "gate_threshold": GATE_THRESHOLD,
        "score_chunk_frames": SCORE_CHUNK_FRAMES,
        "frame_hop": FRAME_HOP,
        "expected_frames": EXPECTED_FRAMES,
        "expected_rows": EXPECTED_ROWS,
        "expected_sources": EXPECTED_SOURCES,
        "expected_speaker_sensitivity_keys": EXPECTED_SPEAKER_KEYS,
    }
    for key, expected in expected_reconstruction.items():
        if reconstruction.get(key) != expected:
            raise ValueError(
                f"AE0 reconstruction field {key!r} differs from freeze"
            )

    primary = payload.get("primary_seed", {})
    if int(primary.get("seed", -1)) != PRIMARY_SEED_EXPECTED:
        raise ValueError("AE0 primary seed differs from freeze")
    if primary.get("short_encoder_relation") != "shared_frozen_short_encoder":
        raise ValueError("AE0 primary seed relation differs from freeze")

    a14 = payload.get("a14_freeze", {})
    hash_specs = (
        ("A14 protocol", "protocol_path", "protocol_sha256"),
        ("A14 result", "result_path", "result_sha256"),
        ("A14 per-cell", "per_cell_path", "per_cell_sha256"),
        ("A14 per-seed", "per_seed_path", "per_seed_sha256"),
        ("A14 runner", "runner_path", "runner_sha256"),
    )
    for role, path_key, hash_key in hash_specs:
        _verify_hash(
            _resolve_path(str(a14.get(path_key, ""))),
            str(a14.get(hash_key, "")),
            role=role,
        )
    _verify_hash(
        _resolve_path(str(primary.get("checkpoint_path", ""))),
        str(primary.get("checkpoint_sha256", "")),
        role="seed17 checkpoint",
    )

    cells = payload.get("cells", {}).get("order", [])
    if cells != [list(cell) for cell in CELL_ORDER]:
        raise ValueError("AE0 cell order differs from the frozen A14 order")
    bootstrap = payload.get("bootstrap", {})
    if int(bootstrap.get("repeats", -1)) != BOOTSTRAP_REPEATS:
        raise ValueError("AE0 bootstrap repeats differ from freeze")
    if int(bootstrap.get("seed", -1)) != BOOTSTRAP_SEED:
        raise ValueError("AE0 bootstrap seed differs from freeze")
    if bootstrap.get("primary_unit") != "source_cluster":
        raise ValueError("AE0 primary bootstrap unit differs from freeze")
    if bootstrap.get("sensitivity_unit") != "speaker_cluster":
        raise ValueError("AE0 sensitivity bootstrap unit differs from freeze")
    if bootstrap.get("seed_rules") != BOOTSTRAP_SEED_RULES:
        raise ValueError("AE0 bootstrap seed rules differ from freeze")

    classification = payload.get("classification", {}).get("precedence", [])
    expected_rules = [
        ("AVAILABILITY_LIMITED", "A_net_c <= 0"),
        ("OBSERVABILITY_LIMITED", "A_net_c > 0 and O_c <= 0.5"),
        (
            "ACTIONABILITY_LIMITED",
            "A_net_c > 0 and O_c > 0.5 and U_gate_c <= 0",
        ),
        (
            "ACTIONABLE",
            "A_net_c > 0 and O_c > 0.5 and U_gate_c > 0",
        ),
    ]
    if [
        (str(rule.get("label", "")), str(rule.get("rule", "")))
        for rule in classification
    ] != expected_rules:
        raise ValueError("AE0 classification precedence differs from freeze")

    return Ae0Config(path=path, payload=payload, sha256=file_sha256(path).upper())


def average_rank_auroc(
    positive: np.ndarray,
    scores: np.ndarray,
) -> float | None:
    """AUROC with average ranks, treating higher scores as positive."""
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive.size != scores.size:
        raise ValueError("positive and scores must have equal length")
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(positive.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positive].sum())
    denominator = float(n_positive * n_negative)
    return (
        rank_sum - float(n_positive * (n_positive + 1)) / 2.0
    ) / denominator


def _pair_wins(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    """Return the average-rank pair-win count for one cluster pair."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if positive_scores.size == 0 or negative_scores.size == 0:
        return 0.0
    negative_sorted = np.sort(negative_scores)
    less = np.searchsorted(negative_sorted, positive_scores, side="left")
    equal = (
        np.searchsorted(negative_sorted, positive_scores, side="right")
        - less
    )
    return float(np.sum(less + 0.5 * equal))


def auroc_pair_matrix(
    positive_by_cluster: Sequence[np.ndarray],
    negative_by_cluster: Sequence[np.ndarray],
) -> np.ndarray:
    """Precompute pooled AUROC pair contributions for cluster resampling."""
    if len(positive_by_cluster) != len(negative_by_cluster):
        raise ValueError("positive and negative cluster lists must match")
    count = len(positive_by_cluster)
    positive_sorted = [
        np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
        for values in positive_by_cluster
    ]
    negative_sorted = [
        np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
        for values in negative_by_cluster
    ]
    matrix = np.zeros((count, count), dtype=np.float64)
    for positive_index, positive in enumerate(positive_sorted):
        if positive.size == 0:
            continue
        for negative_index, negative in enumerate(negative_sorted):
            if negative.size == 0:
                continue
            less = np.searchsorted(negative, positive, side="left")
            equal = (
                np.searchsorted(negative, positive, side="right") - less
            )
            matrix[positive_index, negative_index] = float(
                np.sum(less + 0.5 * equal)
            )
    return matrix


def _mask_for_group(
    bundle: FrameBundle,
    group: Any,
) -> np.ndarray:
    if group == "all":
        return np.ones(bundle.labels.size, dtype=bool)
    if group in ("seen", "unseen"):
        return bundle.domain_codes == DOMAIN_ORDER.index(str(group))
    if isinstance(group, tuple) and len(group) == 2:
        domain, snr = group
        cell = (str(domain), str(snr))
        if cell not in CELL_ORDER:
            raise ValueError(f"unknown AE0 cell: {cell}")
        return bundle.cell_codes == CELL_ORDER.index(cell)
    raise ValueError(f"unknown AE0 group: {group}")


def _cluster_stats(
    *,
    frames: int,
    correction: np.ndarray,
    harm: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    stats = np.zeros((int(n_clusters), len(STAT_COLUMNS)), dtype=np.int64)
    stats[:, 0] = np.bincount(cluster_codes, minlength=n_clusters)
    stats[:, 1] = np.bincount(
        cluster_codes,
        weights=correction.astype(np.int64),
        minlength=n_clusters,
    ).astype(np.int64)
    stats[:, 2] = np.bincount(
        cluster_codes,
        weights=harm.astype(np.int64),
        minlength=n_clusters,
    ).astype(np.int64)
    stats[:, 3] = np.bincount(
        cluster_codes,
        weights=selected.astype(np.int64),
        minlength=n_clusters,
    ).astype(np.int64)
    stats[:, 4] = np.bincount(
        cluster_codes,
        weights=(selected & correction).astype(np.int64),
        minlength=n_clusters,
    ).astype(np.int64)
    stats[:, 5] = np.bincount(
        cluster_codes,
        weights=(selected & harm).astype(np.int64),
        minlength=n_clusters,
    ).astype(np.int64)
    if int(stats[:, 0].sum()) != int(frames):
        raise RuntimeError("cluster frame count is inconsistent")
    return stats


def _event_scores_by_cluster(
    *,
    event: np.ndarray,
    uncertainty: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> list[np.ndarray]:
    event = np.asarray(event, dtype=bool)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64)
    result: list[np.ndarray] = []
    for cluster in range(int(n_clusters)):
        mask = (cluster_codes == cluster) & event
        result.append(
            np.ascontiguousarray(uncertainty[mask], dtype=np.float64)
        )
    return result


def _point_metrics(
    stats: np.ndarray,
    correction_pair_matrix: np.ndarray,
    harm_pair_matrix: np.ndarray,
) -> dict[str, float | int | None]:
    total = np.asarray(stats, dtype=np.int64).sum(axis=0)
    frames = int(total[0])
    correction = int(total[1])
    harm = int(total[2])
    selected = int(total[3])
    selected_correction = int(total[4])
    selected_harm = int(total[5])
    if frames == 0:
        return {
            "frames": 0,
            "correction_all": 0,
            "harm_all": 0,
            "selected": 0,
            "selected_correction": 0,
            "selected_harm": 0,
            **{name: None for name in METRIC_COLUMNS},
        }
    availability_gross = correction / frames
    availability_net = (correction - harm) / frames
    gate_utility = (selected_correction - selected_harm) / frames
    correction_denominator = correction * (frames - correction)
    harm_denominator = harm * (frames - harm)
    observability = (
        float(correction_pair_matrix.sum()) / correction_denominator
        if correction_denominator
        else None
    )
    harm_auroc = (
        float(harm_pair_matrix.sum()) / harm_denominator
        if harm_denominator
        else None
    )
    return {
        "frames": frames,
        "correction_all": correction,
        "harm_all": harm,
        "selected": selected,
        "selected_correction": selected_correction,
        "selected_harm": selected_harm,
        "activation_rate": selected / frames,
        "availability_gross": availability_gross,
        "availability_net": availability_net,
        "gate_utility": gate_utility,
        "observability_auroc": observability,
        "actionability_ratio": (
            gate_utility / availability_net
            if availability_net > 0.0
            else None
        ),
        "harm_auroc": harm_auroc,
        "selected_correction_recall": (
            selected_correction / correction if correction else None
        ),
        "selected_harm_rate": (
            selected_harm / harm if harm else None
        ),
    }


def _bootstrap_metrics(
    stats: np.ndarray,
    correction_pair_matrix: np.ndarray,
    harm_pair_matrix: np.ndarray,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap cluster resamples while reusing each resample for all metrics."""
    if int(repeats) <= 0:
        raise ValueError("bootstrap repeats must be positive")
    stats = np.asarray(stats, dtype=np.int64)
    n_clusters = int(stats.shape[0])
    if n_clusters == 0:
        raise ValueError("bootstrap requires at least one cluster")
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {name: [] for name in METRIC_COLUMNS}
    for _ in range(int(repeats)):
        sampled = rng.choice(
            np.arange(n_clusters, dtype=np.int64),
            size=n_clusters,
            replace=True,
        )
        multiplicity = np.bincount(
            sampled,
            minlength=n_clusters,
        ).astype(np.float64)
        frames = float(multiplicity @ stats[:, 0])
        if frames <= 0.0:
            continue
        correction = float(multiplicity @ stats[:, 1])
        harm = float(multiplicity @ stats[:, 2])
        selected = float(multiplicity @ stats[:, 3])
        selected_correction = float(multiplicity @ stats[:, 4])
        selected_harm = float(multiplicity @ stats[:, 5])
        availability_net = (correction - harm) / frames
        gate_utility = (selected_correction - selected_harm) / frames
        correction_denominator = correction * (frames - correction)
        harm_denominator = harm * (frames - harm)
        observability = (
            float(multiplicity @ correction_pair_matrix @ multiplicity)
            / correction_denominator
            if correction_denominator > 0.0
            else None
        )
        harm_auroc = (
            float(multiplicity @ harm_pair_matrix @ multiplicity)
            / harm_denominator
            if harm_denominator > 0.0
            else None
        )
        values = {
            "activation_rate": selected / frames,
            "availability_gross": correction / frames,
            "availability_net": availability_net,
            "gate_utility": gate_utility,
            "observability_auroc": observability,
            "actionability_ratio": (
                gate_utility / availability_net
                if availability_net > 0.0
                else None
            ),
            "harm_auroc": harm_auroc,
            "selected_correction_recall": (
                selected_correction / correction if correction > 0.0 else None
            ),
            "selected_harm_rate": (
                selected_harm / harm if harm > 0.0 else None
            ),
        }
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))

    point = _point_metrics(
        stats,
        correction_pair_matrix,
        harm_pair_matrix,
    )
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


def classify_layer(
    *,
    availability_net: float | None,
    observability_auroc: float | None,
    gate_utility: float | None,
) -> str:
    """Apply the frozen AE0 classification precedence."""
    if availability_net is None or availability_net <= 0.0:
        return "AVAILABILITY_LIMITED"
    if observability_auroc is None or observability_auroc <= 0.5:
        return "OBSERVABILITY_LIMITED"
    if gate_utility is None or gate_utility <= 0.0:
        return "ACTIONABILITY_LIMITED"
    return "ACTIONABLE"


def _group_label(group: Any) -> str:
    if isinstance(group, tuple) and len(group) == 2:
        return f"{group[0]}/{group[1]}"
    return str(group)


def _sufficient_stats(
    *,
    bundle: FrameBundle,
    mask: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != bundle.labels.size:
        raise ValueError("group mask must match frame arrays")
    selected = bundle.selected[mask]
    labels = bundle.labels[mask]
    short_wrong = (bundle.short_scores[mask] >= DECISION_THRESHOLD) != labels
    refined_wrong = (
        bundle.refined_scores[mask] >= DECISION_THRESHOLD
    ) != labels
    correction = short_wrong & ~refined_wrong
    harm = ~short_wrong & refined_wrong
    uncertainty = np.abs(bundle.short_scores[mask] - DECISION_THRESHOLD)
    local_clusters = np.asarray(cluster_codes, dtype=np.int64)[mask]
    stats = _cluster_stats(
        frames=int(mask.sum()),
        correction=correction,
        harm=harm,
        selected=selected,
        cluster_codes=local_clusters,
        n_clusters=n_clusters,
    )
    correction_pair_matrix = auroc_pair_matrix(
        _event_scores_by_cluster(
            event=correction,
            uncertainty=uncertainty,
            cluster_codes=local_clusters,
            n_clusters=n_clusters,
        ),
        _event_scores_by_cluster(
            event=~correction,
            uncertainty=uncertainty,
            cluster_codes=local_clusters,
            n_clusters=n_clusters,
        ),
    )
    harm_pair_matrix = auroc_pair_matrix(
        _event_scores_by_cluster(
            event=harm,
            uncertainty=uncertainty,
            cluster_codes=local_clusters,
            n_clusters=n_clusters,
        ),
        _event_scores_by_cluster(
            event=~harm,
            uncertainty=uncertainty,
            cluster_codes=local_clusters,
            n_clusters=n_clusters,
        ),
    )
    return stats, correction_pair_matrix, harm_pair_matrix


def _counts_from_bundle(
    bundle: FrameBundle,
    mask: np.ndarray,
) -> dict[str, int]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    labels = bundle.labels[mask]
    short_wrong = (bundle.short_scores[mask] >= DECISION_THRESHOLD) != labels
    refined_wrong = (
        bundle.refined_scores[mask] >= DECISION_THRESHOLD
    ) != labels
    correction = short_wrong & ~refined_wrong
    harm = ~short_wrong & refined_wrong
    selected = bundle.selected[mask]
    return {
        "frames": int(mask.sum()),
        "selected": int(np.count_nonzero(selected)),
        "correction": int(np.count_nonzero(selected & correction)),
        "harm": int(np.count_nonzero(selected & harm)),
        "correction_all": int(np.count_nonzero(correction)),
        "harm_all": int(np.count_nonzero(harm)),
    }


def validate_reconstruction(
    bundle: FrameBundle,
    config: Ae0Config,
) -> dict[str, Any]:
    """Abort before interpretation if any frozen A14 count is not reproduced."""
    expected = config.payload["integrity_checks"]
    observed: dict[str, Any] = {}
    all_counts = _counts_from_bundle(
        bundle,
        np.ones(bundle.labels.size, dtype=bool),
    )
    observed["all"] = {
        "selected": all_counts["selected"],
        "correction": all_counts["correction"],
        "harm": all_counts["harm"],
    }
    for domain in DOMAIN_ORDER:
        observed[domain] = {
            key: value
            for key, value in _counts_from_bundle(
                bundle,
                _mask_for_group(bundle, domain),
            ).items()
            if key in {"selected", "correction", "harm"}
        }
    observed["per_cell"] = []
    for cell in CELL_ORDER:
        counts = _counts_from_bundle(
            bundle,
            _mask_for_group(bundle, cell),
        )
        observed["per_cell"].append(
            {
                "domain": cell[0],
                "snr_db": cell[1],
                "selected": counts["selected"],
                "correction": counts["correction"],
                "harm": counts["harm"],
            }
        )

    for name in ("all", "seen", "unseen"):
        expected_row = {
            key: int(expected[name][key])
            for key in ("selected", "correction", "harm")
        }
        if observed[name] != expected_row:
            raise ValueError(
                f"AE0 integrity failure for {name}: "
                f"{observed[name]} != {expected_row}"
            )
    expected_cells = [
        {
            "domain": str(row["domain"]),
            "snr_db": str(row["snr_db"]),
            "selected": int(row["selected"]),
            "correction": int(row["correction"]),
            "harm": int(row["harm"]),
        }
        for row in expected["per_cell"]
    ]
    if observed["per_cell"] != expected_cells:
        raise ValueError(
            "AE0 integrity failure for per-cell counts: "
            f"{observed['per_cell']} != {expected_cells}"
        )
    return observed


def reconstruct_seed17(
    validated: ValidatedFinalOOD,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    device: str | torch.device = "auto",
    progress_every: int = 10,
) -> FrameBundle:
    """Re-run the frozen seed-17 primary inference to recover all-frame value."""
    rows, metadata = load_sealed_final_ood_rows(validated)
    resolved_device = (
        device if isinstance(device, torch.device) else resolve_device(device)
    )
    checkpoint = next(
        record
        for record in validated.lock_record["verified_checkpoints"]
        if int(record["seed"]) == PRIMARY_SEED_EXPECTED
    )
    model, _, _ = load_adaptive_model(
        Path(str(checkpoint["path"])),
        resolved_device,
    )
    frontend = MfccFrontend(MfccConfig(causal=True)).to(resolved_device)
    source_order = tuple(str(value) for value in metadata["selected_source_keys"])
    speaker_order = tuple(
        str(value) for value in metadata["speaker_sensitivity_keys"]
    )
    source_index = {name: index for index, name in enumerate(source_order)}
    speaker_index = {name: index for index, name in enumerate(speaker_order)}
    data_root = Path(data_root)
    parts: dict[str, list[np.ndarray]] = {
        "labels": [],
        "short_scores": [],
        "refined_scores": [],
        "selected": [],
        "source_codes": [],
        "speaker_codes": [],
        "domain_codes": [],
        "cell_codes": [],
    }
    evaluated_frames = 0
    for index, row in enumerate(rows):
        audio_path, label_path = _row_paths(row, data_root)
        waveform = read_int16_audio(audio_path)
        sample_labels = np.load(label_path)
        if sample_labels.size != waveform.size:
            raise ValueError(
                f"label/audio mismatch for {audio_path}: "
                f"{sample_labels.size} != {waveform.size}"
            )
        n_frames = waveform.size // FRAME_HOP + 1
        labels = causal_frame_labels(sample_labels, n_frames)
        refined_scores, embedded_short, stream_selected, _ = (
            predict_full_adaptive_frames(
                model,
                frontend,
                waveform,
                device=resolved_device,
                chunk_frames=SCORE_CHUNK_FRAMES,
            )
        )
        if refined_scores.size != n_frames or embedded_short.size != n_frames:
            raise RuntimeError("adaptive output frame count is inconsistent")
        if not np.all(stream_selected):
            raise RuntimeError(
                "permissive refinement pass did not cover every frame"
            )
        source = _source_key(row)
        speaker = _speaker_key(row)
        domain = "unseen" if _is_unseen_row(row) else "seen"
        cell = (domain, _snr(row))
        if source not in source_index:
            raise ValueError(f"unexpected AE0 source: {source}")
        if speaker not in speaker_index:
            raise ValueError(f"unexpected AE0 speaker: {speaker}")
        if cell not in CELL_ORDER:
            raise ValueError(f"unexpected AE0 cell: {cell}")
        parts["labels"].append(labels)
        parts["short_scores"].append(embedded_short)
        parts["refined_scores"].append(refined_scores)
        parts["selected"].append(_raw_gate_selection(embedded_short))
        parts["source_codes"].append(
            np.full(labels.size, source_index[source], dtype=np.int64)
        )
        parts["speaker_codes"].append(
            np.full(labels.size, speaker_index[speaker], dtype=np.int64)
        )
        parts["domain_codes"].append(
            np.full(
                labels.size,
                DOMAIN_ORDER.index(domain),
                dtype=np.int64,
            )
        )
        parts["cell_codes"].append(
            np.full(labels.size, CELL_ORDER.index(cell), dtype=np.int64)
        )
        evaluated_frames += int(labels.size)
        if progress_every > 0 and (
            (index + 1) % int(progress_every) == 0
            or index + 1 == len(rows)
        ):
            print(
                f"AE0 seed17: evaluated {index + 1}/{len(rows)} rows",
                flush=True,
            )
    if evaluated_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"AE0 evaluated {evaluated_frames} frames, "
            f"expected {EXPECTED_FRAMES}"
        )
    if not parts["labels"]:
        raise RuntimeError("AE0 reconstruction produced no frames")
    bundle = FrameBundle(
        labels=np.concatenate(parts["labels"]),
        short_scores=np.concatenate(parts["short_scores"]),
        refined_scores=np.concatenate(parts["refined_scores"]),
        selected=np.concatenate(parts["selected"]).astype(bool),
        source_codes=np.concatenate(parts["source_codes"]),
        speaker_codes=np.concatenate(parts["speaker_codes"]),
        domain_codes=np.concatenate(parts["domain_codes"]),
        cell_codes=np.concatenate(parts["cell_codes"]),
        source_order=source_order,
        speaker_order=speaker_order,
    )
    return bundle


def build_group_results(
    bundle: FrameBundle,
    config: Ae0Config,
    *,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    """Compute point and cluster-bootstrap AE0 metrics for every frozen group."""
    results: dict[str, Any] = {}
    domain_indices = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
    for group in GROUP_ORDER:
        mask = _mask_for_group(bundle, group)
        source_stats, source_corr_pair, source_harm_pair = _sufficient_stats(
            bundle=bundle,
            mask=mask,
            cluster_codes=bundle.source_codes,
            n_clusters=len(bundle.source_order),
        )
        speaker_stats, speaker_corr_pair, speaker_harm_pair = _sufficient_stats(
            bundle=bundle,
            mask=mask,
            cluster_codes=bundle.speaker_codes,
            n_clusters=len(bundle.speaker_order),
        )
        if isinstance(group, tuple) and len(group) == 2:
            cell_index = CELL_ORDER.index((str(group[0]), str(group[1])))
            source_seed = (
                BOOTSTRAP_SEED
                + 100
                + cell_index
            )
        elif group in domain_indices:
            source_seed = (
                BOOTSTRAP_SEED
                + 10
                + domain_indices[str(group)]
            )
        else:
            source_seed = BOOTSTRAP_SEED
        source_bootstrap = _bootstrap_metrics(
            source_stats,
            source_corr_pair,
            source_harm_pair,
            repeats=bootstrap_repeats,
            seed=source_seed,
        )
        speaker_bootstrap = _bootstrap_metrics(
            speaker_stats,
            speaker_corr_pair,
            speaker_harm_pair,
            repeats=bootstrap_repeats,
            seed=BOOTSTRAP_SEED + 1,
        )
        point = source_bootstrap["point"]
        layer = classify_layer(
            availability_net=point["availability_net"],
            observability_auroc=point["observability_auroc"],
            gate_utility=point["gate_utility"],
        )
        results[_group_label(group)] = {
            "group": _group_label(group),
            "domain": (
                None if group == "all"
                else str(group)
                if group in domain_indices
                else str(group[0])
            ),
            "snr_db": (
                None if not isinstance(group, tuple) else str(group[1])
            ),
            "classification": layer,
            "point": point,
            "source_cluster_bootstrap": source_bootstrap,
            "speaker_cluster_bootstrap": speaker_bootstrap,
        }
    return results


def _cell_rows(group_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        label = f"{cell[0]}/{cell[1]}"
        row = group_results[label]
        point = row["point"]
        source = row["source_cluster_bootstrap"]
        speaker = row["speaker_cluster_bootstrap"]
        rows.append(
            {
                "domain": cell[0],
                "snr_db": cell[1],
                "classification": row["classification"],
                "frames": point["frames"],
                "correction_all": point["correction_all"],
                "harm_all": point["harm_all"],
                "selected": point["selected"],
                "selected_correction": point["selected_correction"],
                "selected_harm": point["selected_harm"],
                "activation_rate": point["activation_rate"],
                "availability_gross": point["availability_gross"],
                "availability_net": point["availability_net"],
                "gate_utility": point["gate_utility"],
                "observability_auroc": point["observability_auroc"],
                "actionability_ratio": point["actionability_ratio"],
                "harm_auroc": point["harm_auroc"],
                "selected_correction_recall": point[
                    "selected_correction_recall"
                ],
                "selected_harm_rate": point["selected_harm_rate"],
                "source_ci95_low_gate_utility": source[
                    "gate_utility_ci95_low"
                ],
                "source_ci95_high_gate_utility": source[
                    "gate_utility_ci95_high"
                ],
                "source_ci95_low_observability_auroc": source[
                    "observability_auroc_ci95_low"
                ],
                "source_ci95_high_observability_auroc": source[
                    "observability_auroc_ci95_high"
                ],
                "speaker_ci95_low_gate_utility": speaker[
                    "gate_utility_ci95_low"
                ],
                "speaker_ci95_high_gate_utility": speaker[
                    "gate_utility_ci95_high"
                ],
            }
        )
    return rows


def _layer_counts(
    group_results: Mapping[str, Any],
) -> dict[str, int]:
    counts = {
        "AVAILABILITY_LIMITED": 0,
        "OBSERVABILITY_LIMITED": 0,
        "ACTIONABILITY_LIMITED": 0,
        "ACTIONABLE": 0,
    }
    for cell in CELL_ORDER:
        label = f"{cell[0]}/{cell[1]}"
        counts[str(group_results[label]["classification"])] += 1
    return counts


def _summary(
    *,
    config: Ae0Config,
    integrity: Mapping[str, Any],
    group_results: Mapping[str, Any],
    bundle: FrameBundle,
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    layer_counts = _layer_counts(group_results)
    stop_rule = config.payload["stop_rule"]
    stop_triggered = bool(
        layer_counts["AVAILABILITY_LIMITED"] > 0
        and layer_counts["OBSERVABILITY_LIMITED"] == 0
        and layer_counts["ACTIONABILITY_LIMITED"] == 0
    )
    return {
        "version": PROTOCOL_VERSION,
        "date": "2026-09-19",
        "protocol": {
            "path": str(config.path),
            "sha256": config.sha256,
        },
        "scope": {
            "analysis_type": "post_hoc_scientific_analysis",
            "primary_seed": PRIMARY_SEED_EXPECTED,
            "seed_variation": "excluded",
            "training": "none",
            "router_search": "forbidden",
            "gate_tuning": "forbidden",
            "a14_rerun": "forbidden",
            "a15": "forbidden",
        },
        "inputs": {
            "a14_protocol_sha256": config.payload["a14_freeze"][
                "protocol_sha256"
            ],
            "a14_result_sha256": config.payload["a14_freeze"][
                "result_sha256"
            ],
            "a14_per_cell_sha256": config.payload["a14_freeze"][
                "per_cell_sha256"
            ],
            "a14_per_seed_sha256": config.payload["a14_freeze"][
                "per_seed_sha256"
            ],
            "a14_runner_sha256": config.payload["a14_freeze"][
                "runner_sha256"
            ],
            "seed17_checkpoint_sha256": config.payload["primary_seed"][
                "checkpoint_sha256"
            ],
        },
        "reconstruction": {
            "rows": EXPECTED_ROWS,
            "sources": len(bundle.source_order),
            "speaker_sensitivity_keys": len(bundle.speaker_order),
            "frames": int(bundle.labels.size),
            "integrity": dict(integrity),
        },
        "classification": {
            "counts": layer_counts,
            "dominant_failure_layer": max(
                (
                    name
                    for name in (
                        "AVAILABILITY_LIMITED",
                        "OBSERVABILITY_LIMITED",
                        "ACTIONABILITY_LIMITED",
                    )
                ),
                key=lambda name: layer_counts[name],
            ),
            "stop_rule_triggered": stop_triggered,
            "stop_action": (
                stop_rule["stop_action"] if stop_triggered else None
            ),
        },
        "groups": group_results,
        "run_provenance": {
            "started_at": started_at,
            "completed_at": completed_at,
            "single_seed_reconstruction": True,
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
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_lines(result: Mapping[str, Any]) -> list[str]:
    classification = result["classification"]
    counts = classification["counts"]
    lines = [
        "# A-v2 / AE0 - OOD Value Failure Decomposition",
        "",
        f"- Protocol SHA256: `{result['protocol']['sha256']}`",
        f"- Primary seed: `{result['scope']['primary_seed']}`",
        "- Analysis: post-hoc, no training, no router search, no A14 rerun",
        f"- A14 verdict carried forward: `CONDITIONAL_GO`",
        "",
        "## Layer counts",
        "",
        f"- Availability-limited: `{counts['AVAILABILITY_LIMITED']}/12`",
        f"- Observability-limited: `{counts['OBSERVABILITY_LIMITED']}/12`",
        f"- Actionability-limited: `{counts['ACTIONABILITY_LIMITED']}/12`",
        f"- Actionable: `{counts['ACTIONABLE']}/12`",
        f"- Stop rule triggered: `{classification['stop_rule_triggered']}`",
        "",
        "## Aggregate groups",
        "",
    ]
    for group in ("all", "seen", "unseen"):
        row = result["groups"][group]
        point = row["point"]
        lines.extend(
            [
                f"- `{group}`: A_net=`{point['availability_net']:.8f}`, "
                f"O=`{point['observability_auroc']}`, "
                f"U_gate=`{point['gate_utility']:.8f}`, "
                f"classification=`{row['classification']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Cell decomposition",
            "",
            "| Cell | A_net | O | U_gate | P | Classification |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for cell in CELL_ORDER:
        row = result["groups"][f"{cell[0]}/{cell[1]}"]
        point = row["point"]
        lines.append(
            f"| `{cell[0]}/{cell[1]}` | "
            f"`{point['availability_net']:.8f}` | "
            f"`{point['observability_auroc']}` | "
            f"`{point['gate_utility']:.8f}` | "
            f"`{point['actionability_ratio']}` | "
            f"`{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "AE0 uses only seed 17 and the opened A14 OOD data. It does not "
            "authorize A15, router tuning, gate tuning, architecture changes, "
            "or a new confirmatory claim on this same OOD set.",
            "",
        ]
    )
    return lines


def write_artifacts(
    result: Mapping[str, Any],
    group_results: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "ae0_value_decomposition.json"
    csv_path = output_dir / "per_cell.csv"
    report_path = output_dir / "README.md"
    json_path.write_text(
        json.dumps(_json_ready(result), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, _cell_rows(group_results))
    report_path.write_text(
        "\n".join(_report_lines(result)) + "\n",
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "per_cell_csv": csv_path,
        "report": report_path,
    }


def run_ae0(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    device: str | torch.device = "auto",
    progress_every: int = 10,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Path]:
    """Run AE0 exactly once after validating the AE0 and A14 freezes."""
    started_at = datetime.now(timezone.utc).isoformat()
    config = load_config(protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"AE0 output directory already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    validated = validate_final_ood_lock()
    bundle = reconstruct_seed17(
        validated,
        data_root=data_root,
        device=device,
        progress_every=progress_every,
    )
    integrity = validate_reconstruction(bundle, config)
    group_results = build_group_results(
        bundle,
        config,
        bootstrap_repeats=bootstrap_repeats,
    )
    result = _summary(
        config=config,
        integrity=integrity,
        group_results=group_results,
        bundle=bundle,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return write_artifacts(result, group_results, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run A-v2 / AE0 post-hoc value-failure decomposition on the "
            "opened A14 OOD data."
        )
    )
    parser.add_argument(
        "--protocol-path",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=BOOTSTRAP_REPEATS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = run_ae0(
        protocol_path=args.protocol_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device=args.device,
        progress_every=args.progress_every,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
