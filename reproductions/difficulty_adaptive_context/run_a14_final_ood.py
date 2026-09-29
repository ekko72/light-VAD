# -*- coding: utf-8 -*-
"""Run the sealed A14 Final OOD confirmation exactly once.

The A14 protocol is intentionally stricter than the development analyses.
The implementation validates every frozen artifact, manifest hash and
checkpoint hash before it reads any Final OOD manifest row.  The validation
function returns a capability object; only that object can authorize the
loader that reads the sealed rows.
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

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    COUNT_COLUMNS,
    metrics_from_counts,
    speaker_cluster_bootstrap,
    speaker_from_source_key,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    gated_scores,
    load_adaptive_model,
    predict_full_adaptive_frames,
)
from reproductions.difficulty_adaptive_context.train_context import (
    UNSEEN_NOISE,
)
from reproductions.marblenet_vad.dataset import read_manifest
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a14_final_ood_protocol.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a14_final_ood"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"

PROTOCOL_VERSION = "A14-final-ood-confirmatory-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_FINAL_OOD_ACCESS"

SEEDS = (17, 18, 19, 23)
PRIMARY_SEED = 17
SHARED_SHORT_SEEDS = (17, 18, 19)
INDEPENDENT_SHORT_SEED = 23

GATE_THRESHOLD = 0.13
DECISION_THRESHOLD = 0.5
SCORE_CHUNK_FRAMES = 2000
ACTIVATION_WARNING = 0.15
ACTIVATION_FAILURE = 0.20
DEVELOPMENT_ACTIVATION = 0.0505
BOOTSTRAP_REPEATS = 5_000
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_SEED_RULES = {
    "aggregate_source_cluster": BOOTSTRAP_SEED,
    "speaker_cluster_sensitivity": "base + 1",
    "domain_source_cluster": "base + 10 + domain_index",
    "cell_source_cluster": "base + 100 + cell_index",
}

EXPECTED_ROWS = 1_080
EXPECTED_SOURCES = 20
EXPECTED_SPEAKER_KEYS = 16
EXPECTED_NOISE_CLASSES = 9
EXPECTED_SEEN_ROWS = 720
EXPECTED_UNSEEN_ROWS = 360
EXPECTED_SAMPLES = 322_768_800
EXPECTED_FRAMES = 2_017_926

SNR_ORDER = ("-5", "0", "5", "10", "15", "20")
DOMAIN_ORDER = ("seen", "unseen")
CELL_ORDER = tuple(
    (domain, snr) for domain in DOMAIN_ORDER for snr in SNR_ORDER
)

RF384_DILATIONS = (1, 2, 4, 8, 96)
RF384_LOOKBACK = 384
RF384_KERNEL = 5
RF384_MAX_RESIDUAL = 2.0

REQUIRED_WORDING = (
    "four refinement checkpoints were evaluated, of which three share the "
    "same frozen Short encoder and one uses an independently trained Short "
    "encoder"
)

_VALIDATION_TOKEN = object()


@dataclass(frozen=True)
class ValidatedFinalOOD:
    """Capability object issued only after complete frozen-lock validation."""

    protocol: dict[str, Any]
    protocol_path: Path
    lock_record: dict[str, Any]
    _token: object


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_file_hash(
    *,
    role: str,
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path).upper()
    expected = str(expected_sha256).upper()
    if actual != expected:
        raise ValueError(
            f"A14 final-OOD lock hash mismatch for {role}: "
            f"{actual} != {expected}"
        )
    return {
        "role": role,
        "path": str(path),
        "sha256": actual,
    }


def _validate_checkpoint_payload(
    path: Path,
    *,
    seed: int,
) -> None:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    short_config = dict(
        payload.get("short_model_config", payload.get("model_config", {}))
    )
    if not bool(short_config.get("causal", False)):
        raise ValueError(f"seed {seed} checkpoint is not causal")
    if not bool(short_config.get("frame_output", False)):
        raise ValueError(f"seed {seed} checkpoint is not frame-level")
    if str(short_config.get("dilation_profile", "")) != "short":
        raise ValueError(f"seed {seed} checkpoint does not use Short dilation")
    refinement = dict(payload.get("refinement_config", {}))
    if tuple(int(value) for value in refinement.get("dilations", ())) != (
        RF384_DILATIONS
    ):
        raise ValueError(f"seed {seed} checkpoint does not use RF384")
    if int(refinement.get("lookback_frames", -1)) not in (
        -1,
        RF384_LOOKBACK,
    ):
        raise ValueError(f"seed {seed} checkpoint has an unexpected lookback")
    if int(refinement.get("kernel_size", -1)) != RF384_KERNEL:
        raise ValueError(f"seed {seed} checkpoint has an unexpected kernel")
    if float(refinement.get("max_residual", np.nan)) != RF384_MAX_RESIDUAL:
        raise ValueError(
            f"seed {seed} checkpoint has an unexpected residual bound"
        )


def validate_final_ood_lock(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> ValidatedFinalOOD:
    """Validate all A14 locks without reading Final OOD manifest rows."""
    path = Path(protocol_path)
    protocol = _load_json(path)
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported A14 final-OOD protocol version")
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("A14 final-OOD protocol is not frozen before access")

    route = protocol.get("route", {})
    expected_route = {
        "final_ood_access": "SEALED",
        "a14": "CONFIRMATION_ONLY",
        "a14_after_result": "A_CLAIM_FREEZE",
        "a15": "FORBIDDEN",
        "router_tuning_after_a14": "FORBIDDEN",
        "stabilizer_comparison_on_final_ood": "FORBIDDEN",
    }
    if route != expected_route:
        raise ValueError("A14 final-OOD route differs from the frozen route")

    candidate = protocol.get("candidate", {})
    expected_candidate = {
        "name": "ShortShort Adaptive RF384Adaptive RF384",
        "short_model": "Short",
        "adaptive_model": "Adaptive RF384",
        "comparator": "AlwaysRefine RF384",
        "rf_span": 384,
        "dilations": list(RF384_DILATIONS),
        "lookback_frames": RF384_LOOKBACK,
        "kernel_size": RF384_KERNEL,
        "max_residual": RF384_MAX_RESIDUAL,
        "router": "raw causal confidence gate",
        "gate_rule": "abs(short_score - 0.5) <= 0.13",
        "threshold": GATE_THRESHOLD,
        "source_rank_stabilizer": "EXCLUDED",
        "unique_candidate": True,
    }
    if candidate != expected_candidate:
        raise ValueError("A14 final-OOD candidate differs from the freeze")

    artifacts = protocol.get("locks", {}).get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("A14 final-OOD protocol has no artifact locks")
    verified_artifacts: list[dict[str, str]] = []
    artifact_roles: dict[str, dict[str, str]] = {}
    for record in artifacts:
        role = str(record.get("role", ""))
        if not role or role in artifact_roles:
            raise ValueError(f"invalid or duplicate A14 artifact role: {role}")
        verified = _verify_file_hash(
            role=role,
            path=_resolve_path(str(record.get("path", ""))),
            expected_sha256=str(record.get("sha256", "")),
        )
        verified_artifacts.append(verified)
        artifact_roles[role] = verified

    required_artifact_roles = {
        "final_ood_manifest",
        "a11_cost",
        "a12_routing",
        "adaptive_model",
        "evaluate_adaptive",
        "analyze_context_gate",
        "data",
        "marblenet_dataset",
        "marblenet_features",
        "marblenet_model",
        "run_a14_final_ood",
        "test_run_a14_final_ood",
    }
    if set(artifact_roles) != required_artifact_roles:
        raise ValueError("A14 final-OOD artifact lock set is incomplete")
    manifest_lock = artifact_roles["final_ood_manifest"]

    sealed_split = protocol.get("sealed_split", {})
    if sealed_split.get("protocol_role") != "test_untouched_final_ood":
        raise ValueError("A14 final-OOD names the wrong sealed split")
    if manifest_lock["sha256"] != str(
        sealed_split.get("manifest_sha256", "")
    ).upper():
        raise ValueError("A14 manifest hash differs from its lock")
    expected_split_counts = {
        "rows": EXPECTED_ROWS,
        "sources": EXPECTED_SOURCES,
        "speaker_sensitivity_keys": EXPECTED_SPEAKER_KEYS,
        "noise_classes": EXPECTED_NOISE_CLASSES,
        "seen_rows": EXPECTED_SEEN_ROWS,
        "unseen_rows": EXPECTED_UNSEEN_ROWS,
        "samples": EXPECTED_SAMPLES,
        "causal_frames": EXPECTED_FRAMES,
    }
    for name, expected in expected_split_counts.items():
        if int(sealed_split.get(name, -1)) != int(expected):
            raise ValueError(
                f"A14 sealed-split {name} differs from the freeze"
            )

    checkpoints = protocol.get("locks", {}).get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("A14 final-OOD protocol has no checkpoint locks")
    expected_checkpoint_specs = set(SEEDS)
    actual_checkpoint_specs: set[int] = set()
    verified_checkpoints: list[dict[str, Any]] = []
    for record in checkpoints:
        seed = int(record.get("seed", -1))
        if seed in actual_checkpoint_specs:
            raise ValueError(f"duplicate A14 checkpoint lock: seed {seed}")
        actual_checkpoint_specs.add(seed)
        checkpoint_path = _resolve_path(str(record.get("path", "")))
        verified = _verify_file_hash(
            role=str(record.get("role", "")),
            path=checkpoint_path,
            expected_sha256=str(record.get("sha256", "")),
        )
        _validate_checkpoint_payload(checkpoint_path, seed=seed)
        verified.update(
            {
                "seed": seed,
                "short_encoder_relation": str(
                    record.get("short_encoder_relation", "")
                ),
                "role": str(record.get("role", "")),
            }
        )
        verified_checkpoints.append(verified)
    if actual_checkpoint_specs != expected_checkpoint_specs:
        raise ValueError("A14 final-OOD checkpoint lock set is incomplete")
    for record in verified_checkpoints:
        expected_relation = (
            "shared_frozen_short_encoder"
            if int(record["seed"]) in SHARED_SHORT_SEEDS
            else "independently_trained_short_encoder"
        )
        if record["short_encoder_relation"] != expected_relation:
            raise ValueError("A14 Short encoder relation differs from freeze")

    evaluation = protocol.get("evaluation", {})
    expected_evaluation = {
        "primary_seed": PRIMARY_SEED,
        "checkpoint_seeds": list(SEEDS),
        "decision_threshold": DECISION_THRESHOLD,
        "score_chunk_frames": SCORE_CHUNK_FRAMES,
        "causal_inference_start_frame": 0,
        "zero_cache_at_start": True,
        "valid_start_frame": 0,
        "development_activation_reference": DEVELOPMENT_ACTIVATION,
        "activation_warning": ACTIVATION_WARNING,
        "activation_failure": ACTIVATION_FAILURE,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_seed_rules": BOOTSTRAP_SEED_RULES,
        "bootstrap_primary_unit": "source_cluster",
        "bootstrap_sensitivity_unit": "speaker_cluster",
        "cells": [list(cell) for cell in CELL_ORDER],
        "required_wording": REQUIRED_WORDING,
    }
    for name, expected in expected_evaluation.items():
        if evaluation.get(name) != expected:
            raise ValueError(
                f"A14 evaluation field {name!r} differs from the freeze"
            )

    gates = protocol.get("gates", {})
    if gates.get("gate_1", {}).get("utility") != "U_A_minus_S = R_S - R_A":
        raise ValueError("A14 Gate 1 utility differs from the freeze")
    if gates.get("gate_1", {}).get("strong_go_rule") != (
        "U > 0 and source-cluster bootstrap 95% lower CI > 0"
    ):
        raise ValueError("A14 Gate 1 strong-Go rule differs from the freeze")
    if gates.get("gate_2", {}).get("warning_boundary") != ACTIVATION_WARNING:
        raise ValueError("A14 Gate 2 warning boundary differs from the freeze")
    if gates.get("gate_3", {}).get("cell_order") != [
        list(cell) for cell in CELL_ORDER
    ]:
        raise ValueError("A14 Gate 3 cell order differs from the freeze")
    if gates.get("gate_3", {}).get("require_all_cells_positive") is not False:
        raise ValueError("A14 Gate 3 must not require all cells positive")
    if gates.get("gate_3", {}).get("taxonomy", {}).get("F2") != (
        "population value > 0 and selected value <= 0"
    ):
        raise ValueError("A14 F2 rule differs from the freeze")
    if gates.get("gate_3", {}).get("taxonomy", {}).get("F3") != (
        "selected value > 0 and activation > 0.15"
    ):
        raise ValueError("A14 F3 rule differs from the freeze")

    verdicts = protocol.get("verdicts", {})
    if verdicts.get("strong_go", {}).get("activation_rule") != (
        "r_OOD < 0.15"
    ):
        raise ValueError("A14 Strong GO activation rule differs")
    if verdicts.get("no_go", {}).get("aggregate_rule") != (
        "aggregate utility non-positive or (r_OOD > 0.20 and "
        "source-cluster bootstrap 95% lower CI <= 0)"
    ):
        raise ValueError("A14 NO-GO aggregate rule differs")
    if protocol.get("claim_freeze", {}).get("action") != (
        "A14 -> A CLAIM FREEZE"
    ):
        raise ValueError("A14 claim-freeze action differs")

    protocol_sha256 = file_sha256(path).upper()
    lock_record = {
        "version": PROTOCOL_VERSION,
        "status": "VALIDATED_BEFORE_SEALED_ACCESS",
        "protocol_path": str(path),
        "protocol_sha256": protocol_sha256,
        "verified_artifacts": verified_artifacts,
        "verified_checkpoints": verified_checkpoints,
        "sealed_split": sealed_split,
        "sealed_access": {
            "status": "AUTHORIZED_AFTER_FULL_LOCK_VALIDATION",
            "rows_loaded": False,
        },
    }
    return ValidatedFinalOOD(
        protocol=protocol,
        protocol_path=path,
        lock_record=lock_record,
        _token=_VALIDATION_TOKEN,
    )


def _source_digest(split_dir: str, source_relative_path: str) -> str:
    payload = f"{split_dir}\0{source_relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_key(row: Mapping[str, str]) -> str:
    return f"{row.get('split_dir', '')}/{row.get('source_relative_path', '')}"


def _speaker_key(row: Mapping[str, str]) -> str:
    return speaker_from_source_key(_source_key(row))


def _snr(row: Mapping[str, str]) -> str:
    return str(row.get("snr_db", "")).strip()


def _is_unseen_row(row: Mapping[str, str]) -> bool:
    return str(row.get("noise_name", "")) in set(UNSEEN_NOISE)


def select_final_ood_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_sources: int = EXPECTED_SOURCES,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select the frozen hash-ranked source clusters and retain all variants."""
    if not rows:
        raise ValueError("Final OOD manifest is empty")
    source_specs: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        split_dir = str(row.get("split_dir", ""))
        source_relative_path = str(row.get("source_relative_path", ""))
        if not split_dir or not source_relative_path:
            raise ValueError("Final OOD row lacks source identity")
        key = f"{split_dir}/{source_relative_path}"
        digest = _source_digest(split_dir, source_relative_path)
        previous = source_specs.get(key)
        current = (digest, split_dir, source_relative_path)
        if previous is not None and previous != current:
            raise ValueError(f"inconsistent source identity: {key}")
        source_specs[key] = current

    ranked = sorted(
        source_specs.items(),
        key=lambda item: (item[1][0], item[1][1], item[1][2]),
    )
    selected_sources = [name for name, _ in ranked[: int(expected_sources)]]
    if len(selected_sources) != int(expected_sources):
        raise ValueError("not enough source clusters for the frozen selection")
    selected_set = set(selected_sources)
    selected_rows = [
        dict(row)
        for row in rows
        if _source_key(row) in selected_set
    ]
    rank = {name: index for index, name in enumerate(selected_sources)}
    selected_rows.sort(
        key=lambda row: (
            rank[_source_key(row)],
            str(row.get("noise_name", "")),
            float(row.get("snr_db", "nan")),
        )
    )
    metadata = {
        "source_selection_rule": (
            "sort by (sha256(split_dir\\0source_relative_path), "
            "split_dir, source_relative_path), take first 20 sources"
        ),
        "selected_source_keys": selected_sources,
        "selected_source_digests": [source_specs[name][0] for name in selected_sources],
        "rows": len(selected_rows),
        "sources": len(selected_sources),
        "speaker_sensitivity_keys": sorted(
            {_speaker_key(row) for row in selected_rows}
        ),
        "noise_classes": sorted(
            {str(row.get("noise_name", "")) for row in selected_rows}
        ),
        "seen_rows": int(sum(not _is_unseen_row(row) for row in selected_rows)),
        "unseen_rows": int(sum(_is_unseen_row(row) for row in selected_rows)),
        "samples": int(
            sum(int(row.get("output_frames", "0")) for row in selected_rows)
        ),
        "causal_frames": int(
            sum(
                int(row.get("output_frames", "0")) // FRAME_HOP + 1
                for row in selected_rows
            )
        ),
    }
    return selected_rows, metadata


def _validate_selected_rows(
    rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Any],
) -> None:
    expected = {
        "rows": EXPECTED_ROWS,
        "sources": EXPECTED_SOURCES,
        "speaker_sensitivity_keys": EXPECTED_SPEAKER_KEYS,
        "noise_classes": EXPECTED_NOISE_CLASSES,
        "seen_rows": EXPECTED_SEEN_ROWS,
        "unseen_rows": EXPECTED_UNSEEN_ROWS,
        "samples": EXPECTED_SAMPLES,
        "causal_frames": EXPECTED_FRAMES,
    }
    for name, value in expected.items():
        actual = metadata.get(name)
        if name in {
            "speaker_sensitivity_keys",
            "noise_classes",
        }:
            actual = len(actual) if actual is not None else None
        if actual != value:
            raise ValueError(
                f"Final OOD {name} differs from the frozen protocol: "
                f"{actual} != {value}"
            )
    per_source: dict[str, int] = {}
    per_noise: dict[str, int] = {}
    per_snr: dict[str, int] = {}
    for row in rows:
        source = _source_key(row)
        noise = str(row.get("noise_name", ""))
        snr = _snr(row)
        per_source[source] = per_source.get(source, 0) + 1
        per_noise[noise] = per_noise.get(noise, 0) + 1
        per_snr[snr] = per_snr.get(snr, 0) + 1
    if len(per_source) != EXPECTED_SOURCES:
        raise ValueError("Final OOD source-cluster count is inconsistent")
    if set(per_source.values()) != {54}:
        raise ValueError("every frozen source must retain 54 variants")
    if set(per_noise.values()) != {EXPECTED_ROWS // EXPECTED_NOISE_CLASSES}:
        raise ValueError("Final OOD noise-class counts are inconsistent")
    if set(per_snr.values()) != {EXPECTED_ROWS // len(SNR_ORDER)}:
        raise ValueError("Final OOD SNR counts are inconsistent")


def load_sealed_final_ood_rows(
    validated: ValidatedFinalOOD,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read the sealed manifest only after the validation capability is issued."""
    if not isinstance(validated, ValidatedFinalOOD):
        raise TypeError("full A14 final-OOD lock validation is required")
    if validated._token is not _VALIDATION_TOKEN:
        raise RuntimeError("invalid A14 final-OOD validation capability")
    path = _resolve_path(str(validated.protocol["sealed_split"]["manifest_path"]))
    rows = read_manifest(path)
    selected_rows, metadata = select_final_ood_rows(rows)
    _validate_selected_rows(selected_rows, metadata)
    metadata["manifest_path"] = str(path)
    return selected_rows, metadata


def _row_paths(row: Mapping[str, str], data_root: Path) -> tuple[Path, Path]:
    generated = data_root / "generated" / str(row["output_audio_path"])
    labels = data_root / "labels" / str(row["label_relative_path"])
    return generated, labels


def _counts_for_cluster_order(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_order: Sequence[str],
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    refined_scores = np.asarray(refined_scores, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    cluster_ids = np.asarray(cluster_ids, dtype=str).reshape(-1)
    if not (
        labels.size
        == short_scores.size
        == refined_scores.size
        == selected.size
        == cluster_ids.size
    ):
        raise ValueError("frame arrays must have identical lengths")
    lookup = {str(name): index for index, name in enumerate(cluster_order)}
    inverse = np.asarray(
        [lookup[str(name)] for name in cluster_ids],
        dtype=np.int64,
    )
    short_predictions = short_scores >= DECISION_THRESHOLD
    refined_predictions = refined_scores >= DECISION_THRESHOLD
    short_wrong = short_predictions != labels
    refined_wrong = refined_predictions != labels
    values = {
        "frames": np.ones(labels.size, dtype=np.int64),
        "selected": selected.astype(np.int64),
        "correction": (
            selected & short_wrong & ~refined_wrong
        ).astype(np.int64),
        "harm": (
            selected & ~short_wrong & refined_wrong
        ).astype(np.int64),
        "short_error": short_wrong.astype(np.int64),
        "long_error": refined_wrong.astype(np.int64),
        "gated_error": np.where(
            selected,
            refined_wrong,
            short_wrong,
        ).astype(np.int64),
    }
    columns = [
        np.bincount(
            inverse,
            weights=values[name],
            minlength=len(cluster_order),
        )
        for name in COUNT_COLUMNS
    ]
    return np.stack(columns, axis=1).astype(np.int64)


def _raw_gate_selection(short_scores: np.ndarray) -> np.ndarray:
    """Apply the frozen raw causal confidence gate exactly once."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    return np.abs(scores - DECISION_THRESHOLD) <= GATE_THRESHOLD


def _detection_counts(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    prediction = scores >= DECISION_THRESHOLD
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    speech = int(np.count_nonzero(truth))
    silence = int(np.count_nonzero(~truth))
    return tp, fp, fn, tn, speech, silence


def _f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _empty_counts(cluster_count: int) -> np.ndarray:
    return np.zeros((cluster_count, len(COUNT_COLUMNS)), dtype=np.int64)


def _empty_detection_counts() -> np.ndarray:
    return np.zeros(6, dtype=np.int64)


def _accumulate_seed(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    model: Any,
    frontend: torch.nn.Module,
    device: torch.device,
    data_root: Path,
    source_order: Sequence[str],
    speaker_order: Sequence[str],
    progress_every: int,
) -> dict[str, Any]:
    source_index = {name: index for index, name in enumerate(source_order)}
    speaker_index = {name: index for index, name in enumerate(speaker_order)}
    source_counts = _empty_counts(len(source_order))
    speaker_counts = _empty_counts(len(speaker_order))
    cell_source_counts = {
        cell: _empty_counts(len(source_order)) for cell in CELL_ORDER
    }
    cell_speaker_counts = {
        cell: _empty_counts(len(speaker_order)) for cell in CELL_ORDER
    }
    detection = {
        "short": _empty_detection_counts(),
        "refined": _empty_detection_counts(),
        "adaptive": _empty_detection_counts(),
    }
    cell_detection = {
        cell: {
            "short": _empty_detection_counts(),
            "refined": _empty_detection_counts(),
            "adaptive": _empty_detection_counts(),
        }
        for cell in CELL_ORDER
    }
    evaluated_rows = 0
    evaluated_frames = 0
    for index, row in enumerate(rows):
        audio_path, label_path = _row_paths(row, data_root)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
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
                device=device,
                chunk_frames=SCORE_CHUNK_FRAMES,
            )
        )
        if refined_scores.size != n_frames or embedded_short.size != n_frames:
            raise RuntimeError("adaptive output frame count is inconsistent")
        if not np.all(stream_selected):
            raise RuntimeError(
                "permissive refinement pass did not cover every frame"
            )
        selected = _raw_gate_selection(embedded_short)
        adaptive_scores = gated_scores(
            embedded_short,
            refined_scores,
            selected,
        )
        source = _source_key(row)
        speaker = _speaker_key(row)
        domain = "unseen" if _is_unseen_row(row) else "seen"
        cell = (domain, _snr(row))
        if cell not in cell_source_counts:
            raise ValueError(f"unexpected Final OOD cell: {cell}")
        source_ids = np.full(
            labels.size,
            source_index[source],
            dtype=np.int64,
        )
        speaker_ids = np.full(
            labels.size,
            speaker_index[speaker],
            dtype=np.int64,
        )
        source_row_counts = _counts_for_cluster_order(
            labels,
            embedded_short,
            adaptive_scores,
            selected,
            np.asarray([source] * labels.size, dtype=str),
            source_order,
        )
        speaker_row_counts = _counts_for_cluster_order(
            labels,
            embedded_short,
            adaptive_scores,
            selected,
            np.asarray([speaker] * labels.size, dtype=str),
            speaker_order,
        )
        source_counts += source_row_counts
        speaker_counts += speaker_row_counts
        cell_source_counts[cell] += source_row_counts
        cell_speaker_counts[cell] += speaker_row_counts
        for name, scores in (
            ("short", embedded_short),
            ("refined", refined_scores),
            ("adaptive", adaptive_scores),
        ):
            detection[name] += np.asarray(
                _detection_counts(labels, scores),
                dtype=np.int64,
            )
            cell_detection[cell][name] += np.asarray(
                _detection_counts(labels, scores),
                dtype=np.int64,
            )
        evaluated_rows += 1
        evaluated_frames += int(labels.size)
        if progress_every > 0 and (
            (index + 1) % progress_every == 0 or index + 1 == len(rows)
        ):
            print(
                f"seed {seed}: evaluated {index + 1}/{len(rows)} rows",
                flush=True,
            )
    if evaluated_rows != len(rows):
        raise RuntimeError("not every sealed Final OOD row was evaluated")
    if evaluated_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"evaluated {evaluated_frames} frames, expected {EXPECTED_FRAMES}"
        )
    return {
        "seed": int(seed),
        "source_counts": source_counts,
        "speaker_counts": speaker_counts,
        "cell_source_counts": cell_source_counts,
        "cell_speaker_counts": cell_speaker_counts,
        "detection": detection,
        "cell_detection": cell_detection,
        "evaluated_rows": evaluated_rows,
        "evaluated_frames": evaluated_frames,
    }


def _bootstrap(
    counts: np.ndarray,
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return speaker_cluster_bootstrap(
        [np.asarray(counts, dtype=np.int64)],
        cluster_indices=np.arange(counts.shape[0], dtype=np.int64),
        repeats=int(repeats),
        seed=int(seed),
    )


def _metrics_from_detection(
    detection: np.ndarray,
) -> dict[str, Any]:
    tp, fp, fn, tn, speech, silence = (int(value) for value in detection)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "frames": tp + fp + fn + tn,
        "speech": speech,
        "silence": silence,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": _f1_from_counts(tp, fp, fn),
    }


def _seed_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    counts = np.asarray(result["source_counts"], dtype=np.int64)
    metrics = metrics_from_counts(counts.sum(axis=0))
    detection = result["detection"]
    short_metrics = _metrics_from_detection(detection["short"])
    refined_metrics = _metrics_from_detection(detection["refined"])
    adaptive_metrics = _metrics_from_detection(detection["adaptive"])
    return {
        "seed": int(result["seed"]),
        "frames": int(result["evaluated_frames"]),
        "utility": metrics,
        "short": short_metrics,
        "refined": refined_metrics,
        "adaptive": adaptive_metrics,
        "delta_f1_adaptive_minus_short": (
            None
            if short_metrics["f1"] is None
            or adaptive_metrics["f1"] is None
            else float(
                adaptive_metrics["f1"] - short_metrics["f1"]
            )
        ),
    }


def _cell_summary(
    result: Mapping[str, Any],
    *,
    cell: tuple[str, str],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    source_counts = np.asarray(
        result["cell_source_counts"][cell],
        dtype=np.int64,
    )
    speaker_counts = np.asarray(
        result["cell_speaker_counts"][cell],
        dtype=np.int64,
    )
    metrics = metrics_from_counts(source_counts.sum(axis=0))
    detection = result["cell_detection"][cell]
    short_metrics = _metrics_from_detection(detection["short"])
    refined_metrics = _metrics_from_detection(detection["refined"])
    adaptive_metrics = _metrics_from_detection(detection["adaptive"])
    population_value = (
        None
        if metrics["frames"] == 0
        else float(
            (
                metrics["short_error"] - metrics["long_error"]
            )
            / metrics["frames"]
        )
    )
    selected_value = metrics["net_utility_per_selected"]
    if population_value is None or selected_value is None:
        taxonomy = "F1_VALUE_SCARCITY"
    elif population_value <= 0.0 and selected_value <= 0.0:
        taxonomy = "F1_VALUE_SCARCITY"
    elif population_value > 0.0 and selected_value <= 0.0:
        taxonomy = "F2_RANKING_FAILURE"
    elif selected_value > 0.0 and float(metrics["activation_rate"]) > (
        ACTIVATION_WARNING
    ):
        taxonomy = "F3_BUDGET_CALIBRATION_DRIFT"
    else:
        taxonomy = "NO_FAILURE"
    return {
        "domain": str(cell[0]),
        "snr_db": str(cell[1]),
        "utility": metrics,
        "population_value": population_value,
        "selected_value": selected_value,
        "taxonomy": taxonomy,
        "short": short_metrics,
        "refined": refined_metrics,
        "adaptive": adaptive_metrics,
        "delta_f1_adaptive_minus_short": (
            None
            if short_metrics["f1"] is None
            or adaptive_metrics["f1"] is None
            else float(
                adaptive_metrics["f1"] - short_metrics["f1"]
            )
        ),
        "source_cluster_bootstrap": _bootstrap(
            source_counts,
            repeats=bootstrap_repeats,
            seed=bootstrap_seed,
        ),
        "speaker_cluster_bootstrap": _bootstrap(
            speaker_counts,
            repeats=bootstrap_repeats,
            seed=bootstrap_seed + 1,
        ),
    }


def _domain_summary(
    result: Mapping[str, Any],
    *,
    domain: str,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    counts = np.zeros_like(
        np.asarray(result["source_counts"], dtype=np.int64)
    )
    for cell, cell_counts in result["cell_source_counts"].items():
        if cell[0] == domain:
            counts += np.asarray(cell_counts, dtype=np.int64)
    metrics = metrics_from_counts(counts.sum(axis=0))
    bootstrap = _bootstrap(
        counts,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    return {
        "domain": str(domain),
        "utility": metrics,
        "source_cluster_bootstrap": bootstrap,
    }


def _seed_sign(value: float | None) -> str:
    if value is None:
        return "undefined"
    if float(value) > 0.0:
        return "positive"
    if float(value) < 0.0:
        return "negative"
    return "zero"


def _seed23_sensitivity(
    seed_utilities: Mapping[str, float],
) -> dict[str, Any]:
    """Summarize checkpoint sensitivity without treating the four as n=4."""
    mean_17_19 = float(
        np.mean([float(seed_utilities[str(seed)]) for seed in SHARED_SHORT_SEEDS])
    )
    u23 = float(seed_utilities[str(INDEPENDENT_SHORT_SEED)])
    return {
        "mean_shared_short_seeds_17_18_19": mean_17_19,
        "independent_short_seed23": u23,
        "seed23_direction_consistent": (
            _seed_sign(u23) == _seed_sign(mean_17_19)
        ),
        "seed23_positive_direction": u23 > 0.0 and mean_17_19 > 0.0,
        "sign_rule": "sign(U23) == sign(mean(U17,U18,U19))",
    }


def _select_verdict(
    *,
    aggregate_utility: float,
    aggregate_ci_low: float,
    activation: float,
    catastrophic_domain_regression: Sequence[str],
    seed23_positive_direction: bool,
) -> str:
    """Apply the frozen A14 verdict precedence exactly once."""
    activation_without_strong_benefit = (
        float(activation) > ACTIVATION_FAILURE
        and float(aggregate_ci_low) <= 0.0
    )
    if float(aggregate_utility) <= 0.0 or activation_without_strong_benefit:
        return "OOD_GENERALIZATION_NO_GO"
    if (
        float(aggregate_ci_low) > 0.0
        and float(activation) < ACTIVATION_WARNING
        and not catastrophic_domain_regression
        and bool(seed23_positive_direction)
    ):
        return "STRONG_GO"
    return "CONDITIONAL_GO"


def build_final_ood_result(
    validated: ValidatedFinalOOD,
    *,
    seed_results: Mapping[int, Mapping[str, Any]],
    row_metadata: Mapping[str, Any],
    lock_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen gates and verdict precedence to evaluated counts."""
    primary = seed_results[PRIMARY_SEED]
    primary_summary = _seed_summary(primary)
    seed_summaries = {
        str(seed): _seed_summary(seed_results[seed]) for seed in SEEDS
    }
    seed_utilities = {
        str(seed): float(
            seed_summaries[str(seed)]["utility"]["net_utility_per_frame"]
        )
        for seed in SEEDS
    }
    seed23_sensitivity = _seed23_sensitivity(seed_utilities)
    seed23_positive_direction = bool(
        seed23_sensitivity["seed23_positive_direction"]
    )

    aggregate_utility = float(
        primary_summary["utility"]["net_utility_per_frame"]
    )
    aggregate_bootstrap = _bootstrap(primary["source_counts"])
    aggregate_ci_low = float(
        aggregate_bootstrap["net_utility_per_frame_ci95_low"]
    )
    activation = float(primary_summary["utility"]["activation_rate"])
    activation_warning = activation > ACTIVATION_WARNING
    activation_failure = activation > ACTIVATION_FAILURE

    domains = {
        domain: _domain_summary(
            primary,
            domain=domain,
            bootstrap_repeats=BOOTSTRAP_REPEATS,
            bootstrap_seed=BOOTSTRAP_SEED + 10 + index,
        )
        for index, domain in enumerate(DOMAIN_ORDER)
    }
    catastrophic_domain_regression = []
    for domain, summary in domains.items():
        utility = float(summary["utility"]["net_utility_per_frame"])
        ci_high = float(
            summary["source_cluster_bootstrap"][
                "net_utility_per_frame_ci95_high"
            ]
        )
        if utility < 0.0 and ci_high < 0.0:
            catastrophic_domain_regression.append(domain)

    cells = [
        _cell_summary(
            primary,
            cell=cell,
            bootstrap_repeats=BOOTSTRAP_REPEATS,
            bootstrap_seed=BOOTSTRAP_SEED + 100 + index,
        )
        for index, cell in enumerate(CELL_ORDER)
    ]
    cell_utilities = [
        float(cell["utility"]["net_utility_per_frame"]) for cell in cells
    ]
    positive_cells = int(sum(value > 0.0 for value in cell_utilities))
    worst_cell = min(cells, key=lambda cell: float(
        cell["utility"]["net_utility_per_frame"]
    ))
    max_activation_cell = max(
        cells,
        key=lambda cell: float(cell["utility"]["activation_rate"]),
    )
    taxonomy_counts = {
        name: int(sum(cell["taxonomy"] == name for cell in cells))
        for name in (
            "F1_VALUE_SCARCITY",
            "F2_RANKING_FAILURE",
            "F3_BUDGET_CALIBRATION_DRIFT",
            "NO_FAILURE",
        )
    }

    verdict = _select_verdict(
        aggregate_utility=aggregate_utility,
        aggregate_ci_low=aggregate_ci_low,
        activation=activation,
        catastrophic_domain_regression=catastrophic_domain_regression,
        seed23_positive_direction=seed23_positive_direction,
    )

    gate1_status = (
        "STRONG_PASS"
        if aggregate_utility > 0.0 and aggregate_ci_low > 0.0
        else "CONDITIONAL"
        if aggregate_utility > 0.0
        else "FAIL"
    )
    gate2_status = (
        "PASS"
        if not activation_warning
        else "FAILURE"
        if activation_failure
        else "WARNING"
    )
    gate3_status = (
        "NO_CATASTROPHIC_DOMAIN_REGRESSION"
        if not catastrophic_domain_regression
        else "CATASTROPHIC_DOMAIN_REGRESSION"
    )
    result = {
        "version": PROTOCOL_VERSION,
        "date": str(validated.protocol.get("date", "")),
        "protocol": {
            "path": str(validated.protocol_path),
            "sha256": str(lock_record["protocol_sha256"]),
            "version": PROTOCOL_VERSION,
        },
        "route": validated.protocol["route"],
        "candidate": validated.protocol["candidate"],
        "sealed_split": {
            **dict(validated.protocol["sealed_split"]),
            **dict(row_metadata),
        },
        "checkpoint_accounting": {
            "required_wording": REQUIRED_WORDING,
            "primary_seed": PRIMARY_SEED,
            "shared_short_seeds": list(SHARED_SHORT_SEEDS),
            "independent_short_seed": INDEPENDENT_SHORT_SEED,
            "not_independent_end_to_end_seeds": True,
            "no_n_equals_4": True,
            "no_mean_plus_minus_std_across_4_seeds": True,
        },
        "gates": {
            "gate_1_ood_accuracy_utility": {
                "status": gate1_status,
                "utility_definition": "U_A-S = R_S - R_A",
                "aggregate_utility": aggregate_utility,
                "aggregate_source_cluster_bootstrap": aggregate_bootstrap,
                "ci95_lower": aggregate_ci_low,
                "delta_f1_adaptive_minus_short": primary_summary[
                    "delta_f1_adaptive_minus_short"
                ],
            },
            "gate_2_ood_compute_behavior": {
                "status": gate2_status,
                "activation_rate": activation,
                "development_reference": DEVELOPMENT_ACTIVATION,
                "warning_boundary": ACTIVATION_WARNING,
                "failure_boundary": ACTIVATION_FAILURE,
                "drift": float(activation - DEVELOPMENT_ACTIVATION),
            },
            "gate_3_ood_failure_structure": {
                "status": gate3_status,
                "positive_cell_fraction": positive_cells / len(cells),
                "positive_cells": positive_cells,
                "cells": len(cells),
                "worst_cell_utility": float(
                    worst_cell["utility"]["net_utility_per_frame"]
                ),
                "worst_cell": {
                    "domain": worst_cell["domain"],
                    "snr_db": worst_cell["snr_db"],
                },
                "activation_max": float(
                    max_activation_cell["utility"]["activation_rate"]
                ),
                "activation_max_cell": {
                    "domain": max_activation_cell["domain"],
                    "snr_db": max_activation_cell["snr_db"],
                },
                "catastrophic_domain_regression": (
                    catastrophic_domain_regression
                ),
                "taxonomy_counts": taxonomy_counts,
            },
        },
        "domains": domains,
        "cells": cells,
        "seed_sensitivity": {
            "seed_summaries": seed_summaries,
            "seed_net_utility_per_frame": seed_utilities,
            **seed23_sensitivity,
        },
        "verdict": {
            "status": verdict,
            "claim": (
                "STRONG_GO"
                if verdict == "STRONG_GO"
                else "CONDITIONAL_GO"
                if verdict == "CONDITIONAL_GO"
                else "OOD_GENERALIZATION_NO_GO"
            ),
            "activation_warning": activation_warning,
            "activation_failure": activation_failure,
            "a15_allowed": False,
            "next_action": "A14 -> A CLAIM FREEZE",
        },
        "lock_record": dict(lock_record),
        "claim_freeze": {
            "action": "A14 -> A CLAIM FREEZE",
            "a15": "FORBIDDEN",
            "router_or_stabilizer_tuning": "FORBIDDEN",
        },
    }
    return result


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


def _per_cell_rows(
    seed_results: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = seed_results[seed]
        for cell in CELL_ORDER:
            summary = _cell_summary(
                result,
                cell=cell,
                bootstrap_repeats=BOOTSTRAP_REPEATS,
                bootstrap_seed=BOOTSTRAP_SEED + 100 + CELL_ORDER.index(cell),
            )
            utility = summary["utility"]
            rows.append(
                {
                    "seed": int(seed),
                    "domain": summary["domain"],
                    "snr_db": summary["snr_db"],
                    "frames": utility["frames"],
                    "selected": utility["selected"],
                    "activation_rate": utility["activation_rate"],
                    "corrections": utility["correction"],
                    "harms": utility["harm"],
                    "net_utility_per_frame": utility[
                        "net_utility_per_frame"
                    ],
                    "net_utility_per_selected": utility[
                        "net_utility_per_selected"
                    ],
                    "population_value": summary["population_value"],
                    "selected_value": summary["selected_value"],
                    "taxonomy": summary["taxonomy"],
                    "f1_short": summary["short"]["f1"],
                    "f1_adaptive": summary["adaptive"]["f1"],
                    "delta_f1_adaptive_minus_short": summary[
                        "delta_f1_adaptive_minus_short"
                    ],
                    "source_cluster_ci95_low": summary[
                        "source_cluster_bootstrap"
                    ]["net_utility_per_frame_ci95_low"],
                    "source_cluster_ci95_high": summary[
                        "source_cluster_bootstrap"
                    ]["net_utility_per_frame_ci95_high"],
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_lines(result: Mapping[str, Any]) -> list[str]:
    verdict = result["verdict"]["status"]
    gate1 = result["gates"]["gate_1_ood_accuracy_utility"]
    gate2 = result["gates"]["gate_2_ood_compute_behavior"]
    gate3 = result["gates"]["gate_3_ood_failure_structure"]
    lines = [
        "# A14 Final OOD Confirmation Report",
        "",
        f"- Protocol SHA256: `{result['protocol']['sha256']}`",
        f"- Unique candidate: `{result['candidate']['name']}`",
        f"- Router: `{result['candidate']['gate_rule']}`",
        f"- Final verdict: **{verdict}**",
        "- Claim freeze: `A14 -> A CLAIM FREEZE`",
        "- A15: `FORBIDDEN`",
        "",
        "## Checkpoint accounting",
        "",
        REQUIRED_WORDING + ".",
        "",
        "## Gate 1: OOD accuracy utility",
        "",
        f"- Aggregate utility: `{gate1['aggregate_utility']:.8f}`",
        f"- Source-cluster bootstrap 95% CI lower: "
        f"`{gate1['ci95_lower']:.8f}`",
        f"- Delta F1 (Adaptive - Short): "
        f"`{gate1['delta_f1_adaptive_minus_short']}`",
        f"- Status: `{gate1['status']}`",
        "",
        "## Gate 2: OOD compute behavior",
        "",
        f"- `r_OOD`: `{gate2['activation_rate']:.6%}`",
        f"- Development reference: `{gate2['development_reference']:.6%}`",
        f"- Warning boundary: `{gate2['warning_boundary']:.6%}`",
        f"- Failure boundary: `{gate2['failure_boundary']:.6%}`",
        f"- Status: `{gate2['status']}`",
        "",
        "## Gate 3: OOD failure structure",
        "",
        f"- Positive-cell fraction: `{gate3['positive_cell_fraction']:.3%}` "
        f"({gate3['positive_cells']}/{gate3['cells']})",
        f"- Worst-cell utility: `{gate3['worst_cell_utility']:.8f}` "
        f"(`{gate3['worst_cell']['domain']}/{gate3['worst_cell']['snr_db']} dB`)",
        f"- Activation maximum: `{gate3['activation_max']:.6%}` "
        f"(`{gate3['activation_max_cell']['domain']}/"
        f"{gate3['activation_max_cell']['snr_db']} dB`)",
        f"- Catastrophic domain regression: "
        f"`{gate3['catastrophic_domain_regression']}`",
        f"- Taxonomy counts: `{gate3['taxonomy_counts']}`",
        "",
        "## Seed sensitivity",
        "",
    ]
    for seed, value in result["seed_sensitivity"][
        "seed_net_utility_per_frame"
    ].items():
        lines.append(f"- seed{seed}: `{value:.8f}`")
    lines.extend(
        [
            f"- Mean seeds 17/18/19: "
            f"`{result['seed_sensitivity']['mean_shared_short_seeds_17_18_19']:.8f}`",
            f"- Seed23: "
            f"`{result['seed_sensitivity']['independent_short_seed23']:.8f}`",
            f"- Seed23 direction consistent: "
            f"`{result['seed_sensitivity']['seed23_direction_consistent']}`",
            f"- Seed23 positive direction: "
            f"`{result['seed_sensitivity']['seed23_positive_direction']}`",
            "",
            "## Interpretation boundary",
            "",
            "This is composition-level untouched OOD confirmation, not "
            "new-speaker or new-noise OOD. The primary uncertainty unit is "
            "the source cluster; speaker-cluster intervals are sensitivity "
            "checks. The result does not authorize A15, router tuning, gate "
            "tuning, architecture changes, or stabilizer selection on Final "
            "OOD.",
            "",
        ]
    )
    return lines


def write_artifacts(
    result: Mapping[str, Any],
    seed_results: Mapping[int, Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "a14_final_ood.json"
    per_cell_path = output_dir / "per_cell.csv"
    per_seed_path = output_dir / "per_seed.csv"
    report_path = output_dir / "report.md"
    lock_path = output_dir / "lock_record.json"
    claim_path = output_dir / "claim_freeze.json"
    result_path.write_text(
        json.dumps(_json_ready(result), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(per_cell_path, _per_cell_rows(seed_results))
    _write_csv(
        per_seed_path,
        [
            {
                "seed": seed,
                **{
                    key: value
                    for key, value in _seed_summary(seed_results[seed]).items()
                    if key not in {"short", "refined", "adaptive"}
                },
                "f1_short": _seed_summary(seed_results[seed])["short"]["f1"],
                "f1_adaptive": _seed_summary(seed_results[seed])["adaptive"][
                    "f1"
                ],
            }
            for seed in SEEDS
        ],
    )
    report_path.write_text(
        "\n".join(_report_lines(result)) + "\n",
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps(
            _json_ready(result["lock_record"]),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    claim_path.write_text(
        json.dumps(
            {
                "action": "A14 -> A CLAIM FREEZE",
                "verdict": result["verdict"]["status"],
                "a15": "FORBIDDEN",
                "router_or_stabilizer_tuning": "FORBIDDEN",
                "protocol_sha256": result["protocol"]["sha256"],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": result_path,
        "per_cell_csv": per_cell_path,
        "per_seed_csv": per_seed_path,
        "report": report_path,
        "lock_record": lock_path,
        "claim_freeze": claim_path,
    }


def run_final_ood(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    device: str | torch.device = "auto",
    progress_every: int = 10,
) -> dict[str, Path]:
    """Validate the freeze, evaluate Final OOD once, and write all artifacts."""
    started_at = datetime.now(timezone.utc).isoformat()
    validated = validate_final_ood_lock(protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"A14 output directory already exists; refusing to overwrite: "
            f"{output_dir}"
        )
    resolved_device = (
        device if isinstance(device, torch.device) else resolve_device(device)
    )
    rows, row_metadata = load_sealed_final_ood_rows(validated)
    source_order = list(row_metadata["selected_source_keys"])
    speaker_order = list(row_metadata["speaker_sensitivity_keys"])
    frontend = MfccFrontend(MfccConfig(causal=True)).to(resolved_device)
    seed_results: dict[int, dict[str, Any]] = {}
    data_root = Path(data_root)
    for seed in SEEDS:
        checkpoint = next(
            record
            for record in validated.lock_record["verified_checkpoints"]
            if int(record["seed"]) == seed
        )
        model, payload, refinement = load_adaptive_model(
            Path(str(checkpoint["path"])),
            resolved_device,
        )
        if tuple(refinement.dilations) != RF384_DILATIONS:
            raise ValueError(f"seed {seed} loaded a non-RF384 refinement")
        print(
            f"seed {seed}: epoch={payload.get('epoch')} "
            f"device={resolved_device} rows={len(rows)}",
            flush=True,
        )
        seed_results[seed] = _accumulate_seed(
            rows,
            seed=seed,
            model=model,
            frontend=frontend,
            device=resolved_device,
            data_root=data_root,
            source_order=source_order,
            speaker_order=speaker_order,
            progress_every=progress_every,
        )
        del model

    lock_record = dict(validated.lock_record)
    lock_record["sealed_access"] = {
        "status": "LOADED_AFTER_FULL_LOCK_VALIDATION",
        "rows_loaded": True,
        "rows": len(rows),
        "sources": len(source_order),
        "speaker_sensitivity_keys": len(speaker_order),
        "causal_frames": int(
            sum(
                result["evaluated_frames"]
                for result in seed_results.values()
            )
        ),
    }
    lock_record["run_provenance"] = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "primary_seed": PRIMARY_SEED,
        "seeds": list(SEEDS),
        "single_execution": True,
    }
    result = build_final_ood_result(
        validated,
        seed_results=seed_results,
        row_metadata=row_metadata,
        lock_record=lock_record,
    )
    return write_artifacts(result, seed_results, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen A14 lock and run the untouched Final OOD "
            "confirmation exactly once."
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = run_final_ood(
        protocol_path=args.protocol_path,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device=args.device,
        progress_every=args.progress_every,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
