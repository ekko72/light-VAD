# -*- coding: utf-8 -*-
"""Measure the A11 shared-encoder conditional-compute realization.

The primary timing boundary is cached causal MFCC features.  Each case is
executed in a fresh process with one CPU thread so that process peak RSS,
thread-pool state and refinement caches cannot leak between operating points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveStreamingVAD,
    confidence_from_logits,
)
from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    choose_confidence_threshold,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    build_evaluation_items,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
DEFAULT_MANIFEST = (
    DEFAULT_DATA_ROOT / "manifests" / "LibriSpeech_test_medium.tsv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a11_conditional_compute"
)
DEFAULT_RF384_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025_rf384"
    / "best.pt"
)
DEFAULT_RF64_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025"
    / "best.pt"
)
DEFAULT_RF384_PREDICTIONS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
DEFAULT_RF64_PREDICTIONS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf64_fixed130"
    / "frame_predictions.npz"
)
DEFAULT_ACTIVATION_BUDGETS = (0.02, 0.05, 0.10, 0.20)
NEAR_EQUIVALENT_RELATIVE_GAP = 0.05
FRAME_DURATION_MS = 1000.0 * FRAME_HOP / 16_000.0


@dataclass(frozen=True)
class LatencyCase:
    """One isolated process-level timing configuration."""

    name: str
    mode: str
    checkpoint: Path
    threshold: float | None = None
    rf_span: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint"] = str(self.checkpoint)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatencyCase":
        return cls(
            name=str(payload["name"]),
            mode=str(payload["mode"]),
            checkpoint=Path(str(payload["checkpoint"])),
            threshold=(
                None
                if payload.get("threshold") is None
                else float(payload["threshold"])
            ),
            rf_span=(
                None
                if payload.get("rf_span") is None
                else int(payload["rf_span"])
            ),
        )


def binary_detection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    """Return F1, false-alarm rate and miss rate at threshold 0.5."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")
    if labels.size == 0:
        raise ValueError("metrics require at least one frame")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    prediction = scores >= 0.5
    truth = labels.astype(bool)
    tp = int(np.count_nonzero(prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "frames": int(labels.size),
        "speech": int(np.count_nonzero(truth)),
        "silence": int(np.count_nonzero(~truth)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": fp / (fp + tn) if fp + tn else None,
        "miss_rate": fn / (fn + tp) if fn + tp else None,
    }


def _load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"prediction bundle not found: {path}")
    required = {
        "labels",
        "short_scores",
        "full_adaptive_scores",
        "test_mask",
        "calibration_mask",
    }
    with np.load(path) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        return {
            name: np.asarray(payload[name]).copy()
            for name in required
        }


def load_a11_reference(
    rf64_predictions: Path,
    rf384_predictions: Path,
) -> dict[str, np.ndarray]:
    """Load and validate the two frozen prediction bundles used by A11."""
    rf64 = _load_prediction_bundle(rf64_predictions)
    rf384 = _load_prediction_bundle(rf384_predictions)
    common = (
        "labels",
        "short_scores",
        "test_mask",
        "calibration_mask",
    )
    for name in common:
        if rf64[name].shape != rf384[name].shape:
            raise ValueError(
                f"RF64/RF384 {name} shapes differ: "
                f"{rf64[name].shape} vs {rf384[name].shape}"
            )
        if not np.array_equal(rf64[name], rf384[name]):
            raise ValueError(f"RF64/RF384 {name} arrays differ")
    return {
        "labels": rf384["labels"].astype(np.int64),
        "short_scores": rf384["short_scores"].astype(np.float64),
        "rf64_full_scores": rf64["full_adaptive_scores"].astype(np.float64),
        "rf384_full_scores": rf384["full_adaptive_scores"].astype(
            np.float64
        ),
        "test_mask": rf384["test_mask"].astype(bool),
        "calibration_mask": rf384["calibration_mask"].astype(bool),
    }


def derive_activation_thresholds(
    short_scores: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    activation_budgets: Sequence[float],
) -> list[dict[str, Any]]:
    """Choose every target threshold from calibration Short scores only."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if scores.shape != mask.shape:
        raise ValueError("short scores and calibration mask must match")
    if not np.any(mask):
        raise ValueError("calibration mask is empty")
    budgets = sorted(set(float(value) for value in activation_budgets))
    if not budgets:
        raise ValueError("at least one activation budget is required")
    if any(not 0.0 < value <= 1.0 for value in budgets):
        raise ValueError("activation budgets must be in (0, 1]")

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        threshold, selected = choose_confidence_threshold(
            scores[mask],
            activation_rate=budget,
        )
        rows.append(
            {
                "name": f"adaptive_{int(round(budget * 100))}pct",
                "label": f"Adaptive {100.0 * budget:.0f}%",
                "activation_target": float(budget),
                "threshold": float(threshold),
                "calibration_activation_rate": float(np.mean(selected)),
            }
        )
    return rows


def build_accuracy_rows(
    reference: Mapping[str, np.ndarray],
    *,
    activation_budgets: Sequence[float] = DEFAULT_ACTIVATION_BUDGETS,
) -> list[dict[str, Any]]:
    """Compute test-set accuracy for Short, fixed RF and adaptive points."""
    required = {
        "labels",
        "short_scores",
        "rf64_full_scores",
        "rf384_full_scores",
        "test_mask",
        "calibration_mask",
    }
    missing = sorted(required - set(reference))
    if missing:
        raise ValueError(f"reference bundle is missing arrays: {missing}")
    labels = np.asarray(reference["labels"], dtype=np.int64).reshape(-1)
    short_scores = np.asarray(
        reference["short_scores"], dtype=np.float64
    ).reshape(-1)
    rf64_scores = np.asarray(
        reference["rf64_full_scores"], dtype=np.float64
    ).reshape(-1)
    rf384_scores = np.asarray(
        reference["rf384_full_scores"], dtype=np.float64
    ).reshape(-1)
    test_mask = np.asarray(reference["test_mask"], dtype=bool).reshape(-1)
    calibration_mask = np.asarray(
        reference["calibration_mask"], dtype=bool
    ).reshape(-1)
    arrays = (
        short_scores,
        rf64_scores,
        rf384_scores,
        test_mask,
        calibration_mask,
    )
    if any(array.shape != labels.shape for array in arrays):
        raise ValueError("reference arrays must have equal shape")
    if not np.any(test_mask):
        raise ValueError("test mask is empty")
    if np.any(test_mask & calibration_mask):
        raise ValueError("calibration and test masks overlap")

    thresholds = derive_activation_thresholds(
        short_scores,
        calibration_mask,
        activation_budgets=activation_budgets,
    )
    rows: list[dict[str, Any]] = []
    base_specs = (
        (
            "short",
            "Short",
            "short",
            short_scores,
            0.0,
            None,
        ),
        (
            "fixed_rf64",
            "RF64 AlwaysRefine",
            "fixed",
            rf64_scores,
            1.0,
            None,
        ),
        (
            "fixed_rf384",
            "RF384 AlwaysRefine",
            "fixed",
            rf384_scores,
            1.0,
            None,
        ),
    )
    for name, label, kind, scores, activation, threshold in base_specs:
        metrics = binary_detection_metrics(labels[test_mask], scores[test_mask])
        rows.append(
            {
                "name": name,
                "label": label,
                "kind": kind,
                "activation_target": None,
                "activation_rate": float(activation),
                "threshold": threshold,
                **metrics,
            }
        )

    for threshold_row in thresholds:
        threshold = float(threshold_row["threshold"])
        selected = np.abs(short_scores - 0.5) <= threshold
        adaptive_scores = np.where(selected, rf384_scores, short_scores)
        metrics = binary_detection_metrics(
            labels[test_mask],
            adaptive_scores[test_mask],
        )
        rows.append(
            {
                **threshold_row,
                "kind": "adaptive",
                "activation_rate": float(np.mean(selected[test_mask])),
                **metrics,
            }
        )
    return rows


def latency_cases_from_accuracy(
    accuracy_rows: Sequence[Mapping[str, Any]],
    *,
    rf64_checkpoint: Path,
    rf384_checkpoint: Path,
) -> list[LatencyCase]:
    """Create one isolated timing case per required operating point."""
    indexed = {str(row["name"]): row for row in accuracy_rows}
    required = {"short", "fixed_rf64", "fixed_rf384"}
    missing = sorted(required - set(indexed))
    if missing:
        raise ValueError(f"accuracy rows are missing operating points: {missing}")

    cases = [
        LatencyCase(
            name="short_core",
            mode="short_core",
            checkpoint=rf384_checkpoint,
        ),
        LatencyCase(
            name="short",
            mode="short_gate",
            checkpoint=rf384_checkpoint,
            threshold=float(indexed["short"].get("threshold") or 0.0),
        ),
        LatencyCase(
            name="fixed_rf64",
            mode="always_refine",
            checkpoint=rf64_checkpoint,
            rf_span=64,
        ),
        LatencyCase(
            name="fixed_rf384",
            mode="always_refine",
            checkpoint=rf384_checkpoint,
            rf_span=384,
        ),
    ]
    for row in accuracy_rows:
        if row.get("kind") != "adaptive":
            continue
        cases.append(
            LatencyCase(
                name=str(row["name"]),
                mode="adaptive",
                checkpoint=rf384_checkpoint,
                threshold=float(row["threshold"]),
                rf_span=384,
            )
        )
    return cases


def select_benchmark_items(
    items: Sequence[Any],
    limit: int,
) -> list[Any]:
    """Select a deterministic condition-balanced subset of utterances."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(items) <= limit:
        return list(items)

    groups: dict[tuple[str, str, str], list[Any]] = {}
    for item in items:
        key = (
            str(getattr(item, "noise_name", "")),
            str(getattr(item, "snr_db", "")),
            "clean" if bool(getattr(item, "is_clean", False)) else "noisy",
        )
        groups.setdefault(key, []).append(item)
    selected: list[Any] = []
    depth = 0
    while len(selected) < limit:
        progressed = False
        for key in sorted(groups):
            group = groups[key]
            if depth >= len(group):
                continue
            selected.append(group[depth])
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
        depth += 1
    return selected


def _run_latency_pass(
    model: Any,
    features: torch.Tensor,
    case: LatencyCase,
) -> tuple[list[float], int, int, int]:
    """Run one cached-MFCC stream and return per-frame wall times."""
    if features.dim() != 3 or features.shape[0] != 1:
        raise ValueError("latency features must have shape [1, feat, T]")
    if features.shape[-1] == 0:
        raise ValueError("latency features must contain at least one frame")
    durations_ms: list[float] = []
    selected_count = 0
    total_frames = int(features.shape[-1])
    cache_bytes = 0

    if case.mode in ("short_core", "short_gate"):
        state = model.short_model.init_stream_state(features[..., :1])
        for start in range(total_frames):
            started = time.perf_counter_ns()
            encoded, state = model.short_model.encode_stream(
                features[..., start : start + 1],
                state,
            )
            short_logits = model.short_model.classifier(encoded)
            if case.mode == "short_gate":
                selected_count += int(
                    torch.count_nonzero(
                        confidence_from_logits(short_logits)
                        <= float(case.threshold or 0.0)
                    ).item()
                )
            durations_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
        if state is not None:
            cache_bytes = sum(
                int(cache.numel() * cache.element_size())
                for block in state
                for cache in block
                if cache is not None
            )
        return durations_ms, selected_count, cache_bytes, total_frames

    if case.mode == "adaptive":
        threshold = float(case.threshold)
        stream = AdaptiveStreamingVAD(model, activation_threshold=threshold)
    elif case.mode == "always_refine":
        stream = AdaptiveStreamingVAD(model, activation_threshold=0.5)
    else:
        raise ValueError(f"unknown latency case mode: {case.mode}")

    for start in range(total_frames):
        started = time.perf_counter_ns()
        output = stream.forward_components(features[..., start : start + 1])
        durations_ms.append(
            (time.perf_counter_ns() - started) / 1_000_000.0
        )
        selected_count += int(torch.count_nonzero(output.selected).item())
    cache_bytes = int(stream.cache_bytes)
    return durations_ms, selected_count, cache_bytes, total_frames


def benchmark_refinement_kernel(
    model: Any,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    """Measure an isolated one-frame sparse refinement kernel call."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    history_frames = max(int(model.refinement.lookback_frames) + 1, 1)
    encoded = torch.randn(1, model.refinement.config.in_channels, history_frames)
    selected = torch.zeros(1, history_frames, dtype=torch.bool)
    selected[0, -1] = True
    with torch.inference_mode():
        for _ in range(warmup):
            model.refinement.forward_sparse(encoded, selected)
        durations_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model.refinement.forward_sparse(encoded, selected)
            durations_ms.append(
                (time.perf_counter_ns() - started) / 1_000_000.0
            )
    return latency_summary(durations_ms)


def _count_analytical_macs_for_model(model: Any) -> int:
    """Return one-frame Conv1d/Linear MACs using module hooks."""
    total_macs = 0
    handles = []

    def conv_hook(
        module: torch.nn.Conv1d,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal total_macs
        input_tensor = inputs[0]
        kernel = int(module.kernel_size[0])
        in_channels_per_group = int(input_tensor.shape[1]) // int(module.groups)
        total_macs += (
            int(output.numel())
            * in_channels_per_group
            * kernel
        )

    def linear_hook(
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal total_macs
        total_macs += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        dummy_frames = 256
        with torch.inference_mode():
            model.short_model(torch.zeros(1, 64, dummy_frames))
    finally:
        for handle in handles:
            handle.remove()
    if total_macs % dummy_frames:
        raise RuntimeError("analytical Short MAC count is not frame aligned")
    return int(total_macs // dummy_frames)


def latency_summary(durations_ms: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(durations_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency samples must not be empty")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "frames": int(values.size),
        "mean_ms_per_frame": float(np.mean(values)),
        "median_ms_per_frame": float(np.median(values)),
        "p95_ms_per_frame": float(np.quantile(values, 0.95)),
        "p99_ms_per_frame": float(np.quantile(values, 0.99)),
        "total_ms": float(np.sum(values)),
    }


def collect_environment() -> dict[str, Any]:
    """Record process and machine state that can affect CPU timing."""
    process = psutil.Process()
    affinity = sorted(int(value) for value in process.cpu_affinity())
    try:
        cpu_frequency = psutil.cpu_freq()
    except (AttributeError, OSError):
        cpu_frequency = None
    power_scheme = None
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout:
            power_scheme = completed.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    try:
        priority = int(process.nice())
    except (AttributeError, OSError, psutil.Error):
        priority = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "numpy": np.__version__,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "cpu_affinity": affinity,
        "cpu_frequency": (
            None
            if cpu_frequency is None
            else {
                "current_mhz": float(cpu_frequency.current),
                "min_mhz": (
                    None
                    if cpu_frequency.min is None
                    else float(cpu_frequency.min)
                ),
                "max_mhz": (
                    None
                    if cpu_frequency.max is None
                    else float(cpu_frequency.max)
                ),
            }
        ),
        "power_scheme": power_scheme,
        "process_priority": priority,
    }


def configure_worker_process(
    *,
    affinity_index: int,
    high_priority: bool,
) -> dict[str, Any]:
    """Pin and prioritize the timing process on a best-effort basis."""
    process = psutil.Process()
    available = sorted(int(value) for value in process.cpu_affinity())
    if not available:
        raise RuntimeError("process has no CPU affinity mask")
    selected_cpu = available[int(affinity_index) % len(available)]
    process.cpu_affinity([selected_cpu])
    priority_changed = False
    priority_error = None
    if high_priority and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
        try:
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            priority_changed = True
        except (OSError, psutil.Error) as error:
            priority_error = str(error)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "selected_cpu": selected_cpu,
        "available_cpus": available,
        "high_priority_requested": bool(high_priority),
        "high_priority_applied": priority_changed,
        "priority_error": priority_error,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def run_worker(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one latency case in the current process."""
    case = LatencyCase.from_dict(config["case"])
    process_state = configure_worker_process(
        affinity_index=int(config.get("affinity_index", 0)),
        high_priority=bool(config.get("high_priority", True)),
    )
    device = torch.device("cpu")
    model, _, refinement_config = load_adaptive_model(
        case.checkpoint,
        device,
    )
    model.eval()
    model.freeze_short_model()
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    items = build_evaluation_items(
        Path(str(config["manifest"])),
        generated_root=Path(str(config["data_root"])) / "generated",
        label_root=Path(str(config["data_root"])) / "labels",
        librispeech_root=Path(str(config["librispeech_root"])),
        row_sample=int(config["row_sample"]),
        seed=int(config["seed"]),
        include_clean=bool(config.get("include_clean", True)),
        limit=None,
    )
    selected_items = select_benchmark_items(
        items,
        int(config["utterance_limit"]),
    )
    if not selected_items:
        raise RuntimeError("no benchmark utterances were selected")

    feature_tensors: list[torch.Tensor] = []
    utterance_seconds: list[float] = []
    for item in selected_items:
        waveform = read_int16_audio(item.audio_path)
        tensor = torch.from_numpy(
            waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        features = frontend(tensor.to(device, non_blocking=True))
        feature_tensors.append(features)
        utterance_seconds.append(
            float(features.shape[-1]) * FRAME_DURATION_MS / 1000.0
        )
    process = psutil.Process()
    baseline_rss = int(process.memory_info().rss)

    warmup_frames = int(config.get("warmup_frames", 0))
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    if warmup_frames:
        warm_features = feature_tensors[0][
            ..., : min(warmup_frames, feature_tensors[0].shape[-1])
        ]
        with torch.inference_mode():
            _run_latency_pass(model, warm_features, case)

    refinement_kernel = None
    if case.mode in ("adaptive", "always_refine"):
        refinement_kernel = benchmark_refinement_kernel(
            model,
            warmup=int(config.get("micro_warmup", 30)),
            repeats=int(config.get("micro_repeats", 200)),
        )

    all_durations: list[float] = []
    selected_frames = 0
    total_frames = 0
    cache_bytes_values: list[int] = []
    repeats = int(config.get("repeats", 1))
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    with torch.inference_mode():
        for _ in range(repeats):
            for features in feature_tensors:
                durations, selected, cache_bytes, frames = _run_latency_pass(
                    model,
                    features,
                    case,
                )
                all_durations.extend(durations)
                selected_frames += selected
                total_frames += frames
                cache_bytes_values.append(cache_bytes)

    memory_info = process.memory_info()
    peak_rss = int(
        getattr(memory_info, "peak_wset", memory_info.rss)
    )
    latency = latency_summary(all_durations)
    audio_seconds = float(sum(utterance_seconds)) * repeats
    latency["rtf"] = float(latency["total_ms"] / 1000.0 / audio_seconds)
    latency["mean_latency_to_frame_ratio"] = float(
        latency["mean_ms_per_frame"] / FRAME_DURATION_MS
    )
    result = {
        "case": case.to_dict(),
        "latency": latency,
        "activation_rate": (
            float(selected_frames / total_frames) if total_frames else None
        ),
        "selected_frames": int(selected_frames),
        "frames": int(total_frames),
        "audio_seconds": audio_seconds,
        "cache_bytes": int(max(cache_bytes_values, default=0)),
        "cache_bytes_values": sorted(set(cache_bytes_values)),
        "peak_rss_bytes": peak_rss,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_delta_bytes": int(max(0, peak_rss - baseline_rss)),
        "short_macs_per_frame": int(
            _count_analytical_macs_for_model(model)
        ),
        "refinement_macs_per_selected_frame": int(
            model.refinement.estimated_macs_per_selected_frame()
        ),
        "refinement_kernel": refinement_kernel,
        "lookback_frames": int(refinement_config.lookback_frames),
        "utterances": len(selected_items),
        "utterance_seconds": utterance_seconds,
        "frame_durations_ms": all_durations,
        "environment": collect_environment(),
        "process_state": process_state,
    }
    return result


def _run_latency_suite(
    cases: Sequence[LatencyCase],
    *,
    base_config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="a11-workers-",
        dir=output_dir,
    ) as directory:
        temp_dir = Path(directory)
        for case in cases:
            print(f"A11 timing case: {case.name}", flush=True)
            result_path = temp_dir / f"{case.name}.json"
            config_path = temp_dir / f"{case.name}.config.json"
            config = {
                **dict(base_config),
                "case": case.to_dict(),
                "result_path": str(result_path),
            }
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    (
                        "reproductions.difficulty_adaptive_context."
                        "benchmark_a11_conditional_compute"
                    ),
                    "--worker-config",
                    str(config_path),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
            with open(result_path, "r", encoding="utf-8") as handle:
                results[case.name] = json.load(handle)
    return results


def _merge_accuracy_and_latency(
    accuracy_rows: Sequence[Mapping[str, Any]],
    latency_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine frozen prediction metrics and isolated CPU measurements."""
    required_latency = {
        "short",
        "short_core",
        "fixed_rf64",
        "fixed_rf384",
    }
    missing = sorted(required_latency - set(latency_results))
    if missing:
        raise ValueError(f"latency results are missing cases: {missing}")
    short_core = latency_results["short_core"]
    short_gate = latency_results["short"]
    gate_overhead = float(
        short_gate["latency"]["mean_ms_per_frame"]
        - short_core["latency"]["mean_ms_per_frame"]
    )

    rows: list[dict[str, Any]] = []
    for accuracy in accuracy_rows:
        name = str(accuracy["name"])
        latency = latency_results.get(name)
        if latency is None:
            raise ValueError(f"latency result is missing for {name}")
        short_macs = int(latency["short_macs_per_frame"])
        refinement_macs = int(
            latency["refinement_macs_per_selected_frame"]
        )
        benchmark_activation = float(latency["activation_rate"] or 0.0)
        test_activation = float(accuracy["activation_rate"])
        analytical_macs_test = int(
            round(short_macs + test_activation * refinement_macs)
        )
        analytical_macs_benchmark = int(
            round(short_macs + benchmark_activation * refinement_macs)
        )
        measured_mean = float(latency["latency"]["mean_ms_per_frame"])
        refinement_kernel = latency.get("refinement_kernel")
        scheduling_overhead = None
        if accuracy["kind"] == "adaptive" and refinement_kernel is not None:
            modeled_without_scheduling = (
                float(short_gate["latency"]["mean_ms_per_frame"])
                + benchmark_activation
                * float(refinement_kernel["mean_ms_per_frame"])
            )
            scheduling_overhead = measured_mean - modeled_without_scheduling
        rows.append(
            {
                **dict(accuracy),
                "benchmark_activation_rate": benchmark_activation,
                "test_activation_rate": test_activation,
                "mean_ms_per_frame": measured_mean,
                "median_ms_per_frame": float(
                    latency["latency"]["median_ms_per_frame"]
                ),
                "p95_ms_per_frame": float(
                    latency["latency"]["p95_ms_per_frame"]
                ),
                "p99_ms_per_frame": float(
                    latency["latency"]["p99_ms_per_frame"]
                ),
                "rtf": float(latency["latency"]["rtf"]),
                "analytical_macs_per_frame": analytical_macs_test,
                "analytical_macs_per_frame_test_activation": (
                    analytical_macs_test
                ),
                "analytical_macs_per_frame_benchmark_activation": (
                    analytical_macs_benchmark
                ),
                "short_macs_per_frame": short_macs,
                "refinement_macs_per_selected_frame": refinement_macs,
                "cache_bytes": int(latency["cache_bytes"]),
                "peak_rss_bytes": int(latency["peak_rss_bytes"]),
                "peak_rss_delta_bytes": int(
                    latency["peak_rss_delta_bytes"]
                ),
                "router_overhead_ms_per_frame": (
                    gate_overhead
                    if accuracy["kind"] == "adaptive"
                    else None
                ),
                "scheduling_overhead_ms_per_frame": scheduling_overhead,
                "refinement_kernel": refinement_kernel,
                "benchmark_frames": int(latency["frames"]),
                "benchmark_utterances": int(latency["utterances"]),
                "benchmark_seconds": float(latency["audio_seconds"]),
            }
        )

    short_macs = int(rows[0]["short_macs_per_frame"])
    rows.append(
        {
            "name": "short_core",
            "label": "Short encoder + classifier",
            "kind": "overhead",
            "activation_target": None,
            "activation_rate": 0.0,
            "benchmark_activation_rate": 0.0,
            "test_activation_rate": 0.0,
            "threshold": None,
            "mean_ms_per_frame": float(
                short_core["latency"]["mean_ms_per_frame"]
            ),
            "median_ms_per_frame": float(
                short_core["latency"]["median_ms_per_frame"]
            ),
            "p95_ms_per_frame": float(
                short_core["latency"]["p95_ms_per_frame"]
            ),
            "p99_ms_per_frame": float(
                short_core["latency"]["p99_ms_per_frame"]
            ),
            "rtf": float(short_core["latency"]["rtf"]),
            "analytical_macs_per_frame": short_macs,
            "analytical_macs_per_frame_test_activation": short_macs,
            "analytical_macs_per_frame_benchmark_activation": short_macs,
            "short_macs_per_frame": short_macs,
            "refinement_macs_per_selected_frame": 0,
            "cache_bytes": int(short_core["cache_bytes"]),
            "peak_rss_bytes": int(short_core["peak_rss_bytes"]),
            "peak_rss_delta_bytes": int(
                short_core["peak_rss_delta_bytes"]
            ),
            "router_overhead_ms_per_frame": None,
            "scheduling_overhead_ms_per_frame": None,
            "refinement_kernel": None,
            "benchmark_frames": int(short_core["frames"]),
            "benchmark_utterances": int(short_core["utterances"]),
            "benchmark_seconds": float(short_core["audio_seconds"]),
            "gate_overhead_mean_ms_per_frame": gate_overhead,
        }
    )
    return rows


def assess_a11_go(
    rows: Sequence[Mapping[str, Any]],
    *,
    near_equivalent_relative_gap: float = NEAR_EQUIVALENT_RELATIVE_GAP,
) -> dict[str, Any]:
    """Apply the document's accuracy/latency strong and failure gates."""
    if not 0.0 <= near_equivalent_relative_gap < 1.0:
        raise ValueError("near-equivalent gap must be in [0, 1)")
    indexed = {str(row["name"]): row for row in rows}
    required = {"short", "adaptive_5pct", "fixed_rf384"}
    missing = sorted(required - set(indexed))
    if missing:
        raise ValueError(f"GO assessment is missing rows: {missing}")
    short_row = indexed["short"]
    adaptive_row = indexed["adaptive_5pct"]
    refine_row = indexed["fixed_rf384"]
    latency_short = float(short_row["mean_ms_per_frame"])
    latency_adaptive = float(adaptive_row["mean_ms_per_frame"])
    latency_refine = float(refine_row["mean_ms_per_frame"])
    f1_short = float(short_row["f1"])
    f1_adaptive = float(adaptive_row["f1"])
    f1_refine = float(refine_row["f1"])
    relative_refine_gap = (
        (latency_refine - latency_adaptive) / latency_refine
        if latency_refine > 0.0
        else None
    )
    near_equivalent = bool(
        relative_refine_gap is not None
        and relative_refine_gap <= near_equivalent_relative_gap
    )
    f1_order = bool(f1_short < f1_adaptive < f1_refine)
    latency_order = bool(
        latency_short < latency_adaptive < latency_refine
    )
    if near_equivalent:
        status = "NO_GO_EFFICIENCY_OVERHEAD"
        interpretation = (
            "Adaptive 5% is within the predeclared near-equivalent latency "
            "band of AlwaysRefine; conditional-compute efficiency is NO-GO "
            "until the implementation is optimized."
        )
    elif f1_order and latency_order:
        status = "STRONG_GO"
        interpretation = (
            "The 5% operating point satisfies the strict accuracy and "
            "latency ordering required by A11."
        )
    elif latency_order and f1_adaptive > f1_short:
        status = "CONDITIONAL_COMPUTE_GO"
        interpretation = (
            "Conditional compute lowers latency below AlwaysRefine and "
            "improves F1 over Short, but the full strict ordering is not met."
        )
    else:
        status = "MIXED_NOT_STRONG"
        interpretation = (
            "The measured ordering does not satisfy the A11 strong signal."
        )
    savings_fraction = (
        (latency_refine - latency_adaptive)
        / (latency_refine - latency_short)
        if latency_refine > latency_short
        else None
    )
    return {
        "status": status,
        "primary_activation": 0.05,
        "near_equivalent_relative_gap": near_equivalent_relative_gap,
        "f1_short": f1_short,
        "f1_adaptive": f1_adaptive,
        "f1_always_refine": f1_refine,
        "latency_short_mean_ms": latency_short,
        "latency_adaptive_mean_ms": latency_adaptive,
        "latency_always_refine_mean_ms": latency_refine,
        "f1_order_short_lt_adaptive_lt_refine": f1_order,
        "latency_order_short_lt_adaptive_lt_refine": latency_order,
        "relative_refine_latency_gap": relative_refine_gap,
        "adaptive_is_near_always_refine": near_equivalent,
        "conditional_cost_reduction_fraction": savings_fraction,
        "interpretation": interpretation,
    }


def _md_float(
    value: float | None,
    *,
    digits: int = 5,
) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _md_percent(value: float | None, *, digits: int = 3) -> str:
    return (
        "n/a"
        if value is None
        else f"{100.0 * float(value):.{digits}f}%"
    )


def _md_integer(value: int | None) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _md_mib(value: int | None) -> str:
    return "n/a" if value is None else f"{float(value) / 1024.0 / 1024.0:.2f}"


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty A11 CSV")
    excluded = {"refinement_kernel"}
    fieldnames = [
        key
        for key in rows[0]
        if key not in excluded
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def _write_markdown(
    summary: Mapping[str, Any],
    path: Path,
) -> None:
    protocol = summary["protocol"]
    environment = summary["environment"]
    rows = summary["rows"]
    operating = [
        row
        for row in rows
        if row["name"] != "short_core"
    ]
    short_core = next(
        row
        for row in rows
        if row["name"] == "short_core"
    )
    assessment = summary["go_assessment"]
    lines = [
        "# A11 Conditional Compute Realization",
        "",
        "## Protocol",
        "",
        f"- Manifest: `{protocol['manifest']}`",
        f"- RF384 checkpoint: `{protocol['rf384_checkpoint']}`",
        f"- RF64 checkpoint: `{protocol['rf64_checkpoint']}`",
        f"- RF384 predictions: `{protocol['rf384_predictions']}`",
        f"- RF64 predictions: `{protocol['rf64_predictions']}`",
        f"- Benchmark utterances: {protocol['utterance_limit']}",
        f"- Timed repeats per utterance: {protocol['repeats']}",
        f"- Warm-up frames: {protocol['warmup_frames']}",
        f"- Primary timing boundary: cached causal MFCC frames",
        f"- Streaming chunk: {protocol['chunk_frames']} frame(s)",
        f"- CPU: {environment['processor']}",
        f"- Affinity/threads: CPU {protocol['selected_cpu']}, "
        f"{environment['torch_num_threads']} torch thread(s)",
        f"- Power scheme: {environment['power_scheme'] or 'n/a'}",
        "",
        "- The threshold for every adaptive operating point is selected only "
        "from calibration-speaker Short scores. Final-test activation is "
        "observed, not forced.",
        "- Latency excludes MFCC extraction. RTF therefore describes the "
        "cached-feature streaming boundary, not full audio-to-decision RTF.",
        "- Activation percentage is not called compute percentage; MACs and "
        "wall-clock timing are reported separately.",
        "",
        "## Operating Points",
        "",
        "| Operating point | Test activation | Benchmark activation | Threshold | "
        "F1 | FAR | MR | Mean ms/frame | Median | P95 | P99 | RTF | "
        "Analytical MACs/frame (test act.) | Cache bytes | "
        "Peak RSS MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in operating:
        lines.append(
            f"| {row['label']} | "
            f"{_md_percent(row['test_activation_rate'])} | "
            f"{_md_percent(row['benchmark_activation_rate'])} | "
            f"{_md_float(row.get('threshold'), digits=6)} | "
            f"{_md_float(row['f1'])} | {_md_percent(row['far'])} | "
            f"{_md_percent(row['miss_rate'])} | "
            f"{_md_float(row['mean_ms_per_frame'])} | "
            f"{_md_float(row['median_ms_per_frame'])} | "
            f"{_md_float(row['p95_ms_per_frame'])} | "
            f"{_md_float(row['p99_ms_per_frame'])} | "
            f"{_md_float(row['rtf'])} | "
            f"{_md_integer(row['analytical_macs_per_frame_test_activation'])} | "
            f"{_md_integer(row['cache_bytes'])} | "
            f"{_md_mib(row['peak_rss_bytes'])} |"
        )

    lines.extend(
        [
            "",
            "## Router and Scheduling Overhead",
            "",
            f"- Short encoder + classifier mean: "
            f"{short_core['mean_ms_per_frame']:.5f} ms/frame.",
            f"- Short + gate mean: "
            f"{next(row for row in operating if row['name'] == 'short')['mean_ms_per_frame']:.5f} "
            "ms/frame.",
            f"- Gate-only overhead estimate: "
            f"{short_core['gate_overhead_mean_ms_per_frame']:.5f} ms/frame.",
            "- Scheduling overhead below is measured adaptive mean latency "
            "minus `Short+gate + observed activation * isolated refinement "
            "kernel`; it includes history concatenation, dispatch and other "
            "implementation overhead.",
            "",
            "| Adaptive point | Observed activation | Isolated refinement "
            "kernel ms/frame | Estimated scheduling overhead ms/frame |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in operating:
        if row["kind"] != "adaptive":
            continue
        kernel = row.get("refinement_kernel") or {}
        lines.append(
            f"| {row['label']} | "
            f"{_md_percent(row['benchmark_activation_rate'])} | "
            f"{_md_float(kernel.get('mean_ms_per_frame'))} | "
            f"{_md_float(row['scheduling_overhead_ms_per_frame'])} |"
        )

    lines.extend(
        [
            "",
            "## GO Assessment",
            "",
            f"- Primary operating point: adaptive 5%.",
            f"- F1 order `Short < Adaptive < AlwaysRefine`: "
            f"{assessment['f1_order_short_lt_adaptive_lt_refine']}.",
            f"- Latency order `Short < Adaptive < AlwaysRefine`: "
            f"{assessment['latency_order_short_lt_adaptive_lt_refine']}.",
            f"- Adaptive is near AlwaysRefine under the predeclared "
            f"{assessment['near_equivalent_relative_gap']:.1%} relative "
            f"latency-gap rule: {assessment['adaptive_is_near_always_refine']}.",
            f"- Status: **{assessment['status']}**.",
            f"- {assessment['interpretation']}",
            "",
            "## Limitations",
            "",
            "- The benchmark uses one frame per streaming call and cached "
            "MFCC input; it does not measure microphone capture or feature "
            "extraction.",
            "- CPU frequency pinning is not enforced. The active Windows "
            "power scheme and CPU affinity are recorded, but background "
            "frequency changes remain a possible noise source.",
            "- Peak RSS is collected from an isolated worker process per "
            "operating point; it is process peak working set, not tensor-"
            "level peak allocation.",
            "- Analytical MACs count Conv1d and Linear multiply-accumulates "
            "only. They exclude indexing, concatenation, activation, "
            "scheduling and memory traffic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in summary["rows"]
        if row["name"] != "short_core"
    ]
    if not rows:
        raise ValueError("cannot plot an empty A11 result")
    ordered_names = [
        "short",
        "fixed_rf64",
        "adaptive_2pct",
        "adaptive_5pct",
        "adaptive_10pct",
        "adaptive_20pct",
        "fixed_rf384",
    ]
    indexed = {str(row["name"]): row for row in rows}
    plotted = [indexed[name] for name in ordered_names if name in indexed]
    label_offsets = {
        "short": (6, 4),
        "fixed_rf64": (6, -12),
        "adaptive_2pct": (6, -12),
        "adaptive_5pct": (6, 5),
        "adaptive_10pct": (6, -14),
        "adaptive_20pct": (6, 8),
        "fixed_rf384": (-8, 8),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    x = [float(row["mean_ms_per_frame"]) for row in plotted]
    y = [float(row["f1"]) for row in plotted]
    axis.plot(x, y, color="tab:blue", linewidth=1.5, zorder=1)
    for row, x_value, y_value in zip(plotted, x, y):
        marker = "o"
        if row["kind"] == "fixed":
            marker = "s"
        elif row["kind"] == "short":
            marker = "^"
        axis.scatter(
            [x_value],
            [y_value],
            marker=marker,
            s=38,
            zorder=2,
        )
        axis.annotate(
            str(row["label"]),
            (x_value, y_value),
            xytext=label_offsets.get(str(row["name"]), (4, 5)),
            textcoords="offset points",
            fontsize=8,
            horizontalalignment=(
                "right"
                if str(row["name"]) == "fixed_rf384"
                else "left"
            ),
        )
    axis.set_xlabel("Mean latency (ms / cached-MFCC frame)")
    axis.set_ylabel("Final-test F1")
    axis.set_title("A11 accuracy-latency Pareto")
    axis.grid(alpha=0.25)
    axis.margins(x=0.08, y=0.08)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _build_protocol(
    args: argparse.Namespace,
    *,
    thresholds: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "librispeech_root": str(args.librispeech_root),
        "rf384_checkpoint": str(args.rf384_checkpoint),
        "rf64_checkpoint": str(args.rf64_checkpoint),
        "rf384_predictions": str(args.rf384_predictions),
        "rf64_predictions": str(args.rf64_predictions),
        "activation_budgets": [
            float(row["activation_target"]) for row in thresholds
        ],
        "thresholds": [dict(row) for row in thresholds],
        "row_sample": int(args.row_sample),
        "seed": int(args.seed),
        "utterance_limit": int(args.utterances),
        "repeats": int(args.repeats),
        "warmup_frames": int(args.warmup_frames),
        "chunk_frames": int(args.chunk_frames),
        "micro_warmup": int(args.micro_warmup),
        "micro_repeats": int(args.micro_repeats),
        "near_equivalent_relative_gap": float(
            args.near_equivalent_relative_gap
        ),
        "selected_cpu": int(environment["selected_cpu"]),
    }


def aggregate_results(
    *,
    accuracy_rows: Sequence[Mapping[str, Any]],
    latency_results: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _merge_accuracy_and_latency(accuracy_rows, latency_results)
    assessment = assess_a11_go(rows)
    return rows, assessment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark A11 shared-encoder conditional compute."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=DEFAULT_LIBRISPEECH_ROOT,
    )
    parser.add_argument(
        "--rf384-checkpoint",
        type=Path,
        default=DEFAULT_RF384_CHECKPOINT,
    )
    parser.add_argument(
        "--rf64-checkpoint",
        type=Path,
        default=DEFAULT_RF64_CHECKPOINT,
    )
    parser.add_argument(
        "--rf384-predictions",
        type=Path,
        default=DEFAULT_RF384_PREDICTIONS,
    )
    parser.add_argument(
        "--rf64-predictions",
        type=Path,
        default=DEFAULT_RF64_PREDICTIONS,
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
        default=list(DEFAULT_ACTIVATION_BUDGETS),
    )
    parser.add_argument("--row-sample", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--chunk-frames", type=int, default=1)
    parser.add_argument("--micro-warmup", type=int, default=30)
    parser.add_argument("--micro-repeats", type=int, default=200)
    parser.add_argument("--affinity-index", type=int, default=0)
    parser.add_argument(
        "--no-high-priority",
        action="store_true",
    )
    parser.add_argument(
        "--near-equivalent-relative-gap",
        type=float,
        default=NEAR_EQUIVALENT_RELATIVE_GAP,
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.row_sample <= 0:
        parser.error("--row-sample must be positive")
    if args.utterances <= 0:
        parser.error("--utterances must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.chunk_frames != 1:
        parser.error("A11 primary protocol fixes --chunk-frames at 1")
    if args.micro_warmup < 0:
        parser.error("--micro-warmup must be non-negative")
    if args.micro_repeats <= 0:
        parser.error("--micro-repeats must be positive")
    if not 0.0 <= args.near_equivalent_relative_gap < 1.0:
        parser.error("--near-equivalent-relative-gap must be in [0, 1)")
    if not args.activation_budgets:
        parser.error("--activation-budgets must not be empty")
    if any(
        not 0.0 < float(value) <= 1.0
        for value in args.activation_budgets
    ):
        parser.error("--activation-budgets values must be in (0, 1]")
    for path in (
        args.manifest,
        args.data_root,
        args.librispeech_root,
        args.rf384_checkpoint,
        args.rf64_checkpoint,
        args.rf384_predictions,
        args.rf64_predictions,
    ):
        if not path.exists():
            parser.error(f"path does not exist: {path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker_config is not None:
        with open(args.worker_config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        result = run_worker(config)
        result_path = Path(str(config["result_path"]))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return 0

    validate_args(parser, args)
    reference = load_a11_reference(
        args.rf64_predictions,
        args.rf384_predictions,
    )
    accuracy_rows = build_accuracy_rows(
        reference,
        activation_budgets=args.activation_budgets,
    )
    thresholds = [
        row for row in accuracy_rows if row.get("kind") == "adaptive"
    ]
    cases = latency_cases_from_accuracy(
        accuracy_rows,
        rf64_checkpoint=args.rf64_checkpoint,
        rf384_checkpoint=args.rf384_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency_results = _run_latency_suite(
        cases,
        base_config={
            "manifest": str(args.manifest),
            "data_root": str(args.data_root),
            "librispeech_root": str(args.librispeech_root),
            "row_sample": int(args.row_sample),
            "seed": int(args.seed),
            "include_clean": True,
            "utterance_limit": int(args.utterances),
            "repeats": int(args.repeats),
            "warmup_frames": int(args.warmup_frames),
            "micro_warmup": int(args.micro_warmup),
            "micro_repeats": int(args.micro_repeats),
            "affinity_index": int(args.affinity_index),
            "high_priority": not bool(args.no_high_priority),
        },
        output_dir=args.output_dir,
    )
    rows, assessment = aggregate_results(
        accuracy_rows=accuracy_rows,
        latency_results=latency_results,
    )
    environment = dict(
        next(iter(latency_results.values()))["environment"]
    )
    protocol = _build_protocol(
        args,
        thresholds=thresholds,
        environment={
            **environment,
            "selected_cpu": next(
                iter(latency_results.values())
            )["process_state"]["selected_cpu"],
        },
    )
    summary = {
        "protocol": protocol,
        "environment": environment,
        "rows": rows,
        "go_assessment": assessment,
        "latency_results": latency_results,
    }
    with open(
        args.output_dir / "a11_conditional_compute.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(rows, args.output_dir / "a11_conditional_compute.csv")
    _write_markdown(
        summary,
        args.output_dir / "a11_conditional_compute.md",
    )
    _write_plot(
        summary,
        args.output_dir / "a11_accuracy_latency_pareto.png",
    )
    print(
        f"A11 status={assessment['status']}; artifacts written to "
        f"{args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
