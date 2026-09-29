# -*- coding: utf-8 -*-
"""Run the A13 streaming and segment-level behaviour audit.

A13 is deliberately a diagnostic study.  It checks that selective refinement
does not consume future acoustic frames and that its frame-level gains do not
come with an obvious degradation in onset/offset alignment, short-speech
recall, clipping, false activation, or decision flicker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveStreamingVAD,
)
from reproductions.difficulty_adaptive_context.analyze_a10_value_predictability import (
    utterance_segments,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
)
from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


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
DEFAULT_ADAPTIVE_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025_rf384"
    / "best.pt"
)
DEFAULT_A12_RESULTS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a12_routing_robustness"
    / "a12_routing_robustness.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a13_streaming_audit"
)

FRAME_DURATION_SECONDS = 0.01
CONDITION_ORDER = ("clean", "20", "10", "5", "0", "-5")
METHOD_ORDER = ("short", "refine_only", "adaptive")
CRITICAL_GROUPS = ("overall", "noisy", "seen", "unseen")

REQUIRED_BUNDLE_FIELDS = (
    "labels",
    "short_scores",
    "adaptive_scores",
    "full_adaptive_scores",
    "selected",
    "test_mask",
    "calibration_mask",
    "source_key",
    "condition",
    "noise_name",
    "threshold",
)


@dataclass(frozen=True)
class SegmentProtocol:
    """Predeclared frame-to-segment conversion rules."""

    decision_threshold: float = 0.5
    short_speech_max_frames: int = 20
    onset_tolerance_frames: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1)")
        if self.short_speech_max_frames <= 0:
            raise ValueError("short_speech_max_frames must be positive")
        if self.onset_tolerance_frames < 0:
            raise ValueError("onset_tolerance_frames must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class Utterance:
    """One contiguous evaluated utterance variant."""

    start: int
    end: int
    source_key: str
    condition: str
    noise_name: str


def _as_scalar_float(value: np.ndarray, name: str) -> float:
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(scalar[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_a13_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen prediction bundle used by A13."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"prediction bundle not found: {bundle_path}")
    with np.load(bundle_path, allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_BUNDLE_FIELDS) - set(payload.files))
        if missing:
            raise ValueError(
                "prediction bundle is missing fields: " + ", ".join(missing)
            )
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in REQUIRED_BUNDLE_FIELDS
        }

    threshold = _as_scalar_float(arrays["threshold"], "threshold")
    if not 0.0 <= threshold <= 0.5:
        raise ValueError("threshold must be in [0, 0.5]")

    labels = arrays["labels"].astype(np.int64).reshape(-1)
    size = int(labels.size)
    if size == 0:
        raise ValueError("prediction bundle is empty")
    for name in (
        "short_scores",
        "adaptive_scores",
        "full_adaptive_scores",
    ):
        values = arrays[name].astype(np.float64).reshape(-1)
        if values.size != size:
            raise ValueError(
                f"{name} has size {values.size}, expected {size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
        arrays[name] = values

    for name in (
        "selected",
        "test_mask",
        "calibration_mask",
    ):
        values = arrays[name].astype(bool).reshape(-1)
        if values.size != size:
            raise ValueError(
                f"{name} has size {values.size}, expected {size}"
            )
        arrays[name] = values

    for name in ("source_key", "condition", "noise_name"):
        values = arrays[name].astype(str).reshape(-1)
        if values.size != size:
            raise ValueError(
                f"{name} has size {values.size}, expected {size}"
            )
        if np.any(np.char.str_len(values) == 0):
            raise ValueError(f"{name} contains an empty value")
        arrays[name] = values

    arrays["labels"] = labels
    arrays["threshold"] = threshold
    arrays["path"] = str(bundle_path)
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    if not np.any(arrays["test_mask"]):
        raise ValueError("test_mask is empty")
    if not np.any(arrays["calibration_mask"]):
        raise ValueError("calibration_mask is empty")
    if np.any(arrays["test_mask"] & arrays["calibration_mask"]):
        raise ValueError("test and calibration masks overlap")
    return arrays


def build_utterances(bundle: dict[str, Any]) -> list[Utterance]:
    """Return homogeneous, contiguous evaluated utterance variants."""
    raw_segments = utterance_segments(
        bundle["source_key"],
        bundle["noise_name"],
        bundle["condition"],
    )
    result: list[Utterance] = []
    for start, end in raw_segments:
        test_values = bundle["test_mask"][start:end]
        if np.any(test_values) and not np.all(test_values):
            raise ValueError(
                "test_mask crosses an utterance boundary; segment metrics "
                "require whole test utterances"
            )
        if not np.any(test_values):
            continue
        result.append(
            Utterance(
                start=int(start),
                end=int(end),
                source_key=str(bundle["source_key"][start]),
                condition=str(bundle["condition"][start]),
                noise_name=str(bundle["noise_name"][start]),
            )
        )
    if not result:
        raise ValueError("no test utterances were found")
    return result


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous true runs as half-open ``[start, end)`` intervals."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate(([False], values, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _first_true(values: np.ndarray, start: int, end: int) -> int | None:
    indices = np.flatnonzero(values[int(start) : int(end)])
    if indices.size == 0:
        return None
    return int(start) + int(indices[0])


def _first_false(values: np.ndarray, start: int, end: int) -> int | None:
    indices = np.flatnonzero(~values[int(start) : int(end)])
    if indices.size == 0:
        return None
    return int(start) + int(indices[0])


def _safe_quantile(values: Sequence[int], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def evaluate_segment_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    utterances: Sequence[Utterance],
    protocol: SegmentProtocol,
) -> dict[str, Any]:
    """Compute fixed segment metrics for one score stream."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    predictions = scores >= float(protocol.decision_threshold)

    total_frames = 0
    speech_frames = 0
    silence_frames = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    speech_run_count = 0
    short_speech_run_count = 0
    detected_speech_run_count = 0
    detected_short_speech_run_count = 0
    whole_speech_run_misses = 0
    whole_utterance_misses = 0
    short_speech_true_positive = 0
    short_speech_false_negative = 0
    onset_delays: list[int] = []
    offset_delays: list[int] = []
    offset_unresolved = 0
    predicted_run_lengths: list[int] = []
    transition_count = 0

    for utterance in utterances:
        start, end = int(utterance.start), int(utterance.end)
        if not 0 <= start < end <= labels.size:
            raise ValueError(f"invalid utterance bounds: {start}:{end}")
        local_labels = labels[start:end]
        local_predictions = predictions[start:end]
        speech = local_labels == 1
        nonspeech = ~speech

        total_frames += int(local_labels.size)
        speech_frames += int(np.count_nonzero(speech))
        silence_frames += int(np.count_nonzero(nonspeech))
        true_positive += int(np.count_nonzero(local_predictions & speech))
        false_positive += int(np.count_nonzero(local_predictions & nonspeech))
        true_negative += int(np.count_nonzero(~local_predictions & nonspeech))
        false_negative += int(np.count_nonzero(~local_predictions & speech))

        speech_runs = find_runs(speech)
        if speech_runs and not np.any(local_predictions & speech):
            whole_utterance_misses += 1

        for run_start, run_end in speech_runs:
            speech_run_count += 1
            duration = int(run_end - run_start)
            is_short = duration <= int(protocol.short_speech_max_frames)
            if is_short:
                short_speech_run_count += 1
                short_speech_true_positive += int(
                    np.count_nonzero(local_predictions[run_start:run_end])
                )
                short_speech_false_negative += int(
                    np.count_nonzero(~local_predictions[run_start:run_end])
                )

            detection_end = min(
                int(end),
                int(run_end) + int(protocol.onset_tolerance_frames),
            )
            detected = _first_true(
                local_predictions,
                int(run_start),
                detection_end,
            )
            if detected is None:
                whole_speech_run_misses += 1
                continue

            detected_speech_run_count += 1
            onset_delays.append(int(detected - run_start))
            if is_short:
                detected_short_speech_run_count += 1

            offset = _first_false(
                local_predictions,
                int(run_end),
                int(end),
            )
            if offset is None:
                offset_unresolved += 1
            else:
                offset_delays.append(int(offset - run_end))

        local_predicted_runs = find_runs(local_predictions)
        predicted_run_lengths.extend(
            int(run_end - run_start)
            for run_start, run_end in local_predicted_runs
        )
        if local_predictions.size > 1:
            transition_count += int(
                np.count_nonzero(
                    local_predictions[1:] != local_predictions[:-1]
                )
            )

    predicted_run_count = len(predicted_run_lengths)
    single_frame_runs = sum(length == 1 for length in predicted_run_lengths)
    at_most_two_frame_runs = sum(
        length <= 2 for length in predicted_run_lengths
    )
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    short_segment_recall = _safe_rate(
        detected_short_speech_run_count,
        short_speech_run_count,
    )
    short_frame_recall = _safe_rate(
        short_speech_true_positive,
        short_speech_true_positive + short_speech_false_negative,
    )

    return {
        "frames": total_frames,
        "speech_frames": speech_frames,
        "silence_frames": silence_frames,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "speech_runs": speech_run_count,
        "short_speech_runs": short_speech_run_count,
        "detected_speech_runs": detected_speech_run_count,
        "detected_short_speech_runs": detected_short_speech_run_count,
        "short_speech_recall": short_segment_recall,
        "short_speech_frame_recall": short_frame_recall,
        "whole_speech_run_misses": whole_speech_run_misses,
        "whole_utterance_misses": whole_utterance_misses,
        "whole_utterance_miss_rate": _safe_rate(
            whole_utterance_misses,
            sum(
                1
                for utterance in utterances
                if np.any(labels[utterance.start : utterance.end] == 1)
            ),
        ),
        "onset_delay_count": len(onset_delays),
        "onset_delay_median_frames": _safe_quantile(onset_delays, 0.5),
        "onset_delay_p90_frames": _safe_quantile(onset_delays, 0.9),
        "offset_delay_count": len(offset_delays),
        "offset_delay_median_frames": _safe_quantile(offset_delays, 0.5),
        "offset_delay_p90_frames": _safe_quantile(offset_delays, 0.9),
        "offset_unresolved": offset_unresolved,
        "offset_unresolved_rate": _safe_rate(
            offset_unresolved,
            speech_run_count,
        ),
        "speech_clipping_duration_frames": false_negative,
        "speech_clipping_duration_seconds": (
            false_negative * FRAME_DURATION_SECONDS
        ),
        "speech_clipping_rate": _safe_rate(false_negative, speech_frames),
        "false_activation_duration_frames": false_positive,
        "false_activation_duration_seconds": (
            false_positive * FRAME_DURATION_SECONDS
        ),
        "false_activation_rate": _safe_rate(
            false_positive,
            silence_frames,
        ),
        "prediction_runs": predicted_run_count,
        "single_frame_prediction_runs": single_frame_runs,
        "single_frame_prediction_run_rate": _safe_rate(
            single_frame_runs,
            predicted_run_count,
        ),
        "at_most_two_frame_prediction_runs": at_most_two_frame_runs,
        "at_most_two_frame_prediction_run_rate": _safe_rate(
            at_most_two_frame_runs,
            predicted_run_count,
        ),
        "decision_transitions": transition_count,
        "decision_transitions_per_1000_frames": _safe_rate(
            transition_count * 1000.0,
            total_frames,
        ),
        "prediction_runs_per_1000_frames": _safe_rate(
            predicted_run_count * 1000.0,
            total_frames,
        ),
        "median_prediction_run_frames": _safe_quantile(
            predicted_run_lengths,
            0.5,
        ),
    }


def build_segment_groups(
    utterances: Sequence[Utterance],
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    unseen_noise: Sequence[str] = UNSEEN_NOISE,
) -> dict[str, list[Utterance]]:
    """Build stable overall, acoustic-condition, and seen/unseen groups."""
    unseen = {str(value) for value in unseen_noise}
    groups: dict[str, list[Utterance]] = {
        "overall": list(utterances),
        "clean": [
            utterance
            for utterance in utterances
            if utterance.condition == "clean"
        ],
        "noisy": [
            utterance
            for utterance in utterances
            if utterance.condition != "clean"
        ],
        "seen": [
            utterance
            for utterance in utterances
            if utterance.condition != "clean"
            and utterance.noise_name not in unseen
        ],
        "unseen": [
            utterance
            for utterance in utterances
            if utterance.condition != "clean"
            and utterance.noise_name in unseen
        ],
    }
    for condition in conditions:
        groups[f"condition:{condition}"] = [
            utterance
            for utterance in utterances
            if utterance.condition == str(condition)
        ]
    return {name: value for name, value in groups.items() if value}


def build_segment_rows(
    bundle: dict[str, Any],
    utterances: Sequence[Utterance],
    protocol: SegmentProtocol,
    *,
    conditions: Sequence[str] = CONDITION_ORDER,
    unseen_noise: Sequence[str] = UNSEEN_NOISE,
) -> list[dict[str, Any]]:
    """Evaluate Short, AlwaysRefine, and Adaptive on every segment group."""
    methods = {
        "short": bundle["short_scores"],
        "refine_only": bundle["full_adaptive_scores"],
        "adaptive": bundle["adaptive_scores"],
    }
    groups = build_segment_groups(
        utterances,
        conditions=conditions,
        unseen_noise=unseen_noise,
    )
    rows: list[dict[str, Any]] = []
    for group_name, group_utterances in groups.items():
        for method_name in METHOD_ORDER:
            metrics = evaluate_segment_scores(
                bundle["labels"],
                methods[method_name],
                group_utterances,
                protocol,
            )
            rows.append(
                {
                    "method": method_name,
                    "group": group_name,
                    "utterances": len(group_utterances),
                    "sources": len(
                        {utterance.source_key for utterance in group_utterances}
                    ),
                    **metrics,
                }
            )
    return rows


def _tensor_prefix_check(
    name: str,
    first: torch.Tensor,
    second: torch.Tensor,
    prefix_frames: int,
    *,
    atol: float,
    rtol: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prefix_frames <= 0:
        raise ValueError("prefix_frames must be positive")
    if first.dim() != second.dim():
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"rank mismatch: {first.dim()} vs {second.dim()}"
            ),
            **(details or {}),
        }
    if first.shape[:-1] != second.shape[:-1]:
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"shape mismatch before time axis: "
                f"{tuple(first.shape[:-1])} vs "
                f"{tuple(second.shape[:-1])}"
            ),
            **(details or {}),
        }
    if first.shape[-1] < prefix_frames or second.shape[-1] < prefix_frames:
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"prefix of {prefix_frames} frames exceeds tensor lengths "
                f"{first.shape[-1]} and {second.shape[-1]}"
            ),
            **(details or {}),
        }
    first_prefix = first[..., :prefix_frames]
    second_prefix = second[..., :prefix_frames]
    difference = (first_prefix.float() - second_prefix.float()).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = (
        float(difference.mean().item()) if difference.numel() else 0.0
    )
    passed = bool(
        torch.allclose(
            first_prefix,
            second_prefix,
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    return {
        "name": name,
        "passed": passed,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "atol": float(atol),
        "rtol": float(rtol),
        **(details or {}),
    }


def _bool_prefix_check(
    name: str,
    first: torch.Tensor,
    second: torch.Tensor,
    prefix_frames: int,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prefix_frames <= 0:
        raise ValueError("prefix_frames must be positive")
    if first.dim() != second.dim():
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"rank mismatch: {first.dim()} vs {second.dim()}"
            ),
            **(details or {}),
        }
    if first.shape[:-1] != second.shape[:-1]:
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"shape mismatch before time axis: "
                f"{tuple(first.shape[:-1])} vs "
                f"{tuple(second.shape[:-1])}"
            ),
            **(details or {}),
        }
    if first.shape[-1] < prefix_frames or second.shape[-1] < prefix_frames:
        return {
            "name": name,
            "passed": False,
            "reason": (
                f"prefix of {prefix_frames} frames exceeds tensor lengths "
                f"{first.shape[-1]} and {second.shape[-1]}"
            ),
            **(details or {}),
        }
    mismatch = int(
        (first[..., :prefix_frames] != second[..., :prefix_frames])
        .sum()
        .item()
    )
    return {
        "name": name,
        "passed": mismatch == 0,
        "mismatch_count": mismatch,
        **(details or {}),
    }


def _chunk_lengths(total_frames: int, pattern: Sequence[int]) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not pattern or any(int(value) <= 0 for value in pattern):
        raise ValueError("chunk pattern must contain positive lengths")
    result: list[int] = []
    remaining = int(total_frames)
    index = 0
    while remaining > 0:
        length = min(int(pattern[index % len(pattern)]), remaining)
        result.append(length)
        remaining -= length
        index += 1
    return result


@torch.inference_mode()
def _stream_prefix_outputs(
    model: nn.Module,
    features: torch.Tensor,
    *,
    activation_threshold: float,
    pattern: Sequence[int],
) -> dict[str, torch.Tensor | int]:
    stream = AdaptiveStreamingVAD(
        model,
        activation_threshold=float(activation_threshold),
    )
    logits: list[torch.Tensor] = []
    short_logits: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    residual: list[torch.Tensor] = []
    start = 0
    for length in _chunk_lengths(int(features.shape[-1]), pattern):
        output = stream.forward_components(
            features[..., start : start + length]
        )
        logits.append(output.logits)
        short_logits.append(output.short_logits)
        selected.append(output.selected)
        residual.append(output.residual)
        start += length
    return {
        "logits": torch.cat(logits, dim=-1),
        "short_logits": torch.cat(short_logits, dim=-1),
        "selected": torch.cat(selected, dim=-1),
        "residual": torch.cat(residual, dim=-1),
        "cache_bytes": int(stream.cache_bytes),
    }


def _speech_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3 or logits.shape[1] != 2:
        raise ValueError("expected binary logits [B, 2, T]")
    return torch.softmax(logits.transpose(1, 2), dim=-1)[..., 1]


def _module_names(module: nn.Module, types: tuple[type[nn.Module], ...]) -> list[str]:
    return [
        name
        for name, child in module.named_modules()
        if isinstance(child, types)
    ]


def audit_causality(
    model: nn.Module,
    frontend: MfccFrontend,
    *,
    prefix_samples: int,
    total_samples: int,
    activation_threshold: float,
    seed: int = 20260921,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare identical prefixes followed by different futures."""
    if prefix_samples <= 0 or total_samples <= prefix_samples:
        raise ValueError(
            "total_samples must be greater than positive prefix_samples"
        )
    if not 0.0 <= activation_threshold <= 0.5:
        raise ValueError("activation_threshold must be in [0, 0.5]")
    model.eval()
    frontend.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    prefix = 0.05 * torch.randn(
        (1, prefix_samples),
        generator=generator,
        dtype=torch.float32,
    )
    suffix_length = int(total_samples) - int(prefix_samples)
    suffix_one = 0.05 * torch.randn(
        (1, suffix_length),
        generator=generator,
        dtype=torch.float32,
    )
    suffix_two = -0.08 * torch.randn(
        (1, suffix_length),
        generator=generator,
        dtype=torch.float32,
    )
    waveforms = [
        torch.cat((prefix, suffix_one), dim=-1),
        torch.cat((prefix, suffix_two), dim=-1),
    ]

    device = next(model.parameters()).device
    waveforms = [waveform.to(device) for waveform in waveforms]
    features = [frontend(waveform) for waveform in waveforms]
    prefix_frames = int(frontend.config.frames_for_samples(prefix_samples))
    if prefix_frames <= 0:
        raise RuntimeError("prefix produced no MFCC frames")
    if any(feature.shape[-1] < prefix_frames for feature in features):
        raise RuntimeError("frontend produced fewer frames than required")

    checks: list[dict[str, Any]] = []
    checks.append(
        _tensor_prefix_check(
            "fbank_window_prefix_invariance",
            features[0],
            features[1],
            prefix_frames,
            atol=atol,
            rtol=rtol,
            details={"prefix_frames": prefix_frames},
        )
    )
    prefix_only_features = frontend(waveforms[0][..., :prefix_samples])
    checks.append(
        _tensor_prefix_check(
            "padding_prefix_invariance",
            features[0],
            prefix_only_features,
            prefix_frames,
            atol=atol,
            rtol=rtol,
            details={
                "causal_frontend": bool(frontend.config.causal),
                "center": not bool(frontend.config.causal),
            },
        )
    )

    outputs = [
        model.forward_components(
            feature,
            activation_threshold=float(activation_threshold),
        )
        for feature in features
    ]
    checks.append(
        _tensor_prefix_check(
            "speech_probability_prefix_invariance",
            _speech_probabilities(outputs[0].logits),
            _speech_probabilities(outputs[1].logits),
            prefix_frames,
            atol=atol,
            rtol=rtol,
        )
    )
    checks.append(
        _tensor_prefix_check(
            "refinement_cache_prefix_invariance",
            outputs[0].residual,
            outputs[1].residual,
            prefix_frames,
            atol=atol,
            rtol=rtol,
        )
    )
    checks.append(
        _bool_prefix_check(
            "gate_prefix_invariance",
            outputs[0].selected,
            outputs[1].selected,
            prefix_frames,
        )
    )

    batch_features = torch.cat(features, dim=0)
    batch_pattern = (5, 13, 2, 29, 1)
    batch_stream = _stream_prefix_outputs(
        model,
        batch_features,
        activation_threshold=activation_threshold,
        pattern=batch_pattern,
    )
    checks.append(
        _tensor_prefix_check(
            "batching_prefix_isolation",
            batch_stream["logits"],
            torch.cat(
                (outputs[0].logits, outputs[1].logits),
                dim=0,
            ),
            prefix_frames,
            atol=atol,
            rtol=rtol,
            details={"batch_size": 2, "chunk_pattern": list(batch_pattern)},
        )
    )

    chunk_patterns = (
        (1, 7, 31, 64, 3, 127),
        (5, 13, 2, 89, 17),
    )
    for index, pattern in enumerate(chunk_patterns, start=1):
        streamed = _stream_prefix_outputs(
            model,
            features[0],
            activation_threshold=activation_threshold,
            pattern=pattern,
        )
        checks.append(
            _tensor_prefix_check(
                f"chunk_boundary_full_equivalence_{index}",
                streamed["logits"],
                outputs[0].logits,
                prefix_frames,
                atol=atol,
                rtol=rtol,
                details={"chunk_pattern": list(pattern)},
            )
        )
        checks.append(
            _bool_prefix_check(
                f"chunk_boundary_gate_equivalence_{index}",
                streamed["selected"],
                outputs[0].selected,
                prefix_frames,
                details={"chunk_pattern": list(pattern)},
            )
        )
        checks.append(
            _tensor_prefix_check(
                f"chunk_boundary_refinement_cache_equivalence_{index}",
                streamed["residual"],
                outputs[0].residual,
                prefix_frames,
                atol=atol,
                rtol=rtol,
                details={"chunk_pattern": list(pattern)},
            )
        )

    batch_norm_names = _module_names(
        model,
        (nn.BatchNorm1d, nn.SyncBatchNorm),
    )
    normalization_training = [
        name
        for name, child in model.named_modules()
        if isinstance(child, (nn.BatchNorm1d, nn.SyncBatchNorm))
        and child.training
    ]
    checks.append(
        {
            "name": "normalization_eval_mode",
            "passed": not model.training and not normalization_training,
            "model_training": bool(model.training),
            "batch_norm_modules": batch_norm_names,
            "training_batch_norm_modules": normalization_training,
        }
    )

    recurrent_names = _module_names(
        model,
        (nn.RNN, nn.LSTM, nn.GRU, nn.MultiheadAttention),
    )
    checks.append(
        {
            "name": "hidden_state_recurrent_modules",
            "passed": not recurrent_names,
            "recurrent_modules": recurrent_names,
        }
    )
    checks.append(
        {
            "name": "causal_frontend_configuration",
            "passed": bool(frontend.config.causal),
            "causal": bool(frontend.config.causal),
            "center": not bool(frontend.config.causal),
        }
    )

    return {
        "status": "PASS" if all(bool(check["passed"]) for check in checks) else "FAIL",
        "device": str(device),
        "prefix_samples": int(prefix_samples),
        "total_samples": int(total_samples),
        "prefix_frames": prefix_frames,
        "activation_threshold": float(activation_threshold),
        "seed": int(seed),
        "atol": float(atol),
        "rtol": float(rtol),
        "checks": checks,
        "conclusion": (
            "No future acoustic frames are consumed by the tested prefix "
            "when this audit passes."
        ),
    }


def _rows_by_method_group(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["group"]))
        if key in result:
            raise ValueError(f"duplicate segment row: {key}")
        result[key] = row
    return result


def _metric_value(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    return float(value)


def assess_a13(
    rows: Sequence[dict[str, Any]],
    causality: dict[str, Any],
    *,
    a12_status: str | None = None,
    short_speech_recall_tolerance: float = 0.02,
    whole_miss_absolute_tolerance: int = 2,
    clipping_rate_tolerance: float = 0.01,
    false_activation_rate_tolerance: float = 0.005,
    onset_delay_tolerance_frames: float = 1.0,
    offset_delay_tolerance_frames: float = 1.0,
    flicker_run_rate_tolerance: float = 0.02,
    transition_rate_tolerance: float = 2.0,
) -> dict[str, Any]:
    """Apply predeclared A13 non-regression checks against Short."""
    indexed = _rows_by_method_group(rows)
    checks: list[dict[str, Any]] = []
    missing_groups = [
        group
        for group in CRITICAL_GROUPS
        if ("short", group) not in indexed
        or ("adaptive", group) not in indexed
    ]
    checks.append(
        {
            "name": "critical_groups_present",
            "passed": not missing_groups,
            "missing_groups": missing_groups,
            "required_groups": list(CRITICAL_GROUPS),
        }
    )

    for group in CRITICAL_GROUPS:
        short = indexed.get(("short", group))
        adaptive = indexed.get(("adaptive", group))
        if short is None or adaptive is None:
            continue

        short_recall = _metric_value(short, "short_speech_recall")
        adaptive_recall = _metric_value(adaptive, "short_speech_recall")
        if short_recall is not None and adaptive_recall is not None:
            checks.append(
                {
                    "name": "short_speech_recall",
                    "group": group,
                    "passed": (
                        adaptive_recall
                        >= short_recall - short_speech_recall_tolerance
                    ),
                    "short": short_recall,
                    "adaptive": adaptive_recall,
                    "allowed_drop": short_speech_recall_tolerance,
                }
            )

        short_miss = int(short["whole_speech_run_misses"])
        adaptive_miss = int(adaptive["whole_speech_run_misses"])
        allowed_miss = short_miss + max(
            int(whole_miss_absolute_tolerance),
            int(math.ceil(0.01 * max(short["speech_runs"], 1))),
        )
        checks.append(
            {
                "name": "whole_speech_run_misses",
                "group": group,
                "passed": adaptive_miss <= allowed_miss,
                "short": short_miss,
                "adaptive": adaptive_miss,
                "allowed": allowed_miss,
            }
        )

        for metric_name, tolerance in (
            ("speech_clipping_rate", clipping_rate_tolerance),
            ("false_activation_rate", false_activation_rate_tolerance),
        ):
            short_value = _metric_value(short, metric_name)
            adaptive_value = _metric_value(adaptive, metric_name)
            if short_value is None or adaptive_value is None:
                continue
            checks.append(
                {
                    "name": metric_name,
                    "group": group,
                    "passed": adaptive_value <= short_value + tolerance,
                    "short": short_value,
                    "adaptive": adaptive_value,
                    "allowed_increase": tolerance,
                }
            )

        for metric_name, tolerance in (
            ("onset_delay_median_frames", onset_delay_tolerance_frames),
            ("offset_delay_median_frames", offset_delay_tolerance_frames),
        ):
            short_value = _metric_value(short, metric_name)
            adaptive_value = _metric_value(adaptive, metric_name)
            if short_value is None or adaptive_value is None:
                continue
            checks.append(
                {
                    "name": metric_name,
                    "group": group,
                    "passed": adaptive_value <= short_value + tolerance,
                    "short": short_value,
                    "adaptive": adaptive_value,
                    "allowed_increase": tolerance,
                }
            )

        for metric_name, tolerance in (
            (
                "at_most_two_frame_prediction_run_rate",
                flicker_run_rate_tolerance,
            ),
            (
                "decision_transitions_per_1000_frames",
                transition_rate_tolerance,
            ),
        ):
            short_value = _metric_value(short, metric_name)
            adaptive_value = _metric_value(adaptive, metric_name)
            if short_value is None or adaptive_value is None:
                continue
            checks.append(
                {
                    "name": metric_name,
                    "group": group,
                    "passed": adaptive_value <= short_value + tolerance,
                    "short": short_value,
                    "adaptive": adaptive_value,
                    "allowed_increase": tolerance,
                }
            )

    causality_passed = causality.get("status") == "PASS"
    failing = [check for check in checks if not bool(check["passed"])]
    if not causality_passed:
        status = "FAIL_CAUSALITY"
    elif failing:
        status = "FAIL_SEGMENT_REGRESSION"
    else:
        status = "PASS"
    a12_status_text = a12_status if a12_status is not None else "unavailable"
    return {
        "status": status,
        "causality_passed": causality_passed,
        "checks": checks,
        "failure_count": len(failing),
        "failures": failing,
        "note": (
            "A13 is a streaming/segment diagnostic. It does not override "
            f"the loaded A12 condition-level routing status ({a12_status_text})."
        ),
    }


def load_a12_status(path: str | Path) -> dict[str, Any]:
    """Read the current A12 status for the downstream report."""
    result_path = Path(path)
    if not result_path.exists():
        return {"path": str(result_path), "status": None}
    with open(result_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    decision = payload.get("decision") or payload.get("assessment") or {}
    return {
        "path": str(result_path),
        "status": decision.get("status"),
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.3f}%"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_markdown(
    rows: Sequence[dict[str, Any]],
    causality: dict[str, Any],
    assessment: dict[str, Any],
    protocol: SegmentProtocol,
    a12: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# A13 Streaming and Segment Audit",
        "",
        "## Protocol",
        "",
        f"- Decision threshold: `{protocol.decision_threshold:.2f}`.",
        f"- Short speech: at most `{protocol.short_speech_max_frames}` frames "
        f"(`{protocol.short_speech_max_frames * FRAME_DURATION_SECONDS:.2f} s`).",
        f"- Onset tolerance: `{protocol.onset_tolerance_frames}` frames.",
        "- Offset delay is measured to the first non-speech frame after a "
        "ground-truth speech run; unresolved runs are reported separately.",
        "- Adaptive is compared with Short for non-regression. AlwaysRefine "
        "is reported as a diagnostic ceiling.",
        "",
            "## Causality Audit",
            "",
            f"- Device: `{causality['device']}`. CPU avoids backend-dependent "
            "convolution kernels when comparing chunked and full-sequence "
            "execution.",
            "",
            "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in causality["checks"]:
        detail_parts: list[str] = []
        for name in (
            "max_abs_diff",
            "mismatch_count",
            "batch_size",
            "prefix_frames",
            "recurrent_modules",
        ):
            if name in check:
                detail_parts.append(f"{name}={check[name]}")
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | "
            f"{', '.join(detail_parts) or '-'} |"
        )
    lines.extend(
        [
            "",
            f"Causality status: **{causality['status']}**. "
            f"{causality['conclusion']}",
            "",
            "## Overall Segment Metrics",
            "",
            "| Method | Short recall | Whole-run miss | Onset median | "
            "Offset median | Clipping | False activation | Transitions/1k | "
            "Runs <=2 frames |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHOD_ORDER:
        row = next(
            (
                value
                for value in rows
                if value["method"] == method and value["group"] == "overall"
            ),
            None,
        )
        if row is None:
            continue
        lines.append(
            f"| {method} | {_format_percent(row['short_speech_recall'])} | "
            f"{int(row['whole_speech_run_misses'])} | "
            f"{_format_float(row['onset_delay_median_frames'], 2)} | "
            f"{_format_float(row['offset_delay_median_frames'], 2)} | "
            f"{float(row['speech_clipping_duration_seconds']):.3f}s | "
            f"{float(row['false_activation_duration_seconds']):.3f}s | "
            f"{_format_float(row['decision_transitions_per_1000_frames'], 3)} | "
            f"{_format_percent(row['at_most_two_frame_prediction_run_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Adaptive by Group",
            "",
            "| Group | Short recall | Whole-run miss | Onset median | "
            "Offset median | Clipping | False activation | Transitions/1k | "
            "Runs <=2 frames |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["method"] != "adaptive":
            continue
        lines.append(
            f"| {row['group']} | "
            f"{_format_percent(row['short_speech_recall'])} | "
            f"{int(row['whole_speech_run_misses'])} | "
            f"{_format_float(row['onset_delay_median_frames'], 2)} | "
            f"{_format_float(row['offset_delay_median_frames'], 2)} | "
            f"{float(row['speech_clipping_duration_seconds']):.3f}s | "
            f"{float(row['false_activation_duration_seconds']):.3f}s | "
            f"{_format_float(row['decision_transitions_per_1000_frames'], 3)} | "
            f"{_format_percent(row['at_most_two_frame_prediction_run_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"A13 status: **{assessment['status']}**.",
            "",
            f"- Failed non-regression checks: `{assessment['failure_count']}`.",
            f"- A12 status remains: `{a12.get('status')}`.",
            f"- {assessment['note']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        name
        for name in ("overall", "noisy", "seen", "unseen")
        if any(row["group"] == name for row in rows)
    ]
    if not groups:
        return
    method_colors = {
        "short": "tab:blue",
        "refine_only": "tab:orange",
        "adaptive": "tab:green",
    }
    _, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = (
        ("short_speech_recall", "Short-speech recall", True),
        ("onset_delay_median_frames", "Onset median (frames)", False),
        (
            "at_most_two_frame_prediction_run_rate",
            "Prediction runs <= 2 frames",
            True,
        ),
    )
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.24
    for axis, (metric, title, percentage) in zip(axes, metrics):
        for index, method in enumerate(METHOD_ORDER):
            values = []
            for group in groups:
                row = next(
                    (
                        value
                        for value in rows
                        if value["method"] == method and value["group"] == group
                    ),
                    None,
                )
                value = 0.0 if row is None or row.get(metric) is None else row[metric]
                values.append(float(value) * (100.0 if percentage else 1.0))
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=method,
                color=method_colors[method],
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(groups)
        if percentage:
            axis.set_ylabel("Percent")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run(args: argparse.Namespace) -> int:
    bundle = load_a13_bundle(args.bundle)
    utterances = build_utterances(bundle)
    protocol = SegmentProtocol(
        decision_threshold=float(args.decision_threshold),
        short_speech_max_frames=int(args.short_speech_max_frames),
        onset_tolerance_frames=int(args.onset_tolerance_frames),
    )
    rows = build_segment_rows(
        bundle,
        utterances,
        protocol,
        conditions=CONDITION_ORDER,
        unseen_noise=UNSEEN_NOISE,
    )

    device = resolve_device(args.device)
    model, _, _ = load_adaptive_model(args.adaptive_checkpoint, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device).eval()
    gate_threshold = (
        float(bundle["threshold"])
        if args.gate_threshold is None
        else float(args.gate_threshold)
    )
    if not 0.0 <= gate_threshold <= 0.5:
        raise ValueError("gate threshold must be in [0, 0.5]")
    causality = audit_causality(
        model,
        frontend,
        prefix_samples=int(
            round(float(args.causality_prefix_seconds) * 16_000)
        ),
        total_samples=int(round(float(args.causality_total_seconds) * 16_000)),
        activation_threshold=gate_threshold,
        seed=int(args.causality_seed),
    )
    a12 = load_a12_status(args.a12_results)
    assessment = assess_a13(
        rows,
        causality,
        a12_status=a12.get("status"),
    )
    protocol_payload = {
        **protocol.to_dict(),
        "frame_duration_seconds": FRAME_DURATION_SECONDS,
        "gate_threshold": gate_threshold,
        "test_frames": sum(row["frames"] for row in rows if row["method"] == "short"
                           and row["group"] == "overall"),
        "test_utterances": len(utterances),
        "methods": list(METHOD_ORDER),
        "critical_groups": list(CRITICAL_GROUPS),
        "unseen_noise": list(UNSEEN_NOISE),
    }
    summary = {
        "protocol": protocol_payload,
        "bundle": bundle["path"],
        "adaptive_checkpoint": str(args.adaptive_checkpoint),
        "causality": causality,
        "assessment": assessment,
        "a12": a12,
        "segment_rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "a13_streaming_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(rows, args.output_dir / "a13_segment_metrics.csv")
    _write_markdown(
        rows,
        causality,
        assessment,
        protocol,
        a12,
        args.output_dir / "a13_streaming_audit.md",
    )
    _write_plot(rows, args.output_dir / "a13_segment_metrics.png")
    print(
        f"A13 artifacts written to {args.output_dir}; "
        f"status={assessment['status']}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A13 streaming and segment-level audit."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--adaptive-checkpoint",
        type=Path,
        default=DEFAULT_ADAPTIVE_CHECKPOINT,
    )
    parser.add_argument(
        "--a12-results",
        type=Path,
        default=DEFAULT_A12_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--short-speech-max-frames", type=int, default=20)
    parser.add_argument("--onset-tolerance-frames", type=int, default=20)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--causality-prefix-seconds", type=float, default=1.0)
    parser.add_argument("--causality-total-seconds", type=float, default=2.0)
    parser.add_argument("--causality-seed", type=int, default=20260921)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device for the causality audit. CPU is the default because "
            "it avoids backend-dependent convolution kernels when comparing "
            "chunked and full-sequence execution."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
