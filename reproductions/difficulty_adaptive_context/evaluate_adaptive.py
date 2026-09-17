# -*- coding: utf-8 -*-
"""Evaluate the Phase A3 shared-encoder adaptive refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveCausalVAD,
    AdaptiveStreamingVAD,
    RefinementConfig,
    SparseCausalMultiScaleRefinement,
)
from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    choose_confidence_threshold,
    count_frames_by_cluster,
    deterministic_speaker_split,
    metrics_from_counts,
    speaker_cluster_bootstrap,
    speaker_from_source_key,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_context import (
    CONDITION_ORDER,
    binary_metrics,
    load_frame_model,
)
from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import (
    build_marblenet_3x2x64,
    receptive_field,
)
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
DEFAULT_SPEAKER_SPLIT_SEED = 20260917


def load_adaptive_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AdaptiveCausalVAD, dict[str, Any], RefinementConfig]:
    """Load a frozen Short encoder and its trained sparse refinement."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    short_config = dict(
        payload.get("short_model_config", payload.get("model_config", {}))
    )
    if not bool(short_config.get("causal", False)):
        raise ValueError("adaptive checkpoint must use a causal Short encoder")
    if not bool(short_config.get("frame_output", False)):
        raise ValueError("adaptive checkpoint must use frame-level outputs")
    short_model = build_marblenet_3x2x64(
        feat_in=int(short_config.get("feat_in", 64)),
        num_classes=int(short_config.get("num_classes", 2)),
        dropout=float(short_config.get("dropout", 0.0)),
        causal=True,
        dilation_profile=short_config.get("dilation_profile", "short"),
        frame_output=True,
    )
    refinement_config = RefinementConfig.from_dict(
        dict(payload["refinement_config"])
    )
    refinement = SparseCausalMultiScaleRefinement(refinement_config)
    model = AdaptiveCausalVAD(short_model, refinement)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    model.freeze_short_model()
    return model, payload, refinement_config


@torch.inference_mode()
def predict_full_adaptive_frames(
    model: AdaptiveCausalVAD,
    frontend: torch.nn.Module,
    waveform: np.ndarray,
    *,
    device: torch.device,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ungated refinement probabilities and embedded Short probabilities.

    The streaming wrapper is run with a permissive threshold so every frame
    exercises the refinement path.  Budget-specific gates are applied later
    from the observed Short probabilities, which keeps this expensive pass
    independent of the calibration activation target.
    """
    tensor = torch.from_numpy(
        waveform.astype(np.float32) * INT16_SCALE
    ).unsqueeze(0)
    features = frontend(tensor.to(device, non_blocking=True))
    stream = AdaptiveStreamingVAD(
        model,
        activation_threshold=0.5,
    )
    score_parts: list[np.ndarray] = []
    short_parts: list[np.ndarray] = []
    selected_parts: list[np.ndarray] = []
    for start in range(0, features.shape[-1], chunk_frames):
        output = stream.forward_components(
            features[..., start : start + chunk_frames]
        )
        probabilities = torch.softmax(
            output.logits.transpose(1, 2),
            dim=-1,
        )[..., 1]
        score_parts.append(
            probabilities.detach().cpu().numpy().reshape(-1)
        )
        short_probabilities = torch.softmax(
            output.short_logits.transpose(1, 2),
            dim=-1,
        )[..., 1]
        short_parts.append(
            short_probabilities.detach().cpu().numpy().reshape(-1)
        )
        selected_parts.append(
            output.selected.detach().cpu().numpy().reshape(-1)
        )
    if not score_parts:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    return (
        np.concatenate(score_parts).astype(np.float64),
        np.concatenate(short_parts).astype(np.float64),
        np.concatenate(selected_parts).astype(bool),
    )


def gated_scores(
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Use refinement only where the confidence gate is active."""
    short_scores = np.asarray(short_scores, dtype=np.float64)
    refined_scores = np.asarray(refined_scores, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    if short_scores.shape != refined_scores.shape:
        raise ValueError("short and refined score arrays must have equal shape")
    if selected.shape != short_scores.shape:
        raise ValueError("selected mask must match score shape")
    return np.where(selected, refined_scores, short_scores)


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"reference predictions not found: {path}")
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
    lengths = {name: array.size for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"reference arrays have inconsistent lengths: {lengths}"
        )
    return arrays


def _delta_metrics(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    adaptive_scores: np.ndarray,
) -> dict[str, Any]:
    short = binary_metrics(labels, short_scores)
    long = binary_metrics(labels, long_scores)
    adaptive = binary_metrics(labels, adaptive_scores)
    return {
        "frames": int(np.asarray(labels).size),
        "short": short,
        "long": long,
        "adaptive": adaptive,
        "delta_f1_long_minus_short": (
            None
            if short["f1"] is None or long["f1"] is None
            else float(long["f1"] - short["f1"])
        ),
        "delta_f1_adaptive_minus_short": (
            None
            if short["f1"] is None or adaptive["f1"] is None
            else float(adaptive["f1"] - short["f1"])
        ),
        "delta_f1_adaptive_minus_long": (
            None
            if long["f1"] is None or adaptive["f1"] is None
            else float(adaptive["f1"] - long["f1"])
        ),
        "error_reduction_long_vs_short": (
            None
            if short["error"] is None or long["error"] is None
            else float(short["error"] - long["error"])
        ),
        "error_reduction_adaptive_vs_short": (
            None
            if short["error"] is None or adaptive["error"] is None
            else float(short["error"] - adaptive["error"])
        ),
        "error_reduction_adaptive_vs_long": (
            None
            if long["error"] is None or adaptive["error"] is None
            else float(long["error"] - adaptive["error"])
        ),
    }


def _condition_seen_metrics(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    condition: np.ndarray,
    noise_name: np.ndarray,
    unseen_noise: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in CONDITION_ORDER:
        mask = condition == name
        if np.any(mask):
            result[name] = _delta_metrics(
                labels[mask],
                short_scores[mask],
                long_scores[mask],
                adaptive_scores[mask],
            )
    noisy = condition != "clean"
    unseen = noisy & np.isin(noise_name, sorted(unseen_noise))
    seen = noisy & ~unseen
    for name, mask in (("seen", seen), ("unseen", unseen)):
        if np.any(mask):
            result[name] = _delta_metrics(
                labels[mask],
                short_scores[mask],
                long_scores[mask],
                adaptive_scores[mask],
            )
    return result


def _utility_for_mask(
    labels: np.ndarray,
    short_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    selected: np.ndarray,
    speaker_ids: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return count_frames_by_cluster(
        labels,
        short_scores,
        adaptive_scores,
        selected,
        speaker_ids,
        mask=mask,
    )


def _md_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _md_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _report(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    threshold = summary["fixed_gate"]["threshold"]
    primary = summary["fixed_gate"]["test"]
    by_condition = summary["by_condition"]
    if protocol.get("primary_calibration_activation") is None:
        threshold_source = (
            "- Threshold source: fixed before final-test evaluation"
        )
        threshold_note = (
            "The threshold was fixed before final-test evaluation. The "
            "final-test activation rate is observed and is not used to "
            "choose the threshold."
        )
    else:
        threshold_source = (
            "- Calibration activation target: "
            f"{100.0 * protocol['primary_calibration_activation']:.2f}%"
        )
        threshold_note = (
            "The threshold is chosen only on calibration-speaker Short "
            "scores. The final-test activation rate is observed, not forced "
            "to the calibration budget."
        )
    lines = [
        "# Phase A3: Sparse Shared-Encoder Refinement",
        "",
        "## Fixed Protocol",
        "",
        f"- Adaptive checkpoint: `{protocol['adaptive_checkpoint']}`",
        f"- Reference predictions: `{protocol['reference_predictions']}`",
        f"- Calibration speakers: {protocol['calibration_speaker_count']}",
        f"- Final-test speakers: {protocol['test_speaker_count']}",
        f"- Speaker split seed: {protocol['speaker_split_seed']}",
        threshold_source,
        f"- Frozen threshold: {threshold:.6f}",
        f"- Test activation: {primary['activation_rate'] * 100.0:.3f}%",
        f"- Refinement lookback: {protocol['lookback_frames']} frames "
        f"({protocol['lookback_seconds']:.2f} s)",
        f"- Estimated refinement MACs per selected frame: "
        f"{protocol['refinement_macs_per_selected_frame']:,}",
        "",
        threshold_note,
        "",
        "## Final-Test Model Comparison",
        "",
        "| Model | F1 | Error | AUROC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("short", "long", "adaptive"):
        row = primary[name]
        lines.append(
            f"| {name} | {_md_number(row['f1'])} | "
            f"{_md_percent(row['error'])} | {_md_number(row['auc'])} |"
        )
    lines.extend(
        [
            "",
            f"- Adaptive minus Short F1: "
            f"{_md_number(primary['delta_f1_adaptive_minus_short'])}.",
            f"- Adaptive minus Long F1: "
            f"{_md_number(primary['delta_f1_adaptive_minus_long'])}.",
            "",
            "## Final-Test Gate Utility",
            "",
            "| Selected | Correction | Harm | Net / selected | Net / frame |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {primary['utility']['selected']} | "
            f"{primary['utility']['correction']} | "
            f"{primary['utility']['harm']} | "
            f"{_md_percent(primary['utility']['net_utility_per_selected'])} | "
            f"{_md_percent(primary['utility']['net_utility_per_frame'])} |",
            "",
            f"- Speaker-cluster 95% CI for net / selected: "
            f"{_md_percent(primary['bootstrap']['net_utility_per_selected_ci95_low'])} "
            f"to "
            f"{_md_percent(primary['bootstrap']['net_utility_per_selected_ci95_high'])}.",
            "",
            "## By Condition",
            "",
            "| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | "
            "Adaptive-Short | Selected | Net / selected |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in (*CONDITION_ORDER, "seen", "unseen"):
        row = by_condition.get(name)
        if row is None:
            continue
        utility = row.get("utility") or {}
        lines.append(
            f"| {name} | {row['frames']} | "
            f"{_md_number(row['short']['f1'])} | "
            f"{_md_number(row['long']['f1'])} | "
            f"{_md_number(row['adaptive']['f1'])} | "
            f"{_md_number(row['delta_f1_adaptive_minus_short'])} | "
            f"{utility.get('selected', 'n/a')} | "
            f"{_md_percent(utility.get('net_utility_per_selected'))} |"
        )
    for name in ("seen", "unseen"):
        row = by_condition.get(name)
        bootstrap = None if row is None else row.get("utility_bootstrap")
        if not bootstrap:
            continue
        lines.append(
            f"- {name.title()}-noise net / selected 95% CI: "
            f"{_md_percent(bootstrap.get('net_utility_per_selected_ci95_low'))} "
            f"to "
            f"{_md_percent(bootstrap.get('net_utility_per_selected_ci95_high'))}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Adaptive does not need to beat Long-only in absolute F1.",
            "- Accuracy is informative only together with activation rate, "
            "CPU latency, cache and MAC accounting.",
            "- The gate is fixed before final-test evaluation; test top-k "
            "selection is not used.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase A3 sparse adaptive refinement."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adaptive-checkpoint", type=Path, required=True)
    parser.add_argument("--long-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=DEFAULT_LIBRISPEECH_ROOT,
    )
    parser.add_argument("--row-sample", type=int, default=1_080)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-chunk-frames", type=int, default=2_000)
    parser.add_argument("--no-clean", action="store_true")
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
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use an already-fixed low-confidence threshold instead of "
        "selecting one from calibration frames.",
    )
    parser.add_argument(
        "--activation-budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=2_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument(
        "--unseen-noise",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.row_sample is not None and args.row_sample <= 0:
        parser.error("--row-sample must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.score_chunk_frames <= 0:
        parser.error("--score-chunk-frames must be positive")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be in (0, 1)")
    if not 0.0 < args.primary_calibration_activation <= 1.0:
        parser.error(
            "--primary-calibration-activation must be in (0, 1]"
        )
    if args.fixed_threshold is not None and not (
        0.0 <= args.fixed_threshold <= 0.5
    ):
        parser.error("--fixed-threshold must be in [0, 0.5]")
    if any(
        not 0.0 < float(value) <= 1.0
        for value in args.activation_budgets
    ):
        parser.error("--activation-budgets values must be in (0, 1]")
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    device = resolve_device(args.device)

    model, adaptive_payload, refinement_config = load_adaptive_model(
        args.adaptive_checkpoint,
        device,
    )
    _long_model, long_payload, long_config = load_frame_model(
        args.long_checkpoint,
        device,
    )
    long_rf = receptive_field(
        long_config.get("dilation_profile", "baseline")
    )
    valid_start = int(long_rf["lookback_frames"])
    reference = _load_reference(args.reference_predictions)
    reference_speaker_ids = np.asarray(
        [
            speaker_from_source_key(value)
            for value in reference["source_key"]
        ],
        dtype=str,
    )
    (
        calibration_mask,
        test_mask,
        calibration_speakers,
        test_speakers,
    ) = deterministic_speaker_split(
        reference_speaker_ids,
        calibration_fraction=args.calibration_fraction,
        seed=args.speaker_split_seed,
    )
    if args.fixed_threshold is None:
        threshold, _ = choose_confidence_threshold(
            reference["short_scores"][calibration_mask],
            activation_rate=args.primary_calibration_activation,
        )
        threshold_mode = "calibration-only"
    else:
        threshold = float(args.fixed_threshold)
        threshold_mode = "fixed"

    items = build_evaluation_items(
        args.manifest,
        generated_root=args.data_root / "generated",
        label_root=args.data_root / "labels",
        librispeech_root=args.librispeech_root,
        row_sample=args.row_sample,
        seed=args.seed,
        include_clean=not args.no_clean,
        limit=args.limit,
    )
    if not items:
        raise RuntimeError("no evaluation items were selected")

    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    all_refined: list[np.ndarray] = []
    all_stream_selected: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_source: list[np.ndarray] = []
    all_noise: list[np.ndarray] = []
    all_condition: list[np.ndarray] = []
    embedded_short_parts: list[np.ndarray] = []

    print(
        f"device={device}, items={len(items)}, valid_start={valid_start}, "
        f"lookback={refinement_config.lookback_frames} frames, "
        f"threshold_mode={threshold_mode}",
        flush=True,
    )
    for index, item in enumerate(items):
        waveform = read_int16_audio(item.audio_path)
        sample_labels = np.load(item.label_path)
        if sample_labels.size != waveform.size:
            raise ValueError(
                f"label/audio mismatch for {item.audio_path}: "
                f"{sample_labels.size} != {waveform.size}"
            )
        n_frames = waveform.size // FRAME_HOP + 1
        frame_labels = causal_frame_labels(sample_labels, n_frames)
        if n_frames <= valid_start:
            continue

        refined_scores, embedded_short, stream_selected = (
            predict_full_adaptive_frames(
                model,
                frontend,
                waveform,
                device=device,
                chunk_frames=args.score_chunk_frames,
            )
        )
        if refined_scores.size != n_frames or embedded_short.size != n_frames:
            raise RuntimeError("adaptive output frame count is inconsistent")

        labels = frame_labels[valid_start:]
        all_labels.append(labels)
        all_refined.append(refined_scores[valid_start:])
        all_stream_selected.append(stream_selected[valid_start:])
        embedded_short_parts.append(embedded_short[valid_start:])
        all_source.append(
            np.full(
                labels.size,
                item.source_key,
                dtype=f"U{max(1, len(item.source_key))}",
            )
        )
        all_noise.append(
            np.full(
                labels.size,
                item.noise_name,
                dtype=f"U{max(1, len(item.noise_name))}",
            )
        )
        condition = "clean" if item.is_clean else str(item.snr_db)
        all_condition.append(
            np.full(
                labels.size,
                condition,
                dtype=f"U{max(1, len(condition))}",
            )
        )
        if args.progress_every > 0 and (
            (index + 1) % args.progress_every == 0 or index + 1 == len(items)
        ):
            print(f"  evaluated {index + 1}/{len(items)}", flush=True)

    if not all_labels:
        raise RuntimeError("all selected utterances were too short")

    labels = np.concatenate(all_labels)
    full_adaptive_scores = np.concatenate(all_refined)
    embedded_short_scores = np.concatenate(embedded_short_parts)
    stream_selected = np.concatenate(all_stream_selected)
    source_key = np.concatenate(all_source)
    noise_name = np.concatenate(all_noise)
    condition = np.concatenate(all_condition)

    reference_length = int(reference["labels"].size)
    if labels.size != reference_length:
        raise RuntimeError(
            "evaluation frame count does not match reference predictions: "
            f"{labels.size} != {reference_length}"
        )
    if not np.array_equal(labels, reference["labels"]):
        raise RuntimeError(
            "frame labels do not match reference predictions; use the same "
            "manifest, row sample, seed and no-clean setting"
        )
    if not np.array_equal(source_key, reference["source_key"]):
        raise RuntimeError(
            "source keys do not match reference predictions; use the same "
            "manifest, row sample, seed and no-clean setting"
        )
    if not np.array_equal(noise_name, reference["noise_name"]):
        raise RuntimeError(
            "noise names do not match reference predictions; use the same "
            "manifest, row sample, seed and no-clean setting"
        )
    if not np.array_equal(condition, reference["condition"]):
        raise RuntimeError(
            "SNR conditions do not match reference predictions; use the same "
            "manifest, row sample, seed and no-clean setting"
        )
    reference_short_scores = reference["short_scores"].astype(np.float64)
    if not np.allclose(
        embedded_short_scores,
        reference_short_scores,
        rtol=1e-3,
        atol=1e-3,
    ):
        max_error = float(
            np.max(
                np.abs(
                    embedded_short_scores
                    - reference_short_scores
                )
            )
        )
        raise RuntimeError(
            "adaptive checkpoint's embedded Short model does not match the "
            f"reference Short predictions (max abs error {max_error:.6g})"
        )

    embedded_short_error = np.abs(
        embedded_short_scores - reference_short_scores
    )
    embedded_short_max_abs_error = float(np.max(embedded_short_error))
    embedded_short_mean_abs_error = float(np.mean(embedded_short_error))
    embedded_short_p99_abs_error = float(
        np.quantile(embedded_short_error, 0.99)
    )
    # Use the checkpoint's own Short scores for gating and metrics.  The
    # reference file is a protocol-consistency check, not an authoritative
    # replacement for the deployed encoder.
    short_scores = embedded_short_scores
    long_scores = reference["long_scores"].astype(np.float64)
    speaker_ids = np.asarray(
        [speaker_from_source_key(value) for value in source_key],
        dtype=str,
    )
    if not np.all(stream_selected):
        raise RuntimeError(
            "the permissive streaming pass did not evaluate refinement "
            "on every frame"
        )
    # Apply the calibration-only threshold after the dense refinement pass so
    # every activation-budget curve can reuse the same refined scores.
    selected_full = np.abs(embedded_short_scores - 0.5) <= threshold
    adaptive_scores = gated_scores(
        embedded_short_scores,
        full_adaptive_scores,
        selected_full,
    )
    calibration_selected = selected_full[calibration_mask]
    test_selected = selected_full[test_mask]
    streaming_gate_mismatch_frames = 0

    _, calibration_counts = _utility_for_mask(
        labels,
        short_scores,
        adaptive_scores,
        selected_full,
        speaker_ids,
        calibration_mask,
    )
    test_cluster_names, test_counts = _utility_for_mask(
        labels,
        short_scores,
        adaptive_scores,
        selected_full,
        speaker_ids,
        test_mask,
    )
    test_metrics = metrics_from_counts(test_counts.sum(axis=0))
    test_delta = _delta_metrics(
        labels[test_mask],
        short_scores[test_mask],
        long_scores[test_mask],
        adaptive_scores[test_mask],
    )
    test_bootstrap = speaker_cluster_bootstrap(
        [test_counts],
        cluster_indices=np.arange(test_counts.shape[0], dtype=np.int64),
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    unseen_set = {str(value) for value in args.unseen_noise if str(value).strip()}
    noisy_mask = condition != "clean"
    unseen_mask = noisy_mask & np.isin(noise_name, sorted(unseen_set))
    seen_mask = noisy_mask & ~unseen_mask
    test_seen_mask = test_mask & seen_mask
    test_unseen_mask = test_mask & unseen_mask
    unseen_counts: np.ndarray | None = None
    unseen_bootstrap: dict[str, Any] | None = None
    if np.any(test_unseen_mask):
        _, unseen_counts = _utility_for_mask(
            labels,
            short_scores,
            adaptive_scores,
            selected_full,
            speaker_ids,
            test_unseen_mask,
        )
        unseen_bootstrap = speaker_cluster_bootstrap(
            [unseen_counts],
            cluster_indices=np.arange(
                unseen_counts.shape[0],
                dtype=np.int64,
            ),
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed + 1,
        )
    seen_counts: np.ndarray | None = None
    seen_bootstrap: dict[str, Any] | None = None
    if np.any(test_seen_mask):
        _, seen_counts = _utility_for_mask(
            labels,
            short_scores,
            adaptive_scores,
            selected_full,
            speaker_ids,
            test_seen_mask,
        )
        seen_bootstrap = speaker_cluster_bootstrap(
            [seen_counts],
            cluster_indices=np.arange(
                seen_counts.shape[0],
                dtype=np.int64,
            ),
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed + 2,
        )

    curves: list[dict[str, Any]] = []
    for budget in sorted(set(float(value) for value in args.activation_budgets)):
        budget_threshold, _ = choose_confidence_threshold(
            short_scores[calibration_mask],
            activation_rate=budget,
        )
        budget_selected = (
            np.abs(short_scores[test_mask] - 0.5) <= budget_threshold
        )
        budget_selected_full = np.zeros_like(selected_full, dtype=bool)
        budget_selected_full[test_mask] = budget_selected
        budget_adaptive_scores = gated_scores(
            short_scores,
            full_adaptive_scores,
            budget_selected_full,
        )
        _, budget_counts = _utility_for_mask(
            labels,
            short_scores,
            budget_adaptive_scores,
            budget_selected_full,
            speaker_ids,
            test_mask,
        )
        budget_metrics = metrics_from_counts(budget_counts.sum(axis=0))
        curves.append(
            {
                "calibration_activation": float(budget),
                "threshold": float(budget_threshold),
                "test_activation": float(np.mean(budget_selected)),
                "adaptive_f1": binary_metrics(
                    labels[test_mask],
                    budget_adaptive_scores[test_mask],
                )["f1"],
                "net_utility_per_selected": budget_metrics[
                    "net_utility_per_selected"
                ],
                "net_utility_per_frame": budget_metrics[
                    "net_utility_per_frame"
                ],
            }
        )

    by_condition = _condition_seen_metrics(
        labels[test_mask],
        short_scores[test_mask],
        long_scores[test_mask],
        adaptive_scores[test_mask],
        condition[test_mask],
        noise_name[test_mask],
        unseen_set,
    )
    for name in CONDITION_ORDER:
        mask = test_mask & (condition == name)
        if not np.any(mask):
            continue
        _, counts = _utility_for_mask(
            labels,
            short_scores,
            adaptive_scores,
            selected_full,
            speaker_ids,
            mask,
        )
        by_condition[name]["utility"] = metrics_from_counts(
            counts.sum(axis=0)
        )
    for name, mask, bootstrap in (
        ("seen", test_seen_mask, seen_bootstrap),
        ("unseen", test_unseen_mask, unseen_bootstrap),
    ):
        if not np.any(mask):
            continue
        _, counts = _utility_for_mask(
            labels,
            short_scores,
            adaptive_scores,
            selected_full,
            speaker_ids,
            mask,
        )
        by_condition[name]["utility"] = metrics_from_counts(
            counts.sum(axis=0)
        )
        if bootstrap is not None:
            by_condition[name]["utility_bootstrap"] = bootstrap

    summary: dict[str, Any] = {
        "protocol": {
            "manifest": str(args.manifest),
            "reference_predictions": str(args.reference_predictions),
            "adaptive_checkpoint": str(args.adaptive_checkpoint),
            "long_checkpoint": str(args.long_checkpoint),
            "adaptive_epoch": adaptive_payload.get("epoch"),
            "long_epoch": long_payload.get("epoch"),
            "row_sample": args.row_sample,
            "seed": args.seed,
            "utterances": len(all_labels),
            "valid_frames": int(labels.size),
            "valid_start_frame": valid_start,
            "embedded_short_reference_max_abs_error": (
                embedded_short_max_abs_error
            ),
            "embedded_short_reference_mean_abs_error": (
                embedded_short_mean_abs_error
            ),
            "embedded_short_reference_p99_abs_error": (
                embedded_short_p99_abs_error
            ),
            "streaming_gate_threshold_mismatch_frames": (
                streaming_gate_mismatch_frames
            ),
            "speaker_split_seed": args.speaker_split_seed,
            "calibration_fraction": args.calibration_fraction,
            "calibration_speaker_count": len(calibration_speakers),
            "test_speaker_count": len(test_speakers),
            "threshold_mode": threshold_mode,
            "primary_calibration_activation": (
                None
                if args.fixed_threshold is not None
                else float(args.primary_calibration_activation)
            ),
            "lookback_frames": refinement_config.lookback_frames,
            "lookback_seconds": (
                refinement_config.lookback_frames * FRAME_HOP / 16_000.0
            ),
            "refinement_macs_per_selected_frame": (
                model.refinement.estimated_macs_per_selected_frame()
            ),
            "unseen_noise": sorted(unseen_set),
        },
        "fixed_gate": {
            "threshold": float(threshold),
            "calibration_activation_rate": float(np.mean(calibration_selected)),
            "calibration_utility": metrics_from_counts(
                calibration_counts.sum(axis=0)
            ),
            "test": {
                **test_delta,
                "activation_rate": float(np.mean(test_selected)),
                "utility": test_metrics,
                "bootstrap": test_bootstrap,
            },
        },
        "activation_budget_curve": curves,
        "by_condition": by_condition,
        "test_speakers": {
            "calibration": calibration_speakers,
            "test": test_speakers,
            "test_clusters": test_cluster_names.tolist(),
            "seen_counts": None if seen_counts is None else seen_counts.tolist(),
            "seen_bootstrap": seen_bootstrap,
            "unseen_counts": (
                None if unseen_counts is None else unseen_counts.tolist()
            ),
            "unseen_bootstrap": unseen_bootstrap,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "frame_predictions.npz",
        labels=labels,
        short_scores=short_scores,
        long_scores=long_scores,
        adaptive_scores=adaptive_scores,
        full_adaptive_scores=full_adaptive_scores,
        selected=selected_full,
        test_mask=test_mask,
        calibration_mask=calibration_mask,
        embedded_short_scores=embedded_short_scores,
        threshold=np.asarray(threshold, dtype=np.float64),
        speaker_ids=speaker_ids,
        condition=condition,
        noise_name=noise_name,
        source_key=source_key,
    )
    with open(
        args.output_dir / "results.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    with open(
        args.output_dir / "report.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(_report(summary))
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
