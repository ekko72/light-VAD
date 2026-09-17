# -*- coding: utf-8 -*-
"""Evaluate paired Short-RF and Long-RF models for A1-A5."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    EvaluationItem,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
    resolve_label_path,
)
from reproductions.difficulty_adaptive_context.train_context import (
    UNSEEN_NOISE,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import (
    build_marblenet_3x2x64,
    count_parameters,
    receptive_field,
)
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
CONDITION_ORDER = ("clean", "20", "10", "5", "0", "-5")
DIFFICULTY_NAMES = (
    "Very hard",
    "Hard",
    "Medium",
    "Easy",
    "Very easy",
)
BOUNDARY_NAMES = ("0-50 ms", "50-100 ms", "100-200 ms", ">200 ms")


def load_frame_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = dict(payload.get("model_config", {}))
    if not bool(model_config.get("frame_output", False)):
        raise ValueError(
            f"checkpoint is not frame-level: {checkpoint_path}"
        )
    profile = model_config.get("dilation_profile", "baseline")
    model = build_marblenet_3x2x64(
        feat_in=int(model_config.get("feat_in", 64)),
        num_classes=int(model_config.get("num_classes", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
        causal=bool(model_config.get("causal", True)),
        dilation_profile=profile,
        frame_output=True,
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload, model_config


@torch.inference_mode()
def predict_frames(
    model: torch.nn.Module,
    frontend: torch.nn.Module,
    waveform: np.ndarray,
    *,
    device: torch.device,
    chunk_frames: int,
) -> np.ndarray:
    tensor = torch.from_numpy(
        waveform.astype(np.float32) * INT16_SCALE
    ).unsqueeze(0)
    features = frontend(tensor.to(device, non_blocking=True))
    states = None
    scores: list[np.ndarray] = []
    for start in range(0, features.shape[-1], chunk_frames):
        chunk = features[..., start : start + chunk_frames]
        logits, states = model.forward_stream(chunk, states)
        probabilities = torch.softmax(logits.transpose(1, 2), dim=-1)[..., 1]
        scores.append(probabilities.detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores).astype(np.float64)


def boundary_distance_ms(frame_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(frame_labels, dtype=np.int64)
    if labels.size == 0:
        return np.empty(0, dtype=np.float64)
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    if boundaries.size == 0:
        return np.full(labels.size, np.inf, dtype=np.float64)
    indices = np.arange(labels.size, dtype=np.int64)
    distance_frames = np.min(
        np.abs(indices[:, None] - boundaries[None, :]), axis=1
    )
    return distance_frames.astype(np.float64) * 10.0


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predictions = scores >= 0.5
    tp = int(np.count_nonzero((labels == 1) & predictions))
    fp = int(np.count_nonzero((labels == 0) & predictions))
    fn = int(np.count_nonzero((labels == 1) & ~predictions))
    tn = int(np.count_nonzero((labels == 0) & ~predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    return {
        "frames": int(labels.size),
        "speech": int(np.count_nonzero(labels == 1)),
        "silence": int(np.count_nonzero(labels == 0)),
        "accuracy": (
            float((tp + tn) / labels.size) if labels.size else None
        ),
        "error": (
            float((fp + fn) / labels.size) if labels.size else None
        ),
        "precision": precision if labels.size else None,
        "recall": recall if labels.size else None,
        "f1": f1 if labels.size else None,
        "auc": (
            float(roc_auc_score(labels, scores))
            if labels.size and np.unique(labels).size > 1
            else None
        ),
    }


def delta_metrics(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
) -> dict[str, Any]:
    short = binary_metrics(labels, short_scores)
    long = binary_metrics(labels, long_scores)
    return {
        "frames": int(labels.size),
        "short": short,
        "long": long,
        "delta_f1_long_minus_short": (
            None
            if short["f1"] is None or long["f1"] is None
            else float(long["f1"] - short["f1"])
        ),
        "error_reduction_short_minus_long": (
            None
            if short["error"] is None or long["error"] is None
            else float(short["error"] - long["error"])
        ),
        "delta_auc_long_minus_short": (
            None
            if short["auc"] is None or long["auc"] is None
            else float(long["auc"] - short["auc"])
        ),
    }


def condition_key(item: EvaluationItem) -> str:
    return "clean" if item.is_clean else str(item.snr_db)


def boundary_mask(distance_ms: np.ndarray, bucket: int) -> np.ndarray:
    if bucket == 0:
        return distance_ms <= 50.0
    if bucket == 1:
        return (distance_ms > 50.0) & (distance_ms <= 100.0)
    if bucket == 2:
        return (distance_ms > 100.0) & (distance_ms <= 200.0)
    return distance_ms > 200.0


def safe_spearman(
    x: list[float], y: list[float]
) -> dict[str, float | None]:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p_value": None}
    result = spearmanr(x, y, nan_policy="omit")
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def md_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def md_number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(summary: dict[str, Any]) -> str:
    a1 = summary["A1_overall_by_snr"]
    a2 = summary["A2_short_difficulty_buckets"]
    a3 = summary["A3_context_gain_by_snr"]
    a4 = summary["A4_boundary_distance"]
    a5 = summary["A5_seen_unseen_noise"]
    protocol = summary["protocol"]

    lines = [
        "# Difficulty-Adaptive Temporal Context: A1-A5",
        "",
        "## A0 Protocol",
        "",
        f"- Dataset manifest: `{protocol['manifest']}`",
        f"- Evaluation utterances: {protocol['utterances']}",
        f"- Valid frames: {protocol['valid_frames']}",
        f"- Context used for training: {protocol['context_seconds']:.2f} s",
        f"- Target segment: {protocol['target_seconds']:.2f} s",
        f"- Seed: {protocol['seed']}",
        f"- Short-RF: `{protocol['short_profile']}`, "
        f"{protocol['short_rf']['receptive_field_frames']} frames "
        f"({protocol['short_rf']['receptive_field_seconds']:.2f} s)",
        f"- Long-RF: `{protocol['long_profile']}`, "
        f"{protocol['long_rf']['receptive_field_frames']} frames "
        f"({protocol['long_rf']['receptive_field_seconds']:.2f} s)",
        f"- Parameters: Short={protocol['short_parameters']:,}, "
        f"Long={protocol['long_parameters']:,}",
        f"- Unseen noise classes: {', '.join(protocol['unseen_noise'])}",
        "",
        "The two models use the same manifest rows, frame labels, MFCC frontend, "
        "channels, kernels, loss, optimizer, seed and training budget. The only "
        "intentional difference is the five-block dilation profile.",
        "",
        "## A1 Overall Long vs Short",
        "",
        "| SNR | Short F1 | Long F1 | Delta F1 | Short error | Long error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        row = a1.get(condition)
        label = "Clean" if condition == "clean" else condition
        if row is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {md_number(row['short']['f1'])} | "
            f"{md_number(row['long']['f1'])} | "
            f"{md_number(row['delta_f1_long_minus_short'])} | "
            f"{md_percent(row['short']['error'])} | "
            f"{md_percent(row['long']['error'])} |"
        )

    lines.extend(
        [
            "",
            "## A2 Short-Model Difficulty Buckets",
            "",
            "Bucket boundaries are global equal-frequency quintiles of "
            "`confidence = abs(p_short - 0.5)`.",
            "",
            "| Short difficulty | Frames | Confidence range | Short error | "
            "Long error | Error reduction |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in zip(DIFFICULTY_NAMES, a2):
        lines.append(
            f"| {name} | {row['frames']} | "
            f"{row['confidence_min']:.4f}-{row['confidence_max']:.4f} | "
            f"{md_percent(row['short']['error'])} | "
            f"{md_percent(row['long']['error'])} | "
            f"{md_percent(row['error_reduction_short_minus_long'])} |"
        )
    lines.append(
        "- Spearman(difficulty rank, error reduction): "
        f"rho={a2[0]['difficulty_spearman']['rho']}, "
        f"p={a2[0]['difficulty_spearman']['p_value']}."
    )

    lines.extend(
        [
            "",
            "## A3 SNR-by-Context Gain",
            "",
            "| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition in CONDITION_ORDER:
        row = a3.get(condition)
        label = "Clean" if condition == "clean" else condition
        if row is None:
            lines.append(f"| {label} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {md_number(row['delta_f1_long_minus_short'])} | "
            f"{md_percent(row['error_reduction_short_minus_long'])} | "
            f"{md_number(row['delta_auc_long_minus_short'])} |"
        )
    lines.append(
        "- Spearman(SNR rank, Delta F1): "
        f"rho={summary['A3_trend']['rho']}, "
        f"p={summary['A3_trend']['p_value']}. Lower SNR rank means more noise; "
        "a negative rho supports larger gains at lower SNR."
    )

    lines.extend(
        [
            "",
            "## A4 Boundary Distance",
            "",
            "| Distance to nearest GT boundary | Frames | Short error | "
            "Long error | Error reduction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in zip(BOUNDARY_NAMES, a4):
        lines.append(
            f"| {name} | {row['frames']} | "
            f"{md_percent(row['short']['error'])} | "
            f"{md_percent(row['long']['error'])} | "
            f"{md_percent(row['error_reduction_short_minus_long'])} |"
        )

    lines.extend(
        [
            "",
            "## A5 Seen vs Unseen Noise",
            "",
            "| Noise group | Frames | Short F1 | Long F1 | Delta F1 | "
            "Error reduction |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("seen", "unseen"):
        row = a5[name]
        lines.append(
            f"| {name} | {row['frames']} | {md_number(row['short']['f1'])} | "
            f"{md_number(row['long']['f1'])} | "
            f"{md_number(row['delta_f1_long_minus_short'])} | "
            f"{md_percent(row['error_reduction_short_minus_long'])} |"
        )
    lines.extend(
        [
            "",
            f"- Unseen minus seen error reduction: "
            f"{md_percent(summary['A5_unseen_minus_seen_error_reduction'])}.",
            f"- Unseen minus seen Delta F1: "
            f"{md_number(summary['A5_unseen_minus_seen_delta_f1'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Short-RF and Long-RF frame models for A1-A5."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--short-checkpoint", type=Path, required=True)
    parser.add_argument("--long-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=DEFAULT_LIBRISPEECH_ROOT,
    )
    parser.add_argument(
        "--row-sample",
        type=int,
        default=1_080,
        help="Deterministic stratified noisy-manifest rows.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-chunk-frames", type=int, default=2_000)
    parser.add_argument(
        "--unseen-noise",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.row_sample is not None and args.row_sample <= 0:
        parser.error("--row-sample must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.score_chunk_frames <= 0:
        parser.error("--score-chunk-frames must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    device = resolve_device(args.device)

    short_model, short_payload, short_config = load_frame_model(
        args.short_checkpoint, device
    )
    long_model, long_payload, long_config = load_frame_model(
        args.long_checkpoint, device
    )
    short_profile = short_config.get("dilation_profile", "baseline")
    long_profile = long_config.get("dilation_profile", "baseline")
    short_rf = receptive_field(short_profile)
    long_rf = receptive_field(long_profile)
    if int(short_rf["lookback_frames"]) >= int(long_rf["lookback_frames"]):
        raise ValueError(
            "Long-RF checkpoint must have a strictly larger receptive field"
        )
    short_parameters = count_parameters(short_model)
    long_parameters = count_parameters(long_model)
    if short_parameters != long_parameters:
        raise ValueError(
            "A0 requires equal parameter counts: "
            f"{short_parameters} != {long_parameters}"
        )

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

    frontend_config = MfccConfig(causal=True)
    short_frontend = MfccFrontend(frontend_config).to(device)
    long_frontend = MfccFrontend(frontend_config).to(device)
    valid_start = int(long_rf["lookback_frames"])

    condition_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    condition_short: dict[str, list[np.ndarray]] = defaultdict(list)
    condition_long: dict[str, list[np.ndarray]] = defaultdict(list)
    noise_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    noise_short: dict[str, list[np.ndarray]] = defaultdict(list)
    noise_long: dict[str, list[np.ndarray]] = defaultdict(list)

    all_labels: list[np.ndarray] = []
    all_short: list[np.ndarray] = []
    all_long: list[np.ndarray] = []
    all_boundary: list[np.ndarray] = []
    all_noise: list[np.ndarray] = []
    all_condition: list[np.ndarray] = []
    all_source: list[np.ndarray] = []

    print(
        f"device={device}, items={len(items)}, valid_start_frame={valid_start}, "
        f"short_rf={short_rf['receptive_field_frames']} frames, "
        f"long_rf={long_rf['receptive_field_frames']} frames",
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
        short_scores = predict_frames(
            short_model,
            short_frontend,
            waveform,
            device=device,
            chunk_frames=args.score_chunk_frames,
        )
        long_scores = predict_frames(
            long_model,
            long_frontend,
            waveform,
            device=device,
            chunk_frames=args.score_chunk_frames,
        )
        if short_scores.size != n_frames or long_scores.size != n_frames:
            raise RuntimeError("model output frame count is inconsistent")

        labels = frame_labels[valid_start:]
        short_part = short_scores[valid_start:]
        long_part = long_scores[valid_start:]
        boundary_part = boundary_distance_ms(frame_labels)[valid_start:]
        noise_part = np.full(
            labels.size, item.noise_name, dtype=f"U{max(1, len(item.noise_name))}"
        )
        condition_part = np.full(
            labels.size,
            condition_key(item),
            dtype=f"U{max(1, len(condition_key(item)))}",
        )
        source_part = np.full(
            labels.size,
            item.source_key,
            dtype=f"U{max(1, len(item.source_key))}",
        )

        condition = condition_key(item)
        condition_labels[condition].append(labels)
        condition_short[condition].append(short_part)
        condition_long[condition].append(long_part)
        if not item.is_clean:
            noise_labels[item.noise_name].append(labels)
            noise_short[item.noise_name].append(short_part)
            noise_long[item.noise_name].append(long_part)
        all_labels.append(labels)
        all_short.append(short_part)
        all_long.append(long_part)
        all_boundary.append(boundary_part)
        all_noise.append(noise_part)
        all_condition.append(condition_part)
        all_source.append(source_part)
        if args.progress_every > 0 and (
            (index + 1) % args.progress_every == 0 or index + 1 == len(items)
        ):
            print(f"  evaluated {index + 1}/{len(items)}", flush=True)

    if not all_labels:
        raise RuntimeError("all selected utterances were too short")

    labels_array = np.concatenate(all_labels)
    short_array = np.concatenate(all_short)
    long_array = np.concatenate(all_long)
    boundary_array = np.concatenate(all_boundary)
    noise_array = np.concatenate(all_noise)
    condition_array = np.concatenate(all_condition)
    source_array = np.concatenate(all_source)

    a1: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        if condition not in condition_labels:
            continue
        a1[condition] = delta_metrics(
            np.concatenate(condition_labels[condition]),
            np.concatenate(condition_short[condition]),
            np.concatenate(condition_long[condition]),
        )

    confidence = np.abs(short_array - 0.5)
    thresholds = np.quantile(confidence, [0.2, 0.4, 0.6, 0.8])
    bucket_ids = np.digitize(confidence, thresholds, right=False)
    a2: list[dict[str, Any]] = []
    for bucket, name in enumerate(DIFFICULTY_NAMES):
        mask = bucket_ids == bucket
        row = delta_metrics(
            labels_array[mask], short_array[mask], long_array[mask]
        )
        row["name"] = name
        row["confidence_min"] = (
            float(np.min(confidence[mask])) if np.any(mask) else 0.0
        )
        row["confidence_max"] = (
            float(np.max(confidence[mask])) if np.any(mask) else 0.0
        )
        a2.append(row)
    a2_spearman = safe_spearman(
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [
            float(row["error_reduction_short_minus_long"] or 0.0)
            for row in a2
        ],
    )
    for row in a2:
        row["difficulty_spearman"] = a2_spearman

    condition_rank = {
        "clean": 6.0,
        "20": 5.0,
        "10": 4.0,
        "5": 3.0,
        "0": 2.0,
        "-5": 1.0,
    }
    a3_trend = safe_spearman(
        [condition_rank[name] for name in CONDITION_ORDER if name in a1],
        [
            float(a1[name]["delta_f1_long_minus_short"] or 0.0)
            for name in CONDITION_ORDER
            if name in a1
        ],
    )

    a4: list[dict[str, Any]] = []
    for bucket, name in enumerate(BOUNDARY_NAMES):
        mask = boundary_mask(boundary_array, bucket)
        row = delta_metrics(
            labels_array[mask], short_array[mask], long_array[mask]
        )
        row["name"] = name
        a4.append(row)

    unseen_set = set(args.unseen_noise)
    noisy_mask = condition_array != "clean"
    unseen_mask = noisy_mask & np.isin(noise_array, sorted(unseen_set))
    seen_mask = noisy_mask & ~unseen_mask
    a5 = {
        "seen": delta_metrics(
            labels_array[seen_mask],
            short_array[seen_mask],
            long_array[seen_mask],
        ),
        "unseen": delta_metrics(
            labels_array[unseen_mask],
            short_array[unseen_mask],
            long_array[unseen_mask],
        ),
    }
    seen_error_gain = (
        a5["seen"]["error_reduction_short_minus_long"] or 0.0
    )
    unseen_error_gain = (
        a5["unseen"]["error_reduction_short_minus_long"] or 0.0
    )
    seen_f1_gain = a5["seen"]["delta_f1_long_minus_short"] or 0.0
    unseen_f1_gain = a5["unseen"]["delta_f1_long_minus_short"] or 0.0

    summary: dict[str, Any] = {
        "protocol": {
            "manifest": str(args.manifest),
            "row_sample": args.row_sample,
            "seed": args.seed,
            "utterances": len(all_labels),
            "valid_frames": int(labels_array.size),
            "valid_start_frame": valid_start,
            "context_seconds": (
                float(short_config.get("context_samples", 0)) / 16_000.0
            ),
            "target_seconds": (
                float(short_config.get("target_samples", 0)) / 16_000.0
            ),
            "short_checkpoint": str(args.short_checkpoint),
            "long_checkpoint": str(args.long_checkpoint),
            "short_epoch": short_payload.get("epoch"),
            "long_epoch": long_payload.get("epoch"),
            "short_profile": short_profile,
            "long_profile": long_profile,
            "short_rf": short_rf,
            "long_rf": long_rf,
            "short_parameters": short_parameters,
            "long_parameters": long_parameters,
            "unseen_noise": sorted(unseen_set),
        },
        "A1_overall_by_snr": a1,
        "A2_short_difficulty_buckets": a2,
        "A2_confidence_thresholds": thresholds.tolist(),
        "A3_context_gain_by_snr": a1,
        "A3_trend": a3_trend,
        "A4_boundary_distance": a4,
        "A5_seen_unseen_noise": a5,
        "A5_unseen_minus_seen_error_reduction": float(
            unseen_error_gain - seen_error_gain
        ),
        "A5_unseen_minus_seen_delta_f1": float(
            unseen_f1_gain - seen_f1_gain
        ),
        "per_noise": {
            name: delta_metrics(
                np.concatenate(noise_labels[name]),
                np.concatenate(noise_short[name]),
                np.concatenate(noise_long[name]),
            )
            for name in sorted(noise_labels)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "frame_predictions.npz",
        labels=labels_array,
        short_scores=short_array,
        long_scores=long_array,
        boundary_distance_ms=boundary_array,
        noise_name=noise_array,
        condition=condition_array,
        source_key=source_array,
    )
    with open(
        args.output_dir / "results.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    with open(
        args.output_dir / "report.md", "w", encoding="utf-8"
    ) as handle:
        handle.write(build_report(summary))
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
