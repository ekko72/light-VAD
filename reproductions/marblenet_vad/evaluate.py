# -*- coding: utf-8 -*-
"""Evaluate MarbleNet-3x2x64 with overlapping LibriVAD windows.

This follows the inference protocol in the MarbleNet paper:

* a 0.63 s window is shifted with 87.5% overlap by default;
* overlapping window probabilities are reduced with a median filter;
* predictions and labels are reported at the LibriVAD frame protocol
  (25 ms window, 10 ms hop, majority vote inside each frame).

The script also reports sample-level metrics, which are useful while
debugging the pipeline but are not directly comparable to the paper's
frame-level AV A-speech numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

try:
    from .dataset import (
        LibriVADUtterances,
        majority_label,
        window_starts,
    )
    from .features import MfccConfig, MfccFrontend
    from .model import build_marblenet_3x2x64
except ImportError:
    # Allows direct execution with:
    # python reproductions\marblenet_vad\evaluate.py
    from dataset import LibriVADUtterances, majority_label, window_starts
    from features import MfccConfig, MfccFrontend
    from model import build_marblenet_3x2x64


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "marblenet_vad" / "best.pt"
AUC_HISTOGRAM_BINS = 65_536


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.315
) -> float | None:
    """Interpolate TPR at a requested FPR on the empirical ROC curve."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(float(target_fpr), fpr, tpr))


def metrics_from_arrays(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    target_fpr: float = 0.315,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= float(threshold)
    return {
        "examples": int(labels.size),
        "speech": int(np.count_nonzero(labels == 1)),
        "silence": int(np.count_nonzero(labels == 0)),
        "accuracy": (
            float(np.mean(predictions == labels)) if labels.size else None
        ),
        "auc": safe_auc(labels, scores),
        "tpr_at_fpr_0.315": tpr_at_fpr(labels, scores, target_fpr),
    }


class StreamingMetrics:
    """Accumulate binary metrics without retaining all sample scores."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        target_fpr: float = 0.315,
        bins: int = AUC_HISTOGRAM_BINS,
    ) -> None:
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.threshold = float(threshold)
        self.target_fpr = float(target_fpr)
        self.bins = int(bins)
        self.examples = 0
        self.speech = 0
        self.silence = 0
        self.correct = 0
        self.positive_histogram = np.zeros(self.bins, dtype=np.int64)
        self.negative_histogram = np.zeros(self.bins, dtype=np.int64)

    def update(
        self, labels: np.ndarray, scores: np.ndarray
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if labels.shape != scores.shape:
            raise ValueError("labels and scores must have the same shape")
        if labels.size == 0:
            return
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")

        speech = labels == 1
        silence = labels == 0
        if int(np.count_nonzero(speech) + np.count_nonzero(silence)) != labels.size:
            raise ValueError("labels must contain only 0 and 1")

        predictions = scores >= self.threshold
        self.examples += int(labels.size)
        self.speech += int(np.count_nonzero(speech))
        self.silence += int(np.count_nonzero(silence))
        self.correct += int(np.count_nonzero(predictions == speech))

        indices = np.floor(
            np.clip(scores, 0.0, 1.0) * self.bins
        ).astype(np.int64)
        np.minimum(indices, self.bins - 1, out=indices)
        self.positive_histogram += np.bincount(
            indices[speech], minlength=self.bins
        )
        self.negative_histogram += np.bincount(
            indices[silence], minlength=self.bins
        )

    def metrics(self) -> dict[str, Any]:
        if self.examples == 0:
            return {
                "examples": 0,
                "speech": 0,
                "silence": 0,
                "accuracy": None,
                "auc": None,
                "tpr_at_fpr_0.315": None,
            }

        auc: float | None = None
        tpr: float | None = None
        if self.speech and self.silence:
            negatives_before = (
                np.cumsum(self.negative_histogram, dtype=np.float64)
                - self.negative_histogram
            )
            pair_score = float(
                np.dot(
                    self.positive_histogram,
                    negatives_before + 0.5 * self.negative_histogram,
                )
            )
            auc = pair_score / (self.speech * self.silence)

            negatives_from_high = np.cumsum(
                self.negative_histogram[::-1], dtype=np.float64
            )
            positives_from_high = np.cumsum(
                self.positive_histogram[::-1], dtype=np.float64
            )
            fpr = negatives_from_high / self.silence
            tpr_values = positives_from_high / self.speech
            index = int(
                np.searchsorted(fpr, self.target_fpr, side="left")
            )
            if index == 0:
                tpr = 0.0
            elif index >= fpr.size:
                tpr = 1.0
            elif fpr[index] == fpr[index - 1]:
                tpr = float(tpr_values[index])
            else:
                fraction = (
                    (self.target_fpr - fpr[index - 1])
                    / (fpr[index] - fpr[index - 1])
                )
                tpr = float(
                    tpr_values[index - 1]
                    + fraction * (tpr_values[index] - tpr_values[index - 1])
                )

        return {
            "examples": self.examples,
            "speech": self.speech,
            "silence": self.silence,
            "accuracy": self.correct / self.examples,
            "auc": auc,
            "tpr_at_fpr_0.315": tpr,
        }


def aggregate_window_scores(
    n_samples: int,
    starts: np.ndarray,
    window_scores: np.ndarray,
    *,
    segment_samples: int,
    hop_samples: int,
    smoothing: str,
) -> np.ndarray:
    """Reduce overlapping 0.63 s window scores to one score per sample.

    A window contributes its probability to every sample it covers.  The
    reduction is the median (paper default) or mean over contributing
    windows.
    """
    if len(starts) != len(window_scores):
        raise ValueError("starts and window_scores must have the same length")
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)

    # With a fixed hop, a sample is covered by at most
    # ceil(segment_samples / hop_samples) regular windows.  Assigning
    # window i to row i % overlap_count keeps those contributions in
    # distinct rows, avoiding a dense [n_windows, n_samples] matrix.
    overlap_count = max(1, int(np.ceil(segment_samples / hop_samples)))
    values = np.full(
        (overlap_count + 1, n_samples), np.nan, dtype=np.float64
    )
    tail_start = int(starts[-1]) if len(starts) else 0
    has_tail = bool(
        len(starts)
        and tail_start != (len(starts) - 1) * int(hop_samples)
    )
    for index, start in enumerate(starts):
        start = int(start)
        row = (
            overlap_count
            if has_tail and index == len(starts) - 1
            else index % overlap_count
        )
        values[row, start : start + segment_samples] = float(
            window_scores[index]
        )
    if smoothing == "median":
        reduced = np.nanmedian(values, axis=0)
    elif smoothing == "mean":
        reduced = np.nanmean(values, axis=0)
    else:
        raise ValueError(f"unknown smoothing mode: {smoothing}")
    if np.isnan(reduced).any():
        raise RuntimeError("some samples were not covered by any window")
    return np.asarray(reduced, dtype=np.float64)


def frame_arrays(
    sample_scores: np.ndarray,
    sample_labels: np.ndarray,
    *,
    frame_window: int,
    frame_hop: int,
    smoothing: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sample scores/labels to the 25 ms / 10 ms LibriVAD frames."""
    starts = window_starts(len(sample_labels), frame_window, frame_hop)
    frame_scores: list[float] = []
    frame_labels: list[int] = []
    for start in starts:
        score_slice = sample_scores[start : start + frame_window]
        label_slice = sample_labels[start : start + frame_window]
        if smoothing == "median":
            frame_scores.append(float(np.median(score_slice)))
        elif smoothing == "mean":
            frame_scores.append(float(np.mean(score_slice)))
        else:
            raise ValueError(f"unknown smoothing mode: {smoothing}")
        frame_labels.append(majority_label(label_slice))
    return np.asarray(frame_scores), np.asarray(frame_labels, dtype=np.int64)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a training checkpoint and reconstruct the model configuration."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint format: {checkpoint_path}")
    state_dict = payload.get("model", payload)
    model_config = payload.get("model_config", {})
    model = build_marblenet_3x2x64(
        feat_in=int(model_config.get("feat_in", 64)),
        num_classes=int(model_config.get("num_classes", 2)),
        dropout=float(model_config.get("dropout", 0.0)),
        causal=bool(model_config.get("causal", False)),
        dilation_profile=model_config.get("dilation_profile", "baseline"),
        frame_output=bool(model_config.get("frame_output", False)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


@torch.inference_mode()
def predict_windows(
    model: nn.Module,
    frontend: nn.Module,
    waveform: torch.Tensor,
    starts: np.ndarray,
    *,
    segment_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return speech probabilities for each requested window start."""
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    probabilities: list[np.ndarray] = []
    for offset in range(0, starts.size, batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = torch.stack(
            [
                waveform[
                    int(start) : int(start) + int(segment_samples)
                ]
                for start in batch_starts
            ]
        )
        features = frontend(windows.to(device, non_blocking=True))
        logits = model(features)
        probabilities.append(
            torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        )
    return np.concatenate(probabilities).astype(np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MarbleNet on LibriVAD-style mixtures."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--row-sample",
        type=int,
        default=None,
        help=(
            "Deterministically sample this many manifest rows across "
            "noise/SNR conditions before evaluation."
        ),
    )
    parser.add_argument(
        "--segment-samples",
        type=int,
        default=None,
        help="Defaults to the value stored in the training checkpoint.",
    )
    parser.add_argument("--overlap", type=float, default=0.875)
    parser.add_argument(
        "--smoothing", choices=("median", "mean"), default="median"
    )
    parser.add_argument("--frame-window", type=float, default=0.025)
    parser.add_argument("--frame-hop", type=float, default=0.010)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.segment_samples is not None and args.segment_samples <= 0:
        parser.error("--segment-samples must be positive")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    if args.frame_window <= 0.0 or args.frame_hop <= 0.0:
        parser.error("frame window and hop must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.row_sample is not None and args.row_sample < 0:
        parser.error("--row-sample must be non-negative")


def evaluate_manifest(
    *,
    manifest: Path,
    checkpoint_path: Path,
    data_root: Path | None,
    generated_root: Path | None,
    label_root: Path | None,
    limit: int | None,
    row_sample: int | None,
    seed: int,
    segment_samples: int | None,
    overlap: float,
    smoothing: str,
    frame_window_samples: int,
    frame_hop_samples: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, payload = load_model(checkpoint_path, device)
    if segment_samples is None:
        segment_samples = int(
            payload.get("model_config", {}).get("segment_samples", 10_080)
        )
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    model_config = payload.get("model_config", {})
    feature_config = MfccConfig(
        causal=bool(model_config.get("causal", False))
    )
    frontend = MfccFrontend(feature_config).to(device)
    hop_samples = max(
        1, int(round(segment_samples * (1.0 - float(overlap))))
    )
    dataset = LibriVADUtterances(
        manifest,
        generated_root=generated_root,
        label_root=label_root,
        data_root=data_root,
        segment_samples=segment_samples,
        hop_samples=hop_samples,
        limit=limit,
        row_sample=row_sample,
        seed=seed,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "no utterance is long enough for the requested segment length"
        )

    sample_metrics = StreamingMetrics(
        threshold=threshold,
        target_fpr=0.315,
    )
    frame_metrics = StreamingMetrics(
        threshold=threshold,
        target_fpr=0.315,
    )
    per_utterance: list[dict[str, Any]] = []

    print(
        f"device={device}, utterances={len(dataset)}, "
        f"row_sample={row_sample}, "
        f"segment={segment_samples} samples, hop={hop_samples} samples "
        f"({overlap:.1%} overlap), smoothing={smoothing}"
    )
    for index in range(len(dataset)):
        item = dataset[index]
        waveform = item["waveform"]
        assert isinstance(waveform, torch.Tensor)
        sample_labels = (
            item["labels"].numpy().astype(np.int64)
            if isinstance(item["labels"], torch.Tensor)
            else np.asarray(item["labels"], dtype=np.int64)
        )
        starts = (
            item["starts"].numpy().astype(np.int64)
            if isinstance(item["starts"], torch.Tensor)
            else np.asarray(item["starts"], dtype=np.int64)
        )
        window_scores = predict_windows(
            model,
            frontend,
            waveform,
            starts,
            segment_samples=segment_samples,
            batch_size=batch_size,
            device=device,
        )
        sample_scores = aggregate_window_scores(
            len(sample_labels),
            starts,
            window_scores,
            segment_samples=segment_samples,
            hop_samples=hop_samples,
            smoothing=smoothing,
        )
        frame_scores, frame_labels = frame_arrays(
            sample_scores,
            sample_labels,
            frame_window=frame_window_samples,
            frame_hop=frame_hop_samples,
            smoothing=smoothing,
        )

        sample_metrics.update(sample_labels, sample_scores)
        frame_metrics.update(frame_labels, frame_scores)
        per_utterance.append(
            {
                "index": index,
                "sample_id": item.get("sample_id", ""),
                "noise_name": item.get("noise_name", ""),
                "snr_db": item.get("snr_db", ""),
                "samples": int(sample_labels.size),
                "windows": int(starts.size),
                "sample": metrics_from_arrays(
                    sample_labels, sample_scores, threshold=threshold
                ),
                "frame": metrics_from_arrays(
                    frame_labels, frame_scores, threshold=threshold
                ),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print(f"  evaluated {index + 1}/{len(dataset)} utterances")

    summary = {
        "manifest": str(manifest),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "causal": bool(model_config.get("causal", False)),
        "segment_samples": int(segment_samples),
        "hop_samples": int(hop_samples),
        "overlap": float(overlap),
        "smoothing": smoothing,
        "frame_window_samples": int(frame_window_samples),
        "frame_hop_samples": int(frame_hop_samples),
        "threshold": float(threshold),
        "utterances": len(dataset),
        "sample": sample_metrics.metrics(),
        "frame": frame_metrics.metrics(),
        "per_utterance": per_utterance,
    }
    return summary


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    device = resolve_device(args.device)
    feature_config = MfccConfig()
    frame_window_samples = feature_config.samples_for_seconds(args.frame_window)
    frame_hop_samples = feature_config.samples_for_seconds(args.frame_hop)
    if frame_window_samples <= 0 or frame_hop_samples <= 0:
        parser.error("frame window and hop must map to positive sample counts")

    summary = evaluate_manifest(
        manifest=args.manifest,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        generated_root=args.generated_root,
        label_root=args.label_root,
            limit=args.limit,
            row_sample=args.row_sample,
            seed=args.seed,
            segment_samples=args.segment_samples,
        overlap=args.overlap,
        smoothing=args.smoothing,
        frame_window_samples=frame_window_samples,
        frame_hop_samples=frame_hop_samples,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=device,
    )

    output_path = args.output_json
    if output_path is None:
        output_path = args.checkpoint.parent / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

    sample = summary["sample"]
    frame = summary["frame"]
    print(
        "sample: "
        f"accuracy={sample['accuracy']}, auc={sample['auc']}, "
        f"tpr@fpr0.315={sample['tpr_at_fpr_0.315']}"
    )
    print(
        "frame:  "
        f"accuracy={frame['accuracy']}, auc={frame['auc']}, "
        f"tpr@fpr0.315={frame['tpr_at_fpr_0.315']}"
    )
    print(f"artifacts written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
