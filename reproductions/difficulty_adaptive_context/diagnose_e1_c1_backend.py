# -*- coding: utf-8 -*-
"""Diagnose exact C1 reproduction across inference backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
    predict_full_adaptive_frames,
)
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "librivad"
LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
MANIFEST = DATA_ROOT / "manifests" / "LibriSpeech_test_medium.tsv"
CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025_rf384"
    / "best.pt"
)
REFERENCE = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
    / "seed17"
    / "eval_rf384_fixed130"
    / "frame_predictions.npz"
)
VALID_START = 382


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-frames", type=int, default=2000)
    parser.add_argument("--items", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _max_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    error = np.abs(
        np.asarray(actual, dtype=np.float64)
        - np.asarray(expected, dtype=np.float64)
    )
    return {
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
        "p99_abs": float(np.quantile(error, 0.99)),
    }


def main() -> int:
    args = _parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.items <= 0:
        raise ValueError("--items must be positive")
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.disable_mkldnn:
        torch.backends.mkldnn.enabled = False

    device = resolve_device(args.device)
    model, payload, config = load_adaptive_model(CHECKPOINT, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    items = build_evaluation_items(
        MANIFEST,
        generated_root=DATA_ROOT / "generated",
        label_root=DATA_ROOT / "labels",
        librispeech_root=LIBRISPEECH_ROOT,
        row_sample=1080,
        seed=17,
        include_clean=True,
    )
    with np.load(REFERENCE, allow_pickle=False) as reference:
        reference_full = np.asarray(
            reference["full_adaptive_scores"],
            dtype=np.float64,
        )
        reference_embedded_short = np.asarray(
            reference["embedded_short_scores"],
            dtype=np.float64,
        )
        reference_short = np.asarray(
            reference["short_scores"],
            dtype=np.float64,
        )

    cursor = 0
    rows: list[dict[str, object]] = []
    for item_index, item in enumerate(items[: args.items]):
        waveform = read_int16_audio(item.audio_path)
        sample_labels = np.load(item.label_path)
        if sample_labels.size != waveform.size:
            raise ValueError("label/audio mismatch")
        n_frames = waveform.size // FRAME_HOP + 1
        frame_labels = causal_frame_labels(sample_labels, n_frames)
        if n_frames <= VALID_START:
            continue
        retained = n_frames - VALID_START
        expected_end = cursor + retained
        if expected_end > reference_full.size:
            raise ValueError("reference bundle ended before diagnostic item")

        full, embedded_short, selected, _ = predict_full_adaptive_frames(
            model,
            frontend,
            waveform,
            device=device,
            chunk_frames=args.chunk_frames,
        )
        if full.size != n_frames or embedded_short.size != n_frames:
            raise ValueError("adaptive output frame count is inconsistent")
        expected_labels = frame_labels[VALID_START:]
        frozen_labels = reference_full[cursor:expected_end]
        if expected_labels.size != frozen_labels.size:
            raise ValueError("diagnostic frame alignment is inconsistent")

        row: dict[str, object] = {
            "item_index": item_index,
            "sample_id": item.sample_id,
            "source_key": item.source_key,
            "noise_name": item.noise_name,
            "condition": "clean" if item.is_clean else item.snr_db,
            "n_frames": n_frames,
            "retained_frames": retained,
            "full_vs_frozen": _max_error(
                full[VALID_START:],
                reference_full[cursor:expected_end],
            ),
            "embedded_short_vs_frozen": _max_error(
                embedded_short[VALID_START:],
                reference_embedded_short[cursor:expected_end],
            ),
            "embedded_short_vs_reference_short": _max_error(
                embedded_short[VALID_START:],
                reference_short[cursor:expected_end],
            ),
            "selected_frames": int(np.count_nonzero(selected[VALID_START:])),
        }
        rows.append(row)
        cursor = expected_end

    report = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "chunk_frames": args.chunk_frames,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_short_profile": payload.get("short_model_config", {}).get(
            "dilation_profile"
        ),
        "refinement_lookback_frames": config.lookback_frames,
        "refinement_dilations": list(config.dilations),
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
