# -*- coding: utf-8 -*-
"""Frozen AVA-Speech real-recording validation for the A-ANALYSIS story.

The runner is intentionally separate from the hash-locked LibriVAD
A-ANALYSIS runners.  It never reads ``NEW_FINAL_OOD`` or ``FINAL_OOD``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

if os.name != "nt":
    # The frozen checkpoints were serialized on Windows.
    pathlib.WindowsPath = pathlib.PosixPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveCausalVAD,
    AdaptiveStreamingVAD,
    RefinementConfig,
    SparseCausalMultiScaleRefinement,
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
from reproductions.difficulty_adaptive_context.evaluate_context import (
    binary_metrics,
    load_frame_model,
    predict_frames,
)
from reproductions.difficulty_adaptive_context.run_a_analysis_v2_long_term_stats import (
    LongTermStatsFiLM,
)
from reproductions.difficulty_adaptive_context.run_ax1_value_predictability import (
    feature_blocks,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import build_marblenet_3x2x64
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ava_speech_v1"
)
SMOKE_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "ava_speech_v1_smoke"
)

AVA_ROOT = REPO_ROOT / "data" / "ava_speech"
AVA_HF_ROOT = AVA_ROOT / "hf"
AVA_LABEL_CSV = AVA_ROOT / "ava_speech_labels_v1.csv"
AVA_BASELINE_ROOT = AVA_ROOT / "baselines"
AVA_NEMO_PREPROCESSOR = AVA_BASELINE_ROOT / "preprocessor.onnx"
AVA_NEMO_MODEL = (
    AVA_BASELINE_ROOT / "frame_vad_multilingual_marblenet_v2.0.onnx"
)

SHORT_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_short_ext40"
    / "best.pt"
)
LONG_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_long_ext40"
    / "best.pt"
)
ADAPTIVE_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a3_pilot_seed17_gate013_distill025_rf384"
    / "best.pt"
)

PROTOCOL_ID = "AVA-SPEECH-v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_AVA_SCORING"
SPLIT_SEED = 20261001
VALIDATION_FRACTION = 0.20
SOURCE_BOOTSTRAP_REPEATS = 2000
SOURCE_BOOTSTRAP_SEED = 20261002
FRAME_HOP_SAMPLES = 160
SAMPLE_RATE = 16_000
SLICE_LENGTH = 765
LOOKBACK_FRAMES = 382
REMOTE_START = 382
REMOTE_END = 764
DISTANCE_SECONDS = (0.25, 0.5, 1.0, 2.0, 3.0)
DISTANCE_FRAMES = (25, 50, 100, 200, 300)
INTERVENTION_SEEDS = (17, 23, 41)
INTERVENTION_TARGETS_PER_VIDEO = 512
INTERVENTION_BATCH_SIZE = 32
GATE_THRESHOLD = 0.13
RANDOM_GATE_SEEDS = tuple(range(101, 201))
HARD_FRAME_FRACTION = 0.20
LOGLOSS_EPSILON = 1e-12

GBDT_SEED = 17
GBDT_MAX_ITER = 80
GBDT_LEARNING_RATE = 0.08
GBDT_MAX_LEAF_NODES = 15
GBDT_MIN_SAMPLES_LEAF = 50
GBDT_L2_REGULARIZATION = 1e-3
GBDT_TRAIN_FRAMES_PER_VIDEO = 2000
GBDT_WINDOW_FRAMES = 25

LIGHTWEIGHT_TAU_SECONDS = 0.5
LIGHTWEIGHT_SEEDS = (17, 23, 41)
LIGHTWEIGHT_HIDDEN_DIM = 64
LIGHTWEIGHT_EPOCHS = 3
LIGHTWEIGHT_BATCH_SIZE = 32
LIGHTWEIGHT_LR = 1.0e-3
LIGHTWEIGHT_WEIGHT_DECAY = 1.0e-4
LIGHTWEIGHT_CHUNKS_PER_VIDEO = 8
LIGHTWEIGHT_CHUNK_FRAMES = 512
LIGHTWEIGHT_PREFIX_FRAMES = 100
LIGHTWEIGHT_ADDED_CACHE_BYTES = 2 * 64 * 4

POSITIVE_LABELS = {
    "CLEAN_SPEECH",
    "SPEECH_WITH_NOISE",
    "SPEECH_WITH_MUSIC",
}
CONDITION_NAMES = (
    "overall",
    "clean_speech",
    "speech_with_noise",
    "speech_with_music",
    "no_speech",
)
MAIN_METRIC_NAMES = (
    "f1",
    "auc",
    "proper_loss",
    "tpr_at_fpr_0.315",
)
MODEL_ORDER = (
    "Short",
    "Long-RF",
    "Adaptive",
    "AlwaysRefine",
    "WebRTC VAD",
    "Silero VAD",
    "NeMo MarbleNet",
    "Shallow GBDT",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_status_short() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def _stable_hash_int(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _proper_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.clip(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        LOGLOSS_EPSILON,
        1.0 - LOGLOSS_EPSILON,
    )
    if labels.size == 0:
        return float("nan")
    return float(
        -np.mean(
            labels * np.log(scores)
            + (1.0 - labels) * np.log(1.0 - scores)
        )
    )


def _tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_fpr: float = 0.315,
) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    index = int(np.argmin(np.abs(fpr - float(target_fpr))))
    return float(tpr[index])


def _metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    metrics = binary_metrics(labels, scores)
    return {
        "frames": int(labels.size),
        "speech": int(np.count_nonzero(labels == 1)),
        "silence": int(np.count_nonzero(labels == 0)),
        "f1": float(metrics["f1"]) if metrics["f1"] is not None else float("nan"),
        "auc": float(metrics["auc"]) if metrics["auc"] is not None else float("nan"),
        "proper_loss": _proper_loss(labels, scores),
        "tpr_at_fpr_0.315": _tpr_at_fpr(labels, scores),
    }


def _cluster_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    unique = sorted({str(value) for value in groups})
    if not unique:
        return float("nan"), float("nan"), 0
    indices = {
        name: np.flatnonzero(groups.astype(str) == name)
        for name in unique
    }

    def metric_fn(mask: np.ndarray) -> float:
        if metric == "f1":
            value = binary_metrics(labels[mask], scores[mask])["f1"]
            return float(value) if value is not None else float("nan")
        if metric == "auc":
            value = binary_metrics(labels[mask], scores[mask])["auc"]
            return float(value) if value is not None else float("nan")
        if metric == "proper_loss":
            return _proper_loss(labels[mask], scores[mask])
        if metric == "tpr_at_fpr_0.315":
            return _tpr_at_fpr(labels[mask], scores[mask])
        raise ValueError(f"unknown metric: {metric}")

    # With very few source clusters, enumerate the exact bootstrap
    # distribution instead of spending 2,000 expensive metric calls.
    if len(unique) <= 4:
        combinations: list[tuple[tuple[int, ...], int]] = []

        def enumerate_combinations(
            position: int,
            remaining: int,
            current: list[int],
        ) -> None:
            if position == len(unique) - 1:
                value = tuple(current + [remaining])
                multiplicity = math.factorial(len(unique))
                for count in value:
                    multiplicity //= math.factorial(int(count))
                combinations.append((value, int(multiplicity)))
                return
            for count in range(remaining + 1):
                enumerate_combinations(
                    position + 1,
                    remaining - count,
                    current + [count],
                )

        enumerate_combinations(0, len(unique), [])
        values: list[float] = []
        weights: list[int] = []
        for counts, multiplicity in combinations:
            pieces: list[np.ndarray] = []
            for name, count in zip(unique, counts):
                if count:
                    pieces.extend([indices[name]] * int(count))
            if not pieces:
                continue
            values.append(metric_fn(np.concatenate(pieces)))
            weights.append(int(multiplicity))
        finite = np.asarray(values, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        if finite.size == 0 or np.all(~np.isfinite(finite)):
            return float("nan"), float("nan"), len(unique)
        order = np.argsort(finite, kind="mergesort")
        finite = finite[order]
        weight_array = weight_array[order]
        cumulative = np.cumsum(weight_array)
        cumulative /= cumulative[-1]
        low = float(finite[int(np.searchsorted(cumulative, 0.025))])
        high = float(finite[min(int(np.searchsorted(cumulative, 0.975)), finite.size - 1)])
        return low, high, len(unique)

    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        chosen = rng.integers(0, len(unique), size=len(unique))
        pieces: list[np.ndarray] = []
        for index in chosen:
            pieces.append(indices[unique[int(index)]])
        values[repeat] = metric_fn(np.concatenate(pieces))
    return (
        float(np.nanquantile(values, 0.025)),
        float(np.nanquantile(values, 0.975)),
        len(unique),
    )


def _cluster_mean_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, int]:
    """Bootstrap a mean after resampling whole source clusters."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=object).reshape(-1)
    if values.shape != groups.shape:
        raise ValueError("values and groups must have equal shape")
    unique = sorted({str(value) for value in groups})
    if not unique:
        return float("nan"), float("nan"), 0
    indices = {
        name: np.flatnonzero(groups.astype(str) == name)
        for name in unique
    }

    def mean_fn(mask: np.ndarray) -> float:
        local = values[mask]
        if local.size == 0 or np.all(~np.isfinite(local)):
            return float("nan")
        return float(np.nanmean(local))

    if len(unique) <= 4:
        combinations: list[tuple[tuple[int, ...], int]] = []

        def enumerate_combinations(
            position: int,
            remaining: int,
            current: list[int],
        ) -> None:
            if position == len(unique) - 1:
                value = tuple(current + [remaining])
                multiplicity = math.factorial(len(unique))
                for count in value:
                    multiplicity //= math.factorial(int(count))
                combinations.append((value, int(multiplicity)))
                return
            for count in range(remaining + 1):
                enumerate_combinations(
                    position + 1,
                    remaining - count,
                    current + [count],
                )

        enumerate_combinations(0, len(unique), [])
        sampled: list[float] = []
        weights: list[int] = []
        for counts, multiplicity in combinations:
            pieces: list[np.ndarray] = []
            for name, count in zip(unique, counts):
                if count:
                    pieces.extend([indices[name]] * int(count))
            if not pieces:
                continue
            sampled.append(mean_fn(np.concatenate(pieces)))
            weights.append(int(multiplicity))
        finite = np.asarray(sampled, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(finite)
        if not np.any(valid):
            return float("nan"), float("nan"), len(unique)
        finite = finite[valid]
        weight_array = weight_array[valid]
        order = np.argsort(finite, kind="mergesort")
        finite = finite[order]
        weight_array = weight_array[order]
        cumulative = np.cumsum(weight_array)
        cumulative /= cumulative[-1]
        low = float(finite[int(np.searchsorted(cumulative, 0.025))])
        high = float(
            finite[
                min(
                    int(np.searchsorted(cumulative, 0.975)),
                    finite.size - 1,
                )
            ]
        )
        return low, high, len(unique)

    rng = np.random.default_rng(int(seed))
    sampled = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        chosen = rng.integers(0, len(unique), size=len(unique))
        pieces = [indices[unique[int(index)]] for index in chosen]
        sampled[repeat] = mean_fn(np.concatenate(pieces))
    return (
        float(np.nanquantile(sampled, 0.025)),
        float(np.nanquantile(sampled, 0.975)),
        len(unique),
    )


@dataclass(frozen=True)
class AvaVideo:
    video_id: str
    split: str
    audio_path: Path
    json_path: Path
    waveform: np.ndarray
    n_frames: int
    frame_labels: np.ndarray
    frame_conditions: np.ndarray
    source_key: str


def _discover_video_files(split: str) -> list[tuple[str, Path, Path]]:
    root = AVA_HF_ROOT / split
    result: list[tuple[str, Path, Path]] = []
    for audio_path in sorted(root.glob("*_clip.wav")):
        stem = audio_path.stem
        video_id = stem.removeprefix("human_").removesuffix("_clip")
        json_path = audio_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"missing AVA JSON for {audio_path}")
        result.append((video_id, audio_path, json_path))
    return result


def _read_label_intervals() -> dict[str, list[tuple[float, float, str]]]:
    intervals: dict[str, list[tuple[float, float, str]]] = {}
    with AVA_LABEL_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            video_id = str(row[0]).strip()
            try:
                start = float(row[1])
                end = float(row[2])
            except ValueError:
                continue
            label = str(row[3]).strip()
            intervals.setdefault(video_id, []).append((start, end, label))
    for values in intervals.values():
        values.sort(key=lambda item: (item[0], item[1], item[2]))
    return intervals


def _causal_frame_majority(
    sample_values: np.ndarray,
    n_frames: int,
    *,
    n_classes: int,
) -> np.ndarray:
    starts = np.arange(n_frames, dtype=np.int64) * FRAME_HOP - 512 + 56
    ends = starts + 400
    sample_values = np.asarray(sample_values, dtype=np.int64).reshape(-1)
    starts = np.clip(starts, 0, sample_values.size)
    ends = np.clip(ends, 0, sample_values.size)
    result = np.zeros(int(n_frames), dtype=np.int64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            continue
        local = sample_values[int(start) : int(end)]
        counts = np.bincount(local, minlength=int(n_classes))
        result[index] = int(np.argmax(counts))
    return result


def _build_video(
    *,
    video_id: str,
    split: str,
    audio_path: Path,
    json_path: Path,
    intervals: Mapping[str, Sequence[tuple[float, float, str]]],
) -> AvaVideo:
    waveform = read_int16_audio(audio_path)
    n_samples = int(waveform.size)
    n_frames = n_samples // FRAME_HOP + 1
    sample_class = np.zeros(n_samples, dtype=np.int64)
    class_map = {
        "NO_SPEECH": 0,
        "CLEAN_SPEECH": 1,
        "SPEECH_WITH_NOISE": 2,
        "SPEECH_WITH_MUSIC": 3,
    }
    for start, end, label in intervals.get(video_id, ()):
        local_start = int(round((float(start) - 900.0) * SAMPLE_RATE))
        local_end = int(round((float(end) - 900.0) * SAMPLE_RATE))
        local_start = max(0, min(n_samples, local_start))
        local_end = max(0, min(n_samples, local_end))
        if local_end <= local_start:
            continue
        sample_class[local_start:local_end] = int(class_map.get(label, 0))
    frame_labels = causal_frame_labels(sample_class > 0, n_frames)
    frame_conditions = _causal_frame_majority(
        sample_class,
        n_frames,
        n_classes=4,
    )
    if frame_labels.shape != frame_conditions.shape:
        raise RuntimeError("AVA frame labels and conditions are misaligned")
    return AvaVideo(
        video_id=str(video_id),
        split=str(split),
        audio_path=audio_path,
        json_path=json_path,
        waveform=waveform,
        n_frames=int(n_frames),
        frame_labels=frame_labels.astype(np.int64),
        frame_conditions=frame_conditions.astype(np.int64),
        source_key=str(video_id),
    )


def _load_videos(
    *,
    split: str,
    video_ids: Sequence[str] | None = None,
) -> list[AvaVideo]:
    intervals = _read_label_intervals()
    wanted = None if video_ids is None else set(str(value) for value in video_ids)
    videos: list[AvaVideo] = []
    for video_id, audio_path, json_path in _discover_video_files(split):
        if wanted is not None and video_id not in wanted:
            continue
        videos.append(
            _build_video(
                video_id=video_id,
                split=split,
                audio_path=audio_path,
                json_path=json_path,
                intervals=intervals,
            )
        )
    if wanted is not None:
        missing = sorted(wanted - {video.video_id for video in videos})
        if missing:
            raise FileNotFoundError(f"missing requested AVA videos: {missing}")
    return videos


def _build_frontend(device: torch.device) -> MfccFrontend:
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    frontend.eval()
    return frontend


def _load_short_long(
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    MfccFrontend,
    dict[str, Any],
    dict[str, Any],
]:
    short_model, short_payload, short_config = load_frame_model(
        SHORT_CHECKPOINT,
        device,
    )
    long_model, long_payload, long_config = load_frame_model(
        LONG_CHECKPOINT,
        device,
    )
    frontend = _build_frontend(device)
    return (
        short_model,
        long_model,
        frontend,
        short_payload,
        long_payload,
    )


def _count_macs_per_frame(
    model: nn.Module,
    *,
    dummy_frames: int = 256,
    divisor_frames: int | None = None,
) -> float:
    total_macs = 0
    handles = []

    def conv_hook(
        module: nn.Conv1d,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal total_macs
        input_tensor = inputs[0]
        kernel = int(module.kernel_size[0])
        in_channels_per_group = int(input_tensor.shape[1]) // int(module.groups)
        total_macs += int(output.numel()) * in_channels_per_group * kernel

    def linear_hook(
        module: nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal total_macs
        total_macs += int(output.numel()) * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        dummy = torch.zeros(1, 64, int(dummy_frames), device=model_device)
        with torch.inference_mode():
            if hasattr(model, "forward_stream"):
                model.forward_stream(dummy)
            else:
                model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    denominator = int(divisor_frames or dummy_frames)
    return float(total_macs) / float(denominator)


def _stream_cache_bytes(model: nn.Module) -> int:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    dummy = torch.zeros(1, 64, 8, device=model_device)
    states = model.init_stream_state(dummy)
    total = 0
    for block in states:
        for cache in block:
            if cache is not None:
                total += int(cache.numel()) * int(cache.element_size())
    return int(total)


def _benchmark_latency_ms(
    model: nn.Module,
    *,
    device: torch.device,
    frames: int = 256,
    repeats: int = 8,
    warmup: int = 2,
) -> float:
    dummy = torch.zeros(1, 64, int(frames), device=device)
    with torch.inference_mode():
        for _ in range(int(warmup)):
            if hasattr(model, "forward_stream"):
                model.forward_stream(dummy)
            else:
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        for _ in range(int(repeats)):
            if hasattr(model, "forward_stream"):
                model.forward_stream(dummy)
            else:
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - started
    return float(elapsed) / 1e6 / float(repeats) / float(frames)


def _benchmark_adaptive_latency_ms(
    model: AdaptiveCausalVAD,
    *,
    device: torch.device,
    activation_rate: float,
    frames: int = 256,
    repeats: int = 8,
    warmup: int = 2,
) -> float:
    """Estimate Short plus gate-conditioned refinement latency per frame."""
    if not hasattr(model, "forward_components"):
        raise TypeError("adaptive latency benchmark requires forward_components")
    dummy = torch.zeros(1, 64, int(frames), device=device)
    rate = float(np.clip(float(activation_rate), 0.0, 1.0))

    def measure(mask: torch.Tensor) -> float:
        with torch.inference_mode():
            for _ in range(int(warmup)):
                model.forward_components(dummy, activation_mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            for _ in range(int(repeats)):
                model.forward_components(dummy, activation_mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter_ns() - started
        return float(elapsed) / 1e6 / float(repeats) / float(frames)

    short_mask = torch.zeros(
        (1, int(frames)),
        dtype=torch.bool,
        device=device,
    )
    full_mask = torch.ones_like(short_mask)
    short_latency = measure(short_mask)
    full_latency = measure(full_mask)
    return float(
        short_latency
        + rate * max(full_latency - short_latency, 0.0)
    )


def _webrtc_scores(waveform: np.ndarray) -> np.ndarray:
    import webrtcvad

    vad = webrtcvad.Vad(3)
    n_frames = waveform.size // FRAME_HOP + 1
    scores = np.zeros(n_frames, dtype=np.float64)
    for index in range(n_frames):
        start = index * FRAME_HOP
        end = min(waveform.size, start + FRAME_HOP)
        segment = waveform[start:end]
        if segment.size < FRAME_HOP:
            segment = np.pad(segment, (0, FRAME_HOP - segment.size))
        scores[index] = 1.0 if vad.is_speech(
            segment.astype(np.int16).tobytes(),
            SAMPLE_RATE,
        ) else 0.0
    return scores


def _silero_scores(waveform: np.ndarray) -> tuple[np.ndarray, float]:
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=False)
    n_frames = waveform.size // FRAME_HOP + 1
    window = 512
    outputs: list[float] = []
    started = time.perf_counter_ns()
    with torch.inference_mode():
        for start in range(0, waveform.size, window):
            segment = waveform[start : start + window]
            if segment.size < window:
                segment = np.pad(segment, (0, window - segment.size))
            tensor = torch.from_numpy(
                segment.astype(np.float32) / 32768.0
            )
            outputs.append(float(model(tensor, SAMPLE_RATE).item()))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    scores = np.zeros(n_frames, dtype=np.float64)
    for index in range(n_frames):
        causal_end = index * FRAME_HOP + 88
        block = (causal_end // window) - 1
        if block < 0:
            scores[index] = 0.0
        else:
            scores[index] = outputs[min(block, len(outputs) - 1)]
    return scores, elapsed_ms / max(n_frames, 1)


def _nemo_scores(waveform: np.ndarray) -> tuple[np.ndarray, float]:
    import onnxruntime as ort

    preprocessor = ort.InferenceSession(
        str(AVA_NEMO_PREPROCESSOR),
        providers=["CPUExecutionProvider"],
    )
    model = ort.InferenceSession(
        str(AVA_NEMO_MODEL),
        providers=["CPUExecutionProvider"],
    )
    chunk_samples = 10 * SAMPLE_RATE
    feature_parts: list[np.ndarray] = []
    output_parts: list[np.ndarray] = []
    started = time.perf_counter_ns()
    for start in range(0, waveform.size, chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        audio = (
            chunk.astype(np.float32) / 32768.0
        )[None, :]
        length = np.asarray([audio.shape[1]], dtype=np.int64)
        features, feature_length = preprocessor.run(
            None,
            {"audio": audio, "audio_length": length},
        )
        output = model.run(
            None,
            {
                "processed_signal_length": feature_length,
                "processed_signal": features,
            },
        )[0]
        feature_parts.append(np.asarray(features))
        output_parts.append(np.asarray(output).reshape(-1))
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    output = np.concatenate(output_parts).astype(np.float64)
    n_frames = waveform.size // FRAME_HOP + 1
    scores = np.zeros(n_frames, dtype=np.float64)
    for index in range(n_frames):
        causal_end = index * FRAME_HOP + 88
        block = (causal_end // 320) - 1
        if block < 0:
            scores[index] = 0.0
        else:
            scores[index] = output[min(block, output.size - 1)]
    return scores, elapsed_ms / max(n_frames, 1)


def _extract_short_components(
    *,
    model: torch.nn.Module,
    frontend: MfccFrontend,
    waveform: np.ndarray,
    device: torch.device,
    chunk_frames: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(
        waveform.astype(np.float32) * INT16_SCALE
    ).unsqueeze(0)
    features = frontend(tensor.to(device, non_blocking=True))
    state = model.init_stream_state(features)
    encoded_parts: list[torch.Tensor] = []
    probability_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, features.shape[-1], chunk_frames):
            chunk = features[..., start : start + chunk_frames]
            encoded, state = model.encode_stream(chunk, state)
            logits = model.classifier(encoded)
            probabilities = torch.softmax(logits.transpose(1, 2), dim=-1)[..., 1]
            encoded_parts.append(encoded)
            probability_parts.append(probabilities)
    encoded = torch.cat(encoded_parts, dim=-1)
    probabilities = torch.cat(probability_parts, dim=-1)
    return (
        encoded[0].transpose(0, 1).detach().cpu().numpy().astype(np.float32),
        probabilities[0].detach().cpu().numpy().astype(np.float64),
        features[0].transpose(0, 1).detach().cpu().numpy().astype(np.float32),
    )


def _build_gbdt_features(
    *,
    model: torch.nn.Module,
    frontend: MfccFrontend,
    waveform: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hidden, short_scores, mfcc = _extract_short_components(
        model=model,
        frontend=frontend,
        waveform=waveform,
        device=device,
    )
    blocks = feature_blocks(
        short_scores,
        hidden,
        mfcc,
        [(0, short_scores.size)],
        window_frames=GBDT_WINDOW_FRAMES,
    )
    return np.asarray(blocks["X3"], dtype=np.float32), short_scores


def _train_gbdt(
    *,
    short_model: torch.nn.Module,
    frontend: MfccFrontend,
    device: torch.device,
    train_videos: Sequence[AvaVideo],
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    rng = np.random.default_rng(GBDT_SEED)
    features_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for position, video in enumerate(train_videos, start=1):
        features, _ = _build_gbdt_features(
            model=short_model,
            frontend=frontend,
            waveform=video.waveform,
            device=device,
        )
        labels = video.frame_labels
        speech = np.flatnonzero(labels == 1)
        silence = np.flatnonzero(labels == 0)
        per_class = max(1, GBDT_TRAIN_FRAMES_PER_VIDEO // 2)
        chosen: list[np.ndarray] = []
        if speech.size:
            chosen.append(
                rng.choice(
                    speech,
                    size=min(per_class, speech.size),
                    replace=False,
                )
            )
        if silence.size:
            chosen.append(
                rng.choice(
                    silence,
                    size=min(per_class, silence.size),
                    replace=False,
                )
            )
        if chosen:
            indices = np.concatenate(chosen)
            features_parts.append(features[indices])
            labels_parts.append(labels[indices])
        if position % 20 == 0 or position == len(train_videos):
            print(
                f"GBDT feature extraction {position}/{len(train_videos)}",
                flush=True,
            )
    if not features_parts:
        raise RuntimeError("no GBDT training frames were selected")
    features = np.concatenate(features_parts, axis=0)
    labels = np.concatenate(labels_parts).astype(np.int64)
    positive = int(np.count_nonzero(labels == 1))
    negative = int(labels.size - positive)
    weights = np.where(
        labels == 1,
        labels.size / max(2 * positive, 1),
        labels.size / max(2 * negative, 1),
    ).astype(np.float64)
    model = HistGradientBoostingClassifier(
        max_iter=GBDT_MAX_ITER,
        learning_rate=GBDT_LEARNING_RATE,
        max_leaf_nodes=GBDT_MAX_LEAF_NODES,
        min_samples_leaf=GBDT_MIN_SAMPLES_LEAF,
        l2_regularization=GBDT_L2_REGULARIZATION,
        early_stopping=False,
        random_state=GBDT_SEED,
    )
    model.fit(features, labels, sample_weight=weights)
    return model, {
        "train_frames": int(labels.size),
        "train_speech_frames": positive,
        "train_silence_frames": negative,
        "feature_dim": int(features.shape[1]),
    }


def _score_all_models(
    *,
    test_videos: Sequence[AvaVideo],
    short_model: torch.nn.Module,
    long_model: torch.nn.Module,
    adaptive_model: AdaptiveCausalVAD,
    frontend: MfccFrontend,
    device: torch.device,
    gbdt: HistGradientBoostingClassifier,
) -> dict[str, Any]:
    model_scores: dict[str, list[np.ndarray]] = {
        name: []
        for name in MODEL_ORDER
    }
    model_latency: dict[str, list[float]] = {
        name: []
        for name in MODEL_ORDER
    }
    adaptive_gate_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    condition_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for position, video in enumerate(test_videos, start=1):
        print(
            f"AVA scoring {position}/{len(test_videos)}: {video.video_id}",
            flush=True,
        )
        labels = video.frame_labels
        conditions = video.frame_conditions
        sources = np.full(labels.size, video.source_key, dtype=object)

        short_scores = predict_frames(
            short_model,
            frontend,
            video.waveform,
            device=device,
            chunk_frames=2048,
        )
        long_scores = predict_frames(
            long_model,
            frontend,
            video.waveform,
            device=device,
            chunk_frames=2048,
        )
        refined, embedded_short, _selected, adaptive_cache = (
            predict_full_adaptive_frames(
                adaptive_model,
                frontend,
                video.waveform,
                device=device,
                chunk_frames=2048,
            )
        )
        gate = np.abs(embedded_short - 0.5) <= GATE_THRESHOLD
        adaptive_scores = gated_scores(
            embedded_short,
            refined,
            gate,
        )
        adaptive_gate_parts.append(np.asarray(gate, dtype=bool).reshape(-1))
        always_refine = np.asarray(refined, dtype=np.float64)

        webrtc = _webrtc_scores(video.waveform)
        silero, silero_latency = _silero_scores(video.waveform)
        nemo, nemo_latency = _nemo_scores(video.waveform)
        gbdt_features, _ = _build_gbdt_features(
            model=short_model,
            frontend=frontend,
            waveform=video.waveform,
            device=device,
        )
        started = time.perf_counter_ns()
        gbdt_scores = gbdt.predict_proba(gbdt_features)[:, 1]
        gbdt_latency = (
            (time.perf_counter_ns() - started) / 1e6 / max(labels.size, 1)
        )

        values = {
            "Short": short_scores,
            "Long-RF": long_scores,
            "Adaptive": adaptive_scores,
            "AlwaysRefine": always_refine,
            "WebRTC VAD": webrtc,
            "Silero VAD": silero,
            "NeMo MarbleNet": nemo,
            "Shallow GBDT": gbdt_scores,
        }
        for name, scores in values.items():
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            if scores.size != labels.size:
                raise RuntimeError(
                    f"{name} returned {scores.size} scores for {labels.size} frames"
                )
            if not np.all(np.isfinite(scores)):
                raise RuntimeError(f"{name} produced non-finite scores")
            model_scores[name].append(scores)
        model_latency["Silero VAD"].append(float(silero_latency))
        model_latency["NeMo MarbleNet"].append(float(nemo_latency))
        model_latency["Shallow GBDT"].append(float(gbdt_latency))
        labels_parts.append(labels)
        condition_parts.append(conditions)
        source_parts.append(sources)

        del refined, embedded_short, adaptive_scores, always_refine
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = np.concatenate(labels_parts).astype(np.int64)
    conditions = np.concatenate(condition_parts).astype(np.int64)
    sources = np.concatenate(source_parts)
    scores = {
        name: np.concatenate(parts).astype(np.float64)
        for name, parts in model_scores.items()
    }
    latency = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in model_latency.items()
    }
    adaptive_gate = np.concatenate(adaptive_gate_parts).astype(bool)
    activation_rate = float(np.mean(adaptive_gate))
    latency["Adaptive"] = _benchmark_adaptive_latency_ms(
        adaptive_model,
        device=device,
        activation_rate=activation_rate,
    )
    latency["AlwaysRefine"] = _benchmark_adaptive_latency_ms(
        adaptive_model,
        device=device,
        activation_rate=1.0,
    )
    return {
        "labels": labels,
        "conditions": conditions,
        "sources": sources,
        "scores": scores,
        "adaptive_gate_mask": adaptive_gate,
        "latency_ms_per_frame": latency,
        "adaptive_cache_bytes": int(adaptive_cache),
    }


def _condition_mask(conditions: np.ndarray, condition: str) -> np.ndarray:
    if condition == "overall":
        return np.ones(conditions.size, dtype=bool)
    if condition == "clean_speech":
        return conditions == 1
    if condition == "speech_with_noise":
        return conditions == 2
    if condition == "speech_with_music":
        return conditions == 3
    if condition == "no_speech":
        return conditions == 0
    raise ValueError(f"unknown AVA condition: {condition}")


def _main_result_rows(
    *,
    labels: np.ndarray,
    conditions: np.ndarray,
    sources: np.ndarray,
    scores: Mapping[str, np.ndarray],
    latency: Mapping[str, float],
    short_macs: float,
    long_macs: float,
    adaptive_macs: float,
    always_refine_macs: float,
    short_cache: int,
    long_cache: int,
    adaptive_cache: int,
    always_refine_cache: int,
    silero_macs: float,
    bootstrap_repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cost = {
        "Short": (short_macs, short_cache, latency.get("Short", float("nan")), True),
        "Long-RF": (long_macs, long_cache, latency.get("Long-RF", float("nan")), True),
        "Adaptive": (adaptive_macs, adaptive_cache, latency.get("Adaptive", float("nan")), True),
        "AlwaysRefine": (always_refine_macs, always_refine_cache, latency.get("AlwaysRefine", float("nan")), True),
        "WebRTC VAD": (float("nan"), float("nan"), latency.get("WebRTC VAD", float("nan")), True),
        "Silero VAD": (silero_macs, float("nan"), latency.get("Silero VAD", float("nan")), True),
        "NeMo MarbleNet": (float("nan"), float("nan"), latency.get("NeMo MarbleNet", float("nan")), True),
        "Shallow GBDT": (float("nan"), float("nan"), latency.get("Shallow GBDT", float("nan")), True),
    }
    for model_name in MODEL_ORDER:
        model_scores = np.asarray(scores[model_name], dtype=np.float64)
        for condition in CONDITION_NAMES:
            mask = _condition_mask(conditions, condition)
            if not np.any(mask):
                continue
            local_labels = labels[mask]
            local_scores = model_scores[mask]
            local_sources = sources[mask]
            metrics = _metric_dict(local_labels, local_scores)
            row: dict[str, Any] = {
                "model": model_name,
                "condition": condition,
                "frames": int(metrics["frames"]),
                "speech_frames": int(metrics["speech"]),
                "silence_frames": int(metrics["silence"]),
                "source_clusters": int(len(set(str(value) for value in local_sources))),
                "f1": float(metrics["f1"]),
                "auc": float(metrics["auc"]),
                "proper_loss": float(metrics["proper_loss"]),
                "tpr_at_fpr_0.315": float(metrics["tpr_at_fpr_0.315"]),
                "macs_per_frame": cost[model_name][0],
                "streaming_cache_bytes": cost[model_name][1],
                "latency_ms_per_frame": cost[model_name][2],
                "causal_output": cost[model_name][3],
            }
            for metric_name in MAIN_METRIC_NAMES:
                low, high, clusters = _cluster_bootstrap_ci(
                    local_labels,
                    local_scores,
                    local_sources,
                    metric_name,
                    repeats=bootstrap_repeats,
                    seed=(
                        SOURCE_BOOTSTRAP_SEED
                        + 1000 * MODEL_ORDER.index(model_name)
                        + CONDITION_NAMES.index(condition)
                        + 17 * MAIN_METRIC_NAMES.index(metric_name)
                    ),
                )
                row[f"{metric_name}_ci95_low"] = low
                row[f"{metric_name}_ci95_high"] = high
            rows.append(row)
    return rows


def _hard_frame_mask(
    short_scores: np.ndarray,
    sources: np.ndarray,
    *,
    fraction: float = HARD_FRAME_FRACTION,
) -> np.ndarray:
    short_scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    sources = np.asarray(sources, dtype=object).reshape(-1)
    result = np.zeros(short_scores.size, dtype=bool)
    for source in sorted(set(str(value) for value in sources)):
        indices = np.flatnonzero(sources.astype(str) == source)
        if indices.size == 0:
            continue
        count = max(1, int(round(float(fraction) * indices.size)))
        uncertainty = np.abs(short_scores[indices] - 0.5)
        chosen = indices[np.argsort(uncertainty, kind="mergesort")[:count]]
        result[chosen] = True
    return result


def _target_indices(
    video: AvaVideo,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    valid = np.arange(SLICE_LENGTH - 1, video.n_frames, dtype=np.int64)
    labels = video.frame_labels[valid]
    rng = np.random.default_rng(int(seed))
    speech = valid[labels == 1]
    silence = valid[labels == 0]
    chosen: list[np.ndarray] = []
    per_class = max(1, int(count) // 2)
    for pool in (speech, silence):
        if pool.size:
            chosen.append(
                rng.choice(
                    pool,
                    size=min(per_class, pool.size),
                    replace=False,
                )
            )
    if not chosen:
        return np.empty(0, dtype=np.int64)
    result = np.concatenate(chosen)
    if result.size < int(count):
        remaining = np.setdiff1d(valid, result, assume_unique=False)
        if remaining.size:
            extra = rng.choice(
                remaining,
                size=min(int(count) - result.size, remaining.size),
                replace=False,
            )
            result = np.concatenate([result, extra])
    return np.sort(result.astype(np.int64))


def _remote_bounds(distance_frames: int) -> tuple[int, int]:
    if int(distance_frames) < 0 or int(distance_frames) >= LOOKBACK_FRAMES:
        raise ValueError("distance is outside the Long-RF lookback")
    return REMOTE_START, REMOTE_END - int(distance_frames)


def _order_permutation(
    *,
    seed: int,
    video_id: str,
    target_index: int,
    distance_frames: int,
) -> np.ndarray:
    start, end = _remote_bounds(distance_frames)
    length = end - start
    if length <= 1:
        return np.arange(length, dtype=np.int64)
    counter = 0
    while True:
        entropy = _stable_hash_int(
            PROTOCOL_ID,
            "AVA_REMOTE_ORDER",
            int(seed),
            str(video_id),
            int(target_index),
            int(distance_frames),
            int(counter),
        )
        permutation = np.random.default_rng(entropy).permutation(length)
        if not np.array_equal(permutation, np.arange(length)):
            return permutation.astype(np.int64)
        counter += 1
        if counter > 1000:
            raise RuntimeError("could not construct a non-identity permutation")


def _same_utterance_shift(
    *,
    seed: int,
    video_id: str,
    target_index: int,
) -> int:
    max_shift = max(0, int(target_index) - (SLICE_LENGTH - 1))
    return int(
        _stable_hash_int(
            PROTOCOL_ID,
            "AVA_SAME_UTTERANCE_SHIFT",
            int(seed),
            str(video_id),
            int(target_index),
        )
        % (max_shift + 1)
    )


def _same_utterance_indices(
    *,
    target_index: int,
    distance_frames: int,
    shift: int,
) -> np.ndarray:
    lags = np.arange(
        LOOKBACK_FRAMES,
        int(distance_frames),
        -1,
        dtype=np.int64,
    )
    donor_end = int(target_index) - LOOKBACK_FRAMES - int(shift)
    result = donor_end - lags
    if result.size == 0 or np.any(result < 0):
        raise RuntimeError("same-utterance donor is not causal")
    return result.astype(np.int64)


def _loss(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.clip(
        np.asarray(scores, dtype=np.float64),
        LOGLOSS_EPSILON,
        1.0 - LOGLOSS_EPSILON,
    )
    return (
        -labels * np.log(scores)
        - (1.0 - labels) * np.log(1.0 - scores)
    )


def _score_long_slice_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        logits = model(batch)
    probabilities = torch.softmax(logits.transpose(1, 2), dim=-1)[..., 1]
    return probabilities[:, -1].detach().cpu().numpy().astype(np.float64)


def _load_donor_features(
    *,
    donor: AvaVideo,
    frontend: MfccFrontend,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(
        donor.waveform.astype(np.float32) * INT16_SCALE
    ).unsqueeze(0)
    with torch.inference_mode():
        features = frontend(tensor.to(device, non_blocking=True))
    return features[0].detach()


def _choose_donor_end(
    *,
    donor: AvaVideo,
    donor_features: torch.Tensor,
    target_label: int,
    seed: int,
    video_id: str,
    target_index: int,
) -> int:
    valid = np.arange(SLICE_LENGTH - 1, donor.n_frames, dtype=np.int64)
    labels = donor.frame_labels[valid]
    candidates = valid[labels == int(target_label)]
    if candidates.size == 0:
        candidates = valid
    index = int(
        _stable_hash_int(
            PROTOCOL_ID,
            "AVA_DIFFERENT_SOURCE_DONOR",
            int(seed),
            str(video_id),
            int(target_index),
            int(target_label),
            donor.video_id,
        )
        % candidates.size
    )
    return int(candidates[index])


def _run_distance_sweep(
    *,
    test_videos: Sequence[AvaVideo],
    long_model: torch.nn.Module,
    frontend: MfccFrontend,
    device: torch.device,
    baseline_scores: Mapping[str, np.ndarray],
    train_videos: Sequence[AvaVideo],
    targets_per_video: int,
    seeds: Sequence[int],
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_cache: dict[str, torch.Tensor] = {}
    for video in test_videos:
        tensor = torch.from_numpy(
            video.waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        with torch.inference_mode():
            feature_cache[video.video_id] = frontend(
                tensor.to(device, non_blocking=True)
            )[0].detach()

    donor_cache: dict[str, tuple[AvaVideo, torch.Tensor]] = {}
    for target_video in test_videos:
        target_speech = float(np.mean(target_video.frame_labels))
        donor_candidates = sorted(
            train_videos,
            key=lambda donor: (
                abs(float(np.mean(donor.frame_labels)) - target_speech),
                donor.video_id,
            ),
        )
        if not donor_candidates:
            raise RuntimeError("AVA train split has no donor videos")
        donor = donor_candidates[0]
        if donor.video_id not in donor_cache:
            donor_cache[donor.video_id] = (
                donor,
                _load_donor_features(
                    donor=donor,
                    frontend=frontend,
                    device=device,
                ),
            )

    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for video_index, video in enumerate(test_videos):
        target_indices = _target_indices(
            video,
            count=int(targets_per_video),
            seed=SPLIT_SEED + 1000 + video_index,
        )
        target_labels = video.frame_labels[target_indices]
        target_baseline_loss = _loss(
            target_labels,
            _slice_long_scores_for_targets(
                baseline_scores["Long-RF"],
                test_videos,
                video_index,
                target_indices,
            ),
        )
        donor, donor_features = donor_cache[
            sorted(
                donor_cache,
                key=lambda key: (
                    abs(
                        float(np.mean(donor_cache[key][0].frame_labels))
                        - float(np.mean(video.frame_labels))
                    ),
                    key,
                ),
            )[0]
        ]
        features = feature_cache[video.video_id]
        for distance_index, distance_frames in enumerate(DISTANCE_FRAMES):
            remote_start, remote_end = _remote_bounds(distance_frames)
            remote_length = remote_end - remote_start
            lags = np.arange(
                LOOKBACK_FRAMES,
                int(distance_frames),
                -1,
                dtype=np.int64,
            )
            delta = np.empty(
                (
                    target_indices.size,
                    len(seeds),
                    3,
                ),
                dtype=np.float64,
            )
            for batch_start in range(
                0,
                target_indices.size,
                INTERVENTION_BATCH_SIZE,
            ):
                batch_indices = target_indices[
                    batch_start : batch_start + INTERVENTION_BATCH_SIZE
                ]
                batch_labels = video.frame_labels[batch_indices]
                batch_baseline_loss = target_baseline_loss[
                    batch_start : batch_start + batch_indices.size
                ]
                slices = []
                for target_index in batch_indices:
                    start = int(target_index) - (SLICE_LENGTH - 1)
                    slices.append(
                        features[:, start : start + SLICE_LENGTH]
                    )
                base = torch.stack(slices, dim=0)
                for seed_index, seed in enumerate(seeds):
                    # Order shuffle.
                    modified = base.clone()
                    for row, target_index in enumerate(batch_indices):
                        permutation = _order_permutation(
                            seed=int(seed),
                            video_id=video.video_id,
                            target_index=int(target_index),
                            distance_frames=int(distance_frames),
                        )
                        modified[
                            row,
                            :,
                            remote_start:remote_end,
                        ] = base[
                            row,
                            :,
                            torch.as_tensor(
                                remote_start + permutation,
                                dtype=torch.long,
                                device=base.device,
                            ),
                        ]
                    score = _score_long_slice_batch(long_model, modified)
                    delta[
                        batch_start : batch_start + batch_indices.size,
                        seed_index,
                        0,
                    ] = _loss(batch_labels, score) - batch_baseline_loss

                    # Same-utterance alternate time.
                    modified = base.clone()
                    for row, target_index in enumerate(batch_indices):
                        shift = _same_utterance_shift(
                            seed=int(seed),
                            video_id=video.video_id,
                            target_index=int(target_index),
                        )
                        donor_indices = _same_utterance_indices(
                            target_index=int(target_index),
                            distance_frames=int(distance_frames),
                            shift=int(shift),
                        )
                        modified[
                            row,
                            :,
                            remote_start:remote_end,
                        ] = features[
                            :,
                            torch.as_tensor(
                                donor_indices,
                                dtype=torch.long,
                                device=base.device,
                            ),
                        ]
                    score = _score_long_slice_batch(long_model, modified)
                    delta[
                        batch_start : batch_start + batch_indices.size,
                        seed_index,
                        1,
                    ] = _loss(batch_labels, score) - batch_baseline_loss

                    # Different-source replacement.
                    modified = base.clone()
                    for row, target_index in enumerate(batch_indices):
                        donor_end = _choose_donor_end(
                            donor=donor,
                            donor_features=donor_features,
                            target_label=int(video.frame_labels[target_index]),
                            seed=int(seed),
                            video_id=video.video_id,
                            target_index=int(target_index),
                        )
                        donor_indices = donor_end - lags
                        if np.any(donor_indices < 0):
                            raise RuntimeError("donor segment is not causal")
                        modified[
                            row,
                            :,
                            remote_start:remote_end,
                        ] = donor_features[
                            :,
                            torch.as_tensor(
                                donor_indices,
                                dtype=torch.long,
                                device=base.device,
                            ),
                        ]
                    score = _score_long_slice_batch(long_model, modified)
                    delta[
                        batch_start : batch_start + batch_indices.size,
                        seed_index,
                        2,
                    ] = _loss(batch_labels, score) - batch_baseline_loss
                if (
                    batch_start // INTERVENTION_BATCH_SIZE + 1
                ) % 8 == 0 or batch_start + INTERVENTION_BATCH_SIZE >= target_indices.size:
                    print(
                        f"distance sweep {video.video_id} "
                        f"d={distance_frames / 100.0:.2f}s "
                        f"{min(batch_start + INTERVENTION_BATCH_SIZE, target_indices.size)}/"
                        f"{target_indices.size}",
                        flush=True,
                    )
            condition_names = (
                "order_shuffle",
                "same_utterance_alternate",
                "different_source",
            )
            for condition_index, condition in enumerate(condition_names):
                per_target = delta[:, :, condition_index].mean(axis=1)
                source_keys = np.full(
                    per_target.size,
                    video.video_id,
                    dtype=object,
                )
                low, high, clusters = _cluster_mean_bootstrap_ci(
                    per_target,
                    source_keys,
                    repeats=bootstrap_repeats,
                    seed=(
                        SOURCE_BOOTSTRAP_SEED
                        + 100 * distance_index
                        + condition_index
                    ),
                )
                rows.append(
                    {
                        "analysis_type": "distance_sweep",
                        "distance_seconds": float(DISTANCE_SECONDS[distance_index]),
                        "distance_frames": int(distance_frames),
                        "condition": condition,
                        "stratum": "all",
                        "targets": int(per_target.size),
                        "source_clusters": int(clusters),
                        "mean_delta_logloss": float(np.mean(per_target)),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "baseline_logloss": float(np.mean(target_baseline_loss)),
                        "seeds": ",".join(str(value) for value in seeds),
                    }
                )
                raw_records.append(
                    {
                        "video_id": video.video_id,
                        "distance_seconds": float(DISTANCE_SECONDS[distance_index]),
                        "condition": condition,
                        "targets": int(per_target.size),
                        "mean_delta_logloss": float(np.mean(per_target)),
                    }
                )
    return rows, {"raw": raw_records}


def _slice_long_scores_for_targets(
    scores: np.ndarray,
    videos: Sequence[AvaVideo],
    video_index: int,
    target_indices: np.ndarray,
) -> np.ndarray:
    offset = sum(video.n_frames for video in videos[:video_index])
    return np.asarray(scores, dtype=np.float64)[offset + target_indices]


def _mechanism_hard_and_gate_rows(
    *,
    labels: np.ndarray,
    conditions: np.ndarray,
    sources: np.ndarray,
    scores: Mapping[str, np.ndarray],
    refined_scores: np.ndarray,
    embedded_short: np.ndarray,
    bootstrap_repeats: int,
) -> list[dict[str, Any]]:
    hard = _hard_frame_mask(scores["Short"], sources)
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        local = _metric_dict(labels[hard], scores[model_name][hard])
        rows.append(
            {
                "analysis_type": "hard_frame",
                "distance_seconds": float("nan"),
                "distance_frames": -1,
                "condition": "hard_frames_top20",
                "stratum": "all",
                "model": model_name,
                "targets": int(local["frames"]),
                "source_clusters": int(len(set(str(v) for v in sources[hard]))),
                "f1": float(local["f1"]),
                "auc": float(local["auc"]),
                "proper_loss": float(local["proper_loss"]),
                "tpr_at_fpr_0.315": float(local["tpr_at_fpr_0.315"]),
            }
        )
    short_loss = _proper_loss(labels[hard], scores["Short"][hard])
    long_loss = _proper_loss(labels[hard], scores["Long-RF"][hard])
    rows.append(
        {
            "analysis_type": "hard_frame_delta",
            "distance_seconds": float("nan"),
            "distance_frames": -1,
            "condition": "hard_frames_top20",
            "stratum": "all",
            "model": "Long-RF_minus_Short",
            "targets": int(np.count_nonzero(hard)),
            "source_clusters": int(len(set(str(v) for v in sources[hard]))),
            "delta_f1": float(
                binary_metrics(labels[hard], scores["Long-RF"][hard])["f1"]
                - binary_metrics(labels[hard], scores["Short"][hard])["f1"]
            ),
            "delta_proper_loss": float(long_loss - short_loss),
            "short_proper_loss": float(short_loss),
            "long_proper_loss": float(long_loss),
        }
    )

    gate = np.abs(embedded_short - 0.5) <= GATE_THRESHOLD
    gate_scores = gated_scores(embedded_short, refined_scores, gate)
    activation_rate = float(np.mean(gate))
    rng = np.random.default_rng(SOURCE_BOOTSTRAP_SEED + 77)
    random_f1: list[float] = []
    random_loss: list[float] = []
    for seed in RANDOM_GATE_SEEDS:
        random_gate = rng.random(gate.size) < activation_rate
        random_scores = gated_scores(
            embedded_short,
            refined_scores,
            random_gate,
        )
        random_f1.append(
            float(binary_metrics(labels, random_scores)["f1"])
        )
        random_loss.append(_proper_loss(labels, random_scores))
    gate_f1 = float(binary_metrics(labels, gate_scores)["f1"])
    gate_loss = _proper_loss(labels, gate_scores)
    rows.append(
        {
            "analysis_type": "gate_comparison",
            "distance_seconds": float("nan"),
            "distance_frames": -1,
            "condition": "uncertainty_gate_vs_random",
            "stratum": "all",
            "model": "Adaptive",
            "targets": int(labels.size),
            "source_clusters": int(len(set(str(v) for v in sources))),
            "activation_rate": activation_rate,
            "gate_f1": gate_f1,
            "random_f1_mean": float(np.mean(random_f1)),
            "random_f1_ci95_low": float(np.quantile(random_f1, 0.025)),
            "random_f1_ci95_high": float(np.quantile(random_f1, 0.975)),
            "delta_f1_gate_minus_random": float(
                gate_f1 - np.mean(random_f1)
            ),
            "gate_proper_loss": float(gate_loss),
            "random_proper_loss_mean": float(np.mean(random_loss)),
            "random_proper_loss_ci95_low": float(
                np.quantile(random_loss, 0.025)
            ),
            "random_proper_loss_ci95_high": float(
                np.quantile(random_loss, 0.975)
            ),
            "delta_proper_loss_gate_minus_random": float(
                gate_loss - np.mean(random_loss)
            ),
        }
    )
    return rows


def _train_lightweight_adapters(
    *,
    train_videos: Sequence[AvaVideo],
    test_videos: Sequence[AvaVideo],
    short_model: torch.nn.Module,
    frontend: MfccFrontend,
    device: torch.device,
    seeds: Sequence[int],
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(SPLIT_SEED + 404)
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for position, video in enumerate(train_videos, start=1):
        tensor = torch.from_numpy(
            video.waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        with torch.inference_mode():
            features = frontend(tensor.to(device, non_blocking=True))[0].detach()
        labels = torch.from_numpy(video.frame_labels.astype(np.int64))
        max_start = max(
            LIGHTWEIGHT_PREFIX_FRAMES,
            features.shape[-1] - LIGHTWEIGHT_CHUNK_FRAMES - 1,
        )
        for _ in range(LIGHTWEIGHT_CHUNKS_PER_VIDEO):
            start = int(rng.integers(LIGHTWEIGHT_PREFIX_FRAMES, max_start + 1))
            prefix_start = max(0, start - LIGHTWEIGHT_PREFIX_FRAMES)
            chunk_features = features[
                :,
                prefix_start : start + LIGHTWEIGHT_CHUNK_FRAMES,
            ].detach()
            chunk_labels = labels[
                start : start + LIGHTWEIGHT_CHUNK_FRAMES
            ]
            if chunk_features.shape[-1] < (
                LIGHTWEIGHT_PREFIX_FRAMES + LIGHTWEIGHT_CHUNK_FRAMES
            ):
                continue
            chunks.append((chunk_features, chunk_labels))
        if position % 20 == 0 or position == len(train_videos):
            print(
                f"lightweight chunk cache {position}/{len(train_videos)}",
                flush=True,
            )
    if not chunks:
        raise RuntimeError("no lightweight adapter chunks were built")

    test_features: list[torch.Tensor] = []
    test_labels: list[np.ndarray] = []
    test_conditions: list[np.ndarray] = []
    test_sources: list[np.ndarray] = []
    for video in test_videos:
        tensor = torch.from_numpy(
            video.waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        with torch.inference_mode():
            features = frontend(tensor.to(device, non_blocking=True))[0].detach()
        test_features.append(features)
        test_labels.append(video.frame_labels)
        test_conditions.append(video.frame_conditions)
        test_sources.append(
            np.full(video.n_frames, video.video_id, dtype=object)
        )
    test_short_scores_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for features in test_features:
            logits, _ = short_model.forward_stream(
                features.unsqueeze(0).to(device)
            )
            probabilities = torch.softmax(
                logits.transpose(1, 2),
                dim=-1,
            )[..., 1]
            test_short_scores_parts.append(
                probabilities[0].detach().cpu().numpy().astype(np.float64)
            )
    test_short_scores = np.concatenate(test_short_scores_parts)
    test_sources_array = np.concatenate(test_sources)
    hard = _hard_frame_mask(test_short_scores, test_sources_array)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _set_seed(int(seed))
        base = build_marblenet_3x2x64(
            feat_in=64,
            num_classes=2,
            dropout=0.0,
            causal=True,
            dilation_profile="short",
            frame_output=True,
        )
        base.load_state_dict(short_model.state_dict())
        adapter = LongTermStatsFiLM(
            base,
            time_constant_seconds=LIGHTWEIGHT_TAU_SECONDS,
            hidden_dim=LIGHTWEIGHT_HIDDEN_DIM,
        ).to(device)
        adapter._freeze_base()
        optimizer = torch.optim.AdamW(
            adapter.film.parameters(),
            lr=LIGHTWEIGHT_LR,
            weight_decay=LIGHTWEIGHT_WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()
        order = np.arange(len(chunks))
        started = time.time()
        for epoch in range(LIGHTWEIGHT_EPOCHS):
            rng_epoch = np.random.default_rng(int(seed) + 1000 + epoch)
            rng_epoch.shuffle(order)
            losses: list[float] = []
            for batch_start in range(0, len(order), LIGHTWEIGHT_BATCH_SIZE):
                batch_ids = order[
                    batch_start : batch_start + LIGHTWEIGHT_BATCH_SIZE
                ]
                batch_features = torch.stack(
                    [chunks[int(index)][0] for index in batch_ids],
                    dim=0,
                ).to(device)
                batch_labels = torch.stack(
                    [chunks[int(index)][1] for index in batch_ids],
                    dim=0,
                ).to(device)
                logits, _ = adapter.forward_stream(batch_features)
                supervised = logits[
                    :,
                    :,
                    -LIGHTWEIGHT_CHUNK_FRAMES:,
                ]
                supervised_labels = batch_labels[
                    :,
                    -LIGHTWEIGHT_CHUNK_FRAMES:,
                ]
                loss = criterion(supervised, supervised_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            print(
                f"lightweight seed={seed} epoch={epoch + 1}/"
                f"{LIGHTWEIGHT_EPOCHS} loss={np.mean(losses):.5f}",
                flush=True,
            )
        adapter.eval()
        scores_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for features in test_features:
                logits, _ = adapter.forward_stream(
                    features.unsqueeze(0).to(device)
                )
                probabilities = torch.softmax(
                    logits.transpose(1, 2),
                    dim=-1,
                )[..., 1]
                scores_parts.append(
                    probabilities[0].detach().cpu().numpy().astype(np.float64)
                )
        scores = np.concatenate(scores_parts)
        labels = np.concatenate(test_labels).astype(np.int64)
        conditions = np.concatenate(test_conditions).astype(np.int64)
        sources = np.concatenate(test_sources)
        for condition in CONDITION_NAMES:
            mask = _condition_mask(conditions, condition)
            if not np.any(mask):
                continue
            metrics = _metric_dict(labels[mask], scores[mask])
            low, high, clusters = _cluster_bootstrap_ci(
                labels[mask],
                scores[mask],
                sources[mask],
                "f1",
                repeats=bootstrap_repeats,
                seed=SOURCE_BOOTSTRAP_SEED + 9000 + int(seed),
            )
            rows.append(
                {
                    "model": "Short+MFCC EMA FiLM",
                    "seed": int(seed),
                    "condition": condition,
                    "stratum": "all",
                    "frames": int(metrics["frames"]),
                    "source_clusters": int(clusters),
                    "f1": float(metrics["f1"]),
                    "f1_ci95_low": float(low),
                    "f1_ci95_high": float(high),
                    "auc": float(metrics["auc"]),
                    "proper_loss": float(metrics["proper_loss"]),
                    "tpr_at_fpr_0.315": float(metrics["tpr_at_fpr_0.315"]),
                    "added_cache_bytes": int(LIGHTWEIGHT_ADDED_CACHE_BYTES),
                    "long_rf_cache_bytes": int(_stream_cache_bytes(short_model) * 0),
                }
            )
        hard_metrics = _metric_dict(labels[hard], scores[hard])
        rows.append(
            {
                "model": "Short+MFCC EMA FiLM",
                "seed": int(seed),
                "condition": "hard_frames_top20",
                "stratum": "all",
                "frames": int(hard_metrics["frames"]),
                "source_clusters": int(len(set(str(v) for v in sources[hard]))),
                "f1": float(hard_metrics["f1"]),
                "auc": float(hard_metrics["auc"]),
                "proper_loss": float(hard_metrics["proper_loss"]),
                "tpr_at_fpr_0.315": float(hard_metrics["tpr_at_fpr_0.315"]),
                "added_cache_bytes": int(LIGHTWEIGHT_ADDED_CACHE_BYTES),
                "long_rf_cache_bytes": 0,
                "elapsed_seconds": float(time.time() - started),
            }
        )
    return rows, {
        "chunks": int(len(chunks)),
        "train_videos": int(len(train_videos)),
        "test_videos": int(len(test_videos)),
    }


def _build_protocol(output_root: Path) -> dict[str, Any]:
    train_files = _discover_video_files("train")
    test_files = _discover_video_files("test")
    train_ids = sorted(video_id for video_id, _, _ in train_files)
    test_ids = sorted(video_id for video_id, _, _ in test_files)
    if len(test_ids) != 2:
        raise RuntimeError(
            f"expected two official AVA test videos, found {len(test_ids)}"
        )
    rng = np.random.default_rng(SPLIT_SEED)
    permutation = rng.permutation(len(train_ids))
    validation_count = int(round(VALIDATION_FRACTION * len(train_ids)))
    validation_ids = sorted(
        train_ids[int(index)] for index in permutation[:validation_count]
    )
    fit_ids = sorted(set(train_ids) - set(validation_ids))
    records = {
        "label_csv": _file_record(AVA_LABEL_CSV),
        "short_checkpoint": _file_record(SHORT_CHECKPOINT),
        "long_checkpoint": _file_record(LONG_CHECKPOINT),
        "adaptive_checkpoint": _file_record(ADAPTIVE_CHECKPOINT),
        "nemo_preprocessor": _file_record(AVA_NEMO_PREPROCESSOR),
        "nemo_model": _file_record(AVA_NEMO_MODEL),
    }
    audio_records: list[dict[str, Any]] = []
    for split, files in (("train", train_files), ("test", test_files)):
        for video_id, audio_path, _ in files:
            record = _file_record(audio_path)
            record["video_id"] = video_id
            record["split"] = split
            audio_records.append(record)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "platform": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "root": _relative(AVA_ROOT),
            "official_labels": _relative(AVA_LABEL_CSV),
            "label_mapping": {
                "CLEAN_SPEECH": 1,
                "SPEECH_WITH_NOISE": 2,
                "SPEECH_WITH_MUSIC": 3,
                "NO_SPEECH": 0,
            },
            "official_label_epoch_seconds": 900.0,
            "train_video_ids": train_ids,
            "fit_video_ids": fit_ids,
            "validation_video_ids": validation_ids,
            "test_video_ids": test_ids,
            "split_seed": int(SPLIT_SEED),
            "validation_fraction": float(VALIDATION_FRACTION),
            "audio_files": audio_records,
            "source_cluster": "video_id",
            "speaker_disjoint_claim": False,
        },
        "models": {
            "short": {
                "checkpoint": _relative(SHORT_CHECKPOINT),
                "role": "frozen Short-RF reference",
            },
            "long_rf": {
                "checkpoint": _relative(LONG_CHECKPOINT),
                "role": "frozen Long-RF reference",
            },
            "adaptive": {
                "checkpoint": _relative(ADAPTIVE_CHECKPOINT),
                "gate_threshold": float(GATE_THRESHOLD),
                "gate_definition": "abs(P_short - 0.5) <= threshold",
                "refinement": {
                    "in_channels": 128,
                    "kernel_size": 5,
                    "dilations": [1, 2, 4, 8, 96],
                    "max_residual": 2.0,
                },
            },
            "always_refine": {
                "checkpoint": _relative(ADAPTIVE_CHECKPOINT),
                "gate_threshold": 0.5,
            },
        },
        "metrics": {
            "names": list(MAIN_METRIC_NAMES),
            "decision_threshold": 0.5,
            "tpr_fpr_target": 0.315,
            "source_bootstrap_repeats": int(SOURCE_BOOTSTRAP_REPEATS),
            "source_bootstrap_seed": int(SOURCE_BOOTSTRAP_SEED),
            "conditions": list(CONDITION_NAMES),
            "hard_frame_fraction": float(HARD_FRAME_FRACTION),
            "hard_frame_definition": (
                "top fraction of frames by smallest abs(P_short - 0.5), "
                "selected separately within each test video"
            ),
        },
        "mechanism": {
            "slice_length_frames": int(SLICE_LENGTH),
            "lookback_frames": int(LOOKBACK_FRAMES),
            "remote_definition": "lags > d, i.e. older than distance d",
            "distance_seconds": list(DISTANCE_SECONDS),
            "distance_frames": list(DISTANCE_FRAMES),
            "intervention_seeds": list(INTERVENTION_SEEDS),
            "targets_per_video": int(INTERVENTION_TARGETS_PER_VIDEO),
            "conditions": [
                "order_shuffle",
                "same_utterance_alternate",
                "different_source",
            ],
            "different_source_donor": (
                "train-split video with closest overall speech fraction; "
                "donor frame matched to target binary label"
            ),
        },
        "baselines": {
            "webrtc": {
                "mode": 3,
                "frame_ms": 10,
                "mapping": "one 10 ms frame per MFCC grid frame",
            },
            "silero": {
                "window_samples": 512,
                "window_ms": 32,
                "reset_per_recording": True,
                "mapping": "last completed 512-sample window at each 10 ms grid frame",
            },
            "nemo_marblenet": {
                "preprocessor": _relative(AVA_NEMO_PREPROCESSOR),
                "model": _relative(AVA_NEMO_MODEL),
                "chunk_seconds": 10,
                "output_hop_ms": 20,
                "mapping": "last completed 20 ms output at each 10 ms grid frame",
            },
            "gbdt": {
                "seed": int(GBDT_SEED),
                "max_iter": int(GBDT_MAX_ITER),
                "learning_rate": float(GBDT_LEARNING_RATE),
                "max_leaf_nodes": int(GBDT_MAX_LEAF_NODES),
                "min_samples_leaf": int(GBDT_MIN_SAMPLES_LEAF),
                "l2_regularization": float(GBDT_L2_REGULARIZATION),
                "train_frames_per_video": int(GBDT_TRAIN_FRAMES_PER_VIDEO),
                "feature_window_frames": int(GBDT_WINDOW_FRAMES),
                "feature_definition": (
                    "X1 posterior temporal features + 128-dimensional "
                    "Short hidden state + causal MFCC current/trailing "
                    "mean/delta (X3, 330 features)"
                ),
            },
        },
        "lightweight": {
            "name": "Short+MFCC EMA FiLM",
            "time_constant_seconds": float(LIGHTWEIGHT_TAU_SECONDS),
            "hidden_dim": int(LIGHTWEIGHT_HIDDEN_DIM),
            "seeds": list(LIGHTWEIGHT_SEEDS),
            "epochs": int(LIGHTWEIGHT_EPOCHS),
            "batch_size": int(LIGHTWEIGHT_BATCH_SIZE),
            "learning_rate": float(LIGHTWEIGHT_LR),
            "weight_decay": float(LIGHTWEIGHT_WEIGHT_DECAY),
            "chunks_per_video": int(LIGHTWEIGHT_CHUNKS_PER_VIDEO),
            "chunk_frames": int(LIGHTWEIGHT_CHUNK_FRAMES),
            "prefix_frames": int(LIGHTWEIGHT_PREFIX_FRAMES),
            "added_cache_bytes": int(LIGHTWEIGHT_ADDED_CACHE_BYTES),
            "long_rf_reference_cache_bytes": 104 * 1024,
        },
        "input_records": records,
        "output_root": _relative(output_root),
        "output_files": [
            "ava_speech_main_results.csv",
            "ava_speech_mechanism_results.csv",
            "ava_speech_lightweight_results.csv",
            "ava_speech_report.md",
        ],
        "forbidden_inputs": ["NEW_FINAL_OOD", "FINAL_OOD"],
    }


def _freeze(output_root: Path, *, force: bool = False) -> int:
    protocol_path = output_root / "ava_speech_protocol.json"
    hash_path = output_root / "ava_speech_protocol_sha256.txt"
    if protocol_path.exists() and not force:
        raise FileExistsError(
            f"protocol already exists: {protocol_path}; use --force to replace"
        )
    protocol = _build_protocol(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(protocol_path, protocol)
    digest = _sha256_file(protocol_path)
    hash_path.write_text(digest + "\n", encoding="ascii")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_path": _relative(protocol_path),
        "protocol_sha256": digest,
        "frozen_at_utc": _utc_now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": _sha256_file(Path(__file__)),
        "output_files": list(protocol["output_files"]),
    }
    _write_json(output_root / "ava_speech_execution_manifest.json", manifest)
    print(f"frozen protocol: {protocol_path}", flush=True)
    print(f"protocol sha256: {digest}", flush=True)
    return 0


def _validate_protocol(output_root: Path) -> dict[str, Any]:
    protocol_path = output_root / "ava_speech_protocol.json"
    hash_path = output_root / "ava_speech_protocol_sha256.txt"
    if not protocol_path.exists() or not hash_path.exists():
        raise FileNotFoundError("AVA protocol is not frozen")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = hash_path.read_text(encoding="ascii").strip()
    actual = _sha256_file(protocol_path)
    if actual != expected:
        raise RuntimeError(
            f"protocol hash mismatch: expected {expected}, got {actual}"
        )
    for name, record in protocol["input_records"].items():
        path = REPO_ROOT / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"frozen input changed: {name}")
    for record in protocol["data"]["audio_files"]:
        path = REPO_ROOT / str(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"frozen audio changed: {path}")
    return protocol


def _run(
    output_root: Path,
    *,
    smoke: bool = False,
    force: bool = False,
) -> int:
    protocol = _validate_protocol(output_root)
    device = resolve_device("auto")
    _set_seed(SPLIT_SEED)
    if smoke:
        test_ids = protocol["data"]["test_video_ids"][:1]
        fit_ids = protocol["data"]["fit_video_ids"][:2]
        bootstrap_repeats = 50
        mechanism_targets = 32
        mechanism_seeds = (17,)
        lightweight_seeds = (17,)
        lightweight_epochs = 1
    else:
        test_ids = protocol["data"]["test_video_ids"]
        fit_ids = protocol["data"]["fit_video_ids"]
        bootstrap_repeats = int(protocol["metrics"]["source_bootstrap_repeats"])
        mechanism_targets = int(protocol["mechanism"]["targets_per_video"])
        mechanism_seeds = tuple(protocol["mechanism"]["intervention_seeds"])
        lightweight_seeds = tuple(protocol["lightweight"]["seeds"])
        lightweight_epochs = int(protocol["lightweight"]["epochs"])

    print(f"loading AVA test videos: {test_ids}", flush=True)
    test_videos = _load_videos(split="test", video_ids=test_ids)
    print(
        f"loading AVA fit videos: {len(fit_ids)} videos",
        flush=True,
    )
    train_videos = _load_videos(split="train", video_ids=fit_ids)
    short_model, long_model, frontend, _short_payload, _long_payload = (
        _load_short_long(device)
    )
    adaptive_model, _adaptive_payload, _refinement_config = (
        load_adaptive_model(ADAPTIVE_CHECKPOINT, device)
    )

    print("training shallow GBDT baseline", flush=True)
    gbdt, gbdt_info = _train_gbdt(
        short_model=short_model,
        frontend=frontend,
        device=device,
        train_videos=train_videos,
    )
    print("scoring AVA test recordings", flush=True)
    scoring = _score_all_models(
        test_videos=test_videos,
        short_model=short_model,
        long_model=long_model,
        adaptive_model=adaptive_model,
        frontend=frontend,
        device=device,
        gbdt=gbdt,
    )
    labels = scoring["labels"]
    conditions = scoring["conditions"]
    sources = scoring["sources"]
    scores = scoring["scores"]

    short_macs = _count_macs_per_frame(short_model)
    long_macs = _count_macs_per_frame(long_model)
    refinement_macs = float(
        adaptive_model.refinement.estimated_macs_per_selected_frame()
    )
    adaptive_macs = short_macs + float(
        np.mean(scoring["adaptive_gate_mask"])
    ) * refinement_macs
    always_refine_macs = short_macs + refinement_macs
    short_cache = _stream_cache_bytes(short_model)
    long_cache = _stream_cache_bytes(long_model)
    adaptive_cache = int(scoring["adaptive_cache_bytes"])
    always_refine_cache = adaptive_cache
    silero_macs = 0.0
    try:
        from silero_vad import load_silero_vad

        silero_model = load_silero_vad(onnx=False)
        silero_macs = _count_macs_per_frame(
            silero_model,
            dummy_frames=512,
            divisor_frames=32,
        )
    except Exception:
        silero_macs = float("nan")

    latency = dict(scoring["latency_ms_per_frame"])
    latency["Short"] = _benchmark_latency_ms(short_model, device=device)
    latency["Long-RF"] = _benchmark_latency_ms(long_model, device=device)
    latency["WebRTC VAD"] = float("nan")

    main_rows = _main_result_rows(
        labels=labels,
        conditions=conditions,
        sources=sources,
        scores=scores,
        latency=latency,
        short_macs=short_macs,
        long_macs=long_macs,
        adaptive_macs=adaptive_macs,
        always_refine_macs=always_refine_macs,
        short_cache=short_cache,
        long_cache=long_cache,
        adaptive_cache=adaptive_cache,
        always_refine_cache=always_refine_cache,
        silero_macs=silero_macs,
        bootstrap_repeats=bootstrap_repeats,
    )
    _write_csv(output_root / "ava_speech_main_results.csv", main_rows)

    print("running AVA distance intervention sweep", flush=True)
    mechanism_rows, _mechanism_raw = _run_distance_sweep(
        test_videos=test_videos,
        long_model=long_model,
        frontend=frontend,
        device=device,
        baseline_scores=scores,
        train_videos=train_videos,
        targets_per_video=mechanism_targets,
        seeds=mechanism_seeds,
        bootstrap_repeats=bootstrap_repeats,
    )
    # Recover the full adaptive outputs for the gate comparison.  The
    # scoring pass keeps only final scores, so recompute the needed streams.
    refined_parts: list[np.ndarray] = []
    embedded_parts: list[np.ndarray] = []
    for video in test_videos:
        refined, embedded, _selected, _cache = predict_full_adaptive_frames(
            adaptive_model,
            frontend,
            video.waveform,
            device=device,
            chunk_frames=2048,
        )
        refined_parts.append(refined)
        embedded_parts.append(embedded)
    refined_scores = np.concatenate(refined_parts).astype(np.float64)
    embedded_short = np.concatenate(embedded_parts).astype(np.float64)
    mechanism_rows.extend(
        _mechanism_hard_and_gate_rows(
            labels=labels,
            conditions=conditions,
            sources=sources,
            scores=scores,
            refined_scores=refined_scores,
            embedded_short=embedded_short,
            bootstrap_repeats=bootstrap_repeats,
        )
    )
    _write_csv(
        output_root / "ava_speech_mechanism_results.csv",
        mechanism_rows,
    )

    print("training lightweight MFCC EMA adapters", flush=True)
    original_epochs = LIGHTWEIGHT_EPOCHS
    try:
        globals()["LIGHTWEIGHT_EPOCHS"] = int(lightweight_epochs)
        lightweight_rows, lightweight_info = _train_lightweight_adapters(
            train_videos=train_videos,
            test_videos=test_videos,
            short_model=short_model,
            frontend=frontend,
            device=device,
            seeds=lightweight_seeds,
            bootstrap_repeats=bootstrap_repeats,
        )
    finally:
        globals()["LIGHTWEIGHT_EPOCHS"] = original_epochs
    _write_csv(
        output_root / "ava_speech_lightweight_results.csv",
        lightweight_rows,
    )

    report = _build_report(
        protocol=protocol,
        test_videos=test_videos,
        main_rows=main_rows,
        mechanism_rows=mechanism_rows,
        lightweight_rows=lightweight_rows,
        scoring=scoring,
        gbdt_info=gbdt_info,
        lightweight_info=lightweight_info,
        short_cache=short_cache,
        long_cache=long_cache,
        adaptive_cache=adaptive_cache,
    )
    (output_root / "ava_speech_report.md").write_text(
        report,
        encoding="utf-8",
    )
    manifest_path = output_root / "ava_speech_execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed_at_utc": _utc_now(),
            "status": "COMPLETED",
            "test_video_ids": list(test_ids),
            "fit_video_count": len(fit_ids),
            "main_rows": len(main_rows),
            "mechanism_rows": len(mechanism_rows),
            "lightweight_rows": len(lightweight_rows),
        }
    )
    _write_json(manifest_path, manifest)
    print("AVA-Speech run complete", flush=True)
    return 0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _build_report(
    *,
    protocol: Mapping[str, Any],
    test_videos: Sequence[AvaVideo],
    main_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    lightweight_rows: Sequence[Mapping[str, Any]],
    scoring: Mapping[str, Any],
    gbdt_info: Mapping[str, Any],
    lightweight_info: Mapping[str, Any],
    short_cache: int,
    long_cache: int,
    adaptive_cache: int,
) -> str:
    lines = [
        "# AVA-Speech Real-Recording Validation",
        "",
        "## Protocol",
        "",
        f"- Protocol: `{protocol['protocol_id']}`",
        f"- Status: `{protocol['protocol_status']}`",
        f"- Official test videos: {', '.join(protocol['data']['test_video_ids'])}",
        f"- Fit videos: {len(protocol['data']['fit_video_ids'])}",
        f"- Validation videos: {len(protocol['data']['validation_video_ids'])}",
        f"- Source cluster: `video_id`",
        "- Speaker-disjoint evaluation was not proven and is not claimed.",
        "- `NEW_FINAL_OOD` and `FINAL_OOD` were not opened.",
        "",
        "## Main Results",
        "",
        "| Model | Condition | F1 | AUC | Proper loss | TPR@FPR=0.315 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        lines.append(
            "| {model} | {condition} | {f1} | {auc} | {loss} | {tpr} |".format(
                model=row["model"],
                condition=row["condition"],
                f1=_fmt(row.get("f1")),
                auc=_fmt(row.get("auc")),
                loss=_fmt(row.get("proper_loss")),
                tpr=_fmt(row.get("tpr_at_fpr_0.315")),
            )
        )
    lines.extend(
        [
            "",
            "### Cost",
            "",
            "| Model | MACs/frame | Streaming cache (bytes) | Latency (ms/frame) | Causal |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    seen: set[str] = set()
    for row in main_rows:
        if row["model"] in seen:
            continue
        seen.add(row["model"])
        lines.append(
            "| {model} | {macs} | {cache} | {latency} | {causal} |".format(
                model=row["model"],
                macs=_fmt(row.get("macs_per_frame"), 1),
                cache=(
                    "n/a"
                    if row.get("streaming_cache_bytes") is None
                    or not np.isfinite(float(row.get("streaming_cache_bytes", np.nan)))
                    else f"{int(row['streaming_cache_bytes']):,}"
                ),
                latency=_fmt(row.get("latency_ms_per_frame")),
                causal="yes" if row.get("causal_output") else "no",
            )
        )
    lines.extend(
        [
            "",
            f"- Short cache: {short_cache:,} bytes.",
            f"- Long-RF cache: {long_cache:,} bytes.",
            f"- Adaptive cache: {adaptive_cache:,} bytes.",
            f"- GBDT training frames: {gbdt_info.get('train_frames', 0):,}.",
            "",
            "## Mechanism",
            "",
            "### Distance Sweep",
            "",
            "| Distance (s) | Condition | Mean delta log-loss | 95% CI | Targets |",
            "| ---: | --- | ---: | --- | ---: |",
        ]
    )
    for row in mechanism_rows:
        if row.get("analysis_type") != "distance_sweep":
            continue
        lines.append(
            "| {distance} | {condition} | {mean} | [{low}, {high}] | {targets} |".format(
                distance=_fmt(row.get("distance_seconds"), 2),
                condition=row.get("condition"),
                mean=_fmt(row.get("mean_delta_logloss")),
                low=_fmt(row.get("ci95_low")),
                high=_fmt(row.get("ci95_high")),
                targets=row.get("targets", 0),
            )
        )
    lines.extend(
        [
            "",
            "### Hard Frames and Gate",
            "",
            "| Analysis | Model/Condition | Value |",
            "| --- | --- | ---: |",
        ]
    )
    for row in mechanism_rows:
        if row.get("analysis_type") == "hard_frame":
            lines.append(
                "| hard_frame | {model} | F1={f1}, loss={loss} |".format(
                    model=row.get("model"),
                    f1=_fmt(row.get("f1")),
                    loss=_fmt(row.get("proper_loss")),
                )
            )
        elif row.get("analysis_type") == "hard_frame_delta":
            lines.append(
                "| hard_frame_delta | Long-RF minus Short | "
                f"delta F1={_fmt(row.get('delta_f1'))}, "
                f"delta loss={_fmt(row.get('delta_proper_loss'))} |"
            )
        elif row.get("analysis_type") == "gate_comparison":
            lines.append(
                "| gate_comparison | Adaptive uncertainty gate minus random | "
                f"delta F1={_fmt(row.get('delta_f1_gate_minus_random'))}, "
                f"delta loss={_fmt(row.get('delta_proper_loss_gate_minus_random'))}, "
                f"activation={_fmt(row.get('activation_rate'))} |"
            )
    lines.extend(
        [
            "",
            "## Lightweight Alternative",
            "",
            "| Seed | Condition | F1 | AUC | Proper loss | Added cache (bytes) |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lightweight_rows:
        lines.append(
            "| {seed} | {condition} | {f1} | {auc} | {loss} | {cache} |".format(
                seed=row.get("seed"),
                condition=row.get("condition"),
                f1=_fmt(row.get("f1")),
                auc=_fmt(row.get("auc")),
                loss=_fmt(row.get("proper_loss")),
                cache=row.get("added_cache_bytes", "n/a"),
            )
        )
    lines.extend(
        [
            "",
            f"- Lightweight chunks: {lightweight_info.get('chunks', 0):,}.",
            f"- Lightweight added cache: {LIGHTWEIGHT_ADDED_CACHE_BYTES:,} bytes.",
            f"- Long-RF reference cache: {long_cache:,} bytes.",
            "",
            "## Interpretation Guardrails",
            "",
            "- This report is an external-distribution validation, not a "
            "claim that Adaptive/M2 is superior.",
            "- The frozen gate threshold was not tuned on AVA-Speech.",
            "- Results are reported with source-cluster bootstrap intervals; "
            "with only two official test videos, source-level uncertainty is "
            "necessarily coarse.",
        ]
    )
    return "\n".join(lines) + "\n"


def _verify(output_root: Path) -> int:
    required = (
        "ava_speech_protocol.json",
        "ava_speech_protocol_sha256.txt",
        "ava_speech_execution_manifest.json",
        "ava_speech_main_results.csv",
        "ava_speech_mechanism_results.csv",
        "ava_speech_lightweight_results.csv",
        "ava_speech_report.md",
    )
    for name in required:
        path = output_root / name
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty output: {path}")
    protocol = _validate_protocol(output_root)
    main = pd.read_csv(output_root / "ava_speech_main_results.csv")
    mechanism = pd.read_csv(output_root / "ava_speech_mechanism_results.csv")
    lightweight = pd.read_csv(output_root / "ava_speech_lightweight_results.csv")
    if main.empty or mechanism.empty or lightweight.empty:
        raise RuntimeError("one or more AVA result tables are empty")
    for frame in (main, mechanism, lightweight):
        if any("NEW_FINAL_OOD" in str(column) for column in frame.columns):
            raise RuntimeError("forbidden input name leaked into output schema")
    report = (output_root / "ava_speech_report.md").read_text(
        encoding="utf-8"
    )
    if "NEW_FINAL_OOD" not in report:
        raise RuntimeError("report does not record the forbidden-input guard")
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "output_root": _relative(output_root),
                "main_rows": int(len(main)),
                "mechanism_rows": int(len(mechanism)),
                "lightweight_rows": int(len(lightweight)),
                "test_videos": protocol["data"]["test_video_ids"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("freeze", "run", "smoke", "verify"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = (
        SMOKE_OUTPUT_ROOT
        if args.mode == "smoke"
        else (args.output_root or OUTPUT_ROOT)
    )
    output_root = Path(output_root)
    if args.mode == "freeze":
        return _freeze(output_root, force=bool(args.force))
    if args.mode == "smoke":
        if not (output_root / "ava_speech_protocol.json").exists():
            _freeze(output_root, force=True)
        return _run(output_root, smoke=True, force=bool(args.force))
    if args.mode == "run":
        return _run(output_root, smoke=False, force=bool(args.force))
    if args.mode == "verify":
        return _verify(output_root)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
