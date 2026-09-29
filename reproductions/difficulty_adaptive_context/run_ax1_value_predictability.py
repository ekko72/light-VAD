# -*- coding: utf-8 -*-
"""Extract and analyze the AX1 value-predictability ceiling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    deterministic_speaker_split,
)
from reproductions.difficulty_adaptive_context.analyze_a10_value_predictability import (
    binary_entropy,
    select_top_fraction,
    signed_temporal_value,
    utterance_segments,
)
from reproductions.difficulty_adaptive_context.data import (
    FRAME_HOP,
    build_evaluation_items,
    causal_frame_labels,
    read_int16_audio,
)
from reproductions.difficulty_adaptive_context.evaluate_adaptive import (
    load_adaptive_model,
)
from reproductions.marblenet_vad.dataset import INT16_SCALE
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "reproductions" / "difficulty_adaptive_context" / "ax_protocol.json"
FEATURE_NAMES = {
    "uncertainty": ("negative_abs_posterior_deviation",),
    "X0": ("posterior_entropy", "abs_posterior_deviation"),
    "X1": (
        "posterior_entropy",
        "abs_posterior_deviation",
        "posterior_trailing_mean_25",
        "posterior_trailing_std_25",
        "posterior_delta_1",
        "posterior_trailing_abs_delta_mean_25",
        "posterior_trailing_min_25",
        "posterior_trailing_max_25",
        "posterior_trailing_slope_25",
        "posterior_trailing_range_25",
    ),
    "X2": ("X1", "short_hidden_128"),
    "X3": (
        "X2",
        "causal_mfcc_current_64",
        "causal_mfcc_trailing_mean_25",
        "causal_mfcc_delta_1",
    ),
}


@dataclass(frozen=True)
class Ax1Config:
    protocol_path: Path
    payload: Mapping[str, Any]

    def resolve(self, relative_path: str) -> Path:
        return REPO_ROOT / relative_path

    @property
    def output_dir(self) -> Path:
        return self.resolve(str(self.payload["outputs"]["directory"]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_PROTOCOL) -> Ax1Config:
    if not path.exists():
        raise FileNotFoundError(f"AX protocol not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("protocol_id")) != "A-AX1-v1":
        raise ValueError("unexpected AX protocol id")
    return Ax1Config(protocol_path=path, payload=payload)


def validate_frozen_inputs(
    config: Ax1Config,
    *,
    verify_hashes: bool,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("prediction_bundle", "adaptive_checkpoint", "manifest"):
        entry = config.payload["data"][name]
        path = config.resolve(str(entry["path"]))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        if verify_hashes:
            actual = sha256_file(path)
            expected = str(entry["sha256"])
            if actual != expected:
                raise ValueError(
                    f"{name} hash mismatch: expected {expected}, got {actual}"
                )
            hashes[name] = actual
        else:
            hashes[name] = "not_verified"
    return hashes


def load_frozen_bundle(config: Ax1Config) -> dict[str, np.ndarray]:
    path = config.resolve(
        str(config.payload["data"]["prediction_bundle"]["path"])
    )
    required = {
        "labels",
        "short_scores",
        "full_adaptive_scores",
        "calibration_mask",
        "test_mask",
        "source_key",
        "noise_name",
        "condition",
        "speaker_ids",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"prediction bundle is missing: {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    size = int(arrays["labels"].size)
    for name, values in arrays.items():
        if int(values.size) != size:
            raise ValueError(f"{name} has size {values.size}, expected {size}")
    arrays["labels"] = arrays["labels"].astype(np.int64)
    arrays["short_scores"] = arrays["short_scores"].astype(np.float64)
    arrays["full_adaptive_scores"] = arrays[
        "full_adaptive_scores"
    ].astype(np.float64)
    arrays["calibration_mask"] = arrays["calibration_mask"].astype(bool)
    arrays["test_mask"] = arrays["test_mask"].astype(bool)
    for name in ("source_key", "noise_name", "condition", "speaker_ids"):
        arrays[name] = arrays[name].astype(str)
    if np.any(arrays["calibration_mask"] & arrays["test_mask"]):
        raise ValueError("calibration and test masks overlap")
    if not np.all(
        arrays["calibration_mask"] | arrays["test_mask"]
    ):
        raise ValueError("calibration and test masks do not cover all frames")
    return arrays


def _validate_split(
    bundle: Mapping[str, np.ndarray],
    config: Ax1Config,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    split = config.payload["split"]
    calibration_mask, test_mask, calibration, test = (
        deterministic_speaker_split(
            bundle["speaker_ids"],
            calibration_fraction=float(split["calibration_fraction"]),
            seed=int(split["speaker_split_seed"]),
        )
    )
    if not np.array_equal(
        calibration_mask,
        np.asarray(bundle["calibration_mask"], dtype=bool),
    ):
        raise ValueError("reconstructed calibration split does not match bundle")
    if not np.array_equal(
        test_mask,
        np.asarray(bundle["test_mask"], dtype=bool),
    ):
        raise ValueError("reconstructed test split does not match bundle")
    return calibration_mask, test_mask, calibration, test


def _trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(
        ([0.0], np.cumsum(values, dtype=np.float64))
    )
    indices = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, indices - int(window) + 1)
    return cumulative[indices + 1] - cumulative[starts]


def _trailing_count(values: np.ndarray, window: int) -> np.ndarray:
    indices = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, indices - int(window) + 1)
    return (indices - starts + 1).astype(np.float64)


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _trailing_sum(values, window) / _trailing_count(values, window)


def _trailing_variance(values: np.ndarray, window: int) -> np.ndarray:
    count = _trailing_count(values, window)
    total = _trailing_sum(values, window)
    total_sq = _trailing_sum(values * values, window)
    return np.maximum(total_sq / count - (total / count) ** 2, 0.0)


def _trailing_min(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - int(window) + 1)
        result[index] = np.min(values[start : index + 1])
    return result


def _trailing_max(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - int(window) + 1)
        result[index] = np.max(values[start : index + 1])
    return result


def _trailing_slope(values: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(values.size, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - int(window) + 1)
        local = values[start : index + 1]
        if local.size < 2:
            continue
        positions = np.arange(local.size, dtype=np.float64)
        centered = positions - np.mean(positions)
        result[index] = float(
            np.sum(centered * (local - np.mean(local)))
            / np.sum(centered * centered)
        )
    return result


def posterior_temporal_features(
    short_scores: np.ndarray,
    segments: Sequence[tuple[int, int]],
    *,
    window_frames: int,
) -> np.ndarray:
    """Return the frozen X1 feature block."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    features = np.empty((scores.size, 10), dtype=np.float64)
    for start, end in segments:
        local = scores[start:end]
        delta = np.zeros(local.size, dtype=np.float64)
        if local.size > 1:
            delta[1:] = np.diff(local)
        abs_delta = np.abs(delta)
        trailing_mean = _trailing_mean(local, window_frames)
        trailing_std = np.sqrt(_trailing_variance(local, window_frames))
        trailing_min = _trailing_min(local, window_frames)
        trailing_max = _trailing_max(local, window_frames)
        features[start:end, 0] = binary_entropy(local)
        features[start:end, 1] = np.abs(local - 0.5)
        features[start:end, 2] = trailing_mean
        features[start:end, 3] = trailing_std
        features[start:end, 4] = delta
        features[start:end, 5] = _trailing_mean(
            abs_delta, window_frames
        )
        features[start:end, 6] = trailing_min
        features[start:end, 7] = trailing_max
        features[start:end, 8] = _trailing_slope(local, window_frames)
        features[start:end, 9] = trailing_max - trailing_min
    if np.any(~np.isfinite(features)):
        raise RuntimeError("temporal posterior features are not finite")
    return features


def mfcc_context_features(
    mfcc: np.ndarray,
    segments: Sequence[tuple[int, int]],
    *,
    window_frames: int,
) -> np.ndarray:
    """Return current, trailing-mean, and delta causal MFCC blocks."""
    values = np.asarray(mfcc, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("mfcc must have shape [frames, coefficients]")
    channels = int(values.shape[1])
    result = np.empty((values.shape[0], 3 * channels), dtype=np.float64)
    for start, end in segments:
        local = values[start:end]
        delta = np.zeros_like(local)
        if local.shape[0] > 1:
            delta[1:] = np.diff(local, axis=0)
        trailing = np.stack(
            [
                _trailing_mean(local[:, channel], window_frames)
                for channel in range(channels)
            ],
            axis=1,
        )
        result[start:end, :channels] = local
        result[start:end, channels : 2 * channels] = trailing
        result[start:end, 2 * channels :] = delta
    if np.any(~np.isfinite(result)):
        raise RuntimeError("causal MFCC context is not finite")
    return result


def uncertainty_score(short_scores: np.ndarray) -> np.ndarray:
    """Return a descending-compatible uncertainty score."""
    scores = np.asarray(short_scores, dtype=np.float64).reshape(-1)
    return (-np.abs(scores - 0.5))[:, None]


def feature_blocks(
    short_scores: np.ndarray,
    hidden: np.ndarray,
    mfcc: np.ndarray,
    segments: Sequence[tuple[int, int]],
    *,
    window_frames: int,
) -> dict[str, np.ndarray]:
    x1 = posterior_temporal_features(
        short_scores,
        segments,
        window_frames=window_frames,
    )
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] != x1.shape[0]:
        raise ValueError("hidden must have shape [frames, channels]")
    acoustic = mfcc_context_features(
        mfcc,
        segments,
        window_frames=window_frames,
    ).astype(np.float32)
    if acoustic.shape[0] != x1.shape[0]:
        raise ValueError("acoustic context must align with posterior features")
    return {
        "uncertainty": uncertainty_score(short_scores).astype(np.float32),
        "X0": x1[:, :2].astype(np.float32),
        "X1": x1.astype(np.float32),
        "X2": np.concatenate([x1.astype(np.float32), hidden], axis=1),
        "X3": np.concatenate(
            [x1.astype(np.float32), hidden, acoustic],
            axis=1,
        ),
    }


@torch.inference_mode()
def encode_short_stream(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    chunk_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the checkpoint bundle's causal Short encoding exactly."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    encoded_parts: list[torch.Tensor] = []
    probability_parts: list[torch.Tensor] = []
    stream_state = model.short_model.init_stream_state(
        features[..., : min(chunk_frames, features.shape[-1])]
    )
    for start in range(0, features.shape[-1], chunk_frames):
        encoded, stream_state = model.short_model.encode_stream(
            features[..., start : start + chunk_frames],
            stream_state,
        )
        short_logits = model.short_model.classifier(encoded)
        probabilities = torch.softmax(
            short_logits.transpose(1, 2),
            dim=-1,
        )[..., 1]
        encoded_parts.append(encoded)
        probability_parts.append(probabilities)
    return torch.cat(encoded_parts, dim=-1), torch.cat(
        probability_parts,
        dim=-1,
    )


@torch.inference_mode()
def extract_ax1_features(
    config: Ax1Config,
    *,
    device: torch.device,
    output_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    """Rebuild hidden/acoustic features and verify the frozen Short scores."""
    data = config.payload["data"]
    bundle = load_frozen_bundle(config)
    _validate_split(bundle, config)
    checkpoint = config.resolve(str(data["adaptive_checkpoint"]["path"]))
    model, _, _ = load_adaptive_model(checkpoint, device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    items = build_evaluation_items(
        config.resolve(str(data["manifest"]["path"])),
        generated_root=config.resolve(str(data["generated_root"])),
        label_root=config.resolve(str(data["label_root"])),
        librispeech_root=config.resolve(str(data["librispeech_root"])),
        row_sample=int(data["row_sample"]),
        seed=int(data["manifest_seed"]),
        include_clean=bool(data["include_clean"]),
    )
    valid_start = int(data["valid_start_frame"])
    score_chunk_frames = int(data["score_chunk_frames"])
    labels_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    noise_parts: list[np.ndarray] = []
    condition_parts: list[np.ndarray] = []
    short_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    mfcc_parts: list[np.ndarray] = []
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
        tensor = torch.from_numpy(
            waveform.astype(np.float32) * INT16_SCALE
        ).unsqueeze(0)
        features = frontend(tensor.to(device, non_blocking=True))
        encoded, short_probabilities = encode_short_stream(
            model,
            features,
            chunk_frames=score_chunk_frames,
        )
        if encoded.shape[-1] != n_frames:
            raise RuntimeError("encoder output frame count changed")
        labels_parts.append(frame_labels[valid_start:])
        source_parts.append(
            np.full(
                n_frames - valid_start,
                item.source_key,
                dtype=f"U{max(1, len(item.source_key))}",
            )
        )
        noise_name = str(item.noise_name)
        noise_parts.append(
            np.full(
                n_frames - valid_start,
                noise_name,
                dtype=f"U{max(1, len(noise_name))}",
            )
        )
        condition = "clean" if item.is_clean else str(item.snr_db)
        condition_parts.append(
            np.full(
                n_frames - valid_start,
                condition,
                dtype=f"U{max(1, len(condition))}",
            )
        )
        short_parts.append(
            short_probabilities[0, valid_start:]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        hidden_parts.append(
            encoded[0, :, valid_start:]
            .transpose(0, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        mfcc_parts.append(
            features[0, :, valid_start:]
            .transpose(0, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if progress_every > 0 and (
            (index + 1) % progress_every == 0 or index + 1 == len(items)
        ):
            print(f"  extracted {index + 1}/{len(items)}", flush=True)
    if not labels_parts:
        raise RuntimeError("all selected utterances were too short")

    labels = np.concatenate(labels_parts).astype(np.int64)
    source_key = np.concatenate(source_parts).astype(str)
    noise_name = np.concatenate(noise_parts).astype(str)
    condition = np.concatenate(condition_parts).astype(str)
    embedded_short = np.concatenate(short_parts).astype(np.float64)
    hidden = np.concatenate(hidden_parts, axis=0)
    mfcc = np.concatenate(mfcc_parts, axis=0)

    checks = {
        "labels": np.array_equal(labels, bundle["labels"]),
        "source_key": np.array_equal(source_key, bundle["source_key"]),
        "noise_name": np.array_equal(noise_name, bundle["noise_name"]),
        "condition": np.array_equal(condition, bundle["condition"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "reconstructed evaluation metadata differs from frozen bundle: "
            + ", ".join(failed)
        )
    short_error = np.abs(embedded_short - bundle["short_scores"])
    if not np.allclose(
        embedded_short,
        bundle["short_scores"],
        rtol=1e-3,
        atol=1e-3,
    ):
        raise RuntimeError(
            "embedded Short scores differ from the frozen bundle; "
            f"max abs error={float(np.max(short_error)):.6g}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        hidden=hidden,
        mfcc=mfcc,
        labels=labels,
        source_key=source_key,
        noise_name=noise_name,
        condition=condition,
        embedded_short_scores=embedded_short,
    )
    summary = {
        "items": len(items),
        "frames": int(labels.size),
        "hidden_shape": list(hidden.shape),
        "mfcc_shape": list(mfcc.shape),
        "metadata_matches": checks,
        "short_max_abs_error": float(np.max(short_error)),
        "short_mean_abs_error": float(np.mean(short_error)),
        "short_p99_abs_error": float(np.quantile(short_error, 0.99)),
        "output": str(output_path),
    }
    return summary


def _class_values() -> np.ndarray:
    return np.asarray([-1, 0, 1], dtype=np.int64)


def fit_logistic_score(
    features: np.ndarray,
    values: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[calibration_mask])
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=int(max_iter),
        tol=float(tolerance),
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(train_x, values[calibration_mask])
    probabilities = model.predict_proba(scaler.transform(features))
    classes = model.classes_.astype(np.int64)
    return (probabilities @ classes.astype(np.float64)).astype(np.float64)


class _ValueMlp(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(int(input_size), 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def fit_mlp_score(
    features: np.ndarray,
    values: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[calibration_mask]).astype(
        np.float32
    )
    class_ids = np.searchsorted(_class_values(), values[calibration_mask])
    class_counts = np.bincount(class_ids, minlength=3).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("MLP calibration split is missing a value class")
    class_weights = class_ids.size / (3.0 * class_counts)

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = _ValueMlp(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_tensor = torch.from_numpy(train_x)
    target_tensor = torch.from_numpy(class_ids.astype(np.int64))
    weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(
        device
    )
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(
            target_tensor.numel(),
            generator=generator,
        )
        for start in range(0, order.numel(), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            inputs = train_tensor[indices].to(device)
            targets = target_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits,
                targets,
                weight=weight_tensor,
            )
            loss.backward()
            optimizer.step()

    model.eval()
    scores = np.empty(features.shape[0], dtype=np.float64)
    class_tensor = torch.from_numpy(_class_values()).to(device)
    with torch.inference_mode():
        for start in range(0, features.shape[0], int(batch_size)):
            batch = scaler.transform(
                features[start : start + int(batch_size)]
            ).astype(np.float32)
            probabilities = torch.softmax(
                model(torch.from_numpy(batch).to(device)),
                dim=-1,
            )
            scores[start : start + batch.shape[0]] = (
                (probabilities * class_tensor[None, :])
                .sum(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
    return scores


def _selected_metrics(
    values: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size != selected.size:
        raise ValueError("values and selected mask must have equal lengths")
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        return {
            "selected": 0,
            "selected_mean_value": None,
            "oracle_value_recovery": None,
            "net_selected_value": 0,
            "positive_oracle_sum": int(np.sum(np.maximum(values, 0))),
            "correction_selected": 0,
            "harm_selected": 0,
        }
    selected_values = values[selected]
    positive_oracle_sum = int(np.sum(np.maximum(values, 0)))
    net = int(np.sum(selected_values))
    return {
        "selected": selected_count,
        "selected_mean_value": float(net / selected_count),
        "oracle_value_recovery": (
            float(net / positive_oracle_sum)
            if positive_oracle_sum
            else None
        ),
        "net_selected_value": net,
        "positive_oracle_sum": positive_oracle_sum,
        "correction_selected": int(np.count_nonzero(selected_values == 1)),
        "harm_selected": int(np.count_nonzero(selected_values == -1)),
    }


def _cluster_statistics(
    values: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    positive = np.maximum(values, 0).astype(np.int64)
    selected_values = values * selected.astype(np.int64)
    return np.column_stack(
        [
            np.bincount(
                cluster_codes,
                weights=selected.astype(np.int64),
                minlength=n_clusters,
            ),
            np.bincount(
                cluster_codes,
                weights=selected_values,
                minlength=n_clusters,
            ),
            np.bincount(
                cluster_codes,
                weights=positive,
                minlength=n_clusters,
            ),
            np.bincount(
                cluster_codes,
                weights=(selected & (values == 1)).astype(np.int64),
                minlength=n_clusters,
            ),
            np.bincount(
                cluster_codes,
                weights=(selected & (values == -1)).astype(np.int64),
                minlength=n_clusters,
            ),
        ]
    ).astype(np.int64)


def _metrics_from_cluster_statistics(
    statistics: np.ndarray,
) -> dict[str, Any]:
    totals = np.asarray(statistics, dtype=np.int64).sum(axis=0)
    selected, net, positive, correction, harm = (
        int(value) for value in totals
    )
    return {
        "selected": selected,
        "selected_mean_value": (
            float(net / selected) if selected else None
        ),
        "oracle_value_recovery": (
            float(net / positive) if positive else None
        ),
        "net_selected_value": net,
        "positive_oracle_sum": positive,
        "correction_selected": correction,
        "harm_selected": harm,
    }


def cluster_bootstrap_metrics(
    values: np.ndarray,
    selected: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (
        values.size == selected.size == cluster_codes.size
    ):
        raise ValueError("bootstrap arrays must have equal lengths")
    n_clusters = int(cluster_codes.max()) + 1
    statistics = _cluster_statistics(
        values,
        selected,
        cluster_codes,
        n_clusters,
    )
    rng = np.random.default_rng(int(seed))
    mean_values: list[float] = []
    recovery_values: list[float] = []
    selected_values: list[float] = []
    for _ in range(int(repeats)):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        row = _metrics_from_cluster_statistics(statistics[sampled])
        if row["selected_mean_value"] is not None:
            mean_values.append(float(row["selected_mean_value"]))
        if row["oracle_value_recovery"] is not None:
            recovery_values.append(float(row["oracle_value_recovery"]))
        selected_values.append(float(row["selected"]))
    point = _metrics_from_cluster_statistics(statistics)
    result = dict(point)
    for name, samples in (
        ("selected_mean_value", mean_values),
        ("oracle_value_recovery", recovery_values),
        ("selected", selected_values),
    ):
        if samples:
            low, high = np.quantile(samples, [0.025, 0.975])
            result[f"{name}_ci95_low"] = float(low)
            result[f"{name}_ci95_high"] = float(high)
        else:
            result[f"{name}_ci95_low"] = None
            result[f"{name}_ci95_high"] = None
    result["bootstrap_repeats"] = int(repeats)
    result["speaker_clusters"] = n_clusters
    return result


def paired_cluster_bootstrap_difference(
    values: np.ndarray,
    first_selected: np.ndarray,
    second_selected: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    first_selected = np.asarray(first_selected, dtype=bool).reshape(-1)
    second_selected = np.asarray(second_selected, dtype=bool).reshape(-1)
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64).reshape(-1)
    if not (
        values.size
        == first_selected.size
        == second_selected.size
        == cluster_codes.size
    ):
        raise ValueError("paired bootstrap arrays must have equal lengths")
    n_clusters = int(cluster_codes.max()) + 1
    first_stats = _cluster_statistics(
        values,
        first_selected,
        cluster_codes,
        n_clusters,
    )
    second_stats = _cluster_statistics(
        values,
        second_selected,
        cluster_codes,
        n_clusters,
    )
    rng = np.random.default_rng(int(seed))
    mean_differences: list[float] = []
    recovery_differences: list[float] = []
    for _ in range(int(repeats)):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        first = _metrics_from_cluster_statistics(first_stats[sampled])
        second = _metrics_from_cluster_statistics(second_stats[sampled])
        if (
            first["selected_mean_value"] is not None
            and second["selected_mean_value"] is not None
        ):
            mean_differences.append(
                float(first["selected_mean_value"])
                - float(second["selected_mean_value"])
            )
        if (
            first["oracle_value_recovery"] is not None
            and second["oracle_value_recovery"] is not None
        ):
            recovery_differences.append(
                float(first["oracle_value_recovery"])
                - float(second["oracle_value_recovery"])
            )
    point_first = _metrics_from_cluster_statistics(first_stats)
    point_second = _metrics_from_cluster_statistics(second_stats)

    def interval(values_: Sequence[float]) -> tuple[float | None, float | None]:
        if not values_:
            return None, None
        low, high = np.quantile(values_, [0.025, 0.975])
        return float(low), float(high)

    mean_low, mean_high = interval(mean_differences)
    recovery_low, recovery_high = interval(recovery_differences)
    return {
        "first_selected_mean_value": point_first["selected_mean_value"],
        "second_selected_mean_value": point_second["selected_mean_value"],
        "selected_mean_value_difference": (
            None
            if point_first["selected_mean_value"] is None
            or point_second["selected_mean_value"] is None
            else float(point_first["selected_mean_value"])
            - float(point_second["selected_mean_value"])
        ),
        "selected_mean_value_difference_ci95_low": mean_low,
        "selected_mean_value_difference_ci95_high": mean_high,
        "first_oracle_value_recovery": point_first[
            "oracle_value_recovery"
        ],
        "second_oracle_value_recovery": point_second[
            "oracle_value_recovery"
        ],
        "oracle_value_recovery_difference": (
            None
            if point_first["oracle_value_recovery"] is None
            or point_second["oracle_value_recovery"] is None
            else float(point_first["oracle_value_recovery"])
            - float(point_second["oracle_value_recovery"])
        ),
        "oracle_value_recovery_difference_ci95_low": recovery_low,
        "oracle_value_recovery_difference_ci95_high": recovery_high,
    }


def _fixed_selection(
    scores: np.ndarray,
    calibration_mask: np.ndarray,
    test_mask: np.ndarray,
    budget: float,
) -> tuple[float, np.ndarray]:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat_scores.size != calibration_mask.size:
        raise ValueError("scores and selection masks must have equal lengths")
    threshold, _ = select_top_fraction(
        flat_scores[calibration_mask],
        activation_rate=float(budget),
    )
    test_selected = np.zeros(flat_scores.size, dtype=bool)
    test_selected[test_mask] = flat_scores[test_mask] >= threshold
    return threshold, test_selected


def score_sortable(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _oracle_selection(
    values: np.ndarray,
    test_mask: np.ndarray,
    budget: float,
) -> np.ndarray:
    test_indices = np.flatnonzero(test_mask)
    target = int(math.ceil(float(budget) * test_indices.size))
    target = min(max(target, 1), test_indices.size)
    selected = np.zeros(values.size, dtype=bool)
    order = np.argsort(-values[test_indices], kind="stable")
    selected[test_indices[order[:target]]] = True
    return selected


def evaluate_ax1(
    config: Ax1Config,
    *,
    feature_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    bundle = load_frozen_bundle(config)
    calibration_mask, test_mask, calibration_speakers, test_speakers = (
        _validate_split(bundle, config)
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        required = {"hidden", "mfcc"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"feature cache is missing: {missing}")
        hidden = np.asarray(payload["hidden"], dtype=np.float32)
        mfcc = np.asarray(payload["mfcc"], dtype=np.float32)
    if hidden.shape[0] != bundle["labels"].size:
        raise ValueError("feature cache does not match the frozen bundle")

    segments = utterance_segments(
        bundle["source_key"],
        bundle["noise_name"],
        bundle["condition"],
    )
    features = feature_blocks(
        bundle["short_scores"],
        hidden,
        mfcc,
        segments,
        window_frames=int(
            config.payload["feature_ladder"]["causal_window_frames"]
        ),
    )
    values = signed_temporal_value(
        bundle["labels"],
        bundle["short_scores"],
        bundle["full_adaptive_scores"],
    ).astype(np.int64)
    cluster_names, cluster_codes = np.unique(
        bundle["speaker_ids"][test_mask],
        return_inverse=True,
    )
    model_config = config.payload["models"]
    scores: dict[str, np.ndarray] = {
        "uncertainty": features["uncertainty"],
    }
    logistic_config = model_config["logistic"]
    mlp_config = model_config["mlp_ceiling_probe"]
    for name in ("X0", "X1", "X2", "X3"):
        print(f"fitting logistic {name}", flush=True)
        scores[f"{name}_logistic"] = fit_logistic_score(
            features[name],
            values,
            calibration_mask,
            seed=int(logistic_config["random_state"]),
            max_iter=int(logistic_config["max_iter"]),
            tolerance=float(logistic_config["tol"]),
        )
        print(f"fitting MLP {name}", flush=True)
        scores[f"{name}_mlp"] = fit_mlp_score(
            features[name],
            values,
            calibration_mask,
            seed=int(mlp_config["torch_seed"]),
            learning_rate=float(mlp_config["learning_rate"]),
            weight_decay=float(mlp_config["weight_decay"]),
            batch_size=int(mlp_config["batch_size"]),
            epochs=int(mlp_config["epochs"]),
            device=device,
        )

    budgets = tuple(
        float(value) for value in config.payload["selection"]["budgets"]
    )
    bootstrap_config = config.payload["bootstrap"]
    rows: list[dict[str, Any]] = []
    selected_by_budget: dict[float, dict[str, np.ndarray]] = {}
    for budget_index, budget in enumerate(budgets):
        selected_by_budget[budget] = {}
        for model_index, (name, raw_scores) in enumerate(scores.items()):
            threshold, selected = _fixed_selection(
                raw_scores,
                calibration_mask,
                test_mask,
                budget,
            )
            selected_by_budget[budget][name] = selected
            bootstrap = cluster_bootstrap_metrics(
                values[test_mask],
                selected[test_mask],
                cluster_codes,
                repeats=int(bootstrap_config["repeats"]),
                seed=(
                    int(bootstrap_config["seed"])
                    + 10_007 * budget_index
                    + 101 * model_index
                ),
            )
            test_selected = int(np.count_nonzero(selected[test_mask]))
            test_frames = int(np.count_nonzero(test_mask))
            row = {
                "budget": float(budget),
                "model": name,
                "calibration_threshold": float(threshold),
                "test_activation_rate": (
                    float(test_selected / test_frames)
                    if test_frames
                    else None
                ),
                **bootstrap,
                "correction_rate_selected": (
                    float(
                        bootstrap["correction_selected"]
                        / bootstrap["selected"]
                    )
                    if bootstrap["selected"]
                    else None
                ),
                "harm_rate_selected": (
                    float(
                        bootstrap["harm_selected"]
                        / bootstrap["selected"]
                    )
                    if bootstrap["selected"]
                    else None
                ),
            }
            rows.append(row)
        oracle = _oracle_selection(values, test_mask, budget)
        selected_by_budget[budget]["oracle"] = oracle
        oracle_stats = _selected_metrics(values[test_mask], oracle[test_mask])
        rows.append(
            {
                "budget": float(budget),
                "model": "oracle",
                "calibration_threshold": None,
                "test_activation_rate": float(budget),
                **oracle_stats,
                "correction_rate_selected": (
                    float(oracle_stats["correction_selected"] / oracle_stats["selected"])
                    if oracle_stats["selected"]
                    else None
                ),
                "harm_rate_selected": (
                    float(oracle_stats["harm_selected"] / oracle_stats["selected"])
                    if oracle_stats["selected"]
                    else None
                ),
                "bootstrap_repeats": 0,
                "speaker_clusters": int(cluster_names.size),
            }
        )

    paired: dict[str, dict[str, Any]] = {}
    for budget_index, budget in enumerate(budgets):
        baseline = selected_by_budget[budget]["uncertainty"]
        paired[str(budget)] = {}
        for model_index, name in enumerate(scores):
            if name == "uncertainty":
                continue
            paired[str(budget)][name] = paired_cluster_bootstrap_difference(
                values[test_mask],
                selected_by_budget[budget][name][test_mask],
                baseline[test_mask],
                cluster_codes,
                repeats=int(bootstrap_config["repeats"]),
                seed=(
                    int(bootstrap_config["seed"])
                    + 20_011 * budget_index
                    + 211 * model_index
                ),
            )

    return {
        "protocol": {
            "path": str(config.protocol_path),
            "id": config.payload["protocol_id"],
            "feature_names": FEATURE_NAMES,
            "window_frames": int(
                config.payload["feature_ladder"]["causal_window_frames"]
            ),
        },
        "split": {
            "calibration_speakers": calibration_speakers,
            "test_speakers": test_speakers,
            "calibration_frames": int(np.count_nonzero(calibration_mask)),
            "test_frames": int(np.count_nonzero(test_mask)),
        },
        "signed_value": {
            "test_positive": int(np.count_nonzero(values[test_mask] == 1)),
            "test_zero": int(np.count_nonzero(values[test_mask] == 0)),
            "test_negative": int(np.count_nonzero(values[test_mask] == -1)),
            "test_mean_value": float(np.mean(values[test_mask])),
            "test_positive_oracle_sum": int(
                np.sum(np.maximum(values[test_mask], 0))
            ),
        },
        "rows": rows,
        "paired_vs_uncertainty": paired,
        "assessment": assess_ax1(rows, paired),
    }


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[float, str], Mapping[str, Any]]:
    return {
        (float(row["budget"]), str(row["model"])): row
        for row in rows
    }


def assess_ax1(
    rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    lookup = _row_lookup(rows)
    budgets = sorted({float(row["budget"]) for row in rows})
    candidates = [
        name
        for name in lookup
        if name[0] == budgets[0]
        and name[1] not in {"uncertainty", "oracle"}
    ]
    candidate_names = sorted({name for _, name in candidates})
    signal_budgets: list[float] = []
    max_improvement = 0.0
    for budget in budgets:
        for name in candidate_names:
            comparison = paired[str(budget)][name]
            improvement = comparison.get(
                "oracle_value_recovery_difference"
            )
            low = comparison.get(
                "oracle_value_recovery_difference_ci95_low"
            )
            if improvement is not None:
                max_improvement = max(max_improvement, float(improvement))
            if (
                improvement is not None
                and low is not None
                and float(improvement) > 0.0
                and float(low) > 0.0
            ):
                signal_budgets.append(float(budget))
                break
    signal_budgets = sorted(set(signal_budgets))
    richer_signal = (
        len(signal_budgets) >= 2 and max_improvement > 0.02
    )
    near_ceiling = not signal_budgets and max_improvement <= 0.01

    learned_recovery = {
        budget: max(
            float(row["oracle_value_recovery"])
            for (row_budget, name), row in lookup.items()
            if row_budget == budget
            and name not in {"uncertainty", "oracle"}
            and row.get("oracle_value_recovery") is not None
        )
        for budget in budgets
    }
    large_gap = (
        learned_recovery.get(0.01, 0.0) < 0.25
        and learned_recovery.get(0.02, 0.0) < 0.25
        and learned_recovery.get(0.05, 0.0) < 0.50
    )
    if richer_signal:
        status = "RICHER_FEATURES_ADD_OBSERVABLE_SIGNAL"
        interpretation = (
            "At least one higher-information predictor improves value "
            "capture over uncertainty at multiple budgets. This permits "
            "the separately frozen AX2 analysis; it does not authorize a "
            "new router."
        )
    elif near_ceiling:
        status = "UNCERTAINTY_NEAR_OBSERVABLE_CEILING"
        interpretation = (
            "No learned ladder model produced a stable material oracle "
            "value-recovery improvement over uncertainty. Preserve the "
            "simple uncertainty router and characterize the remaining gap."
        )
    else:
        status = "INCONCLUSIVE_OR_IRREDUCIBLE_GAP"
        interpretation = (
            "The result does not satisfy the frozen richer-signal rule, "
            "but improvements are too close to zero to call uncertainty "
            "the observable ceiling. Continue only with the predeclared "
            "AX2 separability diagnostic."
        )
    return {
        "status": status,
        "interpretation": interpretation,
        "signal_budgets": signal_budgets,
        "max_oracle_value_recovery_improvement_over_uncertainty": (
            max_improvement
        ),
        "learned_oracle_value_recovery_by_budget": learned_recovery,
        "criteria": {
            "uncertainty_near_observable_ceiling": near_ceiling,
            "richer_features_add_observable_signal": richer_signal,
            "large_irreducible_routing_gap": large_gap,
        },
    }


def write_budget_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget",
        "model",
        "test_activation_rate",
        "selected",
        "selected_mean_value",
        "selected_mean_value_ci95_low",
        "selected_mean_value_ci95_high",
        "oracle_value_recovery",
        "oracle_value_recovery_ci95_low",
        "oracle_value_recovery_ci95_high",
        "correction_rate_selected",
        "harm_rate_selected",
        "calibration_threshold",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.3f}%"


def _num(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    rows = list(summary["rows"])
    budgets = sorted({float(row["budget"]) for row in rows})
    lines = [
        "# AX1 Value Predictability Ceiling",
        "",
        "## Frozen Split",
        "",
        f"- Calibration speakers/frames: "
        f"{len(summary['split']['calibration_speakers'])}/"
        f"{summary['split']['calibration_frames']:,}",
        f"- Test speakers/frames: "
        f"{len(summary['split']['test_speakers'])}/"
        f"{summary['split']['test_frames']:,}",
        f"- Test signed value: +1 "
        f"{summary['signed_value']['test_positive']:,}, 0 "
        f"{summary['signed_value']['test_zero']:,}, -1 "
        f"{summary['signed_value']['test_negative']:,}.",
        "",
        "All fits and thresholds use calibration speakers only. Test "
        "activation is observed. The oracle row is a test-only upper bound "
        "and is not a deployable router.",
        "",
        "## Budget Results",
        "",
        "| Budget | Model | Activation | E[v\\|selected] | 95% CI | "
        "Oracle recovery | 95% CI |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in budgets:
        for row in rows:
            if float(row["budget"]) != budget:
                continue
            lines.append(
                f"| {100.0 * budget:.0f}% | {row['model']} | "
                f"{_pct(row.get('test_activation_rate'))} | "
                f"{_num(row.get('selected_mean_value'))} | "
                f"[{_num(row.get('selected_mean_value_ci95_low'))}, "
                f"{_num(row.get('selected_mean_value_ci95_high'))}] | "
                f"{_pct(row.get('oracle_value_recovery'))} | "
                f"[{_pct(row.get('oracle_value_recovery_ci95_low'))}, "
                f"{_pct(row.get('oracle_value_recovery_ci95_high'))}] |"
            )
    lines.extend(
        [
            "",
            "## Paired Value Recovery vs Uncertainty",
            "",
            "| Budget | Model | Mean improvement | Paired 95% CI |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    paired = summary["paired_vs_uncertainty"]
    for budget in budgets:
        for name, row in paired[str(budget)].items():
            lines.append(
                f"| {100.0 * budget:.0f}% | {name} | "
                f"{_pct(row.get('oracle_value_recovery_difference'))} | "
                f"[{_pct(row.get('oracle_value_recovery_difference_ci95_low'))}, "
                f"{_pct(row.get('oracle_value_recovery_difference_ci95_high'))}] |"
            )
    assessment = summary["assessment"]
    lines.extend(
        [
            "",
            "## AX1 Assessment",
            "",
            f"- Status: **{assessment['status']}**.",
            f"- Maximum mean oracle-value-recovery improvement over "
            f"uncertainty: "
            f"{_pct(assessment['max_oracle_value_recovery_improvement_over_uncertainty'])}.",
            f"- Budgets passing the paired richer-signal rule: "
            + (
                ", ".join(
                    f"{100.0 * value:.0f}%"
                    for value in assessment["signal_budgets"]
                )
                if assessment["signal_budgets"]
                else "none"
            ),
            f"- Interpretation: {assessment['interpretation']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("extract", "run"),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--feature-cache", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.protocol)
    validate_frozen_inputs(
        config,
        verify_hashes=not args.skip_hash_check,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = (
        args.feature_cache
        if args.feature_cache is not None
        else config.output_dir
        / str(config.payload["outputs"]["feature_cache"])
    )
    device = resolve_device(args.device)
    if args.command == "extract":
        summary = extract_ax1_features(
            config,
            device=device,
            output_path=feature_path,
            progress_every=int(args.progress_every),
        )
        write_json(
            config.output_dir / "ax1_extraction.json",
            summary,
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    if not feature_path.exists():
        raise FileNotFoundError(
            f"AX1 feature cache not found: {feature_path}; run extract first"
        )
    summary = evaluate_ax1(
        config,
        feature_path=feature_path,
        device=device,
    )
    summary["inputs"] = {
        "feature_cache": str(feature_path),
        "feature_cache_sha256": sha256_file(feature_path),
        "hashes": validate_frozen_inputs(config, verify_hashes=True),
    }
    write_json(
        config.output_dir / str(config.payload["outputs"]["summary"]),
        summary,
    )
    write_budget_csv(
        summary["rows"],
        config.output_dir / str(config.payload["outputs"]["budget_table"]),
    )
    write_report(
        summary,
        config.output_dir / str(config.payload["outputs"]["report"]),
    )
    print(
        json.dumps(summary["assessment"], indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
