from __future__ import annotations

import argparse
import audioop
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from .features import (
    FRAME_SIZE_SAMPLES,
    FRAME_STEP_SECONDS,
    SAMPLE_RATE,
    SEQUENCE_FRAMES,
    _mfcc_features,
    _webrtc_labels,
)
from .legacy_models import BATCH_SIZE, load_legacy_checkpoint, num_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "fa6f57855a41380f44f5c95c3d1252fc5856da73"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MODEL_DIR = Path(
    r"C:\Users\20547\Desktop\论文p12\voice-activity-detection-upstream\data\models"
)
DEFAULT_SPEECH_MANIFEST = PROJECT_ROOT / "data" / "splits" / "test_speech.tsv"
DEFAULT_NOISE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "musan_manifest.tsv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "reproductions"
    / "nicklashansen_vad_balanced_musan.json"
)
DEFAULT_MODELS = (
    "net_epoch014.net",
    "net_large_epoch014.net",
    "gru_epoch014.net",
    "gru_large_epoch014.net",
    "densenet_epoch012.net",
    "densenet_large_epoch014.net",
)
NOISE_LEVELS = (
    ("None", None),
    ("-15", -15.0),
    ("-3", -3.0),
)
SLICE_MIN_FRAMES = int(1.0 / 0.03)
SLICE_MAX_FRAMES = int(5.0 / 0.03)


@dataclass(frozen=True)
class AudioInfo:
    frames: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class AudioSlice:
    waveform: np.ndarray
    labels: np.ndarray
    noise: np.ndarray

    @property
    def frames(self) -> np.ndarray:
        return self.waveform.reshape(-1, FRAME_SIZE_SAMPLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the upstream balanced test protocol with the official "
            "nicklashansen VAD checkpoints. QUT-NOISE is replaced by MUSAN noise."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--speech-manifest", type=Path, default=DEFAULT_SPEECH_MANIFEST)
    parser.add_argument("--noise-manifest", type=Path, default=DEFAULT_NOISE_MANIFEST)
    parser.add_argument("--speech-minutes", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--noise-categories",
        default="noise",
        help="Comma-separated MUSAN top-level categories. Default: noise.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Checkpoint filename; repeat to select models. Defaults to all six.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def audio_info(path: Path, cache: dict[Path, AudioInfo]) -> AudioInfo:
    if path not in cache:
        info = sf.info(str(path))
        cache[path] = AudioInfo(
            frames=info.frames,
            sample_rate=info.samplerate,
            channels=info.channels,
        )
    return cache[path]


def read_audio_frames(
    path: Path,
    start_frame: int,
    frame_count: int,
    cache: dict[Path, AudioInfo],
) -> np.ndarray:
    info = audio_info(path, cache)
    if info.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample rate {info.sample_rate}; expected {SAMPLE_RATE}"
        )
    if start_frame < 0 or start_frame + frame_count > info.frames:
        raise ValueError(f"Requested frames outside the file: {path}")

    with sf.SoundFile(str(path)) as file:
        file.seek(start_frame)
        waveform = file.read(
            frame_count,
            dtype="int16",
            always_2d=True,
        )

    if waveform.shape[1] == 1:
        return np.ascontiguousarray(waveform[:, 0], dtype=np.int16)

    # The source manifest is mono, but downmix defensively if it changes.
    return np.rint(waveform.mean(axis=1)).astype(np.int16)


def random_clip(
    rng: random.Random,
    rows: list[dict[str, str]],
    data_root: Path,
    cache: dict[Path, AudioInfo],
    min_frames: int = SLICE_MIN_FRAMES,
    max_frames: int = SLICE_MAX_FRAMES,
) -> np.ndarray:
    while True:
        row = rng.choice(rows)
        path = data_root / row["path"]
        info = audio_info(path, cache)
        available_frames = info.frames // FRAME_SIZE_SAMPLES
        upper = min(max_frames, available_frames)
        if upper < min_frames:
            continue

        frame_count = rng.randint(min_frames, upper)
        sample_count = frame_count * FRAME_SIZE_SAMPLES
        start = rng.randrange(0, info.frames - sample_count + 1)
        return read_audio_frames(path, start, sample_count, cache)


def build_balanced_slices(
    rng: random.Random,
    speech_rows: list[dict[str, str]],
    noise_rows: list[dict[str, str]],
    data_root: Path,
    speech_frames_target: int,
) -> tuple[list[AudioSlice], int, int]:
    cache: dict[Path, AudioInfo] = {}
    clean_slices: list[tuple[np.ndarray, np.ndarray]] = []
    speech_frames = 0

    while speech_frames < speech_frames_target:
        waveform = random_clip(rng, speech_rows, data_root, cache)
        labels = _webrtc_labels(waveform.reshape(-1, FRAME_SIZE_SAMPLES))
        clean_slices.append((waveform, labels))
        speech_frames += len(labels)

    silence_frames = 0
    while silence_frames < speech_frames:
        frame_count = rng.randint(SLICE_MIN_FRAMES, SLICE_MAX_FRAMES)
        waveform = np.zeros(frame_count * FRAME_SIZE_SAMPLES, dtype=np.int16)
        labels = np.zeros(frame_count, dtype=np.int8)
        clean_slices.append((waveform, labels))
        silence_frames += len(labels)

    rng.shuffle(clean_slices)

    slices = []
    for waveform, labels in clean_slices:
        noise = random_clip(
            rng,
            noise_rows,
            data_root,
            cache,
            min_frames=len(labels),
            max_frames=len(labels),
        )
        noise = peak_normalize(noise)
        slices.append(
            AudioSlice(
                waveform=waveform,
                labels=labels,
                noise=noise,
            )
        )
    return slices, speech_frames, silence_frames


def _pcm16_bytes(frames: np.ndarray) -> bytes:
    return np.ascontiguousarray(frames, dtype=np.int16).tobytes()


def _pcm16_array(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).copy()


def peak_normalize(frames: np.ndarray) -> np.ndarray:
    data = _pcm16_bytes(frames)
    peak = audioop.max(data, 2)
    if peak == 0:
        return frames.copy()
    normalized = audioop.mul(data, 2, 32768.0 / peak)
    return _pcm16_array(normalized)


def mix_with_noise(
    speech_frames: np.ndarray,
    noise_frames: np.ndarray,
    gain_during_overlay: float | None,
) -> np.ndarray:
    # Upstream overlays speech onto a peak-normalized noise track. A gain of
    # None means 0 dB, not "add no noise".
    noise_data = _pcm16_bytes(noise_frames)
    if gain_during_overlay:
        noise_data = audioop.mul(
            noise_data,
            2,
            10.0 ** (float(gain_during_overlay) / 20.0),
        )
    return _pcm16_array(
        audioop.add(noise_data, _pcm16_bytes(speech_frames), 2)
    )


def build_features(
    slices: list[AudioSlice],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features: dict[str, list[np.ndarray]] = {name: [] for name, _ in NOISE_LEVELS}
    labels = np.concatenate([item.labels for item in slices]).astype(np.int64)
    alignment = {
        name: np.zeros((3, FRAME_SIZE_SAMPLES), dtype=np.int16)
        for name, _ in NOISE_LEVELS
    }

    for item in slices:
        for name, gain_db in NOISE_LEVELS:
            mixed_waveform = mix_with_noise(item.waveform, item.noise, gain_db)
            mixed_frames = mixed_waveform.reshape(-1, FRAME_SIZE_SAMPLES)
            mfcc, delta = _mfcc_features(
                mixed_frames,
                alignment_frames=alignment[name],
            )
            features[name].append(
                np.hstack((mfcc, delta)).astype(np.float32, copy=False)
            )
            alignment[name] = mixed_frames[-3:]

    return (
        {name: np.concatenate(values, axis=0) for name, values in features.items()},
        labels,
    )


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(
        0,
        len(features) - SEQUENCE_FRAMES + 1,
        6,
        dtype=np.int64,
    )
    usable = (len(starts) // BATCH_SIZE) * BATCH_SIZE
    if usable == 0:
        raise RuntimeError(
            f"Only {len(starts)} windows were generated; need at least {BATCH_SIZE}"
        )
    starts = starts[:usable]
    windows = np.stack(
        [features[start : start + SEQUENCE_FRAMES] for start in starts],
        axis=0,
    )
    targets = labels[starts + SEQUENCE_FRAMES // 2]
    return windows, targets


def move_hidden_to_device(model: torch.nn.Module, device: torch.device) -> None:
    # Older checkpoints store LSTM hidden tensors as ordinary attributes.
    for module in model.modules():
        hidden = getattr(module, "hidden", None)
        if isinstance(hidden, torch.Tensor):
            module.hidden = hidden.to(device)
        elif isinstance(hidden, tuple):
            module.hidden = tuple(
                value.to(device) if isinstance(value, torch.Tensor) else value
                for value in hidden
            )


def predict(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    move_hidden_to_device(model, device)
    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = torch.from_numpy(
                windows[start : start + BATCH_SIZE]
            ).to(device=device, dtype=torch.float32)
            output = model(batch)
            probabilities.append(output[:, 1].cpu().numpy())

    return np.concatenate(probabilities)


def far_at_frr(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_frr: float = 0.01,
) -> dict[str, float | None]:
    speech_scores = np.sort(probabilities[labels == 1])
    if len(speech_scores) == 0:
        return {
            "target_frr": target_frr,
            "achieved_frr": None,
            "far": None,
            "threshold": None,
        }

    index = min(
        len(speech_scores) - 1,
        max(0, math.ceil(target_frr * len(speech_scores)) - 1),
    )
    threshold = speech_scores[index]
    prediction = probabilities >= threshold
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels, prediction.astype(np.int8), labels=[0, 1]).ravel()
    )
    return {
        "target_frr": target_frr,
        "achieved_frr": float(
            false_negative / max(false_negative + true_positive, 1)
        ),
        "far": float(false_positive / max(false_positive + true_negative, 1)),
        "threshold": float(threshold),
    }


def evaluate(
    checkpoint: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    noise_level: str,
) -> dict:
    started = time.perf_counter()
    model = load_legacy_checkpoint(checkpoint, map_location="cpu")
    probabilities = predict(model, windows, device)
    elapsed = time.perf_counter() - started

    prediction = probabilities >= 0.5
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels, prediction.astype(np.int8), labels=[0, 1]).ravel()
    )
    return {
        "checkpoint": checkpoint.name,
        "noise_level": noise_level,
        "parameters": num_parameters(model),
        "frames": int(len(labels)),
        "speech_fraction": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, prediction)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "speech_recall": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "nonspeech_recall": float(
            true_negative / max(true_negative + false_positive, 1)
        ),
        "confusion": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "operating_point": far_at_frr(labels, probabilities),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    if args.speech_minutes <= 0:
        raise ValueError("--speech-minutes must be positive.")

    rng = random.Random(args.seed)
    device = select_device(args.device)
    speech_rows = read_manifest(args.speech_manifest)
    noise_categories = {
        value.strip() for value in args.noise_categories.split(",") if value.strip()
    }
    noise_rows = [
        row
        for row in read_manifest(args.noise_manifest)
        if row["category"] in noise_categories
    ]
    if not speech_rows:
        raise RuntimeError(f"No speech rows found in {args.speech_manifest}")
    if not noise_rows:
        raise RuntimeError(
            f"No noise rows in categories {sorted(noise_categories)} "
            f"from {args.noise_manifest}"
        )

    target_frames = round(args.speech_minutes * 60.0 / FRAME_STEP_SECONDS)
    slices, speech_frames, silence_frames = build_balanced_slices(
        rng,
        speech_rows,
        noise_rows,
        args.data_root,
        target_frames,
    )
    print(
        f"Generated slices={len(slices)} "
        f"speech={speech_frames * FRAME_STEP_SECONDS:.1f}s "
        f"silence={silence_frames * FRAME_STEP_SECONDS:.1f}s "
        f"device={device} seed={args.seed}"
    )

    features, labels = build_features(slices)
    windows_by_level = {}
    targets_by_level = {}
    for noise_level, _ in NOISE_LEVELS:
        windows, targets = build_windows(features[noise_level], labels)
        windows_by_level[noise_level] = windows
        targets_by_level[noise_level] = targets
        print(
            f"Noise {noise_level:>4s}: frames={len(features[noise_level])} "
            f"windows={len(windows)} speech={targets.mean():.2%}"
        )

    reference_targets = next(iter(targets_by_level.values()))
    if any(
        not np.array_equal(reference_targets, targets)
        for targets in targets_by_level.values()
    ):
        raise RuntimeError("Internal label alignment error across noise levels.")

    results = []
    model_names = args.models or list(DEFAULT_MODELS)
    for model_name in model_names:
        checkpoint = args.model_dir / model_name
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for noise_level, _ in NOISE_LEVELS:
            result = evaluate(
                checkpoint,
                windows_by_level[noise_level],
                targets_by_level[noise_level],
                device,
                noise_level,
            )
            results.append(result)
            operating = result["operating_point"]
            print(
                f"{model_name:30s} noise={noise_level:>4s} "
                f"acc={result['accuracy']:.4f} auc={result['auc']:.5f} "
                f"FAR@FRR1%={operating['far']:.4f} "
                f"FRR={operating['achieved_frr']:.4f}"
            )

    payload = {
        "upstream_repository": "nicklashansen/voice-activity-detection",
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol": {
            "speech_dataset": "LibriSpeech test-clean",
            "noise_dataset": "MUSAN",
            "noise_categories": sorted(noise_categories),
            "noise_normalization": "peak normalized to 0 dBFS per sampled clip",
            "noise_levels": {
                name: (
                    "0 dB overlay gain; noise is still present"
                    if gain is None
                    else f"{gain:g} dB overlay gain applied to the noise bed"
                )
                for name, gain in NOISE_LEVELS
            },
            "sample_rate": SAMPLE_RATE,
            "frame_ms": 30,
            "mfcc_coefficients": 12,
            "delta_filter_width": 2,
            "sequence_frames": SEQUENCE_FRAMES,
            "center_frame_label": True,
            "batch_size": BATCH_SIZE,
            "step_size": 6,
            "seed": args.seed,
            "speech_minutes_target": args.speech_minutes,
            "speech_seconds_generated": speech_frames * FRAME_STEP_SECONDS,
            "silence_seconds_generated": silence_frames * FRAME_STEP_SECONDS,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
