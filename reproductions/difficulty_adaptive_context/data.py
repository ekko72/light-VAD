# -*- coding: utf-8 -*-
"""Frame-level data utilities for the temporal-context experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from reproductions.marblenet_vad.dataset import (
    INT16_SCALE,
    read_manifest,
    stratified_row_indices,
)


FRAME_HOP = 160
FRAME_WINDOW = 400
N_FFT = 512


def majority_label(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    return int(np.count_nonzero(labels >= 1) * 2 >= labels.size)


def causal_frame_labels(
    sample_labels: np.ndarray,
    n_frames: int,
    *,
    hop: int = FRAME_HOP,
    window: int = FRAME_WINDOW,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Align 25 ms labels to the causal MFCC output frames.

    ``MfccFrontend(causal=True)`` pads ``n_fft`` samples on the left and
    uses ``center=False``.  Torchaudio centers the shorter analysis window
    inside the FFT window, so frame ``t`` covers the original interval
    ``[t * hop - n_fft + (n_fft - window) // 2, ... + window)``.
    """
    starts, ends = causal_frame_bounds(
        n_frames, hop=hop, window=window, n_fft=n_fft
    )
    labels = np.asarray(sample_labels, dtype=np.int64).reshape(-1)
    starts = np.clip(starts, 0, labels.size)
    ends = np.clip(ends, 0, labels.size)
    result = np.zeros(int(n_frames), dtype=np.int64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end > start:
            result[index] = majority_label(labels[int(start) : int(end)])
    return result


def causal_frame_bounds(
    n_frames: int,
    *,
    hop: int = FRAME_HOP,
    window: int = FRAME_WINDOW,
    n_fft: int = N_FFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the original-sample interval seen by each causal MFCC frame."""
    if int(n_frames) < 0:
        raise ValueError("n_frames must be non-negative")
    if int(hop) <= 0 or int(window) <= 0 or int(n_fft) <= 0:
        raise ValueError("hop, window and n_fft must be positive")
    if int(window) > int(n_fft):
        raise ValueError("window must not exceed n_fft")
    if (int(n_fft) - int(window)) % 2:
        raise ValueError("n_fft - window must be even for symmetric windowing")
    window_offset = (int(n_fft) - int(window)) // 2
    starts = (
        np.arange(int(n_frames), dtype=np.int64) * int(hop)
        - int(n_fft)
        + window_offset
    )
    return starts, starts + int(window)


def frame_times_ms(
    n_frames: int,
    *,
    hop: int = FRAME_HOP,
    window: int = FRAME_WINDOW,
    n_fft: int = N_FFT,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Return the center timestamp of each causal MFCC frame in ms."""
    starts, _ = causal_frame_bounds(
        n_frames, hop=hop, window=window, n_fft=n_fft
    )
    centers = starts.astype(np.float64) + 0.5 * int(window)
    return centers * 1000.0 / float(sample_rate)


def audio_frame_count(path: str | Path) -> int:
    info = sf.info(str(path))
    if int(info.samplerate) != 16_000:
        raise ValueError(f"expected 16 kHz audio, got {info.samplerate}: {path}")
    if int(info.channels) != 1:
        raise ValueError(f"expected mono audio, got {info.channels}: {path}")
    return int(info.frames)


def read_int16_audio(path: str | Path) -> np.ndarray:
    waveform, sample_rate = sf.read(
        str(path), dtype="int16", always_2d=False
    )
    waveform = np.asarray(waveform, dtype=np.int16)
    if int(sample_rate) != 16_000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    if waveform.ndim != 1:
        raise ValueError(f"expected mono audio, got {waveform.shape}: {path}")
    return np.ascontiguousarray(waveform)


def read_window_with_left_pad(
    path: str | Path,
    *,
    start: int,
    length: int,
) -> tuple[np.ndarray, int]:
    """Read ``length`` samples starting at ``start``, zero-padding the left."""
    if length <= 0:
        raise ValueError("length must be positive")
    start = int(start)
    length = int(length)
    left_pad = max(0, -start)
    read_start = max(0, start)
    read_length = length - left_pad
    waveform, sample_rate = sf.read(
        str(path),
        start=read_start,
        frames=read_length,
        dtype="int16",
        always_2d=False,
    )
    waveform = np.asarray(waveform, dtype=np.int16)
    if int(sample_rate) != 16_000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    if waveform.ndim != 1:
        raise ValueError(f"expected mono audio, got {waveform.shape}: {path}")
    if waveform.size < read_length:
        waveform = np.pad(waveform, (0, read_length - waveform.size))
    if left_pad:
        waveform = np.pad(waveform, (left_pad, 0))
    return np.ascontiguousarray(waveform), left_pad


def resolve_generated_audio(row: dict[str, str], root: Path) -> Path:
    return root / row["output_audio_path"]


def resolve_label_path(row: dict[str, str], root: Path) -> Path:
    return root / row["label_relative_path"]


def resolve_clean_audio(row: dict[str, str], root: Path) -> Path:
    return root / row["split_dir"] / row["source_relative_path"]


def filter_noise_classes(
    rows: list[dict[str, str]],
    excluded_noise_names: set[str],
) -> list[dict[str, str]]:
    if not excluded_noise_names:
        return list(rows)
    return [
        row
        for row in rows
        if str(row.get("noise_name", "")) not in excluded_noise_names
    ]


class FrameChunkDataset(Dataset):
    """One context+target chunk per deterministic manifest row.

    Training can rebuild chunk starts at every epoch.  The paired Short/Long
    runs pass the same ``set_epoch`` value, so both models see identical
    chunks while the training data is not limited to one start per row.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        generated_root: str | Path,
        label_root: str | Path,
        context_samples: int,
        target_samples: int,
        row_sample: int | None = None,
        seed: int = 0,
        exclude_noise_names: set[str] | None = None,
        sample_offset: int = 0,
        resample_chunks: bool = False,
    ) -> None:
        if context_samples <= 0 or target_samples <= 0:
            raise ValueError("context_samples and target_samples must be positive")
        if context_samples % FRAME_HOP or target_samples % FRAME_HOP:
            raise ValueError(
                "context_samples and target_samples must be multiples of 160"
            )
        self.manifest = Path(manifest)
        self.generated_root = Path(generated_root)
        self.label_root = Path(label_root)
        self.context_samples = int(context_samples)
        self.target_samples = int(target_samples)
        self.seed = int(seed)
        self.sample_offset = int(sample_offset)
        self.resample_chunks = bool(resample_chunks)
        self._epoch = 0
        rows = read_manifest(self.manifest)
        rows = filter_noise_classes(
            rows, set(exclude_noise_names or set())
        )
        indices = stratified_row_indices(rows, row_sample, seed=seed)
        self.rows = [rows[index] for index in indices]
        self.audio_frames = [
            audio_frame_count(
                resolve_generated_audio(row, self.generated_root)
            )
            for row in self.rows
        ]

        self.chunks: list[tuple[int, int]] = []
        self._build_chunks(self._epoch)

    def _build_chunks(self, epoch: int) -> None:
        rng = np.random.default_rng(
            self.seed + self.sample_offset + int(epoch) * 1_000_003
        )
        chunks: list[tuple[int, int]] = []
        for row_index, frames in enumerate(self.audio_frames):
            max_start = frames - self.target_samples
            if max_start < 0:
                continue
            start = int(rng.integers(0, max_start // FRAME_HOP + 1))
            chunks.append((row_index, start * FRAME_HOP))
        self.chunks = chunks

    def set_epoch(self, epoch: int) -> None:
        """Rebuild training chunk starts for the paired epoch."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch
        if self.resample_chunks:
            self._build_chunks(epoch)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_index, target_start = self.chunks[index]
        row = self.rows[row_index]
        audio_path = resolve_generated_audio(row, self.generated_root)
        label_path = resolve_label_path(row, self.label_root)
        total_samples = self.context_samples + self.target_samples
        input_start = target_start - self.context_samples
        waveform, left_pad = read_window_with_left_pad(
            audio_path,
            start=input_start,
            length=total_samples,
        )

        source_labels = np.load(label_path)
        if source_labels.ndim != 1 or source_labels.dtype != np.int16:
            raise ValueError(f"expected a 1-D int16 label array: {label_path}")
        actual_start = max(0, input_start)
        actual_length = min(
            total_samples - left_pad,
            max(0, source_labels.size - actual_start),
        )
        padded_labels = np.zeros(total_samples, dtype=np.int16)
        padded_labels[
            left_pad : left_pad + actual_length
        ] = source_labels[actual_start : actual_start + actual_length]

        n_frames = total_samples // FRAME_HOP + 1
        labels = causal_frame_labels(padded_labels, n_frames)
        target_frame_start = self.context_samples // FRAME_HOP
        target_labels = labels[target_frame_start:]
        if target_labels.size != self.target_samples // FRAME_HOP + 1:
            raise RuntimeError("frame-label alignment produced an invalid size")

        return (
            torch.from_numpy(
                waveform.astype(np.float32) * INT16_SCALE
            ),
            torch.from_numpy(target_labels.astype(np.int64)),
        )


@dataclass(frozen=True)
class EvaluationItem:
    """One waveform to score with both context models."""

    sample_id: str
    source_key: str
    noise_name: str
    snr_db: str
    audio_path: Path
    label_path: Path
    is_clean: bool


def build_evaluation_items(
    manifest: str | Path,
    *,
    generated_root: str | Path,
    label_root: str | Path,
    librispeech_root: str | Path,
    row_sample: int | None,
    seed: int,
    include_clean: bool,
    limit: int | None = None,
) -> list[EvaluationItem]:
    rows = read_manifest(manifest)
    selected = stratified_row_indices(rows, row_sample, seed=seed)
    selected_rows = [rows[index] for index in selected]
    generated_root = Path(generated_root)
    label_root = Path(label_root)
    librispeech_root = Path(librispeech_root)

    items: list[EvaluationItem] = []
    clean_sources: set[str] = set()
    for row in selected_rows:
        source_key = (
            f"{row.get('split_dir', '')}/"
            f"{row.get('source_relative_path', '')}"
        )
        if include_clean and source_key not in clean_sources:
            clean_sources.add(source_key)
            items.append(
                EvaluationItem(
                    sample_id=row.get("sample_id", ""),
                    source_key=source_key,
                    noise_name="Clean",
                    snr_db="clean",
                    audio_path=resolve_clean_audio(row, librispeech_root),
                    label_path=resolve_label_path(row, label_root),
                    is_clean=True,
                )
            )
        items.append(
            EvaluationItem(
                sample_id=row.get("sample_id", ""),
                source_key=source_key,
                noise_name=row.get("noise_name", ""),
                snr_db=row.get("snr_db", ""),
                audio_path=resolve_generated_audio(row, generated_root),
                label_path=resolve_label_path(row, label_root),
                is_clean=False,
            )
        )
        if limit is not None and len(items) >= int(limit):
            break
    return items
