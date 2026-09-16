# -*- coding: utf-8 -*-
"""Read LibriVAD-style manifests and cut MarbleNet segments.

Manifest columns come from ``scripts/librivad/generate.py``:

* ``output_audio_path``   -- WAV path relative to the generated root
* ``label_relative_path`` -- sample-level ``.npy`` path relative to label root
* ``output_frames``       -- sample count of the mixture

Two views are provided:

* :class:`LibriVADSegments` -- one item per 0.63 s window, used for training
  and window-level validation. The window label is the majority vote of the
  sample-level labels it covers.
* :class:`LibriVADUtterances` -- one item per mixture, used for the
  overlapping-window (87.5%) evaluation in :mod:`evaluate`.

Adaptation note: the paper trains on Speech Commands utterances (all speech)
plus Freesound clips (all non-speech), so it can take one central segment per
sample. A LibriVAD-style mixture already contains both speech and silence, so
segments are enumerated on a stride grid and labelled by majority vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_ROOT = REPO_ROOT / "data" / "librivad" / "generated"
DEFAULT_LABEL_ROOT = REPO_ROOT / "data" / "librivad" / "labels"
DEFAULT_SMOKE_ROOT = REPO_ROOT / "data" / "librivad" / "smoke"

INT16_SCALE = 1.0 / 32768.0


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a LibriVAD-style manifest TSV into a list of row dicts."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows: list[dict[str, str]] = []
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(dict(zip(header, line.split("\t"))))
    if not rows:
        raise ValueError(f"manifest has no data rows: {path}")
    return rows


def resolve_roots(
    generated_root: str | Path | None,
    label_root: str | Path | None,
    data_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve generated/label roots, with ``data_root`` as a shortcut."""
    if data_root is not None:
        base = Path(data_root)
        return (
            Path(generated_root) if generated_root else base / "generated",
            Path(label_root) if label_root else base / "labels",
        )
    return (
        Path(generated_root) if generated_root else DEFAULT_GENERATED_ROOT,
        Path(label_root) if label_root else DEFAULT_LABEL_ROOT,
    )


def load_row_audio(
    row: dict[str, str], generated_root: Path, label_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(int16 waveform, int16 sample labels)`` for one manifest row."""
    audio_path = generated_root / row["output_audio_path"]
    label_path = label_root / row["label_relative_path"]
    audio, sample_rate = sf.read(
        str(audio_path), dtype="int16", always_2d=False
    )
    if audio.ndim != 1:
        raise ValueError(
            f"expected mono audio, got {audio.shape}: {audio_path}"
        )
    if int(sample_rate) != 16_000:
        raise ValueError(
            f"expected 16 kHz audio, got {sample_rate}: {audio_path}"
        )
    labels = np.load(label_path)
    if labels.size != audio.size:
        raise ValueError(
            f"label/audio length mismatch ({labels.size} != {audio.size}): "
            f"{label_path}"
        )
    return np.ascontiguousarray(audio), np.ascontiguousarray(labels)


def read_audio_window(
    row: dict[str, str], generated_root: Path, start: int, length: int
) -> np.ndarray:
    """Read one int16 window without scanning the rest of a long mixture."""
    audio_path = generated_root / row["output_audio_path"]
    audio, sample_rate = sf.read(
        str(audio_path),
        start=int(start),
        frames=int(length),
        dtype="int16",
        always_2d=False,
    )
    audio = np.asarray(audio, dtype=np.int16)
    if audio.ndim != 1:
        raise ValueError(
            f"expected mono audio, got {audio.shape}: {audio_path}"
        )
    if int(sample_rate) != 16_000:
        raise ValueError(
            f"expected 16 kHz audio, got {sample_rate}: {audio_path}"
        )
    if audio.size != int(length):
        raise ValueError(
            f"short read from {audio_path}: {audio.size} != {length}"
        )
    return np.ascontiguousarray(audio)


def window_starts(
    n_samples: int, segment_samples: int, hop_samples: int
) -> list[int]:
    """Stride-grid window starts, with a final window flush to the end."""
    if n_samples < segment_samples:
        return []
    starts = list(range(0, n_samples - segment_samples + 1, hop_samples))
    tail = n_samples - segment_samples
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def majority_label(labels: np.ndarray) -> int:
    """``1`` when at least half of the covered samples are speech."""
    speech = int(np.count_nonzero(labels == 1))
    return int(speech * 2 >= labels.size)


def stratified_row_indices(
    rows: list[dict[str, str]],
    count: int | None,
    *,
    seed: int = 0,
) -> list[int]:
    """Choose manifest rows uniformly across noise/SNR conditions.

    Taking the first ``count`` rows biases training toward the first noise
    category and the first SNR.  This helper shuffles every
    ``(noise_name, snr_db)`` group and then draws one row from each group in
    round-robin order until ``count`` rows are selected.
    """
    if count is None or count >= len(rows):
        return list(range(len(rows)))
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (
            str(row.get("noise_name", "")),
            str(row.get("snr_db", "")),
        )
        groups.setdefault(key, []).append(index)

    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    for key in keys:
        values = np.asarray(groups[key], dtype=np.int64)
        groups[key] = rng.permutation(values).astype(int).tolist()

    selected: list[int] = []
    cursors = {key: 0 for key in keys}
    while len(selected) < count:
        progressed = False
        for key in keys:
            cursor = cursors[key]
            values = groups[key]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[key] = cursor + 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    selected.sort()
    return selected


def speech_ratio(labels: np.ndarray) -> float:
    return float(np.count_nonzero(labels == 1)) / float(labels.size)


def audio_frames_for_row(
    row: dict[str, str], generated_root: Path
) -> int:
    """Read and validate one row's audio header without decoding samples."""
    audio_path = generated_root / row["output_audio_path"]
    info = sf.info(str(audio_path))
    if int(info.samplerate) != 16_000:
        raise ValueError(
            f"expected 16 kHz audio, got {info.samplerate}: {audio_path}"
        )
    if int(info.channels) != 1:
        raise ValueError(
            f"expected mono audio, got {info.channels} channels: {audio_path}"
        )
    return int(info.frames)


@dataclass(frozen=True)
class SegmentRef:
    """One window: which row it came from, where it starts, its label."""

    row: int
    start: int
    label: int


class _RowCache:
    """Tiny LRU cache so each mixture is decoded once per pass."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        generated_root: Path,
        label_root: Path,
        maxsize: int = 8,
    ) -> None:
        self.rows = rows
        self.generated_root = generated_root
        self.label_root = label_root
        self.maxsize = max(1, int(maxsize))
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._order: list[int] = []

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(index)
        if cached is not None:
            self._order.remove(index)
            self._order.append(index)
            return cached
        audio, labels = load_row_audio(
            self.rows[index], self.generated_root, self.label_root
        )
        self._cache[index] = (audio, labels)
        self._order.append(index)
        while len(self._order) > self.maxsize:
            evicted = self._order.pop(0)
            self._cache.pop(evicted, None)
        return audio, labels


class LibriVADSegments(Dataset):
    """Window-level dataset over a LibriVAD-style manifest."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        generated_root: str | Path | None = None,
        label_root: str | Path | None = None,
        data_root: str | Path | None = None,
        segment_samples: int = 10_080,
        stride_samples: int = 2_400,
        limit: int | None = None,
        row_sample: int | None = None,
        balance: bool = False,
        purity: float = 0.0,
        seed: int = 0,
        cache_size: int = 8,
    ) -> None:
        self.manifest = Path(manifest)
        self.generated_root, self.label_root = resolve_roots(
            generated_root, label_root, data_root
        )
        self.segment_samples = int(segment_samples)
        self.stride_samples = int(stride_samples)
        self.rows = read_manifest(self.manifest)
        if limit is not None:
            self.rows = self.rows[: max(0, int(limit))]
        row_indices = stratified_row_indices(
            self.rows, row_sample, seed=seed
        )
        self.rows = [self.rows[index] for index in row_indices]

        self.segments: list[SegmentRef] = []
        skipped_short = 0
        label_cache: dict[str, np.ndarray] = {}
        for row_index in range(len(self.rows)):
            row = self.rows[row_index]
            label_key = row["label_relative_path"]
            labels = label_cache.get(label_key)
            if labels is None:
                labels = np.load(self.label_root / label_key)
                if labels.ndim != 1 or labels.dtype != np.int16:
                    raise ValueError(
                        f"expected a 1-D int16 label array: {label_key}"
                    )
                label_cache[label_key] = labels
            frames = audio_frames_for_row(row, self.generated_root)
            if labels.size != frames:
                raise ValueError(
                    f"label/audio length mismatch ({labels.size} != {frames}): "
                    f"{label_key}"
                )
            starts = window_starts(
                frames, self.segment_samples, self.stride_samples
            )
            if not starts:
                skipped_short += 1
                continue
            for start in starts:
                covered = labels[start : start + self.segment_samples]
                if purity > 0.0 and abs(speech_ratio(covered) - 0.5) < purity:
                    continue
                self.segments.append(
                    SegmentRef(row_index, start, majority_label(covered))
                )
        del label_cache
        self.skipped_short = skipped_short
        self.unbalanced_segments = len(self.segments)
        if balance:
            self.segments = self._balance(self.segments, seed)
        self.balanced_segments = len(self.segments)

    @staticmethod
    def _balance(segments: list[SegmentRef], seed: int) -> list[SegmentRef]:
        """Subsample the larger class so both classes have equal counts."""
        positives = [item for item in segments if item.label == 1]
        negatives = [item for item in segments if item.label == 0]
        target = min(len(positives), len(negatives))
        if target == 0:
            return list(segments)
        rng = np.random.default_rng(seed)
        kept: list[SegmentRef] = []
        for group in (positives, negatives):
            order = rng.permutation(len(group))[:target]
            kept.extend(group[int(index)] for index in sorted(order))
        kept.sort(key=lambda item: (item.row, item.start))
        return kept

    def label_counts(self) -> tuple[int, int]:
        speech = sum(1 for item in self.segments if item.label == 1)
        return speech, len(self.segments) - speech

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment = self.segments[index]
        window = read_audio_window(
            self.rows[segment.row],
            self.generated_root,
            segment.start,
            self.segment_samples,
        ).astype(np.float32)
        waveform = torch.from_numpy(window * INT16_SCALE)
        return waveform, torch.tensor(segment.label, dtype=torch.long)


class LibriVADUtterances(Dataset):
    """One item per mixture, for overlapping-window evaluation."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        generated_root: str | Path | None = None,
        label_root: str | Path | None = None,
        data_root: str | Path | None = None,
        segment_samples: int = 10_080,
        hop_samples: int = 1_260,
        limit: int | None = None,
        row_sample: int | None = None,
        seed: int = 0,
        cache_size: int = 2,
    ) -> None:
        self.manifest = Path(manifest)
        self.generated_root, self.label_root = resolve_roots(
            generated_root, label_root, data_root
        )
        self.segment_samples = int(segment_samples)
        self.hop_samples = int(hop_samples)
        rows = read_manifest(self.manifest)
        row_indices = stratified_row_indices(
            rows, row_sample, seed=seed
        )
        rows = [rows[index] for index in row_indices]
        self._rows = _RowCache(
            rows, self.generated_root, self.label_root, cache_size
        )

        self.indices: list[int] = []
        skipped_short = 0
        for index in range(len(rows)):
            frames = audio_frames_for_row(rows[index], self.generated_root)
            if frames < self.segment_samples:
                skipped_short += 1
                continue
            self.indices.append(index)
            if limit is not None and len(self.indices) >= int(limit):
                break
        self.skipped_short = skipped_short
        self.rows = rows

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_index = self.indices[index]
        audio, labels = self._rows[row_index]
        starts = window_starts(
            audio.size, self.segment_samples, self.hop_samples
        )
        return {
            "row_index": row_index,
            "sample_id": self.rows[row_index].get("sample_id", ""),
            "noise_name": self.rows[row_index].get("noise_name", ""),
            "snr_db": self.rows[row_index].get("snr_db", ""),
            "waveform": torch.from_numpy(audio.astype(np.float32) * INT16_SCALE),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "starts": torch.tensor(starts, dtype=torch.long),
        }
