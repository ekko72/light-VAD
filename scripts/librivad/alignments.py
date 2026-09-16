# -*- coding: utf-8 -*-
"""Forced-alignment parsing and sample-level VAD label generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import textgrids

from config import SAMPLE_RATE


def sample_index(time_s: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Convert seconds to a sample index the same way upstream does.

    Upstream stores alignment times in a ``float32`` array and truncates the
    product towards zero, so this helper keeps the same rounding behaviour.
    """
    return int(np.float32(time_s) * np.float32(sample_rate))


@dataclass(frozen=True)
class Alignment:
    """The ``words`` tier data needed by the upstream label algorithm."""

    path: Path
    end_times: tuple[float, ...]
    texts: tuple[str | None, ...]


@dataclass(frozen=True)
class SilenceInterval:
    """A half-open sample interval in one LibriSpeech utterance."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def load_alignment(path: str | Path) -> Alignment:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Alignment not found: {path}")
    grid = textgrids.TextGrid(str(path))
    if "words" not in grid:
        raise ValueError(f"Alignment has no 'words' tier: {path}")
    intervals = list(grid["words"])
    end_times = tuple(float(interval.xmax) for interval in intervals)
    texts = tuple(
        None if interval.text is None else str(interval.text)
        for interval in intervals
    )
    if any(
        right < left
        for left, right in zip(end_times, end_times[1:])
    ):
        raise ValueError(f"Non-monotonic words tier: {path}")
    return Alignment(path=path, end_times=end_times, texts=texts)


def label_from_alignment(
    alignment: Alignment,
    target_samples: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Create the upstream sample-level label (1=speech, 0=silence)."""
    if target_samples <= 0:
        return np.empty(0, dtype=np.int16)
    label = np.ones(int(target_samples), dtype=np.int16)
    if not alignment.end_times:
        return np.zeros(int(target_samples), dtype=np.int16)

    silent_indices = [
        index
        for index, text in enumerate(alignment.texts)
        if text is None or text == ""
    ]
    if silent_indices and silent_indices[0] == 0:
        end_sample = min(
            sample_index(alignment.end_times[0], sample_rate),
            int(target_samples),
        )
        label[:end_sample] = 0

    for index in silent_indices:
        if index <= 0:
            continue
        start_sample = min(
            sample_index(alignment.end_times[index - 1], sample_rate),
            int(target_samples),
        )
        end_sample = min(
            sample_index(alignment.end_times[index], sample_rate),
            int(target_samples),
        )
        label[start_sample:end_sample] = 0

    last_end_sample = sample_index(alignment.end_times[-1], sample_rate)
    if last_end_sample < target_samples:
        label[last_end_sample:] = 0
    return label


def silence_intervals(
    alignment: Alignment,
    target_samples: int,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[SilenceInterval, ...]:
    """Return the exact silence segments used by upstream's Concat pool."""
    if target_samples <= 0 or not alignment.end_times:
        return ()
    silent_indices = [
        index
        for index, text in enumerate(alignment.texts)
        if text is None or text == ""
    ]
    intervals: list[SilenceInterval] = []
    if not silent_indices:
        return ()

    if silent_indices[0] == 0:
        end = min(
            sample_index(alignment.end_times[0], sample_rate),
            int(target_samples),
        )
        if end > 0:
            intervals.append(SilenceInterval(0, end))
        silent_indices = silent_indices[1:]

    for index in silent_indices:
        if index <= 0:
            continue
        start = min(
            sample_index(alignment.end_times[index - 1], sample_rate),
            int(target_samples),
        )
        end = min(
            sample_index(alignment.end_times[index], sample_rate),
            int(target_samples),
        )
        if end > start:
            intervals.append(SilenceInterval(start, end))
    return tuple(intervals)


def label_counts(label: np.ndarray) -> tuple[int, int]:
    values = np.asarray(label)
    speech = int(np.count_nonzero(values == 1))
    silence = int(np.count_nonzero(values == 0))
    return speech, silence
