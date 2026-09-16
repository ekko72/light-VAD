# -*- coding: utf-8 -*-
"""PCM audio helpers used by the LibriVAD-compatible data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE


def read_pcm16(path: str | Path, expected_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read a mono PCM file as int16 without changing the sample values."""
    path = Path(path)
    info = sf.info(str(path))
    if int(info.samplerate) != expected_sr:
        raise ValueError(
            f"{path} has sample rate {info.samplerate}, expected {expected_sr}"
        )
    if int(info.channels) != 1:
        raise ValueError(f"{path} is not mono (channels={info.channels})")
    data, samplerate = sf.read(
        str(path), dtype="int16", always_2d=False
    )
    if int(samplerate) != expected_sr:
        raise ValueError(
            f"{path} changed sample rate while reading: {samplerate}"
        )
    data = np.asarray(data, dtype=np.int16)
    if data.ndim != 1:
        raise ValueError(f"{path} did not decode as a mono waveform")
    return np.ascontiguousarray(data)


class Pcm16Reader:
    """Seekable mono PCM16 reader with explicit wrapping support."""

    def __init__(self, path: str | Path, expected_sr: int = SAMPLE_RATE):
        self.path = Path(path)
        self.info = sf.info(str(self.path))
        self.samplerate = int(self.info.samplerate)
        self.channels = int(self.info.channels)
        self.frames = int(self.info.frames)
        if self.samplerate != expected_sr:
            raise ValueError(
                f"{self.path} has sample rate {self.samplerate}, "
                f"expected {expected_sr}"
            )
        if self.channels != 1:
            raise ValueError(
                f"{self.path} is not mono (channels={self.channels})"
            )
        self._file = sf.SoundFile(str(self.path), mode="r")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "Pcm16Reader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read(self, start: int, length: int) -> np.ndarray:
        """Read ``length`` samples starting at ``start``, wrapping at EOF."""
        if length < 0:
            raise ValueError("length must be non-negative")
        if self.frames <= 0:
            raise ValueError(f"{self.path} contains no samples")
        if length == 0:
            return np.empty(0, dtype=np.int16)

        start = int(start) % self.frames
        remaining = int(length)
        chunks: list[np.ndarray] = []
        while remaining:
            first_count = min(remaining, self.frames - start)
            self._file.seek(start)
            chunk = self._file.read(
                first_count, dtype="int16", always_2d=False
            )
            chunk = np.asarray(chunk, dtype=np.int16)
            if chunk.ndim != 1:
                raise ValueError(
                    f"{self.path} did not decode as a mono waveform"
                )
            if chunk.size != first_count:
                raise IOError(
                    f"short read from {self.path}: "
                    f"{chunk.size} != {first_count}"
                )
            chunks.append(np.ascontiguousarray(chunk))
            remaining -= first_count
            start = 0
        if len(chunks) == 1:
            return chunks[0]
        return np.ascontiguousarray(np.concatenate(chunks))


def read_segment_pcm16(
    path: str | Path, start: int, length: int
) -> np.ndarray:
    """Read one segment without retaining an open file handle."""
    with Pcm16Reader(path) as reader:
        return reader.read(start, length)


def rms(signal: np.ndarray) -> float:
    """Return RMS in the sample-value domain, matching upstream's RMS()."""
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def rms_db(signal: np.ndarray) -> float:
    """Return 20 log10(RMS) for an int16-domain signal."""
    value = rms(signal)
    if value <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def scale_to_pcm16(signal: np.ndarray) -> np.ndarray:
    """Scale a signal to int16 full scale, matching upstream scale_signal()."""
    values = np.asarray(signal)
    if values.size == 0:
        return np.empty(0, dtype=np.int16)
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return values.astype(np.int16)
    return (values / max_abs * 32767.0).astype(np.int16)
