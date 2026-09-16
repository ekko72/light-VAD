# -*- coding: utf-8 -*-
"""Deterministic speech/noise mixing following LibriVAD's protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from audio import Pcm16Reader, rms, rms_db, scale_to_pcm16


class NoiseCursor:
    """Consume a long noise waveform continuously, wrapping at EOF."""

    def __init__(self, reader: Pcm16Reader):
        self.reader = reader
        self.start = 0

    def next(self, length: int) -> tuple[int, np.ndarray]:
        start = self.start
        segment = self.reader.read(start, length)
        self.start = (start + int(length)) % self.reader.frames
        return start, segment

    def next_non_silent(
        self, length: int, max_attempts: int = 4096
    ) -> tuple[int, np.ndarray, int]:
        """Return the next non-silent segment and count skipped witnesses.

        Some MUSAN recordings contain long digital-silence runs. A silent
        noise window has no defined SNR, so advance the deterministic cursor
        until a usable window is found.
        """
        if length <= 0:
            raise ValueError("length must be positive")
        for skipped in range(max_attempts):
            start, segment = self.next(length)
            if rms(segment) > 0.0:
                return start, segment, skipped
        raise ValueError(
            f"no non-silent noise segment found after {max_attempts} attempts"
        )


@dataclass(frozen=True)
class MixResult:
    output: np.ndarray
    scale: float
    speech_rms_db: float
    noise_rms_db: float
    actual_snr_db: float


def mix_speech_and_noise(
    speech: np.ndarray,
    label: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> MixResult:
    """Mix int16 speech/noise at a speech-region RMS SNR."""
    speech_values = np.asarray(speech, dtype=np.int16)
    label_values = np.asarray(label)
    noise_values = np.asarray(noise, dtype=np.int16)
    if speech_values.ndim != 1 or noise_values.ndim != 1:
        raise ValueError("speech and noise must be one-dimensional")
    if speech_values.size != label_values.size:
        raise ValueError("speech and label lengths differ")
    if speech_values.size != noise_values.size:
        raise ValueError("speech and noise lengths differ")

    speech_only = speech_values[label_values == 1]
    if speech_only.size == 0:
        raise ValueError("label contains no speech samples")

    speech_rms = rms(speech_only)
    noise_rms = rms(noise_values)
    if speech_rms <= 0.0:
        raise ValueError("speech region is silent")
    if noise_rms <= 0.0:
        raise ValueError("noise segment is silent")

    scale = speech_rms / (
        noise_rms * (10.0 ** (float(target_snr_db) / 20.0))
    )
    corrupted = (
        speech_values.astype(np.float64)
        + scale * noise_values.astype(np.float64)
    )
    output = scale_to_pcm16(corrupted)
    speech_rms_db = rms_db(speech_only)
    noise_rms_db = rms_db(noise_values)
    scaled_noise_rms_db = noise_rms_db + float(
        20.0 * np.log10(scale)
    )
    return MixResult(
        output=output,
        scale=float(scale),
        speech_rms_db=speech_rms_db,
        noise_rms_db=noise_rms_db,
        actual_snr_db=speech_rms_db - scaled_noise_rms_db,
    )


def stable_mix_hash(
    *,
    variant: str,
    split: str,
    utterance_ids: Iterable[str],
    noise_name: str,
    target_snr_db: int,
    noise_start_sample: int,
    audio: np.ndarray,
    label: np.ndarray,
) -> str:
    """Hash the generated arrays and the provenance fields that define them."""
    digest = hashlib.sha256()
    fields = (
        variant,
        split,
        "|".join(utterance_ids),
        noise_name,
        str(target_snr_db),
        str(noise_start_sample),
        str(np.asarray(audio).size),
        str(np.asarray(label).size),
    )
    for value in fields:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    digest.update(np.asarray(audio, dtype="<i2").tobytes())
    digest.update(np.asarray(label, dtype="<i2").tobytes())
    return digest.hexdigest()
