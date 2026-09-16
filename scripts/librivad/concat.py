# -*- coding: utf-8 -*-
"""Deterministic implementation of LibriVAD's Concat variant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from alignments import (
    SilenceInterval,
    label_from_alignment,
    load_alignment,
    silence_intervals,
)
from audio import Pcm16Reader, read_segment_pcm16, scale_to_pcm16
from catalog import ConcatEntry, SpeechEntry
from config import SAMPLE_RATE


@dataclass(frozen=True)
class ConcatData:
    """The clean Concat waveform and its sample-level label."""

    audio: np.ndarray
    label: np.ndarray
    first_frames: int
    second_frames: int
    middle_silence_frames: int


class SilencePool:
    """A lazy, deterministic stream of silent intervals from train utterances.

    Upstream concatenates those intervals in parallel and then slices the
    resulting signal. We preserve the same interval order but only read the
    source audio when a Concat pair asks for it.
    """

    def __init__(self, sources: list[SpeechEntry]):
        self.sources = sources
        self._segments: list[tuple[SpeechEntry, SilenceInterval]] = []
        self._source_index = 0
        self._segment_index = 0
        self._offset = 0
        self._total_length = 0
        self._consumed = 0
        self._all_sources_scanned = False

    @property
    def scanned_length(self) -> int:
        return self._total_length

    @property
    def consumed(self) -> int:
        """Monotonic sample offset of the next read in the silence stream."""
        return self._consumed

    def _scan_next_source(self) -> bool:
        if self._source_index >= len(self.sources):
            self._all_sources_scanned = True
            return False
        source = self.sources[self._source_index]
        self._source_index += 1
        frames = int(sf.info(str(source.audio_path)).frames)
        alignment = load_alignment(source.alignment_path)
        for interval in silence_intervals(
            alignment, frames, sample_rate=SAMPLE_RATE
        ):
            if interval.length > 0:
                self._segments.append((source, interval))
                self._total_length += interval.length
        if self._source_index >= len(self.sources):
            self._all_sources_scanned = True
        return True

    def _scan_until_available(self, index: int) -> None:
        while index >= len(self._segments):
            if not self._scan_next_source():
                return

    def read(self, length: int) -> np.ndarray:
        """Read sequentially from the concatenated silence stream."""
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return np.empty(0, dtype=np.int16)
        self._consumed += int(length)
        if self._segment_index >= len(self._segments):
            self._scan_until_available(self._segment_index)
        if not self._segments:
            raise ValueError("No silence segments were extracted")

        # If the request has reached EOF, scan the rest so we can determine
        # whether wrapping is valid and reproduce the upstream modulo cursor.
        remaining = int(length)
        chunks: list[np.ndarray] = []
        while remaining:
            if self._segment_index >= len(self._segments):
                self._scan_until_available(self._segment_index)
                if self._segment_index >= len(self._segments):
                    if not self._all_sources_scanned:
                        self._scan_next_source()
                    if self._segment_index >= len(self._segments):
                        if int(length) > self._total_length:
                            raise ValueError(
                                "Generated silence signal is too short for "
                                "the required length"
                            )
                        self._segment_index = 0
                        self._offset = 0

            source, interval = self._segments[self._segment_index]
            available = interval.length - self._offset
            take = min(remaining, available)
            if take:
                chunks.append(
                    read_segment_pcm16(
                        source.audio_path,
                        interval.start + self._offset,
                        take,
                    )
                )
                remaining -= take
                self._offset += take
            if self._offset >= interval.length:
                self._segment_index += 1
                self._offset = 0
        if len(chunks) == 1:
            return chunks[0]
        return np.ascontiguousarray(np.concatenate(chunks))


def build_concat_data(
    entry: ConcatEntry,
    silence_pool: SilencePool,
) -> ConcatData:
    """Build one Concat example using the upstream label and mix rules."""
    first, second = entry.sources
    wav1, wav2 = Pcm16Reader(first.audio_path), Pcm16Reader(second.audio_path)
    try:
        audio1 = wav1.read(0, wav1.frames)
        audio2 = wav2.read(0, wav2.frames)
    finally:
        wav1.close()
        wav2.close()

    label1 = label_from_alignment(
        load_alignment(first.alignment_path),
        len(audio1),
        SAMPLE_RATE,
    )
    label2 = label_from_alignment(
        load_alignment(second.alignment_path),
        len(audio2),
        SAMPLE_RATE,
    )
    middle_frames = int((len(audio1) + len(audio2)) / 4)
    middle = silence_pool.read(middle_frames)
    combined = np.concatenate(
        (
            audio1.astype(np.float64),
            0.001 * middle.astype(np.float64),
            audio2.astype(np.float64),
        )
    )
    output = scale_to_pcm16(combined)
    label = np.concatenate(
        (
            label1,
            np.zeros(middle_frames, dtype=np.int16),
            label2,
        )
    )
    if output.size != label.size:
        raise AssertionError(
            f"Concat length mismatch for {entry.filename}: "
            f"{output.size} != {label.size}"
        )
    return ConcatData(
        audio=np.ascontiguousarray(output),
        label=np.ascontiguousarray(label),
        first_frames=len(audio1),
        second_frames=len(audio2),
        middle_silence_frames=middle_frames,
    )
