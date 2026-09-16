# -*- coding: utf-8 -*-
"""Protocol-parity checks against a transcription of upstream LibriVAD.

The upstream functions below are transcribed verbatim (modulo formatting) from
the pinned commit so the comparison stays runnable after the temporary
upstream checkout is gone:

* ``upstream_label``  <- ``Scripts/create_labels.py::_generate_label_from_timeframes``
* ``upstream_pair_name`` <- ``Scripts/create_LibriSpeechConcat.py::process_pair``
* ``upstream_mix`` <- ``create_LibriVAD.py::create_LibriVAD`` mixing loop
* ``UpstreamNoiseCursor`` <- the ``noise_idx`` / ``np.tile`` recycling logic

Usage::

    python scripts/librivad/parity.py --samples 25
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import textgrids

from alignments import label_from_alignment, load_alignment
from audio import Pcm16Reader
from catalog import list_speech_entries, make_concat_entries
from config import (
    ALIGNMENT_ROOT,
    LIBRISPEECH_ROOT,
    NOISES_ROOT,
    NOISE_NAMES,
    SAMPLE_RATE,
    SPLIT_SOURCE_DIRS,
)
from mix import NoiseCursor, mix_speech_and_noise


def upstream_label(
    timeframes: np.ndarray,
    words: list[str | None],
    target_samples: int,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Transcription of upstream ``_generate_label_from_timeframes``."""
    if target_samples <= 0:
        return np.array([], dtype=np.int16)
    label = np.ones(target_samples, dtype=np.int16)
    if timeframes.size == 0:
        return np.zeros(target_samples, dtype=np.int16)
    is_silent = [word == "" or word is None for word in words]
    silent_indices = np.where(is_silent)[0]
    if silent_indices.size > 0:
        if silent_indices[0] == 0:
            end_sample = min(int(timeframes[0] * sample_rate), target_samples)
            label[0:end_sample] = 0
        for index in silent_indices:
            if index > 0:
                start_sample = min(
                    int(timeframes[index - 1] * sample_rate), target_samples
                )
                end_sample = min(
                    int(timeframes[index] * sample_rate), target_samples
                )
                label[start_sample:end_sample] = 0
    last_word_end_sample = int(timeframes[-1] * sample_rate)
    if last_word_end_sample < target_samples:
        label[last_word_end_sample:] = 0
    return label


def upstream_timeframes(alignment_path: Path) -> tuple[np.ndarray, list]:
    grid = textgrids.TextGrid(str(alignment_path))
    words_tier = grid["words"]
    timeframes = np.array(
        [interval.xmax for interval in words_tier], dtype=np.float32
    )
    words = [interval.text for interval in words_tier]
    return timeframes, words


def upstream_pair_name(path1: Path, path2: Path) -> str:
    """Transcription of upstream's Concat output filename rules."""
    if path1.parts[-3] != path2.parts[-3]:
        return f"{path1.stem}_+_{path2.name}"
    if path1.parts[-2] != path2.parts[-2]:
        return f"{path1.stem}_+_{'-'.join(path2.stem.split('-')[1:])}.wav"
    return f"{path1.stem}_+_{path2.stem.split('-')[-1]}.wav"


def upstream_mix(
    signal: np.ndarray,
    noise_subset: np.ndarray,
    rms_speech: float,
    snr: float,
) -> tuple[np.ndarray, float]:
    """Transcription of upstream's mixing + ``scale_signal``."""
    scale = rms_speech / (
        float(np.sqrt(np.mean(np.asarray(noise_subset, dtype=np.float64) ** 2)))
        * (10 ** (snr / 20))
    )
    corrupted_signal = signal + scale * noise_subset
    max_abs = np.max(np.abs(corrupted_signal))
    if max_abs == 0:
        return corrupted_signal.astype(np.int16), scale
    return np.int16(corrupted_signal / max_abs * 32767), scale


class UpstreamNoiseCursor:
    """Transcription of upstream's per (noise, SNR) cursor."""

    def __init__(self, noise: np.ndarray):
        self.noise = np.asarray(noise, dtype=np.int16)
        self.duration = len(self.noise)
        self.tiled = np.tile(self.noise, 2)
        self.index = 0

    def next(self, length: int) -> tuple[int, np.ndarray]:
        start = self.index
        if self.index + length > self.duration:
            subset = self.tiled[self.index : self.index + length]
            self.index = (self.index + length) % self.duration
        else:
            subset = self.noise[self.index : self.index + length]
            self.index += length
        return start, subset


def check_labels(samples: int, seed: int, roots: dict) -> list[str]:
    problems: list[str] = []
    rng = random.Random(seed)
    checked = 0
    for split, split_dir in SPLIT_SOURCE_DIRS.items():
        entries = list_speech_entries(
            split,
            librispeech_root=roots["librispeech"],
            alignment_root=roots["alignment"],
        )
        take = min(samples, len(entries))
        for entry in rng.sample(entries, take):
            frames = int(sf.info(str(entry.audio_path)).frames)
            timeframes, words = upstream_timeframes(entry.alignment_path)
            expected = upstream_label(timeframes, words, frames, SAMPLE_RATE)
            actual = label_from_alignment(
                load_alignment(entry.alignment_path), frames, SAMPLE_RATE
            )
            checked += 1
            if expected.size != actual.size:
                problems.append(
                    f"label length mismatch for {entry.relative_path}: "
                    f"{expected.size} != {actual.size}"
                )
                continue
            differing = int(np.count_nonzero(expected != actual))
            if differing:
                problems.append(
                    f"label mismatch for {entry.relative_path}: "
                    f"{differing}/{frames} samples differ"
                )
    print(f"[labels] {checked} utterances compared with upstream")
    return problems


def check_concat_names(roots: dict) -> list[str]:
    problems: list[str] = []
    checked = 0
    for split in SPLIT_SOURCE_DIRS:
        entries = list_speech_entries(
            split,
            librispeech_root=roots["librispeech"],
            alignment_root=roots["alignment"],
        )
        for concat in make_concat_entries(entries):
            first, second = concat.sources
            # Upstream runs on ``.wav`` copies, so compare against the same
            # paths with a ``.wav`` suffix (its different-speaker rule keeps
            # the second file's extension literally).
            expected = upstream_pair_name(
                first.audio_path.with_suffix(".wav"),
                second.audio_path.with_suffix(".wav"),
            )
            # Local pair names keep the source suffix; the generator writes
            # ``.wav``, which is what upstream produces directly.
            actual = concat.output_relative_path.with_suffix(".wav").name
            checked += 1
            if expected != actual:
                problems.append(
                    f"concat name mismatch: {actual} != upstream {expected}"
                )
            expected_dir = second.relative_path.parent
            actual_dir = concat.output_relative_path.parent
            if expected_dir != actual_dir:
                problems.append(
                    f"concat dir mismatch for {actual}: {actual_dir} != "
                    f"upstream {expected_dir}"
                )
    print(f"[concat] {checked} pairings compared with upstream")
    return problems


def check_mix(
    samples: int, seed: int, roots: dict, noises: list[str]
) -> list[str]:
    problems: list[str] = []
    rng = random.Random(seed)
    split = "val"
    entries = list_speech_entries(
        split,
        librispeech_root=roots["librispeech"],
        alignment_root=roots["alignment"],
    )
    if not entries:
        return [f"no {split} utterances available for mixing check"]
    take = min(samples, len(entries))
    chosen = rng.sample(entries, take)
    for name in noises:
        noise_path = roots["noises"] / name / f"{name}_{split}.wav"
        if not noise_path.is_file():
            problems.append(f"noise not available: {noise_path}")
            continue
        with Pcm16Reader(noise_path) as reader:
            noise = reader.read(0, reader.frames)
            for snr in (0, 10):
                # Upstream recreates the noise cursor for every (noise, SNR)
                # pass; matching that reset is required for byte parity.
                cursor = NoiseCursor(reader)
                upstream_cursor = UpstreamNoiseCursor(noise)
                for entry in chosen:
                    audio = Pcm16Reader(entry.audio_path)
                    try:
                        signal = audio.read(0, audio.frames)
                    finally:
                        audio.close()
                    label = label_from_alignment(
                        load_alignment(entry.alignment_path),
                        len(signal),
                        SAMPLE_RATE,
                    )
                    speech_only = signal[label == 1]
                    upstream_rms = float(
                        np.sqrt(
                            np.mean(
                                np.asarray(
                                    speech_only, dtype=np.float64
                                )
                                ** 2
                            )
                        )
                    )
                    ours_start, ours_noise = cursor.next(len(signal))
                    theirs_start, theirs_noise = upstream_cursor.next(
                        len(signal)
                    )
                    if ours_start != theirs_start:
                        problems.append(
                            f"noise cursor drift at {entry.relative_path} "
                            f"snr={snr}: {ours_start} != {theirs_start}"
                        )
                    if not np.array_equal(ours_noise, theirs_noise):
                        problems.append(
                            f"noise segment mismatch at "
                            f"{entry.relative_path} snr={snr}"
                        )
                    ours = mix_speech_and_noise(
                        signal, label, ours_noise, snr
                    )
                    theirs, _ = upstream_mix(
                        signal, theirs_noise, upstream_rms, snr
                    )
                    differing = int(
                        np.count_nonzero(
                            ours.output.astype(np.int32)
                            != theirs.astype(np.int32)
                        )
                    )
                    if differing:
                        problems.append(
                            f"mixture mismatch at {entry.relative_path} "
                            f"noise={name} snr={snr}: {differing} samples"
                        )
    print(
        f"[mix] {take} utterances x {len(noises)} noises x 2 SNRs "
        "compared with upstream"
    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local pipeline numerics with upstream LibriVAD."
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noises",
        default="Babble_noise,Domestic_noise",
        help="Comma-separated noises used for the mixing check.",
    )
    parser.add_argument(
        "--skip-mix", action="store_true", help="Labels and names only."
    )
    parser.add_argument(
        "--librispeech-root", type=Path, default=LIBRISPEECH_ROOT
    )
    parser.add_argument(
        "--alignment-root", type=Path, default=ALIGNMENT_ROOT
    )
    parser.add_argument("--noises-root", type=Path, default=NOISES_ROOT)
    args = parser.parse_args()

    roots = {
        "librispeech": args.librispeech_root,
        "alignment": args.alignment_root,
        "noises": args.noises_root,
    }
    noises = [item.strip() for item in args.noises.split(",") if item.strip()]
    unknown = [name for name in noises if name not in NOISE_NAMES]
    if unknown:
        parser.error(f"unknown noises: {unknown}")

    problems = check_labels(args.samples, args.seed, roots)
    problems += check_concat_names(roots)
    if not args.skip_mix:
        problems += check_mix(args.samples, args.seed, roots, noises)

    if problems:
        for message in problems[:20]:
            print(f"  FAIL {message}")
        if len(problems) > 20:
            print(f"  ... {len(problems) - 20} more")
        print(f"\nparity FAILED ({len(problems)} problems)")
        return 1
    print("\nparity PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
