# -*- coding: utf-8 -*-
"""Generate LibriVAD-compatible noise mixtures for light-VAD.

Layout mirrors upstream LibriVAD (upstream ``Results`` -> local ``generated``)::

    generated/{variant}/{split_dir}/{noise}/{snr}/{utterance}.wav
    labels/{variant}/{split_dir}/{utterance}.npy
    manifests/{variant}_{split}_{size}.tsv

The sample-level label only depends on the clean (or concatenated) waveform,
so it is stored once per sample and referenced from every noise/SNR row.

Protocol reference: upstream ``create_LibriVAD.py``, ``create_labels.py`` and
``create_LibriSpeechConcat.py`` at the commit pinned in ``config.py``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from alignments import label_counts, label_from_alignment, load_alignment
from audio import Pcm16Reader, read_pcm16
from catalog import (
    ConcatEntry,
    SpeechEntry,
    list_speech_entries,
    select_entries_for_variant,
)
from concat import SilencePool, build_concat_data
from config import (
    ALIGNMENT_ROOT,
    ALIGNMENTS_ZIP_SHA256,
    GENERATED_ROOT,
    LABEL_ROOT,
    LIBRISPEECH_ROOT,
    LIBRIVAD_COMMIT,
    LIBRIVAD_DATASET_BASE_URL,
    LIBRIVAD_REPOSITORY,
    MANIFEST_ROOT,
    NOISES_ROOT,
    NOISES_ZIP_SHA256,
    NOISE_NAMES,
    SAMPLE_RATE,
    SIZE_STEPS,
    SNRS_DB,
    SPLIT_SOURCE_DIRS,
    VARIANTS,
)
from manifests import TsvWriter, write_json
from mix import NoiseCursor, mix_speech_and_noise, stable_mix_hash
from noises import noise_relative_path, probe_noise

# Upstream draws no random numbers: selection is a deterministic stride over a
# sorted file list, so the recorded seed is always this constant.
SEED = 0

MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "split_dir",
    "variant",
    "size",
    "sequence_index",
    "batch_index",
    "source_relative_path",
    "alignment_relative_path",
    "speech_utterance_ids",
    "clean_duration_s",
    "speech_duration_s",
    "silence_duration_s",
    "middle_silence_frames",
    "noise_name",
    "noise_source",
    "noise_start_sample",
    "snr_db",
    "measured_snr_db",
    "noise_scale",
    "label_relative_path",
    "output_audio_path",
    "output_frames",
    "seed",
    "mix_hash",
    "protocol_commit",
)


@dataclass(frozen=True)
class PreparedSample:
    """One clean sample (LibriSpeech utterance or Concat pair) plus metadata."""

    entry: SpeechEntry | ConcatEntry
    audio: np.ndarray
    label: np.ndarray
    sequence_index: int
    source_relative_paths: tuple[str, ...]
    alignment_relative_paths: tuple[str, ...]
    utterance_ids: tuple[str, ...]
    silence_start_sample: int | None = None
    middle_silence_frames: int | None = None

    @property
    def frames(self) -> int:
        return int(self.audio.size)

    @property
    def speech_frames(self) -> int:
        return label_counts(self.label)[0]

    @property
    def silence_frames(self) -> int:
        return label_counts(self.label)[1]

    @property
    def output_relative_path(self) -> Path:
        if isinstance(self.entry, ConcatEntry):
            return self.entry.output_relative_path.with_suffix(".wav")
        return self.entry.relative_path.with_suffix(".wav")

    @property
    def label_relative_path(self) -> Path:
        if isinstance(self.entry, ConcatEntry):
            return self.entry.output_relative_path.with_suffix(".npy")
        return self.entry.relative_path.with_suffix(".npy")


def prepare_sample(
    entry: SpeechEntry | ConcatEntry,
    sequence_index: int,
    *,
    alignment_root: Path,
    silence_pool: SilencePool | None,
) -> PreparedSample:
    if isinstance(entry, ConcatEntry):
        if silence_pool is None:
            raise ValueError("Concat samples need a silence pool")
        silence_start = silence_pool.consumed
        data = build_concat_data(entry, silence_pool)
        return PreparedSample(
            entry=entry,
            audio=data.audio,
            label=data.label,
            sequence_index=sequence_index,
            source_relative_paths=tuple(
                source.relative_path.as_posix()
                for source in entry.sources
            ),
            alignment_relative_paths=tuple(
                source.alignment_path.relative_to(alignment_root).as_posix()
                for source in entry.sources
            ),
            utterance_ids=entry.utterance_ids,
            silence_start_sample=silence_start,
            middle_silence_frames=data.middle_silence_frames,
        )

    audio = read_pcm16(entry.audio_path)
    label = label_from_alignment(
        load_alignment(entry.alignment_path),
        audio.size,
        SAMPLE_RATE,
    )
    return PreparedSample(
        entry=entry,
        audio=audio,
        label=label,
        sequence_index=sequence_index,
        source_relative_paths=(entry.relative_path.as_posix(),),
        alignment_relative_paths=(
            entry.alignment_path.relative_to(alignment_root).as_posix(),
        ),
        utterance_ids=(entry.utterance_id,),
    )


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.asarray(audio, dtype=np.int16),
        SAMPLE_RATE,
        subtype="PCM_16",
    )


def write_label(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(label, dtype=np.int16))


def build_silence_pool(
    *,
    librispeech_root: Path,
    alignment_root: Path,
) -> SilencePool:
    """Pool of aligned train silence, exactly as upstream's Concat builder."""
    train_entries = list_speech_entries(
        "train",
        librispeech_root=librispeech_root,
        alignment_root=alignment_root,
    )
    return SilencePool(train_entries)


def generate_variant_split(
    *,
    variant: str,
    split: str,
    size: str,
    limit: int | None,
    batch_size: int,
    noises: list[str],
    snrs: list[int],
    librispeech_root: Path,
    alignment_root: Path,
    noises_root: Path,
    output_root: Path,
    label_root: Path,
    manifest_root: Path,
    progress_every: int,
    noise_cache: dict[tuple[str, str], object],
) -> dict:
    split_dir = SPLIT_SOURCE_DIRS[split]
    entries = select_entries_for_variant(
        variant,
        split,
        size,
        librispeech_root=librispeech_root,
        alignment_root=alignment_root,
        limit=limit,
    )
    total_rows = len(entries) * len(noises) * len(snrs)
    print(
        f"[{variant}/{split}] {len(entries)} clean samples x "
        f"{len(noises)} noises x {len(snrs)} SNRs = {total_rows} rows"
    )

    silence_pool = None
    if variant == "LibriSpeechConcat":
        silence_pool = build_silence_pool(
            librispeech_root=librispeech_root,
            alignment_root=alignment_root,
        )

    manifest_path = manifest_root / f"{variant}_{split}_{size}.tsv"
    started = time.time()
    rows_written = 0
    with TsvWriter(manifest_path, MANIFEST_FIELDS) as writer:
        for batch_index, batch_start in enumerate(
            range(0, len(entries), batch_size)
        ):
            batch = entries[batch_start : batch_start + batch_size]
            prepared: list[PreparedSample] = []
            for offset, entry in enumerate(batch):
                sample = prepare_sample(
                    entry,
                    batch_start + offset,
                    alignment_root=alignment_root,
                    silence_pool=silence_pool,
                )
                write_label(
                    label_root
                    / variant
                    / split_dir
                    / sample.label_relative_path,
                    sample.label,
                )
                prepared.append(sample)

            for noise_name in noises:
                key = (noise_name, split)
                if key not in noise_cache:
                    noise_cache[key] = probe_noise(
                        noise_name, split, root=noises_root
                    )
                noise_file = noise_cache[key]
                with Pcm16Reader(noise_file.path) as reader:
                    for snr in snrs:
                        # Upstream resets the noise cursor for every
                        # (batch, noise, SNR) group.
                        cursor = NoiseCursor(reader)
                        for sample in prepared:
                            (
                                noise_start,
                                segment,
                                noise_skipped_segments,
                            ) = cursor.next_non_silent(sample.frames)
                            result = mix_speech_and_noise(
                                sample.audio,
                                sample.label,
                                segment,
                                snr,
                            )
                            audio_relative = (
                                Path(variant)
                                / split_dir
                                / noise_name
                                / str(snr)
                                / sample.output_relative_path
                            )
                            write_wav(
                                output_root / audio_relative, result.output
                            )
                            mix_hash = stable_mix_hash(
                                variant=variant,
                                split=split,
                                utterance_ids=sample.utterance_ids,
                                noise_name=noise_name,
                                target_snr_db=snr,
                                noise_start_sample=noise_start,
                                audio=result.output,
                                label=sample.label,
                            )
                            writer.write(
                                {
                                    "sample_id": (
                                        f"{variant}:{split}:{noise_name}:"
                                        f"{snr}:"
                                        f"{sample.output_relative_path.stem}"
                                    ),
                                    "split": split,
                                    "split_dir": split_dir,
                                    "variant": variant,
                                    "size": size,
                                    "sequence_index": sample.sequence_index,
                                    "batch_index": batch_index,
                                    "source_relative_path": "|".join(
                                        sample.source_relative_paths
                                    ),
                                    "alignment_relative_path": "|".join(
                                        sample.alignment_relative_paths
                                    ),
                                    "speech_utterance_ids": "|".join(
                                        sample.utterance_ids
                                    ),
                                    "clean_duration_s": (
                                        f"{sample.frames / SAMPLE_RATE:.6f}"
                                    ),
                                    "speech_duration_s": (
                                        f"{sample.speech_frames / SAMPLE_RATE:.6f}"
                                    ),
                                    "silence_duration_s": (
                                        f"{sample.silence_frames / SAMPLE_RATE:.6f}"
                                    ),
                                    "middle_silence_frames": (
                                        "" if sample.middle_silence_frames is None
                                        else sample.middle_silence_frames
                                    ),
                                    "noise_name": noise_name,
                                    "noise_source": noise_relative_path(
                                        noise_name, split
                                    ).as_posix(),
                                    "noise_start_sample": noise_start,
                                    "snr_db": snr,
                                    "measured_snr_db": (
                                        f"{result.actual_snr_db:.6f}"
                                    ),
                                    "noise_scale": f"{result.scale:.12g}",
                                    "label_relative_path": (
                                        f"{variant}/{split_dir}/"
                                        f"{sample.label_relative_path.as_posix()}"
                                    ),
                                    "output_audio_path": (
                                        audio_relative.as_posix()
                                    ),
                                    "output_frames": result.output.size,
                                    "seed": SEED,
                                    "mix_hash": mix_hash,
                                    "protocol_commit": LIBRIVAD_COMMIT,
                                }
                            )
                            rows_written += 1
                            if (
                                progress_every
                                and rows_written % progress_every == 0
                            ):
                                elapsed = max(
                                    time.time() - started, 1e-3
                                )
                                rate = rows_written / elapsed
                                remaining = max(
                                    total_rows - rows_written, 0
                                )
                                eta = remaining / rate if rate else 0.0
                                print(
                                    f"  {rows_written}/{total_rows} rows, "
                                    f"{rate:.2f} rows/s, ETA {eta / 60:.1f} min",
                                    flush=True,
                                )

    elapsed = max(time.time() - started, 1e-3)
    run_info = {
        "variant": variant,
        "split": split,
        "split_dir": split_dir,
        "size": size,
        "size_step": SIZE_STEPS[size],
        "limit": limit,
        "selected_samples": len(entries),
        "manifest_rows": rows_written,
        "batch_size": batch_size,
        "noises": list(noises),
        "snrs_db": list(snrs),
        "seed": SEED,
        "elapsed_s": round(elapsed, 3),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "paths": {
            "librispeech_root": str(librispeech_root),
            "alignment_root": str(alignment_root),
            "noises_root": str(noises_root),
            "output_root": str(output_root),
            "label_root": str(label_root),
            "manifest": str(manifest_path),
        },
        "protocol": {
            "repository": LIBRIVAD_REPOSITORY,
            "commit": LIBRIVAD_COMMIT,
            "dataset_base_url": LIBRIVAD_DATASET_BASE_URL,
            "sample_rate": SAMPLE_RATE,
            "label_dtype": "int16",
            "alignments_zip_sha256": ALIGNMENTS_ZIP_SHA256,
            "noises_zip_sha256": NOISES_ZIP_SHA256,
            "concat_silence_pool": (
                "train-clean-100 aligned silence, deterministic sorted order"
            ),
            "noise_cursor": (
                "continuous per (batch, noise, SNR), wraps at EOF and "
                "skips digital-silence windows"
            ),
        },
        "noise_files": {
            f"{name}/{split}": {
                "relative_path": noise_relative_path(
                    name, split
                ).as_posix(),
                "sha256": noise_cache[(name, split)].sha256,
                "frames": noise_cache[(name, split)].frames,
                "duration_s": round(
                    noise_cache[(name, split)].duration_s, 6
                ),
            }
            for name in noises
        },
    }
    write_json(manifest_root / f"{variant}_{split}_{size}.json", run_info)
    return run_info


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LibriVAD-compatible speech/noise mixtures."
    )
    parser.add_argument(
        "--size", choices=sorted(SIZE_STEPS), default="small"
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated: LibriSpeech,LibriSpeechConcat",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated: train,val,test",
    )
    parser.add_argument(
        "--noises",
        default=",".join(NOISE_NAMES),
        help="Comma-separated noise names (default: all 9).",
    )
    parser.add_argument(
        "--snrs",
        default=",".join(str(value) for value in SNRS_DB),
        help="Comma-separated SNR values in dB.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N clean samples per (variant, split).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2900,
        help="Clean samples held in memory at once (upstream default 2900).",
    )
    parser.add_argument(
        "--librispeech-root", type=Path, default=LIBRISPEECH_ROOT
    )
    parser.add_argument(
        "--alignment-root", type=Path, default=ALIGNMENT_ROOT
    )
    parser.add_argument("--noises-root", type=Path, default=NOISES_ROOT)
    parser.add_argument("--output-root", type=Path, default=GENERATED_ROOT)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the deterministic selection.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_csv(args.variants)
    splits = parse_csv(args.splits)
    noises = parse_csv(args.noises)
    try:
        snrs = [int(value) for value in parse_csv(args.snrs)]
    except ValueError:
        parser.error("--snrs must be a comma-separated list of integers")

    unknown_variants = [value for value in variants if value not in VARIANTS]
    if unknown_variants:
        parser.error(f"unknown variants: {unknown_variants}")
    unknown_noises = [value for value in noises if value not in NOISE_NAMES]
    if unknown_noises:
        parser.error(f"unknown noises: {unknown_noises}")
    unknown_splits = [
        value for value in splits if value not in SPLIT_SOURCE_DIRS
    ]
    if unknown_splits:
        parser.error(f"unknown splits: {unknown_splits}")
    if not variants or not splits or not noises or not snrs:
        parser.error("variants, splits, noises and snrs must not be empty")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    if args.dry_run:
        for variant in variants:
            for split in splits:
                entries = select_entries_for_variant(
                    variant,
                    split,
                    args.size,
                    librispeech_root=args.librispeech_root,
                    alignment_root=args.alignment_root,
                    limit=args.limit,
                )
                first = "-"
                if entries:
                    first = (
                        entries[0].output_relative_path
                        if isinstance(entries[0], ConcatEntry)
                        else entries[0].relative_path
                    ).as_posix()
                print(
                    f"[dry-run] {variant}/{split}: {len(entries)} clean "
                    f"samples, first={first}"
                )
        return 0

    noise_cache: dict[tuple[str, str], object] = {}
    for variant in variants:
        for split in splits:
            generate_variant_split(
                variant=variant,
                split=split,
                size=args.size,
                limit=args.limit,
                batch_size=args.batch_size,
                noises=noises,
                snrs=snrs,
                librispeech_root=args.librispeech_root,
                alignment_root=args.alignment_root,
                noises_root=args.noises_root,
                output_root=args.output_root,
                label_root=args.label_root,
                manifest_root=args.manifest_root,
                progress_every=args.progress_every,
                noise_cache=noise_cache,
            )
    print("LibriVAD-style generation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
