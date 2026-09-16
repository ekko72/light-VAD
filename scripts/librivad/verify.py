# -*- coding: utf-8 -*-
"""Verify generated LibriVAD-style manifests, labels and mixtures.

Every check is independent of the generator: labels are rebuilt from the
TextGrid files, mixtures are recomputed from LibriSpeech + noise, and the
recorded hashes/SNRs are compared against freshly measured values.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from alignments import label_from_alignment, load_alignment
from audio import Pcm16Reader, read_pcm16, rms, rms_db, scale_to_pcm16
from catalog import list_speech_entries, select_entries_for_variant
from concat import ConcatData, SilencePool, build_concat_data
from config import (
    ALIGNMENT_ROOT,
    GENERATED_ROOT,
    LABEL_ROOT,
    LIBRISPEECH_ROOT,
    NOISES_ROOT,
    NOISE_NAMES,
    SAMPLE_RATE,
    SPLIT_SOURCE_DIRS,
)
from manifests import read_tsv
from mix import stable_mix_hash

MAX_REPORTED_FAILURES = 20


def compare_int16(
    expected: np.ndarray, actual: np.ndarray
) -> tuple[int, int]:
    """Return (differing samples, max absolute difference)."""
    if expected.size != actual.size:
        return max(int(expected.size), int(actual.size)), -1
    difference = np.abs(
        expected.astype(np.int32) - actual.astype(np.int32)
    )
    return int(np.count_nonzero(difference)), int(difference.max())


@dataclass
class RowContext:
    """Everything needed to re-derive one manifest row."""

    row: dict
    clean_audio: np.ndarray
    label: np.ndarray
    noise_segment: np.ndarray


@dataclass
class VerificationReport:
    manifest: Path
    rows: int = 0
    checked: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def bump(self, name: str) -> None:
        self.checked[name] = self.checked.get(name, 0) + 1

    def fail(self, name: str, message: str) -> None:
        self.failures.append(f"[{name}] {message}")

    @property
    def ok(self) -> bool:
        return not self.failures


class ManifestVerifier:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.librispeech_root = Path(args.librispeech_root)
        self.alignment_root = Path(args.alignment_root)
        self.noises_root = Path(args.noises_root)
        self.output_root = Path(args.output_root)
        self.label_root = Path(args.label_root)
        self._concat_pools: dict[str, SilencePool] = {}
        self._concat_data: dict[tuple[str, int], ConcatData] = {}
        self._concat_selection: dict[str, list] = {}
        self._concat_progress: dict[str, int] = {}
        self._noise_readers: dict[str, Pcm16Reader] = {}

    def close(self) -> None:
        for reader in self._noise_readers.values():
            reader.close()
        self._noise_readers.clear()

    # ----- helpers -----------------------------------------------------

    def noise_reader(self, name: str, split: str) -> Pcm16Reader:
        key = f"{name}/{split}"
        if key not in self._noise_readers:
            path = self.noises_root / name / f"{name}_{split}.wav"
            self._noise_readers[key] = Pcm16Reader(path)
        return self._noise_readers[key]

    def concat_data(
        self, variant: str, split: str, size: str, index: int
    ) -> ConcatData:
        """Rebuild Concat samples in generation order to keep pool state."""
        key = (split, index)
        if key in self._concat_data:
            return self._concat_data[key]

        pool = self._concat_pools.get(split)
        if pool is None:
            train_entries = list_speech_entries(
                "train",
                librispeech_root=self.librispeech_root,
                alignment_root=self.alignment_root,
            )
            pool = SilencePool(train_entries)
            self._concat_pools[split] = pool
            self._concat_progress[split] = 0

        selection = self._concat_selection.get(split)
        if selection is None:
            selection = select_entries_for_variant(
                variant,
                split,
                size,
                librispeech_root=self.librispeech_root,
                alignment_root=self.alignment_root,
            )
            self._concat_selection[split] = selection

        for position in range(self._concat_progress[split], index + 1):
            entry = selection[position]
            self._concat_data[(split, position)] = build_concat_data(
                entry, pool
            )
            self._concat_progress[split] = position + 1
        return self._concat_data[key]

    def clean_audio(
        self, variant: str, row: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (clean audio, sample-level label) for one manifest row."""
        split_dir = row["split_dir"]
        if variant == "LibriSpeech":
            relative = row["source_relative_path"]
            audio_path = self.librispeech_root / split_dir / relative
            audio = read_pcm16(audio_path)
            label = label_from_alignment(
                load_alignment(
                    self.alignment_root / row["alignment_relative_path"]
                ),
                audio.size,
                SAMPLE_RATE,
            )
            return audio, label

        index = int(row["sequence_index"])
        data = self.concat_data(variant, row["split"], row["size"], index)
        return data.audio, data.label

    # ----- checks ------------------------------------------------------

    def check_label_file(
        self, row: dict, report: VerificationReport
    ) -> np.ndarray | None:
        label_path = (
            self.label_root / row["label_relative_path"]
        )
        if not label_path.is_file():
            report.fail("label-file", f"missing {label_path}")
            return None
        try:
            label = np.load(label_path)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            report.fail("label-file", f"{label_path} unreadable: {exc}")
            return None
        report.bump("label-file")
        if label.dtype != np.int16:
            report.fail(
                "label-dtype",
                f"{label_path} dtype={label.dtype}, expected int16",
            )
        values = np.unique(label)
        if not np.all(np.isin(values, (0, 1))):
            report.fail(
                "label-values",
                f"{label_path} has values {values[:8]}",
            )
        return label

    def check_audio_file(
        self, row: dict, report: VerificationReport
    ) -> np.ndarray | None:
        audio_path = self.output_root / row["output_audio_path"]
        if not audio_path.is_file():
            report.fail("audio-file", f"missing {audio_path}")
            return None
        info = sf.info(str(audio_path))
        report.bump("audio-file")
        if int(info.samplerate) != SAMPLE_RATE:
            report.fail(
                "audio-format",
                f"{audio_path} sample rate {info.samplerate}",
            )
        if int(info.channels) != 1:
            report.fail(
                "audio-format", f"{audio_path} channels {info.channels}"
            )
        if str(info.subtype) != "PCM_16":
            report.fail(
                "audio-format", f"{audio_path} subtype {info.subtype}"
            )
        if int(info.frames) != int(row["output_frames"]):
            report.fail(
                "audio-format",
                f"{audio_path} frames {info.frames} != "
                f"manifest {row['output_frames']}",
            )
        try:
            return read_pcm16(audio_path)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            report.fail("audio-format", f"{audio_path} unreadable: {exc}")
            return None

    def check_regenerated_label(
        self,
        variant: str,
        row: dict,
        clean_label: np.ndarray,
        stored_label: np.ndarray,
        report: VerificationReport,
    ) -> None:
        report.bump("label-regenerated")
        differing, _ = compare_int16(clean_label, stored_label)
        if differing:
            report.fail(
                "label-regenerated",
                f"{row['sample_id']}: {differing} label samples differ",
            )

    def check_mix(
        self,
        row: dict,
        clean_audio: np.ndarray,
        label: np.ndarray,
        stored_audio: np.ndarray,
        report: VerificationReport,
    ) -> None:
        frames = int(row["output_frames"])
        if clean_audio.size != frames:
            report.fail(
                "length",
                f"{row['sample_id']}: clean {clean_audio.size} != "
                f"label/audio {frames}",
            )
            return

        name = row["noise_name"]
        split = row["split"]
        start = int(row["noise_start_sample"])
        reader = self.noise_reader(name, split)
        noise_segment = reader.read(start, frames)

        speech_only = clean_audio[label == 1]
        if speech_only.size == 0:
            report.fail("snr", f"{row['sample_id']}: no speech samples")
            return
        speech_rms = rms(speech_only)
        noise_rms = rms(noise_segment)
        if speech_rms <= 0 or noise_rms <= 0:
            report.fail(
                "snr", f"{row['sample_id']}: zero RMS region"
            )
            return

        snr = float(row["snr_db"])
        expected_scale = speech_rms / (
            noise_rms * (10.0 ** (snr / 20.0))
        )
        recorded_scale = float(row["noise_scale"])
        report.bump("noise-scale")
        if not math.isclose(
            recorded_scale, expected_scale, rel_tol=1e-9, abs_tol=0.0
        ):
            report.fail(
                "noise-scale",
                f"{row['sample_id']}: recorded {recorded_scale} != "
                f"recomputed {expected_scale}",
            )

        measured = rms_db(speech_only) - (
            rms_db(noise_segment) + 20.0 * math.log10(recorded_scale)
        )
        report.bump("snr")
        if abs(measured - snr) > 1e-3:
            report.fail(
                "snr",
                f"{row['sample_id']}: measured {measured:.6f} dB != "
                f"target {snr} dB",
            )
        recorded_measured = float(row["measured_snr_db"])
        if abs(recorded_measured - measured) > 1e-3:
            report.fail(
                "snr",
                f"{row['sample_id']}: manifest measured "
                f"{recorded_measured} != recomputed {measured}",
            )

        expected = scale_to_pcm16(
            clean_audio.astype(np.float64)
            + recorded_scale * noise_segment.astype(np.float64)
        )
        report.bump("mix-bytes")
        if not self.args.skip_audio:
            differing, max_difference = compare_int16(
                expected, stored_audio
            )
            if differing:
                report.fail(
                    "mix-bytes",
                    f"{row['sample_id']}: {differing} samples differ, "
                    f"max |delta| {max_difference}",
                )

        report.bump("mix-hash")
        expected_hash = stable_mix_hash(
            variant=row["variant"],
            split=split,
            utterance_ids=row["speech_utterance_ids"].split("|"),
            noise_name=name,
            target_snr_db=int(row["snr_db"]),
            noise_start_sample=start,
            audio=stored_audio,
            label=label,
        )
        if expected_hash != row["mix_hash"]:
            report.fail(
                "mix-hash",
                f"{row['sample_id']}: manifest {row['mix_hash']} != "
                f"recomputed {expected_hash}",
            )

    def check_selection(
        self, rows: list[dict], report: VerificationReport
    ) -> None:
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for row in rows:
            key = (row["variant"], row["split"], row["size"])
            grouped.setdefault(key, []).append(row)
        for (variant, split, size), group in grouped.items():
            limit = max(int(row["sequence_index"]) for row in group) + 1
            entries = select_entries_for_variant(
                variant,
                split,
                size,
                librispeech_root=self.librispeech_root,
                alignment_root=self.alignment_root,
                limit=limit,
            )
            for row in group:
                index = int(row["sequence_index"])
                entry = entries[index]
                if hasattr(entry, "sources"):
                    expected_ids = "|".join(entry.utterance_ids)
                    expected_paths = "|".join(
                        source.relative_path.as_posix()
                        for source in entry.sources
                    )
                else:
                    expected_ids = entry.utterance_id
                    expected_paths = entry.relative_path.as_posix()
                report.bump("selection")
                if row["speech_utterance_ids"] != expected_ids:
                    report.fail(
                        "selection",
                        f"{row['sample_id']}: ids {expected_ids} != "
                        f"manifest {row['speech_utterance_ids']}",
                    )
                if row["source_relative_path"] != expected_paths:
                    report.fail(
                        "selection",
                        f"{row['sample_id']}: paths {expected_paths} != "
                        f"manifest {row['source_relative_path']}",
                    )

    # ----- driver ------------------------------------------------------

    def verify_manifest(self, manifest: Path) -> VerificationReport:
        report = VerificationReport(manifest=manifest)
        rows = read_tsv(manifest)
        if self.args.limit is not None:
            rows = rows[: self.args.limit]
        report.rows = len(rows)
        print(f"{manifest.name}: verifying {len(rows)} rows")

        for row in rows:
            variant = row["variant"]
            if variant not in ("LibriSpeech", "LibriSpeechConcat"):
                report.fail("variant", f"unknown variant {variant}")
                continue
            if row["split_dir"] != SPLIT_SOURCE_DIRS.get(row["split"]):
                report.fail(
                    "split",
                    f"{row['sample_id']}: split_dir "
                    f"{row['split_dir']} != "
                    f"{SPLIT_SOURCE_DIRS.get(row['split'])}",
                )
            if row["noise_name"] not in NOISE_NAMES:
                report.fail(
                    "noise",
                    f"{row['sample_id']}: unknown noise "
                    f"{row['noise_name']}",
                )

            stored_label = self.check_label_file(row, report)
            stored_audio = self.check_audio_file(row, report)
            if stored_label is None or stored_audio is None:
                continue
            if stored_label.size != stored_audio.size:
                report.fail(
                    "length",
                    f"{row['sample_id']}: label {stored_label.size} != "
                    f"audio {stored_audio.size}",
                )
                continue

            clean_audio, clean_label = self.clean_audio(variant, row)
            if not self.args.skip_labels:
                self.check_regenerated_label(
                    variant, row, clean_label, stored_label, report
                )
            self.check_mix(
                row, clean_audio, stored_label, stored_audio, report
            )

        if self.args.check_selection:
            self.check_selection(rows, report)
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify LibriVAD-style generated data."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="One or more manifest TSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Verify only the first N rows of each manifest.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip the byte-exact remix comparison.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip re-deriving labels from the TextGrid files.",
    )
    parser.add_argument(
        "--check-selection",
        action="store_true",
        help="Re-derive the deterministic size/stride selection.",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verifier = ManifestVerifier(args)
    reports: list[VerificationReport] = []
    try:
        for manifest in args.manifest:
            if not manifest.is_file():
                raise FileNotFoundError(f"Manifest not found: {manifest}")
            reports.append(verifier.verify_manifest(manifest))
    finally:
        verifier.close()

    total_failures = 0
    for report in reports:
        checked = ", ".join(
            f"{name}={count}"
            for name, count in sorted(report.checked.items())
        )
        print(
            f"\n{report.manifest.name}: {report.rows} rows checked"
            + (f" ({checked})" if checked else "")
        )
        for message in report.failures[:MAX_REPORTED_FAILURES]:
            print(f"  FAIL {message}")
        if len(report.failures) > MAX_REPORTED_FAILURES:
            print(
                f"  ... {len(report.failures) - MAX_REPORTED_FAILURES} "
                "more failures"
            )
        print("  result: " + ("PASS" if report.ok else "FAIL"))
        total_failures += len(report.failures)

    if total_failures:
        print(f"\n{total_failures} checks failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
