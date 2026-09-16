# -*- coding: utf-8 -*-
"""Shared constants and paths for the LibriVAD-compatible pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
LIBRIVAD_ROOT = DATA_ROOT / "librivad"

LIBRISPEECH_ROOT = DATA_ROOT / "LibriSpeech"
ALIGNMENT_ROOT = (
    LIBRIVAD_ROOT / "raw" / "Forced_alignments" / "librispeech_alignments"
)
NOISES_ROOT = LIBRIVAD_ROOT / "raw" / "Noises"
NOISES_ZIP = LIBRIVAD_ROOT / "downloads" / "Noises.zip"
ALIGNMENTS_ZIP = LIBRIVAD_ROOT / "downloads" / "Forced_alignments.zip"

GENERATED_ROOT = LIBRIVAD_ROOT / "generated"
LABEL_ROOT = LIBRIVAD_ROOT / "labels"
MANIFEST_ROOT = LIBRIVAD_ROOT / "manifests"

SAMPLE_RATE = 16_000
SPLIT_SOURCE_DIRS = {
    "train": "train-clean-100",
    "val": "dev-clean",
    "test": "test-clean",
}
SPLIT_ORDER = ("train", "val", "test")

NOISE_NAMES = (
    "Babble_noise",
    "SSN_noise",
    "Domestic_noise",
    "Nature_noise",
    "Office_noise",
    "Public_noise",
    "Street_noise",
    "Transport_noise",
    "City_noise",
)
SNRS_DB = (-5, 0, 5, 10, 15, 20)
SIZE_STEPS = {
    "small": 100,
    "medium": 10,
    "large": 1,
}
VARIANTS = ("LibriSpeech", "LibriSpeechConcat")

# Upstream protocol provenance checked on 2026-09-15.
LIBRIVAD_REPOSITORY = "https://github.com/IoannisStylianou/LibriVAD"
LIBRIVAD_COMMIT = "b39c051c08cf3f7b3d748a093484907b0e6e32ec"
LIBRIVAD_DATASET_BASE_URL = (
    "https://hf-mirror.com/datasets/LibriVAD/LibriVAD/resolve/main/Files"
)
ALIGNMENTS_ZIP_SHA256 = (
    "742D5F68B46CF250AC9647398583AC6555B982BDCEDFA30E7D429A0C46982CFC"
)
NOISES_ZIP_SHA256 = (
    "4D6C5279BA628D8FC1895439033BE7FD06861CB1122B41E1696EA492D4E31BB3"
)
NOISES_ZIP_SIZE = 3_315_377_901


def speech_split_dir(split: str) -> Path:
    try:
        return LIBRISPEECH_ROOT / SPLIT_SOURCE_DIRS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc


def alignment_split_dir(split: str, root: Path = ALIGNMENT_ROOT) -> Path:
    try:
        return root / SPLIT_SOURCE_DIRS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split: {split}") from exc
