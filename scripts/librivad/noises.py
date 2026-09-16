# -*- coding: utf-8 -*-
"""Official LibriVAD noise paths, validation, and manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from config import NOISES_ROOT, NOISE_NAMES, SAMPLE_RATE
from download import sha256_file


@dataclass(frozen=True)
class NoiseFile:
    name: str
    split: str
    relative_path: Path
    path: Path
    frames: int
    sample_rate: int
    channels: int
    subtype: str
    sha256: str

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def noise_relative_path(name: str, split: str) -> Path:
    if name not in NOISE_NAMES:
        raise ValueError(f"Unknown LibriVAD noise: {name}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown noise split: {split}")
    return Path("raw") / "Noises" / name / f"{name}_{split}.wav"


def noise_path(
    name: str,
    split: str,
    *,
    root: Path = NOISES_ROOT,
) -> Path:
    if name not in NOISE_NAMES:
        raise ValueError(f"Unknown LibriVAD noise: {name}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown noise split: {split}")
    return root / name / f"{name}_{split}.wav"


def probe_noise(
    name: str,
    split: str,
    *,
    root: Path = NOISES_ROOT,
) -> NoiseFile:
    path = noise_path(name, split, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Noise file not found: {path}")
    info = sf.info(str(path))
    if int(info.samplerate) != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample rate {info.samplerate}, "
            f"expected {SAMPLE_RATE}"
        )
    if int(info.channels) != 1:
        raise ValueError(f"{path} is not mono (channels={info.channels})")
    relative = noise_relative_path(name, split)
    return NoiseFile(
        name=name,
        split=split,
        relative_path=relative,
        path=path,
        frames=int(info.frames),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
        sha256=sha256_file(path),
    )


def validate_selected_noises(
    noises: list[str],
    splits: list[str],
) -> list[NoiseFile]:
    result = [
        probe_noise(name, split)
        for name in noises
        for split in splits
    ]
    return result
