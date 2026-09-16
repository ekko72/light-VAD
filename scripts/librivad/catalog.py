# -*- coding: utf-8 -*-
"""Catalog LibriSpeech utterances and deterministic Concat pairings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import (
    ALIGNMENT_ROOT,
    LIBRISPEECH_ROOT,
    SIZE_STEPS,
    SPLIT_SOURCE_DIRS,
)


@dataclass(frozen=True)
class SpeechEntry:
    """One aligned LibriSpeech utterance."""

    split: str
    relative_path: Path
    audio_path: Path
    alignment_path: Path

    @property
    def utterance_id(self) -> str:
        return self.relative_path.stem


@dataclass(frozen=True)
class ConcatEntry:
    """One adjacent two-utterance Concat example."""

    split: str
    sources: tuple[SpeechEntry, SpeechEntry]
    output_relative_path: Path
    filename: str

    @property
    def utterance_ids(self) -> tuple[str, str]:
        return tuple(source.utterance_id for source in self.sources)


def _read_unaligned_stems(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stems.add(line.split()[0])
    return stems


def list_speech_entries(
    split: str,
    *,
    librispeech_root: Path = LIBRISPEECH_ROOT,
    alignment_root: Path = ALIGNMENT_ROOT,
) -> list[SpeechEntry]:
    """Return sorted, aligned FLAC entries for one LibriSpeech split."""
    if split not in SPLIT_SOURCE_DIRS:
        raise ValueError(f"Unknown split: {split}")
    split_dir = librispeech_root / SPLIT_SOURCE_DIRS[split]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")

    unaligned = _read_unaligned_stems(alignment_root / "unaligned.txt")
    alignment_split_dir = alignment_root / SPLIT_SOURCE_DIRS[split]
    entries: list[SpeechEntry] = []
    for audio_path in sorted(
        split_dir.rglob("*.flac"), key=lambda path: path.as_posix()
    ):
        if audio_path.stem in unaligned:
            continue
        relative_path = audio_path.relative_to(split_dir)
        alignment_path = (
            alignment_split_dir / relative_path.with_suffix(".TextGrid")
        )
        if not alignment_path.is_file():
            raise FileNotFoundError(
                f"Missing forced alignment for {audio_path}: {alignment_path}"
            )
        entries.append(
            SpeechEntry(
                split=split,
                relative_path=relative_path,
                audio_path=audio_path,
                alignment_path=alignment_path,
            )
        )
    return entries


def apply_size_step(entries: list, size: str) -> list:
    """Apply upstream's sorted-list ``[::step]`` size selection."""
    if size not in SIZE_STEPS:
        raise ValueError(f"Unknown size: {size}")
    return entries[:: SIZE_STEPS[size]]


def _concat_filename(first: SpeechEntry, second: SpeechEntry) -> str:
    first_parts = first.relative_path.parts
    second_parts = second.relative_path.parts
    first_stem = first.relative_path.stem
    second_stem = second.relative_path.stem
    if first_parts[0] != second_parts[0]:
        return f"{first_stem}_+_{second.relative_path.name}"
    if first_parts[1] != second_parts[1]:
        suffix = "-".join(second_stem.split("-")[1:])
        return f"{first_stem}_+_{suffix}.flac"
    return f"{first_stem}_+_{second_stem.split('-')[-1]}.flac"


def make_concat_entries(entries: list[SpeechEntry]) -> list[ConcatEntry]:
    """Pair adjacent sorted utterances exactly as the upstream builder."""
    work = list(entries)
    if len(work) % 2:
        work.pop()
    result: list[ConcatEntry] = []
    for index in range(0, len(work), 2):
        first, second = work[index], work[index + 1]
        filename = _concat_filename(first, second)
        output_relative_path = (
            Path(second.relative_path.parts[0])
            / second.relative_path.parts[1]
            / filename
        )
        result.append(
            ConcatEntry(
                split=first.split,
                sources=(first, second),
                output_relative_path=output_relative_path,
                filename=filename,
            )
        )
    return result


def select_entries_for_variant(
    variant: str,
    split: str,
    size: str,
    *,
    librispeech_root: Path = LIBRISPEECH_ROOT,
    alignment_root: Path = ALIGNMENT_ROOT,
    limit: int | None = None,
) -> list[SpeechEntry | ConcatEntry]:
    """Return the deterministic source selection used by the generator."""
    speech_entries = list_speech_entries(
        split,
        librispeech_root=librispeech_root,
        alignment_root=alignment_root,
    )
    if variant == "LibriSpeech":
        selected = apply_size_step(speech_entries, size)
    elif variant == "LibriSpeechConcat":
        concat_entries = make_concat_entries(speech_entries)
        concat_entries.sort(
            key=lambda entry: entry.output_relative_path.as_posix()
        )
        selected = apply_size_step(concat_entries, size)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected
