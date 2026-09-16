# -*- coding: utf-8 -*-
"""Small TSV/JSON helpers for the LibriVAD-compatible manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType


class TsvWriter:
    """Streaming TSV writer so large manifests never sit in memory."""

    def __init__(self, path: str | Path, fields: Sequence[str]):
        self.path = Path(path)
        self.fields = tuple(fields)
        self._handle = None
        self.rows = 0

    def __enter__(self) -> "TsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(
            self.path, "w", encoding="utf-8", newline=""
        )
        self._handle.write("\t".join(self.fields) + "\n")
        return self

    def write(self, row: Mapping[str, object]) -> None:
        if self._handle is None:
            raise RuntimeError("TsvWriter must be used as a context manager")
        self._handle.write(
            "\t".join(str(row.get(field, "")) for field in self.fields)
            + "\n"
        )
        self.rows += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_tsv(
    path: str | Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write(
                "\t".join(str(row.get(field, "")) for field in fields) + "\n"
            )
    return path


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows: list[dict[str, str]] = []
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split("\t")
            rows.append(dict(zip(header, values)))
    return rows


def write_json(path: str | Path, value: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
