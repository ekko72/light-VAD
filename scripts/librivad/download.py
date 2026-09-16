# -*- coding: utf-8 -*-
"""Resumable downloads for the official LibriVAD input archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import (
    ALIGNMENTS_ZIP,
    ALIGNMENTS_ZIP_SHA256,
    ALIGNMENT_ROOT,
    LIBRIVAD_DATASET_BASE_URL,
    LIBRIVAD_ROOT,
    NOISE_NAMES,
    NOISES_ROOT,
    NOISES_ZIP,
    NOISES_ZIP_SHA256,
    NOISES_ZIP_SIZE,
)

USER_AGENT = "light-VAD-librivad-downloader/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def remote_size(url: str, timeout: int = 60) -> int:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers.get("Content-Length") or 0)


def _load_marker(marker: Path) -> set[int]:
    if not marker.is_file():
        return set()
    try:
        values = json.loads(marker.read_text(encoding="utf-8"))
        return {int(value) for value in values}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _write_marker(marker: Path, done: set[int]) -> None:
    marker.write_text(
        json.dumps(sorted(done)), encoding="utf-8"
    )


def _fetch_part(
    url: str,
    start: int,
    end: int,
    destination: Path,
) -> None:
    expected = end - start + 1
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Range": f"bytes={start}-{end}",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                with open(temporary, "wb") as handle:
                    shutil.copyfileobj(response, handle, 1024 * 1024)
            if temporary.stat().st_size != expected:
                raise IOError(
                    f"part size {temporary.stat().st_size} != {expected}"
                )
            os.replace(temporary, destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"Range {start}-{end} failed after retries: {last_error}"
    )


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    jobs: int = 8,
    chunk_mb: int = 8,
) -> Path:
    """Download with range requests, resume markers, and SHA-256 checking."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is None:
        expected_size = remote_size(url)
    if expected_size <= 0:
        raise RuntimeError(f"Could not determine remote size: {url}")

    if destination.is_file() and destination.stat().st_size == expected_size:
        if expected_sha256 is None or (
            sha256_file(destination).lower()
            == expected_sha256.lower()
        ):
            shutil.rmtree(
                destination.parent / f"{destination.stem}.parts",
                ignore_errors=True,
            )
            destination.with_suffix(
                destination.suffix + ".chunks.json"
            ).unlink(missing_ok=True)
            print(f"[skip] complete: {destination}")
            return destination
        print(f"[warn] existing file has wrong hash, rebuilding: {destination}")
        destination.unlink()

    chunk_size = max(1, int(chunk_mb)) * 1024 * 1024
    starts = list(range(0, expected_size, chunk_size))
    marker = destination.with_suffix(destination.suffix + ".chunks.json")
    parts_dir = destination.parent / f"{destination.stem}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    done = _load_marker(marker)
    for index in list(done):
        start = starts[index]
        end = min(start + chunk_size, expected_size) - 1
        part = parts_dir / f"part-{index:04d}.bin"
        if not part.is_file() or part.stat().st_size != end - start + 1:
            done.discard(index)
    _write_marker(marker, done)

    pending = [index for index in range(len(starts)) if index not in done]
    print(
        f"{destination.name}: {expected_size / 1e6:.1f} MB, "
        f"{len(starts)} parts, {len(done)} already complete"
    )
    started = time.time()
    completed = len(done)
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {}
        for index in pending:
            start = starts[index]
            end = min(start + chunk_size, expected_size) - 1
            part = parts_dir / f"part-{index:04d}.bin"
            futures[
                executor.submit(_fetch_part, url, start, end, part)
            ] = index
        try:
            for future in as_completed(futures):
                future.result()
                done.add(futures[future])
                completed += 1
                _write_marker(marker, done)
                elapsed = max(time.time() - started, 0.001)
                speed = (
                    (completed - len(pending) + len(pending))
                    * chunk_size
                    / elapsed
                    / 1e6
                )
                if completed % 10 == 0 or completed == len(starts):
                    print(
                        f"  {completed}/{len(starts)} parts, "
                        f"{speed:.2f} MB/s",
                        flush=True,
                    )
        except Exception:
            print(
                f"  download paused at {len(done)}/{len(starts)} parts; "
                "rerun to resume",
                flush=True,
            )
            raise

    temporary = destination.with_suffix(destination.suffix + ".merging")
    with open(temporary, "wb") as output:
        for index in range(len(starts)):
            part = parts_dir / f"part-{index:04d}.bin"
            with open(part, "rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if temporary.stat().st_size != expected_size:
        raise IOError(
            f"merged size {temporary.stat().st_size} != {expected_size}"
        )
    os.replace(temporary, destination)

    if expected_sha256 is not None:
        actual = sha256_file(destination)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {destination}: "
                f"{actual} != {expected_sha256}"
            )
    shutil.rmtree(parts_dir)
    marker.unlink(missing_ok=True)
    print(f"[ok] downloaded and verified: {destination}")
    return destination


def extract_zip(
    archive: Path,
    destination: Path,
    *,
    force: bool = False,
    sentinel: Path | None = None,
) -> Path:
    archive = Path(archive)
    destination = Path(destination)
    # ``sentinel`` lets a caller that shares one destination directory between
    # several archives decide "already extracted?" from its own subdirectory.
    check = Path(sentinel) if sentinel is not None else destination
    if check.exists() and any(check.iterdir()) and not force:
        print(f"[skip] already extracted: {check}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} -> {destination}")
    with zipfile.ZipFile(archive) as zip_file:
        try:
            zip_file.extractall(destination, filter="data")
        except TypeError:
            # ``filter`` was added to ZipFile.extractall in Python 3.11.
            zip_file.extractall(destination)
    print(f"[ok] extracted: {destination}")
    return destination


def ensure_alignments(jobs: int, chunk_mb: int) -> None:
    if ALIGNMENT_ROOT.is_dir() and any(ALIGNMENT_ROOT.iterdir()):
        print(f"[skip] alignments already available: {ALIGNMENT_ROOT}")
        return
    if not ALIGNMENTS_ZIP.is_file() or (
        sha256_file(ALIGNMENTS_ZIP).lower()
        != ALIGNMENTS_ZIP_SHA256.lower()
    ):
        download_file(
            f"{LIBRIVAD_DATASET_BASE_URL}/Forced_alignments.zip",
            ALIGNMENTS_ZIP,
            expected_sha256=ALIGNMENTS_ZIP_SHA256,
            jobs=jobs,
            chunk_mb=chunk_mb,
        )
    extract_zip(
        ALIGNMENTS_ZIP,
        LIBRIVAD_ROOT / "raw" / "Forced_alignments",
    )


def ensure_noises(jobs: int, chunk_mb: int) -> None:
    expected = [
        NOISES_ROOT / noise / f"{noise}_{split}.wav"
        for noise in NOISE_NAMES
        for split in ("train", "val", "test")
    ]
    if all(path.is_file() for path in expected):
        print(f"[skip] noises already available: {NOISES_ROOT}")
        return
    if not NOISES_ZIP.is_file() or (
        NOISES_ZIP.stat().st_size != NOISES_ZIP_SIZE
    ) or (
        sha256_file(NOISES_ZIP).lower()
        != NOISES_ZIP_SHA256.lower()
    ):
        download_file(
            f"{LIBRIVAD_DATASET_BASE_URL}/Noises.zip",
            NOISES_ZIP,
            expected_size=NOISES_ZIP_SIZE,
            expected_sha256=NOISES_ZIP_SHA256,
            jobs=jobs,
            chunk_mb=chunk_mb,
        )
    extract_zip(
        NOISES_ZIP,
        NOISES_ROOT,
        sentinel=NOISES_ROOT,
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} noise files missing after extraction, "
            f"first: {missing[0]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official LibriVAD inputs."
    )
    parser.add_argument(
        "--target",
        default="noises",
        help="Comma-separated: noises,alignments",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args()
    targets = {
        value.strip() for value in args.target.split(",") if value.strip()
    }
    unknown = targets - {"noises", "alignments"}
    if unknown:
        parser.error(f"unknown targets: {sorted(unknown)}")
    if "alignments" in targets:
        ensure_alignments(args.jobs, args.chunk_mb)
    if "noises" in targets:
        ensure_noises(args.jobs, args.chunk_mb)


if __name__ == "__main__":
    main()
