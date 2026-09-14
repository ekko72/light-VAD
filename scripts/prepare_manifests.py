# -*- coding: utf-8 -*-
"""生成 speech/noise manifest 与固定 train/val/test 划分。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import DATA_ROOT, TARGET_SR

SPEECH_ROOTS = {
    "train": DATA_ROOT / "LibriSpeech" / "train-clean-100",
    "val": DATA_ROOT / "LibriSpeech" / "dev-clean",
    "test": DATA_ROOT / "LibriSpeech" / "test-clean",
}

TRAIN_NOISE_GROUPS = [
    ("music", "fma"),
    ("music", "jamendo"),
    ("music", "rfm"),
    ("noise", "free-sound"),
    ("speech", "us-gov"),
]
TEST_SEEN_NOISE_GROUPS = [("music", "fma-western-art")]
TEST_UNSEEN_NOISE_GROUPS = [
    ("music", "hd-classical"),
    ("noise", "sound-bible"),
    ("speech", "librivox"),
]


def probe(path: Path):
    try:
        info = sf.info(str(path))
        return path, int(info.frames), int(info.samplerate), None
    except Exception as exc:
        return path, 0, 0, exc


def scan_files(files: list[Path], jobs: int = 16):
    result = {}
    errors = []
    total = len(files)
    worker = max(1, jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as pool:
        for idx, (p, frames, sr, err) in enumerate(
            pool.map(probe, files), 1
        ):
            if err is None:
                result[p] = (frames, sr)
            else:
                errors.append((str(p), str(err)))
            if idx % 500 == 0:
                print(f"  {idx}/{total}", flush=True)
    return result, errors


def write_tsv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in header) + "\n")


def build_speech_manifest(jobs: int):
    files = []
    for split, root in SPEECH_ROOTS.items():
        if not root.exists():
            print(f"[错误] 缺少目录: {root}")
            sys.exit(1)
        files.extend(sorted(root.rglob("*.flac"), key=lambda p: p.as_posix()))

    print(f"扫描 LibriSpeech: {len(files)} 个文件 ...", flush=True)
    infos, errors = scan_files(files, jobs)
    for path, err in errors[:20]:
        print(f"  [警告] 无法读取: {path} -> {err}")

    rows = []
    for split, root in SPEECH_ROOTS.items():
        for p in sorted(root.rglob("*.flac"), key=lambda x: x.as_posix()):
            if p not in infos:
                continue
            frames, sr = infos[p]
            rows.append(
                {
                    "path": p.relative_to(DATA_ROOT).as_posix(),
                    "speaker_id": p.relative_to(root).parts[0],
                    "duration_s": f"{frames / sr:.4f}",
                    "sample_rate": str(sr),
                    "split": split,
                }
            )
    manifest = DATA_ROOT / "manifests" / "speech_manifest.tsv"
    write_tsv(
        manifest,
        ["path", "speaker_id", "duration_s", "sample_rate", "split"],
        rows,
    )
    print(f"speech manifest: {manifest} ({len(rows)} 条)")
    return rows


def build_noise_manifest(jobs: int):
    root = DATA_ROOT / "musan"
    if not root.exists():
        print(f"[错误] 缺少目录: {root}")
        sys.exit(1)
    files = sorted(root.rglob("*.wav"), key=lambda p: p.as_posix())
    print(f"扫描 MUSAN: {len(files)} 个文件 ...", flush=True)
    infos, errors = scan_files(files, jobs)
    for path, err in errors[:20]:
        print(f"  [警告] 无法读取: {path} -> {err}")

    rows = []
    for p in files:
        if p not in infos:
            continue
        frames, sr = infos[p]
        rel = p.relative_to(root)
        category = rel.parts[0]
        subcategory = rel.parts[1] if len(rel.parts) > 1 else "root"
        rows.append(
            {
                "path": p.relative_to(DATA_ROOT).as_posix(),
                "category": category,
                "subcategory": subcategory,
                "duration_s": f"{frames / sr:.4f}",
                "sample_rate": str(sr),
            }
        )
    manifest = DATA_ROOT / "manifests" / "musan_manifest.tsv"
    write_tsv(
        manifest,
        ["path", "category", "subcategory", "duration_s", "sample_rate"],
        rows,
    )
    print(f"MUSAN manifest: {manifest} ({len(rows)} 条)")
    return rows


def make_speech_splits(rows: list[dict[str, str]]):
    by_split = {split: [] for split in SPEECH_ROOTS}
    for row in rows:
        by_split[row["split"]].append(row)

    speakers: dict[str, set[str]] = {}
    for split, split_rows in by_split.items():
        speakers[split] = {r["speaker_id"] for r in split_rows}
    overlap = {}
    split_names = list(SPEECH_ROOTS)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            common = sorted(speakers[a] & speakers[b])
            if common:
                overlap[f"{a}∩{b}"] = common
    if overlap:
        print(f"  [警告] speaker 有交集: {overlap}")

    base = DATA_ROOT / "splits"
    for split in split_names:
        header = ["path", "speaker_id", "duration_s", "sample_rate"]
        write_tsv(base / f"{split}_speech.tsv", header, by_split[split])
    print("speech 划分: train / val / test")
    return by_split, overlap


def make_noise_splits(rows: list[dict[str, str]], seed: int):
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["subcategory"])].append(row)

    role: dict[tuple[str, str], str] = {}
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_seen_rows: list[dict[str, str]] = []
    test_unseen_rows: list[dict[str, str]] = []

    for group in TRAIN_NOISE_GROUPS:
        group_rows = sorted(groups.get(group, []), key=lambda r: r["path"])
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_val = int(round(n * 0.15))
        n_test_seen = int(round(n * 0.15))
        n_train = n - n_val - n_test_seen
        train_rows.extend(group_rows[:n_train])
        val_rows.extend(group_rows[n_train : n_train + n_val])
        test_seen_rows.extend(group_rows[n_train + n_val : n_train + n_val + n_test_seen])
        role[group] = "train_group(文件级 70/15/15)"

    for group in TEST_SEEN_NOISE_GROUPS:
        test_seen_rows.extend(groups.get(group, []))
        role[group] = "test_seen_group"

    for group in TEST_UNSEEN_NOISE_GROUPS:
        test_unseen_rows.extend(groups.get(group, []))
        role[group] = "test_unseen_group"

    unknown = sorted(set(groups) - set(role))
    for group in unknown:
        train_rows.extend(groups[group])
        role[group] = "unknown->train(fallback)"
        print(f"  [警告] 未在协议中登记的噪声组: {group[0]}/{group[1]}")

    def sort_rows(rows_, split: str):
        return [
            {**row, "split": split} for row in sorted(rows_, key=lambda r: r["path"])
        ]

    base = DATA_ROOT / "splits"
    header = ["path", "category", "subcategory", "duration_s", "sample_rate", "split"]
    write_tsv(base / "train_noise.tsv", header, sort_rows(train_rows, "train"))
    write_tsv(base / "val_noise.tsv", header, sort_rows(val_rows, "val"))
    write_tsv(base / "test_seen_noise.tsv", header, sort_rows(test_seen_rows, "test_seen"))
    write_tsv(base / "test_unseen_noise.tsv", header, sort_rows(test_unseen_rows, "test_unseen"))
    return role, train_rows, val_rows, test_seen_rows, test_unseen_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    speech_rows = build_speech_manifest(args.jobs)
    noise_rows = build_noise_manifest(args.jobs)
    speech_splits, overlap = make_speech_splits(speech_rows)
    noise_roles, train_n, val_n, seen_n, unseen_n = make_noise_splits(noise_rows, args.seed)

    summary = {
        "seed": args.seed,
        "sample_rate": TARGET_SR,
        "speaker_overlap": overlap,
        "noise_group_roles": {f"{k[0]}/{k[1]}": v for k, v in sorted(noise_roles.items())},
        "counts": {
            "speech_train": len(speech_splits["train"]),
            "speech_val": len(speech_splits["val"]),
            "speech_test": len(speech_splits["test"]),
            "noise_train": len(train_n),
            "noise_val": len(val_n),
            "noise_test_seen": len(seen_n),
            "noise_test_unseen": len(unseen_n),
        },
    }
    summary_path = DATA_ROOT / "splits" / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("划分摘要:")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
