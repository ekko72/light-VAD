# -*- coding: utf-8 -*-
"""下载 G 阶段所需公开数据，支持并发分块与断点续传。

默认下载（已解压的会自动跳过）：
  - LibriSpeech dev-clean / test-clean / train-clean-100
  - MUSAN（约 10.5GB，排序靠后）

用法：
  python scripts/download_data.py [--jobs 8] [--only train-clean-100,musan]
"""

import argparse
import json
import shutil
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"
OPENSLR = "https://www.openslr.org/resources"
HF_MIRROR = "https://hf-mirror.com/datasets/huseinzol05/musan-mirror/resolve/main/musan.tar.gz"
CHUNK_MB = 8
DEFAULT_JOBS = 8

ITEMS = [
    ("dev-clean", f"{OPENSLR}/12/dev-clean.tar.gz", "LibriSpeech/dev-clean"),
    ("test-clean", f"{OPENSLR}/12/test-clean.tar.gz", "LibriSpeech/test-clean"),
    ("train-clean-100", f"{OPENSLR}/12/train-clean-100.tar.gz", "LibriSpeech/train-clean-100"),
    ("musan", HF_MIRROR, "musan"),
]


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def parallel_download(url: str, dest: Path, jobs: int) -> bool:
    """把文件切成若干区间，多连接并发下载后拼接；返回下载是否完整可用。"""
    total = remote_size(url)
    if total <= 0:
        raise RuntimeError(f"无法获取远程文件大小: {url}")

    chunk = CHUNK_MB * 1024 * 1024
    starts = list(range(0, total, chunk))
    marker = dest.with_suffix(dest.suffix + ".chunks.json")
    part_dir = dest.parent / f"{dest.stem}.parts"

    done = set()
    if marker.exists():
        try:
            done = {int(i) for i in json.loads(marker.read_text(encoding="utf-8"))}
        except Exception:
            done = set()

    part_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    finished = [len(done)]
    done_at_start = len(done)
    t0 = time.time()

    def fetch(i: int) -> None:
        start = starts[i]
        end = min(start + chunk, total) - 1
        part = part_dir / f"part-{i:04d}.bin"
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    with open(part, "wb") as f:
                        shutil.copyfileobj(r, f, 1024 * 1024)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 30))
        if last_error is not None:
            raise RuntimeError(f"chunk {i} 多次重试仍失败: {last_error}") from last_error
        if part.stat().st_size != end - start + 1:
            raise RuntimeError(f"chunk {i} 大小不完整: {part}")
        with lock:
            done.add(i)
            finished[0] += 1
            marker.write_text(json.dumps(sorted(done)), encoding="utf-8")

    pending = [i for i in range(len(starts)) if i not in done]
    print(f"  {total/1e6:.1f} MB, 共 {len(starts)} 块, 已完成 {len(done)} 块")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(fetch, i): i for i in pending}
        for fut in as_completed(futures):
            fut.result()
            with lock:
                speed = (finished[0] - done_at_start) * chunk / max(time.time() - t0, 0.001) / 1e6
            print(f"  {finished[0]}/{len(starts)} 块完成, 当前约 {speed:.2f} MB/s", flush=True)

    if marker.exists():
        marker.unlink(missing_ok=True)
    with open(dest, "wb") as out:
        for i in range(len(starts)):
            part = part_dir / f"part-{i:04d}.bin"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)
    if dest.stat().st_size != total:
        raise RuntimeError(f"拼接后大小不一致: {dest.stat().st_size} != {total}")
    shutil.rmtree(part_dir, ignore_errors=True)
    return _verify_tar(dest)


def _verify_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as t:
            return len(t.getmembers()) > 0
    except Exception:
        return False


def unpack(tar_path: Path, target_dir: Path, top_level: str) -> bool:
    extracted = target_dir / top_level
    if extracted.exists() and any(extracted.iterdir()):
        print(f"  [跳过] 已解压: {extracted}")
        return True
    print(f"  解压 {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(target_dir, filter="data")
    if extracted.exists() and any(extracted.iterdir()):
        return True
    print("  [错误] 解压后目录为空，可能解压失败")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--only", default="", help="逗号分隔的条目名，只下载这些")
    args = parser.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else None
    for name, url, top_level in ITEMS:
        if names and name not in names:
            continue
        target = BASE / top_level
        if target.exists() and any(target.iterdir()):
            print(f"== {name} == [跳过] 已解压")
            continue
        print(f"== {name} ==")
        tar = BASE / f"{name}.tar.gz"
        if tar.exists() and _verify_tar(tar):
            print(f"  [跳过] 已存在且完整: {tar}")
        else:
            if tar.exists():
                tar.unlink(missing_ok=True)
            ok = parallel_download(url, tar, args.jobs)
            if not ok:
                print(f"  [错误] {name} 下载/校验失败，请重跑脚本")
                continue
        if not unpack(tar, BASE, top_level):
            continue
        tar.unlink(missing_ok=True)
        print()
    print("全部完成。数据目录:", BASE)


if __name__ == "__main__":
    main()
