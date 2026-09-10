# -*- coding: utf-8 -*-
"""下载 G 阶段所需公开数据（约 2.2GB），带断点续传与完整性校验。

默认下载：
  - LibriSpeech dev-clean   (~337MB)  干净语音，验证集
  - LibriSpeech test-clean  (~346MB)  干净语音，测试集
  - MUSAN                   (~10.5GB) 噪声/音乐库，合成带噪语音（按需下载）

训练集 train-clean-100 (~6.3GB) 默认一起下载，避免后续训练时缺数据。

用法：
  python scripts/download_data.py
"""

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"
OPENSLR = "https://www.openslr.org/resources"

# (名称, 下载URL, 解压后顶层目录, 预计大小MB)
ITEMS = [
    ("dev-clean", f"{OPENSLR}/12/dev-clean.tar.gz", "LibriSpeech/dev-clean", 337),
    ("test-clean", f"{OPENSLR}/12/test-clean.tar.gz", "LibriSpeech/test-clean", 346),
    ("train-clean-100", f"{OPENSLR}/12/train-clean-100.tar.gz", "LibriSpeech/train-clean-100", 6300),
    ("musan", f"{OPENSLR}/17/musan.tar.gz", "musan", 10500),
]


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def download(url: str, dest: Path) -> bool:
    """断点续传下载；返回 True 表示文件完整可用。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    if total <= 0:
        raise RuntimeError(f"无法获取远程文件大小: {url}")

    if dest.exists():
        have = dest.stat().st_size
        if have >= total and _verify_tar(dest):
            print(f"  [跳过] 已存在且完整: {dest.name} ({have/1e6:.1f} MB)")
            return True
        if have < total:
            print(f"  [续传] 本地已有 {have/1e6:.1f} MB，继续下载剩余 {total/1e6:.1f} MB")
        else:
            print(f"  [重下] 本地文件损坏，删除后重新下载")
            dest.unlink()

    start = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(url, headers=headers)
    print(f"  下载 {dest.name} -> {dest}")
    with urllib.request.urlopen(req, timeout=120) as r:
        # 服务器不支持 Range（返回 200）时，从头写而不是追加
        mode = "ab" if (start and r.status == 206) else "wb"
        if mode == "wb" and start:
            print("  服务器不支持断点续传，从头下载")
        done = start if mode == "ab" else 0
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = min(100, done * 100 // total)
                sys.stdout.write(f"\r  {pct:3d}%  {done/1e6:7.1f} / {total/1e6:.1f} MB")
                sys.stdout.flush()
    print()
    if not _verify_tar(dest):
        print("  [错误] 下载完成后完整性校验失败，请重跑脚本")
        return False
    return True


def _verify_tar(path: Path) -> bool:
    """用 tarfile 读一遍整个包，校验 gzip CRC 与 tar 结构。"""
    try:
        with tarfile.open(path, "r:gz") as t:
            members = t.getmembers()
        return len(members) > 0
    except Exception:
        return False


def unpack(tar_path: Path, target_dir: Path, top_level: str) -> bool:
    """解压到 target_dir；若目标目录已存在且非空则跳过。"""
    extracted = target_dir / top_level
    if extracted.exists() and any(extracted.iterdir()):
        print(f"  [跳过] 已解压: {extracted}")
        return True
    print(f"  解压 {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(target_dir, filter="data")
    if extracted.exists() and any(extracted.iterdir()):
        print(f"  完成 -> {extracted}")
        return True
    print("  [错误] 解压后目录为空，可能解压失败")
    return False


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    for name, url, top_level, _mb in ITEMS:
        print(f"== {name} ==")
        tar = BASE / f"{name}.tar.gz"
        if not download(url, tar):
            print("  下载失败，跳过解压")
            continue
        if not unpack(tar, BASE, top_level):
            print("  解压失败，跳过")
            continue
        # 解压成功后删除压缩包，节省磁盘（可注释掉保留）
        tar.unlink(missing_ok=True)
        print()
    print("全部完成。数据目录:", BASE)


if __name__ == "__main__":
    main()
