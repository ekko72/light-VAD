# -*- coding: utf-8 -*-
"""音频抽查：随机取样语音/噪声，打印元信息，并生成可播放与可复核的产物。

覆盖原子任务：
- G0-13 随机读取 LibriSpeech 音频并打印采样率/时长
- G0-14 随机播放/检查若干条语音
- G0-15 随机检查若干条 MUSAN noise
- G0-16 统一采样率到项目目标采样率（读取时重采样，不落盘）

产物写入 data/validation/audio_check/（已被 .gitignore 排除）：
- {speech,noise}_check.tsv    逐条明细
- {speech,noise}_montage.wav  每条 3 s 拼接音频，便于一次听完
- {speech,noise}_check.png    波形 + log-mel 频谱图
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import DATA_ROOT, TARGET_SR, load_mono, read_tsv

SPEECH_MANIFEST = DATA_ROOT / "manifests" / "speech_manifest.tsv"
NOISE_MANIFEST = DATA_ROOT / "manifests" / "musan_manifest.tsv"
CLIP_SECONDS = 3.0
GAP_SECONDS = 0.3
PEAK_TARGET = 0.7
SILENT_DBFS = -60.0
CLIP_FRACTION = 0.001  # 近满幅样本占比超过 0.1% 才判为削波


def rms_dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))


def pick_segment(y: np.ndarray, seconds: float, rng: random.Random) -> np.ndarray:
    n = int(seconds * TARGET_SR)
    if len(y) <= n:
        return y
    start = rng.randrange(0, len(y) - n + 1)
    return y[start : start + n]


def normalize_peak(y: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-9:
        return y.astype(np.float32)
    return (y * (target / peak)).astype(np.float32)


def inspect_one(rel_path: str, rng: random.Random) -> dict:
    path = DATA_ROOT / rel_path
    info = sf.info(str(path))
    y = load_mono(path)
    abs_y = np.abs(y)
    peak = float(np.max(abs_y)) if y.size else 0.0
    clip_fraction = float(np.mean(abs_y >= 0.999)) if y.size else 0.0
    rms_db = rms_dbfs(y)
    if rms_db < SILENT_DBFS:
        flag = "silent"
    elif clip_fraction > CLIP_FRACTION:
        flag = "clipped"
    else:
        flag = "ok"
    return {
        "path": rel_path,
        "source_sr": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "resampled": int(info.samplerate) != TARGET_SR,
        "duration_source_s": info.frames / info.samplerate,
        "duration_target_s": len(y) / TARGET_SR,
        "peak": peak,
        "clip_fraction": clip_fraction,
        "rms_dbfs": rms_db,
        "flag": flag,
        "segment": normalize_peak(pick_segment(y, CLIP_SECONDS, rng)),
    }


def write_summary(records: list[dict], out_tsv: Path) -> None:
    header = [
        "index", "path", "source_sr", "target_sr", "resampled", "channels",
        "subtype", "duration_source_s", "duration_target_s", "peak",
        "clip_fraction", "rms_dbfs", "flag",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for idx, row in enumerate(records, 1):
            values = [
                str(idx), row["path"], str(row["source_sr"]), str(TARGET_SR),
                str(row["resampled"]), str(row["channels"]), str(row["subtype"]),
                f"{row['duration_source_s']:.4f}", f"{row['duration_target_s']:.4f}",
                f"{row['peak']:.5f}", f"{row['clip_fraction']:.6f}",
                f"{row['rms_dbfs']:.3f}", row["flag"],
            ]
            f.write("\t".join(values) + "\n")


def build_montage(records: list[dict], out_wav: Path) -> float:
    gap = np.zeros(int(GAP_SECONDS * TARGET_SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for row in records:
        parts.append(np.asarray(row["segment"], dtype=np.float32))
        parts.append(gap)
    montage = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), montage, TARGET_SR, subtype="PCM_16")
    return len(montage) / TARGET_SR


def plot_checks(records: list[dict], title: str, out_png: Path) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(12, 1.3 * n), squeeze=False)
    for row, ax_wave, ax_spec in zip(records, axes[:, 0], axes[:, 1]):
        seg = row["segment"]
        t = np.arange(len(seg)) / TARGET_SR
        ax_wave.plot(t, seg, linewidth=0.5, color="#3b6ea5")
        ax_wave.set_ylim(-1.05, 1.05)
        ax_wave.set_yticks([])
        ax_wave.tick_params(labelsize=5)
        ax_wave.set_ylabel(Path(row["path"]).name[:30], fontsize=5.5, rotation=0, ha="right", va="center")

        mel = librosa.feature.melspectrogram(y=seg, sr=TARGET_SR, n_mels=64, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)
        ax_spec.imshow(db, aspect="auto", origin="lower", cmap="magma")
        ax_spec.set_yticks([])
        ax_spec.tick_params(labelsize=5)

    axes[0, 0].set_title("waveform (3 s clip, peak-normalised)", fontsize=8)
    axes[0, 1].set_title("log-mel spectrogram", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def verify_resample(out_dir: Path) -> dict:
    """数据本身都是 16 kHz；用 44.1 kHz 合成音单独验证重采样路径。"""
    probe = out_dir / "_resample_probe_44k.wav"
    src_sr = 44100
    t = np.arange(int(0.5 * src_sr)) / src_sr
    tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(probe), tone, src_sr, subtype="PCM_16")
    try:
        y = load_mono(probe)
    finally:
        probe.unlink(missing_ok=True)
    return {
        "source_sr": src_sr,
        "target_sr": TARGET_SR,
        "n_in": len(tone),
        "n_out": len(y),
        "mono_float32": (y.ndim == 1 and y.dtype == np.float32),
        "expected_out": int(0.5 * TARGET_SR),
    }


def play_windows(path: Path) -> None:
    try:
        import winsound
    except ImportError:
        print("  [跳过播放] winsound 仅在 Windows 可用，请手动打开 wav 试听")
        return
    print(f"  [播放] {path.name}（按 Ctrl+C 可中断）")
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def report(records: list[dict], name: str, title: str, args: argparse.Namespace) -> None:
    count = len(records)
    print(f"\n=== {title}：随机 {count} 条（seed={args.seed}）===")
    print(
        f"{'#':>3}  {'source_sr':>9}  {'target_sr':>9}  {'duration_s':>10}  "
        f"{'peak':>6}  {'clip%':>7}  {'rms_dBFS':>8}  {'flag':<8} path"
    )
    for idx, row in enumerate(records, 1):
        print(
            f"{idx:>3}  {row['source_sr']:>9}  {TARGET_SR:>9}  "
            f"{row['duration_source_s']:>10.3f}  {row['peak']:>6.3f}  "
            f"{row['clip_fraction'] * 100:>6.3f}%  {row['rms_dbfs']:>8.2f}  "
            f"{row['flag']:<8} {row['path']}"
        )
    resampled = sum(1 for r in records if r["resampled"])
    print(f"  需要重采样的条目: {resampled}/{count}（其余本来就是 {TARGET_SR} Hz）")
    print(f"  质量标记统计: {dict(Counter(r['flag'] for r in records))}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / f"{name}_check.tsv"
    montage = args.out / f"{name}_montage.wav"
    png = args.out / f"{name}_check.png"
    write_summary(records, summary)
    duration = build_montage(records, montage)
    plot_checks(records, f"{title} · seed={args.seed} · {count} clips", png)
    print(f"  明细表: {summary}")
    print(f"  拼接音频: {montage}（{duration:.1f} s，每条 {CLIP_SECONDS:.0f} s + {GAP_SECONDS} s 间隔，已峰值归一化）")
    print(f"  波形图: {png}")
    if args.play:
        play_windows(montage)


def main() -> None:
    parser = argparse.ArgumentParser(description="随机抽查 LibriSpeech 语音与 MUSAN 噪声")
    parser.add_argument("--count", type=int, default=10, help="每个数据集抽查条数（默认 10）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42，与数据划分一致）")
    parser.add_argument("--dataset", choices=["speech", "noise", "both"], default="both")
    parser.add_argument("--play", action="store_true", help="在 Windows 上播放拼接音频")
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "validation" / "audio_check",
        help="产物输出目录",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    probe = verify_resample(args.out)
    print("=== G0-16 采样率统一验证（合成 44.1 kHz 音）===")
    print(
        f"  {probe['source_sr']} Hz -> {probe['target_sr']} Hz   "
        f"{probe['n_in']} -> {probe['n_out']} samples（期望 {probe['expected_out']}）   "
        f"mono+float32: {probe['mono_float32']}"
    )

    jobs = []
    if args.dataset in ("speech", "both"):
        jobs.append(("speech", SPEECH_MANIFEST, "LibriSpeech 语音"))
    if args.dataset in ("noise", "both"):
        jobs.append(("noise", NOISE_MANIFEST, "MUSAN 噪声"))

    for name, manifest, title in jobs:
        if not manifest.exists():
            print(f"[错误] 缺少 {manifest}")
            print("       请先运行: python scripts\\prepare_manifests.py")
            sys.exit(1)
        rows = read_tsv(manifest)
        if not rows:
            print(f"[错误] {manifest} 为空")
            sys.exit(1)
        count = min(args.count, len(rows))
        sample = rng.sample(rows, count)
        records = [inspect_one(row["path"], rng) for row in sample]
        report(records, name, title, args)

    print(f"\n全部产物位于: {args.out}")


if __name__ == "__main__":
    main()