# -*- coding: utf-8 -*-
"""生成并抽查固定帧尺寸下的 speech/non-speech 标签。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import (
    DATA_ROOT,
    FRAME_LENGTH,
    FRAME_SHIFT,
    TARGET_SR,
    frame_labels,
    load_mono,
    read_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DATA_ROOT / "manifests" / "speech_manifest.tsv")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "validation" / "labels")
    args = parser.parse_args()

    all_rows = [r for r in read_tsv(args.manifest) if r["split"] == "train"]
    rng = random.Random(args.seed)
    selected = rng.sample(all_rows, min(args.limit, len(all_rows)))
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for row in selected:
        audio = load_mono(DATA_ROOT / row["path"])
        labels = frame_labels(audio)
        stem = Path(row["path"]).stem
        np.save(args.out / f"{stem}.npy", labels)
        summary_rows.append(
            {
                "path": row["path"],
                "duration_s": row["duration_s"],
                "num_frames": str(len(labels)),
                "speech_frames": str(int(labels.sum())),
                "speech_ratio": f"{float(labels.mean()):.4f}",
            }
        )

    summary_path = args.out / "label_summary.tsv"
    header = ["path", "duration_s", "num_frames", "speech_frames", "speech_ratio"]
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in summary_rows:
            f.write("\t".join(row[h] for h in header) + "\n")

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 1.7 * len(selected)))
    if len(selected) == 1:
        axes = [axes]
    for ax, row, summary_row in zip(axes, selected, summary_rows):
        audio = load_mono(DATA_ROOT / row["path"])
        labels = frame_labels(audio)
        t = np.arange(len(audio)) / TARGET_SR
        ax.plot(t, audio, color="#4a6fa5", linewidth=0.5)
        label_t = np.arange(len(labels)) * FRAME_SHIFT / TARGET_SR
        ax.fill_between(
            label_t,
            -0.2,
            0.2,
            where=labels.astype(bool),
            color="#e07b39",
            alpha=0.35,
            step="post",
            label="speech frame",
        )
        ax.set_ylabel(f"#{summary_row['speech_ratio']}", fontsize=8)
        ax.set_yticks([])
        if ax is axes[-1]:
            ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    png = DATA_ROOT / "validation" / "label_check.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    print(f"抽查 {len(selected)} 条标签 -> {args.out}")
    print(f"标签汇总 -> {summary_path}")
    print(f"可视化 -> {png}")


if __name__ == "__main__":
    main()
