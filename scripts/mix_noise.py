# -*- coding: utf-8 -*-
"""按指定 SNR 混合 speech + MUSAN noise，并验证实际 SNR。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import DATA_ROOT, frame_labels, load_mono, read_tsv, rms_db


def _segment(noise: np.ndarray, length: int, rng: random.Random):
    if len(noise) >= length:
        start = rng.randrange(len(noise) - length + 1)
        return noise[start : start + length].copy()
    reps = int(np.ceil(length / len(noise)))
    tiled = np.tile(noise, reps)[:length]
    return tiled.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-list", type=Path, default=DATA_ROOT / "splits" / "train_speech.tsv")
    parser.add_argument("--noise-list", type=Path, default=DATA_ROOT / "splits" / "train_noise.tsv")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--snrs", type=str, default="-5,0,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "validation" / "mix_examples")
    args = parser.parse_args()

    speech_rows = read_tsv(args.speech_list)
    noise_rows = read_tsv(args.noise_list)
    snrs = [int(x) for x in args.snrs.split(",") if x.strip()]
    rng = random.Random(args.seed)
    speech_rows = rng.sample(speech_rows, min(args.count, len(speech_rows)))

    args.out.mkdir(parents=True, exist_ok=True)
    check_rows = []
    for sidx, speech_row in enumerate(speech_rows):
        speech = load_mono(DATA_ROOT / speech_row["path"])
        noise_row = rng.choice(noise_rows)
        noise_full = load_mono(DATA_ROOT / noise_row["path"])
        noise = _segment(noise_full, len(speech), rng)
        labels = frame_labels(speech)
        out_dir = args.out / f"sample_{sidx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "clean.wav", speech, 16000)
        np.save(out_dir / "labels.npy", labels)

        rms_speech = float(np.sqrt(np.mean(np.square(speech))))
        rms_noise = float(np.sqrt(np.mean(np.square(noise))))
        for snr in snrs:
            scale = rms_speech / (rms_noise * 10 ** (snr / 20) + 1e-12)
            scaled_noise = noise * scale
            mixture = speech + scaled_noise
            peak = float(np.max(np.abs(mixture)))
            if peak > 0.99:
                mixture = mixture * (0.99 / peak)
            actual = rms_db(speech) - rms_db(scaled_noise)
            sf.write(out_dir / f"noise_snr{snr:+.0f}.wav", scaled_noise, 16000)
            sf.write(out_dir / f"mix_snr{snr:+.0f}.wav", mixture, 16000)
            check_rows.append(
                {
                    "sample": out_dir.name,
                    "speech": speech_row["path"],
                    "noise": noise_row["path"],
                    "target_snr": str(snr),
                    "actual_snr": f"{actual:.3f}",
                    "error_db": f"{actual - snr:.3f}",
                }
            )

        meta = {
            "speech": speech_row["path"],
            "noise": noise_row["path"],
            "noise_category": noise_row["category"],
            "noise_subcategory": noise_row["subcategory"],
            "num_frames": int(labels.shape[0]),
            "speech_frames": int(labels.sum()),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    check_path = args.out / "snr_check.tsv"
    header = ["sample", "speech", "noise", "target_snr", "actual_snr", "error_db"]
    with open(check_path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in check_rows:
            f.write("\t".join(row[h] for h in header) + "\n")

    print(f"生成 {len(speech_rows) * len(snrs)} 条混合样例 -> {args.out}")
    for row in check_rows:
        print(
            f"  {row['sample']}  target {row['target_snr']:>3} dB  "
            f"actual {row['actual_snr']:>7} dB  error {row['error_db']:>7} dB"
        )
    print("SNR 校验表:", check_path)


if __name__ == "__main__":
    main()
