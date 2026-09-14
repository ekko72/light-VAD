# -*- coding: utf-8 -*-
"""数据预处理共享工具：读取、重采样、帧标签。"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import soundfile as sf

TARGET_SR = 16000
FRAME_LENGTH = 480  # 30 ms @16k
FRAME_SHIFT = 160   # 10 ms @16k
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_mono(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """读取音频并统一为 16k 单声道 float32。"""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    else:
        y = y[:, 0]
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def frame_labels(
    y: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    frame_shift: int = FRAME_SHIFT,
    rms_ratio: float = 0.02,
    dilation: int = 2,
) -> np.ndarray:
    """按 clean speech 帧 RMS 生成 speech/non-speech 标签，并做 2 帧扩张。"""
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    windows = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::frame_shift]
    n_frames = (len(y) - frame_length) // frame_shift + 1
    windows = windows[:n_frames]
    rms = np.sqrt(np.mean(np.square(windows), axis=1))
    threshold = max(float(rms.max()) * rms_ratio, 1e-6)
    labels = (rms > threshold).astype(np.uint8)
    if dilation > 0:
        structure = np.ones(dilation * 2 + 1, dtype=bool)
        labels = scipy.ndimage.binary_dilation(labels, structure=structure).astype(np.uint8)
    return labels


def rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(x)) + 1e-12))


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            values = line.split("\t")
            rows.append(dict(zip(header, values)))
    return rows
