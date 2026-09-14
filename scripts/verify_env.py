# -*- coding: utf-8 -*-
"""环境自检脚本：检查 G 阶段（轻量神经 VAD）所需全部依赖。"""
import os
import sys
import platform

# 把 matplotlib 缓存放到项目内，避免系统目录写权限问题
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".mplcache"),
)

print(f"Python: {sys.version.split()[0]} ({platform.machine()})")

ok = True


def check(name, fn):
    global ok
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")


def _import(mod):
    __import__(mod)


for m in [
    "numpy", "scipy", "torch", "torchaudio", "librosa", "soundfile",
    "matplotlib", "onnx", "onnxruntime", "webrtcvad", "tqdm", "yaml",
    "einops", "torchmetrics", "d2l.torch", "docx", "torchvision",
    "python_speech_features", "pydub",
]:
    check(f"import {m}", lambda m=m: _import(m))

# GPU
import torch

print(f"torch: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# webrtcvad 冒烟：合成 320ms 信号，按 20ms 帧逐帧判定
import webrtcvad
import numpy as np

vad = webrtcvad.Vad(1)
sr = 16000
t = np.arange(int(0.32 * sr)) / sr
sig = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
frames = [sig[i:i + 320] for i in range(0, len(sig), 320)]
flags = [vad.is_speech(f.tobytes(), sr) for f in frames]
print(f"webrtcvad: 16 帧判定输出 {flags}")

# librosa 特征冒烟
import librosa

y = librosa.tone(440, sr=sr, duration=0.5)
S = librosa.stft(y)
m = librosa.feature.melspectrogram(y=y, sr=sr)
print(f"librosa: STFT {S.shape}, Mel {m.shape}")

# onnxruntime providers
import onnxruntime as ort

print(f"onnxruntime: {ort.__version__}, providers={ort.get_available_providers()}")

# torch 小矩阵 GPU 计算
if torch.cuda.is_available():
    a = torch.randn(64, 64, device="cuda")
    b = (a @ a.T).sum().item()
    print(f"GPU 矩阵乘法 OK: {b:.3f}")

print("=" * 40)
print("ALL PASS" if ok else "SOME FAILED")
