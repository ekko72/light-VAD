# -*- coding: utf-8 -*-
"""MFCC front-end matching NeMo's ``AudioToMFCCPreprocessor``.

NeMo (``marblenet_3x2x64.yaml``) configures::

    window_size: 0.025
    window_stride: 0.01
    window: "hann"
    n_mels: 64
    n_mfcc: 64
    n_fft: 512

and then calls ``torchaudio.transforms.MFCC(..., dct_type=2, norm="ortho",
log_mels=True)``. This module builds exactly that transform, so the features
are bit-comparable with NeMo's preprocessor for the same waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MfccConfig:
    """Feature configuration of MarbleNet-3x2x64 (NeMo defaults)."""

    sample_rate: int = 16_000
    window_size: float = 0.025
    window_stride: float = 0.01
    n_fft: int = 512
    n_mels: int = 64
    n_mfcc: int = 64
    dct_type: int = 2
    norm: str = "ortho"
    log_mels: bool = True
    causal: bool = False

    @property
    def win_length(self) -> int:
        """Window length in samples (0.025 s -> 400)."""
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        """Hop length in samples (0.01 s -> 160)."""
        return int(self.window_stride * self.sample_rate)

    def frames_for_samples(self, n_samples: int) -> int:
        """Frame count produced for ``n_samples``.

        The causal frontend pads one full FFT window on the left, so its
        frame count is the same as the centered frontend for all inputs.
        """
        return int(n_samples) // self.hop_length + 1

    def samples_for_seconds(self, seconds: float) -> int:
        return int(round(float(seconds) * self.sample_rate))

    def frames_for_seconds(self, seconds: float) -> int:
        return self.frames_for_samples(self.samples_for_seconds(seconds))


class MfccFrontend(nn.Module):
    """``[B, T]`` waveform in [-1, 1] -> ``[B, n_mfcc, frames]`` MFCC."""

    def __init__(self, config: MfccConfig | None = None) -> None:
        super().__init__()
        self.config = config or MfccConfig()
        cfg = self.config
        melkwargs = {
            "f_min": 0.0,
            "f_max": None,
            "n_mels": cfg.n_mels,
            "n_fft": cfg.n_fft,
            "win_length": cfg.win_length,
            "hop_length": cfg.hop_length,
            "window_fn": torch.hann_window,
            "center": not cfg.causal,
        }
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            dct_type=cfg.dct_type,
            norm=cfg.norm,
            log_mels=cfg.log_mels,
            melkwargs=melkwargs,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 2:
            raise ValueError(
                f"expected [B, T] waveform, got {tuple(waveform.shape)}"
            )
        if self.config.causal:
            # One full FFT window of left padding keeps the frame count
            # unchanged while ensuring every STFT frame only sees the
            # current and previous samples.
            waveform = F.pad(waveform, (self.config.n_fft, 0))
        return self.mfcc(waveform)


def main() -> int:
    config = MfccConfig()
    frontend = MfccFrontend(config)
    for seconds in (0.63, 1.0):
        samples = config.samples_for_seconds(seconds)
        waveform = torch.randn(2, samples)
        features = frontend(waveform)
        print(
            f"{seconds:.2f} s -> {samples} samples -> "
            f"{tuple(features.shape)} MFCC "
            f"(expected frames={config.frames_for_samples(samples)})"
        )
    print(
        "window/stride: "
        f"{config.win_length}/{config.hop_length} samples, "
        f"n_fft={config.n_fft}, n_mels={config.n_mels}, "
        f"n_mfcc={config.n_mfcc}, log_mels={config.log_mels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
