# -*- coding: utf-8 -*-
"""Waveform and feature augmentation matching NeMo's implementations.

Sources (NeMo v1.0.0, the version the paper was trained with):

* ``nemo/collections/asr/parts/perturb.py``      -- ShiftPerturbation,
  WhiteNoisePerturbation
* ``nemo/collections/asr/parts/spectr_augment.py`` -- SpecAugment, SpecCutout

The paper applies the waveform perturbation "with a probability of 80%",
while the current NeMo config sets ``prob: 1.0`` for both shift and white
noise. ``WaveformAugmentor.prob`` is therefore configurable and defaults to
the paper value.
"""

from __future__ import annotations

import random

import torch


class WaveformAugmentor:
    """Time shift (+/- 5 ms) and white noise (-90..-46 dB) on raw audio."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        prob: float = 0.8,
        min_shift_ms: float = -5.0,
        max_shift_ms: float = 5.0,
        min_noise_db: int = -90,
        max_noise_db: int = -46,
        rng: random.Random | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.prob = float(prob)
        self.min_shift_ms = float(min_shift_ms)
        self.max_shift_ms = float(max_shift_ms)
        self.min_noise_db = int(min_noise_db)
        self.max_noise_db = int(max_noise_db)
        self.rng = random.Random() if rng is None else rng

    def _shift(self, waveform: torch.Tensor) -> None:
        """Shift one item in place; the vacated samples are zeroed.

        Sign convention copied from NeMo: a positive shift moves the waveform
        earlier (``out[i] = in[i + s]``), a negative shift moves it later.
        """
        length = waveform.shape[-1]
        shift_ms = self.rng.uniform(self.min_shift_ms, self.max_shift_ms)
        shift_samples = int(shift_ms * self.sample_rate // 1000)
        if shift_samples == 0:
            return
        if shift_samples > 0:
            waveform[..., : length - shift_samples] = waveform[
                ..., shift_samples:
            ].clone()
            waveform[..., length - shift_samples :] = 0.0
        else:
            offset = -shift_samples
            waveform[..., offset:] = waveform[..., : length - offset].clone()
            waveform[..., :offset] = 0.0

    def _white_noise(self, waveform: torch.Tensor) -> None:
        level_db = self.rng.randint(self.min_noise_db, self.max_noise_db)
        amplitude = 10.0 ** (level_db / 20.0)
        noise = torch.randn_like(waveform) * amplitude
        waveform.add_(noise)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, T]`` waveform -> augmented copy (``prob`` gates both steps)."""
        out = waveform.clone()
        for index in range(out.shape[0]):
            if self.rng.random() >= self.prob:
                continue
            self._shift(out[index])
            self._white_noise(out[index])
        return out


class SpecAugment:
    """NeMo SpecAugment: ``freq_masks``/``time_masks`` zeroed bands."""

    def __init__(
        self,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_width: int = 15,
        time_width: int = 25,
        mask_value: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self.freq_masks = int(freq_masks)
        self.time_masks = int(time_masks)
        self.freq_width = int(freq_width)
        self.time_width = int(time_width)
        self.mask_value = float(mask_value)
        self.rng = random.Random() if rng is None else rng

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        out = features.clone()
        batch, n_freq, n_time = out.shape
        if n_freq <= self.freq_width or n_time <= self.time_width:
            return out
        for index in range(batch):
            for _ in range(self.freq_masks):
                left = self.rng.randint(0, n_freq - self.freq_width)
                width = self.rng.randint(0, self.freq_width)
                out[index, left : left + width, :] = self.mask_value
            for _ in range(self.time_masks):
                left = self.rng.randint(0, n_time - self.time_width)
                width = self.rng.randint(0, self.time_width)
                out[index, :, left : left + width] = self.mask_value
        return out


class SpecCutout:
    """NeMo SpecCutout: random rectangles zeroed in time/frequency."""

    def __init__(
        self,
        rect_masks: int = 5,
        rect_time: int = 25,
        rect_freq: int = 15,
        rng: random.Random | None = None,
    ) -> None:
        self.rect_masks = int(rect_masks)
        self.rect_time = int(rect_time)
        self.rect_freq = int(rect_freq)
        self.rng = random.Random() if rng is None else rng

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        out = features.clone()
        batch, n_freq, n_time = out.shape
        if n_freq <= self.rect_freq or n_time <= self.rect_time:
            return out
        for index in range(batch):
            for _ in range(self.rect_masks):
                freq_start = self.rng.randint(0, n_freq - self.rect_freq)
                time_start = self.rng.randint(0, n_time - self.rect_time)
                freq_width = self.rng.randint(0, self.rect_freq)
                time_width = self.rng.randint(0, self.rect_time)
                out[
                    index,
                    freq_start : freq_start + freq_width,
                    time_start : time_start + time_width,
                ] = 0.0
        return out


class FeatureAugmentor:
    """Apply SpecAugment then SpecCutout, as NeMo's SpectrogramAugmentation."""

    def __init__(
        self,
        spec_augment: SpecAugment | None = None,
        spec_cutout: SpecCutout | None = None,
    ) -> None:
        self.spec_augment = spec_augment
        self.spec_cutout = spec_cutout

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        out = features
        if self.spec_augment is not None:
            out = self.spec_augment(out)
        if self.spec_cutout is not None:
            out = self.spec_cutout(out)
        return out
