# -*- coding: utf-8 -*-
"""Focused checks for the causal MarbleNet path.

Run directly with:

    ./.venv/Scripts/python.exe reproductions/marblenet_vad/test_causal.py
"""

from __future__ import annotations

import unittest

import torch

try:
    from .features import MfccConfig, MfccFrontend
    from .model import (
        CausalConv1d,
        MarbleNetStreaming,
        build_marblenet_3x2x64,
    )
except ImportError:
    from features import MfccConfig, MfccFrontend
    from model import (
        CausalConv1d,
        MarbleNetStreaming,
        build_marblenet_3x2x64,
    )


class CausalMarbleNetTests(unittest.TestCase):
    def test_causal_conv_streaming_matches_full_sequence(self) -> None:
        torch.manual_seed(0)
        layer = CausalConv1d(
            3, 4, kernel_size=5, dilation=2, causal=True
        ).eval()
        inputs = torch.randn(2, 3, 37)

        with torch.inference_mode():
            full = layer(inputs)
            first, state = layer.forward_with_cache(inputs[..., :11])
            second, _ = layer.forward_with_cache(inputs[..., 11:], state)
            streamed = torch.cat((first, second), dim=-1)

        torch.testing.assert_close(
            streamed, full, atol=1e-6, rtol=1e-5
        )

    def test_marblenet_encoder_does_not_use_future_frames(self) -> None:
        torch.manual_seed(1)
        model = build_marblenet_3x2x64(causal=True).eval()
        features = torch.randn(2, 64, 80)
        changed = features.clone()
        changed[..., 60:] += 5.0

        with torch.inference_mode():
            original = model.encoder(features)
            modified = model.encoder(changed)

        torch.testing.assert_close(
            original[..., :60],
            modified[..., :60],
            atol=1e-6,
            rtol=1e-5,
        )
        self.assertGreater(
            float((original[..., 60:] - modified[..., 60:]).abs().max()),
            1e-4,
        )

    def test_causal_mfcc_does_not_use_future_samples(self) -> None:
        torch.manual_seed(2)
        frontend = MfccFrontend(MfccConfig(causal=True)).eval()
        waveform = torch.randn(1, 8000)
        changed = waveform.clone()
        changed[:, 4000:] += 5.0

        with torch.inference_mode():
            original = frontend(waveform)
            modified = frontend(changed)

        # Frame i ends at original sample i * hop - 1 because of the
        # left-only FFT-window padding. Frames 0..25 end before sample 4000.
        safe_frames = 4000 // frontend.config.hop_length + 1
        torch.testing.assert_close(
            original[..., :safe_frames],
            modified[..., :safe_frames],
            atol=1e-6,
            rtol=1e-5,
        )
        self.assertGreater(
            float(
                (
                    original[..., safe_frames :]
                    - modified[..., safe_frames :]
                )
                .abs()
                .max()
            ),
            1e-4,
        )

    def test_streaming_single_chunk_matches_full_forward(self) -> None:
        torch.manual_seed(3)
        model = build_marblenet_3x2x64(causal=True).eval()
        features = torch.randn(2, 64, 64)

        with torch.inference_mode():
            full = model(features)
            streamed, _ = model.forward_stream(features)
            wrapper = MarbleNetStreaming(model, context_frames=64)
            wrapped = wrapper.forward(features)

        torch.testing.assert_close(
            streamed, full, atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            wrapped, full, atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
