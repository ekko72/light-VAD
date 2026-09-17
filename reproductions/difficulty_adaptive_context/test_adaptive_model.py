# -*- coding: utf-8 -*-
"""Unit checks for the Phase A3 shared adaptive model."""

from __future__ import annotations

import unittest

import torch

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveCausalVAD,
    AdaptiveStreamingVAD,
    RefinementConfig,
    SparseCausalMultiScaleRefinement,
    confidence_from_logits,
    selection_from_confidence,
)
from reproductions.marblenet_vad.model import build_marblenet_3x2x64


class AdaptiveModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.short = build_marblenet_3x2x64(
            causal=True,
            dilation_profile="short",
            frame_output=True,
        ).eval()
        self.refinement = SparseCausalMultiScaleRefinement(
            RefinementConfig(
                in_channels=128,
                num_classes=2,
                kernel_size=5,
                dilations=(1, 2, 4),
                max_residual=2.0,
            )
        ).eval()
        self.model = AdaptiveCausalVAD(self.short, self.refinement).eval()

    def test_sparse_refinement_matches_dense_at_selected_frames(self) -> None:
        encoded = torch.randn(2, 128, 41)
        selected = torch.zeros(2, 41, dtype=torch.bool)
        selected[0, 0] = True
        selected[0, 17] = True
        selected[0, 40] = True
        selected[1, 5] = True
        selected[1, 39] = True

        with torch.inference_mode():
            dense = self.refinement(encoded)
            sparse = self.refinement.forward_sparse(encoded, selected)

        torch.testing.assert_close(
            sparse.permute(0, 2, 1).reshape(-1, 2)[selected.reshape(-1)],
            dense.permute(0, 2, 1).reshape(-1, 2)[selected.reshape(-1)],
            atol=1e-6,
            rtol=1e-5,
        )
        self.assertEqual(
            int(torch.count_nonzero(sparse.permute(0, 2, 1)[~selected])),
            0,
        )

    def test_sparse_refinement_does_not_read_future_frames(self) -> None:
        encoded = torch.randn(1, 128, 30)
        changed = encoded.clone()
        changed[:, :, 21:] += 100.0
        selected = torch.zeros(1, 30, dtype=torch.bool)
        selected[0, 20] = True

        with torch.inference_mode():
            first = self.refinement.forward_sparse(encoded, selected)
            second = self.refinement.forward_sparse(changed, selected)

        torch.testing.assert_close(first, second)

    def test_zero_initialized_output_makes_adaptive_equal_to_short(self) -> None:
        features = torch.randn(2, 64, 71)
        with torch.inference_mode():
            short_logits = self.short(features)
            adaptive = self.model(
                features,
                activation_threshold=0.5,
            )

        torch.testing.assert_close(adaptive, short_logits)

    def test_confidence_gate_selects_low_confidence_frames(self) -> None:
        logits = torch.tensor(
            [
                [
                    [3.0, 0.0, -3.0, 0.0],
                    [-3.0, 0.0, 3.0, 0.0],
                ]
            ]
        )
        confidence = confidence_from_logits(logits)
        selected = selection_from_confidence(confidence, 0.001)

        self.assertEqual(
            selected.tolist(),
            [[False, True, False, True]],
        )

    def test_streaming_matches_full_sequence_with_fixed_gate(self) -> None:
        features = torch.randn(2, 64, 251)
        threshold = 0.30
        with torch.inference_mode():
            full = self.model(
                features,
                activation_threshold=threshold,
            )
            stream = AdaptiveStreamingVAD(
                self.model,
                activation_threshold=threshold,
            )
            chunks = []
            start = 0
            for length in (37, 83, 61):
                chunks.append(stream.forward(features[..., start : start + length]))
                start += length
            chunks.append(stream.forward(features[..., start:]))
            streamed = torch.cat(chunks, dim=-1)

        torch.testing.assert_close(streamed, full, atol=1e-6, rtol=1e-5)
        self.assertGreater(stream.cache_bytes, 0)

    def test_skipped_frames_do_not_change_later_refinement(self) -> None:
        encoded = torch.randn(1, 128, 80)
        first_mask = torch.zeros(1, 80, dtype=torch.bool)
        first_mask[0, 65:] = True
        second_mask = torch.zeros(1, 80, dtype=torch.bool)
        second_mask[0, 64:] = True

        with torch.inference_mode():
            first = self.refinement.forward_sparse(encoded, first_mask)
            second = self.refinement.forward_sparse(encoded, second_mask)

        # The additional activation at frame 64 cannot change frame 65 or
        # later through hidden state; both paths are stateless.
        torch.testing.assert_close(first[..., 65:], second[..., 65:])


if __name__ == "__main__":
    unittest.main()
