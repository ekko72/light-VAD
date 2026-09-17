# -*- coding: utf-8 -*-
"""Focused checks for the difficulty-adaptive temporal-context experiment."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    choose_confidence_threshold,
    count_frames_by_cluster,
    deterministic_speaker_split,
    metrics_from_counts,
    speaker_from_source_key,
)
from reproductions.difficulty_adaptive_context.data import (
    FrameChunkDataset,
    causal_frame_bounds,
    causal_frame_labels,
)
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import (
    build_marblenet_3x2x64,
    count_parameters,
    receptive_field,
)


class ContextExperimentTests(unittest.TestCase):
    def test_speaker_split_is_disjoint_and_deterministic(self) -> None:
        speaker_ids = np.repeat(
            [f"speaker-{index:02d}" for index in range(10)],
            repeats=3,
        )
        first = deterministic_speaker_split(speaker_ids, seed=123)
        second = deterministic_speaker_split(speaker_ids, seed=123)

        calibration, test, calibration_speakers, test_speakers = first
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertFalse(np.any(calibration & test))
        self.assertEqual(set(calibration_speakers) & set(test_speakers), set())
        self.assertEqual(len(calibration_speakers), 5)
        self.assertEqual(len(test_speakers), 5)

    def test_speaker_key_comes_from_source_path(self) -> None:
        self.assertEqual(
            speaker_from_source_key(
                "test-clean/1089/134686/1089-134686-0000.wav"
            ),
            "1089",
        )
        with self.assertRaises(ValueError):
            speaker_from_source_key("")

    def test_fixed_threshold_uses_low_short_confidence(self) -> None:
        scores = np.asarray([0.01, 0.1, 0.4, 0.6, 0.9, 0.99])
        threshold, selected = choose_confidence_threshold(
            scores, activation_rate=0.5
        )

        self.assertEqual(threshold, 0.4)
        self.assertEqual(
            selected.tolist(), [False, True, True, True, False, False]
        )

    def test_gate_counts_track_correction_harm_and_net_utility(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        short_scores = np.asarray([0.9, 0.1, 0.9, 0.1])
        long_scores = np.asarray([0.1, 0.9, 0.9, 0.1])
        selected = np.asarray([True, True, False, False])
        clusters = np.asarray(["s1", "s1", "s2", "s2"])

        names, counts = count_frames_by_cluster(
            labels,
            short_scores,
            long_scores,
            selected,
            clusters,
        )
        metrics = metrics_from_counts(counts.sum(axis=0))

        self.assertEqual(names.tolist(), ["s1", "s2"])
        self.assertEqual(metrics["correction"], 2)
        self.assertEqual(metrics["harm"], 0)
        self.assertEqual(metrics["signed_utility"], 2)
        self.assertAlmostEqual(metrics["net_utility_per_selected"], 1.0)

    def test_short_and_long_have_equal_parameters_and_different_rf(self) -> None:
        short = build_marblenet_3x2x64(
            causal=True, dilation_profile="short", frame_output=True
        )
        long = build_marblenet_3x2x64(
            causal=True, dilation_profile="long", frame_output=True
        )
        short_rf = receptive_field("short")
        long_rf = receptive_field("long")

        self.assertEqual(count_parameters(short), count_parameters(long))
        self.assertEqual(short.dilation_profile, (1, 1, 1, 1, 1))
        self.assertEqual(long.dilation_profile, (1, 2, 3, 4, 4))
        self.assertLess(
            int(short_rf["lookback_frames"]),
            int(long_rf["lookback_frames"]),
        )

    def test_frame_streaming_matches_full_sequence(self) -> None:
        torch.manual_seed(11)
        model = build_marblenet_3x2x64(
            causal=True, dilation_profile="long", frame_output=True
        ).eval()
        features = torch.randn(2, 64, 901)

        with torch.inference_mode():
            full = model(features)
            states = None
            chunks = []
            start = 0
            for length in (73, 251, 177):
                chunk, states = model.forward_stream(
                    features[..., start : start + length], states
                )
                chunks.append(chunk)
                start += length
            chunk, _ = model.forward_stream(features[..., start:], states)
            chunks.append(chunk)
            streamed = torch.cat(chunks, dim=-1)

        torch.testing.assert_close(streamed, full, atol=1e-6, rtol=1e-5)

    def test_frame_bounds_match_causal_mfcc_support(self) -> None:
        frontend = MfccFrontend(MfccConfig(causal=True)).eval()
        n_samples = 1_600
        impulse_at = 400
        waveform = torch.zeros(1, n_samples)
        waveform[0, impulse_at] = 1.0
        with torch.inference_mode():
            baseline = frontend(torch.zeros_like(waveform))
            response = frontend(waveform)
        energy = (response - baseline).abs().sum(dim=1)[0]
        affected = set(
            torch.nonzero(energy > 1e-5).flatten().tolist()
        )
        starts, ends = causal_frame_bounds(int(response.shape[-1]))

        expected = {
            frame
            for frame, (start, end) in enumerate(zip(starts, ends))
            if int(start) <= impulse_at < int(end)
        }
        self.assertEqual(affected, expected)
        self.assertEqual(expected, {3, 4, 5})

    def test_frame_labels_use_the_same_interval_as_mfcc(self) -> None:
        labels = torch.zeros(1_400, dtype=torch.int16).numpy()
        labels[344:744] = 1
        frame_labels = causal_frame_labels(labels, 9)

        # The interval is exactly frame 5's support.  It is also a majority
        # of frames 4 and 6, but not frames 3 or 7.
        self.assertEqual(frame_labels.tolist(), [0, 0, 0, 0, 1, 1, 1, 0, 0])

    def test_training_chunks_can_be_resampled_per_epoch(self) -> None:
        rows = [
            {
                "sample_id": f"sample-{index}",
                "noise_name": "Babble_noise",
                "snr_db": "0",
                "output_audio_path": f"{index}.wav",
                "label_relative_path": f"{index}.npy",
            }
            for index in range(32)
        ]
        with (
            mock.patch(
                "reproductions.difficulty_adaptive_context.data.read_manifest",
                return_value=rows,
            ),
            mock.patch(
                "reproductions.difficulty_adaptive_context.data.audio_frame_count",
                return_value=1_600,
            ),
        ):
            dataset = FrameChunkDataset(
                "unused.tsv",
                generated_root="generated",
                label_root="labels",
                context_samples=800,
                target_samples=800,
                seed=7,
                resample_chunks=True,
            )
            epoch_zero = list(dataset.chunks)
            dataset.set_epoch(1)
            epoch_one = list(dataset.chunks)

        self.assertEqual(len(epoch_zero), len(rows))
        self.assertNotEqual(epoch_zero, epoch_one)


if __name__ == "__main__":
    unittest.main()
