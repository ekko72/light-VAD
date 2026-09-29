# -*- coding: utf-8 -*-
"""Unit checks for the A13 streaming and segment audit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveCausalVAD,
    RefinementConfig,
    SparseCausalMultiScaleRefinement,
)
from reproductions.difficulty_adaptive_context.run_a13_streaming_audit import (
    SegmentProtocol,
    Utterance,
    _chunk_lengths,
    _tensor_prefix_check,
    assess_a13,
    audit_causality,
    build_segment_rows,
    evaluate_segment_scores,
    find_runs,
    load_a13_bundle,
)
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import build_marblenet_3x2x64


class A13StreamingAuditTests(unittest.TestCase):
    def test_find_runs_returns_half_open_intervals(self) -> None:
        self.assertEqual(
            find_runs(np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)),
            [(1, 3), (4, 5), (7, 8)],
        )
        self.assertEqual(find_runs(np.zeros(4, dtype=bool)), [])

    def test_segment_metrics_cover_alignment_and_flicker(self) -> None:
        labels = np.asarray(
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            dtype=np.int64,
        )
        scores = np.asarray(
            [0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]
        )
        utterances = [
            Utterance(
                start=0,
                end=12,
                source_key="a",
                condition="20",
                noise_name="Babble_noise",
            )
        ]
        protocol = SegmentProtocol(
            decision_threshold=0.5,
            short_speech_max_frames=4,
            onset_tolerance_frames=2,
        )

        metrics = evaluate_segment_scores(
            labels,
            scores,
            utterances,
            protocol,
        )

        self.assertEqual(metrics["speech_runs"], 2)
        self.assertEqual(metrics["short_speech_runs"], 2)
        self.assertAlmostEqual(metrics["short_speech_recall"], 0.5)
        self.assertEqual(metrics["whole_speech_run_misses"], 1)
        self.assertEqual(metrics["whole_utterance_misses"], 0)
        self.assertEqual(metrics["onset_delay_median_frames"], 1.0)
        self.assertEqual(metrics["offset_delay_median_frames"], 0.0)
        self.assertEqual(metrics["speech_clipping_duration_frames"], 2)
        self.assertEqual(metrics["false_activation_duration_frames"], 1)
        self.assertEqual(metrics["prediction_runs"], 2)
        self.assertEqual(metrics["at_most_two_frame_prediction_runs"], 2)

    def test_bundle_loader_rejects_mixed_test_utterance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            size = 4
            np.savez(
                path,
                labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
                short_scores=np.full(size, 0.4),
                adaptive_scores=np.full(size, 0.4),
                full_adaptive_scores=np.full(size, 0.6),
                selected=np.ones(size, dtype=bool),
                test_mask=np.asarray([True, False, True, True]),
                calibration_mask=np.asarray([False, True, False, False]),
                source_key=np.asarray(["a", "a", "a", "a"]),
                condition=np.asarray(["20"] * size),
                noise_name=np.asarray(["Babble_noise"] * size),
                threshold=np.asarray(0.13),
            )

            bundle = load_a13_bundle(path)
            from reproductions.difficulty_adaptive_context.run_a13_streaming_audit import (
                build_utterances,
            )

            with self.assertRaisesRegex(ValueError, "test_mask crosses"):
                build_utterances(bundle)

    def test_chunk_lengths_cover_every_frame(self) -> None:
        lengths = _chunk_lengths(31, (1, 7, 13))
        self.assertEqual(sum(lengths), 31)
        self.assertTrue(all(length > 0 for length in lengths))

    def test_prefix_tensor_check_detects_future_leak(self) -> None:
        first = torch.tensor([[[1.0, 2.0, 3.0]]])
        second = first.clone()
        second[..., 0] += 0.1

        check = _tensor_prefix_check(
            "future",
            first,
            second,
            2,
            atol=1e-6,
            rtol=1e-5,
        )

        self.assertFalse(check["passed"])
        self.assertAlmostEqual(check["max_abs_diff"], 0.1, places=6)

    def test_segment_rows_compare_three_methods(self) -> None:
        bundle = {
            "labels": np.asarray([0, 1, 1, 0], dtype=np.int64),
            "short_scores": np.asarray([0.1, 0.6, 0.6, 0.1]),
            "full_adaptive_scores": np.asarray([0.1, 0.9, 0.9, 0.1]),
            "adaptive_scores": np.asarray([0.1, 0.9, 0.9, 0.1]),
        }
        utterances = [
            Utterance(0, 4, "a", "20", "Babble_noise"),
        ]
        rows = build_segment_rows(
            bundle,
            utterances,
            SegmentProtocol(),
        )

        self.assertEqual(
            {(row["method"], row["group"]) for row in rows},
            {
                ("short", "overall"),
                ("short", "noisy"),
                ("short", "seen"),
                ("short", "condition:20"),
                ("refine_only", "overall"),
                ("refine_only", "noisy"),
                ("refine_only", "seen"),
                ("refine_only", "condition:20"),
                ("adaptive", "overall"),
                ("adaptive", "noisy"),
                ("adaptive", "seen"),
                ("adaptive", "condition:20"),
            },
        )

    def test_assessment_requires_causality_and_non_regression(self) -> None:
        base = {
            "short_speech_recall": 0.8,
            "whole_speech_run_misses": 2,
            "speech_runs": 100,
            "speech_clipping_rate": 0.1,
            "false_activation_rate": 0.05,
            "onset_delay_median_frames": 2.0,
            "offset_delay_median_frames": 1.0,
            "at_most_two_frame_prediction_run_rate": 0.2,
            "decision_transitions_per_1000_frames": 10.0,
        }
        rows = []
        for group in ("overall", "noisy", "seen", "unseen"):
            rows.extend(
                (
                    {"method": "short", "group": group, **base},
                    {
                        "method": "adaptive",
                        "group": group,
                        **{
                            **base,
                            "short_speech_recall": 0.79,
                        },
                    },
                )
            )

        passed = assess_a13(rows, {"status": "PASS"})
        failed = assess_a13(rows, {"status": "FAIL"})

        self.assertEqual(passed["status"], "PASS")
        self.assertEqual(failed["status"], "FAIL_CAUSALITY")

    def test_assessment_note_tracks_loaded_a12_status(self) -> None:
        base = {
            "short_speech_recall": 0.8,
            "whole_speech_run_misses": 2,
            "speech_runs": 100,
            "speech_clipping_rate": 0.1,
            "false_activation_rate": 0.05,
            "onset_delay_median_frames": 2.0,
            "offset_delay_median_frames": 1.0,
            "at_most_two_frame_prediction_run_rate": 0.2,
            "decision_transitions_per_1000_frames": 10.0,
        }
        rows = []
        for group in ("overall", "noisy", "seen", "unseen"):
            rows.extend(
                (
                    {"method": "short", "group": group, **base},
                    {"method": "adaptive", "group": group, **base},
                )
            )

        assessment = assess_a13(
            rows,
            {"status": "PASS"},
            a12_status="GO_WITH_SOURCE_RANK_STABILIZER",
        )

        self.assertIn("GO_WITH_SOURCE_RANK_STABILIZER", assessment["note"])
        self.assertNotIn("NO_GO", assessment["note"])

    def test_causality_audit_matches_prefix_with_different_futures(self) -> None:
        torch.manual_seed(11)
        short = build_marblenet_3x2x64(
            causal=True,
            dilation_profile="short",
            frame_output=True,
        ).eval()
        refinement = SparseCausalMultiScaleRefinement(
            RefinementConfig(
                in_channels=128,
                num_classes=2,
                kernel_size=5,
                dilations=(1, 2, 4),
                max_residual=2.0,
            )
        ).eval()
        model = AdaptiveCausalVAD(short, refinement).eval()
        frontend = MfccFrontend(MfccConfig(causal=True)).eval()

        audit = audit_causality(
            model,
            frontend,
            prefix_samples=960,
            total_samples=1_280,
            activation_threshold=0.13,
            seed=7,
        )

        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(check["passed"] for check in audit["checks"]))

    def test_causality_audit_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation_threshold"):
            audit_causality(
                nn.Linear(1, 1),
                MfccFrontend(MfccConfig(causal=True)).eval(),
                prefix_samples=800,
                total_samples=1_280,
                activation_threshold=0.6,
            )

    def test_causality_audit_rejects_non_causal_frontend(self) -> None:
        torch.manual_seed(11)
        short = build_marblenet_3x2x64(
            causal=True,
            dilation_profile="short",
            frame_output=True,
        ).eval()
        refinement = SparseCausalMultiScaleRefinement(
            RefinementConfig(
                in_channels=128,
                num_classes=2,
                kernel_size=5,
                dilations=(1, 2, 4),
                max_residual=2.0,
            )
        ).eval()
        model = AdaptiveCausalVAD(short, refinement).eval()

        audit = audit_causality(
            model,
            MfccFrontend(MfccConfig(causal=False)).eval(),
            prefix_samples=960,
            total_samples=1_280,
            activation_threshold=0.13,
            seed=7,
        )

        frontend_check = next(
            check
            for check in audit["checks"]
            if check["name"] == "causal_frontend_configuration"
        )
        self.assertEqual(audit["status"], "FAIL")
        self.assertFalse(frontend_check["passed"])
        self.assertFalse(frontend_check["causal"])


if __name__ == "__main__":
    unittest.main()
