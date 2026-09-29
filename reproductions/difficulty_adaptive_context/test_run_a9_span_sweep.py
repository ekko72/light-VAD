# -*- coding: utf-8 -*-
"""Unit checks for the A9 span-sweep runner."""

from __future__ import annotations

from argparse import Namespace
import tempfile
import unittest
from pathlib import Path

from reproductions.difficulty_adaptive_context.run_a9_span_sweep import (
    _assess_go,
    _evaluation_command,
    _parse_checkpoint_overrides,
    _write_markdown,
)


class A9SpanSweepRunnerTests(unittest.TestCase):
    def test_evaluation_command_uses_requested_seed(self) -> None:
        args = Namespace(
            test_manifest=Path("test.tsv"),
            long_checkpoint=Path("long.pt"),
            reference_predictions=Path("reference.npz"),
            data_root=Path("data"),
            librispeech_root=Path("LibriSpeech"),
            eval_row_sample=1080,
            device="cpu",
            score_chunk_frames=2000,
            fixed_threshold=0.13,
            bootstrap_repeats=2000,
            unseen_noise=["SSN_noise"],
            latency_warmup=20,
            latency_repeats=100,
            latency_selected_frames=64,
            skip_latency_benchmark=False,
        )

        command = _evaluation_command(
            args,
            seed=23,
            span=128,
            checkpoint=Path("adaptive.pt"),
            output_dir=Path("evaluation"),
        )

        seed_index = command.index("--seed")
        self.assertEqual(command[seed_index + 1], "23")

    def test_checkpoint_override_parser(self) -> None:
        self.assertEqual(
            _parse_checkpoint_overrides(
                [r"17:64:C:\checkpoints\rf64.pt"]
            ),
            {(17, 64): Path(r"C:\checkpoints\rf64.pt")},
        )
        with self.assertRaisesRegex(ValueError, "seed:span:path"):
            _parse_checkpoint_overrides(["17:64"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _parse_checkpoint_overrides(
                ["17:64:first.pt", "17:64:second.pt"]
            )

    def test_go_assessment_compares_rf64_and_rf384(self) -> None:
        rows = []
        for condition, low_utility, high_utility in (
            ("test", 0.02, 0.08),
            ("clean", 0.01, 0.03),
        ):
            for span, utility in ((64, low_utility), (384, high_utility)):
                rows.append(
                    {
                        "seed": 17,
                        "rf_span": span,
                        "condition": condition,
                        "adaptive_net_utility_per_selected": utility,
                    }
                )

        assessment = _assess_go(rows, [64, 384])

        self.assertTrue(assessment["all_seed_provisional_go"])
        self.assertAlmostEqual(
            assessment["seed_results"][0]["u384_minus_u64"],
            0.06,
        )

    def test_markdown_cost_accepts_missing_latency(self) -> None:
        row = {
            "seed": 17,
            "rf_span": 64,
            "condition": "test",
            "lookback_seconds": 0.64,
            "adaptive_f1": 0.8,
            "refine_only_f1": 0.81,
            "short_f1": 0.79,
            "long_f1": 0.82,
            "activation_rate": 0.05,
            "correction_per_selected": 0.04,
            "harm_per_selected": 0.02,
            "adaptive_net_utility_per_selected": 0.02,
            "adaptive_net_utility_per_frame": 0.001,
            "net_utility_ci95_low": 0.01,
            "net_utility_ci95_high": 0.03,
            "refinement_macs_per_selected_frame": 4096,
            "refinement_macs_per_call": 262144,
            "cache_bytes": 32768,
            "cpu_median_ms_per_call": None,
            "cpu_p95_ms_per_call": None,
        }
        assessment = {
            "seed_results": [],
            "all_seed_provisional_go": False,
            "note": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a9.md"
            _write_markdown([row], assessment, path)
            report = path.read_text(encoding="utf-8")

        self.assertIn("4,096", report)
        self.assertIn("262,144", report)
        self.assertIn("n/a", report)


if __name__ == "__main__":
    unittest.main()
