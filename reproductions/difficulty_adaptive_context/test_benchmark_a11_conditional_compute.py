# -*- coding: utf-8 -*-
"""Unit checks for the A11 conditional-compute benchmark."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reproductions.difficulty_adaptive_context.benchmark_a11_conditional_compute import (
    _count_analytical_macs_for_model,
    _merge_accuracy_and_latency,
    _write_csv,
    _write_markdown,
    assess_a11_go,
    binary_detection_metrics,
    build_accuracy_rows,
    collect_environment,
    latency_summary,
)


class _TinyShortModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv1d(
            64,
            4,
            kernel_size=3,
            padding=1,
            groups=2,
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        logits = self.classifier(encoded.transpose(1, 2))
        return logits.transpose(1, 2)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.short_model = _TinyShortModel()


def _accuracy_row(
    name: str,
    *,
    kind: str,
    activation: float,
    f1: float,
) -> dict[str, object]:
    return {
        "name": name,
        "label": name,
        "kind": kind,
        "activation_rate": activation,
        "f1": f1,
        "far": 0.1,
        "miss_rate": 0.2,
        "threshold": 0.1 if kind == "adaptive" else None,
    }


def _latency_result(
    *,
    mean_ms: float,
    activation: float,
    rtf: float,
    cache_bytes: int,
    kernel_ms: float | None = None,
) -> dict[str, object]:
    return {
        "latency": {
            "mean_ms_per_frame": mean_ms,
            "median_ms_per_frame": mean_ms * 0.9,
            "p95_ms_per_frame": mean_ms * 1.5,
            "p99_ms_per_frame": mean_ms * 1.8,
            "rtf": rtf,
        },
        "activation_rate": activation,
        "frames": 100,
        "utterances": 2,
        "audio_seconds": 10.0,
        "short_macs_per_frame": 100,
        "refinement_macs_per_selected_frame": 200,
        "cache_bytes": cache_bytes,
        "peak_rss_bytes": 1_000_000,
        "peak_rss_delta_bytes": 500_000,
        "refinement_kernel": (
            None
            if kernel_ms is None
            else {
                "mean_ms_per_frame": kernel_ms,
                "median_ms_per_frame": kernel_ms,
                "p95_ms_per_frame": kernel_ms,
                "p99_ms_per_frame": kernel_ms,
            }
        ),
    }


class A11ConditionalComputeTests(unittest.TestCase):
    def test_thresholds_use_calibration_only(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        calibration_mask = np.asarray(
            [True, True, True, True, False, False, False, False]
        )
        test_mask = ~calibration_mask
        rf_scores = np.asarray(
            [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
        )
        first_short = np.asarray(
            [0.50, 0.51, 0.30, 0.40, 0.50, 0.52, 0.30, 0.40]
        )
        second_short = np.asarray(
            [0.50, 0.51, 0.30, 0.40, 0.53, 0.30, 0.40, 0.45]
        )
        base = {
            "labels": labels,
            "rf64_full_scores": rf_scores,
            "rf384_full_scores": rf_scores,
            "calibration_mask": calibration_mask,
            "test_mask": test_mask,
        }

        first_rows = build_accuracy_rows(
            {**base, "short_scores": first_short},
            activation_budgets=[0.5],
        )
        second_rows = build_accuracy_rows(
            {**base, "short_scores": second_short},
            activation_budgets=[0.5],
        )

        first_adaptive = next(
            row for row in first_rows if row["kind"] == "adaptive"
        )
        second_adaptive = next(
            row for row in second_rows if row["kind"] == "adaptive"
        )
        self.assertAlmostEqual(first_adaptive["threshold"], 0.01)
        self.assertEqual(
            first_adaptive["threshold"],
            second_adaptive["threshold"],
        )
        self.assertEqual(
            first_adaptive["calibration_activation_rate"],
            second_adaptive["calibration_activation_rate"],
        )
        self.assertEqual(first_adaptive["activation_rate"], 0.25)
        self.assertEqual(second_adaptive["activation_rate"], 0.0)

    def test_binary_detection_metrics(self) -> None:
        metrics = binary_detection_metrics(
            labels=np.asarray([1, 1, 1, 0, 0, 0]),
            scores=np.asarray([0.9, 0.4, 0.8, 0.6, 0.2, 0.1]),
        )

        self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["far"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["miss_rate"], 1.0 / 3.0)

    def test_latency_summary_reports_percentiles(self) -> None:
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["frames"], 4)
        self.assertAlmostEqual(summary["mean_ms_per_frame"], 2.5)
        self.assertAlmostEqual(summary["median_ms_per_frame"], 2.5)
        self.assertAlmostEqual(summary["p95_ms_per_frame"], 3.85)
        self.assertAlmostEqual(summary["p99_ms_per_frame"], 3.97)

    def test_environment_collection_handles_system_text(self) -> None:
        environment = collect_environment()

        self.assertIn("platform", environment)
        self.assertIn("power_scheme", environment)
        self.assertIn("cpu_affinity", environment)
        self.assertIn("torch_num_threads", environment)

    def test_macs_count_grouped_conv_once(self) -> None:
        macs = _count_analytical_macs_for_model(_TinyModel())

        # Conv: 4 output channels * (64 / 2 input channels/group) * 3 taps.
        # Linear: 2 outputs * 4 inputs.
        self.assertEqual(macs, 4 * 32 * 3 + 2 * 4)

    def test_merge_tracks_macs_cache_router_and_scheduling_overhead(self) -> None:
        accuracy_rows = [
            _accuracy_row(
                "short",
                kind="short",
                activation=0.0,
                f1=0.70,
            ),
            _accuracy_row(
                "fixed_rf64",
                kind="fixed",
                activation=1.0,
                f1=0.80,
            ),
            _accuracy_row(
                "fixed_rf384",
                kind="fixed",
                activation=1.0,
                f1=0.81,
            ),
            _accuracy_row(
                "adaptive_5pct",
                kind="adaptive",
                activation=0.05,
                f1=0.75,
            ),
        ]
        latency_results = {
            "short_core": _latency_result(
                mean_ms=1.00,
                activation=0.0,
                rtf=0.01,
                cache_bytes=1_000,
            ),
            "short": _latency_result(
                mean_ms=1.20,
                activation=0.0,
                rtf=0.012,
                cache_bytes=1_100,
            ),
            "fixed_rf64": _latency_result(
                mean_ms=2.50,
                activation=1.0,
                rtf=0.025,
                cache_bytes=2_000,
                kernel_ms=1.30,
            ),
            "fixed_rf384": _latency_result(
                mean_ms=3.00,
                activation=1.0,
                rtf=0.030,
                cache_bytes=3_000,
                kernel_ms=1.80,
            ),
            "adaptive_5pct": _latency_result(
                mean_ms=2.00,
                activation=0.04,
                rtf=0.020,
                cache_bytes=2_500,
                kernel_ms=4.00,
            ),
        }

        rows = _merge_accuracy_and_latency(
            accuracy_rows,
            latency_results,
        )
        indexed = {row["name"]: row for row in rows}
        adaptive = indexed["adaptive_5pct"]

        self.assertEqual(adaptive["analytical_macs_per_frame"], 110)
        self.assertEqual(
            adaptive["analytical_macs_per_frame_test_activation"],
            110,
        )
        self.assertEqual(
            adaptive["analytical_macs_per_frame_benchmark_activation"],
            108,
        )
        self.assertEqual(adaptive["cache_bytes"], 2_500)
        self.assertAlmostEqual(adaptive["rtf"], 0.020)
        self.assertAlmostEqual(
            adaptive["router_overhead_ms_per_frame"],
            0.20,
        )
        self.assertAlmostEqual(
            adaptive["scheduling_overhead_ms_per_frame"],
            0.64,
        )
        self.assertAlmostEqual(
            indexed["short_core"]["gate_overhead_mean_ms_per_frame"],
            0.20,
        )

    def test_go_assessment_strong_go(self) -> None:
        assessment = assess_a11_go(
            [
                _accuracy_row(
                    "short",
                    kind="short",
                    activation=0.0,
                    f1=0.70,
                )
                | {"mean_ms_per_frame": 1.0},
                _accuracy_row(
                    "adaptive_5pct",
                    kind="adaptive",
                    activation=0.05,
                    f1=0.75,
                )
                | {"mean_ms_per_frame": 2.0},
                _accuracy_row(
                    "fixed_rf384",
                    kind="fixed",
                    activation=1.0,
                    f1=0.80,
                )
                | {"mean_ms_per_frame": 3.0},
            ]
        )

        self.assertEqual(assessment["status"], "STRONG_GO")
        self.assertTrue(
            assessment["f1_order_short_lt_adaptive_lt_refine"]
        )
        self.assertTrue(
            assessment["latency_order_short_lt_adaptive_lt_refine"]
        )

    def test_go_assessment_efficiency_failure(self) -> None:
        assessment = assess_a11_go(
            [
                _accuracy_row(
                    "short",
                    kind="short",
                    activation=0.0,
                    f1=0.70,
                )
                | {"mean_ms_per_frame": 1.0},
                _accuracy_row(
                    "adaptive_5pct",
                    kind="adaptive",
                    activation=0.05,
                    f1=0.75,
                )
                | {"mean_ms_per_frame": 2.95},
                _accuracy_row(
                    "fixed_rf384",
                    kind="fixed",
                    activation=1.0,
                    f1=0.80,
                )
                | {"mean_ms_per_frame": 3.0},
            ]
        )

        self.assertEqual(assessment["status"], "NO_GO_EFFICIENCY_OVERHEAD")
        self.assertTrue(assessment["adaptive_is_near_always_refine"])

    def test_go_assessment_conditional_middle_state(self) -> None:
        assessment = assess_a11_go(
            [
                _accuracy_row(
                    "short",
                    kind="short",
                    activation=0.0,
                    f1=0.70,
                )
                | {"mean_ms_per_frame": 1.0},
                _accuracy_row(
                    "adaptive_5pct",
                    kind="adaptive",
                    activation=0.05,
                    f1=0.80,
                )
                | {"mean_ms_per_frame": 2.0},
                _accuracy_row(
                    "fixed_rf384",
                    kind="fixed",
                    activation=1.0,
                    f1=0.80,
                )
                | {"mean_ms_per_frame": 3.0},
            ]
        )

        self.assertEqual(assessment["status"], "CONDITIONAL_COMPUTE_GO")
        self.assertFalse(
            assessment["f1_order_short_lt_adaptive_lt_refine"]
        )

    def test_csv_and_markdown_reporting(self) -> None:
        rows = [
            {
                "name": "short_core",
                "label": "Short encoder + classifier",
                "kind": "overhead",
                "test_activation_rate": 0.0,
                "benchmark_activation_rate": 0.0,
                "threshold": None,
                "f1": None,
                "far": None,
                "miss_rate": None,
                "mean_ms_per_frame": 1.0,
                "median_ms_per_frame": 0.9,
                "p95_ms_per_frame": 1.5,
                "p99_ms_per_frame": 1.8,
                "rtf": 0.01,
                "analytical_macs_per_frame": 100,
                "analytical_macs_per_frame_test_activation": 100,
                "analytical_macs_per_frame_benchmark_activation": 100,
                "cache_bytes": 1000,
                "peak_rss_bytes": 1_000_000,
                "gate_overhead_mean_ms_per_frame": 0.2,
                "refinement_kernel": None,
                "scheduling_overhead_ms_per_frame": None,
            },
            {
                "name": "short",
                "label": "Short",
                "kind": "short",
                "test_activation_rate": 0.0,
                "benchmark_activation_rate": 0.0,
                "threshold": None,
                "f1": 0.70,
                "far": 0.10,
                "miss_rate": 0.20,
                "mean_ms_per_frame": 1.2,
                "median_ms_per_frame": 1.1,
                "p95_ms_per_frame": 1.7,
                "p99_ms_per_frame": 1.9,
                "rtf": 0.012,
                "analytical_macs_per_frame": 100,
                "analytical_macs_per_frame_test_activation": 100,
                "analytical_macs_per_frame_benchmark_activation": 100,
                "cache_bytes": 1100,
                "peak_rss_bytes": 1_000_000,
            },
            {
                "name": "adaptive_5pct",
                "label": "Adaptive 5%",
                "kind": "adaptive",
                "test_activation_rate": 0.05,
                "benchmark_activation_rate": 0.04,
                "threshold": 0.1,
                "f1": 0.75,
                "far": 0.08,
                "miss_rate": 0.15,
                "mean_ms_per_frame": 2.0,
                "median_ms_per_frame": 1.9,
                "p95_ms_per_frame": 2.5,
                "p99_ms_per_frame": 2.8,
                "rtf": 0.020,
                "analytical_macs_per_frame": 110,
                "analytical_macs_per_frame_test_activation": 110,
                "analytical_macs_per_frame_benchmark_activation": 108,
                "cache_bytes": 2500,
                "peak_rss_bytes": 1_200_000,
                "refinement_kernel": {
                    "mean_ms_per_frame": 4.0,
                },
                "scheduling_overhead_ms_per_frame": 0.64,
            },
        ]
        summary = {
            "protocol": {
                "manifest": "manifest.tsv",
                "rf384_checkpoint": "rf384.pt",
                "rf64_checkpoint": "rf64.pt",
                "rf384_predictions": "rf384.npz",
                "rf64_predictions": "rf64.npz",
                "utterance_limit": 2,
                "repeats": 1,
                "warmup_frames": 10,
                "chunk_frames": 1,
                "selected_cpu": 0,
            },
            "environment": {
                "processor": "Test CPU",
                "torch_num_threads": 1,
                "power_scheme": "Balanced",
            },
            "rows": rows,
            "go_assessment": {
                "f1_order_short_lt_adaptive_lt_refine": True,
                "latency_order_short_lt_adaptive_lt_refine": True,
                "near_equivalent_relative_gap": 0.05,
                "adaptive_is_near_always_refine": False,
                "status": "STRONG_GO",
                "interpretation": "Synthetic report check.",
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "a11.csv"
            markdown_path = Path(directory) / "a11.md"
            _write_csv(rows[1:], csv_path)
            _write_markdown(summary, markdown_path)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[1]["name"], "adaptive_5pct")
        self.assertIn("| Adaptive 5% | 5.000% | 4.000% |", markdown)
        self.assertIn("Status: **STRONG_GO**", markdown)
        self.assertIn("Activation percentage is not called compute percentage", markdown)


if __name__ == "__main__":
    unittest.main()
