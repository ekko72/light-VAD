# -*- coding: utf-8 -*-
"""Unit checks for the A10 temporal-value predictor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.analyze_a10_value_predictability import (
    _write_markdown,
    fit_value_score,
    paired_speaker_cluster_bootstrap,
    select_top_fraction,
    signed_temporal_value,
    temporal_value_map,
)


class A10ValuePredictabilityTests(unittest.TestCase):
    def test_signed_temporal_value(self) -> None:
        value = signed_temporal_value(
            labels=np.asarray([1, 1, 0, 0]),
            short_scores=np.asarray([0.6, 0.4, 0.6, 0.4]),
            refined_scores=np.asarray([0.6, 0.6, 0.4, 0.4]),
        )

        np.testing.assert_array_equal(value, [0, 1, 1, 0])

    def test_select_top_fraction_is_score_based(self) -> None:
        threshold, selected = select_top_fraction(
            np.asarray([0.1, 0.9, 0.5, 0.7]),
            activation_rate=0.5,
        )

        self.assertEqual(threshold, 0.7)
        np.testing.assert_array_equal(selected, [False, True, False, True])

    def test_logistic_value_score_orders_signed_value(self) -> None:
        rng = np.random.default_rng(20260919)
        positive = rng.normal(
            loc=[2.0, 0.0, 0.0, 0.0],
            scale=0.25,
            size=(300, 4),
        )
        zero = rng.normal(
            loc=[0.0, 0.0, 0.0, 0.0],
            scale=0.25,
            size=(300, 4),
        )
        negative = rng.normal(
            loc=[0.0, 0.0, 2.0, 0.0],
            scale=0.25,
            size=(300, 4),
        )
        features = np.concatenate([positive, zero, negative])
        values = np.concatenate(
            [
                np.ones(300, dtype=np.int8),
                np.zeros(300, dtype=np.int8),
                -np.ones(300, dtype=np.int8),
            ]
        )
        mask = np.ones(values.size, dtype=bool)

        score, metadata = fit_value_score(
            features,
            values,
            mask,
            random_seed=17,
        )

        self.assertGreater(score[:300].mean(), score[300:600].mean())
        self.assertGreater(score[300:600].mean(), score[600:].mean())
        self.assertEqual(metadata["classes"], [-1, 0, 1])

    def test_paired_bootstrap_reports_positive_difference(self) -> None:
        first = np.zeros((4, 7), dtype=np.int64)
        second = np.zeros((4, 7), dtype=np.int64)
        first[:, 0] = 100
        second[:, 0] = 100
        first[:, 1] = 10
        second[:, 1] = 10
        first[:, 2] = 6
        second[:, 2] = 2
        first[:, 3] = 1
        second[:, 3] = 4

        result = paired_speaker_cluster_bootstrap(
            first,
            second,
            repeats=100,
            seed=17,
        )

        self.assertAlmostEqual(
            result["difference_net_utility_per_selected"],
            0.7,
        )
        self.assertGreater(
            result["difference_net_utility_per_selected_ci95_low"],
            0.0,
        )

    def test_markdown_uses_gate_specific_bootstrap_intervals(self) -> None:
        paired = {
            "first_net_utility_per_selected_ci95_low": 0.01,
            "first_net_utility_per_selected_ci95_high": 0.02,
            "second_net_utility_per_selected_ci95_low": 0.03,
            "second_net_utility_per_selected_ci95_high": 0.04,
            "difference_net_utility_per_selected": -0.02,
            "difference_net_utility_per_selected_ci95_low": -0.03,
            "difference_net_utility_per_selected_ci95_high": -0.01,
        }
        gate = {
            "test_activation_rate": 0.05,
            "test_correction_rate_selected": 0.10,
            "test_harm_rate_selected": 0.02,
            "test_net_utility_per_selected": 0.08,
            "test_net_utility_per_frame": 0.004,
            "test_f1": 0.75,
            "cluster_bootstrap": paired,
        }
        random_gate = {
            "test_activation_rate": {"mean": 0.05},
            "test_correction_rate_selected": {"mean": 0.08},
            "test_harm_rate_selected": {"mean": 0.03},
            "test_net_utility_per_selected": {
                "mean": 0.05,
                "ci95_low": 0.04,
                "ci95_high": 0.06,
            },
            "test_net_utility_per_frame": {"mean": 0.0025},
            "test_f1": {"mean": 0.74},
        }
        summary = {
            "protocol": {
                "predictions": "predictions.npz",
                "adaptive_checkpoint": "adaptive.pt",
                "lookback_frames": 384,
                "lookback_seconds": 6.144,
                "window_frames": 25,
                "window_seconds": 0.4,
                "calibration_speaker_count": 2,
                "test_speaker_count": 2,
                "primary_activation": 0.05,
                "feature_names": ["entropy", "change", "variance", "switch"],
            },
            "signed_value": {
                split: {
                    "frames": 10,
                    "positive": 2,
                    "zero": 7,
                    "negative": 1,
                    "mean_value": 0.1,
                    "correction_rate": 0.2,
                    "harm_rate": 0.1,
                }
                for split in ("calibration", "test")
            },
            "primary": {
                "uncertainty": gate,
                "value": gate,
                "random": random_gate,
                "value_minus_uncertainty": paired,
            },
            "activation_budget_rows": [
                {
                    "calibration_activation_target": 0.05,
                    "uncertainty": gate,
                    "value": gate,
                    "random": random_gate,
                }
            ],
            "temporal_value_map": {
                "max_mean_value": 0.2,
                "max_mean_value_count": 3,
            },
            "go_assessment": {
                "status": "UNCERTAINTY_PROXY",
                "criteria": {
                    "value_net_utility_above_uncertainty": False,
                    "paired_cluster_ci_excludes_zero": False,
                },
                "interpretation": "Test interpretation.",
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            _write_markdown(summary, path)
            markdown = path.read_text(encoding="utf-8")

        self.assertIn("| value | 5.000% |", markdown)
        self.assertIn("[1.000%, 2.000%]", markdown)
        self.assertIn("| uncertainty | 5.000% |", markdown)
        self.assertIn("[3.000%, 4.000%]", markdown)

    def test_temporal_value_map_has_entropy_variance_shape(self) -> None:
        features = np.column_stack(
            [
                np.linspace(0.0, 1.0, 100),
                np.zeros(100),
                np.linspace(0.0, 1.0, 100),
                np.zeros(100),
            ]
        )
        values = np.zeros(100, dtype=np.int8)
        values[:20] = -1
        values[-20:] = 1
        mask = np.ones(100, dtype=bool)

        result = temporal_value_map(
            features,
            values,
            mask,
            bins=5,
        )

        self.assertEqual(len(result["count"]), 5)
        self.assertEqual(len(result["count"][0]), 5)
        self.assertGreater(result["max_mean_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
