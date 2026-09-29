# -*- coding: utf-8 -*-
"""Unit checks for the AX1 feature and metric implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.run_ax1_value_predictability import (
    _fixed_selection,
    _selected_metrics,
    cluster_bootstrap_metrics,
    encode_short_stream,
    feature_blocks,
    fit_logistic_score,
    fit_mlp_score,
    load_config,
    mfcc_context_features,
    posterior_temporal_features,
    uncertainty_score,
)


class Ax1ValuePredictabilityTests(unittest.TestCase):
    def test_frozen_protocol_loads(self) -> None:
        config = load_config()

        self.assertEqual(config.payload["protocol_id"], "A-AX1-v1")
        self.assertEqual(
            config.output_dir.name,
            "ax1_value_predictability",
        )

    def test_posterior_features_are_causal_and_reset_per_segment(self) -> None:
        scores = np.asarray([0.2, 0.4, 0.8, 0.5, 0.3], dtype=np.float64)
        segments = [(0, 3), (3, 5)]
        features = posterior_temporal_features(
            scores,
            segments,
            window_frames=3,
        )

        self.assertEqual(features.shape, (5, 10))
        self.assertAlmostEqual(features[0, 2], scores[0])
        self.assertAlmostEqual(features[0, 4], 0.0)
        self.assertAlmostEqual(features[3, 4], 0.0)
        self.assertAlmostEqual(features[4, 4], -0.2)

    def test_mfcc_context_has_current_mean_and_delta_blocks(self) -> None:
        mfcc = np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 8.0],
            ],
            dtype=np.float64,
        )
        features = mfcc_context_features(
            mfcc,
            [(0, 3)],
            window_frames=2,
        )

        np.testing.assert_allclose(features[:, :2], mfcc)
        np.testing.assert_allclose(features[1, 2:4], [2.0, 3.0])
        np.testing.assert_allclose(features[2, 4:6], [2.0, 4.0])

    def test_feature_blocks_have_nested_shapes(self) -> None:
        scores = np.linspace(0.1, 0.9, 12)
        hidden = np.arange(12 * 128, dtype=np.float32).reshape(12, 128)
        mfcc = np.arange(12 * 64, dtype=np.float32).reshape(12, 64)
        blocks = feature_blocks(
            scores,
            hidden,
            mfcc,
            [(0, 12)],
            window_frames=5,
        )

        self.assertEqual(blocks["uncertainty"].shape, (12, 1))
        self.assertEqual(blocks["X0"].shape, (12, 2))
        self.assertEqual(blocks["X1"].shape, (12, 10))
        self.assertEqual(blocks["X2"].shape, (12, 138))
        self.assertEqual(blocks["X3"].shape, (12, 330))

    def test_short_stream_matches_direct_encoder(self) -> None:
        torch.manual_seed(17)
        model = torch.nn.Sequential()
        model.short_model = torch.nn.Module()
        model.short_model.init_stream_state = lambda features: None

        def encode_stream(
            features: torch.Tensor,
            states: object,
        ) -> tuple[torch.Tensor, object]:
            del states
            return features, None

        model.short_model.encode_stream = encode_stream
        model.short_model.classifier = torch.nn.Conv1d(
            2,
            2,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            model.short_model.classifier.weight.zero_()
            model.short_model.classifier.weight[1, 0, 0] = 1.0
        features = torch.zeros(1, 2, 5)
        features[:, 0, :] = torch.arange(5)

        encoded, probabilities = encode_short_stream(
            model,
            features,
            chunk_frames=2,
        )

        torch.testing.assert_close(encoded, features)
        self.assertEqual(tuple(probabilities.shape), (1, 5))

    def test_selected_metrics_and_oracle_recovery(self) -> None:
        values = np.asarray([1, 1, -1, 0], dtype=np.int64)
        selected = np.asarray([True, True, False, False])

        metrics = _selected_metrics(values, selected)

        self.assertEqual(metrics["selected"], 2)
        self.assertEqual(metrics["selected_mean_value"], 1.0)
        self.assertEqual(metrics["oracle_value_recovery"], 1.0)
        self.assertEqual(metrics["correction_selected"], 2)

    def test_uncertainty_score_ranks_low_confidence_first(self) -> None:
        scores = np.asarray([0.49, 0.9, 0.5])
        uncertainty = uncertainty_score(scores)

        self.assertEqual(int(np.argmax(uncertainty)), 2)
        self.assertEqual(int(np.argmin(uncertainty)), 1)

    def test_fixed_selection_uses_calibration_threshold(self) -> None:
        scores = np.asarray([0.1, 0.9, 0.2, 0.95])
        calibration = np.asarray([True, True, False, False])
        test = ~calibration

        threshold, selected = _fixed_selection(
            scores,
            calibration,
            test,
            budget=0.5,
        )

        self.assertEqual(threshold, 0.9)
        self.assertEqual(selected.tolist(), [False, False, False, True])

    def test_fixed_selection_accepts_two_dimensional_uncertainty(self) -> None:
        scores = uncertainty_score(np.asarray([0.4, 0.9, 0.6, 0.5]))
        calibration = np.asarray([True, True, False, False])
        test = ~calibration

        threshold, selected = _fixed_selection(
            scores,
            calibration,
            test,
            budget=0.5,
        )

        self.assertEqual(scores.shape, (4, 1))
        self.assertAlmostEqual(threshold, -0.1)
        self.assertEqual(selected.tolist(), [False, False, True, True])

    def test_cluster_bootstrap_returns_intervals(self) -> None:
        values = np.asarray([1, 0, -1, 1, 0, -1] * 4, dtype=np.int64)
        selected = np.asarray([True, False, False] * 8)
        clusters = np.repeat(np.arange(4), 6)

        result = cluster_bootstrap_metrics(
            values,
            selected,
            clusters,
            repeats=200,
            seed=17,
        )

        self.assertEqual(result["selected"], 8)
        self.assertGreater(result["selected_mean_value_ci95_low"], 0.0)

    def test_logistic_score_prefers_positive_value_region(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(300, 3))
        values = np.zeros(300, dtype=np.int64)
        values[features[:, 0] > 1.0] = 1
        values[features[:, 0] < -1.0] = -1
        mask = np.ones(300, dtype=bool)

        scores = fit_logistic_score(
            features,
            values,
            mask,
            seed=17,
            max_iter=500,
            tolerance=1e-5,
        )

        self.assertGreater(scores[features[:, 0] > 1.0].mean(), 0.0)
        self.assertLess(scores[features[:, 0] < -1.0].mean(), 0.0)

    def test_mlp_score_runs_on_cpu(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(180, 2)).astype(np.float32)
        values = np.zeros(180, dtype=np.int64)
        values[features[:, 0] > 1.0] = 1
        values[features[:, 0] < -1.0] = -1
        mask = np.ones(180, dtype=bool)

        scores = fit_mlp_score(
            features,
            values,
            mask,
            seed=17,
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=32,
            epochs=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(scores.shape, (180,))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
