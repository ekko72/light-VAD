# -*- coding: utf-8 -*-
"""Checks for memory-bounded evaluation metrics."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from .evaluate import StreamingMetrics
except ImportError:
    from evaluate import StreamingMetrics


class StreamingMetricsTests(unittest.TestCase):
    def test_metrics_match_sklearn_reference(self) -> None:
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, size=20_000, dtype=np.int64)
        labels[0] = 0
        labels[1] = 1
        scores = rng.random(labels.size)

        metrics = StreamingMetrics(threshold=0.5, bins=65_536)
        metrics.update(labels, scores)
        summary = metrics.metrics()

        fpr, tpr, _ = roc_curve(labels, scores)
        expected_tpr = float(np.interp(0.315, fpr, tpr))
        self.assertAlmostEqual(
            summary["auc"],
            float(roc_auc_score(labels, scores)),
            places=4,
        )
        self.assertAlmostEqual(
            summary["tpr_at_fpr_0.315"],
            expected_tpr,
            places=4,
        )
        self.assertAlmostEqual(
            summary["accuracy"],
            float(np.mean((scores >= 0.5) == labels)),
        )

    def test_accumulator_handles_multiple_updates(self) -> None:
        metrics = StreamingMetrics(threshold=0.5, bins=65_536)
        metrics.update(
            np.array([0, 1, 0], dtype=np.int64),
            np.array([0.1, 0.9, 0.8], dtype=np.float64),
        )
        metrics.update(
            np.array([1, 0], dtype=np.int64),
            np.array([0.7, 0.2], dtype=np.float64),
        )
        summary = metrics.metrics()

        self.assertEqual(summary["examples"], 5)
        self.assertEqual(summary["speech"], 2)
        self.assertEqual(summary["silence"], 3)
        self.assertAlmostEqual(summary["accuracy"], 0.8)
        self.assertAlmostEqual(
            summary["auc"],
            float(roc_auc_score([0, 1, 0, 1, 0], [0.1, 0.9, 0.8, 0.7, 0.2])),
        )


if __name__ == "__main__":
    unittest.main()
