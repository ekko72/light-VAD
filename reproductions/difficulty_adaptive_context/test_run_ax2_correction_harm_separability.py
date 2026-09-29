# -*- coding: utf-8 -*-
"""Unit checks for the AX2 correction-versus-harm runner."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from reproductions.difficulty_adaptive_context.run_ax2_correction_harm_separability import (
    Ax2Config,
    assess_ax2,
    binary_classification_metrics,
    binary_targets,
    bootstrap_cluster_counts,
    bootstrap_curve_metrics,
    evaluate_ax2,
    fit_binary_logistic_probability,
    fit_binary_mlp_probability,
    load_config,
    paired_curve_difference,
    prepare_binary_curve,
    verify_hash,
)


def _row(
    target: str,
    *,
    auroc_low: float,
    lift_low: float,
) -> dict[str, object]:
    return {
        "target": target,
        "model": "X3_logistic",
        "auroc_ci95_low": auroc_low,
        "normalized_auprc_lift_ci95_low": lift_low,
    }


def _paired_direction(
    *,
    auroc_low: float,
    auroc_high: float,
    lift_low: float,
    lift_high: float,
) -> dict[str, object]:
    return {
        "auroc_difference_ci95_low": auroc_low,
        "auroc_difference_ci95_high": auroc_high,
        "normalized_auprc_lift_difference_ci95_low": lift_low,
        "normalized_auprc_lift_difference_ci95_high": lift_high,
    }


def _brute_force_bootstrap_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    auroc = np.full(cluster_counts.shape[0], np.nan, dtype=np.float64)
    auprc = np.full(cluster_counts.shape[0], np.nan, dtype=np.float64)
    normalized_lift = np.full(
        cluster_counts.shape[0],
        np.nan,
        dtype=np.float64,
    )
    cluster_indices = [
        np.flatnonzero(cluster_codes == cluster)
        for cluster in range(int(cluster_counts.shape[1]))
    ]
    for repeat, counts in enumerate(cluster_counts):
        expanded = np.concatenate(
            [
                np.repeat(cluster_indices[cluster], count)
                for cluster, count in enumerate(counts)
                if count > 0
            ],
        )
        sample_target = target[expanded]
        sample_probability = probabilities[expanded]
        if np.unique(sample_target).size != 2:
            continue
        prevalence = float(np.mean(sample_target))
        sample_auprc = float(
            average_precision_score(sample_target, sample_probability)
        )
        auroc[repeat] = float(
            roc_auc_score(sample_target, sample_probability)
        )
        auprc[repeat] = sample_auprc
        normalized_lift[repeat] = (
            sample_auprc - prevalence
        ) / (1.0 - prevalence)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "normalized_auprc_lift": normalized_lift,
    }


class Ax2CorrectionHarmSeparabilityTests(unittest.TestCase):
    def test_frozen_protocol_loads(self) -> None:
        config = load_config()

        self.assertEqual(config.payload["protocol_id"], "A-AX2-v1")
        self.assertEqual(
            config.payload["feature_ladder"]["primary_predictor"],
            "X3_logistic",
        )
        self.assertEqual(
            config.output_dir.name,
            "ax2_correction_harm_separability",
        )

    def test_verify_hash_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.bin"
            path.write_bytes(b"first")
            first = verify_hash(path, "", verify=False)
            self.assertEqual(first, "not_verified")
            with self.assertRaises(ValueError):
                verify_hash(path, "0" * 64, verify=True)

    def test_binary_targets_are_one_versus_rest(self) -> None:
        targets = binary_targets(np.asarray([-1, 0, 1, 1]))

        self.assertEqual(
            targets["correction"].tolist(),
            [False, False, True, True],
        )
        self.assertEqual(
            targets["harm"].tolist(),
            [True, False, False, False],
        )

    def test_binary_metrics_match_sklearn_and_are_normalized(self) -> None:
        target = np.asarray([False, False, True, True])
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])

        metrics = binary_classification_metrics(target, probabilities)
        prevalence = 0.5
        expected_auprc = float(
            average_precision_score(target, probabilities)
        )

        self.assertAlmostEqual(metrics["prevalence"], prevalence)
        self.assertAlmostEqual(
            metrics["auroc"],
            float(roc_auc_score(target, probabilities)),
        )
        self.assertAlmostEqual(metrics["auprc"], expected_auprc)
        self.assertAlmostEqual(
            metrics["normalized_auprc_lift"],
            (expected_auprc - prevalence) / (1.0 - prevalence),
        )

    def test_logistic_probabilities_are_finite_and_discriminative(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(240, 3))
        target = features[:, 0] > 0.0
        calibration = np.ones(240, dtype=bool)

        probabilities = fit_binary_logistic_probability(
            features,
            target,
            calibration,
            seed=17,
            max_iter=500,
            tolerance=1e-5,
        )

        self.assertEqual(probabilities.shape, (240,))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertGreater(
            probabilities[target].mean(),
            probabilities[~target].mean(),
        )

    def test_mlp_probabilities_run_deterministically_on_cpu(self) -> None:
        rng = np.random.default_rng(23)
        features = rng.normal(size=(180, 2)).astype(np.float32)
        target = features[:, 0] > 0.0
        calibration = np.ones(180, dtype=bool)
        arguments = {
            "seed": 23,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 32,
            "epochs": 3,
            "device": torch.device("cpu"),
        }

        first = fit_binary_mlp_probability(
            features,
            target,
            calibration,
            **arguments,
        )
        second = fit_binary_mlp_probability(
            features,
            target,
            calibration,
            **arguments,
        )

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.isfinite(first)))

    def test_grouped_bootstrap_matches_brute_force(self) -> None:
        target = np.asarray(
            [True, False, True, False, False, True, False, True],
        )
        probabilities = np.asarray(
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.4],
        )
        clusters = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        prepared = prepare_binary_curve(target, probabilities, clusters)
        counts = bootstrap_cluster_counts(
            int(clusters.max()) + 1,
            repeats=64,
            seed=20260919,
        )

        observed = bootstrap_curve_metrics(
            prepared,
            counts,
            group_chunk_size=2,
            weight_chunk_size=3,
        )
        expected = _brute_force_bootstrap_metrics(
            target,
            probabilities,
            clusters,
            counts,
        )

        for metric in ("auroc", "auprc", "normalized_auprc_lift"):
            np.testing.assert_allclose(
                observed[metric],
                expected[metric],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        point = prepared["point_metrics"]
        self.assertAlmostEqual(
            point["auroc"],
            float(roc_auc_score(target, probabilities)),
        )
        self.assertAlmostEqual(
            point["auprc"],
            float(average_precision_score(target, probabilities)),
        )

    def test_paired_difference_uses_same_resampled_clusters(self) -> None:
        target = np.asarray([True, False, True, False, True, False])
        first_probability = np.asarray([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
        second_probability = np.asarray([0.8, 0.3, 0.6, 0.2, 0.5, 0.4])
        clusters = np.asarray([0, 0, 1, 1, 2, 2])
        counts = bootstrap_cluster_counts(3, repeats=16, seed=17)

        first = prepare_binary_curve(target, first_probability, clusters)
        second = prepare_binary_curve(target, second_probability, clusters)
        paired = paired_curve_difference(first, second, counts)

        self.assertAlmostEqual(
            paired["auroc_difference"],
            paired["first_auroc"] - paired["second_auroc"],
        )
        self.assertAlmostEqual(
            paired["normalized_auprc_lift_difference"],
            paired["first_normalized_auprc_lift"]
            - paired["second_normalized_auprc_lift"],
        )
        self.assertLessEqual(
            paired["auroc_difference_ci95_low"],
            paired["auroc_difference_ci95_high"],
        )
        self.assertLessEqual(
            paired["normalized_auprc_lift_difference_ci95_low"],
            paired["normalized_auprc_lift_difference_ci95_high"],
        )

    def test_assessment_branches_are_frozen(self) -> None:
        cases = [
            (
                "BIDIRECTIONAL_SEPARABILITY",
                [_row("correction", auroc_low=0.6, lift_low=0.1),
                 _row("harm", auroc_low=0.6, lift_low=0.1)],
                _paired_direction(
                    auroc_low=-0.1,
                    auroc_high=0.1,
                    lift_low=-0.1,
                    lift_high=0.1,
                ),
            ),
            (
                "ASYMMETRIC_SEPARABILITY",
                [_row("correction", auroc_low=0.6, lift_low=0.1),
                 _row("harm", auroc_low=0.4, lift_low=-0.1)],
                _paired_direction(
                    auroc_low=0.05,
                    auroc_high=0.4,
                    lift_low=0.05,
                    lift_high=0.4,
                ),
            ),
            (
                "UNCERTAIN_ROBUST_ASYMMETRY",
                [_row("correction", auroc_low=0.6, lift_low=0.1),
                 _row("harm", auroc_low=0.4, lift_low=-0.1)],
                _paired_direction(
                    auroc_low=-0.05,
                    auroc_high=0.2,
                    lift_low=-0.05,
                    lift_high=0.2,
                ),
            ),
            (
                "NO_ROBUST_DIRECTION_SEPARABILITY",
                [_row("correction", auroc_low=0.4, lift_low=-0.1),
                 _row("harm", auroc_low=0.4, lift_low=-0.1)],
                _paired_direction(
                    auroc_low=-0.1,
                    auroc_high=0.1,
                    lift_low=-0.1,
                    lift_high=0.1,
                ),
            ),
        ]
        for expected_status, rows, direction in cases:
            with self.subTest(expected_status=expected_status):
                paired = {
                    "correction": {},
                    "harm": {},
                    "correction_vs_harm": direction,
                }
                assessment = assess_ax2(rows, paired)
                self.assertEqual(assessment["status"], expected_status)

    def test_evaluate_ax2_tiny_synthetic_context(self) -> None:
        base = load_config()
        payload = copy.deepcopy(base.payload)
        payload["bootstrap"]["repeats"] = 32
        config = Ax2Config(base.protocol_path, payload)
        n_frames = 8
        features = {
            level: np.linspace(
                0.0,
                1.0,
                n_frames * 2,
                dtype=np.float32,
            ).reshape(n_frames, 2)
            for level in ("X0", "X1", "X2", "X3")
        }
        targets = {
            "correction": np.asarray(
                [True, False, False, False, False, False, True, False],
            ),
            "harm": np.asarray(
                [False, False, True, False, False, False, False, True],
            ),
        }
        calibration_mask = np.asarray(
            [True, True, True, True, False, False, False, False],
        )
        test_mask = ~calibration_mask
        context = {
            "features": features,
            "targets": targets,
            "calibration_mask": calibration_mask,
            "test_mask": test_mask,
            "cluster_names": np.asarray(["s0", "s1", "s2", "s3"]),
            "cluster_codes": np.asarray([0, 1, 2, 3]),
            "calibration_speakers": ["s0", "s1", "s2", "s3"],
            "test_speakers": ["s0", "s1", "s2", "s3"],
        }

        def fake_logistic(
            features_value: np.ndarray,
            target: np.ndarray,
            calibration: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            del features_value, target, calibration, kwargs
            return np.linspace(0.1, 0.9, n_frames)

        def fake_mlp(
            features_value: np.ndarray,
            target: np.ndarray,
            calibration: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            del features_value, target, calibration, kwargs
            return np.linspace(0.1, 0.9, n_frames)

        with patch(
            "reproductions.difficulty_adaptive_context."
            "run_ax2_correction_harm_separability."
            "fit_binary_logistic_probability",
            side_effect=fake_logistic,
        ), patch(
            "reproductions.difficulty_adaptive_context."
            "run_ax2_correction_harm_separability."
            "fit_binary_mlp_probability",
            side_effect=fake_mlp,
        ):
            summary = evaluate_ax2(
                config,
                context=context,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(summary["rows"]), 16)
        self.assertEqual(summary["split"]["test_frames"], 4)
        self.assertIn(summary["assessment"]["status"], {
            "BIDIRECTIONAL_SEPARABILITY",
            "ASYMMETRIC_SEPARABILITY",
            "UNCERTAIN_ROBUST_ASYMMETRY",
            "NO_ROBUST_DIRECTION_SEPARABILITY",
        })
        for target_name in ("correction", "harm"):
            primary = summary["bootstrap"]["primary"][target_name]
            self.assertIn("auroc_ci95_low", primary)
            self.assertIn("normalized_auprc_lift_ci95_low", primary)


if __name__ == "__main__":
    unittest.main()
