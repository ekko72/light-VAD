# -*- coding: utf-8 -*-
"""Synthetic unit checks for the frozen CAR runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from reproductions.cross_architecture_replication.run_car import (
    MODEL_SEEDS,
    _aggregate_indicator,
    _aggregate_values,
    _bootstrap_cluster_weights,
    _bootstrap_mean_metrics,
    _bootstrap_ratio_metrics,
    _cluster_codes,
    _cluster_sums,
    _point_ratio,
    _refuse_overwrite,
    _score_full_history,
    _score_window_batch,
    _taxonomy_from_scores,
    encode_stream_chunks,
    score_horizon,
)
from reproductions.prior_factored_vad.models import TinyGRUConfig, TinyGRUVAD


def _synthetic_model() -> TinyGRUVAD:
    torch.manual_seed(20260920)
    model = TinyGRUVAD(
        TinyGRUConfig(
            input_size=4,
            hidden_size=6,
            num_layers=2,
            num_classes=2,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def _synthetic_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.randn(1, 4, 700, generator=generator)


class FiniteHistoryTests(unittest.TestCase):
    def test_finite_window_scores_only_the_explicit_target_window(self) -> None:
        model = _synthetic_model()
        features = _synthetic_features()
        with torch.inference_mode():
            for horizon in (64, 128, 256, 384):
                for target in (20, 100, 400):
                    observed = _score_window_batch(
                        model,
                        features,
                        [target],
                        horizon=horizon,
                    )[0]
                    start = max(0, target - horizon + 1)
                    sequence = features[
                        0, :, start : target + 1
                    ].transpose(0, 1)
                    logits = model(
                        sequence.transpose(0, 1).unsqueeze(0)
                    )[:, :, -1]
                    expected = float(
                        torch.softmax(logits, dim=-1)[0, 1]
                    )
                    self.assertAlmostEqual(
                        float(observed),
                        expected,
                        places=7,
                    )

    def test_future_frames_do_not_change_a_target_score(self) -> None:
        model = _synthetic_model()
        features = _synthetic_features()
        target = 200
        with torch.inference_mode():
            for horizon in (64, 128, 256, 384):
                observed = _score_window_batch(
                    model,
                    features,
                    [target],
                    horizon=horizon,
                )[0]
                future = features.clone()
                future[:, :, target + 1 :] += 100.0
                perturbed = _score_window_batch(
                    model,
                    future,
                    [target],
                    horizon=horizon,
                )[0]
                self.assertEqual(float(observed), float(perturbed))

            full_observed = _score_full_history(
                model,
                features,
                [target],
            )[0]
            full_future = features.clone()
            full_future[:, :, target + 1 :] += 100.0
            full_perturbed = _score_full_history(
                model,
                full_future,
                [target],
            )[0]
            self.assertEqual(float(full_observed), float(full_perturbed))

    def test_reset_isolates_frames_before_the_window_start(self) -> None:
        model = _synthetic_model()
        features = _synthetic_features()
        target = 300
        with torch.inference_mode():
            for horizon in (64, 128, 256, 384):
                start = max(0, target - horizon + 1)
                observed = _score_window_batch(
                    model,
                    features,
                    [target],
                    horizon=horizon,
                )[0]
                reset = features.clone()
                reset[:, :, :start] += 100.0
                perturbed = _score_window_batch(
                    model,
                    reset,
                    [target],
                    horizon=horizon,
                )[0]
                self.assertEqual(float(observed), float(perturbed))

    def test_chunked_stream_matches_carried_hidden_state(self) -> None:
        model = _synthetic_model()
        features = _synthetic_features()
        with torch.inference_mode():
            full = model.forward_features(features)
            chunked = encode_stream_chunks(
                model,
                features,
                chunk_size=97,
            )
        self.assertTrue(
            torch.allclose(
                full,
                chunked,
                rtol=1e-6,
                atol=1e-7,
            )
        )

    def test_batched_targets_are_isolated(self) -> None:
        model = _synthetic_model()
        features = _synthetic_features()
        targets = [20, 150, 251, 377]
        with torch.inference_mode():
            separate = np.asarray(
                [
                    _score_window_batch(
                        model,
                        features,
                        [target],
                        horizon=128,
                    )[0]
                    for target in targets
                ]
            )
            for batch_size in (2, 4):
                batched = score_horizon(
                    model,
                    features,
                    targets,
                    horizon=128,
                    batch_size=batch_size,
                )
                np.testing.assert_array_equal(separate, batched)


class TaxonomyAndAggregationTests(unittest.TestCase):
    def test_taxonomy_matches_the_frozen_four_class_definitions(self) -> None:
        labels = np.ones(4, dtype=np.int64)
        short_scores = np.asarray([0.9, 0.1, 0.1, 0.9])
        long_scores = np.asarray([0.8, 0.8, 0.2, 0.2])
        taxonomy = _taxonomy_from_scores(
            labels,
            short_scores,
            long_scores,
        )
        expected = {
            "SS": np.asarray([True, False, False, False]),
            "R": np.asarray([False, True, False, False]),
            "I": np.asarray([False, False, True, False]),
            "H": np.asarray([False, False, False, True]),
        }
        for name, mask in expected.items():
            np.testing.assert_array_equal(taxonomy[name], mask)
        np.testing.assert_allclose(
            taxonomy["value"],
            taxonomy["short_loss"] - taxonomy["long_loss"],
        )

    def test_five_seed_aggregate_is_equal_weighted(self) -> None:
        values = {
            seed: np.full(3, float(index))
            for index, seed in enumerate(MODEL_SEEDS)
        }
        indicators = {
            seed: {"indicator": np.full(3, float(index + 1))}
            for index, seed in enumerate(MODEL_SEEDS)
        }
        np.testing.assert_allclose(
            _aggregate_values(values),
            np.full(3, 2.0),
        )
        np.testing.assert_allclose(
            _aggregate_indicator(indicators, "indicator"),
            np.full(3, 3.0),
        )


class ClusterBootstrapTests(unittest.TestCase):
    def test_cluster_codes_ignore_values_outside_the_valid_mask(self) -> None:
        values = np.asarray(
            ["calibration-a", "test-a", "calibration-b", "test-b"]
        )
        valid = np.asarray([False, True, False, True])
        codes = _cluster_codes(
            values,
            ("test-a", "test-b"),
            valid_mask=valid,
        )
        np.testing.assert_array_equal(codes, [-1, 0, -1, 1])
        with self.assertRaises(ValueError):
            _cluster_codes(values, ("test-a", "test-b"))

    def test_cluster_sums_reject_invalid_codes_in_selected_frames(self) -> None:
        with self.assertRaises(ValueError):
            _cluster_sums(
                np.asarray([1.0, 2.0]),
                np.asarray([True, False]),
                np.asarray([-1, 0]),
                1,
            )
        with self.assertRaises(ValueError):
            _cluster_sums(
                np.asarray([1.0, 2.0]),
                np.asarray([False, True]),
                np.asarray([0, 1]),
                1,
            )

    def test_source_cluster_weights_are_deterministic(self) -> None:
        first = _bootstrap_cluster_weights(
            6,
            repeats=11,
            seed=20260920,
        )
        second = _bootstrap_cluster_weights(
            6,
            repeats=11,
            seed=20260920,
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first.sum(axis=0),
            np.full(11, 6),
        )

    def test_source_cluster_mean_bootstrap_uses_cluster_codes(self) -> None:
        codes = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        data = SimpleNamespace(
            source_codes=codes,
            speaker_codes=codes,
            reference=SimpleNamespace(
                source_order=("a", "b", "c"),
                speaker_order=("s0", "s1", "s2"),
            ),
        )
        values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = _bootstrap_mean_metrics(
            data=data,
            group="synthetic",
            mask=np.ones(values.size, dtype=bool),
            metrics={"mean": values},
            unit="source_cluster",
        )["mean"]
        self.assertAlmostEqual(float(result["estimate"]), 2.5)
        self.assertIsNotNone(result["ci95_low"])
        self.assertIsNotNone(result["ci95_high"])
        self.assertLessEqual(result["ci95_low"], result["ci95_high"])

    def test_ratio_bootstrap_handles_denominators_and_subsets(self) -> None:
        codes = np.asarray([0, 0, 1, 1], dtype=np.int64)
        data = SimpleNamespace(
            source_codes=codes,
            speaker_codes=codes,
            reference=SimpleNamespace(
                source_order=("a", "b"),
                speaker_order=("s0", "s1"),
            ),
        )
        numerator = np.asarray([True, False, True, False])
        denominator = np.ones(4, dtype=bool)
        result = _bootstrap_ratio_metrics(
            data=data,
            group="synthetic",
            numerators={"rate": numerator},
            denominators={"rate": denominator},
            unit="source_cluster",
        )["rate"]
        self.assertAlmostEqual(float(result["estimate"]), 0.5)
        self.assertIsNotNone(result["ci95_low"])
        self.assertIsNotNone(result["ci95_high"])

        empty = np.zeros(4, dtype=bool)
        undefined = _bootstrap_ratio_metrics(
            data=data,
            group="synthetic-empty",
            numerators={"rate": empty},
            denominators={"rate": empty},
            unit="source_cluster",
        )["rate"]
        self.assertIsNone(undefined["estimate"])
        self.assertIsNone(undefined["ci95_low"])
        self.assertIsNone(undefined["ci95_high"])
        self.assertIsNone(_point_ratio(1.0, 0.0))

        with self.assertRaises(ValueError):
            _bootstrap_ratio_metrics(
                data=data,
                group="synthetic-invalid",
                numerators={"rate": np.asarray([True, True, False, False])},
                denominators={"rate": np.asarray([False, True, True, True])},
                unit="source_cluster",
            )
        with self.assertRaises(ValueError):
            _bootstrap_ratio_metrics(
                data=data,
                group="synthetic-missing",
                numerators={"rate": numerator},
                denominators={},
                unit="source_cluster",
            )


class OverwriteGuardTests(unittest.TestCase):
    def test_overwrite_guard_rejects_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _refuse_overwrite(
                [output_dir / "missing.csv"],
                stage="synthetic",
            )
            existing = output_dir / "existing.csv"
            existing.write_text("value\n1\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _refuse_overwrite([existing], stage="synthetic")


if __name__ == "__main__":
    unittest.main()
