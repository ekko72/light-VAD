# -*- coding: utf-8 -*-
"""Unit checks for the frozen A-v2 method study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reproductions.difficulty_adaptive_context import run_a2_method_study as a2


def _frame_data(
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    dev_mask: np.ndarray | None = None,
    test_mask: np.ndarray | None = None,
    hidden: np.ndarray | None = None,
    mfcc: np.ndarray | None = None,
) -> a2.FrameData:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    refined_scores = np.asarray(refined_scores, dtype=np.float64)
    size = labels.size
    if train_mask is None:
        train_mask = np.ones(size, dtype=bool)
    if dev_mask is None:
        dev_mask = np.zeros(size, dtype=bool)
    if test_mask is None:
        test_mask = np.zeros(size, dtype=bool)
    value, taxonomy = a2.taxonomy_from_scores(
        labels,
        short_scores,
        refined_scores,
    )
    if hidden is None:
        hidden = np.zeros((size, 2), dtype=np.float32)
    if mfcc is None:
        mfcc = np.zeros((size, 2), dtype=np.float32)
    return a2.FrameData(
        labels=labels,
        short_scores=short_scores,
        refined_scores=refined_scores,
        speaker_ids=np.asarray([f"s{(i % 4) + 1}" for i in range(size)]),
        source_key=np.asarray([f"src{i // 3}" for i in range(size)]),
        noise_name=np.asarray(["seen_noise"] * size),
        condition=np.asarray(["5"] * size),
        value=value,
        taxonomy=taxonomy,
        train_mask=np.asarray(train_mask, dtype=bool),
        dev_mask=np.asarray(dev_mask, dtype=bool),
        test_mask=np.asarray(test_mask, dtype=bool),
        speakers={
            "TRAIN": ["s1"],
            "DEV": ["s2"],
            "INTERNAL_TEST": ["s3", "s4"],
        },
        hidden=np.asarray(hidden, dtype=np.float32),
        mfcc=np.asarray(mfcc, dtype=np.float32),
    )


def _budget_rows(
    m1_seed_deltas: list[float],
    m2_seed_deltas: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, deltas in (("M1", m1_seed_deltas), ("M2", m2_seed_deltas)):
        for seed, delta in zip(a2.TRAINING_SEEDS, deltas):
            for domain, domain_delta in (
                ("all", delta),
                ("seen", 0.0),
                ("unseen", 0.0),
                ("clean", 0.0),
            ):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "budget": a2.PRIMARY_BUDGET,
                        "domain": domain,
                        "utility": domain_delta,
                    }
                )
    for seed in a2.TRAINING_SEEDS:
        for domain in ("all", "seen", "unseen", "clean"):
            rows.append(
                {
                    "method": "M0",
                    "seed": seed,
                    "budget": a2.PRIMARY_BUDGET,
                    "domain": domain,
                    "utility": 0.0,
                }
            )
    return rows


def _cluster_rows(
    m1_point: float,
    m2_point: float,
) -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "seed": "aggregate_5_seeds",
            "budget": a2.PRIMARY_BUDGET,
            "unit": "source_cluster",
            "point": point,
            "ci95_low": point - 0.001,
            "ci95_high": point + 0.001,
        }
        for method, point in (("M1", m1_point), ("M2", m2_point))
    ]


def _compute_rows(
    m1_activation: float = 0.05,
    m2_activation: float = 0.05,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, activation in (("M1", m1_activation), ("M2", m2_activation)):
        advantage = 0.8
        rows.append(
            {
                "method": method,
                "budget": a2.PRIMARY_BUDGET,
                "activation": activation,
                "conditional_advantage_ms_per_frame": advantage,
                "conditional_advantage_fraction": advantage / 3.0,
            }
        )
    return rows


class TaxonomyAndMetricTests(unittest.TestCase):
    def test_taxonomy_and_signed_value_agree(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        short = np.asarray([0.9, 0.4, 0.4, 0.9, 0.4, 0.9])
        refined = np.asarray([0.4, 0.4, 0.4, 0.9, 0.9, 0.9])
        value, taxonomy = a2.taxonomy_from_scores(labels, short, refined)
        np.testing.assert_array_equal(
            taxonomy,
            np.asarray(["R", "SS", "I", "SS", "H", "SS"]),
        )
        np.testing.assert_array_equal(value, np.asarray([1, 0, 0, 0, -1, 0]))

    def test_utility_uses_actual_short_to_refined_execution(self) -> None:
        data = _frame_data(
            labels=np.asarray([0, 0, 1, 1]),
            short_scores=np.asarray([0.4, 0.9, 0.6, 0.1]),
            refined_scores=np.asarray([0.4, 0.1, 0.9, 0.9]),
        )
        metrics = a2.method_metrics(
            data,
            np.asarray([False, True, False, True]),
            np.ones(4, dtype=bool),
        )
        self.assertEqual(metrics["frames"], 4)
        self.assertEqual(metrics["selected"], 2)
        self.assertAlmostEqual(metrics["utility"], 0.5)
        self.assertAlmostEqual(metrics["activation_rate"], 0.5)
        self.assertEqual(metrics["correction"], 2)
        self.assertEqual(metrics["harm"], 0)
        self.assertAlmostEqual(metrics["value_captured"], 1.0)


class SplitTests(unittest.TestCase):
    def test_three_way_split_is_disjoint_and_complete(self) -> None:
        speakers = [f"spk{index:03d}" for index in range(40)]
        train, dev, test, roles = a2.three_way_speaker_split(speakers)
        self.assertEqual(int(train.sum()), 24)
        self.assertEqual(int(dev.sum()), 8)
        self.assertEqual(int(test.sum()), 8)
        self.assertFalse(np.any(train & dev))
        self.assertFalse(np.any(train & test))
        self.assertFalse(np.any(dev & test))
        self.assertTrue(np.all(train | dev | test))
        self.assertEqual(len(set(roles["TRAIN"])), 24)
        self.assertEqual(len(set(roles["DEV"])), 8)
        self.assertEqual(len(set(roles["INTERNAL_TEST"])), 8)

    def test_split_fingerprint_is_order_independent_and_deterministic(self) -> None:
        speakers = [f"spk{index:03d}" for index in range(40)]
        train, dev, test, _ = a2.three_way_speaker_split(speakers)
        masks = {"TRAIN": train, "DEV": dev, "INTERNAL_TEST": test}
        first = a2.split_fingerprint(np.asarray(speakers), masks)
        order = np.arange(len(speakers))[::-1]
        shuffled = np.asarray(speakers)[order]
        shuffled_masks = {
            name: np.asarray(mask)[order] for name, mask in masks.items()
        }
        second = a2.split_fingerprint(shuffled, shuffled_masks)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


class SelectionTests(unittest.TestCase):
    def test_calibration_uses_only_dev_rows(self) -> None:
        calibration = np.asarray([True, True, True, False, False])
        first = np.asarray([0.1, 0.9, 0.8, 0.2, 0.3])
        second = np.asarray([0.1, 0.9, 0.8, 100.0, 200.0])
        threshold_first = a2.calibrate_threshold(
            first,
            calibration,
            2.0 / 3.0,
        )
        threshold_second = a2.calibrate_threshold(
            second,
            calibration,
            2.0 / 3.0,
        )
        self.assertEqual(threshold_first, threshold_second)
        selected = a2.apply_threshold(second, ~calibration, threshold_second)
        np.testing.assert_array_equal(selected, [False, False, False, True, True])

    def test_paired_cluster_bootstrap_point_and_single_cluster_interval(self) -> None:
        result = a2.paired_cluster_bootstrap(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray(["a", "a", "a"]),
            repeats=50,
            seed=7,
        )
        self.assertAlmostEqual(result["point"], 2.0)
        self.assertAlmostEqual(result["ci95_low"], 2.0)
        self.assertAlmostEqual(result["ci95_high"], 2.0)
        with self.assertRaises(ValueError):
            a2.paired_cluster_bootstrap(
                np.asarray([1.0]),
                np.asarray(["a", "b"]),
                repeats=1,
                seed=1,
            )


class ModelScoringTests(unittest.TestCase):
    def test_m1_and_m2_predictions_have_declared_scale(self) -> None:
        features = np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32)
        m1 = a2.build_model("M1", 2)
        with torch.no_grad():
            for parameter in m1.parameters():
                parameter.zero_()
            m1.net[-1].bias.fill_(0.25)
        m1_scores = a2.predict_method(
            m1,
            "M1",
            features,
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(m1_scores, [0.25, 0.25], atol=1e-7)

        m2 = a2.build_model("M2", 2)
        with torch.no_grad():
            for parameter in m2.parameters():
                parameter.zero_()
            m2.primary.bias.fill_(1.0)
        m2_scores = a2.predict_method(
            m2,
            "M2",
            features,
            device=torch.device("cpu"),
        )
        expected = 1.0 / (1.0 + np.exp(-1.0))
        np.testing.assert_allclose(m2_scores, [expected, expected], atol=1e-7)

    def test_primary_tie_break_prefers_m1(self) -> None:
        assessment = a2.assess_primary(
            budget_rows=_budget_rows(
                [0.01, 0.01, 0.01, 0.01, -0.001],
                [0.01, 0.01, 0.01, 0.01, -0.001],
            ),
            cluster_rows=_cluster_rows(0.01, 0.01),
            cell_rows=[],
            compute_rows=_compute_rows(),
        )
        self.assertEqual(assessment["primary_method"], "M1")
        self.assertEqual(assessment["status"], "METHOD_GO")

    def test_passing_lower_point_candidate_wins_over_control_failure(self) -> None:
        assessment = a2.assess_primary(
            budget_rows=_budget_rows(
                [0.03, 0.03, 0.03, 0.03, -0.001],
                [0.01, 0.01, 0.01, 0.01, -0.001],
            ),
            cluster_rows=_cluster_rows(0.03, 0.01),
            cell_rows=[],
            compute_rows=_compute_rows(m1_activation=0.20),
        )
        self.assertEqual(assessment["primary_method"], "M2")
        self.assertEqual(assessment["status"], "METHOD_GO")


class ReplicateEvaluationTests(unittest.TestCase):
    def test_replicate_scores_are_evaluated_against_replicate_data(self) -> None:
        size = 20
        labels = np.asarray([0, 1] * (size // 2), dtype=np.int64)
        short = np.linspace(0.05, 0.95, size)
        refined = np.asarray(
            [0.1 if label == 0 else 0.9 for label in labels],
            dtype=np.float64,
        )
        dev = np.zeros(size, dtype=bool)
        dev[:5] = True
        test = np.zeros(size, dtype=bool)
        test[5:] = True
        first = _frame_data(
            labels,
            short,
            refined,
            dev_mask=dev,
            test_mask=test,
        )
        second = _frame_data(
            labels,
            1.0 - short,
            refined,
            dev_mask=dev,
            test_mask=test,
        )
        scores = {
            "M0": a2.uncertainty_score(first.short_scores),
            "M1": np.linspace(0.0, 1.0, size),
            "M2": np.linspace(0.0, 1.0, size),
        }
        first_metrics = a2.evaluate_replicate_scores(first, scores)
        second_scores = dict(scores)
        second_scores["M0"] = a2.uncertainty_score(second.short_scores)
        second_metrics = a2.evaluate_replicate_scores(second, second_scores)
        self.assertEqual(first_metrics["M0"]["frames"], int(test.sum()))
        self.assertNotEqual(
            first_metrics["M0"]["utility"],
            second_metrics["M0"]["utility"],
        )


class GuardTests(unittest.TestCase):
    def test_new_final_ood_remains_sealed(self) -> None:
        valid = {
            "new_final_ood": {
                "touched": False,
                "result_generation": "forbidden",
            }
        }
        a2.assert_new_final_ood_untouched(valid)
        with self.assertRaises(ValueError):
            a2.assert_new_final_ood_untouched(
                {
                    "new_final_ood": {
                        "touched": True,
                        "result_generation": "forbidden",
                    }
                }
            )
        with self.assertRaises(RuntimeError):
            a2.load_new_final_ood(Path("forbidden.npz"))

    def test_claim_once_rejects_a_second_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "attempt.json"
            a2.claim_once(
                marker,
                command="smoke",
                protocol_sha256="a" * 64,
            )
            with self.assertRaises(FileExistsError):
                a2.claim_once(
                    marker,
                    command="smoke",
                    protocol_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
