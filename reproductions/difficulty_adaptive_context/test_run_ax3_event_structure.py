# -*- coding: utf-8 -*-
"""Unit checks for the AX3 event-structure diagnostic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.run_ax3_event_structure import (
    Ax3Config,
    _ci_excludes_zero,
    _cluster_value_counts,
    _paired_metric_differences,
    assess_ax3,
    bootstrap_cluster_weights,
    build_event_structure,
    load_config,
    summarize_bins,
    summarize_contrast,
    verify_hash,
)


def _contrast_row(
    contrast_id: str,
    *,
    supported: bool,
    mean_diff: float,
    mean_low: float,
    mean_high: float,
    positive_low: float = 0.01,
    positive_high: float = 0.1,
) -> dict[str, object]:
    return {
        "id": contrast_id,
        "supported": supported,
        "mean_value_difference": mean_diff,
        "mean_value_ci95_low": mean_low,
        "mean_value_ci95_high": mean_high,
        "p_positive_difference": 0.02,
        "p_positive_ci95_low": positive_low,
        "p_positive_ci95_high": positive_high,
        "p_negative_difference": 0.0,
        "p_negative_ci95_low": -0.01,
        "p_negative_ci95_high": 0.01,
    }


class Ax3EventStructureTests(unittest.TestCase):
    def test_frozen_protocol_loads(self) -> None:
        config = load_config()

        self.assertEqual(config.payload["protocol_id"], "A-AX3-v1")
        self.assertEqual(
            config.output_dir.name,
            "ax3_event_structure",
        )

    def test_verify_hash_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.bin"
            path.write_bytes(b"first")
            self.assertEqual(
                verify_hash(path, "", verify=False),
                "not_verified",
            )
            with self.assertRaises(ValueError):
                verify_hash(path, "0" * 64, verify=True)

    def test_event_structure_bins_onsets_offsets_and_run_lengths(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int64)
        scores = np.asarray(
            [0.1, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.9, 0.1]
        )

        result = build_event_structure(
            labels,
            scores,
            [(0, labels.size)],
            decision_threshold=0.5,
            uncertainty_threshold=0.13,
        )

        self.assertEqual(
            result["onset_distance"].tolist(),
            [
                "not_in_speech",
                "not_in_speech",
                "0",
                "1-2",
                "1-2",
                "not_in_speech",
                "not_in_speech",
                "0",
                "not_in_speech",
            ],
        )
        self.assertEqual(
            result["offset_distance"].tolist(),
            [
                "pre_first_speech",
                "pre_first_speech",
                "in_speech",
                "in_speech",
                "in_speech",
                "0",
                "1-2",
                "in_speech",
                "0",
            ],
        )
        self.assertEqual(
            result["speech_run_length"].tolist(),
            [
                "not_in_speech",
                "not_in_speech",
                "3-5",
                "3-5",
                "3-5",
                "not_in_speech",
                "not_in_speech",
                "1-2",
                "not_in_speech",
            ],
        )
        self.assertEqual(
            result["silence_run_length"].tolist(),
            [
                "1-2",
                "1-2",
                "in_speech",
                "in_speech",
                "in_speech",
                "1-2",
                "1-2",
                "in_speech",
                "1-2",
            ],
        )

    def test_event_structure_resets_distances_at_utterance_boundaries(self) -> None:
        labels = np.asarray([0, 1, 1, 0, 1, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.8, 0.8, 0.1, 0.5, 0.5, 0.5])

        result = build_event_structure(
            labels,
            scores,
            [(0, 3), (3, 7)],
            decision_threshold=0.5,
            uncertainty_threshold=0.13,
        )

        self.assertEqual(result["onset_distance"][4], "0")
        self.assertEqual(result["offset_distance"][3], "pre_first_speech")
        self.assertEqual(result["speech_run_length"][4], "3-5")
        self.assertEqual(
            result["recent_posterior_transition"][4],
            "0",
        )
        self.assertEqual(
            result["recent_uncertainty_duration"][3:].tolist(),
            ["0", "1-2", "1-2", "3-5"],
        )

    def test_event_structure_transition_age_and_uncertainty_streak(self) -> None:
        labels = np.ones(6, dtype=np.int64)
        scores = np.asarray([0.45, 0.6, 0.6, 0.49, 0.49, 0.7])

        result = build_event_structure(
            labels,
            scores,
            [(0, scores.size)],
            decision_threshold=0.5,
            uncertainty_threshold=0.05,
        )

        self.assertEqual(
            result["recent_posterior_transition"].tolist(),
            ["none", "0", "1-2", "0", "1-2", "0"],
        )
        self.assertEqual(
            result["recent_uncertainty_duration"].tolist(),
            ["1-2", "0", "0", "1-2", "1-2", "0"],
        )

    def test_bootstrap_weights_are_shared_and_resample_clusters(self) -> None:
        first = bootstrap_cluster_weights(7, repeats=20, seed=17)
        second = bootstrap_cluster_weights(7, repeats=20, seed=17)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (7, 20))
        np.testing.assert_array_equal(
            first.sum(axis=0),
            np.full(20, 7, dtype=np.int64),
        )

    def test_bin_summary_matches_manual_counts_and_rates(self) -> None:
        labels = np.asarray(["focal", "focal", "reference", "reference"])
        values = np.asarray([1, -1, 0, 1], dtype=np.int64)
        clusters = np.asarray([0, 0, 1, 1], dtype=np.int64)
        weights = np.eye(2, dtype=np.int64)

        rows = summarize_bins(
            "synthetic",
            labels,
            values,
            clusters,
            bins=["focal", "reference"],
            weights=weights,
        )

        focal = rows[0]
        reference = rows[1]
        self.assertEqual(focal["frames"], 2)
        self.assertEqual(focal["corrections"], 1)
        self.assertEqual(focal["harms"], 1)
        self.assertAlmostEqual(focal["p_positive"], 0.5)
        self.assertAlmostEqual(focal["p_negative"], 0.5)
        self.assertAlmostEqual(focal["mean_value"], 0.0)
        self.assertAlmostEqual(reference["p_positive"], 0.5)
        self.assertAlmostEqual(reference["p_negative"], 0.0)
        self.assertAlmostEqual(reference["mean_value"], 0.5)
        self.assertAlmostEqual(focal["correction_share"], 0.5)
        self.assertAlmostEqual(reference["harm_share"], 0.0)

    def test_paired_metric_difference_uses_same_cluster_resample(self) -> None:
        values = np.asarray([1, 1, -1, 0, 1, -1], dtype=np.int64)
        clusters = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        focal_mask = np.asarray([True, True, False, False, True, False])
        reference_mask = np.asarray([False, False, True, True, False, True])
        weights = np.eye(3, dtype=np.int64)
        focal = _cluster_value_counts(
            focal_mask,
            values,
            clusters,
            3,
        )
        reference = _cluster_value_counts(
            reference_mask,
            values,
            clusters,
            3,
        )

        differences, intervals = _paired_metric_differences(
            focal,
            reference,
            weights,
        )

        self.assertAlmostEqual(differences["p_positive"], 1.0)
        self.assertAlmostEqual(differences["p_negative"], -2.0 / 3.0)
        self.assertAlmostEqual(differences["mean_value"], 5.0 / 3.0)
        self.assertTrue(
            _ci_excludes_zero(
                intervals["mean_value_ci95_low"],
                intervals["mean_value_ci95_high"],
            )
        )

    def test_contrast_summary_marks_supported_rule(self) -> None:
        values = np.asarray([1, 1, 0, -1, 1, 0], dtype=np.int64)
        clusters = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        contrast = {
            "id": "synthetic",
            "dimension": "synthetic",
            "focal_bins": ["focal"],
            "reference_bins": ["reference"],
            "_material_effect_threshold": 0.1,
            "_minimum_frames_each_side": 1,
        }
        row = summarize_contrast(
            contrast,
            values,
            clusters,
            focal_mask=np.asarray([True, True, False, False, True, False]),
            reference_mask=np.asarray([False, False, True, True, False, True]),
            weights=np.eye(3, dtype=np.int64),
            calibration_mask=np.asarray([True, True, False, True, False, False]),
            test_mask=np.asarray([False, False, True, False, True, True]),
        )

        self.assertTrue(row["supported"])
        self.assertEqual(row["focal_frames"], 3)
        self.assertEqual(row["reference_frames"], 3)

    def test_assessment_has_three_frozen_branches(self) -> None:
        supported = assess_ax3(
            [
                _contrast_row(
                    "supported",
                    supported=True,
                    mean_diff=0.02,
                    mean_low=0.01,
                    mean_high=0.03,
                )
            ]
        )
        limited = assess_ax3(
            [
                _contrast_row(
                    "limited",
                    supported=False,
                    mean_diff=0.002,
                    mean_low=-0.001,
                    mean_high=0.005,
                )
            ]
        )
        no_stable = assess_ax3(
            [
                _contrast_row(
                    "none",
                    supported=False,
                    mean_diff=0.001,
                    mean_low=-0.002,
                    mean_high=0.004,
                    positive_low=-0.01,
                    positive_high=0.02,
                )
            ]
        )

        self.assertEqual(
            supported["status"],
            "EVENT_STRUCTURE_SUPPORTED",
        )
        self.assertEqual(
            limited["status"],
            "EVENT_STRUCTURE_LIMITED",
        )
        self.assertEqual(
            no_stable["status"],
            "NO_STABLE_EVENT_STRUCTURE",
        )


if __name__ == "__main__":
    unittest.main()
