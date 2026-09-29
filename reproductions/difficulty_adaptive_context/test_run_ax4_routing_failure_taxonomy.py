# -*- coding: utf-8 -*-
"""Unit checks for the AX4 routing failure taxonomy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.run_ax4_routing_failure_taxonomy import (
    Ax4Config,
    assess_ax4,
    bootstrap_cluster_weights,
    build_cells,
    classify_failure,
    cluster_cell_counts,
    load_config,
    raw_gate_selection,
    signed_values,
    summarize_cell,
    verify_hash,
)


def _row(
    *,
    section: str,
    domain: str,
    condition: str,
    target_ssr: float | None,
    failure_class: str,
) -> dict[str, object]:
    return {
        "section": section,
        "domain": domain,
        "condition": condition,
        "target_ssr": target_ssr,
        "mean_value": 0.01,
        "selected_mean_value": (
            -0.01 if failure_class != "NO_FAILURE" else 0.01
        ),
        "activation_rate": (
            0.16
            if failure_class == "F3_BUDGET_CALIBRATION_DRIFT"
            else 0.05
        ),
        "failure_class": failure_class,
    }


class Ax4RoutingFailureTaxonomyTests(unittest.TestCase):
    def test_frozen_protocol_loads(self) -> None:
        config = load_config()

        self.assertEqual(config.payload["protocol_id"], "A-AX4-v1")
        self.assertEqual(
            config.output_dir.name,
            "ax4_routing_failure_taxonomy",
        )
        self.assertEqual(
            config.payload["cells"]["major_cell_count"],
            30,
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

    def test_signed_values_match_frozen_definition(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        short = np.asarray([0.1, 0.4, 0.4, 0.6])
        refined = np.asarray([0.9, 0.9, 0.1, 0.9])

        values = signed_values(labels, short, refined)

        self.assertEqual(values.tolist(), [-1, 1, 0, 0])

    def test_raw_gate_uses_frozen_abs_threshold(self) -> None:
        scores = np.asarray([0.37, 0.5, 0.63, 0.64])

        selected = raw_gate_selection(scores, threshold=0.13)

        self.assertEqual(selected.tolist(), [True, True, True, False])

    def test_bootstrap_weights_resample_clusters_deterministically(self) -> None:
        first = bootstrap_cluster_weights(7, repeats=20, seed=17)
        second = bootstrap_cluster_weights(7, repeats=20, seed=17)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first.sum(axis=0),
            np.full(20, 7, dtype=np.int64),
        )

    def test_cluster_counts_separate_population_and_selection(self) -> None:
        values = np.asarray([1, -1, 0, 1, -1, 0], dtype=np.int64)
        selected = np.asarray([True, False, True, True, True, False])
        clusters = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

        counts = cluster_cell_counts(values, selected, clusters, 2)

        self.assertEqual(counts["frames"].tolist(), [3, 3])
        self.assertEqual(counts["corrections"].tolist(), [1, 1])
        self.assertEqual(counts["harms"].tolist(), [1, 1])
        self.assertEqual(counts["selected"].tolist(), [2, 2])
        self.assertEqual(counts["selected_corrections"].tolist(), [1, 1])
        self.assertEqual(counts["selected_harms"].tolist(), [0, 1])

    def test_failure_taxonomy_has_frozen_priority(self) -> None:
        self.assertEqual(
            classify_failure(
                population_mean_value=-0.02,
                selected_mean_value=-0.01,
                activation_rate=0.05,
                activation_warning_threshold=0.15,
            ),
            "F1_VALUE_SCARCITY",
        )
        self.assertEqual(
            classify_failure(
                population_mean_value=0.02,
                selected_mean_value=-0.01,
                activation_rate=0.05,
                activation_warning_threshold=0.15,
            ),
            "F2_RANKING_FAILURE",
        )
        self.assertEqual(
            classify_failure(
                population_mean_value=0.02,
                selected_mean_value=0.03,
                activation_rate=0.16,
                activation_warning_threshold=0.15,
            ),
            "F3_BUDGET_CALIBRATION_DRIFT",
        )
        self.assertEqual(
            classify_failure(
                population_mean_value=0.02,
                selected_mean_value=0.03,
                activation_rate=0.15,
                activation_warning_threshold=0.15,
            ),
            "NO_FAILURE",
        )

    def test_summarize_cell_reports_population_and_selected_value(self) -> None:
        cell = {
            "section": "acoustic_cell",
            "domain": "seen",
            "condition": "-5",
            "target_ssr": None,
            "actual_ssr": 0.5,
            "labels": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "short_scores": np.asarray([0.45, 0.55, 0.45, 0.45]),
            "refined_scores": np.asarray([0.9, 0.45, 0.1, 0.9]),
            "source_key": np.asarray(["a", "a", "b", "b"]),
            "speaker_ids": np.asarray(["s1", "s1", "s2", "s2"]),
            "threshold": 0.13,
        }

        row = summarize_cell(
            cell,
            bootstrap_repeats=20,
            bootstrap_seed=17,
            activation_warning_threshold=0.15,
        )

        self.assertEqual(row["frames"], 4)
        self.assertEqual(row["selected"], 4)
        self.assertAlmostEqual(row["p_positive"], 0.25)
        self.assertAlmostEqual(row["p_negative"], 0.5)
        self.assertAlmostEqual(row["mean_value"], -0.25)
        self.assertAlmostEqual(row["selected_mean_value"], -0.25)
        self.assertEqual(row["failure_class"], "F1_VALUE_SCARCITY")
        self.assertIn("mean_value_ci95_low", row)
        self.assertIn("speaker_sensitivity", row)

    def test_build_cells_recreates_ssr_and_acoustic_cells(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        bundle = {
            "labels": labels,
            "short_scores": scores.copy(),
            "refined_scores": scores.copy(),
            "source_key": np.asarray([f"utt-{i}" for i in range(8)]),
            "speaker_ids": np.asarray(["s1"] * 8),
            "condition": np.asarray(["20"] * 8),
            "noise_name": np.asarray(["seen"] * 8),
            "test_mask": np.ones(8, dtype=bool),
            "threshold": 0.13,
        }
        payload = {
            "cells": {
                "ssr_sweep": {
                    "domains": ["all"],
                    "conditions": ["20"],
                    "unseen_noise": ["unseen"],
                    "target_ssr": [0.5],
                    "frames_per_stratum": 2,
                    "resample_seed": 17,
                },
                "acoustic_cell": {
                    "domains": ["all"],
                    "conditions": ["20"],
                    "unseen_noise": ["unseen"],
                },
                "major_cell_count": 2,
            }
        }
        config = Ax4Config(
            protocol_path=Path("synthetic"),
            payload=payload,
        )

        cells = build_cells(bundle, config)

        self.assertEqual(
            [cell["section"] for cell in cells],
            ["ssr_sweep", "acoustic_cell"],
        )
        self.assertEqual(cells[0]["labels"].size, 2)
        self.assertAlmostEqual(cells[0]["actual_ssr"], 0.5)
        self.assertEqual(cells[1]["labels"].size, 8)

    def test_assessment_triggers_ax5_only_for_f3(self) -> None:
        focus = [
            {
                "section": "ssr_sweep",
                "domain": "seen",
                "condition": "all_noisy",
                "target_ssr": 0.9,
            },
            {
                "section": "acoustic_cell",
                "domain": "seen",
                "condition": "-5",
                "target_ssr": None,
            },
        ]
        rows = [
            _row(
                section="ssr_sweep",
                domain="seen",
                condition="all_noisy",
                target_ssr=0.9,
                failure_class="F3_BUDGET_CALIBRATION_DRIFT",
            ),
            _row(
                section="acoustic_cell",
                domain="seen",
                condition="-5",
                target_ssr=None,
                failure_class="F1_VALUE_SCARCITY",
            ),
        ]

        assessment = assess_ax4(rows, focus_cells=focus)

        self.assertEqual(
            assessment["status"],
            "F3_BUDGET_CALIBRATION_DRIFT_DETECTED",
        )
        self.assertTrue(assessment["ax5_triggered"])

    def test_assessment_reports_mixed_failure_without_ax5(self) -> None:
        focus = [
            {
                "section": "ssr_sweep",
                "domain": "seen",
                "condition": "all_noisy",
                "target_ssr": 0.9,
            },
            {
                "section": "acoustic_cell",
                "domain": "seen",
                "condition": "-5",
                "target_ssr": None,
            },
        ]
        rows = [
            _row(
                section="ssr_sweep",
                domain="seen",
                condition="all_noisy",
                target_ssr=0.9,
                failure_class="F1_VALUE_SCARCITY",
            ),
            _row(
                section="acoustic_cell",
                domain="seen",
                condition="-5",
                target_ssr=None,
                failure_class="F2_RANKING_FAILURE",
            ),
        ]

        assessment = assess_ax4(rows, focus_cells=focus)

        self.assertEqual(assessment["status"], "MIXED_FAILURE")
        self.assertFalse(assessment["ax5_triggered"])


if __name__ == "__main__":
    unittest.main()
