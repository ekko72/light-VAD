# -*- coding: utf-8 -*-
"""Unit checks for the A12 routing robustness runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.run_a12_routing_robustness import (
    assess_a12,
    build_acoustic_rows,
    load_a11_cost_model,
    load_a12_bundle,
    project_cost,
    source_rank_selection,
    source_rank_uncertainty,
)


def _cost_model() -> dict[str, float | int | str]:
    return {
        "source": "test",
        "short_gate_ms_per_frame": 2.0,
        "adaptive_5pct_ms_per_frame": 2.5,
        "adaptive_5pct_activation": 0.05,
        "refinement_ms_per_selected_frame": 0.5,
        "projected_base_ms_per_frame": 2.475,
        "always_refine_ms_per_frame": 3.0,
        "short_macs_per_frame": 100,
        "refinement_macs_per_selected_frame": 20,
        "cache_bytes": 1000,
    }


def _row(
    *,
    section: str,
    domain: str,
    condition: str,
    target_ssr: float | None,
    gate: str,
    activation: float,
    utility: float,
    ci_low: float | None = None,
    major: bool = True,
) -> dict[str, object]:
    return {
        "section": section,
        "domain": domain,
        "condition": condition,
        "target_ssr": target_ssr,
        "gate": gate,
        "activation_rate": activation,
        "net_utility_per_selected": utility,
        "net_utility_per_selected_ci95_low": (
            utility if ci_low is None else ci_low
        ),
        "is_major_development_cell": major,
    }


class A12RoutingRobustnessTests(unittest.TestCase):
    def test_load_bundle_uses_adaptive_refinement_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            scores = np.asarray([0.1, 0.6, 0.4, 0.9])
            refined_scores = np.asarray([0.9, 0.4, 0.8, 0.2])
            reference_long_scores = np.asarray([0.8, 0.3, 0.7, 0.1])
            np.savez(
                path,
                labels=labels,
                short_scores=scores,
                full_adaptive_scores=refined_scores,
                long_scores=reference_long_scores,
                test_mask=np.asarray([False, False, True, True]),
                calibration_mask=np.asarray([True, True, False, False]),
                source_key=np.asarray(["a", "a", "b", "b"]),
                condition=np.asarray(["20", "20", "20", "20"]),
                noise_name=np.asarray(
                    ["Babble_noise"] * 4,
                    dtype="<U32",
                ),
                threshold=np.asarray(0.13),
            )

            bundle = load_a12_bundle(path)

        self.assertEqual(bundle["threshold"], 0.13)
        self.assertEqual(bundle["labels"].dtype, np.int64)
        self.assertTrue(
            np.array_equal(bundle["refined_scores"], refined_scores)
        )
        self.assertFalse(
            np.array_equal(
                bundle["refined_scores"],
                reference_long_scores,
            )
        )
        self.assertEqual(bundle["source_key"].tolist(), ["a", "a", "b", "b"])

    def test_source_rank_gate_is_within_source_and_deterministic(self) -> None:
        scores = np.asarray([0.50, 0.51, 0.80, 0.20, 0.40, 0.70])
        source = np.asarray(["a", "a", "a", "b", "b", "b"])
        mask = np.ones(scores.size, dtype=bool)

        first = source_rank_uncertainty(scores, source, mask=mask)
        second = source_rank_uncertainty(scores, source, mask=mask)
        selected = source_rank_selection(
            first,
            mask=mask,
            activation_rate=1.0 / 3.0,
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(selected.tolist(), [True, False, False, False, True, False])
        self.assertEqual(source_rank_selection(
            first,
            mask=mask,
            activation_rate=1.0,
        ).sum(), scores.size)

    def test_a11_cost_model_projects_activation_linearly(self) -> None:
        cost_model = _cost_model()
        low = project_cost(0.05, cost_model)
        high = project_cost(0.10, cost_model)

        self.assertAlmostEqual(low["estimated_ms_per_frame"], 2.5)
        self.assertAlmostEqual(high["estimated_ms_per_frame"], 2.525)
        self.assertEqual(low["analytical_macs_per_frame"], 101)
        self.assertEqual(high["analytical_macs_per_frame"], 102)
        self.assertAlmostEqual(
            low["estimated_relative_to_always_refine"],
            2.5 / 3.0,
        )

    def test_assessment_returns_go_when_all_major_cells_are_positive(self) -> None:
        ssr = [
            _row(
                section="ssr_sweep",
                domain="all",
                condition="all_noisy",
                target_ssr=0.5,
                gate="raw",
                activation=0.05,
                utility=0.2,
            )
        ]
        acoustic = [
            _row(
                section="acoustic_cell",
                domain="seen",
                condition="0",
                target_ssr=None,
                gate="raw",
                activation=0.05,
                utility=0.1,
            )
        ]

        decision = assess_a12(ssr, acoustic)

        self.assertEqual(decision["status"], "GO")
        self.assertEqual(decision["point_failure_count"], 0)

    def test_assessment_reports_no_go_when_rank_stabilizer_fails(self) -> None:
        raw = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="raw",
            activation=0.06,
            utility=-0.1,
        )
        rank = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="source_rank",
            activation=0.06,
            utility=-0.08,
        )

        decision = assess_a12([], [raw, rank])

        self.assertEqual(decision["status"], "NO_GO")
        self.assertEqual(decision["unresolved_failure_count"], 1)

    def test_assessment_allows_rank_stabilizer_to_resolve_raw_failure(self) -> None:
        raw = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="raw",
            activation=0.06,
            utility=-0.1,
        )
        rank = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="source_rank",
            activation=0.05,
            utility=0.1,
        )

        decision = assess_a12([], [raw, rank])

        self.assertEqual(decision["status"], "GO_WITH_SOURCE_RANK_STABILIZER")
        self.assertEqual(decision["resolved_failure_count"], 1)
        self.assertEqual(decision["unresolved_failure_count"], 0)

    def test_assessment_marks_point_only_rank_repair_as_conditional(self) -> None:
        raw = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="raw",
            activation=0.06,
            utility=-0.1,
        )
        rank = _row(
            section="acoustic_cell",
            domain="seen",
            condition="0",
            target_ssr=None,
            gate="source_rank",
            activation=0.05,
            utility=0.01,
            ci_low=-0.02,
        )

        decision = assess_a12([], [raw, rank])

        self.assertEqual(
            decision["status"],
            "GO_WITH_SOURCE_RANK_STABILIZER",
        )
        self.assertEqual(decision["resolved_point_only_count"], 1)
        self.assertIn("conditional", decision["interpretation"])

    def test_a11_cost_loader_reads_measured_kernel(self) -> None:
        payload = {
            "rows": [
                {
                    "name": "short",
                    "mean_ms_per_frame": 2.0,
                    "analytical_macs_per_frame": 100,
                    "refinement_macs_per_selected_frame": 20,
                },
                {
                    "name": "adaptive_5pct",
                    "mean_ms_per_frame": 2.5,
                    "test_activation_rate": 0.05,
                    "refinement_kernel": {
                        "mean_ms_per_frame": 0.5,
                    },
                },
                {
                    "name": "fixed_rf384",
                    "mean_ms_per_frame": 3.0,
                    "cache_bytes": 1000,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a11.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            model = load_a11_cost_model(path)

        self.assertEqual(model["refinement_macs_per_selected_frame"], 20)
        self.assertAlmostEqual(model["projected_base_ms_per_frame"], 2.475)

    def test_acoustic_rows_report_actual_ssr(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        short_scores = np.asarray(
            [0.45, 0.55, 0.20, 0.80, 0.60, 0.30, 0.70, 0.40]
        )
        refined_scores = np.asarray(
            [0.10, 0.90, 0.05, 0.95, 0.90, 0.20, 0.80, 0.30]
        )
        bundle = {
            "labels": labels,
            "short_scores": short_scores,
            "refined_scores": refined_scores,
            "source_key": np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"]),
            "condition": np.asarray(["0"] * 8),
            "noise_name": np.asarray(["seen", "seen", "seen", "seen", "unseen", "unseen", "unseen", "unseen"]),
            "test_mask": np.ones(8, dtype=bool),
            "threshold": 0.13,
        }

        rows = build_acoustic_rows(
            bundle,
            conditions=("0",),
            unseen_noise=("unseen",),
            rank_selection=np.zeros(8, dtype=bool),
            cost_model=_cost_model(),
            bootstrap_repeats=1,
            bootstrap_seed=17,
        )

        overall_all = next(
            row
            for row in rows
            if row["domain"] == "all"
            and row["condition"] == "overall"
            and row["gate"] == "raw"
        )
        self.assertAlmostEqual(overall_all["actual_ssr"], 0.5)


if __name__ == "__main__":
    unittest.main()
