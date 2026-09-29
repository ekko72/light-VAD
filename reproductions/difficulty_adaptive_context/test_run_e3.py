# -*- coding: utf-8 -*-
"""Focused checks for the frozen E3 BVRA runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from reproductions.difficulty_adaptive_context import run_e3


def _population(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    full_scores: np.ndarray,
    sources: np.ndarray | None = None,
    segments: tuple[run_e3.Segment, ...] | None = None,
) -> run_e3.E3Population:
    labels = np.asarray(labels, dtype=np.int64)
    short_scores = np.asarray(short_scores, dtype=np.float64)
    full_scores = np.asarray(full_scores, dtype=np.float64)
    if sources is None:
        sources = np.zeros(labels.size, dtype=np.int64)
    sources = np.asarray(sources, dtype=str)
    if segments is None:
        segments = (
            run_e3.Segment(
                start=0,
                end=int(labels.size),
                source_key=str(sources[0]),
                condition="clean",
                noise_name="none",
            ),
        )
    frame = pd.DataFrame(
        {
            "global_index": np.arange(labels.size, dtype=np.int64),
            "source_key": sources,
            "speaker_id": np.full(labels.size, "speaker", dtype=object),
            "noise_name": np.full(labels.size, "none", dtype=object),
            "condition": np.full(labels.size, "clean", dtype=object),
            "seen": np.ones(labels.size, dtype=bool),
            "label": labels,
            "short_score": short_scores,
            "embedded_short_score": short_scores,
            "full_score": full_scores,
        }
    )
    return run_e3.E3Population(
        frame=frame,
        labels=labels,
        short_scores=short_scores,
        full_scores=full_scores,
        global_index=frame["global_index"].to_numpy(dtype=np.int64),
        source_keys=sources,
        speaker_ids=frame["speaker_id"].astype(str).to_numpy(),
        noise_names=frame["noise_name"].astype(str).to_numpy(),
        conditions=frame["condition"].astype(str).to_numpy(),
        seen=frame["seen"].to_numpy(dtype=bool),
        segments=segments,
        sources=run_e3._source_index(sources),
    )


class ThresholdTaxonomyTests(unittest.TestCase):
    def test_taxonomy_and_i_over_r(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        short_scores = np.asarray([0.1, 0.9, 0.1, 0.9], dtype=np.float64)
        full_scores = np.asarray([0.2, 0.8, 0.9, 0.1], dtype=np.float64)
        ss, r, i, h = run_e3._taxonomy_at(
            labels,
            short_scores,
            full_scores,
            threshold=0.5,
        )
        self.assertEqual(ss.tolist(), [True, False, False, False])
        self.assertEqual(r.tolist(), [False, False, True, False])
        self.assertEqual(i.tolist(), [False, True, False, False])
        self.assertEqual(h.tolist(), [False, False, False, True])

        population = _population(
            labels=labels,
            short_scores=short_scores,
            full_scores=full_scores,
        )
        deployed = next(
            row
            for row in run_e3._threshold_rows(population)
            if row["deployed_threshold"]
        )
        self.assertAlmostEqual(deployed["P_R"], 0.25)
        self.assertAlmostEqual(deployed["P_I"], 0.25)
        self.assertAlmostEqual(deployed["P_H"], 0.25)
        self.assertAlmostEqual(deployed["NetRefinability"], 0.0)
        self.assertAlmostEqual(deployed["I_over_R"], 1.0)

    def test_event_bins_exclude_missing_distances(self) -> None:
        masks = run_e3._event_bin_masks(
            np.asarray([0.0, 1.0, 2.0, run_e3.MISSING_DISTANCE])
        )
        self.assertEqual(int(np.count_nonzero(masks["0"])), 1)
        self.assertEqual(int(np.count_nonzero(masks["1-2"])), 2)
        self.assertEqual(
            sum(int(np.count_nonzero(mask)) for mask in masks.values()),
            3,
        )


class JitterTests(unittest.TestCase):
    def test_jitter_is_deterministic_and_legal(self) -> None:
        labels = np.asarray(
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            dtype=np.int64,
        )
        segments = (
            run_e3.Segment(0, 6, "s1", "clean", "none"),
            run_e3.Segment(6, 10, "s1", "clean", "none"),
        )
        first = run_e3._jitter_labels(
            labels,
            segments,
            width_frames=2,
            seed=113,
        )
        second = run_e3._jitter_labels(
            labels,
            segments,
            width_frames=2,
            seed=113,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(np.isin(first, (0, 1))))
        for segment in segments:
            runs = run_e3._find_runs(
                first[segment.start : segment.end] == 1
            )
            self.assertTrue(all(run_end - run_start >= 1 for run_start, run_end in runs))

    def test_jitter_does_not_escape_segment_bounds(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 1, 1], dtype=np.int64)
        segments = (
            run_e3.Segment(0, 2, "s1", "clean", "none"),
            run_e3.Segment(4, 6, "s1", "clean", "none"),
        )
        jittered = run_e3._jitter_labels(
            labels,
            segments,
            width_frames=5,
            seed=227,
        )
        self.assertEqual(jittered[2:4].tolist(), [0, 0])
        for segment in segments:
            runs = run_e3._find_runs(
                jittered[segment.start : segment.end] == 1
            )
            self.assertTrue(all(run_end - run_start >= 1 for run_start, run_end in runs))


class BootstrapTests(unittest.TestCase):
    def test_source_cluster_bootstrap_is_deterministic(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        sources = run_e3._source_index(
            np.asarray(["a", "a", "b", "b"], dtype=str)
        )
        sums, counts = run_e3._source_sums(values, None, sources)
        first = run_e3._bootstrap_mean_from_sums(
            sums,
            counts,
            seed=12345,
        )
        second = run_e3._bootstrap_mean_from_sums(
            sums,
            counts,
            seed=12345,
        )
        self.assertEqual(first, second)

        left_sums, left_counts = run_e3._source_sums(
            np.asarray([2.0, 3.0, 5.0, 7.0]),
            None,
            sources,
        )
        right_sums, right_counts = run_e3._source_sums(
            np.asarray([1.0, 1.0, 2.0, 2.0]),
            None,
            sources,
        )
        first_difference = run_e3._bootstrap_difference_from_sums(
            left_sums,
            left_counts,
            right_sums,
            right_counts,
            seed=54321,
        )
        second_difference = run_e3._bootstrap_difference_from_sums(
            left_sums,
            left_counts,
            right_sums,
            right_counts,
            seed=54321,
        )
        self.assertEqual(first_difference, second_difference)


class GateAndClaimTests(unittest.TestCase):
    def test_status_priority(self) -> None:
        decision_rows = [
            {"group": "P_PLUS_NO_FLIP", "frame_share": 0.50},
        ]
        gates = {
            "G1": {"passed": False},
            "G2": {"passed": False},
            "G3": {"passed": False},
            "G4": {"passed": False},
        }
        self.assertEqual(
            run_e3._select_e3_status(gates, decision_rows, 0.01),
            "VALUE_STRUCTURE_CONDITIONAL",
        )
        gates["G1"]["passed"] = True
        self.assertEqual(
            run_e3._select_e3_status(gates, decision_rows, 0.01),
            "DECISION_ONLY_SPARSITY",
        )
        gates["G2"]["passed"] = True
        gates["G3"]["passed"] = True
        gates["G4"]["passed"] = True
        self.assertEqual(
            run_e3._select_e3_status(gates, decision_rows, 0.01),
            "ROBUST_VALUE_STRUCTURE",
        )
        gates["G3"]["passed"] = False
        gates["G4"]["passed"] = False
        self.assertEqual(
            run_e3._select_e3_status(gates, decision_rows, 0.01),
            "BOUNDARY_SENSITIVE_VALUE_STRUCTURE",
        )
        gates["G3"]["passed"] = True
        self.assertEqual(
            run_e3._select_e3_status(gates, decision_rows, 0.01),
            "VALUE_STRUCTURE_CONDITIONAL",
        )

    def test_claim_mappings(self) -> None:
        supported_gate = {"aggregate_positive": True}
        rows_supported = [
            {"group": "P_PLUS_NO_FLIP", "frame_share": 0.01},
            {"group": "P_PLUS_ANY", "frame_share": 0.10},
        ]
        self.assertEqual(
            run_e3._claim_b(supported_gate, rows_supported, 0.01),
            "SUPPORTED",
        )
        rows_broad = [
            {"group": "P_PLUS_NO_FLIP", "frame_share": 0.50},
            {"group": "P_PLUS_ANY", "frame_share": 0.60},
        ]
        self.assertEqual(
            run_e3._claim_b(supported_gate, rows_broad, 0.01),
            "TOO_STRONG",
        )
        self.assertEqual(
            run_e3._claim_b(
                {"aggregate_positive": False},
                rows_supported,
                0.01,
            ),
            "NOT_SUPPORTED",
        )
        self.assertEqual(run_e3._claim_c(0.01, 0.10), "SUPPORTED")
        self.assertEqual(run_e3._claim_c(0.01, 0.01), "CONDITIONAL")
        self.assertEqual(run_e3._claim_c(0.10, 0.20), "NOT_SUPPORTED")
        boundary_rows = [{"boundary_exclusion_ms": 100.0, "mean_v_log": 0.1}]
        self.assertEqual(
            run_e3._claim_f("SUPPORTED", boundary_rows),
            "NOT_SUPPORTED",
        )
        self.assertEqual(
            run_e3._claim_f("NOT_SUPPORTED", boundary_rows),
            "UNRESOLVED",
        )


class ManifestTests(unittest.TestCase):
    def test_verifier_ignores_manifest_self_and_detects_hash_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            output.mkdir(parents=True)
            summary_path = output / "e3_final_summary.json"
            artifact_path = output / "artifact.txt"
            manifest_path = output / "e3_execution_manifest.json"
            summary_path.write_text(
                json.dumps({"E3_STATUS": "TEST_STATUS"}),
                encoding="utf-8",
            )
            artifact_path.write_text("original", encoding="utf-8")
            manifest = {
                "ARTIFACTS": [
                    {
                        "path": str(artifact_path.relative_to(root)),
                        "sha256": run_e3._sha256_file(artifact_path),
                    }
                ]
            }
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with (
                mock.patch.object(run_e3, "REPO_ROOT", root),
                mock.patch.object(run_e3, "OUTPUT_ROOT", output),
                mock.patch.object(
                    run_e3,
                    "EXECUTION_MANIFEST_PATH",
                    manifest_path,
                ),
                mock.patch.object(
                    run_e3,
                    "REQUIRED_OUTPUT_FILES",
                    (
                        "e3_final_summary.json",
                        "e3_execution_manifest.json",
                        "artifact.txt",
                    ),
                ),
                mock.patch.object(run_e3, "FIGURE_FILES", ()),
            ):
                verified = run_e3._verify_required_artifacts()
                self.assertEqual(
                    verified["summary"]["E3_STATUS"],
                    "TEST_STATUS",
                )
                artifact_path.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "manifest artifact hash mismatch",
                ):
                    run_e3._verify_required_artifacts()


class LosoTests(unittest.TestCase):
    def test_unary_loso_uses_direct_means_and_contrasts_use_differences(
        self,
    ) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        short_scores = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float64)
        full_scores = np.asarray([0.1, 0.9, 0.3, 0.7], dtype=np.float64)
        population = _population(
            labels=labels,
            short_scores=short_scores,
            full_scores=full_scores,
            sources=np.asarray(["s1", "s1", "s2", "s2"], dtype=str),
        )
        distances = {
            "onset_distance": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "offset_distance": np.asarray(
                [run_e3.MISSING_DISTANCE] * 4
            ),
            "posterior_transition_distance": np.asarray(
                [run_e3.MISSING_DISTANCE] * 4
            ),
        }
        rows = run_e3._loso_rows(population, distances)
        v_log = run_e3._logloss_frame(
            labels,
            short_scores,
        ) - run_e3._logloss_frame(labels, full_scores)
        _, r, _, h = run_e3._taxonomy_at(
            labels,
            short_scores,
            full_scores,
            run_e3.DECISION_THRESHOLD,
        )
        unary = next(
            row for row in rows if row["statistic"] == "mean_v_log"
        )
        self.assertAlmostEqual(unary["full_estimate"], float(np.mean(v_log)))
        net = next(
            row for row in rows if row["statistic"] == "net_refinability"
        )
        expected_net = float(np.mean(r) - np.mean(h))
        self.assertAlmostEqual(net["full_estimate"], expected_net)

        contrast = next(
            row
            for row in rows
            if row["statistic"]
            == "onset_distance:near_0_2_minus_far_3_plus"
        )
        expected_contrast = float(np.mean(v_log[:3]) - v_log[3])
        self.assertAlmostEqual(
            contrast["full_estimate"],
            expected_contrast,
        )


if __name__ == "__main__":
    unittest.main()
