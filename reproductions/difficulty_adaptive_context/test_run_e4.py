# -*- coding: utf-8 -*-
"""Focused checks for the frozen E4 RSD runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from reproductions.difficulty_adaptive_context import run_e4


class AlignmentAndTaxonomyTests(unittest.TestCase):
    def test_replicate_alignment_requires_identical_labels_and_masks(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        test_mask = np.asarray([True, True, False, True], dtype=bool)
        replicates = {
            seed: {
                "labels": labels.copy(),
                "test_mask": test_mask.copy(),
            }
            for seed in run_e4.MODEL_SEEDS
        }

        result = run_e4._validate_replicate_alignment(
            labels=labels,
            test_mask=test_mask,
            replicates=replicates,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["n_frames_total"], 4)
        self.assertEqual(result["n_frames_test"], 3)
        self.assertEqual(result["n_replicates"], 5)

        replicates[run_e4.MODEL_SEEDS[-1]]["labels"] = np.asarray(
            [0, 0, 0, 1],
            dtype=np.int64,
        )
        with self.assertRaises(ValueError):
            run_e4._validate_replicate_alignment(
                labels=labels,
                test_mask=test_mask,
                replicates=replicates,
            )

    def test_taxonomy_codes_are_exhaustive_and_disjoint(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        short_scores = np.asarray([0.1, 0.9, 0.1, 0.9])
        long_scores = np.asarray([0.2, 0.8, 0.9, 0.1])

        taxonomy = run_e4._taxonomy_from_scores(
            labels,
            short_scores,
            long_scores,
        )

        self.assertEqual(taxonomy["SS"].tolist(), [True, False, False, False])
        self.assertEqual(taxonomy["R"].tolist(), [False, False, True, False])
        self.assertEqual(taxonomy["I"].tolist(), [False, True, False, False])
        self.assertEqual(taxonomy["H"].tolist(), [False, False, False, True])
        self.assertEqual(
            run_e4._taxonomy_codes(taxonomy).tolist(),
            [0, 2, 1, 3],
        )

    def test_stability_categories_and_replicate_counts(self) -> None:
        r_matrix = np.asarray(
            [
                [False, False, False, False, False, True],
                [False, False, False, False, True, True],
                [False, False, False, True, True, True],
                [False, False, True, True, True, True],
                [False, True, True, True, True, True],
            ],
            dtype=bool,
        )
        k_r = run_e4._count_matrix_rows(r_matrix)
        masks = run_e4._stability_masks(k_r)

        self.assertEqual(k_r.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            masks["NEVER_R"].tolist(),
            [True, False, False, False, False, False],
        )
        self.assertEqual(
            masks["UNSTABLE_R"].tolist(),
            [False, True, True, False, False, False],
        )
        self.assertEqual(
            masks["STABLE_R"].tolist(),
            [False, False, False, True, True, True],
        )
        self.assertEqual(
            masks["CORE_R"].tolist(),
            [False, False, False, False, True, True],
        )

    def test_independence_pmf_and_k_distribution_rows(self) -> None:
        pmf = run_e4._independence_k_pmf([0.5, 0.5])
        np.testing.assert_allclose(pmf, [0.25, 0.5, 0.25])

        k_r = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        rows = run_e4._k_distribution_rows(
            k_r=k_r,
            k_i=k_r,
            k_h=k_r,
            independence_pmf=np.asarray(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=np.float64,
            ),
        )

        self.assertEqual(len(rows), 18)
        k_r_rows = [row for row in rows if row["variable"] == "K_R"]
        self.assertEqual(
            [row["frames"] for row in k_r_rows],
            [1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            [row["independence_expected_P"] for row in k_r_rows],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(
            sum(float(row["P"]) for row in k_r_rows),
            1.0,
        )


class NullAndBootstrapTests(unittest.TestCase):
    def test_permutation_preserves_each_source_condition_stratum(self) -> None:
        mask = np.asarray(
            [True, True, False, True, False, True, True, False],
            dtype=bool,
        )
        strata = np.asarray(
            ["a", "a", "a", "a", "b", "b", "b", "b"],
            dtype=str,
        )
        permuted = run_e4._permute_within_strata(
            mask,
            strata,
            np.random.default_rng(123),
        )

        for stratum in np.unique(strata):
            selected = strata == stratum
            self.assertEqual(
                int(np.count_nonzero(mask[selected])),
                int(np.count_nonzero(permuted[selected])),
            )

    def test_permutation_overlap_is_deterministic(self) -> None:
        r_matrix = np.asarray(
            [
                [True, True, False, False, True, False, False, False],
                [False, True, True, False, False, True, False, False],
                [False, False, True, True, False, False, True, False],
                [True, False, False, True, False, True, False, False],
                [False, True, False, False, True, False, True, False],
            ],
            dtype=bool,
        )
        source_keys = np.asarray(
            ["a", "a", "a", "a", "b", "b", "b", "b"],
            dtype=str,
        )
        conditions = np.asarray(
            ["c", "c", "c", "c", "d", "d", "d", "d"],
            dtype=str,
        )
        with mock.patch.object(
            run_e4,
            "PERMUTATION_SEEDS",
            (101, 211),
        ):
            first_rows, first_summary = run_e4._permutation_overlap_rows(
                r_matrix=r_matrix,
                source_keys=source_keys,
                conditions=conditions,
            )
            second_rows, second_summary = run_e4._permutation_overlap_rows(
                r_matrix=r_matrix,
                source_keys=source_keys,
                conditions=conditions,
            )

        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first_rows), 2)

    def test_null_overlap_uses_the_frozen_maximum_reference(self) -> None:
        k_r = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        rows = run_e4._null_overlap_rows(
            k_r=k_r,
            independence_pmf=np.asarray(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=np.float64,
            ),
            permutation_rows=[
                {"P_K_R_GE_3": 0.1, "P_K_R_GE_4": 0.0},
                {"P_K_R_GE_3": 0.2, "P_K_R_GE_4": 0.1},
            ],
        )
        by_name = {row["statistic"]: row for row in rows}

        self.assertAlmostEqual(by_name["K_R_GE_3"]["observed"], 0.5)
        self.assertAlmostEqual(by_name["K_R_GE_3"]["null_reference"], 1.0)
        self.assertFalse(by_name["K_R_GE_3"]["exceeds_null_point"])
        self.assertAlmostEqual(by_name["K_R_GE_4"]["observed"], 1.0 / 3.0)
        self.assertAlmostEqual(by_name["K_R_GE_4"]["null_reference"], 0.05)
        self.assertTrue(by_name["K_R_GE_4"]["exceeds_null_point"])

    def test_source_cluster_bootstrap_is_deterministic(self) -> None:
        sources = run_e4._source_index(
            np.asarray(["a", "a", "b", "b"], dtype=str)
        )
        values = np.asarray([1.0, 2.0, 4.0, 8.0])

        with mock.patch.object(run_e4, "BOOTSTRAP_REPEATS", 100):
            first_ratio = run_e4._bootstrap_ratio(
                values,
                None,
                sources,
                seed=123,
            )
            second_ratio = run_e4._bootstrap_ratio(
                values,
                None,
                sources,
                seed=123,
            )
            first_difference = run_e4._bootstrap_difference(
                values,
                np.asarray([True, True, False, False]),
                np.asarray([False, False, True, True]),
                sources,
                seed=456,
            )
            second_difference = run_e4._bootstrap_difference(
                values,
                np.asarray([True, True, False, False]),
                np.asarray([False, False, True, True]),
                sources,
                seed=456,
            )

        self.assertEqual(first_ratio, second_ratio)
        self.assertEqual(first_difference, second_difference)


class DecompositionAndScoringTests(unittest.TestCase):
    def test_disagreement_decomposition_counts_non_r_fates(self) -> None:
        r_matrix = np.asarray(
            [
                [True, True, True, True, False, True],
                [False, True, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, False, True, False, True],
                [False, False, False, False, False, True],
            ],
            dtype=bool,
        )
        k_r = run_e4._count_matrix_rows(r_matrix)
        ss = np.zeros(r_matrix.shape, dtype=bool)
        i = np.zeros(r_matrix.shape, dtype=bool)
        h = np.zeros(r_matrix.shape, dtype=bool)

        i[1, 0] = True
        ss[2, 0] = True
        h[3, 0] = True
        i[4, 0] = True

        ss[2, 1] = True
        i[3, 1] = True
        h[4, 1] = True

        ss[3, 2] = True
        i[4, 2] = True

        i[4, 3] = True

        rows = run_e4._disagreement_decomposition_rows(
            k_r=k_r,
            r_matrix=r_matrix,
            taxonomy_matrices={"SS": ss, "I": i, "H": h},
        )
        aggregate = next(
            row for row in rows if row["K_R_category"] == "1-4"
        )

        self.assertEqual(aggregate["frames"], 4)
        self.assertEqual(aggregate["non_R_assignments"], 10)
        self.assertAlmostEqual(aggregate["P_non_R_SS"], 0.3)
        self.assertAlmostEqual(aggregate["P_non_R_I"], 0.5)
        self.assertAlmostEqual(aggregate["P_non_R_H"], 0.2)

    def test_proper_scoring_uses_frame_median_not_median_of_frame_means(
        self,
    ) -> None:
        masks = {
            "NEVER_R": np.asarray([True, False, False, False]),
            "UNSTABLE_R": np.asarray([False, True, False, False]),
            "STABLE_R": np.asarray([False, False, True, False]),
            "CORE_R": np.asarray([False, False, False, True]),
        }
        observables = {
            "mean_v_log": np.asarray([1.0, 3.0, 5.0, 7.0]),
            "median_v_log": np.asarray([10.0, 20.0, 30.0, 40.0]),
            "mean_v_brier": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "positive_value_count": np.asarray([1, 2, 3, 4]),
        }

        rows = run_e4._proper_scoring_rows(
            masks=masks,
            observables=observables,
        )

        self.assertAlmostEqual(rows[0]["median_v_log"], 10.0)
        self.assertAlmostEqual(rows[1]["median_v_log"], 20.0)
        self.assertAlmostEqual(rows[2]["median_v_log"], 30.0)
        self.assertAlmostEqual(rows[3]["median_v_log"], 40.0)


def _bootstrap_row(
    statistic: str,
    estimate: float,
    low: float,
    high: float,
    frames: int = 10_000,
) -> dict[str, object]:
    return {
        "statistic": statistic,
        "estimate": float(estimate),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "frames": int(frames),
        "direction_supported": bool(low > 0.0 or high < 0.0),
    }


def _gate_fixture(
    *,
    baseline_reproduced: bool = True,
    g1: bool = True,
    g2: bool = True,
    nontrivial: bool = True,
    source_codes: np.ndarray | None = None,
) -> dict[str, object]:
    n_frames = 10_000
    k_r = np.zeros(n_frames, dtype=np.int64)
    if nontrivial:
        k_r[:10] = 4
        k_r[10:30] = 3
        k_r[30:100] = 1
    else:
        k_r[:10] = 4
        k_r[10:30] = 3
        k_r[30:100] = 1
    masks = run_e4._stability_masks(k_r)
    if source_codes is None:
        source_codes = np.arange(n_frames, dtype=np.int64) % 100
    source_codes = np.asarray(source_codes, dtype=np.int64)

    observed3 = float(np.mean(k_r >= 3))
    observed4 = float(np.mean(k_r >= 4))
    if g1:
        reference3 = observed3 - 0.001
        reference4 = observed4 - 0.0005
        boot3_low = 0.0005
        boot4_low = 0.0001
    else:
        reference3 = observed3
        reference4 = observed4
        boot3_low = -0.0001
        boot4_low = -0.0001
    excess3 = observed3 - reference3
    excess4 = observed4 - reference4
    null_rows = [
        {
            "statistic": "K_R_GE_3",
            "observed": observed3,
            "null_reference": reference3,
            "observed_minus_null": excess3,
            "exceeds_null_point": bool(observed3 > reference3),
        },
        {
            "statistic": "K_R_GE_4",
            "observed": observed4,
            "null_reference": reference4,
            "observed_minus_null": excess4,
            "exceeds_null_point": bool(observed4 > reference4),
        },
    ]
    contrast_low = 0.005 if g2 else -0.005
    bootstrap_rows = [
        _bootstrap_row(
            "observed_minus_null_K_GE_3",
            excess3,
            boot3_low,
            max(boot3_low + 0.001, excess3 + 0.001),
        ),
        _bootstrap_row(
            "observed_minus_null_K_GE_4",
            excess4,
            boot4_low,
            max(boot4_low + 0.001, excess4 + 0.001),
        ),
        _bootstrap_row("stable_mean_v_log", 0.02, 0.01, 0.03),
        _bootstrap_row(
            "stable_minus_unstable_v_log",
            0.015,
            contrast_low,
            0.025,
        ),
        _bootstrap_row(
            "posterior_transition_0_vs_beyond_stable",
            0.01,
            0.005,
            0.015,
        ),
        _bootstrap_row("onset_0_vs_beyond_stable", 0.01, 0.005, 0.015),
        _bootstrap_row("offset_0_vs_beyond_stable", 0.01, 0.005, 0.015),
        _bootstrap_row(
            "stable_minus_unstable_short_margin",
            0.01,
            0.005,
            0.015,
        ),
        _bootstrap_row("seen_vs_unseen_stable", 0.01, 0.005, 0.015),
        _bootstrap_row(
            "stable_vs_unstable_non_r_i_share",
            0.02,
            0.01,
            0.03,
        ),
    ]
    seed_rows = [
        {
            "scope": "seed",
            "seed": 41,
            "NetRefinability": 0.01,
            "mean_v_log": 0.02,
        },
        {
            "scope": "aggregate",
            "seed": "aggregate",
            "NetRefinability": 0.01,
            "mean_v_log": 0.02,
        },
    ]
    return {
        "baseline": {"BASELINE_REPRODUCED": baseline_reproduced},
        "seed_rows": seed_rows,
        "k_r": k_r,
        "masks": masks,
        "null_rows": null_rows,
        "bootstrap_rows": bootstrap_rows,
        "loso_rows": [
            {
                "statistic": "stable_minus_unstable_v_log",
                "sign_consistent": bool(g2),
            },
            {
                "statistic": "P_STABLE_R",
                "sign_consistent": True,
            },
        ],
        "source_codes": source_codes,
    }


class GateAndArtifactTests(unittest.TestCase):
    def test_status_precedence_and_source_support(self) -> None:
        stable_core = run_e4._build_gates_and_claims(
            **_gate_fixture()
        )
        self.assertEqual(stable_core["E4_STATUS"], "STABLE_REFINABLE_CORE")
        self.assertTrue(stable_core["NONTRIVIAL_STABLE"])

        population = run_e4._build_gates_and_claims(
            **_gate_fixture(g2=False)
        )
        self.assertEqual(
            population["E4_STATUS"],
            "POPULATION_STABLE_INSTANCE_RELATIVE",
        )

        mostly_model = run_e4._build_gates_and_claims(
            **_gate_fixture(g1=False)
        )
        self.assertEqual(
            mostly_model["E4_STATUS"],
            "MOSTLY_MODEL_RELATIVE_REFINABILITY",
        )

        unsupported_source = run_e4._build_gates_and_claims(
            **_gate_fixture(
                source_codes=np.zeros(10_000, dtype=np.int64),
            )
        )
        self.assertFalse(unsupported_source["NONTRIVIAL_SOURCE_SUPPORT"])
        self.assertEqual(
            unsupported_source["E4_STATUS"],
            "POPULATION_STABLE_INSTANCE_RELATIVE",
        )

        invalid = run_e4._build_gates_and_claims(
            **_gate_fixture(baseline_reproduced=False)
        )
        self.assertEqual(invalid["E4_STATUS"], "INCONCLUSIVE_OR_INVALID")

        bad_length = _gate_fixture()
        bad_length["source_codes"] = bad_length["source_codes"][:-1]
        with self.assertRaises(ValueError):
            run_e4._build_gates_and_claims(**bad_length)

    def test_validate_frozen_protocol_checks_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_hash_path = root / "protocol.sha256"
            runner_hash_path = root / "runner.sha256"
            protocol_path.write_text(
                json.dumps(
                    {
                        "PROTOCOL_ID": run_e4.PROTOCOL_ID,
                        "PROTOCOL_STATUS": run_e4.PROTOCOL_STATUS,
                    }
                ),
                encoding="utf-8",
            )
            protocol_hash_path.write_text(
                run_e4._sha256_file(protocol_path) + "\n",
                encoding="ascii",
            )
            runner_hash_path.write_text(
                run_e4._runner_hash() + "\n",
                encoding="ascii",
            )

            with mock.patch.multiple(
                run_e4,
                PROTOCOL_PATH=protocol_path,
                PROTOCOL_HASH_PATH=protocol_hash_path,
                RUNNER_HASH_PATH=runner_hash_path,
            ):
                result = run_e4._validate_frozen_protocol()
                self.assertEqual(
                    result["protocol_sha256"],
                    run_e4._sha256_file(protocol_path),
                )

                protocol_path.write_text(
                    protocol_path.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    run_e4._validate_frozen_protocol()

    def test_verify_required_artifacts_checks_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            summary_path = output / "e4_final_summary.json"
            report_path = output / "e4_final_report.md"
            claim_path = output / "e4_claim_freeze.md"
            manifest_path = output / "e4_execution_manifest.json"
            run_e4._write_json(
                summary_path,
                {
                    "E4_STATUS": "POPULATION_STABLE_INSTANCE_RELATIVE",
                },
            )
            report_path.write_text("report\n", encoding="utf-8")
            claim_path.write_text("claim\n", encoding="utf-8")
            artifacts = [
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256": run_e4._sha256_file(path),
                }
                for path in (summary_path, report_path, claim_path)
            ]
            run_e4._write_json(
                manifest_path,
                {"ARTIFACTS": artifacts},
            )
            required = (
                summary_path.name,
                report_path.name,
                claim_path.name,
                manifest_path.name,
            )

            with mock.patch.multiple(
                run_e4,
                REPO_ROOT=root,
                OUTPUT_ROOT=output,
                EXECUTION_MANIFEST_PATH=manifest_path,
                REQUIRED_OUTPUT_FILES=required,
                FIGURE_FILES=(),
            ):
                result = run_e4._verify_required_artifacts()
                self.assertTrue(result["summary"]["E4_STATUS"])

                report_path.write_text("changed\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_e4._verify_required_artifacts()


if __name__ == "__main__":
    unittest.main()
