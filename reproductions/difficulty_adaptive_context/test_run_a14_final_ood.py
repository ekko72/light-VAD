# -*- coding: utf-8 -*-
"""Unit checks for the frozen A14 Final OOD confirmation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from reproductions.difficulty_adaptive_context.run_a14_final_ood import (
    ACTIVATION_FAILURE,
    ACTIVATION_WARNING,
    BOOTSTRAP_SEED,
    BOOTSTRAP_SEED_RULES,
    CELL_ORDER,
    DEFAULT_DATA_ROOT,
    EXPECTED_FRAMES,
    EXPECTED_NOISE_CLASSES,
    EXPECTED_ROWS,
    EXPECTED_SAMPLES,
    EXPECTED_SEEN_ROWS,
    EXPECTED_SOURCES,
    EXPECTED_SPEAKER_KEYS,
    EXPECTED_UNSEEN_ROWS,
    GATE_THRESHOLD,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    REQUIRED_WORDING,
    RF384_DILATIONS,
    RF384_KERNEL,
    RF384_LOOKBACK,
    RF384_MAX_RESIDUAL,
    SEEDS,
    SHARED_SHORT_SEEDS,
    ValidatedFinalOOD,
    _bootstrap,
    _detection_counts,
    _raw_gate_selection,
    _report_lines,
    _seed23_sensitivity,
    _seed_summary,
    _select_verdict,
    file_sha256,
    load_sealed_final_ood_rows,
    run_final_ood,
    validate_final_ood_lock,
    write_artifacts,
)
from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    COUNT_COLUMNS,
    metrics_from_counts,
)


ARTIFACT_ROLES = (
    "final_ood_manifest",
    "a11_cost",
    "a12_routing",
    "adaptive_model",
    "evaluate_adaptive",
    "analyze_context_gate",
    "data",
    "marblenet_dataset",
    "marblenet_features",
    "marblenet_model",
    "run_a14_final_ood",
    "test_run_a14_final_ood",
)


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return file_sha256(path).upper()


def _checkpoint_payload(seed: int) -> dict[str, object]:
    return {
        "seed": int(seed),
        "short_model_config": {
            "causal": True,
            "frame_output": True,
            "dilation_profile": "short",
        },
        "refinement_config": {
            "dilations": list(RF384_DILATIONS),
            "lookback_frames": RF384_LOOKBACK,
            "kernel_size": RF384_KERNEL,
            "max_residual": RF384_MAX_RESIDUAL,
        },
    }


def _write_checkpoint(path: Path, seed: int) -> str:
    torch.save(_checkpoint_payload(seed), path)
    return file_sha256(path).upper()


def _protocol_payload(root: Path) -> dict[str, object]:
    manifest = root / "manifest.tsv"
    manifest_hash = _write_bytes(manifest, b"sealed manifest placeholder\n")
    artifacts: list[dict[str, str]] = []
    for role in ARTIFACT_ROLES:
        if role == "final_ood_manifest":
            path = manifest
            digest = manifest_hash
        elif role == "run_a14_final_ood":
            path = root / "run_a14_final_ood.py"
            digest = _write_bytes(path, b"runner placeholder\n")
        elif role == "test_run_a14_final_ood":
            path = root / "test_run_a14_final_ood.py"
            digest = _write_bytes(path, b"test placeholder\n")
        else:
            path = root / "artifacts" / f"{role}.txt"
            digest = _write_bytes(path, f"{role}\n".encode("ascii"))
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "sha256": digest,
            }
        )

    checkpoints = []
    for seed in SEEDS:
        path = root / "checkpoints" / f"seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.append(
            {
                "role": f"checkpoint_seed{seed}",
                "seed": seed,
                "path": str(path),
                "sha256": _write_checkpoint(path, seed),
                "short_encoder_relation": (
                    "shared_frozen_short_encoder"
                    if seed in SHARED_SHORT_SEEDS
                    else "independently_trained_short_encoder"
                ),
            }
        )

    return {
        "version": PROTOCOL_VERSION,
        "status": PROTOCOL_STATUS,
        "date": "2026-09-19",
        "route": {
            "final_ood_access": "SEALED",
            "a14": "CONFIRMATION_ONLY",
            "a14_after_result": "A_CLAIM_FREEZE",
            "a15": "FORBIDDEN",
            "router_tuning_after_a14": "FORBIDDEN",
            "stabilizer_comparison_on_final_ood": "FORBIDDEN",
        },
        "candidate": {
            "name": "ShortShort Adaptive RF384Adaptive RF384",
            "short_model": "Short",
            "adaptive_model": "Adaptive RF384",
            "comparator": "AlwaysRefine RF384",
            "rf_span": 384,
            "dilations": list(RF384_DILATIONS),
            "lookback_frames": RF384_LOOKBACK,
            "kernel_size": RF384_KERNEL,
            "max_residual": RF384_MAX_RESIDUAL,
            "router": "raw causal confidence gate",
            "gate_rule": "abs(short_score - 0.5) <= 0.13",
            "threshold": GATE_THRESHOLD,
            "source_rank_stabilizer": "EXCLUDED",
            "unique_candidate": True,
        },
        "locks": {
            "artifacts": artifacts,
            "checkpoints": checkpoints,
        },
        "sealed_split": {
            "protocol_role": "test_untouched_final_ood",
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_hash,
            "rows": EXPECTED_ROWS,
            "sources": EXPECTED_SOURCES,
            "speaker_sensitivity_keys": EXPECTED_SPEAKER_KEYS,
            "noise_classes": EXPECTED_NOISE_CLASSES,
            "seen_rows": EXPECTED_SEEN_ROWS,
            "unseen_rows": EXPECTED_UNSEEN_ROWS,
            "samples": EXPECTED_SAMPLES,
            "causal_frames": EXPECTED_FRAMES,
        },
        "evaluation": {
            "primary_seed": 17,
            "checkpoint_seeds": list(SEEDS),
            "decision_threshold": 0.5,
            "score_chunk_frames": 2000,
            "causal_inference_start_frame": 0,
            "zero_cache_at_start": True,
            "valid_start_frame": 0,
            "development_activation_reference": 0.0505,
            "activation_warning": ACTIVATION_WARNING,
            "activation_failure": ACTIVATION_FAILURE,
            "bootstrap_repeats": 5000,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_seed_rules": BOOTSTRAP_SEED_RULES,
            "bootstrap_primary_unit": "source_cluster",
            "bootstrap_sensitivity_unit": "speaker_cluster",
            "cells": [list(cell) for cell in CELL_ORDER],
            "required_wording": REQUIRED_WORDING,
        },
        "gates": {
            "gate_1": {
                "utility": "U_A_minus_S = R_S - R_A",
                "strong_go_rule": (
                    "U > 0 and source-cluster bootstrap 95% lower CI > 0"
                ),
            },
            "gate_2": {
                "warning_boundary": ACTIVATION_WARNING,
            },
            "gate_3": {
                "cell_order": [list(cell) for cell in CELL_ORDER],
                "require_all_cells_positive": False,
                "taxonomy": {
                    "F2": "population value > 0 and selected value <= 0",
                    "F3": "selected value > 0 and activation > 0.15",
                },
            },
        },
        "verdicts": {
            "strong_go": {"activation_rule": "r_OOD < 0.15"},
            "no_go": {
                "aggregate_rule": (
                    "aggregate utility non-positive or (r_OOD > 0.20 and "
                    "source-cluster bootstrap 95% lower CI <= 0)"
                )
            },
        },
        "claim_freeze": {"action": "A14 -> A CLAIM FREEZE"},
    }


def _write_protocol(root: Path) -> Path:
    path = root / "protocol.json"
    path.write_text(
        json.dumps(_protocol_payload(root), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _counts(
    *,
    frames: int,
    selected: int,
    correction: int,
    harm: int,
    short_error: int,
    long_error: int,
) -> np.ndarray:
    values = {
        "frames": frames,
        "selected": selected,
        "correction": correction,
        "harm": harm,
        "short_error": short_error,
        "long_error": long_error,
        "gated_error": short_error - correction + harm,
    }
    return np.asarray([values[name] for name in COUNT_COLUMNS], dtype=np.int64)


class A14FinalOODTests(unittest.TestCase):
    def test_invalid_lock_fails_before_manifest_loader_or_output_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = _write_protocol(root)
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["sealed_split"]["manifest_sha256"] = "0" * 64
            protocol_path.write_text(
                json.dumps(protocol, indent=2) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "must_not_exist"
            with mock.patch(
                "reproductions.difficulty_adaptive_context."
                "run_a14_final_ood.load_sealed_final_ood_rows"
            ) as loader:
                with self.assertRaises(ValueError):
                    run_final_ood(
                        protocol_path=protocol_path,
                        output_dir=output_dir,
                        data_root=DEFAULT_DATA_ROOT,
                        device="cpu",
                    )
            loader.assert_not_called()
            self.assertFalse(output_dir.exists())

    def test_manifest_and_checkpoint_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = _write_protocol(root)
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

            manifest = Path(protocol["sealed_split"]["manifest_path"])
            manifest.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "lock hash mismatch"):
                validate_final_ood_lock(protocol_path)

            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            checkpoint = Path(protocol["locks"]["checkpoints"][0]["path"])
            checkpoint.write_bytes(b"tampered checkpoint\n")
            with self.assertRaisesRegex(ValueError, "lock hash mismatch"):
                validate_final_ood_lock(protocol_path)

    def test_capability_token_is_required_before_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = _write_protocol(root)
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            forged = ValidatedFinalOOD(
                protocol=protocol,
                protocol_path=protocol_path,
                lock_record={},
                _token=object(),
            )
            with mock.patch(
                "reproductions.difficulty_adaptive_context."
                "run_a14_final_ood.read_manifest"
            ) as reader:
                with self.assertRaisesRegex(RuntimeError, "capability"):
                    load_sealed_final_ood_rows(forged)
            reader.assert_not_called()

    def test_raw_gate_uses_frozen_inclusive_boundary(self) -> None:
        scores = np.asarray([0.36, 0.37, 0.5, 0.63, 0.64])
        self.assertEqual(
            _raw_gate_selection(scores).tolist(),
            [False, True, True, True, False],
        )

    def test_detection_counts_match_seed_accumulator_shape(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.8, 0.4, 0.9], dtype=np.float64)
        counts = _detection_counts(labels, scores)
        self.assertEqual(counts, (1, 1, 1, 1, 2, 2))
        self.assertEqual(np.asarray(counts).shape, (6,))

    def test_seed_summary_accepts_detection_mapping(self) -> None:
        source_counts = np.asarray(
            [
                _counts(
                    frames=10,
                    selected=2,
                    correction=2,
                    harm=0,
                    short_error=4,
                    long_error=2,
                )
            ],
            dtype=np.int64,
        )
        summary = _seed_summary(
            {
                "seed": 17,
                "evaluated_frames": 10,
                "source_counts": source_counts,
                "detection": {
                    "short": np.asarray([2, 2, 2, 4, 4, 6]),
                    "refined": np.asarray([2, 1, 2, 5, 4, 6]),
                    "adaptive": np.asarray([2, 2, 2, 4, 4, 6]),
                },
            }
        )
        self.assertEqual(summary["seed"], 17)
        self.assertEqual(summary["short"]["f1"], 0.5)
        self.assertEqual(summary["adaptive"]["f1"], 0.5)
        self.assertAlmostEqual(summary["utility"]["net_utility_per_frame"], 0.2)

    def test_utility_is_exactly_short_error_minus_adaptive_error(self) -> None:
        counts = _counts(
            frames=20,
            selected=8,
            correction=3,
            harm=1,
            short_error=10,
            long_error=8,
        )
        metrics = metrics_from_counts(counts)
        self.assertAlmostEqual(
            metrics["net_utility_per_frame"],
            0.10,
        )
        self.assertAlmostEqual(
            metrics["short_error_rate"] - metrics["long_error_rate"],
            0.10,
        )
        self.assertEqual(metrics["signed_utility"], 2)

    def test_bootstrap_units_and_cell_order_are_frozen(self) -> None:
        counts = np.asarray(
            [
                _counts(
                    frames=10,
                    selected=2,
                    correction=2,
                    harm=0,
                    short_error=4,
                    long_error=2,
                ),
                _counts(
                    frames=10,
                    selected=1,
                    correction=0,
                    harm=1,
                    short_error=4,
                    long_error=5,
                ),
            ]
        )
        bootstrap = _bootstrap(counts, repeats=20, seed=BOOTSTRAP_SEED)
        self.assertIn("net_utility_per_frame_ci95_low", bootstrap)
        self.assertEqual(
            CELL_ORDER,
            (
                ("seen", "-5"),
                ("seen", "0"),
                ("seen", "5"),
                ("seen", "10"),
                ("seen", "15"),
                ("seen", "20"),
                ("unseen", "-5"),
                ("unseen", "0"),
                ("unseen", "5"),
                ("unseen", "10"),
                ("unseen", "15"),
                ("unseen", "20"),
            ),
        )
        self.assertEqual(len(CELL_ORDER), 12)

    def test_seed23_sensitivity_does_not_create_n_equals_four(self) -> None:
        positive = _seed23_sensitivity(
            {"17": 0.1, "18": 0.2, "19": 0.3, "23": 0.4}
        )
        self.assertTrue(positive["seed23_direction_consistent"])
        self.assertTrue(positive["seed23_positive_direction"])
        self.assertAlmostEqual(
            positive["mean_shared_short_seeds_17_18_19"],
            0.2,
        )
        negative = _seed23_sensitivity(
            {"17": 0.1, "18": 0.2, "19": 0.3, "23": -0.4}
        )
        self.assertFalse(negative["seed23_direction_consistent"])
        self.assertFalse(negative["seed23_positive_direction"])

    def test_verdict_precedence_is_exact(self) -> None:
        strong = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.01,
            activation=0.1,
            catastrophic_domain_regression=(),
            seed23_positive_direction=True,
        )
        self.assertEqual(strong, "STRONG_GO")

        nonpositive = _select_verdict(
            aggregate_utility=0.0,
            aggregate_ci_low=0.1,
            activation=0.01,
            catastrophic_domain_regression=(),
            seed23_positive_direction=True,
        )
        self.assertEqual(nonpositive, "OOD_GENERALIZATION_NO_GO")

        activation_failure = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.0,
            activation=ACTIVATION_FAILURE + 0.001,
            catastrophic_domain_regression=(),
            seed23_positive_direction=True,
        )
        self.assertEqual(activation_failure, "OOD_GENERALIZATION_NO_GO")

        activation_boundary = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.01,
            activation=ACTIVATION_FAILURE,
            catastrophic_domain_regression=(),
            seed23_positive_direction=True,
        )
        self.assertEqual(activation_boundary, "CONDITIONAL_GO")

        crossed_ci = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.0,
            activation=0.1,
            catastrophic_domain_regression=(),
            seed23_positive_direction=True,
        )
        self.assertEqual(crossed_ci, "CONDITIONAL_GO")

        catastrophic = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.01,
            activation=0.1,
            catastrophic_domain_regression=("unseen",),
            seed23_positive_direction=True,
        )
        self.assertEqual(catastrophic, "CONDITIONAL_GO")

        seed23_disagrees = _select_verdict(
            aggregate_utility=0.1,
            aggregate_ci_low=0.01,
            activation=0.1,
            catastrophic_domain_regression=(),
            seed23_positive_direction=False,
        )
        self.assertEqual(seed23_disagrees, "CONDITIONAL_GO")

    def test_report_and_six_artifacts_contain_frozen_wording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "a14"
            result = {
                "verdict": {"status": "CONDITIONAL_GO"},
                "protocol": {"sha256": "ABC"},
                "candidate": {
                    "name": "ShortShort Adaptive RF384Adaptive RF384",
                    "gate_rule": "abs(short_score - 0.5) <= 0.13",
                },
                "gates": {
                    "gate_1_ood_accuracy_utility": {
                        "aggregate_utility": 0.1,
                        "ci95_lower": 0.01,
                        "delta_f1_adaptive_minus_short": 0.02,
                        "status": "STRONG_PASS",
                    },
                    "gate_2_ood_compute_behavior": {
                        "activation_rate": 0.08,
                        "development_reference": 0.0505,
                        "warning_boundary": 0.15,
                        "failure_boundary": 0.20,
                        "status": "PASS",
                    },
                    "gate_3_ood_failure_structure": {
                        "positive_cell_fraction": 1.0,
                        "positive_cells": 12,
                        "cells": 12,
                        "worst_cell_utility": 0.01,
                        "worst_cell": {"domain": "unseen", "snr_db": "-5"},
                        "activation_max": 0.1,
                        "activation_max_cell": {
                            "domain": "unseen",
                            "snr_db": "-5",
                        },
                        "catastrophic_domain_regression": [],
                        "taxonomy_counts": {
                            "F1_VALUE_SCARCITY": 0,
                            "F2_RANKING_FAILURE": 0,
                            "F3_BUDGET_CALIBRATION_DRIFT": 0,
                            "NO_FAILURE": 12,
                        },
                    },
                },
                "seed_sensitivity": {
                    "seed_net_utility_per_frame": {
                        "17": 0.1,
                        "18": 0.1,
                        "19": 0.1,
                        "23": 0.1,
                    },
                    "mean_shared_short_seeds_17_18_19": 0.1,
                    "independent_short_seed23": 0.1,
                    "seed23_direction_consistent": True,
                    "seed23_positive_direction": True,
                },
                "lock_record": {"protocol_sha256": "ABC"},
            }
            seed_results = {seed: {"seed": seed} for seed in SEEDS}
            with mock.patch(
                "reproductions.difficulty_adaptive_context."
                "run_a14_final_ood._per_cell_rows",
                return_value=[],
            ), mock.patch(
                "reproductions.difficulty_adaptive_context."
                "run_a14_final_ood._seed_summary",
                return_value={"short": {"f1": 0.1}, "adaptive": {"f1": 0.2}},
            ):
                outputs = write_artifacts(result, seed_results, output_dir)

            self.assertEqual(
                set(outputs),
                {
                    "json",
                    "per_cell_csv",
                    "per_seed_csv",
                    "report",
                    "lock_record",
                    "claim_freeze",
                },
            )
            report = outputs["report"].read_text(encoding="utf-8")
            self.assertIn(REQUIRED_WORDING, report)
            self.assertIn("A14 -> A CLAIM FREEZE", report)
            self.assertIn("FORBIDDEN", report)


if __name__ == "__main__":
    unittest.main()
