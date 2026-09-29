# -*- coding: utf-8 -*-
"""Unit checks for the frozen A-v2 / AE mechanism runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.run_ae_mechanism import (
    CELL_ORDER,
    PROTOCOL_SHA256,
    AeData,
    _ae4_pair_rows,
    _build_ae5_rows,
    _build_event_features,
    _classify_ae5,
    _feature_bin_mask,
    _horizon_arrays,
    _refuse_overwrite,
    _state_arrays,
    file_sha256,
)


def _scores(labels: np.ndarray, correct: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    return np.where(labels == correct, 0.9, 0.1).astype(np.float64)


def _synthetic_data(
    *,
    labels: np.ndarray,
    short_correct: np.ndarray,
    correctness: np.ndarray,
    selected: np.ndarray | None = None,
    segments: list[tuple[int, int]] | None = None,
) -> AeData:
    labels = np.asarray(labels, dtype=np.int64)
    short_correct = np.asarray(short_correct, dtype=bool)
    correctness = np.asarray(correctness, dtype=bool)
    if correctness.shape != (labels.size, 5):
        raise ValueError("synthetic correctness must have shape (frames, 5)")
    if selected is None:
        selected = np.zeros(labels.size, dtype=bool)
    source_key = np.asarray(["source-a"] * labels.size, dtype=str)
    speaker_ids = np.asarray(["speaker-a"] * labels.size, dtype=str)
    condition = np.asarray(["clean"] * labels.size, dtype=str)
    noise_name = np.asarray(["none"] * labels.size, dtype=str)
    test_mask = np.ones(labels.size, dtype=bool)
    group_masks = {group: test_mask.copy() for group in CELL_ORDER}
    spans = {
        span: {
            "full_adaptive_scores": _scores(labels, correctness[:, index])
        }
        for index, span in enumerate((64, 128, 256, 384, 512))
    }
    return AeData(
        protocol=None,  # type: ignore[arg-type]
        spans=spans,
        labels=labels,
        short_scores=_scores(labels, short_correct),
        selected=np.asarray(selected, dtype=bool),
        test_mask=test_mask,
        calibration_mask=np.zeros(labels.size, dtype=bool),
        source_key=source_key,
        speaker_ids=speaker_ids,
        condition=condition,
        noise_name=noise_name,
        source_order=("source-a",),
        speaker_order=("speaker-a",),
        cell_labels=np.asarray(["clean"] * labels.size, dtype=object),
        group_masks=group_masks,
        segments=(
            segments
            if segments is not None
            else [(0, labels.size)]
        ),
    )


class AeMechanismTests(unittest.TestCase):
    def test_protocol_hash_is_frozen(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "reproductions"
            / "difficulty_adaptive_context"
            / "ae_protocol_freeze.json"
        )
        self.assertEqual(file_sha256(path), PROTOCOL_SHA256)

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

    def test_ae1_states_follow_frozen_short_and_rf384_outcomes(self) -> None:
        labels = np.asarray([1, 1, 1, 0, 1, 1], dtype=np.int64)
        short_correct = np.asarray([True, False, False, True, True, False])
        correctness = np.tile(
            np.asarray([True, False, True, True, True, True]),
            (5, 1),
        ).T
        data = _synthetic_data(
            labels=labels,
            short_correct=short_correct,
            correctness=correctness,
        )

        states = _state_arrays(data)

        np.testing.assert_array_equal(
            states["short_sufficient"],
            np.asarray([True, False, False, True, True, False]),
        )
        np.testing.assert_array_equal(
            states["refinable"],
            np.asarray([False, False, True, False, False, True]),
        )
        np.testing.assert_array_equal(
            states["irreducible"],
            np.asarray([False, True, False, False, False, False]),
        )
        self.assertFalse(np.any(states["harm"]))

    def test_ae2_horizons_require_correct_suffix_for_stability(self) -> None:
        labels = np.asarray([1, 1, 1, 0, 1, 1], dtype=np.int64)
        correctness = np.asarray(
            [
                [True, True, True, True, True],
                [False, True, True, True, True],
                [False, False, True, True, True],
                [True, True, True, True, True],
                [False, False, True, True, False],
                [False, True, True, True, True],
            ],
            dtype=bool,
        )
        data = _synthetic_data(
            labels=labels,
            short_correct=np.zeros(labels.size, dtype=bool),
            correctness=correctness,
        )

        horizons = _horizon_arrays(data)

        np.testing.assert_array_equal(
            horizons["first_correct"],
            np.asarray([64, 128, 256, 64, 256, 128], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            horizons["stable_sufficient"],
            np.asarray([64, 128, 256, 64, -1, 128], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            horizons["unstable_correct"],
            np.asarray([False, False, False, False, True, False]),
        )
        self.assertTrue(horizons["non_monotone"][4])
        self.assertFalse(np.any(horizons["never_correct"]))

    def test_ae3_event_features_reset_at_utterance_boundaries(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 0, 1, 1], dtype=np.int64)
        correctness = np.zeros((labels.size, 5), dtype=bool)
        data = _synthetic_data(
            labels=labels,
            short_correct=np.zeros(labels.size, dtype=bool),
            correctness=correctness,
            selected=np.asarray([False, True, True, False, False, False, False]),
            segments=[(0, 4), (4, 7)],
        )

        features = _build_event_features(data)

        np.testing.assert_array_equal(
            features["onset_distance"],
            np.asarray([-2, -1, 0, 1, -1, 0, 1], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            features["speech_run_length"],
            np.asarray([0, 0, 1, 0, 0, 2, 2], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            features["uncertainty_persistence"],
            np.asarray([0, 2, 2, 0, 0, 0, 0], dtype=np.float64),
        )

    def test_ae3_distance_bins_use_absolute_distance(self) -> None:
        values = np.asarray([-30, -4, -1, 0, 1, 4, 25, 51], dtype=np.float64)

        np.testing.assert_array_equal(
            _feature_bin_mask("onset_distance", values, "0"),
            np.asarray([False, False, False, True, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            _feature_bin_mask("onset_distance", values, "2-4"),
            np.asarray([False, True, False, False, False, True, False, False]),
        )
        np.testing.assert_array_equal(
            _feature_bin_mask("onset_distance", values, "26-50"),
            np.asarray([True, False, False, False, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            _feature_bin_mask("onset_distance", values, ">50"),
            np.asarray([False, False, False, False, False, False, False, True]),
        )

    def test_ae4_pair_agreement_reports_sign_and_condition_directions(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
        data = _synthetic_data(
            labels=labels,
            short_correct=np.zeros(labels.size, dtype=bool),
            correctness=np.zeros((labels.size, 5), dtype=bool),
        )
        seed_values = {
            17: {"value": np.asarray([1, 1, -1, 0], dtype=np.int8)},
            18: {"value": np.asarray([1, -1, -1, 0], dtype=np.int8)},
        }

        rows = _ae4_pair_rows(data, seed_values=seed_values)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["comparison_role"],
            "shared_short_encoder_checkpoint_variation",
        )
        self.assertAlmostEqual(rows[0]["sign_agreement"], 0.75)
        self.assertAlmostEqual(
            rows[0]["condition_level_direction_agreement"],
            0.0,
        )

    def test_ae5_uses_nonempty_seed_rows_with_pair_columns(self) -> None:
        ae0_payload = {
            "groups": {
                "all": {
                    "point": {
                        "availability_net": 0.1,
                        "observability_auroc": 0.7,
                        "gate_utility": 0.05,
                        "activation_rate": 0.1,
                    }
                }
            }
        }
        ae1_rows = [
            {
                "group": "all",
                "availability_net": 0.1,
                "gate_utility": 0.05,
                "activation_rate": 0.1,
                "state_rate_refinable": 0.2,
                "state_rate_refinement_harm": 0.1,
                "state_rate_rf384_irreducible": 0.3,
            }
        ]
        ae2_rows = [
            {
                "group": "all",
                "stable_sufficient_long_rate": 0.1,
                "mean_stable_sufficient_span": 128.0,
                "non_monotone_rate": 0.2,
            }
        ]
        ae3_rows = [
            {
                "feature": "uncertainty_persistence",
                "group": "all",
                "availability_net": 0.1,
            }
        ]
        ae4_rows = [
            {
                "left_seed": "",
                "right_seed": "",
                "group": "all",
                "role": "shared_short_encoder_checkpoint",
                "availability_net": 0.2,
            }
        ]
        ae4_pairs = [
            {
                "left_seed": "17",
                "right_seed": "18",
                "comparison_role": "shared_short_encoder_checkpoint_variation",
                "sign_agreement": 0.75,
                "condition_level_direction_agreement": 0.5,
            }
        ]

        rows = _build_ae5_rows(
            ae0_payload=ae0_payload,
            ae1_rows=ae1_rows,
            ae2_rows=ae2_rows,
            ae3_rows=ae3_rows,
            ae4_rows=ae4_rows,
            ae4_pairs=ae4_pairs,
        )

        all_row = next(row for row in rows if row["group"] == "all")
        self.assertAlmostEqual(all_row["checkpoint_availability_net"], 0.2)
        self.assertAlmostEqual(all_row["checkpoint_value_agreement"], 0.75)
        self.assertEqual(all_row["classification"], "REPLICATED_VALUE")

    def test_ae5_classification_precedence(self) -> None:
        common = {
            "observability_auroc": 0.7,
            "gate_utility": 0.1,
            "stable_long_rate": 0.1,
            "checkpoint_agreement": 0.6,
            "replication_agreement": 0.6,
        }

        self.assertEqual(
            _classify_ae5(availability_net=0.0, **common)[0],
            "VALUE_SCARCITY",
        )
        self.assertEqual(
            _classify_ae5(
                availability_net=0.1,
                **{**common, "observability_auroc": 0.5},
            )[0],
            "OBSERVABILITY_LIMITED",
        )
        self.assertEqual(
            _classify_ae5(
                availability_net=0.1,
                **{**common, "gate_utility": 0.0},
            )[0],
            "ACTIONABILITY_LIMITED",
        )
        self.assertEqual(
            _classify_ae5(
                availability_net=0.1,
                **{**common, "stable_long_rate": 0.5},
            )[0],
            "LONG_HORIZON_DEMAND",
        )
        self.assertEqual(
            _classify_ae5(
                availability_net=0.1,
                **{**common, "checkpoint_agreement": 0.4},
            )[0],
            "MODEL_RELATIVE_VALUE",
        )
        self.assertEqual(
            _classify_ae5(availability_net=0.1, **common)[0],
            "REPLICATED_VALUE",
        )
        self.assertEqual(
            _classify_ae5(
                availability_net=0.1,
                **{
                    **common,
                    "checkpoint_agreement": 0.5,
                    "replication_agreement": 0.5,
                },
            )[0],
            "TEMPORAL_SPAN_NOT_LIMITING",
        )


if __name__ == "__main__":
    unittest.main()
