# -*- coding: utf-8 -*-
"""Focused checks for the frozen E2 RCCD runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from reproductions.difficulty_adaptive_context import run_e2


def _meta(
    *,
    sample_id: str,
    source_key: str,
    speaker_id: str,
    noise_name: str,
    noise_instance: tuple[str, int],
    snr_db: float,
) -> run_e2.ManifestMeta:
    return run_e2.ManifestMeta(
        sample_id=sample_id,
        source_key=source_key,
        speaker_id=speaker_id,
        noise_name=noise_name,
        noise_source=noise_instance[0],
        noise_start_sample=noise_instance[1],
        snr_db=float(snr_db),
        noise_instance=noise_instance,
        clean_duration_s=1.0,
        speech_duration_s=0.5,
        silence_duration_s=0.5,
    )


def _match_record(
    *,
    item_index: int,
    sample_id: str,
    source_key: str,
    speaker_id: str,
    noise_name: str,
    condition: str = "0",
    noise_instance: tuple[str, int] = ("noise-a", 0),
    snr_db: float = 5.0,
    mean_speech_state: float = 0.5,
    target_positions: tuple[int, ...] = (600, 700),
    anchor_positions: tuple[int, ...] = tuple(
        range(500, 701, 20)
    ),
    n_frames: int = 900,
) -> run_e2.MatchRecord:
    record = SimpleNamespace(
        item_index=int(item_index),
        source_key=str(source_key),
        speaker_id=str(speaker_id),
        noise_name=str(noise_name),
        condition=str(condition),
        n_frames=int(n_frames),
    )
    targets = np.asarray(target_positions, dtype=np.int64)
    anchors = np.asarray(anchor_positions, dtype=np.int64)
    return run_e2.MatchRecord(
        record=record,
        meta=_meta(
            sample_id=sample_id,
            source_key=source_key,
            speaker_id=speaker_id,
            noise_name=noise_name,
            noise_instance=noise_instance,
            snr_db=snr_db,
        ),
        target_positions=targets,
        target_speech_state=np.full(
            targets.size,
            float(mean_speech_state),
            dtype=np.float64,
        ),
        target_rms=np.ones(targets.size, dtype=np.float64),
        anchor_positions=anchors,
        anchor_speech_state=np.full(
            anchors.size,
            float(mean_speech_state),
            dtype=np.float64,
        ),
        anchor_rms=np.ones(anchors.size, dtype=np.float64),
        anchor_indices=np.arange(anchors.size, dtype=np.int64),
        representative_indices=np.arange(
            anchors.size,
            dtype=np.int64,
        ),
        mean_remote_speech_state=float(mean_speech_state),
        median_remote_rms=1.0,
    )


def _contrast(
    effect: float,
    *,
    low: float = 0.001,
    status: str = "PRIMARY_INTERPRETABLE",
) -> dict[str, object]:
    return {
        "status": status,
        "effect": float(effect),
        "ci95_low": float(low),
        "ci95_high": float(effect) + 0.1,
    }


def _decision_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    conditions = list(run_e2.REPLACEMENT_CONDITIONS)
    feasibility = pd.DataFrame(
        {
            "condition": conditions,
            "feasibility_status": [
                (
                    "PRIMARY_INTERPRETABLE"
                    if condition
                    in {
                        run_e2.C1_SAME_UTT_DIFFERENT_TIME,
                        run_e2.C3_SAME_NOISE_CLASS,
                        run_e2.C6_SPEECH_STATE_MISMATCH,
                        run_e2.C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
                    }
                    else "INCOMPARABLE"
                )
                for condition in conditions
            ],
        }
    )
    full_column = run_e2._score_column(run_e2.C0_FULL)
    frame = pd.DataFrame(
        {
            "embedded_short_score": [0.1, 0.9],
            "short_score": [0.1, 0.9],
            full_column: [0.2, 0.8],
            "full_score": [0.2, 0.8],
        }
    )
    contrasts = {
        "P1_UTTERANCE": _contrast(0.01, low=0.002),
        "P2_NOISE_INSTANCE": _contrast(0.0),
        "P3_NOISE_CLASS": _contrast(0.01, low=0.002),
        "P4_SNR": _contrast(0.0),
        "P5_SPEECH_STATE": _contrast(0.0),
        "P6_RESIDUAL_IDENTITY": _contrast(0.08, low=0.01),
    }
    event_lookup = {
        (
            run_e2.C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
            "S1_ONSET_NEAR",
        ): {
            "excess_over_matched_stable": 0.006,
            "excess_ci95_low": 0.001,
        },
        (
            run_e2.C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE,
            "S2_TRANSITION_NEAR",
        ): {
            "excess_over_matched_stable": 0.0,
            "excess_ci95_low": -0.001,
        },
    }
    loso = {
        name: {
            "sign_consistent": True,
            "all_loso_positive": True,
        }
        for name in (
            "P1_UTTERANCE",
            "P2_NOISE_INSTANCE",
            "P3_NOISE_CLASS",
            "P4_SNR",
            "P5_SPEECH_STATE",
            "P6_RESIDUAL_IDENTITY",
        )
    }
    donor_seed = {
        "contrasts": {
            name: {"positive_seed_count": 5}
            for name in (
                "P1_UTTERANCE",
                "P2_NOISE_INSTANCE",
                "P3_NOISE_CLASS",
                "P4_SNR",
                "P5_SPEECH_STATE",
                "P6_RESIDUAL_IDENTITY",
            )
        }
    }
    return (
        feasibility,
        frame,
        contrasts,
        event_lookup,
        loso,
        donor_seed,
    )


def _summary_payload() -> dict[str, object]:
    effect = {"estimate": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    payload: dict[str, object] = {
        "E2_STATUS": "BACKGROUND_REFERENCE_CONDITIONAL",
        "BASELINE_REPRODUCED": True,
        "E1_C4_REPLAY_REPRODUCED": True,
        "MATCHING_FEASIBILITY_BY_CONDITION": {
            condition: "PRIMARY_INTERPRETABLE"
            for condition in run_e2.REPLACEMENT_CONDITIONS
        },
        "ONSET_EFFECT": effect,
        "TRANSITION_EFFECT": effect,
        "OFFSET_EFFECT": effect,
        "ONSET_MINUS_OFFSET": effect,
        "SEEN_EFFECT": {},
        "UNSEEN_EFFECT": {},
        "R_SURVIVAL_BY_CONDITION": {},
        "SOURCE_LOSO_STABILITY": {},
        "DONOR_SEED_STABILITY": {},
        "E1_RECONCILIATION": {},
        "PROTOCOL_DEVIATIONS": [],
        "TRAINING_PERFORMED": False,
        "NEW_FINAL_OOD_TOUCHED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
    }
    for key in (
        "SAME_UTT_EFFECT",
        "SAME_NOISE_INSTANCE_EFFECT",
        "SAME_NOISE_CLASS_EFFECT",
        "WRONG_SNR_EFFECT",
        "DIFFERENT_CLASS_EFFECT",
        "SPEECH_STATE_MISMATCH_EFFECT",
        "BEST_METADATA_MATCHED_EFFECT",
        "P1_UTTERANCE_CONTRAST",
        "P2_NOISE_INSTANCE_CONTRAST",
        "P3_NOISE_CLASS_CONTRAST",
        "P4_SNR_CONTRAST",
        "P5_SPEECH_STATE_CONTRAST",
        "P6_RESIDUAL_IDENTITY_EFFECT",
    ):
        payload[key] = dict(effect)
    return payload


def _quality_rows(
    condition: str,
    *,
    snr: float = 0.0,
    state_mismatch: float = 0.0,
    rows: int = 20,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(rows):
        result.append(
            {
                "condition": condition,
                "target_item_index": index,
                "target_source_key": f"target-source-{index}",
                "donor_item_index": 1000 + index,
                "donor_source_key": f"donor-source-{index}",
                "donor_noise_instance": f"donor-instance-{index}",
                "matched_frames": 100,
                "snr_abs_difference": float(snr),
                "rms_abs_log_difference_mean": 0.01,
                "speech_state_abs_difference_mean": float(state_mismatch),
                "noise_class_match": True,
                "noise_instance_match": False,
                "source_overlap": False,
                "speaker_overlap": False,
                "different_utterance": True,
            }
        )
    return result


class FrozenContractTests(unittest.TestCase):
    def test_frozen_constants_and_required_summary_keys(self) -> None:
        self.assertEqual(
            run_e2.DONOR_SEEDS,
            (101, 211, 307, 401, 503),
        )
        self.assertEqual(run_e2.C6_MIN_SPEECH_STATE_MISMATCH, 0.25)
        self.assertEqual(run_e2.MATERIAL_CONTRAST_DELTA, 0.005)
        self.assertEqual(run_e2.BOOTSTRAP_REPEATS, 2_000)
        self.assertEqual(run_e2.E2_EXPECTED_NOISY_TEST_FRAMES, 240_565)
        self.assertEqual(run_e2.E2_EXPECTED_NOISY_TEST_RECORDS, 396)
        self.assertEqual(run_e2.E2_EXPECTED_NOISY_TEST_SOURCES, 96)
        self.assertEqual(run_e2.E2_EXPECTED_NOISY_TEST_SPEAKERS, 20)
        self.assertEqual(
            run_e2.EXPECTED_E1_RUNNER_SHA256,
            "A429DCBF8ADF4FF9A4C1601F75D941E623A171A81F6FA084B7D7FAFBA3265BB3",
        )
        self.assertEqual(
            run_e2.EXPECTED_E1_PROTOCOL_SHA256,
            "D61BB571BA60454AC0AF28491AEF278967B53D7764E58C8BBA797C1313C886E6",
        )
        self.assertEqual(run_e2.E1_C4_REFERENCE_DELTA, 0.032131332797710244)
        self.assertEqual(
            set(run_e2.REQUIRED_SUMMARY_KEYS),
            {
                "E2_STATUS",
                "BASELINE_REPRODUCED",
                "MATCHING_FEASIBILITY_BY_CONDITION",
                "SAME_UTT_EFFECT",
                "SAME_NOISE_INSTANCE_EFFECT",
                "SAME_NOISE_CLASS_EFFECT",
                "WRONG_SNR_EFFECT",
                "DIFFERENT_CLASS_EFFECT",
                "SPEECH_STATE_MISMATCH_EFFECT",
                "BEST_METADATA_MATCHED_EFFECT",
                "P1_UTTERANCE_CONTRAST",
                "P2_NOISE_INSTANCE_CONTRAST",
                "P3_NOISE_CLASS_CONTRAST",
                "P4_SNR_CONTRAST",
                "P5_SPEECH_STATE_CONTRAST",
                "P6_RESIDUAL_IDENTITY_EFFECT",
                "ONSET_EFFECT",
                "TRANSITION_EFFECT",
                "OFFSET_EFFECT",
                "ONSET_MINUS_OFFSET",
                "SEEN_EFFECT",
                "UNSEEN_EFFECT",
                "R_SURVIVAL_BY_CONDITION",
                "SOURCE_LOSO_STABILITY",
                "DONOR_SEED_STABILITY",
                "E1_RECONCILIATION",
                "PROTOCOL_DEVIATIONS",
                "TRAINING_PERFORMED",
                "NEW_FINAL_OOD_TOUCHED",
                "NEXT_EXPERIMENT_AUTHORIZED",
            },
        )

    def test_parser_exposes_only_frozen_execution_modes(self) -> None:
        parser = run_e2._build_parser()
        action = next(
            action
            for action in parser._actions
            if "--mode" in action.option_strings
        )
        self.assertEqual(
            tuple(action.choices),
            (
                "freeze_protocol",
                "feasibility",
                "baseline",
                "e1_c4_audit",
                "run",
                "verify",
            ),
        )


class MatchingRuleTests(unittest.TestCase):
    def test_c1_legal_mask_rejects_self_overlap_and_nonpast_anchors(self) -> None:
        target = _match_record(
            item_index=1,
            sample_id="target",
            source_key="source",
            speaker_id="speaker",
            noise_name="noise",
            target_positions=(600, 700),
            anchor_positions=(500, 504, 604, 600, 700),
        )
        mask = run_e2._same_utterance_legal_mask(
            target=target,
            donor=target,
            anchor_indices=target.anchor_indices,
        )
        expected = np.asarray(
            [
                [True, False, True, False, False],
                [True, True, False, True, False],
            ],
            dtype=bool,
        )
        self.assertTrue(np.array_equal(mask, expected))

    def test_c1_vectorized_legal_mask_matches_naive_reference(self) -> None:
        target_positions = (
            96,
            191,
            192,
            287,
            288,
            381,
            382,
            383,
            384,
            479,
            480,
            575,
            576,
            671,
            672,
            767,
            768,
            769,
        )
        anchor_positions = (
            96,
            191,
            192,
            287,
            288,
            381,
            382,
            383,
            384,
            479,
            480,
            575,
            576,
            671,
            672,
            767,
            768,
            769,
        )
        target = _match_record(
            item_index=1,
            sample_id="target",
            source_key="source",
            speaker_id="speaker",
            noise_name="noise",
            target_positions=target_positions,
            anchor_positions=anchor_positions,
        )

        expected = np.ones(
            (len(target_positions), len(anchor_positions)),
            dtype=bool,
        )
        for row, target_frame in enumerate(target_positions):
            original_positions = {
                int(target_frame) - int(tap)
                for tap in run_e2.REMOTE_WINDOW_TAPS
                if int(target_frame) >= int(tap)
            }
            for column, anchor in enumerate(anchor_positions):
                donor_positions = {
                    int(anchor) - int(tap)
                    for tap in run_e2.REMOTE_WINDOW_TAPS
                    if int(anchor) >= int(tap)
                }
                expected[row, column] = bool(
                    int(anchor) != int(target_frame)
                    and max(donor_positions) < int(target_frame)
                    and not donor_positions.intersection(
                        original_positions
                    )
                )

        mask = run_e2._same_utterance_legal_mask(
            target=target,
            donor=target,
            anchor_indices=target.anchor_indices,
        )
        self.assertTrue(np.array_equal(mask, expected))

    def test_anchor_cost_order_matches_lexsort_with_lazy_ties(self) -> None:
        costs = np.asarray(
            [2.0, 1.0, 1.0, 4.0, 1.0, 2.0, 3.0, np.inf, 3.0],
            dtype=np.float64,
        )
        finite = np.flatnonzero(np.isfinite(costs))

        def tie_breaker(index: int) -> int:
            return (
                1469598103934665603
                ^ ((index + 1) * 1099511628211)
            ) & ((1 << 64) - 1)

        expected = finite[
            np.lexsort(
                (
                    np.asarray(
                        [tie_breaker(int(index)) for index in finite],
                        dtype=np.uint64,
                    ),
                    costs[finite],
                )
            )
        ]
        actual = run_e2._ordered_finite_candidates(
            cost_row=costs,
            finite=finite,
            tie_breaker=tie_breaker,
        )
        self.assertTrue(np.array_equal(actual, expected))

    def test_c2_through_c7_hard_constraints(self) -> None:
        target = _match_record(
            item_index=1,
            sample_id="target",
            source_key="source-a",
            speaker_id="speaker-a",
            noise_name="noise-a",
            condition="condition-a",
            noise_instance=("recording-a", 100),
            snr_db=5.0,
            mean_speech_state=0.5,
        )
        different_source = {
            "source_key": "source-b",
            "speaker_id": "speaker-b",
        }
        same_instance = _match_record(
            item_index=2,
            sample_id="same-instance",
            noise_name="noise-a",
            condition="condition-a",
            noise_instance=("recording-a", 100),
            snr_db=5.0,
            **different_source,
        )
        same_class = _match_record(
            item_index=3,
            sample_id="same-class",
            noise_name="noise-a",
            condition="condition-a",
            noise_instance=("recording-b", 0),
            snr_db=5.0,
            **different_source,
        )
        wrong_snr = _match_record(
            item_index=4,
            sample_id="wrong-snr",
            noise_name="noise-a",
            condition="condition-b",
            noise_instance=("recording-b", 0),
            snr_db=20.0,
            **different_source,
        )
        different_class = _match_record(
            item_index=5,
            sample_id="different-class",
            noise_name="noise-b",
            condition="condition-a",
            noise_instance=("recording-c", 0),
            snr_db=5.0,
            **different_source,
        )
        state_mismatch = _match_record(
            item_index=6,
            sample_id="state-mismatch",
            noise_name="noise-a",
            condition="condition-a",
            noise_instance=("recording-b", 0),
            snr_db=5.0,
            mean_speech_state=0.8,
            **different_source,
        )
        too_little_state_mismatch = _match_record(
            item_index=7,
            sample_id="small-state-mismatch",
            noise_name="noise-a",
            condition="condition-a",
            noise_instance=("recording-b", 0),
            snr_db=5.0,
            mean_speech_state=0.74,
            **different_source,
        )

        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_instance,
                condition=run_e2.C2_SAME_NOISE_INSTANCE,
            )
        )
        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_class,
                condition=run_e2.C3_SAME_NOISE_CLASS,
            )
        )
        self.assertFalse(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_instance,
                condition=run_e2.C3_SAME_NOISE_CLASS,
            )
        )
        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=wrong_snr,
                condition=run_e2.C4_SAME_CLASS_WRONG_SNR,
            )
        )
        self.assertFalse(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_class,
                condition=run_e2.C4_SAME_CLASS_WRONG_SNR,
            )
        )
        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=different_class,
                condition=(
                    run_e2.C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR
                ),
            )
        )
        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=state_mismatch,
                condition=run_e2.C6_SPEECH_STATE_MISMATCH,
            )
        )
        self.assertFalse(
            run_e2._hard_candidate_ok(
                target=target,
                donor=too_little_state_mismatch,
                condition=run_e2.C6_SPEECH_STATE_MISMATCH,
            )
        )
        self.assertTrue(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_class,
                condition=(
                    run_e2.C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE
                ),
            )
        )
        self.assertFalse(
            run_e2._hard_candidate_ok(
                target=target,
                donor=same_instance,
                condition=(
                    run_e2.C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE
                ),
            )
        )

    def test_feasibility_gates_and_c4_wrong_snr_acceptance(self) -> None:
        def row(
            condition: str,
            *,
            snr: float,
            state: float = 0.0,
        ) -> dict[str, object]:
            return run_e2._feasibility_row(
                condition=condition,
                target_count=20,
                eligible_target_frames=2_000,
                eligible_sources={
                    f"target-source-{index}" for index in range(20)
                },
                quality=pd.DataFrame(
                    _quality_rows(
                        condition,
                        snr=snr,
                        state_mismatch=state,
                    )
                ),
            )

        wrong_snr = row(run_e2.C4_SAME_CLASS_WRONG_SNR, snr=15.0)
        self.assertEqual(
            wrong_snr["feasibility_status"],
            "PRIMARY_INTERPRETABLE",
        )
        self.assertNotIn(
            "intended same-SNR matching failed",
            str(wrong_snr["feasibility_reason"]),
        )

        same_snr_c4 = row(run_e2.C4_SAME_CLASS_WRONG_SNR, snr=0.0)
        self.assertEqual(
            same_snr_c4["feasibility_status"],
            "INCOMPARABLE",
        )
        self.assertIn(
            "mean SNR mismatch below 10 dB",
            str(same_snr_c4["feasibility_reason"]),
        )

        same_class = row(run_e2.C3_SAME_NOISE_CLASS, snr=0.5)
        self.assertEqual(
            same_class["feasibility_status"],
            "INCOMPARABLE",
        )
        self.assertIn(
            "intended same-SNR matching failed",
            str(same_class["feasibility_reason"]),
        )

        state_mismatch = row(
            run_e2.C6_SPEECH_STATE_MISMATCH,
            snr=0.0,
            state=0.24,
        )
        self.assertEqual(
            state_mismatch["feasibility_status"],
            "INCOMPARABLE",
        )
        self.assertIn(
            "speech-state mismatch below frozen minimum",
            str(state_mismatch["feasibility_reason"]),
        )


class OrderingTests(unittest.TestCase):
    def _feasibility_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "condition": list(run_e2.REPLACEMENT_CONDITIONS),
                "feasibility_status": [
                    (
                        "PRIMARY_INTERPRETABLE"
                        if index % 2 == 0
                        else "INCOMPARABLE"
                    )
                    for index in range(len(run_e2.REPLACEMENT_CONDITIONS))
                ],
            }
        )

    def test_feasibility_mode_works_before_protocol_freeze(self) -> None:
        feasibility = self._feasibility_table()
        with (
            mock.patch.object(
                run_e2,
                "_assert_metadata_stage_before_freeze",
            ),
            mock.patch.object(
                run_e2,
                "_validate_frozen_protocol",
                side_effect=AssertionError(
                    "feasibility must not require a frozen protocol"
                ),
            ),
            mock.patch.object(
                run_e2.e1,
                "_load_population",
                return_value=object(),
            ),
            mock.patch.object(
                run_e2,
                "_validate_population",
                return_value={},
            ),
            mock.patch.object(
                run_e2,
                "_manifest_meta_for_population",
                return_value={},
            ),
            mock.patch.object(
                run_e2,
                "_build_match_records",
                return_value={},
            ),
            mock.patch.object(
                run_e2,
                "build_match_plan",
                return_value=({}, []),
            ),
            mock.patch.object(
                run_e2,
                "build_feasibility",
                return_value=feasibility,
            ),
            mock.patch.object(run_e2, "_write_csv"),
        ):
            result = run_e2._run_feasibility_mode(SimpleNamespace())
        self.assertEqual(result, 0)

    def test_freeze_fails_without_metadata_feasibility(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=run_e2.OUTPUT_ROOT
        ) as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(
                    run_e2,
                    "_outcome_stage_paths",
                    return_value=(),
                ),
                mock.patch.object(
                    run_e2,
                    "PROTOCOL_PATH",
                    root / "protocol.json",
                ),
                mock.patch.object(
                    run_e2,
                    "PROTOCOL_HASH_PATH",
                    root / "protocol.sha256",
                ),
                mock.patch.object(
                    run_e2,
                    "RUNNER_HASH_PATH",
                    root / "runner.sha256",
                ),
                mock.patch.object(
                    run_e2,
                    "FEASIBILITY_PATH",
                    root / "missing-feasibility.csv",
                ),
                mock.patch.object(
                    run_e2,
                    "MATCH_QUALITY_PATH",
                    root / "missing-quality.csv",
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    run_e2._freeze_protocol()

    def test_freeze_records_metadata_hashes_and_statuses(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=run_e2.OUTPUT_ROOT
        ) as temporary:
            root = Path(temporary)
            feasibility_path = root / "e2_matching_feasibility.csv"
            quality_path = root / "e2_match_quality.csv"
            feasibility = self._feasibility_table()
            feasibility.to_csv(feasibility_path, index=False)
            pd.DataFrame({"condition": ["C2"]}).to_csv(
                quality_path,
                index=False,
            )
            protocol_path = root / "e2_protocol_freeze.json"
            with (
                mock.patch.object(
                    run_e2,
                    "_outcome_stage_paths",
                    return_value=(),
                ),
                mock.patch.object(
                    run_e2,
                    "PROTOCOL_PATH",
                    protocol_path,
                ),
                mock.patch.object(
                    run_e2,
                    "PROTOCOL_HASH_PATH",
                    root / "e2_protocol_sha256.txt",
                ),
                mock.patch.object(
                    run_e2,
                    "RUNNER_HASH_PATH",
                    root / "e2_runner_sha256.txt",
                ),
                mock.patch.object(
                    run_e2,
                    "FEASIBILITY_PATH",
                    feasibility_path,
                ),
                mock.patch.object(
                    run_e2,
                    "MATCH_QUALITY_PATH",
                    quality_path,
                ),
            ):
                result = run_e2._freeze_protocol()
                self.assertEqual(result, 0)
                payload = json.loads(
                    protocol_path.read_text(encoding="utf-8")
                )
            metadata = payload["metadata_feasibility"]
            self.assertEqual(
                metadata["condition_statuses"],
                run_e2._feasibility_status_map(feasibility),
            )
            artifacts = {
                item["path"]: item["sha256"]
                for item in metadata["artifacts"]
            }
            self.assertEqual(
                artifacts[
                    str(feasibility_path.relative_to(
                        run_e2.REPO_ROOT
                    )).replace("\\", "/")
                ],
                run_e2._sha256_file(feasibility_path),
            )
            self.assertEqual(
                artifacts[
                    str(quality_path.relative_to(
                        run_e2.REPO_ROOT
                    )).replace("\\", "/")
                ],
                run_e2._sha256_file(quality_path),
            )

    def test_outcome_modes_require_frozen_protocol(self) -> None:
        cases = (
            (
                run_e2._run_baseline_mode,
                {"args": SimpleNamespace(), "device": object()},
            ),
            (
                run_e2._run_e1_c4_audit_mode,
                {"args": SimpleNamespace(), "device": object()},
            ),
            (
                run_e2._run_formal_analysis,
                {"args": SimpleNamespace(), "device": object()},
            ),
            (
                run_e2._run_verify_mode,
                {"args": SimpleNamespace()},
            ),
        )
        for function, kwargs in cases:
            with (
                self.subTest(function=function.__name__),
                mock.patch.object(
                    run_e2,
                    "_validate_frozen_protocol",
                    side_effect=RuntimeError(
                        "frozen protocol is required"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "frozen protocol is required",
                ):
                    function(**kwargs)


class DecisionTests(unittest.TestCase):
    def test_decision_rule_precedence_and_all_statuses(self) -> None:
        (
            feasibility,
            frame,
            contrasts,
            event_lookup,
            loso,
            donor_seed,
        ) = _decision_inputs()

        supported = run_e2._decision_status(
            baseline={"passed": True},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=contrasts,
            event_lookup=event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            supported["E2_STATUS"],
            "BACKGROUND_REFERENCE_SUPPORTED",
        )
        self.assertTrue(all(supported[key] for key in ("B1", "B2", "B3", "B4", "B5")))

        conditional_contrasts = dict(contrasts)
        conditional_contrasts["P1_UTTERANCE"] = _contrast(
            0.006,
            low=-0.001,
        )
        conditional_contrasts["P3_NOISE_CLASS"] = _contrast(0.0)
        conditional_contrasts["P6_RESIDUAL_IDENTITY"] = _contrast(0.0)
        conditional = run_e2._decision_status(
            baseline={"passed": True},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=conditional_contrasts,
            event_lookup=event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            conditional["E2_STATUS"],
            "BACKGROUND_REFERENCE_CONDITIONAL",
        )

        residual_contrasts = dict(contrasts)
        residual_contrasts["P1_UTTERANCE"] = _contrast(0.0)
        residual_contrasts["P3_NOISE_CLASS"] = _contrast(0.0)
        residual_contrasts["P4_SNR"] = _contrast(0.0)
        residual_contrasts["P5_SPEECH_STATE"] = _contrast(0.0)
        residual_contrasts["P6_RESIDUAL_IDENTITY"] = _contrast(
            0.08,
            low=0.01,
        )
        residual = run_e2._decision_status(
            baseline={"passed": True},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=residual_contrasts,
            event_lookup=event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            residual["E2_STATUS"],
            "RESIDUAL_CONTEXT_IDENTITY_EFFECT",
        )

        speech_contrasts = dict(residual_contrasts)
        speech_contrasts["P5_SPEECH_STATE"] = _contrast(
            0.08,
            low=0.002,
        )
        speech_contrasts["P6_RESIDUAL_IDENTITY"] = _contrast(
            0.10,
            low=0.002,
        )
        speech = run_e2._decision_status(
            baseline={"passed": True},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=speech_contrasts,
            event_lookup=event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            speech["E2_STATUS"],
            "SPEECH_STATE_COMPATIBILITY_SUPPORTED",
        )

        unresolved_contrasts = {
            name: _contrast(0.0, low=-0.001)
            for name in contrasts
        }
        unresolved_event_lookup = {
            key: {
                **value,
                "excess_over_matched_stable": 0.0,
                "excess_ci95_low": -0.001,
            }
            for key, value in event_lookup.items()
        }
        unresolved = run_e2._decision_status(
            baseline={"passed": True},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=unresolved_contrasts,
            event_lookup=unresolved_event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            unresolved["E2_STATUS"],
            "COMPATIBILITY_UNRESOLVED",
        )

        invalid = run_e2._decision_status(
            baseline={"passed": False},
            e1_c4_replay={"passed": True},
            feasibility=feasibility,
            frame=frame,
            condition_stats={},
            contrasts=contrasts,
            event_lookup=event_lookup,
            loso=loso,
            donor_seed=donor_seed,
            e1_unchanged=True,
        )
        self.assertEqual(
            invalid["E2_STATUS"],
            "INCONCLUSIVE_OR_INVALID",
        )

    def test_claim_freeze_uses_exact_required_headings(self) -> None:
        summary = _summary_payload()
        classifications = {
            "R1": "SUPPORTED",
            "R2": "PARTIALLY_SUPPORTED",
            "R8": "NOT_SUPPORTED",
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "claim.md"
            with mock.patch.object(
                run_e2,
                "CLAIM_FREEZE_PATH",
                target,
            ):
                run_e2._write_claim_freeze(
                    summary=summary,
                    condition_statistics={},
                    contrasts={},
                    reconciliation_classifications=classifications,
                )
            lines = target.read_text(encoding="utf-8").splitlines()
        for heading in (
            "OBSERVATIONS",
            "SUPPORTED_INTERPRETATIONS",
            "PARTIALLY_SUPPORTED_INTERPRETATIONS",
            "ALTERNATIVE_EXPLANATIONS",
            "UNSUPPORTED_CLAIMS",
        ):
            self.assertIn(heading, lines)


class ArtifactVerifierTests(unittest.TestCase):
    def _materialize_outputs(
        self,
        root: Path,
    ) -> tuple[Path, Path]:
        protocol_path = root / "e2_protocol_freeze.json"
        protocol_path.write_text("{}\n", encoding="utf-8")
        for name in run_e2.REQUIRED_OUTPUT_FILES:
            path = root / name.rstrip("/")
            if name == "e2_figures/":
                path.mkdir(parents=True, exist_ok=True)
            elif name not in {
                "e2_final_summary.json",
                "e2_execution_manifest.json",
            }:
                path.write_text("{}\n", encoding="utf-8")
        summary_path = root / "e2_final_summary.json"
        summary_path.write_text(
            json.dumps(_summary_payload()),
            encoding="utf-8",
        )
        pd.DataFrame({"global_index": [0, 1]}).to_parquet(
            root / "e2_frame_results.parquet",
            index=False,
        )
        figures = root / "e2_figures"
        for name in run_e2.FIGURE_FILES:
            (figures / name).write_bytes(b"png")
        manifest_path = root / "e2_execution_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "protocol_id": run_e2.PROTOCOL_ID,
                    "protocol_sha256": run_e2._sha256_file(
                        protocol_path
                    ),
                    "runner_sha256": run_e2._sha256_file(
                        Path(run_e2.__file__)
                    ),
                    "training_performed": False,
                    "new_final_ood_touched": False,
                    "next_experiment_authorized": False,
                    "e1_artifacts": {
                        "before": {},
                        "after": {},
                    },
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )
        return protocol_path, figures

    def test_verifier_accepts_complete_outputs_and_rejects_figure_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path, figures = self._materialize_outputs(root)
            with (
                mock.patch.object(run_e2, "REPO_ROOT", root),
                mock.patch.object(run_e2, "OUTPUT_ROOT", root),
                mock.patch.object(
                    run_e2,
                    "PROTOCOL_PATH",
                    protocol_path,
                ),
                mock.patch.object(
                    run_e2,
                    "EXECUTION_MANIFEST_PATH",
                    root / "e2_execution_manifest.json",
                ),
                mock.patch.object(
                    run_e2,
                    "FINAL_SUMMARY_PATH",
                    root / "e2_final_summary.json",
                ),
                mock.patch.object(
                    run_e2,
                    "FRAME_RESULTS_PATH",
                    root / "e2_frame_results.parquet",
                ),
                mock.patch.object(
                    run_e2,
                    "FIGURES_ROOT",
                    figures,
                ),
                mock.patch.object(
                    run_e2,
                    "E1_RESULTS_ROOT",
                    root / "missing-e1",
                ),
                mock.patch.object(
                    run_e2,
                    "E2_EXPECTED_NOISY_TEST_FRAMES",
                    2,
                ),
            ):
                verified = run_e2._verify_required_artifacts()
                self.assertTrue(
                    verified["summary"]["E2_STATUS"]
                    in run_e2.ALLOWED_E2_STATUSES
                )
                (figures / "figure_6_extra.png").write_bytes(b"png")
                with self.assertRaises(ValueError):
                    run_e2._verify_required_artifacts()


if __name__ == "__main__":
    unittest.main()
