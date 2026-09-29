# -*- coding: utf-8 -*-
"""Focused unit checks for the frozen E1 intervention runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

from reproductions.difficulty_adaptive_context import run_e1


def _dummy_item(sample_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        audio_path=Path("dummy") / f"{sample_id}.wav",
    )


class _ChunkSensitiveClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(value.shape[-1]))
        return value + value.mean(dim=-1, keepdim=True)


class _PointwiseClassifier(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2.0 - 0.5


def _record(
    *,
    item_index: int,
    start: int,
    labels: np.ndarray,
    sample_id: str,
    source_key: str,
    speaker_id: str,
    noise_name: str = "clean",
    condition: str = "clean",
    is_clean: bool = True,
    n_frames: int | None = None,
) -> run_e1.Record:
    labels = np.asarray(labels, dtype=np.int64)
    retained = int(labels.size)
    short_scores = np.where(labels == 1, 0.9, 0.1).astype(np.float64)
    full_scores = short_scores.copy()
    event_features = {
        "onset_distance": np.arange(retained, dtype=np.float64),
        "offset_distance": np.arange(retained, dtype=np.float64) + 1.0,
        "posterior_transition_distance": np.arange(
            retained,
            dtype=np.float64,
        ),
        "uncertainty_persistence": np.ones(retained, dtype=np.float64),
    }
    return run_e1.Record(
        item_index=int(item_index),
        item=_dummy_item(sample_id),
        start=int(start),
        end=int(start) + retained,
        n_frames=int(n_frames if n_frames is not None else retained + 382),
        labels=labels,
        short_scores=short_scores,
        full_scores=full_scores,
        frozen_full_scores=full_scores.copy(),
        source_key=str(source_key),
        speaker_id=str(speaker_id),
        noise_name=str(noise_name),
        condition=str(condition),
        is_clean=bool(is_clean),
        event_features=event_features,
    )


def _frame_rows_for_schema_test() -> list[dict[str, object]]:
    record = _record(
        item_index=7,
        start=0,
        labels=np.asarray([0, 1, 1], dtype=np.int64),
        sample_id="schema",
        source_key="source-1",
        speaker_id="speaker-1",
    )
    record = run_e1.Record(
        **{
            **record.__dict__,
            "short_scores": np.asarray([0.1, 0.9, 0.0], dtype=np.float64),
            "full_scores": np.asarray([0.1, 0.9, 0.9], dtype=np.float64),
            "frozen_full_scores": np.asarray(
                [0.1, 0.9, 0.9],
                dtype=np.float64,
            ),
        }
    )
    population = run_e1.Population(
        records=[record],
        labels=record.labels.copy(),
        short_scores=record.short_scores.copy(),
        full_scores=record.full_scores.copy(),
        frozen_full_scores=record.frozen_full_scores.copy(),
        test_mask=np.ones(record.retained_frames, dtype=bool),
        source_key=np.full(
            record.retained_frames,
            record.source_key,
            dtype=object,
        ),
        speaker_ids=np.full(
            record.retained_frames,
            record.speaker_id,
            dtype=object,
        ),
        noise_name=np.full(
            record.retained_frames,
            record.noise_name,
            dtype=object,
        ),
        condition=np.full(
            record.retained_frames,
            record.condition,
            dtype=object,
        ),
        selected=np.zeros(record.retained_frames, dtype=bool),
    )
    targets = np.asarray(
        [
            run_e1.VALID_START,
            run_e1.VALID_START + 1,
            run_e1.VALID_START + 2,
        ],
        dtype=np.int64,
    )
    full = record.frozen_full_scores.copy()
    scores: dict[str, np.ndarray] = {
        "embedded_short": record.short_scores.copy(),
        "full": full,
        "C2_ZERO": full.copy(),
    }
    for seed in run_e1.INTERVENTION_SEEDS:
        scores[f"C3_{seed}"] = full.copy()
        scores[f"C4_{seed}"] = full.copy()
        scores[f"C5_{seed}"] = full.copy()
    return run_e1._frame_rows_from_scores(
        population=population,
        record=record,
        target_positions=targets,
        scores=scores,
    )


def _stats(mean: float, low: float) -> dict[str, object]:
    return {
        "mean_delta_logloss": float(mean),
        "delta_logloss": {"ci95_low": float(low)},
    }


def _survival(
    *,
    drop: float = 0.10,
    source_min: float = 0.03,
    domain_min: float = 0.03,
) -> dict[str, object]:
    return {
        "survival_drop": float(drop),
        "leave_one_source_out": {"min": float(source_min)},
        "leave_one_domain_out": {"min": float(domain_min)},
    }


def _seed_summary(
    *,
    c3_positive: int,
    c4_positive: int,
) -> list[dict[str, object]]:
    return [
        {
            "condition": "C3_REMOTE_PERMUTE",
            "row_type": "seed_summary",
            "positive_seed_count": int(c3_positive),
        },
        {
            "condition": "C4_REMOTE_MATCHED_REPLACE",
            "row_type": "seed_summary",
            "positive_seed_count": int(c4_positive),
        },
    ]


def _required_summary_payload() -> dict[str, object]:
    return {
        "E1_STATUS": "REMOTE_INFORMATION_CONDITIONAL",
        "PRIMARY_QUESTION": (
            "Does RF384 benefit depend on correctly related remote "
            "temporal information?"
        ),
        "FULL_BASELINE_REPRODUCED": True,
        "REMOTE_PERMUTE_DELTA_LOGLOSS": 0.01,
        "REMOTE_PERMUTE_CI95": [-0.01, 0.03],
        "MATCHED_REPLACE_DELTA_LOGLOSS": 0.02,
        "MATCHED_REPLACE_CI95": [-0.02, 0.04],
        "ZERO_DELTA_LOGLOSS": 0.05,
        "LOCAL_CONTROL_DELTA_LOGLOSS": 0.04,
        "ORIGINAL_R_PREVALENCE": 0.1,
        "R_SURVIVAL_FULL": 1.0,
        "R_SURVIVAL_PERMUTE": 0.8,
        "R_SURVIVAL_MATCHED": 0.82,
        "INTERVENTION_SEED_AGREEMENT": "C3 4/5 positive; C4 4/5 positive",
        "SEEN_EFFECT": 0.01,
        "UNSEEN_EFFECT": 0.02,
        "TRANSITION_NEAR_EFFECT": 0.03,
        "TRANSITION_FAR_EFFECT": 0.005,
        "A9_AE2_RECONCILIATION": "T4",
        "PROTOCOL_DEVIATIONS": [],
        "NEW_FINAL_OOD_TOUCHED": False,
        "TRAINING_PERFORMED": False,
        "NEXT_EXPERIMENT_AUTHORIZED": False,
        "PROTOCOL_VERSION": run_e1.PROTOCOL_ID,
        "V1_INVALID_RETAINED": True,
        "SCIENTIFIC_PROTOCOL_CHANGED": False,
        "IMPLEMENTATION_CORRECTION": True,
        "INTERVENTION_EXECUTED_BEFORE_V2_FREEZE": False,
        "AUTHORIZATION_DISCLOSURE": run_e1.AUTHORIZATION_DISCLOSURE,
    }


class V2ScoringPathCorrectionTests(unittest.TestCase):
    def test_a_chunked_classifier_logits_reproduce_reference_semantics(self) -> None:
        classifier = _ChunkSensitiveClassifier()
        model = SimpleNamespace(
            short_model=SimpleNamespace(classifier=classifier)
        )
        encoded = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
        actual = run_e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=2,
        )
        expected = torch.cat(
            [
                encoded[..., 0:2]
                + encoded[..., 0:2].mean(dim=-1, keepdim=True),
                encoded[..., 2:4]
                + encoded[..., 2:4].mean(dim=-1, keepdim=True),
                encoded[..., 4:5]
                + encoded[..., 4:5].mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        full = encoded + encoded.mean(dim=-1, keepdim=True)
        self.assertEqual(classifier.calls, [2, 2, 1])
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, full))

    def test_b_embedded_short_output_is_unchanged(self) -> None:
        classifier = _PointwiseClassifier()
        model = SimpleNamespace(
            short_model=SimpleNamespace(classifier=classifier)
        )
        encoded = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        direct = classifier(encoded)
        chunked = run_e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=2,
        )
        self.assertTrue(torch.equal(chunked, direct))

    def test_c_manual_refinement_arithmetic_is_unchanged(self) -> None:
        branch_sum = torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]],
            dtype=torch.float32,
        )
        short_logits = torch.tensor(
            [[[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]]],
            dtype=torch.float32,
        )
        time_index = torch.tensor([0, 1, 2], dtype=torch.long)
        weight = torch.tensor(
            [[[1.5], [-0.5]], [[-0.25], [0.75]]],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.1, -0.2], dtype=torch.float32)
        model = SimpleNamespace(
            refinement=SimpleNamespace(
                activation=torch.nn.Identity(),
                output=SimpleNamespace(weight=weight, bias=bias),
                config=SimpleNamespace(max_residual=0.25),
            )
        )
        actual = run_e1._score_branch_sum(
            branch_sum,
            short_logits,
            time_index,
            model,
        )
        raw = branch_sum @ weight[:, :, 0].T + bias
        residual = 0.25 * torch.tanh(raw)
        current = short_logits[0, :, time_index].transpose(0, 1)
        expected = torch.softmax(current + residual, dim=-1)[:, 1]
        self.assertTrue(torch.equal(actual, expected))

    def test_d_no_future_frame_access_is_introduced(self) -> None:
        encoded = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
        branch = SimpleNamespace(kernel_size=3, dilation=2)
        target = torch.tensor([4], dtype=torch.long)
        windows = run_e1._gather_branch_windows(encoded, target, branch)
        expected = torch.tensor([[[0.0, 2.0, 4.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(windows, expected))

    def test_e_chunk_boundaries_do_not_alter_target_alignment(self) -> None:
        classifier = _PointwiseClassifier()
        model = SimpleNamespace(
            short_model=SimpleNamespace(classifier=classifier)
        )
        encoded = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
        chunked = run_e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=2,
        )
        direct = classifier(encoded)
        target = torch.tensor([1, 2, 4, 5], dtype=torch.long)
        self.assertTrue(
            torch.equal(
                chunked[0, :, target],
                direct[0, :, target],
            )
        )
        self.assertTrue(torch.equal(chunked, direct))

    def test_f_batch_and_source_isolation_remain_intact(self) -> None:
        original = torch.arange(
            2 * 1 * (len(run_e1.REMOTE_TAPS) + 1),
            dtype=torch.float32,
        ).reshape(2, 1, len(run_e1.REMOTE_TAPS) + 1)
        donor = original + 100.0
        targets = np.asarray([382, 384], dtype=np.int64)
        modified = run_e1._replace_remote_windows(
            original,
            donor,
            targets,
        )
        self.assertTrue(
            torch.equal(modified[0, :, 0], original[0, :, 0])
        )
        self.assertTrue(
            torch.equal(modified[0, :, 1:4], donor[0, :, 1:4])
        )
        self.assertTrue(
            torch.equal(modified[1, :, 0:4], donor[1, :, 0:4])
        )
        self.assertTrue(
            torch.equal(modified[1, :, 4], original[1, :, 4])
        )
        self.assertTrue(
            torch.equal(modified[0, :, 4], original[0, :, 4])
        )

    def test_g_correction_affects_classifier_evaluation_semantics_only(self) -> None:
        classifier = _ChunkSensitiveClassifier()
        model = SimpleNamespace(
            short_model=SimpleNamespace(classifier=classifier)
        )
        encoded = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
        before = encoded.clone()
        actual = run_e1._short_logits_for_encoded(
            encoded,
            model,
            chunk_frames=2,
        )
        self.assertTrue(torch.equal(encoded, before))
        self.assertEqual(classifier.calls, [2, 2, 1])
        self.assertEqual(actual.shape, encoded.shape)
        self.assertFalse(
            torch.equal(
                actual,
                encoded + encoded.mean(dim=-1, keepdim=True),
            )
        )


class RemoteTemporalPrimitiveTests(unittest.TestCase):
    def test_remote_window_order_and_literal_branch_mapping(self) -> None:
        self.assertEqual(
            run_e1.REMOTE_WINDOW_TAPS,
            (384, 288, 192, 96),
        )
        encoded = torch.arange(512, dtype=torch.float32).reshape(1, 1, -1)
        branch = SimpleNamespace(kernel_size=5, dilation=96)
        windows = run_e1._gather_branch_windows(
            encoded,
            torch.as_tensor([400], dtype=torch.long),
            branch,
        )
        self.assertEqual(
            windows[0, 0].tolist(),
            [16.0, 112.0, 208.0, 304.0, 400.0],
        )

    def test_remote_availability_boundaries(self) -> None:
        availability = run_e1._remote_availability(
            np.asarray([382, 383, 384], dtype=np.int64),
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [
                [False, True, True, True],
                [False, True, True, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(availability, expected))

    def test_remote_source_map_is_deterministic_and_fixed_point_free(self) -> None:
        record = _record(
            item_index=1,
            start=0,
            labels=np.ones(3, dtype=np.int64),
            sample_id="map",
            source_key="s1",
            speaker_id="p1",
        )
        targets = np.asarray([382, 383, 384, 500], dtype=np.int64)
        first = run_e1._remote_source_map_for_targets(
            seed=101,
            record=record,
            target_local_indices=targets,
        )
        second = run_e1._remote_source_map_for_targets(
            seed=101,
            record=record,
            target_local_indices=targets,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first[0, 0], 0)
        self.assertEqual(first[1, 0], 0)
        for row, target in enumerate(targets):
            available = [
                index
                for index, tap in enumerate(run_e1.REMOTE_WINDOW_TAPS)
                if int(target) >= tap
            ]
            for destination in available:
                self.assertIn(int(first[row, destination]), available)
                self.assertNotEqual(
                    int(first[row, destination]),
                    int(destination),
                )

    def test_derangement_is_deterministic_and_has_no_fixed_points(self) -> None:
        for size in (2, 3, 4, 7):
            first = run_e1._derangement(
                seed=211,
                kind="test",
                sample_id=f"size-{size}",
                target_index=5,
                size=size,
            )
            second = run_e1._derangement(
                seed=211,
                kind="test",
                sample_id=f"size-{size}",
                target_index=5,
                size=size,
            )
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(
                np.any(first == np.arange(size, dtype=np.int64))
            )

    def test_donor_choice_excludes_target_and_prefers_different_speaker(self) -> None:
        target = _record(
            item_index=1,
            start=0,
            labels=np.asarray([0, 1, 1], dtype=np.int64),
            sample_id="target",
            source_key="s1",
            speaker_id="p1",
            noise_name="SSN_noise",
            condition="0",
            is_clean=False,
        )
        same_speaker = _record(
            item_index=2,
            start=3,
            labels=np.asarray([0, 1, 1], dtype=np.int64),
            sample_id="same-speaker",
            source_key="s1",
            speaker_id="p1",
            noise_name="SSN_noise",
            condition="0",
            is_clean=False,
        )
        different_speaker = _record(
            item_index=3,
            start=6,
            labels=np.asarray([0, 1, 1], dtype=np.int64),
            sample_id="different-speaker",
            source_key="s2",
            speaker_id="p2",
            noise_name="SSN_noise",
            condition="0",
            is_clean=False,
        )
        donor, quality = run_e1._donor_choice(
            target=target,
            candidates=[same_speaker, different_speaker],
            seed=101,
        )
        self.assertEqual(donor.item_index, different_speaker.item_index)
        self.assertTrue(bool(quality["different_utterance"]))
        self.assertTrue(bool(quality["different_speaker"]))
        self.assertTrue(bool(quality["different_source"]))

        records = [target, same_speaker, different_speaker]
        selections, quality_rows = run_e1._build_seed_donors(
            records,
            np.ones(9, dtype=bool),
        )
        second_selections, _ = run_e1._build_seed_donors(
            records,
            np.ones(9, dtype=bool),
        )
        self.assertEqual(selections, second_selections)
        for seed in run_e1.INTERVENTION_SEEDS:
            for record in records:
                donor_index = selections[seed][record.item_index]
                self.assertNotEqual(donor_index, record.item_index)
        self.assertEqual(len(quality_rows), len(records) * len(run_e1.INTERVENTION_SEEDS))
        self.assertTrue(
            all(
                bool(row["different_utterance"])
                and bool(row["target_in_frozen_test_split"])
                and bool(row["donor_in_frozen_test_split"])
                for row in quality_rows
            )
        )

    def test_remote_helpers_preserve_current_and_unavailable_slots(self) -> None:
        original = torch.arange(
            2 * 3 * (len(run_e1.REMOTE_TAPS) + 1),
            dtype=torch.float32,
        ).reshape(2, 3, len(run_e1.REMOTE_TAPS) + 1)
        targets = np.asarray([382, 384], dtype=np.int64)
        zeroed = run_e1._zero_remote_windows(original, targets)
        self.assertTrue(
            torch.equal(
                zeroed[:, :, run_e1.CURRENT_WINDOW_SLOT],
                original[:, :, run_e1.CURRENT_WINDOW_SLOT],
            )
        )
        self.assertTrue(
            torch.count_nonzero(
                zeroed[:, :, run_e1.REMOTE_WINDOW_SLOTS]
            ).item()
            == 0
        )

        record = _record(
            item_index=1,
            start=0,
            labels=np.ones(3, dtype=np.int64),
            sample_id="slots",
            source_key="s1",
            speaker_id="p1",
        )
        source_map = run_e1._remote_source_map_for_targets(
            seed=101,
            record=record,
            target_local_indices=targets,
        )
        permuted = run_e1._permute_remote_windows(
            original,
            source_map,
            targets,
        )
        self.assertTrue(
            torch.equal(
                permuted[:, :, run_e1.CURRENT_WINDOW_SLOT],
                original[:, :, run_e1.CURRENT_WINDOW_SLOT],
            )
        )
        self.assertTrue(
            torch.equal(
                permuted[0, :, 0],
                original[0, :, 0],
            )
        )

        donor = torch.ones_like(original)
        replaced = run_e1._replace_remote_windows(
            original,
            donor,
            targets,
        )
        self.assertTrue(
            torch.equal(
                replaced[:, :, run_e1.CURRENT_WINDOW_SLOT],
                original[:, :, run_e1.CURRENT_WINDOW_SLOT],
            )
        )
        self.assertTrue(torch.equal(replaced[0, :, 0], original[0, :, 0]))
        self.assertTrue(torch.equal(replaced[0, :, 1:4], donor[0, :, 1:4]))
        self.assertTrue(torch.equal(replaced[1, :, 0:4], donor[1, :, 0:4]))

    def test_local_source_maps_change_only_noncurrent_taps(self) -> None:
        record = _record(
            item_index=1,
            start=0,
            labels=np.ones(4, dtype=np.int64),
            sample_id="local",
            source_key="s1",
            speaker_id="p1",
        )
        targets = np.asarray([382, 383, 384, 500], dtype=np.int64)
        first = run_e1._local_source_maps_for_targets(
            seed=101,
            record=record,
            target_local_indices=targets,
        )
        second = run_e1._local_source_maps_for_targets(
            seed=101,
            record=record,
            target_local_indices=targets,
        )
        self.assertTrue(np.array_equal(first, second))
        allowed = set(run_e1.LOCAL_TAPS_NONCURRENT)
        for row, target in enumerate(targets):
            for index, original_tap in enumerate(
                run_e1.LOCAL_TAPS_NONCURRENT
            ):
                mapped = int(first[row, index])
                self.assertIn(mapped, allowed)
                self.assertNotEqual(mapped, 0)
                if int(target) >= original_tap:
                    self.assertNotEqual(mapped, original_tap)


class AnalysisAndDecisionTests(unittest.TestCase):
    def test_frame_rows_include_c2_score_alias_and_seen_group(self) -> None:
        rows = _frame_rows_for_schema_test()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["seen_group"], "clean")
        self.assertIn("score_C2_REMOTE_ZERO", rows[0])
        self.assertEqual(
            rows[0]["score_C2_REMOTE_ZERO"],
            rows[0]["C2_REMOTE_ZERO"],
        )
        frame = pd.DataFrame(rows)
        predicted = run_e1._predict_mean_realization(
            frame,
            "C2_REMOTE_ZERO",
        )
        self.assertTrue(
            np.allclose(
                predicted,
                frame["C2_REMOTE_ZERO"].to_numpy(dtype=np.float64),
            )
        )

    def test_synthetic_frame_reaches_all_analysis_tables(self) -> None:
        frame = pd.DataFrame(_frame_rows_for_schema_test())
        tables = run_e1._collect_analysis_tables(frame)
        self.assertIn("decision", tables)
        self.assertIn(
            tables["decision"]["E1_STATUS"],
            {
                "REMOTE_INFORMATION_SUPPORTED",
                "REMOTE_INFORMATION_CONDITIONAL",
                "REMOTE_INFORMATION_NOT_SUPPORTED",
                "INCONCLUSIVE_OR_INVALID",
            },
        )
        self.assertTrue(tables["intervention_rows"])
        self.assertTrue(tables["taxonomy_rows"])
        self.assertTrue(tables["stratified_rows"])

    def test_decision_rule_precedence(self) -> None:
        survival = {
            "C3_REMOTE_PERMUTE": _survival(),
            "C4_REMOTE_MATCHED_REPLACE": _survival(),
        }
        supported = run_e1._decision_rules_status(
            {
                "C3_REMOTE_PERMUTE": _stats(0.02, 0.01),
                "C4_REMOTE_MATCHED_REPLACE": _stats(0.03, 0.01),
                "C2_REMOTE_ZERO": _stats(0.20, 0.10),
                "C5_LOCAL_PERMUTE": _stats(0.04, 0.01),
            },
            _seed_summary(c3_positive=5, c4_positive=4),
            survival,
        )
        self.assertEqual(
            supported["E1_STATUS"],
            "REMOTE_INFORMATION_SUPPORTED",
        )
        self.assertTrue(supported["R1"])
        self.assertTrue(supported["R5"])

        conditional = run_e1._decision_rules_status(
            {
                "C3_REMOTE_PERMUTE": _stats(0.02, -0.001),
                "C4_REMOTE_MATCHED_REPLACE": _stats(0.03, 0.001),
                "C2_REMOTE_ZERO": _stats(0.20, 0.10),
                "C5_LOCAL_PERMUTE": _stats(0.04, 0.01),
            },
            _seed_summary(c3_positive=5, c4_positive=5),
            survival,
        )
        self.assertEqual(
            conditional["E1_STATUS"],
            "REMOTE_INFORMATION_CONDITIONAL",
        )
        self.assertFalse(conditional["R1"])
        self.assertTrue(conditional["R5"])

        not_supported = run_e1._decision_rules_status(
            {
                "C3_REMOTE_PERMUTE": _stats(0.0, -0.001),
                "C4_REMOTE_MATCHED_REPLACE": _stats(0.0, -0.001),
                "C2_REMOTE_ZERO": _stats(0.2, 0.1),
                "C5_LOCAL_PERMUTE": _stats(0.02, 0.01),
            },
            _seed_summary(c3_positive=2, c4_positive=2),
            survival,
        )
        self.assertEqual(
            not_supported["E1_STATUS"],
            "REMOTE_INFORMATION_NOT_SUPPORTED",
        )
        self.assertTrue(not_supported["C3_C4_NEAR_ZERO"])
        self.assertTrue(not_supported["C5_CLEAR_POSITIVE"])

        inconclusive = run_e1._decision_rules_status(
            {
                "C3_REMOTE_PERMUTE": _stats(0.0, -0.001),
                "C4_REMOTE_MATCHED_REPLACE": _stats(0.0, -0.001),
                "C2_REMOTE_ZERO": _stats(0.0, -0.001),
                "C5_LOCAL_PERMUTE": _stats(0.0, -0.001),
            },
            _seed_summary(c3_positive=2, c4_positive=2),
            survival,
        )
        self.assertEqual(
            inconclusive["E1_STATUS"],
            "INCONCLUSIVE_OR_INVALID",
        )

    def test_supported_interpretation_covers_all_statuses(self) -> None:
        statuses = (
            "REMOTE_INFORMATION_SUPPORTED",
            "REMOTE_INFORMATION_CONDITIONAL",
            "REMOTE_INFORMATION_NOT_SUPPORTED",
            "INCONCLUSIVE_OR_INVALID",
        )
        texts = [run_e1._supported_interpretation(status) for status in statuses]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertTrue(all(texts))
        self.assertNotIn("causal mechanism", texts[0].lower())
        self.assertNotIn("architecture capacity is proven", texts[2].lower())

    def test_replay_validation_is_deterministic_and_rejects_score_change(self) -> None:
        frame = pd.DataFrame(
            {
                "global_index": np.asarray([0, 1], dtype=np.int64),
                "item_index": np.asarray([0, 0], dtype=np.int64),
                "frame_in_utterance": np.asarray([382, 383], dtype=np.int64),
                "label": np.asarray([0, 1], dtype=np.int64),
                "full_score": np.asarray([0.1, 0.9], dtype=np.float64),
                "score_C3_REMOTE_PERMUTE_101": np.asarray(
                    [0.2, 0.8],
                    dtype=np.float64,
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            protocol_path = Path(temporary) / "e1_protocol_freeze.json"
            protocol_path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(run_e1, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(
                    run_e1,
                    "_collect_frame_results",
                    return_value=frame.copy(),
                ),
            ):
                payload = run_e1._run_replay_validation(
                    population=None,
                    reference_frame=frame,
                    model=None,
                    frontend=None,
                    device=torch.device("cpu"),
                    donor_selection={},
                    donor_records={},
                    chunk_frames=2,
                    write_result=False,
                )
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["max_abs_score_error"], 0.0)

            changed = frame.copy()
            changed.loc[0, "full_score"] += 1e-6
            with (
                mock.patch.object(run_e1, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(
                    run_e1,
                    "_collect_frame_results",
                    return_value=changed,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    run_e1._run_replay_validation(
                        population=None,
                        reference_frame=frame,
                        model=None,
                        frontend=None,
                        device=torch.device("cpu"),
                        donor_selection={},
                        donor_records={},
                        chunk_frames=2,
                        write_result=False,
                    )

    def test_artifact_verifier_checks_schema_and_exact_figure_set(self) -> None:
        payload = _required_summary_payload()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "e1_protocol_freeze.json"
            protocol_path.write_text("{}\n", encoding="utf-8")
            for name in run_e1.REQUIRED_OUTPUT_FILES:
                path = root / run_e1._output_name(name)
                if name == "e1_figures/":
                    path.mkdir(parents=True, exist_ok=True)
                elif name != "e1_final_summary.json":
                    path.write_text("{}\n", encoding="utf-8")
            (root / run_e1._output_name("e1_final_summary.json")).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            pd.DataFrame({"global_index": [0, 1]}).to_parquet(
                root / run_e1._output_name("e1_frame_results.parquet"),
                index=False,
            )
            figures = root / run_e1._output_name("e1_figures")
            for name in run_e1.FIGURE_FILES:
                (figures / name).write_bytes(b"png")
            manifest = {
                "protocol_sha256": run_e1._sha256_file(protocol_path),
                "artifacts": [],
            }
            (
                root / run_e1._output_name("e1_execution_manifest.json")
            ).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with (
                mock.patch.object(run_e1, "OUTPUT_ROOT", root),
                mock.patch.object(run_e1, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(run_e1, "TEST_FRAMES", 2),
            ):
                verified = run_e1._verify_required_artifacts()
                self.assertEqual(
                    verified["summary"]["E1_STATUS"],
                    payload["E1_STATUS"],
                )
                (figures / "figure_6_extra.png").write_bytes(b"png")
                with self.assertRaises(ValueError):
                    run_e1._verify_required_artifacts()

    def test_claim_freeze_uses_exact_required_headings(self) -> None:
        summary = _required_summary_payload()
        summary["SUPPORTED_INTERPRETATION"] = (
            "No supported interpretation is available."
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(run_e1, "OUTPUT_ROOT", root):
                run_e1._write_claim_freeze(
                    summary=summary,
                    reconciliation_explanation="No mechanism preference.",
                )
            text = (
                root / run_e1._output_name("e1_claim_freeze.md")
            ).read_text(encoding="utf-8")
        for heading in (
            "OBSERVATIONS",
            "SUPPORTED_INTERPRETATIONS",
            "ALTERNATIVE_EXPLANATIONS",
            "UNSUPPORTED_CLAIMS",
        ):
            self.assertIn(f"\n{heading}\n", "\n" + text)


if __name__ == "__main__":
    unittest.main()
