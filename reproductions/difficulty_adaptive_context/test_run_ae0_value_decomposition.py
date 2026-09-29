# -*- coding: utf-8 -*-
"""Unit checks for the AE0 OOD value-failure decomposition."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from reproductions.difficulty_adaptive_context.run_ae0_value_decomposition import (
    Ae0Config,
    CELL_ORDER,
    DOMAIN_ORDER,
    FrameBundle,
    _counts_from_bundle,
    _point_metrics,
    _sufficient_stats,
    average_rank_auroc,
    classify_layer,
    validate_reconstruction,
)


def _bundle(
    *,
    labels: np.ndarray,
    short_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    source_codes: np.ndarray,
    speaker_codes: np.ndarray,
    domain_codes: np.ndarray,
    cell_codes: np.ndarray,
) -> FrameBundle:
    return FrameBundle(
        labels=np.asarray(labels, dtype=np.int64),
        short_scores=np.asarray(short_scores, dtype=np.float64),
        refined_scores=np.asarray(refined_scores, dtype=np.float64),
        selected=np.asarray(selected, dtype=bool),
        source_codes=np.asarray(source_codes, dtype=np.int64),
        speaker_codes=np.asarray(speaker_codes, dtype=np.int64),
        domain_codes=np.asarray(domain_codes, dtype=np.int64),
        cell_codes=np.asarray(cell_codes, dtype=np.int64),
        source_order=("source-a", "source-b"),
        speaker_order=("speaker-a",),
    )


def _zero_integrity_payload() -> dict[str, object]:
    return {
        "integrity_checks": {
            "all": {"selected": 0, "correction": 0, "harm": 0},
            "seen": {"selected": 0, "correction": 0, "harm": 0},
            "unseen": {"selected": 0, "correction": 0, "harm": 0},
            "per_cell": [
                {
                    "domain": domain,
                    "snr_db": snr,
                    "selected": 0,
                    "correction": 0,
                    "harm": 0,
                }
                for domain, snr in CELL_ORDER
            ],
        }
    }


class Ae0ValueDecompositionTests(unittest.TestCase):
    def test_oracle_correction_harm_and_gate_utility(self) -> None:
        bundle = _bundle(
            labels=np.asarray([0, 1, 0, 1, 0, 1]),
            short_scores=np.asarray([0.9, 0.1, 0.1, 0.9, 0.1, 0.9]),
            refined_scores=np.asarray([0.1, 0.9, 0.9, 0.1, 0.1, 0.9]),
            selected=np.asarray([True, True, False, False, False, False]),
            source_codes=np.asarray([0, 0, 0, 1, 1, 1]),
            speaker_codes=np.zeros(6, dtype=np.int64),
            domain_codes=np.zeros(6, dtype=np.int64),
            cell_codes=np.zeros(6, dtype=np.int64),
        )

        counts = _counts_from_bundle(
            bundle,
            np.ones(bundle.labels.size, dtype=bool),
        )
        self.assertEqual(counts["frames"], 6)
        self.assertEqual(counts["correction_all"], 2)
        self.assertEqual(counts["harm_all"], 2)
        self.assertEqual(counts["selected"], 2)
        self.assertEqual(counts["correction"], 2)
        self.assertEqual(counts["harm"], 0)

        stats, correction_pairs, harm_pairs = _sufficient_stats(
            bundle=bundle,
            mask=np.ones(bundle.labels.size, dtype=bool),
            cluster_codes=bundle.source_codes,
            n_clusters=len(bundle.source_order),
        )
        point = _point_metrics(stats, correction_pairs, harm_pairs)
        self.assertAlmostEqual(point["availability_gross"], 2.0 / 6.0)
        self.assertAlmostEqual(point["availability_net"], 0.0)
        self.assertAlmostEqual(point["gate_utility"], 2.0 / 6.0)
        self.assertAlmostEqual(point["observability_auroc"], 0.5)
        self.assertIsNone(point["actionability_ratio"])

    def test_average_rank_auroc_handles_ties_and_undefined_classes(self) -> None:
        positive = np.asarray([False, True, True, False])
        scores = np.asarray([0.1, 0.2, 0.2, 0.3])

        self.assertAlmostEqual(
            average_rank_auroc(positive, scores),
            0.5,
        )
        self.assertIsNone(
            average_rank_auroc(
                np.asarray([True, True]),
                np.asarray([0.4, 0.4]),
            ),
        )
        self.assertIsNone(
            average_rank_auroc(
                np.asarray([False, False]),
                np.asarray([0.4, 0.4]),
            )
        )

    def test_classification_precedence_covers_all_four_layers(self) -> None:
        self.assertEqual(
            classify_layer(
                availability_net=0.0,
                observability_auroc=0.9,
                gate_utility=0.1,
            ),
            "AVAILABILITY_LIMITED",
        )
        self.assertEqual(
            classify_layer(
                availability_net=0.1,
                observability_auroc=0.5,
                gate_utility=0.1,
            ),
            "OBSERVABILITY_LIMITED",
        )
        self.assertEqual(
            classify_layer(
                availability_net=0.1,
                observability_auroc=0.6,
                gate_utility=0.0,
            ),
            "ACTIONABILITY_LIMITED",
        )
        self.assertEqual(
            classify_layer(
                availability_net=0.1,
                observability_auroc=0.6,
                gate_utility=0.1,
            ),
            "ACTIONABLE",
        )
        self.assertEqual(
            classify_layer(
                availability_net=0.1,
                observability_auroc=None,
                gate_utility=0.1,
            ),
            "OBSERVABILITY_LIMITED",
        )

    def test_integrity_mismatch_is_rejected_before_interpretation(self) -> None:
        empty = _bundle(
            labels=np.empty(0, dtype=np.int64),
            short_scores=np.empty(0),
            refined_scores=np.empty(0),
            selected=np.empty(0, dtype=bool),
            source_codes=np.empty(0, dtype=np.int64),
            speaker_codes=np.empty(0, dtype=np.int64),
            domain_codes=np.empty(0, dtype=np.int64),
            cell_codes=np.empty(0, dtype=np.int64),
        )
        payload = _zero_integrity_payload()
        config = Ae0Config(
            path=Path("synthetic-ae0-protocol.json"),
            payload=payload,
            sha256="0" * 64,
        )
        validate_reconstruction(empty, config)

        payload["integrity_checks"]["all"]["selected"] = 1
        with self.assertRaises(ValueError):
            validate_reconstruction(empty, config)


if __name__ == "__main__":
    unittest.main()
