# -*- coding: utf-8 -*-
"""Focused checks for deterministic LibriVAD row sampling."""

from __future__ import annotations

import unittest

try:
    from .dataset import stratified_row_indices
except ImportError:
    from dataset import stratified_row_indices


class ManifestSamplingTests(unittest.TestCase):
    def test_stratified_sampling_is_deterministic_and_balanced(self) -> None:
        rows = [
            {"noise_name": f"noise-{noise}", "snr_db": str(snr)}
            for noise in range(3)
            for snr in range(2)
            for _ in range(10)
        ]

        first = stratified_row_indices(rows, 12, seed=7)
        second = stratified_row_indices(rows, 12, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(first, sorted(first))
        for noise in range(3):
            for snr in range(2):
                selected = [
                    index
                    for index in first
                    if rows[index]
                    == {"noise_name": f"noise-{noise}", "snr_db": str(snr)}
                ]
                self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
