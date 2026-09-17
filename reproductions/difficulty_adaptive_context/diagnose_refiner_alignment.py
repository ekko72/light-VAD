# -*- coding: utf-8 -*-
"""Diagnose whether the learned refiner tracks the Long temporal probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reproductions.difficulty_adaptive_context.train_context import (
    UNSEEN_NOISE,
)


CONDITION_ORDER = ("clean", "20", "10", "5", "0", "-5")
PROBABILITY_EPSILON = 1e-6


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probability.astype(np.float64),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _analysis(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    group_mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(group_mask, dtype=bool) & np.asarray(selected, dtype=bool)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"selected": 0}

    truth = labels[mask] >= 1
    short_margin = _logit(short_scores[mask])
    long_margin = _logit(long_scores[mask])
    refined_margin = _logit(refined_scores[mask])
    long_residual = long_margin - short_margin
    refined_residual = refined_margin - short_margin

    short_correct = short_scores[mask] >= 0.5
    short_correct = short_correct == truth
    long_correct = (long_scores[mask] >= 0.5) == truth
    refined_correct = (refined_scores[mask] >= 0.5) == truth

    correction = int(np.count_nonzero(~short_correct & long_correct))
    harm = int(np.count_nonzero(short_correct & ~long_correct))
    refined_correction = int(np.count_nonzero(~short_correct & refined_correct))
    refined_harm = int(np.count_nonzero(short_correct & ~refined_correct))

    correction_mask = ~short_correct & long_correct
    harm_mask = short_correct & ~long_correct
    nonzero = (np.abs(long_residual) > PROBABILITY_EPSILON) & (
        np.abs(refined_residual) > PROBABILITY_EPSILON
    )
    same_direction = (
        long_residual[nonzero] * refined_residual[nonzero] > 0.0
    )

    long_magnitude = np.abs(long_residual[nonzero])
    refined_magnitude = np.abs(refined_residual[nonzero])
    median_ratio = (
        float(np.median(refined_magnitude / long_magnitude))
        if long_magnitude.size
        else None
    )

    return {
        "selected": count,
        "probe": {
            "correction": correction,
            "harm": harm,
            "net": correction - harm,
        },
        "refiner": {
            "correction": refined_correction,
            "harm": refined_harm,
            "net": refined_correction - refined_harm,
        },
        "on_probe_corrections": {
            "frames": int(np.count_nonzero(correction_mask)),
            "refiner_correct": int(
                np.count_nonzero(correction_mask & refined_correct)
            ),
            "refiner_correct_rate": _safe_rate(
                int(np.count_nonzero(correction_mask & refined_correct)),
                int(np.count_nonzero(correction_mask)),
            ),
        },
        "on_probe_harms": {
            "frames": int(np.count_nonzero(harm_mask)),
            "refiner_correct": int(
                np.count_nonzero(harm_mask & refined_correct)
            ),
            "refiner_correct_rate": _safe_rate(
                int(np.count_nonzero(harm_mask & refined_correct)),
                int(np.count_nonzero(harm_mask)),
            ),
        },
        "residual_alignment": {
            "nonzero_frames": int(np.count_nonzero(nonzero)),
            "same_direction": int(np.count_nonzero(same_direction)),
            "same_direction_rate": _safe_rate(
                int(np.count_nonzero(same_direction)),
                int(np.count_nonzero(nonzero)),
            ),
            "pearson": _safe_correlation(long_residual, refined_residual),
            "median_refined_to_probe_magnitude_ratio": median_ratio,
        },
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase A3 Refiner-vs-Long Alignment",
        "",
        f"- Frame predictions: `{summary['predictions']}`",
        "- Margins use `logit(p)` differences from the shared Short output.",
        "- This is a diagnostic only; it does not alter thresholds or checkpoints.",
        "",
        "| Group | Selected | Probe net | Refiner net | On probe corrections: "
        "refiner correct | On probe harms: refiner correct | Residual same "
        "direction | Median residual magnitude ratio | Pearson |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary["groups"].items():
        if int(row.get("selected", 0)) == 0:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        corrections = row["on_probe_corrections"]
        harms = row["on_probe_harms"]
        alignment = row["residual_alignment"]
        lines.append(
            f"| {name} | {row['selected']} | {row['probe']['net']} | "
            f"{row['refiner']['net']} | "
            f"{corrections['refiner_correct']}/"
            f"{corrections['frames']} ({_percent(corrections['refiner_correct_rate'])}) | "
            f"{harms['refiner_correct']}/{harms['frames']} "
            f"({_percent(harms['refiner_correct_rate'])}) | "
            f"{alignment['same_direction']}/{alignment['nonzero_frames']} "
            f"({_percent(alignment['same_direction_rate'])}) | "
            f"{_number(alignment['median_refined_to_probe_magnitude_ratio'])} | "
            f"{_number(alignment['pearson'])} |"
        )
    lines.extend(
        [
            "",
            "`On probe corrections` asks how often the refiner also corrects "
            "frames where Long corrects Short. `On probe harms` asks how often "
            "the refiner avoids Long's wrong flips. A positive Pearson and a "
            "small magnitude ratio indicate an underpowered refiner; a near-zero "
            "or negative Pearson indicates a mismatched refinement signal.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose refiner and Long-probe residual alignment."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--unseen-noise",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = np.load(args.predictions)
    labels = payload["labels"].astype(np.int64)
    short_scores = payload["short_scores"].astype(np.float64)
    long_scores = payload["long_scores"].astype(np.float64)
    refined_scores = payload["full_adaptive_scores"].astype(np.float64)
    selected = payload["selected"].astype(bool)
    test_mask = payload["test_mask"].astype(bool)
    condition = payload["condition"].astype(str)
    noise_name = payload["noise_name"].astype(str)

    unseen_names = {
        str(value) for value in args.unseen_noise if str(value).strip()
    }
    noisy_mask = condition != "clean"
    unseen_mask = noisy_mask & np.isin(noise_name, sorted(unseen_names))
    seen_mask = noisy_mask & ~unseen_mask

    masks: dict[str, np.ndarray] = {
        "overall": test_mask,
        "seen": test_mask & seen_mask,
        "unseen": test_mask & unseen_mask,
        "clean": test_mask & (condition == "clean"),
    }
    for name in CONDITION_ORDER:
        mask = test_mask & (condition == name)
        if np.any(mask):
            masks[name] = mask

    summary = {
        "predictions": str(args.predictions),
        "unseen_noise": sorted(unseen_names),
        "groups": {
            name: _analysis(
                labels,
                short_scores,
                long_scores,
                refined_scores,
                selected,
                mask,
            )
            for name, mask in masks.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "alignment.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        args.output_dir / "report.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(_report(summary))
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
