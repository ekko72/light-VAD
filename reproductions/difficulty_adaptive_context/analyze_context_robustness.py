# -*- coding: utf-8 -*-
"""Offline robustness analysis for the difficulty-adaptive context experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reproductions.difficulty_adaptive_context.evaluate_context import (
    BOUNDARY_NAMES,
    CONDITION_ORDER,
    DIFFICULTY_NAMES,
)


def _grouped_counts(
    labels: np.ndarray,
    scores: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    predictions = np.asarray(scores, dtype=np.float64) >= 0.5
    labels = np.asarray(labels, dtype=np.int64)
    flat_index = (
        inverse * 4
        + (predictions.astype(np.int64) << 1)
        + labels.astype(np.int64)
    )
    counts = np.bincount(flat_index, minlength=n_groups * 4)
    return counts.reshape(n_groups, 4).astype(np.int64)


def _metrics_from_counts(counts: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(value) for value in counts)
    frames = tn + fp + fn + tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    return {
        "frames": frames,
        "speech": tp + fn,
        "silence": tn + fp,
        "error": (fp + fn) / max(frames, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _paired_metrics(
    short_counts: np.ndarray, long_counts: np.ndarray
) -> dict[str, Any]:
    short = _metrics_from_counts(short_counts)
    long = _metrics_from_counts(long_counts)
    return {
        "frames": int(short["frames"]),
        "short": short,
        "long": long,
        "delta_f1_long_minus_short": float(long["f1"] - short["f1"]),
        "error_reduction_short_minus_long": float(
            short["error"] - long["error"]
        ),
    }


def _subset_counts(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _grouped_counts(
            labels[mask], short_scores[mask], inverse[mask], n_groups
        ),
        _grouped_counts(
            labels[mask], long_scores[mask], inverse[mask], n_groups
        ),
    )


def _bootstrap_paired(
    short_counts: np.ndarray,
    long_counts: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    n_groups = short_counts.shape[0]
    rng = np.random.default_rng(seed)
    error_reduction = np.empty(repeats, dtype=np.float64)
    delta_f1 = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.integers(0, n_groups, size=n_groups)
        row = _paired_metrics(
            short_counts[sampled].sum(axis=0),
            long_counts[sampled].sum(axis=0),
        )
        error_reduction[index] = row[
            "error_reduction_short_minus_long"
        ]
        delta_f1[index] = row["delta_f1_long_minus_short"]
    error_low, error_high = np.quantile(error_reduction, [0.025, 0.975])
    f1_low, f1_high = np.quantile(delta_f1, [0.025, 0.975])
    return {
        "error_reduction_ci95_low": float(error_low),
        "error_reduction_ci95_high": float(error_high),
        "delta_f1_ci95_low": float(f1_low),
        "delta_f1_ci95_high": float(f1_high),
    }


def _bootstrap_unseen_gap(
    seen_short: np.ndarray,
    seen_long: np.ndarray,
    unseen_short: np.ndarray,
    unseen_long: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    n_groups = seen_short.shape[0]
    rng = np.random.default_rng(seed)
    error_gap = np.empty(repeats, dtype=np.float64)
    f1_gap = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.integers(0, n_groups, size=n_groups)
        seen = _paired_metrics(
            seen_short[sampled].sum(axis=0),
            seen_long[sampled].sum(axis=0),
        )
        unseen = _paired_metrics(
            unseen_short[sampled].sum(axis=0),
            unseen_long[sampled].sum(axis=0),
        )
        error_gap[index] = (
            unseen["error_reduction_short_minus_long"]
            - seen["error_reduction_short_minus_long"]
        )
        f1_gap[index] = (
            unseen["delta_f1_long_minus_short"]
            - seen["delta_f1_long_minus_short"]
        )
    error_low, error_high = np.quantile(error_gap, [0.025, 0.975])
    f1_low, f1_high = np.quantile(f1_gap, [0.025, 0.975])
    return {
        "error_reduction_gap_ci95_low": float(error_low),
        "error_reduction_gap_ci95_high": float(error_high),
        "delta_f1_gap_ci95_low": float(f1_low),
        "delta_f1_gap_ci95_high": float(f1_high),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Context Robustness Checks",
        "",
        "## Protocol",
        "",
        f"- Predictions: `{summary['predictions']}`",
        f"- Utterance clusters: {summary['clusters']}",
        f"- Frames: {summary['frames']}",
        f"- Bootstrap repeats: {summary['bootstrap_repeats']}",
        "- Confidence thresholds come from the original global A2 buckets.",
        "",
        "## Very Hard Frames by SNR",
        "",
        "| SNR | Frames | Short error | Long error | Error reduction | "
        "95% CI | F1 delta | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        row = summary["very_hard_by_condition"].get(condition)
        if row is None:
            continue
        label = "Clean" if condition == "clean" else condition
        lines.append(
            f"| {label} | {row['frames']} | "
            f"{100.0 * row['short']['error']:.3f}% | "
            f"{100.0 * row['long']['error']:.3f}% | "
            f"{100.0 * row['error_reduction_short_minus_long']:.3f}% | "
            f"[{100.0 * row['error_reduction_ci95_low']:.3f}, "
            f"{100.0 * row['error_reduction_ci95_high']:.3f}]% | "
            f"{row['delta_f1_long_minus_short']:.4f} | "
            f"[{row['delta_f1_ci95_low']:.4f}, "
            f"{row['delta_f1_ci95_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Utterance-Clustered Bootstrap",
            "",
            "| Check | Point estimate | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    checks = summary["cluster_bootstrap"]
    for label, row in checks.items():
        if "error_reduction" in row:
            point = 100.0 * row["error_reduction"]
            low = 100.0 * row["error_reduction_ci95_low"]
            high = 100.0 * row["error_reduction_ci95_high"]
            unit = "%"
        else:
            point = 100.0 * row["error_reduction_gap"]
            low = 100.0 * row["error_reduction_gap_ci95_low"]
            high = 100.0 * row["error_reduction_gap_ci95_high"]
            unit = "%"
        lines.append(
            f"| {label} | {point:.3f}{unit} | "
            f"[{low:.3f}, {high:.3f}]{unit} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze context-prediction robustness by SNR and utterance."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")

    predictions = np.load(args.predictions)
    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)

    labels = predictions["labels"].astype(np.int64)
    short_scores = predictions["short_scores"].astype(np.float64)
    long_scores = predictions["long_scores"].astype(np.float64)
    condition = predictions["condition"].astype(str)
    source_key = predictions["source_key"].astype(str)
    boundary_distance = predictions["boundary_distance_ms"].astype(
        np.float64
    )
    _source_ids, inverse = np.unique(source_key, return_inverse=True)
    n_groups = int(inverse.max()) + 1

    thresholds = np.asarray(
        results["A2_confidence_thresholds"], dtype=np.float64
    )
    confidence = np.abs(short_scores - 0.5)
    bucket_ids = np.digitize(confidence, thresholds, right=False)

    very_hard_by_condition: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(CONDITION_ORDER):
        mask = (condition == name) & (bucket_ids == 0)
        if not np.any(mask):
            continue
        short_counts, long_counts = _subset_counts(
            labels,
            short_scores,
            long_scores,
            inverse,
            n_groups,
            mask,
        )
        row = _paired_metrics(
            short_counts.sum(axis=0), long_counts.sum(axis=0)
        )
        row.update(
            _bootstrap_paired(
                short_counts,
                long_counts,
                repeats=args.bootstrap_repeats,
                seed=args.seed + index,
            )
        )
        very_hard_by_condition[name] = row

    all_short, all_long = _subset_counts(
        labels,
        short_scores,
        long_scores,
        inverse,
        n_groups,
        np.ones(labels.shape, dtype=bool),
    )
    very_hard_short, very_hard_long = _subset_counts(
        labels,
        short_scores,
        long_scores,
        inverse,
        n_groups,
        bucket_ids == 0,
    )

    noise_name = predictions["noise_name"].astype(str)
    unseen_names = set(results["protocol"]["unseen_noise"])
    noisy = condition != "clean"
    unseen = noisy & np.isin(noise_name, sorted(unseen_names))
    seen = noisy & ~unseen
    seen_short, seen_long = _subset_counts(
        labels, short_scores, long_scores, inverse, n_groups, seen
    )
    unseen_short, unseen_long = _subset_counts(
        labels, short_scores, long_scores, inverse, n_groups, unseen
    )
    boundary_short, boundary_long = _subset_counts(
        labels,
        short_scores,
        long_scores,
        inverse,
        n_groups,
        boundary_distance <= 200.0,
    )

    all_row = _paired_metrics(all_short.sum(0), all_long.sum(0))
    very_hard_row = _paired_metrics(
        very_hard_short.sum(0), very_hard_long.sum(0)
    )
    boundary_row = _paired_metrics(
        boundary_short.sum(0), boundary_long.sum(0)
    )
    seen_row = _paired_metrics(seen_short.sum(0), seen_long.sum(0))
    unseen_row = _paired_metrics(unseen_short.sum(0), unseen_long.sum(0))
    seen_gap = (
        seen_row["error_reduction_short_minus_long"]
    )
    unseen_gap = (
        unseen_row["error_reduction_short_minus_long"]
    )

    cluster_bootstrap: dict[str, dict[str, float]] = {}
    for label, short_counts, long_counts in (
        ("All valid frames", all_short, all_long),
        ("Very hard frames", very_hard_short, very_hard_long),
        ("Within 200 ms of a boundary", boundary_short, boundary_long),
    ):
        row = {
            "error_reduction": (
                _paired_metrics(
                    short_counts.sum(0), long_counts.sum(0)
                )["error_reduction_short_minus_long"]
            )
        }
        row.update(
            _bootstrap_paired(
                short_counts,
                long_counts,
                repeats=args.bootstrap_repeats,
                seed=args.seed + len(cluster_bootstrap),
            )
        )
        cluster_bootstrap[label] = row
    cluster_bootstrap["Unseen minus seen error reduction"] = {
        "error_reduction_gap": float(unseen_gap - seen_gap),
        **_bootstrap_unseen_gap(
            seen_short,
            seen_long,
            unseen_short,
            unseen_long,
            repeats=args.bootstrap_repeats,
            seed=args.seed + 100,
        ),
    }

    summary = {
        "predictions": str(args.predictions),
        "results": str(args.results),
        "clusters": n_groups,
        "frames": int(labels.size),
        "bootstrap_repeats": args.bootstrap_repeats,
        "confidence_thresholds": thresholds.tolist(),
        "very_hard_by_condition": very_hard_by_condition,
        "overall": all_row,
        "very_hard": very_hard_row,
        "within_200ms": boundary_row,
        "seen": seen_row,
        "unseen": unseen_row,
        "cluster_bootstrap": cluster_bootstrap,
        "boundary_names": list(BOUNDARY_NAMES),
        "difficulty_names": list(DIFFICULTY_NAMES),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "robustness.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        args.output_dir / "robustness.md", "w", encoding="utf-8"
    ) as handle:
        handle.write(_markdown_report(summary))
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
