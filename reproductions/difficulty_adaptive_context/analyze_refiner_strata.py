# -*- coding: utf-8 -*-
"""Stratify the fixed-gate refiner utility by difficulty, noise and utterance.

This is a diagnostic over an already-frozen evaluation.  It never changes the
threshold, the checkpoint or the activation budget.  The question it answers
is whether the refiner's utility is spread over the selected region or
concentrated in a few confidence strata, noise types or utterances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reproductions.difficulty_adaptive_context.train_context import (
    UNSEEN_NOISE,
)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _utility(
    labels: np.ndarray,
    short_scores: np.ndarray,
    candidate_scores: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int]:
    """Return correction/harm counts for substituting ``candidate_scores``."""
    truth = np.asarray(labels)[mask] >= 1
    baseline_correct = (np.asarray(short_scores)[mask] >= 0.5) == truth
    candidate_correct = (np.asarray(candidate_scores)[mask] >= 0.5) == truth
    correction = int(np.count_nonzero(~baseline_correct & candidate_correct))
    harm = int(np.count_nonzero(baseline_correct & ~candidate_correct))
    return correction, harm


def _row(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    refined_scores: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    frames = int(np.count_nonzero(mask))
    if frames == 0:
        return {"frames": 0}
    probe_correction, probe_harm = _utility(
        labels, short_scores, long_scores, mask
    )
    refiner_correction, refiner_harm = _utility(
        labels, short_scores, refined_scores, mask
    )
    probe_net = probe_correction - probe_harm
    refiner_net = refiner_correction - refiner_harm
    return {
        "frames": frames,
        "probe": {
            "correction": probe_correction,
            "harm": probe_harm,
            "net": probe_net,
            "net_per_frame": _rate(probe_net, frames),
        },
        "refiner": {
            "correction": refiner_correction,
            "harm": refiner_harm,
            "net": refiner_net,
            "net_per_frame": _rate(refiner_net, frames),
        },
        "retention": (
            _rate(refiner_net, probe_net)
            if probe_net > 0
            else None
        ),
    }


def _strata_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    group_mask: np.ndarray,
    confidence: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    """Split the group's selected frames into equal-count confidence bins."""
    subset = np.flatnonzero(np.asarray(group_mask, dtype=bool) & selected)
    if subset.size == 0:
        return []
    order = subset[np.argsort(confidence[subset], kind="stable")]
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(np.array_split(order, bins)):
        if chunk.size == 0:
            continue
        chunk_mask = np.zeros_like(selected, dtype=bool)
        chunk_mask[chunk] = True
        row = _row(
            labels,
            short_scores,
            long_scores,
            refined_scores,
            chunk_mask,
        )
        confidence_chunk = confidence[chunk]
        row["stratum"] = index + 1
        row["confidence_low"] = float(confidence_chunk.min())
        row["confidence_high"] = float(confidence_chunk.max())
        rows.append(row)
    return rows


def _cluster_rows(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    group_mask: np.ndarray,
    source_key: np.ndarray,
) -> dict[str, Any]:
    """Summarise how concentrated the utility is over utterances."""
    subset = np.flatnonzero(np.asarray(group_mask, dtype=bool) & selected)
    if subset.size == 0:
        return {"clusters": 0}
    keys, inverse = np.unique(source_key[subset], return_inverse=True)
    count = np.bincount(inverse, minlength=keys.size)

    truth = labels[subset] >= 1
    short_correct = (short_scores[subset] >= 0.5) == truth

    def _per_cluster(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        correct = (scores[subset] >= 0.5) == truth
        correction = np.bincount(
            inverse,
            weights=(~short_correct & correct).astype(np.float64),
            minlength=keys.size,
        )
        harm = np.bincount(
            inverse,
            weights=(short_correct & ~correct).astype(np.float64),
            minlength=keys.size,
        )
        return correction, harm

    probe_correction, probe_harm = _per_cluster(long_scores)
    refiner_correction, refiner_harm = _per_cluster(refined_scores)
    refiner_net = refiner_correction - refiner_harm

    def _top_share(values: np.ndarray) -> float | None:
        total = float(values.sum())
        if total <= 0.0:
            return None
        top = max(1, int(np.ceil(0.10 * values.size)))
        ordered = np.sort(values)[::-1]
        return float(ordered[:top].sum() / total)

    worst = np.argsort(refiner_net)[:5]
    return {
        "clusters": int(keys.size),
        "positive_clusters": int(np.count_nonzero(refiner_net > 0)),
        "negative_clusters": int(np.count_nonzero(refiner_net < 0)),
        "neutral_clusters": int(np.count_nonzero(refiner_net == 0)),
        "top_decile_share_refiner_correction": _top_share(
            refiner_correction
        ),
        "top_decile_share_refiner_harm": _top_share(refiner_harm),
        "top_decile_share_probe_correction": _top_share(probe_correction),
        "top_decile_share_probe_harm": _top_share(probe_harm),
        "worst_clusters": [
            {
                "source_key": str(keys[index]),
                "selected": int(count[index]),
                "refiner_net": int(refiner_net[index]),
                "probe_net": int(
                    probe_correction[index] - probe_harm[index]
                ),
            }
            for index in worst
        ],
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase A3 Refiner Utility Strata",
        "",
        f"- Frame predictions: `{summary['predictions']}`",
        f"- Test activation: {_percent(summary['test_activation'])}",
        "- Diagnostic only: thresholds, checkpoints and budgets are frozen.",
        "- Stratum 1 is the most uncertain selected region.",
        "",
        "## Confidence Strata Inside The Selected Region",
        "",
        "| Group | Stratum | Confidence range | Selected | Long net/selected "
        "| Refiner net/selected | Retention |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for group, rows in summary["strata"].items():
        for row in rows:
            lines.append(
                f"| {group} | {row['stratum']} | "
                f"{row['confidence_low']:.4f}"
                f"-{row['confidence_high']:.4f} | {row['frames']} | "
                f"{_percent(row['probe']['net_per_frame'])} | "
                f"{_percent(row['refiner']['net_per_frame'])} | "
                f"{_percent(row['retention'])} |"
            )

    lines.extend(
        [
            "",
            "## Per Noise Type",
            "",
            "| Noise | Group | Selected | Long net/selected | "
            "Refiner net/selected |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in summary["per_noise"].items():
        lines.append(
            f"| {name} | {row['group']} | {row['frames']} | "
            f"{_percent(row['probe']['net_per_frame'])} | "
            f"{_percent(row['refiner']['net_per_frame'])} |"
        )

    lines.extend(
        [
            "",
            "## Utterance Concentration",
            "",
            "| Group | Utterances | Positive | Negative | Top-10% share of "
            "refiner corrections | Top-10% share of refiner harms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, row in summary["clusters"].items():
        if not row.get("clusters"):
            lines.append(f"| {group} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {group} | {row['clusters']} | {row['positive_clusters']} | "
            f"{row['negative_clusters']} | "
            f"{_percent(row['top_decile_share_refiner_correction'])} | "
            f"{_percent(row['top_decile_share_refiner_harm'])} |"
        )

    worst = summary["clusters"].get("unseen", {}).get("worst_clusters", [])
    if worst:
        lines.extend(
            [
                "",
                "### Worst unseen utterances by refiner net",
                "",
                "| Utterance | Selected | Refiner net | Long net |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in worst:
            lines.append(
                f"| {row['source_key']} | {row['selected']} | "
                f"{row['refiner_net']} | {row['probe_net']} |"
            )

    lines.extend(
        [
            "",
            "`Retention` is the refiner net utility divided by the Long probe "
            "net utility on the same frames. A flat retention profile means the "
            "refiner degrades uniformly; a falling profile means it fails on "
            "the marginal, near-threshold frames.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stratify refiner utility by difficulty and cluster."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument(
        "--unseen-noise",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.strata <= 0:
        raise SystemExit("--strata must be positive")

    payload = np.load(args.predictions)
    labels = payload["labels"].astype(np.int64)
    short_scores = payload["short_scores"].astype(np.float64)
    long_scores = payload["long_scores"].astype(np.float64)
    refined_scores = payload["full_adaptive_scores"].astype(np.float64)
    selected = payload["selected"].astype(bool)
    test_mask = payload["test_mask"].astype(bool)
    condition = payload["condition"].astype(str)
    noise_name = payload["noise_name"].astype(str)
    source_key = payload["source_key"].astype(str)

    unseen_names = {
        str(value) for value in args.unseen_noise if str(value).strip()
    }
    noisy_mask = condition != "clean"
    unseen_mask = noisy_mask & np.isin(noise_name, sorted(unseen_names))
    seen_mask = noisy_mask & ~unseen_mask
    confidence = np.abs(short_scores - 0.5)

    groups = {
        "overall": test_mask,
        "seen": test_mask & seen_mask,
        "unseen": test_mask & unseen_mask,
    }
    strata = {
        name: _strata_rows(
            labels,
            short_scores,
            long_scores,
            refined_scores,
            selected,
            mask,
            confidence,
            args.strata,
        )
        for name, mask in groups.items()
    }
    clusters = {
        name: _cluster_rows(
            labels,
            short_scores,
            long_scores,
            refined_scores,
            selected,
            mask,
            source_key,
        )
        for name, mask in groups.items()
    }

    per_noise: dict[str, Any] = {}
    for name in sorted(set(noise_name[test_mask & noisy_mask])):
        mask = test_mask & noisy_mask & (noise_name == name)
        if not np.any(mask):
            continue
        row = _row(
            labels,
            short_scores,
            long_scores,
            refined_scores,
            mask & selected,
        )
        if row.get("frames", 0) == 0:
            continue
        row["group"] = "unseen" if name in unseen_names else "seen"
        per_noise[str(name)] = row

    summary = {
        "predictions": str(args.predictions),
        "strata": int(args.strata),
        "unseen_noise": sorted(unseen_names),
        "test_activation": float(np.mean(selected[test_mask])),
        "strata_rows": strata,
        "per_noise": per_noise,
        "clusters": clusters,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "strata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    report_input = dict(summary)
    report_input["strata"] = strata
    with open(
        args.output_dir / "report.md",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(_report(report_input))
    print(f"artifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
