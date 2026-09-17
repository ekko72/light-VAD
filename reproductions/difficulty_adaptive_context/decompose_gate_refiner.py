# -*- coding: utf-8 -*-
"""Decompose fixed-gate utility into gate quality and refiner quality."""

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


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _metrics_for_mask(
    labels: np.ndarray,
    short_scores: np.ndarray,
    long_scores: np.ndarray,
    refined_scores: np.ndarray,
    selected: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    selected = np.asarray(selected, dtype=bool) & mask
    truth = labels[mask] >= 1
    short_prediction = short_scores[mask] >= 0.5
    long_prediction = long_scores[mask] >= 0.5
    refined_prediction = refined_scores[mask] >= 0.5
    selected_frame = selected[mask]

    short_correct = short_prediction == truth
    long_correct = long_prediction == truth
    refined_correct = refined_prediction == truth

    long_correction = int(
        np.count_nonzero(selected_frame & ~short_correct & long_correct)
    )
    long_harm = int(
        np.count_nonzero(selected_frame & short_correct & ~long_correct)
    )
    refined_correction = int(
        np.count_nonzero(selected_frame & ~short_correct & refined_correct)
    )
    refined_harm = int(
        np.count_nonzero(selected_frame & short_correct & ~refined_correct)
    )
    selected_count = int(np.count_nonzero(selected_frame))

    both_correct = int(
        np.count_nonzero(selected_frame & long_correct & refined_correct)
    )
    long_only_correct = int(
        np.count_nonzero(selected_frame & long_correct & ~refined_correct)
    )
    refined_only_correct = int(
        np.count_nonzero(selected_frame & ~long_correct & refined_correct)
    )
    both_wrong = int(
        np.count_nonzero(selected_frame & ~long_correct & ~refined_correct)
    )

    long_signed = long_correction - long_harm
    refined_signed = refined_correction - refined_harm
    return {
        "frames": int(np.count_nonzero(mask)),
        "selected": selected_count,
        "activation_rate": _rate(selected_count, int(np.count_nonzero(mask))),
        "short_error": int(np.count_nonzero(~short_correct)),
        "long_error": int(np.count_nonzero(~long_correct)),
        "refined_error": int(np.count_nonzero(~refined_correct)),
        "probe_long": {
            "correction": long_correction,
            "harm": long_harm,
            "signed_utility": long_signed,
            "net_per_selected": _rate(long_signed, selected_count),
            "correction_rate_selected": _rate(
                long_correction, selected_count
            ),
            "harm_rate_selected": _rate(long_harm, selected_count),
        },
        "refiner": {
            "correction": refined_correction,
            "harm": refined_harm,
            "signed_utility": refined_signed,
            "net_per_selected": _rate(refined_signed, selected_count),
            "correction_rate_selected": _rate(
                refined_correction, selected_count
            ),
            "harm_rate_selected": _rate(refined_harm, selected_count),
        },
        "selected_overlap": {
            "both_correct": both_correct,
            "long_only_correct": long_only_correct,
            "refiner_only_correct": refined_only_correct,
            "both_wrong": both_wrong,
        },
    }


def _diagnosis(row: dict[str, Any]) -> str:
    selected = int(row["selected"])
    if selected == 0:
        return "no selected frames"
    long_net = row["probe_long"]["net_per_selected"]
    refiner_net = row["refiner"]["net_per_selected"]
    if long_net > 0.0 and refiner_net > 0.0:
        return "gate and refiner aligned"
    if long_net > 0.0 and refiner_net <= 0.0:
        return "refiner failure on a valid Long-positive region"
    if long_net <= 0.0 and refiner_net > 0.0:
        return "refiner gain, but gate is not aligned with Long probe"
    return "both gate region and refiner are weak"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase A3 Gate-vs-Refiner Decomposition",
        "",
        f"- Frame predictions: `{summary['predictions']}`",
        f"- Fixed confidence threshold: {summary['threshold']:.6f}",
        "- Prediction rule: probability >= 0.50",
        "",
        "| Group | Frames | Selected | Activation | Long correction | "
        "Long harm | Long net/selected | Refiner correction | Refiner harm | "
        "Refiner net/selected | Diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- |",
    ]
    for name, row in summary["groups"].items():
        probe = row["probe_long"]
        refiner = row["refiner"]
        lines.append(
            f"| {name} | {row['frames']} | {row['selected']} | "
            f"{_percent(row['activation_rate'])} | "
            f"{probe['correction']} | {probe['harm']} | "
            f"{_percent(probe['net_per_selected'])} | "
            f"{refiner['correction']} | {refiner['harm']} | "
            f"{_percent(refiner['net_per_selected'])} | "
            f"{_diagnosis(row)} |"
        )
    lines.extend(
        [
            "",
            "A positive Long net utility means the selected region contains "
            "transferable temporal value in the original Short/Long probe. "
            "A non-positive Refiner net utility then isolates the failure to "
            "the learned refinement.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose fixed-gate adaptive utility."
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
    threshold = float(payload["threshold"])

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

    groups = {
        name: _metrics_for_mask(
            labels,
            short_scores,
            long_scores,
            refined_scores,
            selected,
            mask,
        )
        for name, mask in masks.items()
    }
    summary = {
        "predictions": str(args.predictions),
        "threshold": threshold,
        "unseen_noise": sorted(unseen_names),
        "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "decomposition.json",
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
