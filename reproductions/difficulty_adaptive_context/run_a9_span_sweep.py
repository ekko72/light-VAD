# -*- coding: utf-8 -*-
"""Run and aggregate the A9 temporal-span/value/cost sweep."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from reproductions.difficulty_adaptive_context.adaptive_model import (
    RF_SPAN_DILATION_PROFILES,
    RefinementConfig,
)
from reproductions.difficulty_adaptive_context.analyze_context_gate import (
    count_frames_by_cluster,
    metrics_from_counts,
    speaker_cluster_bootstrap,
)
from reproductions.difficulty_adaptive_context.data import FRAME_HOP
from reproductions.difficulty_adaptive_context.evaluate_context import (
    CONDITION_ORDER,
    binary_metrics,
)
from reproductions.difficulty_adaptive_context.train_context import UNSEEN_NOISE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
DEFAULT_LIBRISPEECH_ROOT = REPO_ROOT / "data" / "LibriSpeech"
DEFAULT_MANIFEST_ROOT = DEFAULT_DATA_ROOT / "manifests"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "a9_span_sweep"
)
DEFAULT_SHORT_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_short_ext40"
    / "best.pt"
)
DEFAULT_LONG_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_long_ext40"
    / "best.pt"
)
DEFAULT_REFERENCE_PREDICTIONS = (
    REPO_ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "formal_eval_ext40"
    / "frame_predictions.npz"
)


def _threshold_tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _parse_checkpoint_overrides(values: Iterable[str]) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "--reuse-checkpoint must use seed:span:path, "
                f"got {value!r}"
            )
        seed, span, path_text = parts
        key = (int(seed), int(span))
        if key in result:
            raise ValueError(f"duplicate checkpoint override for {key}")
        result[key] = Path(path_text)
    return result


def _checkpoint_span(path: Path) -> tuple[int, tuple[int, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = RefinementConfig.from_dict(dict(payload["refinement_config"]))
    return config.lookback_frames, config.dilations


def _train_command(
    args: argparse.Namespace,
    *,
    span: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "reproductions.difficulty_adaptive_context.train_adaptive",
        "--train-manifest",
        str(args.train_manifest),
        "--val-manifest",
        str(args.val_manifest),
        "--data-root",
        str(args.data_root),
        "--short-checkpoint",
        str(args.short_checkpoint),
        "--long-checkpoint",
        str(args.long_checkpoint),
        "--results-dir",
        str(output_dir),
        "--train-rows",
        str(args.train_rows),
        "--val-rows",
        str(args.val_rows),
        "--context-seconds",
        str(args.context_seconds),
        "--target-seconds",
        str(args.target_seconds),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--log-every",
        str(args.log_every),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--max-lr",
        str(args.max_lr),
        "--min-lr",
        str(args.min_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--training-threshold",
        str(args.training_threshold),
        "--activation-threshold",
        str(args.activation_threshold),
        "--label-weight",
        str(args.label_weight),
        "--distill-weight",
        str(args.distill_weight),
        "--kernel-size",
        "5",
        "--rf-span",
        str(span),
        "--max-residual",
        str(args.max_residual),
        "--exclude-noise-names",
        *args.exclude_noise_names,
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    return command


def _evaluation_command(
    args: argparse.Namespace,
    *,
    seed: int,
    span: int,
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "reproductions.difficulty_adaptive_context.evaluate_adaptive",
        "--manifest",
        str(args.test_manifest),
        "--adaptive-checkpoint",
        str(checkpoint),
        "--long-checkpoint",
        str(args.long_checkpoint),
        "--reference-predictions",
        str(args.reference_predictions),
        "--data-root",
        str(args.data_root),
        "--librispeech-root",
        str(args.librispeech_root),
        "--row-sample",
        str(args.eval_row_sample),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--score-chunk-frames",
        str(args.score_chunk_frames),
        "--fixed-threshold",
        str(args.fixed_threshold),
        "--bootstrap-repeats",
        str(args.bootstrap_repeats),
        "--unseen-noise",
        *args.unseen_noise,
        "--latency-warmup",
        str(args.latency_warmup),
        "--latency-repeats",
        str(args.latency_repeats),
        "--latency-selected-frames",
        str(args.latency_selected_frames),
        "--output-dir",
        str(output_dir),
    ]
    if args.skip_latency_benchmark:
        command.append("--skip-latency-benchmark")
    return command


def _run_training_and_evaluation(
    args: argparse.Namespace,
    *,
    span: int,
    seed: int,
    overrides: dict[tuple[int, int], Path],
) -> tuple[Path, Path]:
    tag = _threshold_tag(args.fixed_threshold)
    seed_dir = args.output_dir / f"seed{seed}"
    train_dir = seed_dir / f"train_rf{span}"
    evaluation_dir = seed_dir / f"eval_rf{span}_fixed{tag}"
    checkpoint = overrides.get((seed, span), train_dir / "best.pt")

    if (seed, span) in overrides:
        lookback, dilations = _checkpoint_span(checkpoint)
        if lookback != span:
            raise ValueError(
                f"override {checkpoint} has lookback {lookback}, "
                f"expected {span}"
            )
        expected_dilations = RF_SPAN_DILATION_PROFILES[span]
        if dilations != expected_dilations:
            raise ValueError(
                f"override {checkpoint} has dilations {dilations}, "
                f"expected {expected_dilations}"
            )
        print(f"reuse seed={seed} span={span}: {checkpoint}", flush=True)
    elif checkpoint.exists() and not args.force:
        print(f"skip existing seed={seed} span={span}: {checkpoint}", flush=True)
    else:
        _run_command(
            _train_command(
                args,
                span=span,
                seed=seed,
                output_dir=train_dir,
            ),
            dry_run=args.dry_run,
        )

    if (
        (evaluation_dir / "results.json").exists()
        and not args.force
        and not args.rerun_evaluations
    ):
        print(
            f"skip existing evaluation seed={seed} span={span}: "
            f"{evaluation_dir}",
            flush=True,
        )
    else:
        _run_command(
            _evaluation_command(
                args,
                seed=seed,
                span=span,
                checkpoint=checkpoint,
                output_dir=evaluation_dir,
            ),
            dry_run=args.dry_run,
        )
    return checkpoint, evaluation_dir


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        required = {
            "labels",
            "short_scores",
            "long_scores",
            "adaptive_scores",
            "full_adaptive_scores",
            "selected",
            "test_mask",
            "speaker_ids",
            "condition",
            "noise_name",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        return {
            name: np.asarray(payload[name]).copy()
            for name in required
        }


def _condition_rows(
    predictions: dict[str, np.ndarray],
    *,
    seed: int,
    span: int,
    protocol: dict[str, Any],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    labels = predictions["labels"]
    short_scores = predictions["short_scores"]
    long_scores = predictions["long_scores"]
    adaptive_scores = predictions["adaptive_scores"]
    refine_scores = predictions["full_adaptive_scores"]
    gated_selected = predictions["selected"]
    all_selected = np.ones_like(gated_selected, dtype=bool)
    test_mask = predictions["test_mask"]
    speaker_ids = predictions["speaker_ids"]
    condition = predictions["condition"]
    noise_name = predictions["noise_name"]
    unseen = {str(value) for value in protocol["unseen_noise"]}

    specs: list[tuple[str, np.ndarray]] = [("test", test_mask)]
    for name in CONDITION_ORDER:
        mask = test_mask & (condition == name)
        if np.any(mask):
            specs.append((name, mask))
    noisy = test_mask & (condition != "clean")
    unseen_mask = noisy & np.isin(noise_name, sorted(unseen))
    seen_mask = noisy & ~unseen_mask
    for name, mask in (("seen", seen_mask), ("unseen", unseen_mask)):
        if np.any(mask):
            specs.append((name, mask))
    for name in sorted(set(str(value) for value in noise_name[test_mask])):
        mask = test_mask & (noise_name == name)
        if name.lower() != "clean" and np.any(mask):
            specs.append((f"noise:{name}", mask))

    rows: list[dict[str, Any]] = []
    latency = protocol.get("cpu_latency") or {}
    for index, (name, mask) in enumerate(specs):
        try:
            _, gated_counts = count_frames_by_cluster(
                labels,
                short_scores,
                adaptive_scores,
                gated_selected,
                speaker_ids,
                mask=mask,
            )
            _, refine_counts = count_frames_by_cluster(
                labels,
                short_scores,
                refine_scores,
                all_selected,
                speaker_ids,
                mask=mask,
            )
        except ValueError:
            continue
        gated_metrics = metrics_from_counts(gated_counts.sum(axis=0))
        refine_metrics = metrics_from_counts(refine_counts.sum(axis=0))
        bootstrap = speaker_cluster_bootstrap(
            [gated_counts],
            cluster_indices=np.arange(gated_counts.shape[0], dtype=np.int64),
            repeats=bootstrap_repeats,
            seed=bootstrap_seed + seed * 10_000 + span * 20 + index,
        )
        adaptive = binary_metrics(labels[mask], adaptive_scores[mask])
        refine = binary_metrics(labels[mask], refine_scores[mask])
        short = binary_metrics(labels[mask], short_scores[mask])
        long = binary_metrics(labels[mask], long_scores[mask])
        rows.append(
            {
                "seed": int(seed),
                "rf_span": int(span),
                "condition": name,
                "frames": int(gated_metrics["frames"]),
                "short_f1": short["f1"],
                "long_f1": long["f1"],
                "refine_only_f1": refine["f1"],
                "adaptive_f1": adaptive["f1"],
                "selected": int(gated_metrics["selected"]),
                "activation_rate": gated_metrics["activation_rate"],
                "correction": int(gated_metrics["correction"]),
                "harm": int(gated_metrics["harm"]),
                "correction_per_selected": gated_metrics[
                    "correction_rate_selected"
                ],
                "harm_per_selected": gated_metrics["harm_rate_selected"],
                "adaptive_net_utility_per_selected": gated_metrics[
                    "net_utility_per_selected"
                ],
                "adaptive_net_utility_per_frame": gated_metrics[
                    "net_utility_per_frame"
                ],
                "net_utility_ci95_low": bootstrap.get(
                    "net_utility_per_selected_ci95_low"
                ),
                "net_utility_ci95_high": bootstrap.get(
                    "net_utility_per_selected_ci95_high"
                ),
                "refine_only_net_utility_per_selected": refine_metrics[
                    "net_utility_per_selected"
                ],
                "refinement_macs_per_selected_frame": protocol[
                    "refinement_macs_per_selected_frame"
                ],
                "refinement_macs_per_call": protocol[
                    "refinement_macs_per_call"
                ],
                "cache_bytes": protocol.get("streaming_cache_bytes"),
                "cpu_median_ms_per_call": latency.get(
                    "median_ms_per_call"
                ),
                "cpu_p95_ms_per_call": latency.get("p95_ms_per_call"),
                "lookback_frames": protocol["lookback_frames"],
                "lookback_seconds": protocol["lookback_seconds"],
            }
        )
    return rows


def _assess_go(rows: list[dict[str, Any]], spans: list[int]) -> dict[str, Any]:
    primary = {
        (int(row["seed"]), int(row["rf_span"])): row
        for row in rows
        if row["condition"] == "test"
    }
    indexed = {
        (
            int(row["seed"]),
            int(row["rf_span"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    seeds = sorted({seed for seed, _ in primary})
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        if 64 not in spans or 384 not in spans:
            continue
        low = primary.get((seed, 64))
        high = primary.get((seed, 384))
        if low is None or high is None:
            continue
        delta = float(
            high["adaptive_net_utility_per_selected"]
            - low["adaptive_net_utility_per_selected"]
        )
        span_utilities = [
            {
                "rf_span": span,
                "adaptive_net_utility_per_selected": indexed[
                    (seed, span, "test")
                ]["adaptive_net_utility_per_selected"],
            }
            for span in sorted(spans)
            if (seed, span, "test") in indexed
        ]
        condition_deltas: dict[str, float] = {}
        condition_names = sorted(
            {
                condition
                for row_seed, row_span, condition in indexed
                if row_seed == seed
                and row_span in (64, 384)
                and condition != "test"
            }
        )
        for condition in condition_names:
            low_row = indexed.get((seed, 64, condition))
            high_row = indexed.get((seed, 384, condition))
            if low_row is None or high_row is None:
                continue
            condition_deltas[condition] = float(
                high_row["adaptive_net_utility_per_selected"]
                - low_row["adaptive_net_utility_per_selected"]
            )
        positive = sum(value > 0 for value in condition_deltas.values())
        required = max(1, (len(condition_deltas) + 1) // 2)
        seed_results.append(
            {
                "seed": seed,
                "u384_minus_u64": delta,
                "span_utilities": span_utilities,
                "condition_deltas": condition_deltas,
                "positive_condition_count": positive,
                "condition_count": len(condition_deltas),
                "conditions_required": required,
                "provisional_go": bool(
                    delta > 0 and positive >= required
                ),
            }
        )
    return {
        "spans": sorted(spans),
        "seed_results": seed_results,
        "all_seed_provisional_go": (
            bool(seed_results)
            and all(result["provisional_go"] for result in seed_results)
        ),
        "note": (
            "This is an A9 mechanism assessment. The overall A9 GO still "
            "depends on a stable transition/saturation pattern and any "
            "multi-seed replication requested by the experiment protocol."
        ),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["condition"] == "test"]


def _md_integer(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def _md_float(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_markdown(
    rows: list[dict[str, Any]],
    assessment: dict[str, Any],
    path: Path,
) -> None:
    primary = _primary_rows(rows)
    lines = [
        "# A9 Temporal Span-Value-Cost Sweep",
        "",
        "## Primary Test",
        "",
        "| Seed | RF | Lookback | Adaptive F1 | Refine F1 | Short F1 | "
        "Long F1 | Activation | Correction/selected | Harm/selected | "
        "Net/selected | 95% CI | Net/frame |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        primary,
        key=lambda value: (int(value["seed"]), int(value["rf_span"])),
    ):
        lines.append(
            f"| {row['seed']} | {row['rf_span']} | "
            f"{row['lookback_seconds']:.2f}s | "
            f"{row['adaptive_f1']:.5f} | {row['refine_only_f1']:.5f} | "
            f"{row['short_f1']:.5f} | {row['long_f1']:.5f} | "
            f"{row['activation_rate'] * 100.0:.3f}% | "
            f"{row['correction_per_selected'] * 100.0:.3f}% | "
            f"{row['harm_per_selected'] * 100.0:.3f}% | "
            f"{row['adaptive_net_utility_per_selected'] * 100.0:.3f}% | "
            f"[{row['net_utility_ci95_low'] * 100.0:.3f}%, "
            f"{row['net_utility_ci95_high'] * 100.0:.3f}%] | "
            f"{row['adaptive_net_utility_per_frame'] * 100.0:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Cost",
            "",
            "| RF | MACs/selected frame | MACs/call | Cache bytes | "
            "CPU median ms/call | CPU p95 ms/call |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seen_spans: set[int] = set()
    for row in primary:
        span = int(row["rf_span"])
        if span in seen_spans:
            continue
        seen_spans.add(span)
        lines.append(
            f"| {span} | "
            f"{_md_integer(row['refinement_macs_per_selected_frame'])} | "
            f"{_md_integer(row['refinement_macs_per_call'])} | "
            f"{_md_integer(row['cache_bytes'])} | "
            f"{_md_float(row['cpu_median_ms_per_call'])} | "
            f"{_md_float(row['cpu_p95_ms_per_call'])} |"
        )

    lines.extend(
        [
            "",
            "## By Condition",
            "",
            "| Seed | RF | Condition | Frames | Adaptive F1 | Refine F1 | "
            "Activation | Net/selected |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows,
        key=lambda value: (
            int(value["seed"]),
            int(value["rf_span"]),
            str(value["condition"]),
        ),
    ):
        if row["condition"] == "test":
            continue
        lines.append(
            f"| {row['seed']} | {row['rf_span']} | {row['condition']} | "
            f"{row['frames']} | {row['adaptive_f1']:.5f} | "
            f"{row['refine_only_f1']:.5f} | "
            f"{row['activation_rate'] * 100.0:.3f}% | "
            f"{row['adaptive_net_utility_per_selected'] * 100.0:.3f}% |"
        )

    lines.extend(["", "## GO Assessment", ""])
    for result in assessment["seed_results"]:
        lines.append(
            f"- Seed {result['seed']}: U384-U64 = "
            f"{result['u384_minus_u64'] * 100.0:.3f} percentage points; "
            f"positive SNR/domain cells "
            f"{result['positive_condition_count']}/"
            f"{result['condition_count']}; provisional GO: "
            f"{result['provisional_go']}."
        )
    lines.extend(
        [
            f"- All evaluated seeds pass the provisional rule: "
            f"{assessment['all_seed_provisional_go']}.",
            f"- {assessment['note']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = sorted(
        _primary_rows(rows),
        key=lambda value: (int(value["seed"]), int(value["rf_span"])),
    )
    if not primary:
        return
    seed = int(primary[0]["seed"])
    primary = [row for row in primary if int(row["seed"]) == seed]
    spans = [int(row["rf_span"]) for row in primary]
    _, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    condition_names = [
        "test",
        *CONDITION_ORDER,
        "seen",
        "unseen",
    ]
    for condition in condition_names:
        condition_rows = sorted(
            (
                row
                for row in rows
                if int(row["seed"]) == seed
                and row["condition"] == condition
            ),
            key=lambda value: int(value["rf_span"]),
        )
        if not condition_rows:
            continue
        axes[0].plot(
            [int(row["rf_span"]) for row in condition_rows],
            [
                float(row["adaptive_net_utility_per_selected"])
                for row in condition_rows
            ],
            marker="o",
            label=condition,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Gate utility")
    axes[0].set_xlabel("RF span (frames)")
    axes[0].set_ylabel("Net utility / selected")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].plot(
        spans,
        [float(row["adaptive_f1"]) for row in primary],
        marker="o",
        label="Adaptive",
    )
    axes[1].plot(
        spans,
        [float(row["refine_only_f1"]) for row in primary],
        marker="s",
        label="Always refine",
    )
    axes[1].plot(
        spans,
        [float(row["short_f1"]) for row in primary],
        linestyle="--",
        marker=".",
        label="Short",
    )
    axes[1].plot(
        spans,
        [float(row["long_f1"]) for row in primary],
        linestyle=":",
        marker=".",
        label="Long teacher",
    )
    axes[1].set_title("F1")
    axes[1].set_xlabel("RF span (frames)")
    axes[1].set_ylabel("F1")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        spans,
        [float(row["cache_bytes"]) / 1024.0 for row in primary],
        color="tab:red",
        marker="o",
    )
    axes[2].set_xlabel("RF span (frames)")
    axes[2].set_ylabel("Cache KiB", color="tab:red")
    axes[2].tick_params(axis="y", labelcolor="tab:red")
    latency_axis = axes[2].twinx()
    latency_values = [row["cpu_median_ms_per_call"] for row in primary]
    if all(value is not None for value in latency_values):
        latency_axis.plot(
            spans,
            [float(value) for value in latency_values],
            color="tab:blue",
            marker="s",
        )
        latency_axis.set_ylabel("CPU ms/call", color="tab:blue")
        latency_axis.tick_params(axis="y", labelcolor="tab:blue")
    else:
        latency_axis.set_axis_off()
    axes[2].set_title("Cost")

    for axis in axes:
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def aggregate(args: argparse.Namespace) -> int:
    tag = _threshold_tag(args.fixed_threshold)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in args.seeds:
        for span in args.spans:
            evaluation_dir = (
                args.output_dir
                / f"seed{seed}"
                / f"eval_rf{span}_fixed{tag}"
            )
            results_path = evaluation_dir / "results.json"
            predictions_path = evaluation_dir / "frame_predictions.npz"
            if not results_path.exists() or not predictions_path.exists():
                missing.append(str(evaluation_dir))
                continue
            with open(results_path, "r", encoding="utf-8") as handle:
                results = json.load(handle)
            rows.extend(
                _condition_rows(
                    _load_prediction(predictions_path),
                    seed=seed,
                    span=span,
                    protocol=results["protocol"],
                    bootstrap_repeats=args.bootstrap_repeats,
                    bootstrap_seed=args.bootstrap_seed,
                )
            )
    if missing:
        raise RuntimeError(
            "A9 evaluations are missing; run evaluation first: "
            + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("A9 aggregation produced no rows")

    assessment = _assess_go(rows, list(args.spans))
    summary = {
        "protocol": {
            "seeds": list(args.seeds),
            "spans": list(args.spans),
            "fixed_threshold": float(args.fixed_threshold),
            "test_manifest": str(args.test_manifest),
            "reference_predictions": str(args.reference_predictions),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "go_assessment": assessment,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_dir / "a9_span_sweep.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(rows, args.output_dir / "a9_span_sweep.csv")
    _write_markdown(
        rows,
        assessment,
        args.output_dir / "a9_span_sweep.md",
    )
    _write_plot(rows, args.output_dir / "a9_span_sweep.png")
    print(f"A9 artifacts written to {args.output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A9 temporal-span/value/cost sweep."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "train", "evaluate", "aggregate"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--spans", type=int, nargs="+", default=[64, 128, 256, 384, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument(
        "--reuse-checkpoint",
        action="append",
        default=[],
        metavar="SEED:SPAN:PATH",
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "LibriSpeech_train_medium.tsv")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "LibriSpeech_val_medium.tsv")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "LibriSpeech_test_medium.tsv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--librispeech-root", type=Path, default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--short-checkpoint", type=Path, default=DEFAULT_SHORT_CHECKPOINT)
    parser.add_argument("--long-checkpoint", type=Path, default=DEFAULT_LONG_CHECKPOINT)
    parser.add_argument("--reference-predictions", type=Path, default=DEFAULT_REFERENCE_PREDICTIONS)
    parser.add_argument("--train-rows", type=int, default=8_640)
    parser.add_argument("--val-rows", type=int, default=432)
    parser.add_argument("--eval-row-sample", type=int, default=1_080)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--target-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--training-threshold", type=float, default=0.13)
    parser.add_argument("--activation-threshold", type=float, default=0.13)
    parser.add_argument("--fixed-threshold", type=float, default=0.13)
    parser.add_argument("--label-weight", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--max-residual", type=float, default=2.0)
    parser.add_argument("--score-chunk-frames", type=int, default=2_000)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--latency-selected-frames", type=int, default=64)
    parser.add_argument("--skip-latency-benchmark", action="store_true")
    parser.add_argument("--exclude-noise-names", nargs="*", default=list(UNSEEN_NOISE))
    parser.add_argument("--unseen-noise", nargs="*", default=list(UNSEEN_NOISE))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun-evaluations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    unsupported = sorted(set(args.spans) - set(RF_SPAN_DILATION_PROFILES))
    if unsupported:
        parser.error(f"unsupported --spans values: {unsupported}")
    if not args.spans:
        parser.error("--spans must not be empty")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    if not 0.0 <= args.fixed_threshold <= 0.5:
        parser.error("--fixed-threshold must be in [0, 0.5]")
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")
    for value in (
        args.train_rows,
        args.val_rows,
        args.eval_row_sample,
        args.batch_size,
        args.epochs,
        args.latency_repeats,
        args.latency_selected_frames,
    ):
        if value <= 0:
            parser.error("row, batch, epoch and latency counts must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive when provided")
    if args.latency_warmup < 0:
        parser.error("--latency-warmup must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        overrides = _parse_checkpoint_overrides(args.reuse_checkpoint)
    except ValueError as error:
        parser.error(str(error))

    if args.mode in ("all", "train"):
        for seed in args.seeds:
            for span in args.spans:
                _run_training_and_evaluation(
                    args,
                    span=span,
                    seed=seed,
                    overrides=overrides,
                )
    elif args.mode == "evaluate":
        for seed in args.seeds:
            for span in args.spans:
                tag = _threshold_tag(args.fixed_threshold)
                train_dir = (
                    args.output_dir / f"seed{seed}" / f"train_rf{span}"
                )
                checkpoint = overrides.get(
                    (seed, span),
                    train_dir / "best.pt",
                )
                evaluation_dir = (
                    args.output_dir
                    / f"seed{seed}"
                    / f"eval_rf{span}_fixed{tag}"
                )
                if (
                    (evaluation_dir / "results.json").exists()
                    and not args.force
                    and not args.rerun_evaluations
                ):
                    print(
                        f"skip existing evaluation seed={seed} span={span}: "
                        f"{evaluation_dir}",
                        flush=True,
                    )
                    continue
                _run_command(
                    _evaluation_command(
                        args,
                        seed=seed,
                        span=span,
                        checkpoint=checkpoint,
                        output_dir=evaluation_dir,
                    ),
                    dry_run=args.dry_run,
                )

    if args.dry_run and args.mode in ("all", "aggregate"):
        print("dry run: skipping aggregation", flush=True)
    elif args.mode in ("all", "aggregate"):
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
