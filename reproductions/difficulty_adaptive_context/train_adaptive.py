# -*- coding: utf-8 -*-
"""Train the Phase A3 sparse refinement on top of a frozen Short model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from reproductions.difficulty_adaptive_context.adaptive_model import (
    AdaptiveCausalVAD,
    RefinementConfig,
    SparseCausalMultiScaleRefinement,
    confidence_from_logits,
    selection_from_confidence,
    speech_probabilities,
)
from reproductions.difficulty_adaptive_context.data import FrameChunkDataset
from reproductions.difficulty_adaptive_context.evaluate_context import (
    load_frame_model,
)
from reproductions.difficulty_adaptive_context.train_context import (
    UNSEEN_NOISE,
    binary_metrics,
    build_loader,
    set_seed,
)
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.train import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"


def binary_utility_metrics(
    labels: torch.Tensor,
    short_logits: torch.Tensor,
    adaptive_logits: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float | int]:
    """Count correction, harm and signed utility in a target-frame batch."""
    labels = labels.reshape(-1)
    short_prediction = (
        short_logits[:, 1].reshape(-1) >= short_logits[:, 0].reshape(-1)
    )
    adaptive_prediction = (
        adaptive_logits[:, 1].reshape(-1) >= adaptive_logits[:, 0].reshape(-1)
    )
    selected_flat = selected.reshape(-1)
    truth = labels >= 1
    correct_short = short_prediction == truth
    correct_adaptive = adaptive_prediction == truth
    correction = int(
        torch.count_nonzero(
            selected_flat & (~correct_short) & correct_adaptive
        ).item()
    )
    harm = int(
        torch.count_nonzero(
            selected_flat & correct_short & (~correct_adaptive)
        ).item()
    )
    selected_count = int(torch.count_nonzero(selected_flat).item())
    return {
        "frames": int(labels.numel()),
        "selected": selected_count,
        "correction": correction,
        "harm": harm,
        "signed_utility": correction - harm,
        "net_utility_per_selected": (
            float(correction - harm) / selected_count
            if selected_count
            else 0.0
        ),
    }


def train_epoch(
    model: AdaptiveCausalVAD,
    teacher: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    frontend: nn.Module,
    device: torch.device,
    *,
    context_samples: int,
    training_threshold: float,
    label_weight: float,
    distill_weight: float,
    log_every: int,
    max_steps: int | None,
    start_step: int,
) -> tuple[dict[str, Any], int]:
    model.train()
    model.freeze_short_model()
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)

    loss_sum = 0.0
    label_loss_sum = 0.0
    distill_loss_sum = 0.0
    selected_sum = 0
    frame_sum = 0
    global_step = start_step

    target_start = int(context_samples) // 160
    for waveforms, labels in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = frontend(waveforms)

        with torch.no_grad():
            encoded = model.short_model.encoder(features)
            short_logits = model.short_model.classifier(encoded)
            short_target = short_logits[:, :, target_start:]
            selected = selection_from_confidence(
                confidence_from_logits(short_target),
                training_threshold,
            )
            if teacher is None:
                teacher_margin = None
            else:
                teacher_logits = teacher(features)[:, :, target_start:]
                teacher_margin = (
                    teacher_logits[:, 1] - teacher_logits[:, 0]
                )

        # Replace the dense all-false mask with the actual target-only mask.
        # Constructing the full mask explicitly keeps the refinement stateless.
        full_selected = torch.zeros(
            (encoded.shape[0], encoded.shape[-1]),
            dtype=torch.bool,
            device=encoded.device,
        )
        full_selected[:, target_start:] = selected
        residual = model.refinement.forward_sparse(encoded, full_selected)
        adaptive_logits = short_logits + residual
        adaptive_target = adaptive_logits[:, :, target_start:]

        if not bool(selected.any()):
            global_step += 1
            if log_every > 0 and global_step % log_every == 0:
                print(
                    f"  step {global_step}: no selected frames, skipped",
                    flush=True,
                )
            continue

        student_margin = (
            adaptive_target[:, 1] - adaptive_target[:, 0]
        )[selected]
        target_labels = labels[selected]
        label_loss = F.binary_cross_entropy_with_logits(
            student_margin,
            target_labels.to(dtype=student_margin.dtype),
        )
        if teacher_margin is None or distill_weight <= 0:
            distill_loss = student_margin.new_zeros(())
        else:
            soft_targets = torch.sigmoid(teacher_margin)[selected]
            distill_loss = F.binary_cross_entropy_with_logits(
                student_margin,
                soft_targets,
            )
        loss = label_weight * label_loss + distill_weight * distill_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        selected_count = int(selected.sum().item())
        selected_sum += selected_count
        frame_sum += int(labels.numel())
        loss_sum += float(loss.detach().cpu()) * selected_count
        label_loss_sum += float(label_loss.detach().cpu()) * selected_count
        distill_loss_sum += float(distill_loss.detach().cpu()) * selected_count
        global_step += 1

        if log_every > 0 and global_step % log_every == 0:
            denominator = max(selected_sum, 1)
            print(
                f"  step {global_step}: "
                f"loss={loss_sum / denominator:.5f}, "
                f"label={label_loss_sum / denominator:.5f}, "
                f"distill={distill_loss_sum / denominator:.5f}, "
                f"selected={selected_sum / max(frame_sum, 1):.3%}",
                flush=True,
            )
        if max_steps is not None and global_step >= max_steps:
            break

    denominator = max(selected_sum, 1)
    return (
        {
            "loss": loss_sum / denominator,
            "label_loss": label_loss_sum / denominator,
            "distill_loss": distill_loss_sum / denominator,
            "selected_frames": selected_sum,
            "target_frames": frame_sum,
            "activation_rate": selected_sum / max(frame_sum, 1),
        },
        global_step,
    )


@torch.inference_mode()
def validate(
    model: AdaptiveCausalVAD,
    teacher: nn.Module | None,
    loader: DataLoader,
    frontend: nn.Module,
    device: torch.device,
    *,
    context_samples: int,
    activation_threshold: float,
) -> dict[str, Any]:
    model.eval()
    model.freeze_short_model()
    if teacher is not None:
        teacher.eval()

    labels_parts: list[np.ndarray] = []
    short_parts: list[np.ndarray] = []
    adaptive_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    utility_totals: dict[str, float | int] = {
        "frames": 0,
        "selected": 0,
        "correction": 0,
        "harm": 0,
        "signed_utility": 0,
    }
    target_start = int(context_samples) // 160

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = frontend(waveforms)
        output = model.forward_components(
            features,
            activation_threshold=activation_threshold,
        )
        short_target = output.short_logits[:, :, target_start:]
        adaptive_target = output.logits[:, :, target_start:]
        selected = output.selected[:, target_start:]
        if teacher is not None:
            teacher_logits = teacher(features)[:, :, target_start:]
        else:
            teacher_logits = short_target.new_empty(0)

        labels_parts.append(labels.detach().cpu().numpy().reshape(-1))
        short_parts.append(
            speech_probabilities(short_target).detach().cpu().numpy().reshape(-1)
        )
        adaptive_parts.append(
            speech_probabilities(adaptive_target)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        if teacher is not None:
            teacher_parts.append(
                speech_probabilities(teacher_logits)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        utility = binary_utility_metrics(
            labels,
            short_target,
            adaptive_target,
            selected,
        )
        for key in utility_totals:
            utility_totals[key] += utility[key]

    labels_array = np.concatenate(labels_parts)
    short_array = np.concatenate(short_parts)
    adaptive_array = np.concatenate(adaptive_parts)
    result = {
        "short": binary_metrics(labels_array, short_array),
        "adaptive": binary_metrics(labels_array, adaptive_array),
        "utility": utility_totals,
    }
    result["utility"]["net_utility_per_selected"] = (
        float(utility_totals["signed_utility"])
        / max(int(utility_totals["selected"]), 1)
    )
    if teacher_parts:
        teacher_array = np.concatenate(teacher_parts)
        result["teacher"] = binary_metrics(labels_array, teacher_array)
        result["teacher_agreement"] = float(
            np.mean((adaptive_array >= 0.5) == (teacher_array >= 0.5))
        )
    return result


def save_checkpoint(
    path: Path,
    *,
    model: AdaptiveCausalVAD,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    refinement_config: RefinementConfig,
    short_config: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "refinement_config": refinement_config.to_dict(),
            "short_model_config": short_config,
            "history": history,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Phase A3 sparse causal refinement."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--short-checkpoint", type=Path, required=True)
    parser.add_argument("--long-checkpoint", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=8_640)
    parser.add_argument("--val-rows", type=int, default=432)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--target-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--training-threshold", type=float, default=0.25)
    parser.add_argument("--activation-threshold", type=float, default=0.24)
    parser.add_argument("--label-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--max-residual", type=float, default=2.0)
    parser.add_argument(
        "--exclude-noise-names",
        nargs="*",
        default=list(UNSEEN_NOISE),
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.train_rows is not None and args.train_rows <= 0:
        parser.error("--train-rows must be positive")
    if args.val_rows is not None and args.val_rows <= 0:
        parser.error("--val-rows must be positive")
    if args.context_seconds <= 0 or args.target_seconds <= 0:
        parser.error("context and target durations must be positive")
    if not 0.0 <= args.training_threshold <= 0.5:
        parser.error("--training-threshold must be in [0, 0.5]")
    if not 0.0 <= args.activation_threshold <= 0.5:
        parser.error("--activation-threshold must be in [0, 0.5]")
    if args.label_weight < 0 or args.distill_weight < 0:
        parser.error("loss weights must be non-negative")
    if args.label_weight == 0 and args.distill_weight == 0:
        parser.error("at least one loss weight must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    context_samples = int(round(args.context_seconds * 16_000))
    target_samples = int(round(args.target_seconds * 16_000))
    if context_samples % 160 or target_samples % 160:
        parser.error("context and target durations must align to 10 ms frames")

    unseen = tuple(
        name for name in args.exclude_noise_names if str(name).strip()
    )
    train_dataset = FrameChunkDataset(
        args.train_manifest,
        generated_root=args.data_root / "generated",
        label_root=args.data_root / "labels",
        context_samples=context_samples,
        target_samples=target_samples,
        row_sample=args.train_rows,
        seed=args.seed,
        exclude_noise_names=set(unseen),
        resample_chunks=True,
    )
    val_dataset = FrameChunkDataset(
        args.val_manifest,
        generated_root=args.data_root / "generated",
        label_root=args.data_root / "labels",
        context_samples=context_samples,
        target_samples=target_samples,
        row_sample=args.val_rows,
        seed=args.seed + 1,
        exclude_noise_names=set(unseen),
        sample_offset=10_000,
        resample_chunks=False,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("adaptive datasets are empty")
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 1,
    )

    short_model, _, short_config = load_frame_model(
        args.short_checkpoint,
        device,
    )
    if not bool(short_config.get("causal", False)):
        raise ValueError("Short checkpoint must be causal")
    if not bool(short_config.get("frame_output", False)):
        raise ValueError("Short checkpoint must have frame outputs")
    short_model.requires_grad_(False)
    short_model.eval()
    refinement_config = RefinementConfig(
        in_channels=128,
        num_classes=2,
        kernel_size=args.kernel_size,
        dilations=tuple(int(value) for value in args.dilations),
        max_residual=args.max_residual,
    )
    refinement = SparseCausalMultiScaleRefinement(refinement_config)
    model = AdaptiveCausalVAD(short_model, refinement).to(device)
    model.freeze_short_model()

    teacher: nn.Module | None = None
    if args.long_checkpoint is not None:
        teacher, _, _ = load_frame_model(args.long_checkpoint, device)
        teacher.eval()
        teacher.requires_grad_(False)

    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    optimizer = torch.optim.AdamW(
        model.refinement.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * len(train_loader), 1),
        eta_min=args.min_lr,
    )

    print(
        f"device={device}, short={args.short_checkpoint}, "
        f"teacher={args.long_checkpoint}, "
        f"train_chunks={len(train_dataset)}, "
        f"val_chunks={len(val_dataset)}, "
        f"refinement_params={sum(p.numel() for p in model.refinement.parameters()):,}, "
        f"lookback={model.refinement.lookback_frames} frames, "
        f"training_threshold={args.training_threshold:.4f}, "
        f"activation_threshold={args.activation_threshold:.4f}",
        flush=True,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_utility = -math.inf
    global_step = 0
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        train_metrics, global_step = train_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            frontend,
            device,
            context_samples=context_samples,
            training_threshold=args.training_threshold,
            label_weight=args.label_weight,
            distill_weight=args.distill_weight,
            log_every=args.log_every,
            max_steps=args.max_steps,
            start_step=global_step,
        )
        val_metrics = validate(
            model,
            teacher,
            val_loader,
            frontend,
            device,
            context_samples=context_samples,
            activation_threshold=args.activation_threshold,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        utility_key = "net_utility_per_selected"
        utility = float(val_metrics["utility"][utility_key])
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"selected={train_metrics['activation_rate']:.3%} "
            f"val_short_f1={val_metrics['short']['f1']:.5f} "
            f"val_adaptive_f1={val_metrics['adaptive']['f1']:.5f} "
            f"val_activation={val_metrics['utility']['selected'] / max(val_metrics['utility']['frames'], 1):.3%} "
            f"val_utility={utility:.5f}",
            flush=True,
        )
        improved = utility > best_utility
        best_utility = max(best_utility, utility)
        save_checkpoint(
            args.results_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            refinement_config=refinement_config,
            short_config=short_config,
            history=history,
        )
        if improved:
            save_checkpoint(
                args.results_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                refinement_config=refinement_config,
                short_config=short_config,
                history=history,
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    summary = {
        "config": vars(args),
        "short_model_config": short_config,
        "refinement_config": refinement_config.to_dict(),
        "unseen_noise": list(unseen),
        "history": history,
        "best_utility_per_selected": best_utility,
        "global_step": global_step,
    }
    with open(
        args.results_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"artifacts written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
