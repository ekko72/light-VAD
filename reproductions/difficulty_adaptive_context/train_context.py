# -*- coding: utf-8 -*-
"""Train the paired Short-RF and Long-RF frame-level MarbleNet models."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from reproductions.difficulty_adaptive_context.data import FrameChunkDataset
from reproductions.marblenet_vad.features import MfccConfig, MfccFrontend
from reproductions.marblenet_vad.model import (
    DILATION_PROFILES,
    build_marblenet_3x2x64,
    count_parameters,
    receptive_field,
)
from reproductions.marblenet_vad.train import learning_rate_at, resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "librivad"
UNSEEN_NOISE = ("SSN_noise", "Street_noise", "Transport_noise")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: FrameChunkDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def frame_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    context_samples: int,
    criterion: nn.Module,
) -> torch.Tensor:
    target_start = int(context_samples) // 160
    target_logits = logits[:, :, target_start:]
    if target_logits.shape[-1] != labels.shape[-1]:
        raise ValueError(
            "target frame mismatch: "
            f"{target_logits.shape[-1]} logits vs {labels.shape[-1]} labels"
        )
    return criterion(
        target_logits.transpose(1, 2).reshape(-1, target_logits.shape[1]),
        labels.reshape(-1),
    )


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predictions = scores >= 0.5
    result: dict[str, float | int | None] = {
        "frames": int(labels.size),
        "speech": int(np.count_nonzero(labels == 1)),
        "silence": int(np.count_nonzero(labels == 0)),
        "error": (
            float(np.mean(predictions != labels)) if labels.size else None
        ),
        "f1": (
            float(f1_score(labels, predictions, zero_division=0))
            if labels.size and np.unique(labels).size > 1
            else None
        ),
        "auc": (
            float(roc_auc_score(labels, scores))
            if labels.size and np.unique(labels).size > 1
            else None
        ),
    }
    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    frontend: nn.Module,
    device: torch.device,
    *,
    context_samples: int,
    start_step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_ratio: float,
    hold_ratio: float,
    decay_power: float,
    log_every: int,
    max_steps: int | None,
) -> tuple[dict[str, Any], int]:
    model.train()
    loss_sum = 0.0
    frame_count = 0
    global_step = start_step
    for waveforms, labels in loader:
        if max_steps is not None and global_step >= max_steps:
            break
        lr = learning_rate_at(
            global_step,
            total_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_ratio=warmup_ratio,
            hold_ratio=hold_ratio,
            power=decay_power,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        features = frontend(waveforms.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = frame_loss(
            logits,
            labels,
            context_samples=context_samples,
            criterion=criterion,
        )
        loss.backward()
        optimizer.step()

        batch_frames = int(labels.numel())
        loss_sum += float(loss.detach().cpu()) * batch_frames
        frame_count += batch_frames
        global_step += 1
        if log_every > 0 and global_step % log_every == 0:
            print(
                f"  step {global_step}/{total_steps}: "
                f"loss={loss_sum / max(frame_count, 1):.5f}, lr={lr:.6f}",
                flush=True,
            )
        if max_steps is not None and global_step >= max_steps:
            break
    return (
        {
            "loss": loss_sum / max(frame_count, 1),
            "frames": frame_count,
        },
        global_step,
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    frontend: nn.Module,
    device: torch.device,
    *,
    context_samples: int,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    frame_count = 0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for waveforms, labels in loader:
        features = frontend(waveforms.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = frame_loss(
            logits,
            labels,
            context_samples=context_samples,
            criterion=criterion,
        )
        target_start = context_samples // 160
        probabilities = torch.softmax(
            logits[:, :, target_start:].transpose(1, 2), dim=-1
        )[..., 1]
        loss_sum += float(loss.detach().cpu()) * int(labels.numel())
        frame_count += int(labels.numel())
        all_labels.append(labels.detach().cpu().numpy().reshape(-1))
        all_scores.append(probabilities.detach().cpu().numpy().reshape(-1))

    labels_array = (
        np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    )
    scores_array = (
        np.concatenate(all_scores) if all_scores else np.empty(0, dtype=float)
    )
    result = binary_metrics(labels_array, scores_array)
    result["loss"] = loss_sum / max(frame_count, 1)
    return result


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "model_config": model_config,
            "history": history,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frame-level causal MarbleNet context variant."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--rf-profile",
        choices=("short", "long"),
        required=True,
    )
    parser.add_argument("--train-rows", type=int, default=8_640)
    parser.add_argument("--val-rows", type=int, default=432)
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--target-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--fixed-train-chunks",
        action="store_true",
        help="Reuse one chunk start per row instead of resampling each epoch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-lr", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--hold-ratio", type=float, default=0.25)
    parser.add_argument("--decay-power", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument(
        "--exclude-noise-names",
        nargs="*",
        default=list(UNSEEN_NOISE),
        help=(
            "Noise classes excluded from training for A5. Pass an empty "
            "list to train on all classes."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer and history from a checkpoint.",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
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
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.hold_ratio < 1.0:
        parser.error("--hold-ratio must be in [0, 1)")
    if args.warmup_ratio + args.hold_ratio >= 1.0:
        parser.error("warmup + hold ratios must be less than 1")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    context_samples = int(round(args.context_seconds * 16_000))
    target_samples = int(round(args.target_seconds * 16_000))
    largest_lookback = int(receptive_field("long")["lookback_frames"]) * 160
    if context_samples < largest_lookback:
        parser.error(
            "--context-seconds must cover the Long-RF lookback "
            f"({largest_lookback / 16_000:.2f} seconds)"
        )

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
        resample_chunks=not args.fixed_train_chunks,
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
        raise RuntimeError("context datasets are empty")

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

    model_config = {
        "feat_in": 64,
        "num_classes": 2,
        "repeat": 2,
        "channels": 64,
        "dropout": 0.0,
        "causal": True,
        "frame_output": True,
        "dilation_profile": args.rf_profile,
        "frontend": "mfcc_causal",
        "context_samples": context_samples,
        "target_samples": target_samples,
    }
    model = build_marblenet_3x2x64(
        feat_in=64,
        num_classes=2,
        dropout=0.0,
        causal=True,
        dilation_profile=args.rf_profile,
        frame_output=True,
    ).to(device)
    frontend = MfccFrontend(MfccConfig(causal=True)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.max_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
        payload = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))

    steps_per_epoch = len(train_loader)
    total_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(args.epochs) * steps_per_epoch
    )
    rf = receptive_field(args.rf_profile)
    print(
        f"device={device}, profile={args.rf_profile}, "
        f"rf={rf['receptive_field_frames']} frames "
        f"({rf['receptive_field_seconds']:.2f} s), "
        f"parameters={count_parameters(model):,}, "
        f"train_chunks={len(train_dataset)}, "
        f"val_chunks={len(val_dataset)}, "
        f"unseen_noise={unseen}, total_steps={total_steps}",
        flush=True,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = -math.inf
    for record in history:
        value = record.get("val", {}).get("auc")
        if value is not None:
            best_val_auc = max(best_val_auc, float(value))
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_dataset.set_epoch(epoch)
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            frontend,
            device,
            context_samples=context_samples,
            start_step=global_step,
            total_steps=total_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
            hold_ratio=args.hold_ratio,
            decay_power=args.decay_power,
            log_every=args.log_every,
            max_steps=args.max_steps,
        )
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            frontend,
            device,
            context_samples=context_samples,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_error={val_metrics['error']:.5f} "
            f"val_f1={val_metrics['f1']:.5f} "
            f"val_auc={val_metrics['auc']:.5f}",
            flush=True,
        )

        improved = False
        if val_metrics["auc"] is not None:
            improved = float(val_metrics["auc"]) > best_val_auc
            best_val_auc = max(best_val_auc, float(val_metrics["auc"]))
        save_checkpoint(
            args.results_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            history=history,
            model_config=model_config,
        )
        if improved:
            save_checkpoint(
                args.results_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                history=history,
                model_config=model_config,
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    summary = {
        "config": vars(args),
        "model_config": model_config,
        "receptive_field": rf,
        "unseen_noise": list(unseen),
        "history": history,
        "last_epoch": last_epoch,
        "global_step": global_step,
    }
    with open(
        args.results_dir / "history.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"artifacts written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
