# -*- coding: utf-8 -*-
"""Train MarbleNet-3x2x64 on LibriVAD-style manifests.

The model and feature pipeline follow the MarbleNet paper and NeMo's
``marblenet_3x2x64.yaml``.  LibriVAD-style mixtures are used as the local
adaptation of the paper's SCF train/validation/test protocol:

* each sample is cut into 0.63 s windows;
* a window label is the majority vote of its sample-level labels;
* training windows are class-balanced by default.

The training recipe keeps the paper's SGD momentum, weight decay,
Warmup-Hold-Decay schedule, and waveform/spectrogram augmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from .augment import (
        FeatureAugmentor,
        SpecAugment,
        SpecCutout,
        WaveformAugmentor,
    )
    from .dataset import LibriVADSegments
    from .features import MfccConfig, MfccFrontend
    from .model import (
        DILATION_PROFILES,
        build_marblenet_3x2x64,
        count_parameters,
        receptive_field,
    )
except ImportError:
    # Allows direct execution with:
    # python reproductions\marblenet_vad\train.py
    from augment import (
        FeatureAugmentor,
        SpecAugment,
        SpecCutout,
        WaveformAugmentor,
    )
    from dataset import LibriVADSegments
    from features import MfccConfig, MfccFrontend
    from model import (
        DILATION_PROFILES,
        build_marblenet_3x2x64,
        count_parameters,
        receptive_field,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "marblenet_vad"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable smoke runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def learning_rate_at(
    step: int,
    total_steps: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_ratio: float,
    hold_ratio: float,
    power: float,
) -> float:
    """NeMo-style Warmup-Hold-Decay learning rate for one optimizer step.

    ``step`` is zero-based.  The warmup is linear from zero to ``max_lr``;
    the hold phase stays at ``max_lr``; the decay phase uses a polynomial
    from ``max_lr`` to ``min_lr``.
    """
    if total_steps <= 1:
        return float(max_lr)
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= hold_ratio < 1.0:
        raise ValueError("hold_ratio must be in [0, 1)")
    if warmup_ratio + hold_ratio >= 1.0:
        raise ValueError("warmup_ratio + hold_ratio must be less than 1")

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    hold_steps = max(0, int(total_steps * hold_ratio))
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    step = min(max(int(step), 0), total_steps - 1)

    if step < warmup_steps:
        return float(max_lr * (step + 1) / warmup_steps)
    if step < warmup_steps + hold_steps:
        return float(max_lr)
    decay_step = step - warmup_steps - hold_steps
    progress = min(1.0, decay_step / decay_steps)
    return float(
        (max_lr - min_lr) * (1.0 - progress) ** power + min_lr
    )


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC-AUC, or ``None`` when an evaluation slice has one class."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def build_data_loader(
    dataset: LibriVADSegments,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    frontend: nn.Module,
    waveform_augmentor: WaveformAugmentor | None,
    feature_augmentor: FeatureAugmentor | None,
    device: torch.device,
    *,
    start_step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_ratio: float,
    hold_ratio: float,
    power: float,
    log_every: int,
    max_steps: int | None,
) -> tuple[dict[str, Any], int]:
    """Run one training pass and return metrics plus the next global step."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    global_step = start_step

    for batch_index, (waveforms, labels) in enumerate(loader):
        if max_steps is not None and global_step >= max_steps:
            break

        lr = learning_rate_at(
            global_step,
            total_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_ratio=warmup_ratio,
            hold_ratio=hold_ratio,
            power=power,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Keep waveform augmentation on CPU, then move features to the
        # selected device.  This avoids a separate random-number stream on
        # the GPU and keeps smoke runs deterministic.
        if waveform_augmentor is not None:
            waveforms = waveform_augmentor(waveforms)
        features = frontend(waveforms.to(device, non_blocking=True))
        if feature_augmentor is not None:
            features = feature_augmentor(features)

        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        predictions = logits.detach().argmax(dim=1)
        total_correct += int((predictions == labels).sum().cpu())
        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(
            torch.softmax(logits.detach(), dim=1)[:, 1]
            .cpu()
            .numpy()
        )
        global_step += 1

        if log_every > 0 and global_step % log_every == 0:
            print(
                f"  step {global_step}/{total_steps}: "
                f"loss={total_loss / max(total_examples, 1):.4f}, "
                f"lr={lr:.6f}",
                flush=True,
            )
        if max_steps is not None and global_step >= max_steps:
            break

    labels_array = (
        np.concatenate(all_labels) if all_labels else np.empty(0, dtype=int)
    )
    scores_array = (
        np.concatenate(all_scores) if all_scores else np.empty(0, dtype=float)
    )
    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "auc": safe_auc(labels_array, scores_array),
        "examples": total_examples,
        "lr": learning_rate_at(
            max(global_step - 1, 0),
            total_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_ratio=warmup_ratio,
            hold_ratio=hold_ratio,
            power=power,
        ),
    }
    return metrics, global_step


@torch.inference_mode()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    frontend: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the model on fixed 0.63 s validation windows."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for waveforms, labels in loader:
        features = frontend(waveforms.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().cpu())
        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(
            torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        )

    labels_array = (
        np.concatenate(all_labels) if all_labels else np.empty(0, dtype=int)
    )
    scores_array = (
        np.concatenate(all_scores) if all_scores else np.empty(0, dtype=float)
    )
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "auc": safe_auc(labels_array, scores_array),
        "examples": total_examples,
    }


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
        description="Train MarbleNet-3x2x64 on LibriVAD-style data."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
        help="Training manifest TSV.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        required=True,
        help="Validation manifest TSV.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing generated/ and labels/ (for smoke data).",
    )
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--train-rows",
        type=int,
        default=None,
        help=(
            "Deterministically sample this many training manifest rows, "
            "stratified by noise/SNR. Useful for full medium/large manifests."
        ),
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=None,
        help=(
            "Deterministically sample this many validation manifest rows, "
            "stratified by noise/SNR."
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--segment-samples", type=int, default=10_080)
    parser.add_argument("--train-stride", type=int, default=2_400)
    parser.add_argument("--val-stride", type=int, default=2_400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--causal",
        action="store_true",
        help=(
            "Use left-only convolutions and a causal MFCC frontend. "
            "The checkpoint records this setting for evaluation."
        ),
    )
    parser.add_argument(
        "--rf-profile",
        choices=tuple(DILATION_PROFILES),
        default="baseline",
        help=(
            "Dilation profile used to change temporal receptive field "
            "without changing parameter count."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Warm-start model and optimizer state from a checkpoint. "
            "With --causal this converts an existing non-causal model into "
            "a causal model before continuing training."
        ),
    )

    parser.add_argument("--max-lr", type=float, default=0.01)
    parser.add_argument("--min-lr", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--hold-ratio", type=float, default=0.45)
    parser.add_argument("--decay-power", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)

    parser.add_argument("--waveform-aug-prob", type=float, default=0.8)
    parser.add_argument("--no-waveform-augment", action="store_true")
    parser.add_argument("--no-specaugment", action="store_true")
    parser.add_argument("--no-spec-cutout", action="store_true")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Do not balance speech and silence training windows.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.segment_samples <= 0 or args.train_stride <= 0 or args.val_stride <= 0:
        parser.error("segment and stride values must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.train_rows is not None and args.train_rows < 0:
        parser.error("--train-rows must be non-negative")
    if args.val_rows is not None and args.val_rows < 0:
        parser.error("--val-rows must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not 0.0 <= args.waveform_aug_prob <= 1.0:
        parser.error("--waveform-aug-prob must be in [0, 1]")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.max_lr <= 0 or args.min_lr <= 0 or args.min_lr > args.max_lr:
        parser.error("learning rates must satisfy 0 < min_lr <= max_lr")
    try:
        learning_rate_at(
            0,
            100,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
            hold_ratio=args.hold_ratio,
            power=args.decay_power,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = LibriVADSegments(
        args.train_manifest,
        generated_root=args.generated_root,
        label_root=args.label_root,
        data_root=args.data_root,
        segment_samples=args.segment_samples,
        stride_samples=args.train_stride,
        limit=args.limit,
        row_sample=args.train_rows,
        balance=not args.no_balance,
        seed=args.seed,
    )
    val_dataset = LibriVADSegments(
        args.val_manifest,
        generated_root=args.generated_root,
        label_root=args.label_root,
        data_root=args.data_root,
        segment_samples=args.segment_samples,
        stride_samples=args.val_stride,
        limit=args.limit,
        row_sample=args.val_rows,
        balance=False,
        seed=args.seed,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            "train or validation dataset has no 0.63 s windows; "
            "check the manifest and --segment-samples"
        )

    train_loader = build_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_data_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = {
        "feat_in": 64,
        "num_classes": 2,
        "repeat": 2,
        "channels": 64,
        "dropout": args.dropout,
        "segment_samples": args.segment_samples,
        "causal": bool(args.causal),
        "frontend": "mfcc_causal" if args.causal else "mfcc_centered",
        "dilation_profile": args.rf_profile,
        "frame_output": False,
    }
    model = build_marblenet_3x2x64(
        feat_in=model_config["feat_in"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
        causal=model_config["causal"],
        dilation_profile=model_config["dilation_profile"],
        frame_output=model_config["frame_output"],
    ).to(device)
    feature_config = MfccConfig(causal=args.causal)
    frontend = MfccFrontend(feature_config).to(device)
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
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected checkpoint format: {args.resume}")
        state_dict = payload.get("model", payload)
        model.load_state_dict(state_dict)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        history = list(payload.get("history", []))
        source_config = payload.get("model_config", {})
        source_causal = bool(source_config.get("causal", False))
        if source_causal != bool(args.causal):
            print(
                "resume: warm-starting "
                f"{'causal' if source_causal else 'non-causal'} weights into "
                f"a {'causal' if args.causal else 'non-causal'} model"
            )

    waveform_augmentor = None
    if not args.no_waveform_augment:
        waveform_augmentor = WaveformAugmentor(
            sample_rate=feature_config.sample_rate,
            prob=args.waveform_aug_prob,
            rng=random.Random(args.seed),
        )
    spec_augment = (
        None if args.no_specaugment else SpecAugment(rng=random.Random(args.seed + 1))
    )
    spec_cutout = (
        None if args.no_spec_cutout else SpecCutout(rng=random.Random(args.seed + 2))
    )
    feature_augmentor = (
        None
        if spec_augment is None and spec_cutout is None
        else FeatureAugmentor(spec_augment, spec_cutout)
    )

    steps_per_epoch = len(train_loader)
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else args.epochs * steps_per_epoch
    )
    print(
        f"device={device}, causal={args.causal}, "
        f"train_windows={len(train_dataset)} "
        f"({train_dataset.label_counts()[0]} speech / "
        f"{train_dataset.label_counts()[1]} silence), "
        f"val_windows={len(val_dataset)}, "
        f"parameters={count_parameters(model):,}, "
        f"rf={receptive_field(args.rf_profile)}, "
        f"total_steps={total_steps}, start_epoch={start_epoch}"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = -math.inf
    best_val_loss = math.inf
    for record in history:
        val = record.get("val", {})
        if val.get("auc") is not None:
            best_val_auc = max(best_val_auc, float(val["auc"]))
        if val.get("loss") is not None:
            best_val_loss = min(best_val_loss, float(val["loss"]))
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_metrics, global_step = run_train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            frontend,
            waveform_augmentor,
            feature_augmentor,
            device,
            start_step=global_step,
            total_steps=total_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_ratio=args.warmup_ratio,
            hold_ratio=args.hold_ratio,
            power=args.decay_power,
            log_every=args.log_every,
            max_steps=args.max_steps,
        )
        val_metrics = run_validation(
            model, val_loader, criterion, frontend, device
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
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']}",
            flush=True,
        )

        improved = False
        if val_metrics["auc"] is not None:
            improved = val_metrics["auc"] > best_val_auc
            best_val_auc = max(best_val_auc, val_metrics["auc"])
        elif val_metrics["loss"] < best_val_loss:
            improved = True
        best_val_loss = min(best_val_loss, val_metrics["loss"])

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

    with open(
        args.results_dir / "history.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "config": vars(args),
                "model_config": model_config,
                "history": history,
                "last_epoch": last_epoch,
                "global_step": global_step,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"artifacts written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
