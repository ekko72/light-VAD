from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .legacy_models import BATCH_SIZE, FEATURES, FRAMES, load_legacy_checkpoint, num_parameters


CHECKPOINTS = (
    "net_epoch014.net",
    "net_large_epoch014.net",
    "gru_epoch014.net",
    "gru_large_epoch014.net",
    "densenet_epoch012.net",
    "densenet_large_epoch014.net",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            r"C:\Users\20547\Desktop\论文p12\voice-activity-detection-upstream\data\models"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != BATCH_SIZE:
        raise ValueError(
            f"The original models hard-code BATCH_SIZE={BATCH_SIZE}; "
            "this smoke check intentionally uses the same value."
        )

    sample = torch.randn(args.batch_size, FRAMES, FEATURES)
    print(f"Input: {tuple(sample.shape)}")

    for filename in CHECKPOINTS:
        path = args.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        model = load_legacy_checkpoint(path)
        with torch.inference_mode():
            output = model(sample)

        print(
            f"{filename:30s} params={num_parameters(model):6d} "
            f"output={tuple(output.shape)} sum={output.sum(dim=1).mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
