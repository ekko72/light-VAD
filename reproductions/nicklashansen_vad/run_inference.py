from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from .features import PreparedAudio, center_windows, prepare_audio
from .legacy_models import load_legacy_checkpoint, num_parameters


DEFAULT_MODEL_DIR = Path(
    r"C:\Users\20547\Desktop\论文p12\voice-activity-detection-upstream\data\models"
)
DEFAULT_MANIFEST = Path("data/splits/test_speech.tsv")
DEFAULT_MODELS = (
    "net_large_epoch014.net",
    "gru_large_epoch014.net",
    "densenet_large_epoch014.net",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original nicklashansen VAD checkpoints on LibriSpeech."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Checkpoint filename; repeat to compare several. Defaults to 3 large models.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as file:
        header = file.readline().rstrip("\n").split("\t")
        rows = []
        for line in file:
            values = line.rstrip("\n").split("\t")
            if values:
                rows.append(dict(zip(header, values)))
    return rows


def select_rows(rows: list[dict[str, str]], limit: int, seed: int):
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def load_prepared(rows: list[dict[str, str]], data_root: Path):
    prepared = []
    for row in rows:
        path = data_root / row["path"]
        prepared.append(prepare_audio(path))
    return prepared


def predict(
    model: torch.nn.Module,
    samples: list[PreparedAudio],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.concatenate([center_windows(sample)[0] for sample in samples])
    labels = np.concatenate([center_windows(sample)[1] for sample in samples])
    probabilities = []

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            if len(chunk) < batch_size:
                padding = np.zeros(
                    (batch_size - len(chunk), chunk.shape[1], chunk.shape[2]),
                    dtype=np.float32,
                )
                chunk = np.concatenate((chunk, padding))

            output = model(torch.from_numpy(chunk).to(device))
            probabilities.append(output[: min(batch_size, len(windows) - start), 1])

    return torch.cat(probabilities).cpu().numpy(), labels


def far_at_frr(labels: np.ndarray, probabilities: np.ndarray, target_frr=0.01):
    thresholds = np.unique(probabilities)
    candidates = []
    for threshold in thresholds:
        prediction = probabilities >= threshold
        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(
                labels,
                prediction.astype(np.int8),
                labels=[0, 1],
            ).ravel()
        )
        frr = false_negative / max(false_negative + true_positive, 1)
        far = false_positive / max(false_positive + true_negative, 1)
        if frr <= target_frr:
            candidates.append((frr, far, float(threshold)))

    if not candidates:
        return {
            "target_frr": target_frr,
            "achieved_frr": None,
            "far": None,
            "threshold": None,
        }
    achieved_frr, far, threshold = min(candidates)
    return {
        "target_frr": target_frr,
        "achieved_frr": achieved_frr,
        "far": far,
        "threshold": threshold,
    }


def evaluate(
    checkpoint: Path,
    samples: list[PreparedAudio],
    batch_size: int,
    device: torch.device,
) -> dict:
    model = load_legacy_checkpoint(checkpoint, map_location="cpu")
    probabilities, labels = predict(model, samples, batch_size, device)
    prediction = (probabilities >= 0.5).astype(np.int8)
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    )

    metrics = {
        "checkpoint": checkpoint.name,
        "parameters": num_parameters(model),
        "frames": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "auc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None,
        "speech_recall": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "nonspeech_recall": float(
            true_negative / max(true_negative + false_positive, 1)
        ),
        "confusion": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "operating_point": far_at_frr(labels, probabilities),
    }
    return metrics


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    data_root = Path(__file__).resolve().parents[2] / "data"
    rows = select_rows(read_manifest(args.manifest), args.limit, args.seed)
    samples = load_prepared(rows, data_root)
    speech_frames = sum(int(sample.labels.sum()) for sample in samples)
    all_frames = sum(len(sample.labels) for sample in samples)

    print(
        f"Samples={len(samples)} frames={all_frames} "
        f"WebRTC speech={speech_frames / max(all_frames, 1):.3%} "
        f"device={device}"
    )

    model_names = args.models or list(DEFAULT_MODELS)
    results = []
    for model_name in model_names:
        checkpoint = args.model_dir / model_name
        result = evaluate(checkpoint, samples, args.batch_size, device)
        results.append(result)
        print(
            f"{model_name:30s} params={result['parameters']:6d} "
            f"acc={result['accuracy']:.4f} auc={result['auc']:.5f} "
            f"speech_recall={result['speech_recall']:.4f} "
            f"nonspeech_recall={result['nonspeech_recall']:.4f}"
        )
        operating = result["operating_point"]
        print(
            "  FRR={achieved_frr:.3%} FAR={far:.3%} threshold={threshold:.6f}".format(
                **operating
            )
        )

    payload = {
        "manifest": str(args.manifest),
        "sample_count": len(samples),
        "frame_count": all_frames,
        "webrtc_speech_fraction": speech_frames / max(all_frames, 1),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
