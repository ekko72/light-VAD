# -*- coding: utf-8 -*-
"""Audited recovery for an interrupted A-v2 result-assembly run.

The primary M1/M2 checkpoints were already trained and saved before the
locked runner failed while assembling domain-level bootstrap rows. This
driver reuses those checkpoints and does not retrain the primary models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from reproductions.difficulty_adaptive_context import run_a2_method_study as a2


_ORIGINAL_EVALUATE_PRIMARY = a2.evaluate_primary


def _saved_primary_models(
    data: a2.FrameData,
    x3: np.ndarray,
    *,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], np.ndarray]]:
    """Load the already-trained primary checkpoints and recompute scores."""

    model_dir = output_dir / "a2_models"
    manifest: list[dict[str, Any]] = []
    scores: dict[tuple[str, int], np.ndarray] = {}
    for method in ("M1", "M2"):
        for seed in a2.TRAINING_SEEDS:
            model_path = model_dir / f"{method.lower()}_seed{seed}.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"recovery requires saved primary model: {model_path}"
                )
            payload = torch.load(
                model_path, map_location=device, weights_only=False
            )
            if payload.get("protocol_id") != a2.PROTOCOL_ID:
                raise ValueError(f"protocol id mismatch in {model_path}")
            if payload.get("method") != method or int(payload.get("seed")) != int(seed):
                raise ValueError(f"metadata mismatch in {model_path}")
            if int(payload["input_dim"]) != int(x3.shape[1]):
                raise ValueError(f"input dimension mismatch in {model_path}")
            model = a2.build_model(method, int(payload["input_dim"])).to(device)
            model.load_state_dict(payload["state_dict"])
            scores[(method, seed)] = a2.predict_method(
                model, method, x3, device=device
            )
            history = list(payload["history"])
            manifest.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "model_path": a2.relpath(model_path),
                    "model_sha256": a2.sha256_file(model_path),
                    "final_train_loss": float(history[-1]["loss"]),
                    "train_frames": int(data.train_mask.sum()),
                    "history": history,
                    "protocol_id": a2.PROTOCOL_ID,
                }
            )
            print(f"[A2 recovery] loaded {method} seed {seed}", flush=True)
    return manifest, scores


def _evaluate_primary_complete_domains(
    data: a2.FrameData,
    scores: Mapping[tuple[str, int], np.ndarray],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen evaluator with a complete domain-index lookup.

    `_domain_masks` includes `all` and `clean`, while the original
    `DOMAIN_ORDER` only included `seen` and `unseen`. Keeping `seen` and
    `unseen` in their original positions preserves their bootstrap seeds.
    Only `seen`/`unseen` cells are retained, matching the frozen output.
    """

    original_order = a2.DOMAIN_ORDER
    a2.DOMAIN_ORDER = ("seen", "unseen", "all", "clean")
    try:
        evaluation = _ORIGINAL_EVALUATE_PRIMARY(
            data,
            scores,
            output_dir=output_dir,
        )
    finally:
        a2.DOMAIN_ORDER = original_order
    evaluation["cell_rows"] = [
        row
        for row in evaluation["cell_rows"]
        if row["domain"] in original_order
    ]
    return evaluation


def _load_protocol(output_dir: Path) -> a2.FrozenProtocol:
    protocol_path = output_dir / "a2_method_protocol_freeze.json"
    hash_path = output_dir / "a2_protocol_sha256.txt"
    return a2.load_frozen_protocol(protocol_path, hash_path=hash_path)


def _load_frozen_data(
    protocol: a2.FrozenProtocol,
) -> tuple[a2.FrameData, torch.device]:
    a2.verify_protocol_locks(protocol.payload)
    locks = protocol.payload["locks"]
    data = a2.load_frame_data(
        frame_bundle_path=a2.resolve_path(locks["frame_bundle"]["path"]),
        feature_path=a2.resolve_path(locks["feature_cache"]["path"]),
    )
    device = a2.resolve_device("cuda")
    return data, device


def preflight(output_dir: Path) -> dict[str, Any]:
    """Validate recovery evaluation without writing result artifacts."""

    protocol = _load_protocol(output_dir)
    data, device = _load_frozen_data(protocol)
    x3, _ = a2.build_feature_arrays(data)
    _, scores = _saved_primary_models(
        data,
        x3,
        device=device,
        output_dir=output_dir,
    )
    evaluation = _evaluate_primary_complete_domains(
        data,
        scores,
        output_dir=output_dir,
    )
    compute_rows = a2.compute_estimates(
        data,
        scores,
        output_dir=output_dir,
        device=device,
    )
    assessment = a2.assess_primary(
        budget_rows=evaluation["budget_rows"],
        cluster_rows=evaluation["cluster_rows"],
        cell_rows=evaluation["cell_rows"],
        compute_rows=compute_rows,
    )
    selected = assessment["candidates"][assessment["primary_method"]]
    return {
        "status": assessment["status"],
        "primary_method": assessment["primary_method"],
        "primary_budget": assessment["primary_budget"],
        "delta_utility": selected["point_delta_utility"],
        "ci95_low": selected["ci95_low"],
        "ci95_high": selected["ci95_high"],
        "seen_delta": selected["seen_delta"],
        "unseen_delta": selected["unseen_delta"],
        "max_activation": selected["max_activation"],
        "cell_rows": len(evaluation["cell_rows"]),
    }


def complete(output_dir: Path, progress_every: int) -> dict[str, Any]:
    """Resume result assembly and independent sensitivity from saved models."""

    protocol = _load_protocol(output_dir)
    original_train = a2.train_primary_models
    original_evaluate = a2.evaluate_primary
    a2.train_primary_models = _saved_primary_models
    a2.evaluate_primary = _evaluate_primary_complete_domains
    try:
        result = a2.run_protocol(
            protocol=protocol,
            output_dir=output_dir,
            device_name="cuda",
            progress_every=progress_every,
            skip_independent=False,
        )
    finally:
        a2.train_primary_models = original_train
        a2.evaluate_primary = original_evaluate

    a2.claim_once(
        output_dir / a2.INTERNAL_RUN_COMPLETE_MARKER_NAME,
        command="internal-test-complete",
        protocol_sha256=protocol.sha256,
    )

    deviation = {
        "kind": "infrastructure_recovery",
        "stage": "after all primary checkpoints were trained, before result files",
        "failure": (
            "ValueError in domain bootstrap seed lookup because `all` and "
            "`clean` are returned by `_domain_masks` but omitted from "
            "`DOMAIN_ORDER`"
        ),
        "action": (
            "Reused all ten saved primary checkpoints, preserved the original "
            "seen/unseen bootstrap seed positions, reran deterministic "
            "evaluation and independent sensitivity, and made no model, "
            "threshold, feature, loss, budget, or seed selection."
        ),
        "primary_models_retrained": False,
        "new_final_ood_touched": False,
        "next_search_authorized": False,
    }
    result["PROTOCOL_DEVIATIONS"] = [deviation]
    a2.write_json(output_dir / "a2_final_summary.json", result)

    audit = {
        "recovery_script": a2.relpath(Path(__file__)),
        "recovery_script_sha256": a2.sha256_file(Path(__file__)),
        "protocol_sha256": protocol.sha256,
        "internal_run_started": json.loads(
            (output_dir / a2.INTERNAL_RUN_STARTED_MARKER_NAME).read_text(
                encoding="utf-8"
            )
        ),
        "deviation": deviation,
    }
    a2.write_json(output_dir / "a2_recovery_audit.json", audit)
    report_path = output_dir / "a2_final_report.md"
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Recovery Audit\n\n"
            "- The primary checkpoints were trained once and reused after a "
            "result-assembly domain-index error.\n"
            "- No primary model was retrained; no model, threshold, feature, "
            "loss, budget, or seed search was performed.\n"
            "- See `a2_recovery_audit.json` for the exact deviation record.\n"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "complete"))
    parser.add_argument("--output-dir", type=Path, default=a2.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "preflight":
        payload = preflight(output_dir)
    else:
        complete_marker = output_dir / a2.INTERNAL_RUN_COMPLETE_MARKER_NAME
        if complete_marker.exists():
            raise ValueError("A-v2 internal-test recovery is already complete")
        payload = complete(output_dir, args.progress_every)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
