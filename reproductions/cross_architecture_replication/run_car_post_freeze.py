# -*- coding: utf-8 -*-
"""Run post-freeze CAR evaluation and analysis with integration fixes.

The frozen runner was hashed before training. Its ``load_rf384_reference``
constructor omitted the already-validated ``full_adaptive_scores`` and
``long_scores`` arrays from the reference object. This wrapper preserves
the frozen runner and protocol, injects those two fields from the hashed
RF384 bundle, records the amendment, and delegates evaluation and
analysis to the frozen functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import reproductions.cross_architecture_replication.run_car as car


def _write_amendment(results_dir: Path) -> Path:
    path = results_dir / "car_implementation_amendments.json"
    record = car.load_car_protocol(verify_hashes=True)["source_hashes"][
        "car_runner"
    ]
    car._write_json(
        path,
        {
            "protocol_id": car.PROTOCOL_ID,
            "protocol_sha256": car.file_sha256(
                car.DEFAULT_RESULTS_DIR / "car_protocol_freeze.json"
            ),
            "created_at": car._utc_now(),
            "frozen_runner_sha256": str(record["sha256"]),
            "post_freeze_wrapper_sha256": car.file_sha256(Path(__file__)),
            "amendments": [
                {
                    "target": "CarReference construction",
                    "issue": (
                        "The frozen runner omitted full_adaptive_scores and "
                        "long_scores when constructing CarReference."
                    ),
                    "resolution": (
                        "Load the two arrays from the already hash-verified "
                        "RF384 prediction bundle and pass them to the frozen "
                        "CarReference class."
                    ),
                    "changes_training": False,
                    "changes_evaluation_semantics": False,
                    "changes_analysis_semantics": False,
                },
                {
                    "target": "score_horizon target slices",
                    "issue": (
                        "The frozen _score_window_batch used a truth-value "
                        "test on a NumPy target array."
                    ),
                    "resolution": (
                        "Convert target slices to lists before delegating to "
                        "the unchanged scoring implementation."
                    ),
                    "changes_training": False,
                    "changes_evaluation_semantics": False,
                    "changes_analysis_semantics": False,
                },
                {
                    "target": "CAR analysis overwrite guard",
                    "issue": (
                        "The frozen run_analyze_stage included the completed "
                        "car_gru_training_runs.csv in its analysis overwrite "
                        "guard."
                    ),
                    "resolution": (
                        "Exclude only car_gru_training_runs.csv from the "
                        "analysis-stage overwrite check; all analysis outputs "
                        "remain protected from overwrite."
                    ),
                    "changes_training": False,
                    "changes_evaluation_semantics": False,
                    "changes_analysis_semantics": False,
                }
            ],
            "NEXT_METHOD_SEARCH_AUTHORIZED": False,
            "NEW_FINAL_OOD_TOUCHED": False,
        },
    )
    return path


def _install_reference_compatibility() -> None:
    original_reference = car.CarReference

    class CompatibleCarReference(original_reference):
        def __init__(self, **kwargs: Any) -> None:
            if (
                "full_adaptive_scores" not in kwargs
                or "long_scores" not in kwargs
            ):
                protocol = kwargs["protocol"]
                record = protocol["data_hashes"][
                    "rf384_prediction_bundle"
                ]
                arrays = car._load_npz_arrays(
                    car.REPO_ROOT / str(record["path"])
                )
                kwargs["full_adaptive_scores"] = np.asarray(
                    arrays["full_adaptive_scores"],
                    dtype=np.float64,
                )
                kwargs["long_scores"] = np.asarray(
                    arrays["long_scores"],
                    dtype=np.float64,
                )
            super().__init__(**kwargs)

    car.CarReference = CompatibleCarReference


def _install_score_compatibility() -> None:
    original_score = car._score_window_batch

    def compatible_score(
        model: Any,
        features: Any,
        targets: Any,
        *,
        horizon: int,
    ) -> np.ndarray:
        target_list = list(targets)
        if not target_list:
            return np.empty(0, dtype=np.float64)
        return original_score(
            model,
            features,
            target_list,
            horizon=horizon,
        )

    car._score_window_batch = compatible_score


def _install_analysis_overwrite_compatibility() -> None:
    original_refuse = car._refuse_overwrite

    def compatible_refuse(paths: Any, *, stage: str) -> None:
        filtered = [
            path
            for path in paths
            if Path(path).name != "car_gru_training_runs.csv"
        ]
        original_refuse(filtered, stage=stage)

    car._refuse_overwrite = compatible_refuse


def _install_manifest_amendment(results_dir: Path) -> None:
    original_collect = car._collect_car_artifacts
    amendment_path = results_dir / "car_implementation_amendments.json"

    def collect_with_amendment(output_dir: Path) -> list[Path]:
        paths = list(original_collect(output_dir))
        if amendment_path.is_file():
            paths.append(amendment_path)
        return paths

    car._collect_car_artifacts = collect_with_amendment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CAR evaluation/analysis with a documented post-freeze "
            "integration fix."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("evaluate", "analyze"),
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=car.DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir).resolve()
    if results_dir != car.DEFAULT_RESULTS_DIR.resolve():
        raise ValueError(
            "CAR paths are frozen to "
            f"{car.DEFAULT_RESULTS_DIR}; got {results_dir}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    protocol = car.load_car_protocol(verify_hashes=True)
    amendment_path = _write_amendment(results_dir)
    _install_reference_compatibility()
    _install_score_compatibility()
    _install_manifest_amendment(results_dir)

    if args.stage == "evaluate":
        car.run_evaluate_stage(
            protocol=protocol,
            results_dir=results_dir,
            device=car.resolve_device(args.device),
        )
    else:
        _install_analysis_overwrite_compatibility()
        car.run_analyze_stage(
            protocol=protocol,
            results_dir=results_dir,
        )
    print(
        f"CAR_POST_FREEZE_STAGE={args.stage} "
        f"AMENDMENT={amendment_path.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
