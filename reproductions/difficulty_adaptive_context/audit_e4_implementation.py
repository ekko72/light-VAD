from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = (
    ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e4_refinability_stability_decomposition"
)
AUDIT_ROOT = (
    ROOT
    / "results"
    / "difficulty_adaptive_context"
    / "e4_implementation_audit"
)

EXPECTED_PROTOCOL_SHA256 = (
    "066516F8E208C3F4F6F4A74BE2EA92AE5C91993B995C2022C61CC8DBE783626E"
)
EXPECTED_RUNNER_SHA256 = (
    "33B2EF3D0E4745432F832E2FE69BD4762AA21030B96B73FC01B73D29CF422212"
)
CLAIM_C_STABLE_SHARE_MAX = 0.50
CLAIM_C_CORE_SHARE_MAX = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, value: dict[str, Any]) -> None:
    lines = [
        "# E4 Post-hoc Implementation Audit",
        "",
        "This audit does not replace or modify the frozen E4 protocol, runner, "
        "or official result.",
        "",
        f"- Audit type: `{value['AUDIT_TYPE']}`",
        f"- Official E4 status unchanged: "
        f"`{str(value['OFFICIAL_E4_STATUS_UNCHANGED']).lower()}`",
        f"- Authorized repair performed: "
        f"`{str(value['AUTHORIZED_REPAIR_PERFORMED']).lower()}`",
        f"- Scientific status established by this audit: "
        f"`{str(value['SCIENTIFIC_STATUS_ESTABLISHED_BY_AUDIT']).lower()}`",
        "",
        "## Defect",
        "",
        value["DEFECT_SUMMARY"],
        "",
        "## Derived Intended Gates",
        "",
        f"- Population reproducible: "
        f"`{str(value['DERIVED_POPULATION_REPRODUCIBLE']).lower()}`",
        f"- G1 stable core: "
        f"`{str(value['DERIVED_G1_STABLE_CORE']).lower()}`",
        f"- G2 stable core value: "
        f"`{str(value['DERIVED_G2_STABLE_CORE_VALUE']).lower()}`",
        f"- G3 structured instability: "
        f"`{str(value['DERIVED_G3_STRUCTURED_INSTABILITY']).lower()}`",
        f"- Nontrivial stable/core population: "
        f"`{str(value['DERIVED_NONTRIVIAL_STABLE']).lower()}`",
        "",
        "## Derived Intended Result",
        "",
        f"- Derived status under frozen rules: "
        f"`{value['DERIVED_STATUS_UNDER_FROZEN_RULES']}`",
        f"- Claim A: `{value['DERIVED_CLAIM_A']}`",
        f"- Claim B: `{value['DERIVED_CLAIM_B']}`",
        f"- Claim C: `{value['DERIVED_CLAIM_C']}`",
        f"- Claim D: `{value['DERIVED_CLAIM_D']}`",
        f"- Claim E: `{value['DERIVED_CLAIM_E']}`",
        "",
        "## Boundary",
        "",
        "The derived result is post-hoc and must not be reported as the official "
        "frozen E4 result without an explicitly authorized versioned repair and "
        "rerun.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit() -> dict[str, Any]:
    summary_path = OFFICIAL_ROOT / "e4_final_summary.json"
    baseline_path = OFFICIAL_ROOT / "e4_baseline_reproduction.json"
    protocol_path = OFFICIAL_ROOT / "e4_protocol_freeze.json"
    runner_path = (
        ROOT
        / "reproductions"
        / "difficulty_adaptive_context"
        / "run_e4.py"
    )

    summary = _read_json(summary_path)
    baseline = _read_json(baseline_path)
    protocol_sha256 = _sha256(protocol_path)
    runner_sha256 = _sha256(runner_path)
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen E4 protocol hash does not match")
    if runner_sha256 != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("frozen E4 runner hash does not match")

    seed_rows = [
        row
        for row in baseline.get("seed_taxonomy", [])
        if str(row.get("scope")) == "seed"
    ]
    aggregate = baseline.get("aggregate_taxonomy", {})
    population_reproducible = bool(
        baseline.get("BASELINE_REPRODUCED", False)
        and seed_rows
        and all(
            float(row.get("NetRefinability", float("nan"))) > 0.0
            for row in seed_rows
        )
        and float(aggregate.get("mean_v_log", float("nan"))) > 0.0
    )

    g1 = bool(summary.get("G1_STABLE_CORE", False))
    g2 = bool(summary.get("G2_STABLE_CORE_VALUE", False))
    g3 = bool(summary.get("G3_STRUCTURED_INSTABILITY", False))
    details = summary.get("E4_DETAILS", {})
    nontrivial_stable = bool(details.get("NONTRIVIAL_STABLE", False))
    p_stable_given_ever = float(
        summary.get("P_STABLE_GIVEN_EVER_R", float("nan"))
    )
    p_core_given_ever = float(
        summary.get("P_CORE_GIVEN_EVER_R", float("nan"))
    )

    if not population_reproducible:
        derived_status = "INCONCLUSIVE_OR_INVALID"
    elif g1 and g2 and nontrivial_stable:
        derived_status = "STABLE_REFINABLE_CORE"
    elif g1:
        derived_status = "POPULATION_STABLE_INSTANCE_RELATIVE"
    else:
        derived_status = "MOSTLY_MODEL_RELATIVE_REFINABILITY"

    claim_a = "SUPPORTED" if population_reproducible else "NOT_SUPPORTED"
    claim_b = (
        "SUPPORTED"
        if derived_status == "STABLE_REFINABLE_CORE"
        else "CONDITIONAL"
        if g1
        else "NOT_SUPPORTED"
    )
    claim_c = (
        "SUPPORTED"
        if population_reproducible
        and (
            not g1
            or p_stable_given_ever <= CLAIM_C_STABLE_SHARE_MAX
            or p_core_given_ever <= CLAIM_C_CORE_SHARE_MAX
        )
        else "CONDITIONAL"
        if g1
        else "NOT_SUPPORTED"
    )

    g3_details = details.get("G3", {})
    short_margin_pass = bool(
        g3_details.get("stable_minus_unstable_short_margin", False)
    )
    non_short_candidates = [
        bool(value)
        for key, value in g3_details.items()
        if key != "stable_minus_unstable_short_margin"
    ]
    if short_margin_pass and not any(non_short_candidates):
        claim_d = "SUPPORTED"
    elif not short_margin_pass:
        claim_d = "NOT_SUPPORTED"
    else:
        claim_d = "UNRESOLVED"

    claim_e = (
        "SUPPORTED_AS_HYPOTHESIS"
        if population_reproducible
        and (
            p_stable_given_ever <= CLAIM_C_STABLE_SHARE_MAX
            or p_core_given_ever <= CLAIM_C_CORE_SHARE_MAX
        )
        else "UNRESOLVED"
    )

    return {
        "AUDIT_TYPE": "POSTHOC_IMPLEMENTATION_AUDIT",
        "OFFICIAL_E4_STATUS_UNCHANGED": True,
        "AUTHORIZED_REPAIR_PERFORMED": False,
        "SCIENTIFIC_STATUS_ESTABLISHED_BY_AUDIT": False,
        "OFFICIAL_E4_STATUS": str(summary.get("E4_STATUS", "")),
        "PROTOCOL_SHA256": protocol_sha256,
        "RUNNER_SHA256": runner_sha256,
        "OFFICIAL_SUMMARY_SHA256": _sha256(summary_path),
        "OFFICIAL_BASELINE_SHA256": _sha256(baseline_path),
        "DEFECT_SUMMARY": (
            "The frozen runner passes baseline['seed_taxonomy'] into the gate "
            "builder. That field excludes the aggregate taxonomy row, although "
            "the gate builder requires it to compute population_reproducible."
        ),
        "DERIVED_POPULATION_REPRODUCIBLE": population_reproducible,
        "DERIVED_G1_STABLE_CORE": g1,
        "DERIVED_G2_STABLE_CORE_VALUE": g2,
        "DERIVED_G3_STRUCTURED_INSTABILITY": g3,
        "DERIVED_NONTRIVIAL_STABLE": nontrivial_stable,
        "DERIVED_STATUS_UNDER_FROZEN_RULES": derived_status,
        "DERIVED_CLAIM_A": claim_a,
        "DERIVED_CLAIM_B": claim_b,
        "DERIVED_CLAIM_C": claim_c,
        "DERIVED_CLAIM_D": claim_d,
        "DERIVED_CLAIM_E": claim_e,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="write audit artifacts under the separate audit directory",
    )
    args = parser.parse_args()
    audit = build_audit()
    if args.write:
        _write_json(AUDIT_ROOT / "e4_posthoc_implementation_audit.json", audit)
        _write_markdown(AUDIT_ROOT / "e4_posthoc_implementation_audit.md", audit)
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
