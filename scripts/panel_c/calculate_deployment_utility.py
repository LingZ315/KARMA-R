#!/usr/bin/env python3
"""Calculate the frozen cost-complete deployment utility horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    require_analysis_input_freeze,
    require_frozen_execution_state,
    write_json_new,
)
from karmavl.panel_c.statistics import utility_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-target", type=Path, required=True)
    parser.add_argument("--analysis-input-freeze", type=Path, required=True)
    parser.add_argument("--onboarding-costs", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    args = parser.parse_args()
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    require_analysis_input_freeze(args.analysis_input_freeze, paired_rows=args.paired_target)
    contract = load_json(args.analysis_contract)
    onboarding = load_json(args.onboarding_costs)
    rows = onboarding.get("folds", [])
    if len(rows) != int(contract["cohort"]["outer_folds"]):
        raise ValueError("onboarding ledger must cover every outer fold")
    v702 = str(contract.get("protocol_version")) == "1.0.2"
    h2_field = "H2_U_gpu_seconds" if v702 else "H2_gpu_seconds"
    baseline_field = "B_U_star_gpu_seconds" if v702 else "B_f_star_gpu_seconds"
    if any(h2_field not in row or baseline_field not in row for row in rows):
        raise ValueError("onboarding ledger does not match the frozen Gate-4 policy fields")
    h2 = sum(float(row[h2_field]) for row in rows)
    baseline = sum(float(row[baseline_field]) for row in rows)
    if min(h2, baseline) < 0:
        raise ValueError("onboarding costs cannot be negative")
    result = utility_summary(
        load_jsonl(args.paired_target),
        onboarding_h2=h2,
        onboarding_b_f_star=baseline,
        horizons=contract["cost"]["horizons"],
        cost_coefficient=float(contract["cost"]["lambda"]),
    )
    result["onboarding_gpu_seconds"] = {"H2": h2, "B_f_star": baseline}
    result["gate"] = "Gate 4: utility-selected H2_U versus utility-selected B_U_star"
    result["formulation"] = contract["cost"]["utility"]
    write_json_new(args.output, result)
    print(json.dumps({"horizons": len(result["horizons"])}, sort_keys=True))


if __name__ == "__main__":
    main()
