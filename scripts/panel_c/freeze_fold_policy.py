#!/usr/bin/env python3
"""Hash-freeze one outer fold's complete dual-gate policy before target inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    require_frozen_execution_state,
    sha256_file,
    write_json_new,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--semantic-features", type=Path, required=True)
    parser.add_argument("--candidate-profiles", type=Path, required=True)
    parser.add_argument("--h1-router", type=Path)
    parser.add_argument("--h1-5-router", type=Path)
    parser.add_argument("--h2-router", type=Path, required=True)
    parser.add_argument("--simple-controls", type=Path, required=True)
    parser.add_argument("--selected-h2-gate3", type=Path)
    parser.add_argument("--selected-h2-gate4", type=Path)
    parser.add_argument("--selected-B-A-star", type=Path)
    parser.add_argument("--selected-B-U-star", type=Path)
    # v7.0.1 aliases remain accepted only for its historical contract.
    parser.add_argument("--selected-margin", type=Path)
    parser.add_argument("--selected-B-f-star", type=Path)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    contract = load_json(args.analysis_contract)
    v702 = str(contract.get("protocol_version")) == "1.0.2"
    if v702 and any(
        value is None
        for value in (
            args.h1_router,
            args.h1_5_router,
            args.selected_h2_gate3,
            args.selected_h2_gate4,
            args.selected_B_A_star,
            args.selected_B_U_star,
        )
    ):
        raise PermissionError("v7.0.2 requires complete H1/H1.5 and dual Gate-3/Gate-4 policies")
    split = load_json(args.split_manifest)
    fold = next((row for row in split["folds"] if row["fold_id"] == args.fold_id), None)
    if fold is None or str(fold["held_out_source"]) != args.held_out_source:
        raise ValueError("fold/held-out-source mismatch")
    gate3_path = args.selected_h2_gate3 or args.selected_margin
    gate4_path = args.selected_h2_gate4 or args.selected_margin
    baseline_a_path = args.selected_B_A_star or args.selected_B_f_star
    baseline_u_path = args.selected_B_U_star or args.selected_B_f_star
    if any(path is None for path in (gate3_path, gate4_path, baseline_a_path, baseline_u_path)):
        raise ValueError("fold-local selector artifacts are incomplete")
    gate3 = load_json(gate3_path)
    gate4 = load_json(gate4_path)
    baseline_a = load_json(baseline_a_path)
    baseline_u = load_json(baseline_u_path)
    selections = (gate3, gate4, baseline_a, baseline_u)
    if any(row.get("fold_id") != args.fold_id for row in selections):
        raise ValueError("fold-local selections belong to another fold")
    if any(row.get("held_out_source_used") is not False for row in selections):
        raise PermissionError("held-out source reached a policy selector")
    if v702:
        expected_objectives = (
            (gate3, "accuracy"),
            (gate4, "utility"),
            (baseline_a, "accuracy"),
            (baseline_u, "utility"),
        )
        if any(row.get("selection_objective") != objective for row, objective in expected_objectives):
            raise PermissionError("Gate-3/Gate-4 selector objectives are not symmetric")
    if contract["outer_fold_selection"]["global_cross_fold_selection_for_primary"] is not False:
        raise PermissionError("analysis contract permits global primary comparator selection")
    bindings = {
        "analysis_contract": args.analysis_contract,
        "split_manifest": args.split_manifest,
        "code_manifest": args.code_manifest,
        "semantic_features": args.semantic_features,
        "candidate_profiles": args.candidate_profiles,
        "h2_router": args.h2_router,
        "simple_controls": args.simple_controls,
    }
    if v702:
        bindings.update(
            {
                "h1_router": args.h1_router,
                "h1_5_router": args.h1_5_router,
                "selected_h2_gate3_accuracy_policy": gate3_path,
                "selected_h2_gate4_utility_policy": gate4_path,
                "selected_B_A_star": baseline_a_path,
                "selected_B_U_star": baseline_u_path,
            }
        )
    else:
        bindings.update(
            {
                "selected_margin": gate3_path,
                "selected_B_f_star": baseline_a_path,
            }
        )
    payload = {
        "schema_version": 2 if v702 else 1,
        "artifact_role": "panel_c_fold_policy_freeze",
        "status": "FOLD_POLICY_FROZEN_PRE_TARGET_OUTCOME",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fold_id": args.fold_id,
        "held_out_source": args.held_out_source,
        "H1_policy": "frozen_router" if args.h1_router else "historical_v7_0_1_not_separately_bound",
        "H1_5_policy": "frozen_router" if args.h1_5_router else "historical_v7_0_1_not_separately_bound",
        "H2_gate3_accuracy_policy": {
            "selected_margin": gate3["selected_margin"],
            "selection_objective": gate3.get("selection_objective", "legacy"),
        },
        "H2_gate4_utility_policy": {
            "selected_margin": gate4["selected_margin"],
            "selection_objective": gate4.get("selection_objective", "legacy"),
        },
        "B_A_star_method": baseline_a["selected_B_f_star"],
        "B_U_star_method": baseline_u["selected_B_f_star"],
        "selected_margin": gate3["selected_margin"],
        "selected_B_f_star": baseline_a["selected_B_f_star"],
        "target_outcomes_accessed": False,
        "global_policy_pool_used": False,
        "bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in bindings.items()
        },
    }
    write_json_new(args.output, payload)
    print(json.dumps({"status": payload["status"], "fold": args.fold_id}, sort_keys=True))


if __name__ == "__main__":
    main()
