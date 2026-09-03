#!/usr/bin/env python3
"""Fit the six frozen simple controls within one outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    add_frozen_execution_arguments,
    load_json,
    load_jsonl,
    require_frozen_execution_from_args,
    write_json_new,
)
from karmavl.panel_c.controls import fit_simple_controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--profile-scores", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--model-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    contract = load_json(args.analysis_contract)
    pool = load_json(args.model_pool)
    semantics = {
        str(row["query_id"]): {
            "primary_class": row["primary_class"],
            "subtype": row.get("subtype"),
            "ambiguity": bool(row.get("ambiguity")),
        }
        for row in load_jsonl(args.semantics)
    }
    profile_rows = load_jsonl(args.profile_scores)
    calibration_rows = load_jsonl(args.calibration_scores)
    for role, rows in (("profile", profile_rows), ("calibration", calibration_rows)):
        if any(
            str(row.get("fold_id")) != args.fold_id
            or row.get("role") != role
            or str(row.get("source")) == args.held_out_source
            for row in rows
        ):
            raise PermissionError(f"{role} rows violate the outer-fold source boundary")
    payload = fit_simple_controls(
        profile_rows=profile_rows,
        calibration_rows=calibration_rows,
        semantics=semantics,
        profiles=load_json(args.profiles),
        incumbent_route=str(pool["incumbent"]["route_id"]),
        candidate_routes=[str(row["route_id"]) for row in pool["candidates"]],
        class_order=list(contract["semantic_schema"]["class_order"]),
        subtype_order=list(contract["semantic_schema"]["subtype_order"]),
        minimum_support=int(contract["feature_estimation"]["minimum_class_or_subtype_support"]),
        cost_coefficient=float(contract["policy"]["cost_coefficient_per_gpu_second"]),
        logistic_l2=float(contract["hyperparameters"]["fixed"]["logistic_raw_l2"]),
        logistic_iterations=int(
            contract["hyperparameters"]["fixed"]["logistic_raw_maximum_iterations"]
        ),
    )
    payload["fold_id"] = args.fold_id
    payload["held_out_source"] = args.held_out_source
    write_json_new(args.output, payload)
    print(json.dumps({"controls": 6, "target_outcomes_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
