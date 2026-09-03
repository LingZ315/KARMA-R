#!/usr/bin/env python3
"""Assemble and calculate the fold-nested Panel-C primary endpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    require_frozen_execution_state,
    require_policy_bundle_freeze,
    require_target_route_bundle_freeze,
    sha256_file,
    write_json_new,
    write_jsonl_new,
)
from karmavl.panel_c.statistics import primary_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Bundle schema: {folds:[{fold_id,semantics,target_score_files:[...]}]}"
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--utility-paired-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--analysis-freeze-output", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--policy-bundle-lock", type=Path, required=True)
    parser.add_argument("--target-routes", type=Path, required=True)
    parser.add_argument("--target-route-freeze", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
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
    started_at = datetime.now(timezone.utc)
    require_policy_bundle_freeze(args.policy_bundle_lock, bundle_path=args.policy_bundle)
    _, frozen_route_rows = require_target_route_bundle_freeze(
        args.target_route_freeze,
        routes_path=args.target_routes,
        policy_bundle_lock=args.policy_bundle_lock,
    )
    routes_by_fold: dict[str, list[dict[str, Any]]] = {}
    for row in frozen_route_rows:
        routes_by_fold.setdefault(str(row["fold_id"]), []).append(row)
    bundle = load_json(args.bundle)
    contract = load_json(args.analysis_contract)
    expected_folds = int(contract["cohort"]["outer_folds"])
    if len(bundle.get("folds", [])) != expected_folds:
        raise ValueError("analysis bundle does not contain every outer fold")
    requirements = contract["method_cost_requirements"]
    paired: list[dict[str, Any]] = []
    utility_paired: list[dict[str, Any]] = []
    input_bindings: dict[str, str] = {}
    seen_folds: set[str] = set()
    for spec in bundle["folds"]:
        fold_id = str(spec["fold_id"])
        if fold_id in seen_folds:
            raise ValueError("duplicate fold in analysis bundle")
        seen_folds.add(fold_id)
        routing = routes_by_fold.get(fold_id, [])
        if not routing:
            raise ValueError("frozen target-route ledger is missing an analysis fold")
        semantics = {str(row["query_id"]): row for row in load_jsonl(Path(spec["semantics"]))}
        score_map: dict[tuple[str, str], dict[str, Any]] = {}
        for path_text in spec["target_score_files"]:
            score_path = Path(path_text)
            rows = load_jsonl(score_path)
            if any(row.get("role") != "target" or str(row.get("fold_id")) != fold_id for row in rows):
                raise ValueError("target score file contains another role or fold")
            for row in rows:
                key = (str(row["query_id"]), str(row["route_id"]))
                if key in score_map:
                    raise ValueError("duplicate target score")
                score_map[key] = row
            input_bindings[f"{fold_id}/score/{score_path.name}"] = sha256_file(score_path)
        for route in routing:
            query_id = str(route["query_id"])
            h2_route = str(route["h2_gate3_accuracy_selected_route"])
            baseline_route = str(route["b_A_star_selected_route"])
            baseline_id = str(route["b_A_star_baseline_id"])
            h2_score = score_map[(query_id, h2_route)]
            baseline_score = score_map[(query_id, baseline_route)]
            semantic_cost = float(semantics[query_id].get("feature_gpu_seconds", 0.0))
            baseline_feature = (
                semantic_cost if requirements[baseline_id]["needs_semantic_classifier"] else 0.0
            )
            paired.append(
                {
                    "query_id": query_id,
                    "fold_id": fold_id,
                    "source": route["held_out_source"],
                    "h2_gate3_accuracy_selected_route": h2_route,
                    "b_A_star_baseline_id": baseline_id,
                    "b_A_star_selected_route": baseline_route,
                    "h2_correct": bool(h2_score["correct"]),
                    "b_f_star_correct": bool(baseline_score["correct"]),
                    "h2_serving_gpu_seconds": semantic_cost
                    + float(h2_score["generation_gpu_seconds"]),
                    "b_f_star_serving_gpu_seconds": baseline_feature
                    + float(baseline_score["generation_gpu_seconds"]),
                }
            )
            h2_u_route = str(route["h2_gate4_utility_selected_route"])
            baseline_u_route = str(route["b_U_star_selected_route"])
            baseline_u_id = str(route["b_U_star_baseline_id"])
            h2_u_score = score_map[(query_id, h2_u_route)]
            baseline_u_score = score_map[(query_id, baseline_u_route)]
            baseline_u_feature = (
                semantic_cost
                if requirements[baseline_u_id]["needs_semantic_classifier"]
                else 0.0
            )
            utility_paired.append(
                {
                    "query_id": query_id,
                    "fold_id": fold_id,
                    "source": route["held_out_source"],
                    "h2_gate4_utility_selected_route": h2_u_route,
                    "b_U_star_baseline_id": baseline_u_id,
                    "b_U_star_selected_route": baseline_u_route,
                    "h2_correct": bool(h2_u_score["correct"]),
                    "b_f_star_correct": bool(baseline_u_score["correct"]),
                    "h2_serving_gpu_seconds": semantic_cost
                    + float(h2_u_score["generation_gpu_seconds"]),
                    "b_f_star_serving_gpu_seconds": baseline_u_feature
                    + float(baseline_u_score["generation_gpu_seconds"]),
                }
            )
        input_bindings[f"{fold_id}/semantics"] = sha256_file(Path(spec["semantics"]))
    input_bindings["policy_bundle"] = sha256_file(args.policy_bundle)
    input_bindings["policy_bundle_lock"] = sha256_file(args.policy_bundle_lock)
    input_bindings["target_routes"] = sha256_file(args.target_routes)
    input_bindings["target_route_freeze"] = sha256_file(args.target_route_freeze)
    if len(paired) != int(contract["cohort"]["pooled_target_n"]):
        raise ValueError("paired target count differs from the frozen cohort")
    if len(utility_paired) != len(paired):
        raise ValueError("Gate-4 utility pairs do not match the complete target cohort")
    write_jsonl_new(args.paired_output, paired)
    write_jsonl_new(args.utility_paired_output, utility_paired)
    summary = {
        "schema_version": 1,
        "artifact_role": "panel_c_primary_endpoint_result",
        "endpoint": contract["primary_endpoint"],
        "result": primary_endpoint(paired),
        "practical_threshold": float(
            str(contract["primary_endpoint"]["practical_superiority"]).split(">=")[-1]
        ),
        "input_sha256": input_bindings,
    }
    write_json_new(args.summary_output, summary)
    freeze = {
        "schema_version": 1,
        "artifact_role": "panel_c_primary_analysis_input_freeze",
        "status": "PAIRED_PRIMARY_INPUT_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bindings": {
            "paired_target_rows": {
                "path": str(args.paired_output.resolve()),
                "sha256": sha256_file(args.paired_output),
            },
            "primary_summary": {
                "path": str(args.summary_output.resolve()),
                "sha256": sha256_file(args.summary_output),
            },
            "utility_paired_target_rows": {
                "path": str(args.utility_paired_output.resolve()),
                "sha256": sha256_file(args.utility_paired_output),
            },
            "analysis_contract": {
                "path": str(args.analysis_contract.resolve()),
                "sha256": sha256_file(args.analysis_contract),
            },
            "code_manifest": {
                "path": str(args.code_manifest.resolve()),
                "sha256": sha256_file(args.code_manifest),
            },
        },
    }
    write_json_new(args.analysis_freeze_output, freeze)
    receipt = {
        "schema_version": 1,
        "artifact_role": "panel_c_primary_analysis_execution_receipt",
        "status": "COMPLETED",
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bindings": {
            "primary_summary": {
                "path": str(args.summary_output.resolve()),
                "sha256": sha256_file(args.summary_output),
            },
            "paired_target_rows": {
                "path": str(args.paired_output.resolve()),
                "sha256": sha256_file(args.paired_output),
            },
            "utility_paired_target_rows": {
                "path": str(args.utility_paired_output.resolve()),
                "sha256": sha256_file(args.utility_paired_output),
            },
            "policy_bundle_lock": {
                "path": str(args.policy_bundle_lock.resolve()),
                "sha256": sha256_file(args.policy_bundle_lock),
            },
            "target_route_freeze": {
                "path": str(args.target_route_freeze.resolve()),
                "sha256": sha256_file(args.target_route_freeze),
            },
        },
    }
    write_json_new(args.execution_receipt, receipt)
    print(json.dumps({"folds": len(seen_folds), "target_n": len(paired)}, sort_keys=True))


if __name__ == "__main__":
    main()
