#!/usr/bin/env python3
"""Apply one fold's frozen Gate-3/Gate-4 policies without opening outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    require_frozen_execution_state,
    require_fold_policy_freeze,
    require_policy_bundle_freeze,
    sha256_file,
    write_json_new,
    write_jsonl_new,
)
from karmavl.panel_c.controls import predict_simple_controls
from karmavl.panel_c.routing import choose_routes, predict_router


def _bound(freeze: dict, name: str, path: Path) -> None:
    if sha256_file(path) != freeze["bindings"][name]["sha256"]:
        raise PermissionError(f"target routing input differs from policy freeze: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--h2-features", type=Path, required=True)
    parser.add_argument("--h2-model", type=Path, required=True)
    parser.add_argument("--simple-controls", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--selected-h2-gate3", type=Path, required=True)
    parser.add_argument("--selected-h2-gate4", type=Path, required=True)
    parser.add_argument("--selected-B-A-star", type=Path, required=True)
    parser.add_argument("--selected-B-U-star", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--policy-bundle-lock", type=Path, required=True)
    parser.add_argument("--target-semantic-receipt", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    freeze = require_fold_policy_freeze(args.policy_freeze, expected_fold_id=args.fold_id)
    _, bundle = require_policy_bundle_freeze(
        args.policy_bundle_lock,
        bundle_path=args.policy_bundle,
        expected_fold_id=args.fold_id,
    )
    bundled_fold = next(row for row in bundle["folds"] if str(row["fold_id"]) == args.fold_id)
    if bundled_fold["fold_policy_freeze_sha256"] != sha256_file(args.policy_freeze):
        raise PermissionError("per-fold policy freeze differs from the complete policy bundle")
    for name, path in (
        ("analysis_contract", args.analysis_contract),
        ("split_manifest", args.split_manifest),
        ("h2_router", args.h2_model),
        ("simple_controls", args.simple_controls),
        ("candidate_profiles", args.profiles),
        ("selected_h2_gate3_accuracy_policy", args.selected_h2_gate3),
        ("selected_h2_gate4_utility_policy", args.selected_h2_gate4),
        ("selected_B_A_star", args.selected_B_A_star),
        ("selected_B_U_star", args.selected_B_U_star),
    ):
        _bound(freeze, name, path)
    semantic_receipt = load_json(args.target_semantic_receipt)
    if (
        semantic_receipt.get("artifact_role") != "panel_c_semantic_inference_execution_receipt"
        or semantic_receipt.get("status") != "COMPLETED"
        or semantic_receipt.get("role") != "target"
        or str(semantic_receipt.get("fold_id")) != args.fold_id
    ):
        raise PermissionError("target semantic receipt has the wrong fold, role, or status")
    semantic_binding = semantic_receipt.get("bindings", {}).get("output", {})
    semantic_path = Path(str(semantic_binding.get("path", "")))
    if semantic_path.resolve() != args.semantics.resolve() or semantic_binding.get(
        "sha256"
    ) != sha256_file(args.semantics):
        raise PermissionError("target semantic ledger differs from its execution receipt")
    policy_binding = semantic_receipt.get("bindings", {}).get("policy_bundle_lock", {})
    if policy_binding.get("sha256") != sha256_file(args.policy_bundle_lock):
        raise PermissionError("target semantic receipt binds another policy bundle")
    split = load_json(args.split_manifest)
    fold = next(row for row in split["folds"] if row["fold_id"] == args.fold_id)
    target_ids = [str(value) for value in fold["roles"]["target"]["query_ids"]]
    target_set = set(target_ids)
    features = load_jsonl(args.h2_features)
    if {str(row["query_id"]) for row in features} != target_set:
        raise ValueError("H2 target features do not match the frozen held-out source")
    semantics_rows = load_jsonl(args.semantics)
    semantics = {
        str(row["query_id"]): {
            "primary_class": row["primary_class"],
            "subtype": row.get("subtype"),
            "ambiguity": bool(row.get("ambiguity")),
        }
        for row in semantics_rows
    }
    profiles = load_json(args.profiles)
    contract = load_json(args.analysis_contract)
    all_routes = list(profiles["all_routes"])
    incumbent = next(route for route in all_routes if route not in profiles["candidate_routes"])
    candidate_costs = {
        route: float(profiles["per_route"][route]["mean_generation_gpu_seconds"])
        for route in profiles["candidate_routes"]
    }
    incumbent_accuracy = {
        query_id: float(
            profiles["per_route"][incumbent]["class_accuracy"][semantics[query_id]["primary_class"]]
        )
        for query_id in target_ids
    }
    predictions = predict_router(load_json(args.h2_model), features)
    routing_arguments = {
        "candidate_costs": candidate_costs,
        "incumbent_route": incumbent,
        "incumbent_accuracy_by_query": incumbent_accuracy,
        "incumbent_cost": float(
            profiles["per_route"][incumbent]["mean_generation_gpu_seconds"]
        ),
        "cost_coefficient": float(contract["policy"]["cost_coefficient_per_gpu_second"]),
    }
    gate3_selector = load_json(args.selected_h2_gate3)
    gate4_selector = load_json(args.selected_h2_gate4)
    if gate3_selector.get("selection_objective") != "accuracy":
        raise PermissionError("Gate-3 H2 selector is not accuracy-selected")
    if gate4_selector.get("selection_objective") != "utility":
        raise PermissionError("Gate-4 H2 selector is not utility-selected")
    h2_a = choose_routes(
        predictions,
        margin=float(gate3_selector["selected_margin"]),
        **routing_arguments,
    )
    h2_u = choose_routes(
        predictions,
        margin=float(gate4_selector["selected_margin"]),
        **routing_arguments,
    )
    control_rows = predict_simple_controls(
        load_json(args.simple_controls), query_ids=target_ids, semantics=semantics, profiles=profiles
    )
    controls = {
        (str(row["query_id"]), str(row["baseline_id"])): str(row["selected_route"])
        for row in control_rows
    }
    baseline_a_selector = load_json(args.selected_B_A_star)
    baseline_u_selector = load_json(args.selected_B_U_star)
    if baseline_a_selector.get("selection_objective") != "accuracy":
        raise PermissionError("Gate-3 simple comparator is not accuracy-selected")
    if baseline_u_selector.get("selection_objective") != "utility":
        raise PermissionError("Gate-4 simple comparator is not utility-selected")
    baseline_a = str(baseline_a_selector["selected_B_f_star"])
    baseline_u = str(baseline_u_selector["selected_B_f_star"])
    h2_a_by_query = {str(row["query_id"]): row for row in h2_a}
    h2_u_by_query = {str(row["query_id"]): row for row in h2_u}
    output = [
        {
            "fold_id": args.fold_id,
            "held_out_source": fold["held_out_source"],
            "query_id": query_id,
            "h2_gate3_accuracy_selected_route": h2_a_by_query[query_id]["selected_route"],
            "h2_gate3_candidate_route": h2_a_by_query[query_id]["candidate_route"],
            "h2_gate3_margin": h2_a_by_query[query_id]["margin"],
            "b_A_star_baseline_id": baseline_a,
            "b_A_star_selected_route": controls[(query_id, baseline_a)],
            "h2_gate4_utility_selected_route": h2_u_by_query[query_id]["selected_route"],
            "h2_gate4_candidate_route": h2_u_by_query[query_id]["candidate_route"],
            "h2_gate4_margin": h2_u_by_query[query_id]["margin"],
            "b_U_star_baseline_id": baseline_u,
            "b_U_star_selected_route": controls[(query_id, baseline_u)],
            "route_scope": "confirmatory_selected_routes_only",
            "target_outcomes_accessed": False,
        }
        for query_id in target_ids
    ]
    write_jsonl_new(args.output, output)
    receipt = {
        "schema_version": 1,
        "artifact_role": "panel_c_fold_target_routing_execution_receipt",
        "status": "ROUTES_GENERATED_PRE_CANDIDATE_INFERENCE",
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fold_id": args.fold_id,
        "held_out_source": fold["held_out_source"],
        "target_queries": len(output),
        "target_outcomes_accessed": False,
        "target_candidate_responses_generated": False,
        "bindings": {
            "policy_freeze": {
                "path": str(args.policy_freeze.resolve()),
                "sha256": sha256_file(args.policy_freeze),
            },
            "target_routes": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
            "h2_target_features": {
                "path": str(args.h2_features.resolve()),
                "sha256": sha256_file(args.h2_features),
            },
            "code_manifest": {
                "path": str(args.code_manifest.resolve()),
                "sha256": sha256_file(args.code_manifest),
            },
            "policy_bundle_lock": {
                "path": str(args.policy_bundle_lock.resolve()),
                "sha256": sha256_file(args.policy_bundle_lock),
            },
            "target_semantic_receipt": {
                "path": str(args.target_semantic_receipt.resolve()),
                "sha256": sha256_file(args.target_semantic_receipt),
            },
        },
    }
    write_json_new(args.execution_receipt, receipt)
    print(json.dumps({"fold": args.fold_id, "target_routes": len(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
