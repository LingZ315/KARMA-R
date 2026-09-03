#!/usr/bin/env python3
"""Combine all fold route ledgers and freeze them before target responses exist."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    parse_utc,
    require_frozen_execution_state,
    require_policy_bundle_freeze,
    sha256_file,
    write_json_new,
    write_jsonl_new,
)


ROUTE_FIELDS = {
    "fold_id",
    "held_out_source",
    "query_id",
    "h2_gate3_accuracy_selected_route",
    "b_A_star_baseline_id",
    "b_A_star_selected_route",
    "h2_gate4_utility_selected_route",
    "b_U_star_baseline_id",
    "b_U_star_selected_route",
    "route_scope",
    "target_outcomes_accessed",
}


def _receipt_map(paths: list[Path], role: str) -> dict[str, tuple[Path, dict]]:
    output: dict[str, tuple[Path, dict]] = {}
    for path in paths:
        receipt = load_json(path)
        fold_id = str(receipt.get("fold_id", ""))
        if fold_id in output:
            raise ValueError(f"duplicate {role} receipt for {fold_id}")
        output[fold_id] = (path, receipt)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-route", action="append", type=Path, required=True)
    parser.add_argument("--fold-routing-receipt", action="append", type=Path, required=True)
    parser.add_argument("--target-semantic-receipt", action="append", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--policy-bundle-lock", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()

    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    policy_lock, _ = require_policy_bundle_freeze(
        args.policy_bundle_lock, bundle_path=args.policy_bundle
    )
    policy_time = parse_utc(policy_lock["created_at_utc"], field="policy lock created_at_utc")
    split = load_json(args.split_manifest)
    expected = {str(row["fold_id"]): row for row in split["folds"]}
    if len(expected) != 5 or any(len(values) != 5 for values in (args.fold_route, args.fold_routing_receipt, args.target_semantic_receipt)):
        raise ValueError("exactly five folds, route ledgers, routing receipts, and semantic receipts are required")
    routing_receipts = _receipt_map(args.fold_routing_receipt, "routing")

    semantic_receipts: dict[str, tuple[Path, dict]] = {}
    for path in args.target_semantic_receipt:
        receipt = load_json(path)
        if receipt.get("artifact_role") != "panel_c_semantic_inference_execution_receipt" or receipt.get("role") != "target":
            raise PermissionError("target semantic receipt has the wrong role")
        fold_id = str(receipt.get("fold_id", ""))
        if not fold_id:
            # The semantic runner may be invoked once for the all-fold target
            # ledger; that single receipt can be copied into five fold-indexed
            # orchestration receipts, but an explicit fold_id is mandatory here.
            raise PermissionError("target semantic receipt must declare fold_id")
        if fold_id in semantic_receipts:
            raise ValueError("duplicate target semantic receipt")
        if parse_utc(receipt["started_at_utc"], field="semantic started_at_utc") <= policy_time:
            raise PermissionError("target semantic inference did not start after the policy-bundle freeze")
        semantic_receipts[fold_id] = (path, receipt)

    route_files: dict[str, Path] = {}
    combined: list[dict] = []
    for path in args.fold_route:
        rows = load_jsonl(path)
        fold_ids = {str(row.get("fold_id")) for row in rows}
        if len(fold_ids) != 1:
            raise ValueError("one fold route file must contain exactly one fold")
        fold_id = next(iter(fold_ids))
        if fold_id in route_files or fold_id not in expected:
            raise ValueError("duplicate or unknown fold route file")
        route_files[fold_id] = path
        expected_ids = {str(value) for value in expected[fold_id]["roles"]["target"]["query_ids"]}
        if {str(row.get("query_id")) for row in rows} != expected_ids:
            raise ValueError("fold route file does not cover the exact held-out target IDs")
        if any(not ROUTE_FIELDS <= set(row) for row in rows):
            raise ValueError("fold route file is missing a confirmatory route field")
        if any(row.get("target_outcomes_accessed") is not False for row in rows):
            raise PermissionError("fold route file lacks an outcome-blind assertion")
        combined.extend(rows)
    if set(route_files) != set(expected) or set(routing_receipts) != set(expected) or set(semantic_receipts) != set(expected):
        raise ValueError("route chronology inputs do not cover all five folds")

    for fold_id in sorted(expected):
        route_path = route_files[fold_id]
        routing_path, routing = routing_receipts[fold_id]
        semantic_path, semantic = semantic_receipts[fold_id]
        if routing.get("artifact_role") != "panel_c_fold_target_routing_execution_receipt":
            raise PermissionError("routing receipt has the wrong artifact role")
        if routing.get("target_candidate_responses_generated") is not False:
            raise PermissionError("routing receipt is not pre-response")
        if routing["bindings"]["target_routes"]["sha256"] != sha256_file(route_path):
            raise PermissionError("routing receipt does not bind its fold route file")
        semantic_completed = parse_utc(semantic["completed_at_utc"], field="semantic completed_at_utc")
        routing_started = parse_utc(routing["started_at_utc"], field="routing started_at_utc")
        if routing_started <= semantic_completed:
            raise PermissionError("target routing did not start after target semantic inference")
        # Bindings are retained below; variables are intentionally referenced
        # here to make both receipts part of the primary evidence chain.
        if not routing_path.is_file() or not semantic_path.is_file():
            raise FileNotFoundError("a chronology receipt disappeared")

    combined.sort(key=lambda row: (str(row["fold_id"]), str(row["query_id"])))
    if len({str(row["query_id"]) for row in combined}) != len(combined):
        raise ValueError("a target query appears in more than one frozen fold route")
    write_jsonl_new(args.output, combined)
    created = datetime.now(timezone.utc)
    lock_bindings: dict[str, Path] = {
        "target_routes": args.output,
        "policy_bundle": args.policy_bundle,
        "policy_bundle_lock": args.policy_bundle_lock,
        "split_manifest": args.split_manifest,
        "preregistration_archive": args.preregistration_archive,
        "code_manifest": args.code_manifest,
        "authorization_lock": args.authorization_lock,
    }
    for fold_id in sorted(expected):
        lock_bindings[f"{fold_id}_routing_receipt"] = routing_receipts[fold_id][0]
        lock_bindings[f"{fold_id}_semantic_receipt"] = semantic_receipts[fold_id][0]
    lock = {
        "schema_version": 1,
        "artifact_role": "panel_c_target_route_bundle_freeze",
        "status": "ALL_TARGET_ROUTES_FROZEN_PRE_CANDIDATE_INFERENCE",
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "route_file_sha256": sha256_file(args.output),
        "protocol_archive_sha256": sha256_file(args.preregistration_archive),
        "policy_bundle_sha256": sha256_file(args.policy_bundle),
        "target_queries": len(combined),
        "target_candidate_responses_generated": False,
        "target_outcomes_accessed": False,
        "bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in lock_bindings.items()
        },
    }
    write_json_new(args.output_lock, lock)
    print(json.dumps({"status": lock["status"], "target_queries": len(combined)}, sort_keys=True))


if __name__ == "__main__":
    main()
