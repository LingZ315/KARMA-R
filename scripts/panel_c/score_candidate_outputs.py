#!/usr/bin/env python3
"""Score one route within one exact fold role under the outcome firewall."""

from __future__ import annotations

import argparse
import hashlib
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
from karmavl.panel_c.scoring import score_response


def _fold(split: dict[str, Any], fold_id: str) -> dict[str, Any]:
    matches = [row for row in split["folds"] if str(row["fold_id"]) == fold_id]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate outer fold")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--role", choices=("profile", "calibration", "policy", "target"), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--policy-bundle-lock", type=Path)
    parser.add_argument("--target-routes", type=Path)
    parser.add_argument("--target-route-freeze", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    args = parser.parse_args()
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    started_at = datetime.now(timezone.utc)
    frozen_target_routes: list[dict[str, Any]] | None = None
    if args.role == "target":
        if any(
            value is None
            for value in (
                args.policy_bundle,
                args.policy_bundle_lock,
                args.target_routes,
                args.target_route_freeze,
                args.execution_receipt,
            )
        ):
            raise PermissionError("target scoring requires complete policy/route freezes and a receipt")
        require_policy_bundle_freeze(
            args.policy_bundle_lock,
            bundle_path=args.policy_bundle,
            expected_fold_id=args.fold_id,
        )
        _, frozen_target_routes = require_target_route_bundle_freeze(
            args.target_route_freeze,
            routes_path=args.target_routes,
            policy_bundle_lock=args.policy_bundle_lock,
            expected_fold_id=args.fold_id,
        )
    elif any(
        value is not None
        for value in (
            args.policy_bundle,
            args.policy_bundle_lock,
            args.target_routes,
            args.target_route_freeze,
        )
    ):
        raise ValueError("non-target role scoring must not receive target freeze artifacts")

    # Validate the scoped path before opening any answer-bearing file.
    expected_name = f"{args.role}_reference_answers.jsonl"
    if args.outcomes.name != expected_name or args.outcomes.parent.name != args.fold_id:
        raise PermissionError(
            f"scorer accepts only {args.fold_id}/{expected_name}; a pooled answer ledger is forbidden"
        )
    split = load_json(args.split_manifest)
    fold = _fold(split, args.fold_id)
    expected_ids = {str(value) for value in fold["roles"][args.role]["query_ids"]}
    inputs = {str(row["query_id"]): row for row in load_jsonl(args.input)}
    manifests = {str(row["query_id"]): row for row in load_jsonl(args.row_manifest)}
    predictions = load_jsonl(args.predictions)
    routes = {str(row["route_id"]) for row in predictions}
    if len(routes) != 1:
        raise ValueError("one prediction ledger must contain exactly one frozen route")
    prediction_map = {str(row["query_id"]): row for row in predictions}
    score_ids = expected_ids
    if args.role == "target":
        route_id = next(iter(routes))
        selected_fields = (
            "h2_gate3_accuracy_selected_route",
            "b_A_star_selected_route",
            "h2_gate4_utility_selected_route",
            "b_U_star_selected_route",
        )
        required_global_ids = {
            str(row["query_id"])
            for row in frozen_target_routes
            if route_id in {str(row[field]) for field in selected_fields}
        }
        score_ids = {
            str(row["query_id"])
            for row in frozen_target_routes
            if str(row.get("fold_id")) == args.fold_id
            and route_id in {str(row[field]) for field in selected_fields}
        }
        if set(prediction_map) != required_global_ids:
            raise PermissionError("target predictions are not the exact all-fold frozen minimal route subset")
    elif set(prediction_map) != expected_ids:
        raise ValueError("prediction ledger does not cover the exact fold role")

    # This is the first point at which the scoped reference ledger is opened.
    outcome_rows = load_jsonl(args.outcomes)
    if {str(row["query_id"]) for row in outcome_rows} != expected_ids:
        raise PermissionError("fold-scoped answer ledger query IDs do not match the frozen role")
    outcomes = {str(row["query_id"]): row for row in outcome_rows}
    held = str(fold["held_out_source"])
    scored: list[dict[str, Any]] = []
    for query_id in sorted(score_ids):
        input_row = inputs[query_id]
        manifest = manifests[query_id]
        source = str(manifest["source"])
        if (args.role == "target") != (source == held):
            raise PermissionError("held-out-source role boundary failed")
        prompt = str(input_row.get("prompt", input_row.get("question", "")))
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prediction = prediction_map[query_id]
        if prediction.get("prompt_sha256") != prompt_hash or manifest.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"prompt hash mismatch for {query_id}")
        if prediction.get("image_sha256") != manifest.get("normalized_image_sha256"):
            raise ValueError(f"image hash mismatch for {query_id}")
        outcome = outcomes[query_id]
        scorer = str(outcome["scorer"])
        if scorer != str(manifest["scorer"]):
            raise ValueError(f"scorer drift for {query_id}")
        correct = bool(prediction.get("success")) and score_response(
            prediction.get("response", ""), outcome["answer"], scorer, prompt
        )
        scored.append(
            {
                "fold_id": args.fold_id,
                "held_out_source": held,
                "role": args.role,
                "source": source,
                "query_id": query_id,
                "route_id": next(iter(routes)),
                "correct": bool(correct),
                "generation_gpu_seconds": float(prediction.get("generation_gpu_seconds", 0.0)),
                "success": bool(prediction.get("success")),
            }
        )
    write_jsonl_new(args.output, scored)
    if args.execution_receipt is not None:
        receipt = {
            "schema_version": 1,
            "artifact_role": "panel_c_target_scoring_execution_receipt"
            if args.role == "target"
            else "panel_c_non_target_scoring_execution_receipt",
            "status": "COMPLETED",
            "role": args.role,
            "fold_id": args.fold_id,
            "route_id": next(iter(routes)),
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "bindings": {
                "predictions": {
                    "path": str(args.predictions.resolve()),
                    "sha256": sha256_file(args.predictions),
                },
                "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
                "authorization_lock": {
                    "path": str(args.authorization_lock.resolve()),
                    "sha256": sha256_file(args.authorization_lock),
                },
            },
        }
        if args.role == "target":
            receipt["bindings"]["policy_bundle_lock"] = {
                "path": str(args.policy_bundle_lock.resolve()),
                "sha256": sha256_file(args.policy_bundle_lock),
            }
            receipt["bindings"]["target_route_freeze"] = {
                "path": str(args.target_route_freeze.resolve()),
                "sha256": sha256_file(args.target_route_freeze),
            }
        write_json_new(args.execution_receipt, receipt)
    print(json.dumps({"fold": args.fold_id, "role": args.role, "rows": len(scored)}, sort_keys=True))


if __name__ == "__main__":
    main()
