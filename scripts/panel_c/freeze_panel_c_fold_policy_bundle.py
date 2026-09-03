#!/usr/bin/env python3
"""Serialize and hash-freeze all five outer-fold policies before target inference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    parse_utc,
    require_frozen_execution_state,
    require_fold_policy_freeze,
    sha256_file,
    write_json_new,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-policy-freeze", action="append", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--model-pool", type=Path, required=True)
    parser.add_argument("--environment-specification", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()

    authorization = require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    if len(args.fold_policy_freeze) != 5:
        raise ValueError("exactly five per-fold policy freezes are required")
    split = load_json(args.split_manifest)
    contract = load_json(args.analysis_contract)
    pool = load_json(args.model_pool)
    expected = {
        str(row["fold_id"]): str(row["held_out_source"]) for row in split.get("folds", [])
    }
    if len(expected) != 5 or int(contract["cohort"]["outer_folds"]) != 5:
        raise ValueError("the frozen design must contain exactly five outer folds")

    frozen: dict[str, tuple[Path, dict]] = {}
    for path in args.fold_policy_freeze:
        payload = require_fold_policy_freeze(path)
        fold_id = str(payload["fold_id"])
        if fold_id in frozen:
            raise ValueError("duplicate per-fold policy freeze")
        if expected.get(fold_id) != str(payload.get("held_out_source")):
            raise ValueError("fold policy held-out source differs from the split manifest")
        if int(payload.get("schema_version", 1)) < 2:
            raise PermissionError("v7.0.2 requires dual-gate schema-v2 fold policies")
        frozen[fold_id] = (path, payload)
    if set(frozen) != set(expected):
        raise ValueError("per-fold policy freezes do not cover the exact split manifest")

    action_space = [str(pool["incumbent"]["route_id"]), *[str(row["route_id"]) for row in pool["candidates"]]]
    folds = []
    for fold_id in sorted(frozen):
        path, payload = frozen[fold_id]
        bindings = payload["bindings"]
        folds.append(
            {
                "fold_id": fold_id,
                "held_out_source": expected[fold_id],
                "H1_policy": bindings["h1_router"],
                "H1_5_policy": bindings["h1_5_router"],
                "H2_gate3_accuracy_policy": payload["H2_gate3_accuracy_policy"],
                "H2_gate4_utility_policy": payload["H2_gate4_utility_policy"],
                "B_A_star_method": payload["B_A_star_method"],
                "B_U_star_method": payload["B_U_star_method"],
                "selected_margins": {
                    "gate3_accuracy": payload["H2_gate3_accuracy_policy"]["selected_margin"],
                    "gate4_utility": payload["H2_gate4_utility_policy"]["selected_margin"],
                },
                "selected_hyperparameters": {
                    "H1_router_sha256": bindings["h1_router"]["sha256"],
                    "H1_5_router_sha256": bindings["h1_5_router"]["sha256"],
                    "H2_router_sha256": bindings["h2_router"]["sha256"],
                    "simple_controls_sha256": bindings["simple_controls"]["sha256"],
                },
                "model_action_space_metadata": {
                    "eligible_routes": action_space,
                    "incumbent_eligible": True,
                    "explicit_source_metadata_available": False,
                },
                "fold_policy_freeze": str(path.resolve()),
                "fold_policy_freeze_sha256": sha256_file(path),
                "fold_policy_created_at_utc": payload["created_at_utc"],
            }
        )

    created = datetime.now(timezone.utc)
    if any(parse_utc(row["fold_policy_created_at_utc"], field="fold_policy_created_at_utc") >= created for row in folds):
        raise PermissionError("all per-fold policies must predate the complete bundle")
    bundle = {
        "schema_version": 1,
        "artifact_role": "panel_c_complete_fold_policy_bundle",
        "status": "FROZEN_PRE_TARGET_INFERENCE",
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "target_outcomes_accessed": False,
        "target_candidate_responses_generated": False,
        "gate3_contract": "accuracy-selected H2_A versus accuracy-selected B_A_star on target accuracy",
        "gate4_contract": "utility-selected H2_U versus utility-selected B_U_star on target utility",
        "folds": folds,
    }
    write_json_new(args.output_bundle, bundle)
    bindings = {
        "policy_bundle": args.output_bundle,
        "analysis_contract": args.analysis_contract,
        "split_manifest": args.split_manifest,
        "model_pool": args.model_pool,
        "environment_specification": args.environment_specification,
        "code_manifest": args.code_manifest,
        "preregistration_archive": args.preregistration_archive,
        "authorization_lock": args.authorization_lock,
    }
    lock = {
        "schema_version": 1,
        "artifact_role": "panel_c_fold_policy_bundle_freeze",
        "status": "ALL_FOLD_POLICIES_FROZEN_PRE_TARGET_INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy_bundle_sha256": sha256_file(args.output_bundle),
        "protocol_archive_sha256": sha256_file(args.preregistration_archive),
        "external_timestamp_utc": authorization["external_timestamp_utc"],
        "target_outcomes_accessed": False,
        "target_candidate_responses_generated": False,
        "bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in bindings.items()
        },
    }
    write_json_new(args.output_lock, lock)
    print(json.dumps({"status": lock["status"], "folds": len(folds)}, sort_keys=True))


if __name__ == "__main__":
    main()
