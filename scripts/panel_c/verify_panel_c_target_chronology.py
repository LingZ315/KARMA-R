#!/usr/bin/env python3
"""Verify Panel-C target chronology from hash-bound execution receipts.

Candidate response and scorer-output bodies are never parsed. Their SHA-256
digests are checked as opaque bytes. Filesystem modification times are reported
only as secondary evidence and never rescue a failed primary receipt chain.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from karmavl.panel_c.common import (
    load_json,
    parse_utc,
    require_policy_bundle_freeze,
    require_target_route_bundle_freeze,
    sha256_file,
)


def _bound_path(receipt_path: Path, row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", "")))
    return path if path.is_absolute() else receipt_path.parent / path


def _verify_binding(receipt_path: Path, receipt: dict[str, Any], name: str) -> Path:
    row = receipt.get("bindings", {}).get(name)
    if not isinstance(row, dict):
        raise PermissionError(f"receipt is missing binding: {name}")
    path = _bound_path(receipt_path, row)
    if not path.is_file() or sha256_file(path) != str(row.get("sha256")):
        raise PermissionError(f"receipt binding failed: {name}")
    return path


def _load_receipts(paths: list[Path], role: str) -> list[tuple[Path, dict[str, Any]]]:
    output = []
    for path in paths:
        receipt = load_json(path)
        if receipt.get("artifact_role") != role or receipt.get("status") != "COMPLETED":
            raise PermissionError(f"wrong or incomplete receipt: {path}")
        output.append((path, receipt))
    if not output:
        raise ValueError(f"at least one {role} receipt is required")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--policy-bundle-lock", type=Path, required=True)
    parser.add_argument("--target-routes", type=Path, required=True)
    parser.add_argument("--target-route-freeze", type=Path, required=True)
    parser.add_argument("--target-semantic-receipt", action="append", type=Path, required=True)
    parser.add_argument("--target-inference-receipt", action="append", type=Path, required=True)
    parser.add_argument("--target-scoring-receipt", action="append", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    policy_lock, _ = require_policy_bundle_freeze(
        args.policy_bundle_lock, bundle_path=args.policy_bundle
    )
    route_lock, _ = require_target_route_bundle_freeze(
        args.target_route_freeze,
        routes_path=args.target_routes,
        policy_bundle_lock=args.policy_bundle_lock,
    )
    policy_time = parse_utc(policy_lock["created_at_utc"], field="policy lock created_at_utc")
    route_time = parse_utc(route_lock["created_at_utc"], field="route lock created_at_utc")
    if route_time <= policy_time:
        raise PermissionError("route freeze does not postdate complete policy freeze")

    semantic_receipts = _load_receipts(
        args.target_semantic_receipt, "panel_c_semantic_inference_execution_receipt"
    )
    semantic_times = []
    for path, receipt in semantic_receipts:
        if receipt.get("role") != "target" or receipt.get("target_candidate_responses_generated") is not False:
            raise PermissionError("semantic receipt is not an outcome-free target-stage receipt")
        _verify_binding(path, receipt, "output")
        if _verify_binding(path, receipt, "policy_bundle_lock").resolve() != args.policy_bundle_lock.resolve():
            raise PermissionError("semantic receipt binds another policy lock")
        started = parse_utc(receipt["started_at_utc"], field="semantic started_at_utc")
        completed = parse_utc(receipt["completed_at_utc"], field="semantic completed_at_utc")
        if not policy_time < started < completed < route_time:
            raise PermissionError("target semantic chronology violates policy/route boundaries")
        semantic_times.append((started, completed))

    inference_receipts = _load_receipts(
        args.target_inference_receipt, "panel_c_candidate_inference_execution_receipt"
    )
    candidate_outputs: dict[str, datetime] = {}
    secondary_mtimes: list[dict[str, Any]] = []
    for path, receipt in inference_receipts:
        if receipt.get("role") != "target" or receipt.get("target_outcomes_accessed") is not False:
            raise PermissionError("candidate receipt is not a blind target-stage receipt")
        if _verify_binding(path, receipt, "policy_bundle_lock").resolve() != args.policy_bundle_lock.resolve():
            raise PermissionError("candidate receipt binds another policy lock")
        if _verify_binding(path, receipt, "target_route_freeze").resolve() != args.target_route_freeze.resolve():
            raise PermissionError("candidate receipt binds another route lock")
        if _verify_binding(path, receipt, "target_routes").resolve() != args.target_routes.resolve():
            raise PermissionError("candidate receipt binds another route ledger")
        output = _verify_binding(path, receipt, "output")
        started = parse_utc(receipt["started_at_utc"], field="candidate started_at_utc")
        completed = parse_utc(receipt["completed_at_utc"], field="candidate completed_at_utc")
        if not route_time < started < completed:
            raise PermissionError("target candidate output predates policy or route freeze")
        output_hash = sha256_file(output)
        if output_hash in candidate_outputs:
            raise ValueError("duplicate target candidate output binding")
        candidate_outputs[output_hash] = completed
        secondary_mtimes.append(
            {
                "path": str(output),
                "mtime_utc": datetime.fromtimestamp(output.stat().st_mtime, timezone.utc).isoformat(),
                "consistent_with_route_freeze": datetime.fromtimestamp(
                    output.stat().st_mtime, timezone.utc
                )
                >= route_time,
            }
        )

    scoring_receipts = _load_receipts(
        args.target_scoring_receipt, "panel_c_target_scoring_execution_receipt"
    )
    score_completed: list[datetime] = []
    scored_prediction_hashes: set[str] = set()
    for path, receipt in scoring_receipts:
        prediction = _verify_binding(path, receipt, "predictions")
        prediction_hash = sha256_file(prediction)
        if prediction_hash not in candidate_outputs:
            raise PermissionError("scorer receipt references an unverified target prediction ledger")
        started = parse_utc(receipt["started_at_utc"], field="scoring started_at_utc")
        completed = parse_utc(receipt["completed_at_utc"], field="scoring completed_at_utc")
        if not candidate_outputs[prediction_hash] < started < completed:
            raise PermissionError("target scorer did not postdate its candidate inference")
        _verify_binding(path, receipt, "output")
        scored_prediction_hashes.add(prediction_hash)
        score_completed.append(completed)
    if scored_prediction_hashes != set(candidate_outputs):
        raise PermissionError("not every sealed candidate output has a scoring receipt")

    analysis = load_json(args.analysis_receipt)
    if analysis.get("artifact_role") != "panel_c_primary_analysis_execution_receipt" or analysis.get("status") != "COMPLETED":
        raise PermissionError("primary analysis receipt is missing or incomplete")
    analysis_started = parse_utc(analysis["started_at_utc"], field="analysis started_at_utc")
    analysis_completed = parse_utc(analysis["completed_at_utc"], field="analysis completed_at_utc")
    if not max(score_completed) < analysis_started < analysis_completed:
        raise PermissionError("primary analysis did not postdate all target scoring")
    _verify_binding(args.analysis_receipt, analysis, "primary_summary")

    report = {
        "schema_version": 1,
        "artifact_role": "panel_c_target_chronology_verification",
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "primary_evidence": {
            "policy_freeze_precedes_target_semantics": True,
            "policy_freeze_precedes_target_candidate_inference": True,
            "route_freeze_precedes_target_candidate_inference": True,
            "target_scoring_postdates_corresponding_candidate_inference": True,
            "primary_analysis_postdates_all_target_scoring": True,
            "hash_bindings_verified": True,
        },
        "filesystem_mtime_secondary_only": secondary_mtimes,
        "candidate_response_bodies_parsed": False,
        "target_outcome_bodies_parsed_by_verifier": False,
        "counts": {
            "semantic_receipts": len(semantic_receipts),
            "candidate_inference_receipts": len(inference_receipts),
            "scoring_receipts": len(scoring_receipts),
        },
    }
    material = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(material)
    print(json.dumps({"status": "PASS", "candidate_outputs": len(candidate_outputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
