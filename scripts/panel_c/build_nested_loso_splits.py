#!/usr/bin/env python3
"""Bind the frozen split IDs to fold-local decision scopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import load_json, sha256_file, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_json(args.split_manifest)
    contract = load_json(args.analysis_contract)
    if len(split["folds"]) != int(contract["cohort"]["outer_folds"]):
        raise ValueError("outer-fold count drift")
    folds = []
    target_union: set[str] = set()
    for fold in split["folds"]:
        roles = {role: fold["roles"][role] for role in ("profile", "calibration", "policy", "target")}
        target = set(map(str, roles["target"]["query_ids"]))
        if target_union & target:
            raise ValueError("target query appears in multiple outer folds")
        target_union.update(target)
        folds.append(
            {
                "fold_id": fold["fold_id"],
                "held_out_source": fold["held_out_source"],
                "roles": roles,
                "selection_scope": {
                    "profiles": "fold-local profile",
                    "router_and_controls": "fold-local calibration",
                    "margin_and_B_f_star": "fold-local policy",
                    "held_out_target_used": False,
                },
            }
        )
    if len(target_union) != int(contract["cohort"]["pooled_target_n"]):
        raise ValueError("pooled target count drift")
    payload = {
        "schema_version": 1,
        "artifact_role": "panel_c_nested_loso_split_plan",
        "status": "READY_FOR_EXTERNAL_TIMESTAMP",
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "analysis_contract_sha256": sha256_file(args.analysis_contract),
        "target_outcomes_accessed": False,
        "folds": folds,
    }
    write_json_new(args.output, payload)
    print(json.dumps({"status": payload["status"], "target_n": len(target_union)}, sort_keys=True))


if __name__ == "__main__":
    main()

