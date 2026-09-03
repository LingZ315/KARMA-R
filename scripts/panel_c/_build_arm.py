"""Shared CLI implementation for H1, H1.5, and H2 feature builders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    require_frozen_execution_state,
    write_jsonl_new,
)
from karmavl.panel_c.features import build_arm_feature_rows


def run(arm: str) -> None:
    parser = argparse.ArgumentParser(description=f"Build {arm} features from frozen profiles")
    parser.add_argument("--role", choices=("calibration", "policy", "target"), required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
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
    semantic_rows = load_jsonl(args.semantics)
    semantics = {
        str(row["query_id"]): {
            "primary_class": row["primary_class"],
            "subtype": row.get("subtype"),
            "ambiguity": bool(row.get("ambiguity")),
        }
        for row in semantic_rows
    }
    query_payload = load_json(args.query_ids)
    query_ids = [str(value) for value in query_payload["query_ids"]]
    rows = build_arm_feature_rows(
        query_ids=query_ids,
        semantics=semantics,
        profiles=load_json(args.profiles),
        arm=arm,
    )
    write_jsonl_new(args.output, rows)
    print(json.dumps({"arm": arm, "role": args.role, "rows": len(rows)}, sort_keys=True))
