#!/usr/bin/env python3
"""Build one fold's candidate profiles using only its profile outcomes."""

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
from karmavl.panel_c.features import build_fold_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--profile-scores", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--model-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    contract = load_json(args.analysis_contract)
    pool = load_json(args.model_pool)
    semantic_rows = load_jsonl(args.semantics)
    semantics = {
        str(row["query_id"]): {
            "primary_class": row["primary_class"],
            "subtype": row.get("subtype"),
            "ambiguity": bool(row.get("ambiguity")),
        }
        for row in semantic_rows
    }
    incumbent = str(pool["incumbent"]["route_id"])
    candidates = [str(row["route_id"]) for row in pool["candidates"]]
    score_rows = load_jsonl(args.profile_scores)
    if any(
        str(row.get("fold_id")) != args.fold_id
        or row.get("role") != "profile"
        or str(row.get("source")) == args.held_out_source
        for row in score_rows
    ):
        raise PermissionError("profile rows violate the outer-fold source boundary")
    profile = build_fold_profiles(
        score_rows,
        semantics,
        candidate_routes=candidates,
        all_routes=[incumbent, *candidates],
        class_order=list(contract["semantic_schema"]["class_order"]),
        subtype_order=list(contract["semantic_schema"]["subtype_order"]),
        minimum_support=int(contract["feature_estimation"]["minimum_class_or_subtype_support"]),
    )
    profile["fold_id"] = args.fold_id
    profile["held_out_source"] = args.held_out_source
    write_json_new(args.output, profile)
    print(json.dumps({"routes": len(profile["per_route"]), "target_outcomes_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
