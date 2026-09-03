#!/usr/bin/env python3
"""Verify actual outer-fold source isolation without reading any outcome ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import load_json, load_jsonl, write_json_new
from karmavl.panel_c.nested import verify_nested_loso


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--derive-source-from-target-roles", action="store_true")
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = load_json(args.split_manifest)
    if (args.source_map is None) == (not args.derive_source_from_target_roles):
        parser.error("choose exactly one of --source-map or --derive-source-from-target-roles")
    if args.source_map is not None:
        source_rows = load_jsonl(args.source_map)
        source_by_query = {str(row["query_id"]): str(row["source"]) for row in source_rows}
        if len(source_by_query) != len(source_rows):
            raise ValueError("duplicate query_id in source map")
    else:
        source_by_query: dict[str, str] = {}
        for fold in split["folds"]:
            source = str(fold["held_out_source"])
            for query_id in fold["roles"]["target"]["query_ids"]:
                query_id = str(query_id)
                if query_id in source_by_query:
                    raise ValueError("query is target in more than one source fold")
                source_by_query[query_id] = source
    audit = verify_nested_loso(
        split_manifest=split,
        source_by_query=source_by_query,
        analysis_contract=load_json(args.analysis_contract),
    )
    write_json_new(args.output, audit)
    print(json.dumps({"status": audit["status"], "folds": len(audit["folds"])}, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
