#!/usr/bin/env python3
"""Combine one fold/role score matrix after validating route-complete coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    add_frozen_execution_arguments,
    load_json,
    load_jsonl,
    require_frozen_execution_from_args,
    write_jsonl_new,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--role", choices=("profile", "calibration", "policy", "target"), required=True)
    parser.add_argument("--model-pool", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    pool = load_json(args.model_pool)
    routes = [pool["incumbent"]["route_id"], *[row["route_id"] for row in pool["candidates"]]]
    rows = [row for path in args.input for row in load_jsonl(path)]
    if any(
        str(row.get("fold_id")) != args.fold_id
        or row.get("role") != args.role
        or ((args.role == "target") != (str(row.get("source")) == args.held_out_source))
        for row in rows
    ):
        raise PermissionError("score matrix violates the fold/role/source boundary")
    by_route = {route: [row for row in rows if str(row["route_id"]) == route] for route in routes}
    if any(not values for values in by_route.values()) or set(row["route_id"] for row in rows) != set(routes):
        raise ValueError("score matrix does not cover the exact frozen route pool")
    query_sets = [{str(row["query_id"]) for row in values} for values in by_route.values()]
    if len({frozenset(values) for values in query_sets}) != 1:
        raise ValueError("route score ledgers cover different query sets")
    output = sorted(rows, key=lambda row: (str(row["query_id"]), str(row["route_id"])))
    write_jsonl_new(args.output, output)
    print(json.dumps({"routes": len(routes), "queries": len(query_sets[0])}, sort_keys=True))


if __name__ == "__main__":
    main()
