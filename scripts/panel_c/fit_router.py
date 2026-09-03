#!/usr/bin/env python3
"""Fit a frozen router on one outer fold's calibration rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    add_frozen_execution_arguments,
    load_jsonl,
    require_frozen_execution_from_args,
    write_json_new,
)
from karmavl.panel_c.routing import fit_router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--learner", choices=("bilinear_logistic", "linear_logistic"), required=True)
    parser.add_argument("--l2", type=float, required=True)
    parser.add_argument("--maximum-iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    score_rows = load_jsonl(args.calibration_scores)
    if any(str(row.get("role")) != "calibration" for row in score_rows):
        raise ValueError("router fitting accepts calibration rows only")
    if any(
        str(row.get("fold_id")) != args.fold_id
        or str(row.get("source")) == args.held_out_source
        for row in score_rows
    ):
        raise PermissionError("calibration rows violate the outer-fold source boundary")
    if any(row.get("target_outcome") is True for row in score_rows):
        raise ValueError("target outcome reached router fitting")
    correctness = {
        (str(row["query_id"]), str(row["route_id"])): bool(row["correct"])
        for row in score_rows
    }
    model = fit_router(
        load_jsonl(args.features),
        correctness,
        learner=args.learner,
        l2=args.l2,
        maximum_iterations=args.maximum_iterations,
    )
    model["fold_id"] = args.fold_id
    model["held_out_source"] = args.held_out_source
    write_json_new(args.output, model)
    print(json.dumps({"learner": args.learner, "target_outcomes_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
