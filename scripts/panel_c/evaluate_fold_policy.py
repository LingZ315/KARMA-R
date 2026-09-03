#!/usr/bin/env python3
"""Evaluate H2 margins and six controls on one fold's policy role only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from karmavl.panel_c.common import (
    add_frozen_execution_arguments,
    load_json,
    load_jsonl,
    require_frozen_execution_from_args,
    write_jsonl_new,
)
from karmavl.panel_c.controls import predict_simple_controls
from karmavl.panel_c.routing import choose_routes, predict_router


def _score_map(rows: list[dict[str, Any]], fold_id: str, held: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not rows or any(str(row.get("fold_id")) != fold_id or row.get("role") != "policy" for row in rows):
        raise ValueError("policy evaluation received a row outside its fold-local policy role")
    if any(str(row.get("source")) == held for row in rows):
        raise PermissionError("held-out source reached fold-local policy evaluation")
    result = {(str(row["query_id"]), str(row["route_id"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate policy score row")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--h2-features", type=Path, required=True)
    parser.add_argument("--h2-model", type=Path, required=True)
    parser.add_argument("--simple-controls", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--policy-scores", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--margin-output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    contract = load_json(args.analysis_contract)
    profiles = load_json(args.profiles)
    controls = load_json(args.simple_controls)
    semantics_rows = load_jsonl(args.semantics)
    semantics = {
        str(row["query_id"]): {
            "primary_class": row["primary_class"],
            "subtype": row.get("subtype"),
            "ambiguity": bool(row.get("ambiguity")),
            "feature_gpu_seconds": float(row.get("feature_gpu_seconds", 0.0)),
        }
        for row in semantics_rows
    }
    scores = _score_map(load_jsonl(args.policy_scores), args.fold_id, args.held_out_source)
    query_ids = sorted({query_id for query_id, _route in scores})
    all_routes = list(profiles["all_routes"])
    if set(scores) != {(query_id, route) for query_id in query_ids for route in all_routes}:
        raise ValueError("policy score matrix does not cover the frozen route pool")
    predictions = predict_router(load_json(args.h2_model), load_jsonl(args.h2_features))
    candidate_costs = {
        route: float(profiles["per_route"][route]["mean_generation_gpu_seconds"])
        for route in profiles["candidate_routes"]
    }
    incumbent = next(route for route in all_routes if route not in profiles["candidate_routes"])
    incumbent_cost = float(profiles["per_route"][incumbent]["mean_generation_gpu_seconds"])
    incumbent_accuracy = {
        query_id: float(
            profiles["per_route"][incumbent]["class_accuracy"][
                str(semantics[query_id]["primary_class"])
            ]
        )
        for query_id in query_ids
    }
    coefficient = float(contract["policy"]["cost_coefficient_per_gpu_second"])
    margin_rows: list[dict[str, Any]] = []
    for margin in contract["policy"]["margin_grid"]:
        selections = choose_routes(
            predictions,
            candidate_costs=candidate_costs,
            incumbent_route=incumbent,
            incumbent_accuracy_by_query=incumbent_accuracy,
            incumbent_cost=incumbent_cost,
            margin=float(margin),
            cost_coefficient=coefficient,
        )
        for selected in selections:
            query_id = str(selected["query_id"])
            route = str(selected["selected_route"])
            score = scores[(query_id, route)]
            serving = float(semantics[query_id]["feature_gpu_seconds"]) + float(
                score["generation_gpu_seconds"]
            )
            margin_rows.append(
                {
                    "fold_id": args.fold_id,
                    "held_out_source": args.held_out_source,
                    "role": "policy",
                    "source": score["source"],
                    "query_id": query_id,
                    "margin": float(margin),
                    "selected_route": route,
                    "correct": bool(score["correct"]),
                    "serving_gpu_seconds": serving,
                    "realized_utility": float(bool(score["correct"])) - coefficient * serving,
                    "target_outcome": False,
                }
            )

    control_predictions = predict_simple_controls(
        controls, query_ids=query_ids, semantics=semantics, profiles=profiles
    )
    requirements = contract["method_cost_requirements"]
    baseline_rows: list[dict[str, Any]] = []
    for selected in control_predictions:
        query_id = str(selected["query_id"])
        route = str(selected["selected_route"])
        baseline = str(selected["baseline_id"])
        score = scores[(query_id, route)]
        feature_cost = (
            float(semantics[query_id]["feature_gpu_seconds"])
            if requirements[baseline]["needs_semantic_classifier"]
            else 0.0
        )
        serving = feature_cost + float(score["generation_gpu_seconds"])
        baseline_rows.append(
            {
                "fold_id": args.fold_id,
                "held_out_source": args.held_out_source,
                "role": "policy",
                "source": score["source"],
                "query_id": query_id,
                "baseline_id": baseline,
                "selected_route": route,
                "correct": bool(score["correct"]),
                "serving_gpu_seconds": serving,
                "target_outcome": False,
            }
        )
    write_jsonl_new(args.margin_output, margin_rows)
    write_jsonl_new(args.baseline_output, baseline_rows)
    print(
        json.dumps(
            {
                "fold": args.fold_id,
                "policy_queries": len(query_ids),
                "margin_rows": len(margin_rows),
                "baseline_rows": len(baseline_rows),
                "held_out_source_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
