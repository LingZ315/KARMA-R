"""Fold-local fitting and prediction for the six frozen simple controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from karmavl.kbs_replication_b.learners import LinearLogisticRouter

from .features import semantic_query_vector


def _best_route(rows: list[dict[str, Any]], routes: list[str], cost_coefficient: float) -> str:
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_route[str(row["route_id"])].append(row)
    if set(by_route) != set(routes):
        raise ValueError("control-fitting route coverage mismatch")
    summary = {}
    for route, values in by_route.items():
        accuracy = sum(float(bool(row["correct"])) for row in values) / len(values)
        cost = sum(float(row["generation_gpu_seconds"]) for row in values) / len(values)
        summary[route] = (accuracy, accuracy - cost_coefficient * cost, cost)
    fixed_rank = {route: index for index, route in enumerate(routes)}
    return min(
        routes,
        key=lambda route: (
            -summary[route][0],
            -summary[route][1],
            summary[route][2],
            fixed_rank[route],
        ),
    )


def fit_simple_controls(
    *,
    profile_rows: Iterable[dict[str, Any]],
    calibration_rows: Iterable[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    profiles: dict[str, Any],
    incumbent_route: str,
    candidate_routes: list[str],
    class_order: list[str],
    subtype_order: list[str],
    minimum_support: int,
    cost_coefficient: float,
    logistic_l2: float,
    logistic_iterations: int,
) -> dict[str, Any]:
    profile = list(profile_rows)
    calibration = list(calibration_rows)
    if any(row.get("target_outcome") is True for row in [*profile, *calibration]):
        raise ValueError("target outcome reached simple-control fitting")
    all_routes = [incumbent_route, *candidate_routes]
    profile_cost = {
        route: profiles["per_route"][route]["mean_generation_gpu_seconds"] for route in all_routes
    }
    cheapest = min(candidate_routes, key=lambda route: (float(profile_cost[route]), route))
    # v7.0.2 matches the deployable action space of the strongest adaptive
    # controls to H2: incumbent plus every frozen candidate route.
    global_best = _best_route(calibration, all_routes, cost_coefficient)
    by_class: dict[str, str] = {}
    for class_name in class_order:
        selected = [
            row
            for row in calibration
            if semantics[str(row["query_id"])]["primary_class"] == class_name
        ]
        query_support = len({str(row["query_id"]) for row in selected})
        by_class[class_name] = (
            _best_route(selected, all_routes, cost_coefficient)
            if query_support >= minimum_support
            else global_best
        )

    route_index = {route: index for index, route in enumerate(all_routes)}
    pairs = [row for row in calibration if str(row["route_id"]) in route_index]
    query_matrix = np.stack(
        [
            semantic_query_vector(
                semantics[str(row["query_id"])],
                class_order=class_order,
                subtype_order=subtype_order,
            )
            for row in pairs
        ]
    )
    candidate_matrix = np.zeros((len(pairs), len(all_routes)), dtype=np.float64)
    for index, row in enumerate(pairs):
        candidate_matrix[index, route_index[str(row["route_id"])]] = 1.0
    labels = np.asarray([float(bool(row["correct"])) for row in pairs], dtype=np.float64)
    raw_model = LinearLogisticRouter(l2=logistic_l2, maximum_iterations=logistic_iterations).fit(
        query_matrix, candidate_matrix, labels
    )
    return {
        "incumbent_only": {"route_id": incumbent_route},
        "cheapest_candidate": {"route_id": cheapest},
        "static_calibration_global_best": {"route_id": global_best},
        "static_class_conditional_best": {
            "route_by_class": by_class,
            "fallback_route": global_best,
        },
        "logistic_raw": {
            "candidate_routes": all_routes,
            "eligible_routes": all_routes,
            "coefficients": raw_model.coefficients_.tolist(),
            "l2": logistic_l2,
            "maximum_iterations": logistic_iterations,
            "training_contract": {
                "input_features": "frozen semantic primary-class one-hot, subtype one-hot, ambiguity indicator, plus eligible-route one-hot",
                "normalization": "none; all design inputs are binary indicators",
                "solver": "deterministic full-batch Newton/IRLS with direct linear solve",
                "regularization": "L2 on all non-intercept coefficients; intercept unpenalized",
                "convergence": "maximum absolute Newton step below 1e-10 or maximum_iterations",
                "class_weighting": "none",
                "labels": "per-query per-route binary correctness on fold-local calibration rows",
                "action_space": all_routes,
                "target_driven_tuning": False,
            },
        },
        "nearest_profile": {
            "candidate_routes": all_routes,
            "eligible_routes": all_routes,
            "cost_coefficient": cost_coefficient,
            "training_contract": {
                "representation": "[global profile accuracy, active primary-class accuracy, active subtype accuracy or class fallback]",
                "distance_metric": "unweighted Euclidean distance to the ideal vector [1,1,1]",
                "tie_break": ["smaller_distance", "lower_profile_generation_gpu_seconds", "fixed_action_space_order"],
                "candidate_pool": all_routes,
                "incumbent_representation_rule": "identical profile construction and smoothing as every candidate route",
                "target_driven_tuning": False,
            },
        },
        "matched_action_space": {
            "eligible_routes": all_routes,
            "incumbent_eligible": True,
            "strong_adaptive_controls": [
                "static_class_conditional_best",
                "logistic_raw",
                "nearest_profile",
            ],
            "static_global_incumbent_eligible": True,
            "source_id_conditional_control": False,
        },
        "profile_costs": profile_cost,
        "target_outcomes_used": False,
    }


def predict_simple_controls(
    controls: dict[str, Any],
    *,
    query_ids: Iterable[str],
    semantics: dict[str, dict[str, Any]],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    class_order = list(profiles["class_order"])
    subtype_order = list(profiles["subtype_order"])
    routes = list(controls["logistic_raw"]["candidate_routes"])
    route_index = {route: index for index, route in enumerate(routes)}
    raw = controls["logistic_raw"]
    raw_model = LinearLogisticRouter(
        l2=float(raw["l2"]), maximum_iterations=int(raw["maximum_iterations"])
    )
    raw_model.coefficients_ = np.asarray(raw["coefficients"], dtype=np.float64)
    output: list[dict[str, Any]] = []
    for query_id in query_ids:
        query_id = str(query_id)
        semantic = semantics[query_id]
        query_vector = semantic_query_vector(
            semantic, class_order=class_order, subtype_order=subtype_order
        )
        static_class = controls["static_class_conditional_best"]["route_by_class"].get(
            semantic["primary_class"], controls["static_class_conditional_best"]["fallback_route"]
        )
        raw_probabilities = {}
        for route in routes:
            candidate = np.zeros((1, len(routes)), dtype=np.float64)
            candidate[0, route_index[route]] = 1.0
            raw_probabilities[route] = float(
                raw_model.predict_proba(query_vector[None, :], candidate)[0]
            )
        logistic_route = min(
            routes,
            key=lambda route: (-raw_probabilities[route], route_index[route]),
        )
        primary = str(semantic["primary_class"])
        subtype = semantic.get("subtype")
        nearest_distance = {}
        for route in routes:
            profile = profiles["per_route"][route]
            class_accuracy = float(profile["class_accuracy"][primary])
            conditional_accuracy = class_accuracy
            if subtype in subtype_order and int(profile["subtype_support"][subtype]) >= int(
                profiles["minimum_support"]
            ):
                conditional_accuracy = float(profile["subtype_accuracy"][subtype])
            representation = np.asarray(
                [
                    float(profile["global_accuracy"]),
                    class_accuracy,
                    conditional_accuracy,
                ],
                dtype=np.float64,
            )
            nearest_distance[route] = float(np.linalg.norm(np.ones(3) - representation))
        fixed_rank = {route: index for index, route in enumerate(routes)}
        nearest_route = min(
            routes,
            key=lambda route: (
                nearest_distance[route],
                float(controls["profile_costs"][route]),
                fixed_rank[route],
            ),
        )
        selections = {
            "incumbent_only": controls["incumbent_only"]["route_id"],
            "cheapest_candidate": controls["cheapest_candidate"]["route_id"],
            "static_calibration_global_best": controls["static_calibration_global_best"]["route_id"],
            "static_class_conditional_best": static_class,
            "logistic_raw": logistic_route,
            "nearest_profile": nearest_route,
        }
        output.extend(
            {"query_id": query_id, "baseline_id": name, "selected_route": route}
            for name, route in selections.items()
        )
    return output
