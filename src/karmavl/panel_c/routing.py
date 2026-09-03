"""Frozen router fitting, serialization, and route selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from karmavl.kbs_replication_b.learners import BilinearLogisticRouter, LinearLogisticRouter


def _matrix(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=np.float64)


def fit_router(
    feature_rows: Iterable[dict[str, Any]],
    correctness: dict[tuple[str, str], bool],
    *,
    learner: str,
    l2: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    rows = list(feature_rows)
    if not rows:
        raise ValueError("router fitting requires calibration rows")
    labels = np.asarray(
        [float(correctness[(str(row["query_id"]), str(row["route_id"]))]) for row in rows],
        dtype=np.float64,
    )
    query = _matrix(rows, "query_features")
    candidate = _matrix(rows, "candidate_features")
    if learner == "bilinear_logistic":
        model: Any = BilinearLogisticRouter(l2=l2, maximum_iterations=maximum_iterations)
    elif learner == "linear_logistic":
        model = LinearLogisticRouter(l2=l2, maximum_iterations=maximum_iterations)
    else:
        raise ValueError(f"unsupported frozen learner: {learner}")
    model.fit(query, candidate, labels)
    payload: dict[str, Any] = {
        "learner": learner,
        "l2": float(l2),
        "maximum_iterations": int(maximum_iterations),
        "coefficients": model.coefficients_.tolist(),
        "query_dimension": int(query.shape[1]),
        "candidate_dimension": int(candidate.shape[1]),
        "target_outcomes_used": False,
    }
    return payload


def predict_router(model_payload: dict[str, Any], feature_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(feature_rows)
    learner = str(model_payload["learner"])
    if learner == "bilinear_logistic":
        model: Any = BilinearLogisticRouter(
            l2=float(model_payload["l2"]),
            maximum_iterations=int(model_payload["maximum_iterations"]),
        )
        model.query_dimension_ = int(model_payload["query_dimension"])
        model.candidate_dimension_ = int(model_payload["candidate_dimension"])
    elif learner == "linear_logistic":
        model = LinearLogisticRouter(
            l2=float(model_payload["l2"]),
            maximum_iterations=int(model_payload["maximum_iterations"]),
        )
    else:
        raise ValueError(f"unsupported frozen learner: {learner}")
    model.coefficients_ = np.asarray(model_payload["coefficients"], dtype=np.float64)
    probabilities = model.predict_proba(
        _matrix(rows, "query_features"), _matrix(rows, "candidate_features")
    )
    return [
        {
            "query_id": str(row["query_id"]),
            "route_id": str(row["route_id"]),
            "arm": str(row["arm"]),
            "predicted_accuracy": float(probability),
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def choose_routes(
    predictions: Iterable[dict[str, Any]],
    *,
    candidate_costs: dict[str, float],
    incumbent_route: str,
    incumbent_accuracy_by_query: dict[str, float],
    incumbent_cost: float,
    margin: float,
    cost_coefficient: float,
) -> list[dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_query[str(row["query_id"])].append(row)
    output: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_query.items()):
        candidate = min(
            rows,
            key=lambda row: (
                -(
                    float(row["predicted_accuracy"])
                    - cost_coefficient * float(candidate_costs[str(row["route_id"])])
                ),
                str(row["route_id"]),
            ),
        )
        candidate_route = str(candidate["route_id"])
        candidate_utility = float(candidate["predicted_accuracy"]) - cost_coefficient * float(
            candidate_costs[candidate_route]
        )
        incumbent_utility = float(incumbent_accuracy_by_query[query_id]) - cost_coefficient * float(
            incumbent_cost
        )
        selected = candidate_route if candidate_utility > incumbent_utility + margin else incumbent_route
        output.append(
            {
                "query_id": query_id,
                "candidate_route": candidate_route,
                "selected_route": selected,
                "margin": float(margin),
                "candidate_estimated_utility": candidate_utility,
                "incumbent_estimated_utility": incumbent_utility,
            }
        )
    return output

