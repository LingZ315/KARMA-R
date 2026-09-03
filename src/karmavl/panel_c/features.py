"""Fold-local semantic and candidate-profile feature construction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def semantic_query_vector(
    row: dict[str, Any], *, class_order: list[str], subtype_order: list[str]
) -> np.ndarray:
    forbidden = {"source", "source_id", "dataset", "dataset_id", "benchmark", "source_one_hot"}
    if forbidden & {key.casefold() for key in row}:
        raise ValueError("explicit source metadata reached semantic router feature construction")
    vector = np.zeros(len(class_order) + len(subtype_order) + 1, dtype=np.float64)
    primary = str(row["primary_class"])
    if primary not in class_order:
        raise ValueError(f"unknown primary semantic class: {primary}")
    vector[class_order.index(primary)] = 1.0
    subtype = row.get("subtype")
    if subtype not in (None, "", "none"):
        subtype = str(subtype)
        if subtype not in subtype_order:
            raise ValueError(f"unknown semantic subtype: {subtype}")
        vector[len(class_order) + subtype_order.index(subtype)] = 1.0
    vector[-1] = float(bool(row.get("ambiguity")))
    return vector


def _global(successes: int, support: int) -> float:
    return (float(successes) + 1.0) / (float(support) + 2.0)


def _conditional(successes: int, support: int, global_accuracy: float, minimum: int) -> float:
    if support < minimum:
        return global_accuracy
    return (float(successes) + 2.0 * float(global_accuracy)) / (float(support) + 2.0)


def _profile_summary(
    rows: list[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    *,
    class_order: list[str],
    subtype_order: list[str],
    minimum_support: int,
) -> dict[str, Any]:
    support = len(rows)
    if support == 0:
        raise ValueError("profile summary requires observations")
    successes = sum(int(bool(row["correct"])) for row in rows)
    global_accuracy = _global(successes, support)
    class_accuracy: dict[str, float] = {}
    class_support: dict[str, int] = {}
    for name in class_order:
        selected = [row for row in rows if semantics[str(row["query_id"])]["primary_class"] == name]
        class_support[name] = len(selected)
        class_accuracy[name] = _conditional(
            sum(int(bool(row["correct"])) for row in selected),
            len(selected),
            global_accuracy,
            minimum_support,
        )
    subtype_accuracy: dict[str, float] = {}
    subtype_support: dict[str, int] = {}
    for name in subtype_order:
        selected = [row for row in rows if semantics[str(row["query_id"])].get("subtype") == name]
        subtype_support[name] = len(selected)
        subtype_accuracy[name] = _conditional(
            sum(int(bool(row["correct"])) for row in selected),
            len(selected),
            global_accuracy,
            minimum_support,
        )
    return {
        "global_accuracy": global_accuracy,
        "global_support": support,
        "class_accuracy": class_accuracy,
        "class_support": class_support,
        "subtype_accuracy": subtype_accuracy,
        "subtype_support": subtype_support,
        "mean_generation_gpu_seconds": sum(float(row["generation_gpu_seconds"]) for row in rows)
        / support,
    }


def build_fold_profiles(
    scored_rows: Iterable[dict[str, Any]],
    semantics: dict[str, dict[str, Any]],
    *,
    candidate_routes: list[str],
    all_routes: list[str],
    class_order: list[str],
    subtype_order: list[str],
    minimum_support: int,
) -> dict[str, Any]:
    rows = list(scored_rows)
    if not rows or any(str(row.get("role")) != "profile" for row in rows):
        raise ValueError("candidate profiles may use only fold-local profile rows")
    if any(row.get("target_outcome") is True for row in rows):
        raise ValueError("target outcome reached profile construction")
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row["query_id"])
        if query_id not in semantics:
            raise ValueError(f"missing semantic row: {query_id}")
        by_route[str(row["route_id"])].append(row)
    if set(by_route) != set(all_routes):
        raise ValueError("profile route coverage differs from the frozen route pool")
    query_sets = [{str(row["query_id"]) for row in values} for values in by_route.values()]
    if len({frozenset(value) for value in query_sets}) != 1:
        raise ValueError("routes do not cover the same profile queries")
    per_route = {
        route: _profile_summary(
            by_route[route],
            semantics,
            class_order=class_order,
            subtype_order=subtype_order,
            minimum_support=minimum_support,
        )
        for route in all_routes
    }
    pooled_rows = [row for route in candidate_routes for row in by_route[route]]
    pooled = _profile_summary(
        pooled_rows,
        semantics,
        class_order=class_order,
        subtype_order=subtype_order,
        minimum_support=minimum_support,
    )
    return {
        "class_order": class_order,
        "subtype_order": subtype_order,
        "minimum_support": minimum_support,
        "candidate_routes": candidate_routes,
        "all_routes": all_routes,
        "per_route": per_route,
        "pooled_candidates": pooled,
        "target_outcomes_used": False,
    }


def _candidate_vector(summary: dict[str, Any], class_order: list[str], subtype_order: list[str]) -> list[float]:
    return (
        [float(summary["global_accuracy"]), math.log1p(int(summary["global_support"]))]
        + [float(summary["class_accuracy"][name]) for name in class_order]
        + [math.log1p(int(summary["class_support"][name])) for name in class_order]
        + [float(summary["subtype_accuracy"][name]) for name in subtype_order]
        + [math.log1p(int(summary["subtype_support"][name])) for name in subtype_order]
    )


def arm_candidate_vectors(profiles: dict[str, Any], arm: str) -> dict[str, list[float]]:
    class_order = list(profiles["class_order"])
    subtype_order = list(profiles["subtype_order"])
    output: dict[str, list[float]] = {}
    for route in profiles["candidate_routes"]:
        specific = profiles["per_route"][route]
        if arm == "H1":
            summary = profiles["pooled_candidates"]
        elif arm == "H1_5":
            global_accuracy = float(specific["global_accuracy"])
            global_support = int(specific["global_support"])
            summary = {
                "global_accuracy": global_accuracy,
                "global_support": global_support,
                "class_accuracy": {name: global_accuracy for name in class_order},
                "class_support": {name: global_support for name in class_order},
                "subtype_accuracy": {name: global_accuracy for name in subtype_order},
                "subtype_support": {name: global_support for name in subtype_order},
            }
        elif arm == "H2":
            summary = specific
        else:
            raise ValueError(f"unknown arm: {arm}")
        output[route] = _candidate_vector(summary, class_order, subtype_order)
    return output


def build_arm_feature_rows(
    *,
    query_ids: Iterable[str],
    semantics: dict[str, dict[str, Any]],
    profiles: dict[str, Any],
    arm: str,
) -> list[dict[str, Any]]:
    candidate_vectors = arm_candidate_vectors(profiles, arm)
    class_order = list(profiles["class_order"])
    subtype_order = list(profiles["subtype_order"])
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        query_id = str(query_id)
        query_vector = semantic_query_vector(
            semantics[query_id], class_order=class_order, subtype_order=subtype_order
        ).tolist()
        for route in profiles["candidate_routes"]:
            rows.append(
                {
                    "query_id": query_id,
                    "route_id": route,
                    "arm": arm,
                    "query_features": query_vector,
                    "candidate_features": candidate_vectors[route],
                }
            )
    return rows

