"""Method-specific, cost-complete Panel-C utility accounting."""

from __future__ import annotations

from typing import Any, Iterable


def total_gpu_seconds(*, onboarding_gpu_seconds: float, serving_gpu_seconds: float, queries: int) -> float:
    if onboarding_gpu_seconds < 0 or serving_gpu_seconds < 0 or queries < 1:
        raise ValueError("costs must be non-negative and queries positive")
    return float(onboarding_gpu_seconds) + int(queries) * float(serving_gpu_seconds)


def deployment_utility(
    *,
    accuracy: float,
    onboarding_gpu_seconds: float,
    serving_gpu_seconds: float,
    queries: int,
    cost_coefficient: float,
) -> float:
    """Historical normalized convention: accuracy minus amortized cost/query."""

    if not 0.0 <= accuracy <= 1.0 or cost_coefficient < 0:
        raise ValueError("accuracy must be in [0,1] and cost coefficient non-negative")
    total = total_gpu_seconds(
        onboarding_gpu_seconds=onboarding_gpu_seconds,
        serving_gpu_seconds=serving_gpu_seconds,
        queries=queries,
    )
    return float(accuracy) - float(cost_coefficient) * total / int(queries)


def method_serving_cost(
    records: Iterable[dict[str, Any]], *, requirements: dict[str, Any]
) -> float:
    """Calculate only components the method actually requires."""

    rows = list(records)
    if not rows:
        raise ValueError("serving cost requires records")
    needed = {
        "feature_gpu_seconds": bool(requirements["needs_semantic_classifier"]),
        "router_gpu_seconds": bool(requirements["needs_learned_router"]),
        "selected_vlm_gpu_seconds": True,
    }
    totals: list[float] = []
    for row in rows:
        value = 0.0
        for field, enabled in needed.items():
            component = float(row.get(field, 0.0))
            if component < 0:
                raise ValueError(f"negative cost component: {field}")
            if enabled:
                value += component
            elif component != 0.0:
                raise ValueError(f"method was charged an unused component: {field}")
        totals.append(value)
    return sum(totals) / len(totals)


def validate_method_requirements(methods: dict[str, Any]) -> None:
    required = {
        "incumbent_only",
        "cheapest_candidate",
        "static_calibration_global_best",
        "static_class_conditional_best",
        "logistic_raw",
        "nearest_profile",
        "H1",
        "H1_5",
        "H2",
    }
    if set(methods) != required:
        raise ValueError(f"method cost table mismatch: {sorted(set(methods) ^ required)}")
    for name, row in methods.items():
        if "needs_semantic_classifier" not in row or "needs_learned_router" not in row:
            raise ValueError(f"missing serving requirements for {name}")
    if not methods["H2"]["needs_semantic_classifier"]:
        raise ValueError("H2 must include per-query semantic feature-generation cost")
    for name in ("incumbent_only", "cheapest_candidate", "static_calibration_global_best"):
        if methods[name]["needs_semantic_classifier"]:
            raise ValueError(f"simple control is incorrectly charged semantic inference: {name}")

