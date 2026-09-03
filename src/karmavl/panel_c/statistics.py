"""Frozen Panel-C primary endpoint, macro-source, bootstrap, and utility summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .costs import deployment_utility


def _validated(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    if not values:
        raise ValueError("primary analysis requires paired target rows")
    ids = [str(row["query_id"]) for row in values]
    if len(ids) != len(set(ids)):
        raise ValueError("each target query must appear exactly once")
    required = {"source", "h2_correct", "b_f_star_correct"}
    if any(not required <= set(row) for row in values):
        raise ValueError("paired target row is incomplete")
    return values


def primary_endpoint(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = _validated(rows)
    by_source: dict[str, list[float]] = defaultdict(list)
    differences = []
    for row in values:
        difference = float(bool(row["h2_correct"])) - float(bool(row["b_f_star_correct"]))
        differences.append(difference)
        by_source[str(row["source"])].append(difference)
    source_effects = {source: float(np.mean(items)) for source, items in sorted(by_source.items())}
    return {
        "pooled_query_weighted_effect": float(np.mean(differences)),
        "macro_source_effect": float(np.mean(list(source_effects.values()))),
        "median_source_effect": float(np.median(list(source_effects.values()))),
        "positive_source_count": sum(value > 0 for value in source_effects.values()),
        "negative_source_count": sum(value < 0 for value in source_effects.values()),
        "zero_source_count": sum(value == 0 for value in source_effects.values()),
        "source_effects": source_effects,
        "target_n": len(values),
    }


def source_stratified_paired_bootstrap(
    rows: Iterable[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    values = _validated(rows)
    if replicates < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    by_source: dict[str, np.ndarray] = defaultdict(list)  # type: ignore[assignment]
    temp: dict[str, list[float]] = defaultdict(list)
    for row in values:
        temp[str(row["source"])].append(
            float(bool(row["h2_correct"])) - float(bool(row["b_f_star_correct"]))
        )
    by_source = {source: np.asarray(items, dtype=np.float64) for source, items in temp.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    pooled = np.empty(replicates, dtype=np.float64)
    macro = np.empty(replicates, dtype=np.float64)
    sources = sorted(by_source)
    source_draws = {source: np.empty(replicates, dtype=np.float64) for source in sources}
    total_n = sum(len(by_source[source]) for source in sources)
    for draw in range(replicates):
        source_means = []
        pooled_sum = 0.0
        for source in sources:
            values_source = by_source[source]
            sampled = values_source[rng.integers(0, len(values_source), size=len(values_source))]
            source_means.append(float(sampled.mean()))
            source_draws[source][draw] = float(sampled.mean())
            pooled_sum += float(sampled.sum())
        pooled[draw] = pooled_sum / total_n
        macro[draw] = float(np.mean(source_means))
    alpha = (1.0 - confidence) / 2.0
    pooled_ci = np.quantile(pooled, [alpha, 1.0 - alpha])
    macro_ci = np.quantile(macro, [alpha, 1.0 - alpha])
    source_ci = {
        source: [
            float(np.quantile(source_draws[source], alpha)),
            float(np.quantile(source_draws[source], 1.0 - alpha)),
        ]
        for source in sources
    }
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "pooled_ci": [float(pooled_ci[0]), float(pooled_ci[1])],
        "macro_ci": [float(macro_ci[0]), float(macro_ci[1])],
        "source_specific_descriptive_ci": source_ci,
        "probability_h2_gt_b_f_star": float(np.mean(pooled > 0)),
    }


def utility_summary(
    rows: Iterable[dict[str, Any]],
    *,
    onboarding_h2: float,
    onboarding_b_f_star: float,
    horizons: Iterable[int],
    cost_coefficient: float,
) -> dict[str, Any]:
    values = _validated(rows)
    h2_accuracy = float(np.mean([float(bool(row["h2_correct"])) for row in values]))
    baseline_accuracy = float(
        np.mean([float(bool(row["b_f_star_correct"])) for row in values])
    )
    h2_serving = float(np.mean([float(row["h2_serving_gpu_seconds"]) for row in values]))
    baseline_serving = float(
        np.mean([float(row["b_f_star_serving_gpu_seconds"]) for row in values])
    )
    output = {}
    for horizon in horizons:
        output[str(int(horizon))] = {
            "H2": deployment_utility(
                accuracy=h2_accuracy,
                onboarding_gpu_seconds=onboarding_h2,
                serving_gpu_seconds=h2_serving,
                queries=int(horizon),
                cost_coefficient=cost_coefficient,
            ),
            "B_f_star": deployment_utility(
                accuracy=baseline_accuracy,
                onboarding_gpu_seconds=onboarding_b_f_star,
                serving_gpu_seconds=baseline_serving,
                queries=int(horizon),
                cost_coefficient=cost_coefficient,
            ),
        }
        output[str(int(horizon))]["difference"] = (
            output[str(int(horizon))]["H2"] - output[str(int(horizon))]["B_f_star"]
        )
    return {
        "units": "accuracy proportion minus 0.01 times amortized GPU-seconds/query",
        "horizons": output,
    }
