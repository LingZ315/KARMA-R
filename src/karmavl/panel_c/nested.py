"""Outer-fold isolation and fold-local control selection for Panel C."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


ROLES = ("profile", "calibration", "policy", "target")
TRAINING_ROLES = ("profile", "calibration", "policy")


def _sets_for_fold(fold: dict[str, Any]) -> dict[str, set[str]]:
    return {role: {str(value) for value in fold["roles"][role]["query_ids"]} for role in ROLES}


def verify_nested_loso(
    *,
    split_manifest: dict[str, Any],
    source_by_query: dict[str, str],
    analysis_contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify actual source membership and the declared dependency boundary.

    The function reads identifiers and source membership only.  It accepts no
    prediction, answer, correctness, score, or utility ledger.
    """

    selection = analysis_contract.get("outer_fold_selection", {})
    dependency_contract_ok = (
        selection.get("B_star_scope") == "fold_local_policy_only"
        and selection.get("h2_hyperparameter_scope") == "fold_local_non_target_only"
        and selection.get("global_cross_fold_selection_for_primary") is False
    )
    folds_out: list[dict[str, Any]] = []
    all_target_ids: set[str] = set()
    role_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    failures: list[str] = []

    for fold in split_manifest["folds"]:
        fold_id = str(fold["fold_id"])
        held = str(fold["held_out_source"])
        role_sets = _sets_for_fold(fold)
        for left_index, left in enumerate(ROLES):
            for right in ROLES[left_index + 1 :]:
                if role_sets[left] & role_sets[right]:
                    failures.append(f"{fold_id}: {left}/{right} overlap")
        for role, query_ids in role_sets.items():
            missing = sorted(query_ids - set(source_by_query))
            if missing:
                failures.append(f"{fold_id}/{role}: {len(missing)} IDs absent from source map")
            for query_id in query_ids:
                role_occurrences[query_id].append((fold_id, role))
        target_sources = sorted({source_by_query[q] for q in role_sets["target"] if q in source_by_query})
        training_sources = {
            role: sorted({source_by_query[q] for q in role_sets[role] if q in source_by_query})
            for role in TRAINING_ROLES
        }
        target_ok = target_sources == [held]
        training_ok = all(held not in training_sources[role] for role in TRAINING_ROLES)
        unique_target_ok = all_target_ids.isdisjoint(role_sets["target"])
        all_target_ids.update(role_sets["target"])
        if not target_ok:
            failures.append(f"{fold_id}: target source mismatch")
        if not training_ok:
            failures.append(f"{fold_id}: held-out source present in a training role")
        if not unique_target_ok:
            failures.append(f"{fold_id}: target query appears in more than one outer fold")
        folds_out.append(
            {
                "fold": fold_id,
                "held_out_source": held,
                "profile_sources": training_sources["profile"],
                "calibration_sources": training_sources["calibration"],
                "policy_sources": training_sources["policy"],
                "target_sources": target_sources,
                "B_star_selected_with_heldout_source": not (training_ok and dependency_contract_ok),
                "h2_hyperparameters_selected_with_heldout_source": not (
                    training_ok and dependency_contract_ok
                ),
                "target_outcomes_accessed": False,
                "role_counts": {role: len(role_sets[role]) for role in ROLES},
            }
        )

    selected_total = int(analysis_contract["cohort"]["pooled_target_n"])
    if len(all_target_ids) != selected_total:
        failures.append(f"unique outer targets={len(all_target_ids)}, expected={selected_total}")
    reused = {
        query_id: occurrences
        for query_id, occurrences in role_occurrences.items()
        if any(role == "target" for _fold, role in occurrences) and len(occurrences) > 1
    }
    if not dependency_contract_ok:
        failures.append("analysis contract does not prohibit global cross-fold primary selection")

    graph_edges: list[dict[str, str]] = []
    for row in folds_out:
        prefix = row["fold"]
        graph_edges.extend(
            [
                {"from": f"{prefix}/profile", "to": f"{prefix}/candidate_profiles"},
                {"from": f"{prefix}/calibration", "to": f"{prefix}/router_and_controls"},
                {"from": f"{prefix}/policy", "to": f"{prefix}/margin"},
                {"from": f"{prefix}/policy", "to": f"{prefix}/B_f_star"},
                {"from": f"{prefix}/router_and_controls", "to": f"{prefix}/target_predictions"},
                {"from": f"{prefix}/B_f_star", "to": f"{prefix}/target_predictions"},
            ]
        )
    return {
        "schema_version": 1,
        "artifact_role": "panel_c_nested_loso_fold_isolation_audit",
        "status": "PASS" if not failures else "FAIL",
        "target_outcomes_accessed": False,
        "cross_fold_reuse": {
            "acknowledged": True,
            "queries_target_once_and_reused_in_other_non_target_roles": len(reused),
            "permitted_only_with_fold_local_decisions": True,
        },
        "dependency_graph": {
            "edges": graph_edges,
            "forbidden_edges": [
                "heldout_source_or_target -> B_f_star",
                "heldout_source_or_target -> H2_hyperparameters",
                "other_outer_fold_policy_pool -> primary_B_f_star",
            ],
        },
        "folds": folds_out,
        "failures": failures,
    }


def _validate_policy_rows(
    rows: Iterable[dict[str, Any]], *, fold_id: str, held_out_source: str
) -> list[dict[str, Any]]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("fold-local policy selection requires rows")
    for row in materialized:
        if str(row.get("fold_id")) != fold_id or str(row.get("role")) != "policy":
            raise ValueError("policy selector received a row outside its fold-local policy split")
        if str(row.get("source")) == held_out_source:
            raise ValueError("held-out source reached a fold-local selector")
        if row.get("target_outcome") is True:
            raise ValueError("target outcome reached a fold-local selector")
    return materialized


def select_fold_b_star(
    rows: Iterable[dict[str, Any]],
    *,
    fold_id: str,
    held_out_source: str,
    cost_coefficient: float,
    objective: str = "accuracy",
    method_order: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Choose a fold-local simple comparator under a frozen gate objective."""

    if objective not in {"accuracy", "utility"}:
        raise ValueError("simple-comparator objective must be accuracy or utility")

    materialized = _validate_policy_rows(rows, fold_id=fold_id, held_out_source=held_out_source)
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_baseline[str(row["baseline_id"])].append(row)
    coverage = {name: {str(row["query_id"]) for row in values} for name, values in by_baseline.items()}
    if not coverage or len({frozenset(ids) for ids in coverage.values()}) != 1:
        raise ValueError("every simple control must cover the same fold-local policy queries")
    summary: dict[str, dict[str, float | int]] = {}
    for name, values in by_baseline.items():
        accuracy = sum(float(bool(row["correct"])) for row in values) / len(values)
        mean_cost = sum(float(row["serving_gpu_seconds"]) for row in values) / len(values)
        summary[name] = {
            "accuracy": accuracy,
            "utility": accuracy - cost_coefficient * mean_cost,
            "mean_serving_gpu_seconds": mean_cost,
            "n": len(values),
        }
    fixed_order = list(method_order) if method_order is not None else sorted(summary)
    if set(fixed_order) != set(summary) or len(fixed_order) != len(set(fixed_order)):
        raise ValueError("fixed method order must contain every simple control exactly once")
    rank = {name: index for index, name in enumerate(fixed_order)}
    if objective == "accuracy":
        key = lambda name: (  # noqa: E731 - visible preregistered ranking tuple
            -float(summary[name]["accuracy"]),
            -float(summary[name]["utility"]),
            float(summary[name]["mean_serving_gpu_seconds"]),
            rank[name],
        )
        tie_break = ["higher_accuracy", "higher_utility", "lower_serving_gpu_cost", "fixed_method_order"]
    else:
        key = lambda name: (  # noqa: E731 - visible preregistered ranking tuple
            -float(summary[name]["utility"]),
            -float(summary[name]["accuracy"]),
            float(summary[name]["mean_serving_gpu_seconds"]),
            rank[name],
        )
        tie_break = ["higher_utility", "higher_accuracy", "lower_serving_gpu_cost", "fixed_method_order"]
    selected = min(summary, key=key)
    return {
        "fold_id": fold_id,
        "held_out_source": held_out_source,
        "selection_role": "policy",
        "selected_B_f_star": selected,
        "selection_objective": objective,
        "tie_break_order": tie_break,
        "fixed_method_order": fixed_order,
        "ranking": summary,
        "held_out_source_used": False,
        "target_outcomes_used": False,
    }


def select_fold_margin(
    rows: Iterable[dict[str, Any]],
    *,
    fold_id: str,
    held_out_source: str,
    margin_grid: Iterable[float],
    objective: str = "utility",
    cost_coefficient: float | None = None,
    fixed_margin_order: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Select a fold-local H2 margin under an accuracy or utility objective."""

    if objective not in {"accuracy", "utility"}:
        raise ValueError("H2 policy objective must be accuracy or utility")

    materialized = _validate_policy_rows(rows, fold_id=fold_id, held_out_source=held_out_source)
    grid_order = [float(value) for value in margin_grid]
    legacy_utility_only = cost_coefficient is None and all(
        "correct" not in row and "serving_gpu_seconds" not in row for row in materialized
    )
    fixed_order = (
        [float(value) for value in fixed_margin_order]
        if fixed_margin_order is not None
        else (list(reversed(grid_order)) if legacy_utility_only else grid_order)
    )
    if set(fixed_order) != set(grid_order):
        raise ValueError("fixed margin order must contain every frozen margin exactly once")
    if len(fixed_order) != len(set(fixed_order)):
        raise ValueError("frozen margin order contains duplicates")
    allowed = set(fixed_order)
    by_margin: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        margin = float(row["margin"])
        if margin not in allowed:
            raise ValueError(f"margin outside frozen grid: {margin}")
        by_margin[margin].append(row)
    if set(by_margin) != allowed:
        raise ValueError("not every frozen margin was evaluated on fold-local policy data")
    coverage = {
        margin: {str(row["query_id"]) for row in values} for margin, values in by_margin.items()
    }
    if not legacy_utility_only and len({frozenset(ids) for ids in coverage.values()}) != 1:
        raise ValueError("every H2 margin must cover the same fold-local policy queries")
    summary: dict[float, dict[str, float | int]] = {}
    for margin, values in by_margin.items():
        accuracy = (
            sum(float(bool(row["correct"])) for row in values) / len(values)
            if not legacy_utility_only
            else 0.0
        )
        mean_cost = (
            sum(float(row["serving_gpu_seconds"]) for row in values) / len(values)
            if not legacy_utility_only
            else 0.0
        )
        if cost_coefficient is None or legacy_utility_only:
            utility = sum(float(row["realized_utility"]) for row in values) / len(values)
        else:
            utility = accuracy - cost_coefficient * mean_cost
        summary[margin] = {
            "accuracy": accuracy,
            "utility": utility,
            "mean_serving_gpu_seconds": mean_cost,
            "n": len(values),
        }
    rank = {margin: index for index, margin in enumerate(fixed_order)}
    if objective == "accuracy":
        key = lambda margin: (  # noqa: E731
            -float(summary[margin]["accuracy"]),
            -float(summary[margin]["utility"]),
            float(summary[margin]["mean_serving_gpu_seconds"]),
            rank[margin],
        )
        tie_break = ["higher_accuracy", "higher_utility", "lower_serving_gpu_cost", "fixed_margin_order"]
    else:
        key = lambda margin: (  # noqa: E731
            -float(summary[margin]["utility"]),
            -float(summary[margin]["accuracy"]),
            float(summary[margin]["mean_serving_gpu_seconds"]),
            rank[margin],
        )
        tie_break = ["higher_utility", "higher_accuracy", "lower_serving_gpu_cost", "fixed_margin_order"]
    selected = min(fixed_order, key=key)
    return {
        "fold_id": fold_id,
        "held_out_source": held_out_source,
        "selection_role": "policy",
        "selected_margin": selected,
        "selection_objective": objective,
        "tie_break_order": tie_break,
        "fixed_margin_order": fixed_order,
        "ranking": {str(key): value for key, value in sorted(summary.items())},
        "mean_policy_utility": {
            str(key): float(value["utility"]) for key, value in sorted(summary.items())
        },
        "held_out_source_used": False,
        "target_outcomes_used": False,
    }
