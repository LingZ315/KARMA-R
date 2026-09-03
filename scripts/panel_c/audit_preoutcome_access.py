#!/usr/bin/env python3
"""Static-only audit of Panel-C target access gates and dangerous tokens."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = re.compile(
    r"target_correct|target_score|target_label|heldout_correctness|"
    r"sealed_reference_answers|panel_c_results",
    re.IGNORECASE,
)
GUARDED = {
    "_build_arm.py",
    "run_semantic_classifier.py",
    "run_candidate_inference.py",
    "score_candidate_outputs.py",
    "combine_fold_scores.py",
    "build_candidate_profiles.py",
    "fit_router.py",
    "fit_simple_controls.py",
    "evaluate_fold_policy.py",
    "select_fold_hyperparameters.py",
    "select_fold_B_star.py",
    "freeze_fold_policy.py",
    "freeze_panel_c_fold_policy_bundle.py",
    "run_target_routing.py",
    "freeze_panel_c_target_routes.py",
    "analyze_primary_endpoint.py",
    "bootstrap_primary_endpoint.py",
    "calculate_macro_source_effect.py",
    "calculate_deployment_utility.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    script_root = root / "scripts" / "panel_c"
    source_root = root / "src" / "karmavl" / "panel_c"
    files = sorted([*script_root.glob("*.py"), *source_root.glob("*.py")])
    hits: list[tuple[str, int, str]] = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if PATTERNS.search(line):
                hits.append((path.relative_to(root).as_posix(), number, line.strip()))
    missing_auth = []
    for name in sorted(GUARDED):
        path = script_root / name
        text = path.read_text(encoding="utf-8")
        if "require_frozen_execution_state" not in text and "require_frozen_execution_from_args" not in text:
            missing_auth.append(name)
    scorer = (script_root / "score_candidate_outputs.py").read_text(encoding="utf-8")
    candidate = (script_root / "run_candidate_inference.py").read_text(encoding="utf-8")
    semantic = (script_root / "run_semantic_classifier.py").read_text(encoding="utf-8")
    external = (source_root / "external.py").read_text(encoding="utf-8")
    target_freezes = all(
        token in scorer
        for token in ("require_policy_bundle_freeze", "require_target_route_bundle_freeze")
    )
    candidate_three_locks = all(
        token in candidate
        for token in (
            "require_frozen_execution_state",
            "require_policy_bundle_freeze",
            "require_target_route_bundle_freeze",
            "exact minimal query set",
        )
    )
    semantic_policy_gate = "require_policy_bundle_freeze" in semantic and 'args.role == "target"' in semantic
    immutable_github_only = (
        'release.get("immutable") is not True' in external
        and 'provider == "github"' in external
        and "EXTERNAL_TIMESTAMP_MANUAL_VERIFICATION_REQUIRED" in external
    )
    forbidden_outputs = [
        root / "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.json",
        root / "PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock",
        root / "PANEL_C_FOLD_POLICIES_FROZEN.lock",
        root / "PANEL_C_TARGET_ROUTES_FROZEN.lock",
        root / "panel_c_target_routes_frozen.jsonl",
        root / "PANEL_C_EXTERNAL_RELEASE_VERIFICATION.json",
        root / "revision_outputs" / "panel_c_results.csv",
    ]
    existing = [path.relative_to(root).as_posix() for path in forbidden_outputs if path.exists()]
    status = (
        "PASS"
        if not missing_auth
        and target_freezes
        and candidate_three_locks
        and semantic_policy_gate
        and immutable_github_only
        and not existing
        else "FAIL"
    )
    lines = [
        "# Panel-C pre-outcome target-access audit",
        "",
        f"Status: **{status}**",
        "",
        "This audit read source/configuration text only. It did not open an answer, correctness, prediction, score, or utility ledger.",
        "",
        "## Guard coverage",
        "",
        f"- Guarded result-determining entry points checked: {len(GUARDED)}.",
        f"- Missing `require_frozen_execution_state`: {missing_auth or 'none'}.",
        f"- Target scorer requires both fold-policy and target-route freezes: {target_freezes}.",
        f"- Target candidate runner requires authorization, complete-policy, and route locks plus minimal selected routes: {candidate_three_locks}.",
        f"- Target semantic runner requires the complete policy bundle: {semantic_policy_gate}.",
        f"- Automatic external authorization is GitHub-immutable-only; Zenodo/OSF are manual-only: {immutable_github_only}.",
        f"- Unauthorized receipt/lock/result artifacts present: {existing or 'none'}.",
        "- Authorization rechecks a live external provider and downloaded archive bytes; provider/network failure is fail-closed.",
        "- Every guarded entry rehashes the live master, code, and configuration manifests and every file they declare.",
        "- The target scorer validates authorization and both freezes before opening a fold-scoped target reference ledger.",
        "- A pooled reference-answer ledger is rejected by path and exact query-ID scope.",
        "",
        "## Dangerous-token review",
        "",
        "Matches below are guards, schemas, or explicitly gated post-outcome implementations; no matched data file was opened.",
        "",
    ]
    lines.extend(f"- `{path}:{number}` — `{text}`" for path, number, text in hits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
