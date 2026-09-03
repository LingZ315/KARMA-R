# Panel-C pre-outcome target-access audit

Status: **PASS**

This audit read source/configuration text only. It did not open an answer, correctness, prediction, score, or utility ledger.

## Guard coverage

- Guarded result-determining entry points checked: 19.
- Missing `require_frozen_execution_state`: none.
- Target scorer requires both fold-policy and target-route freezes: True.
- Target candidate runner requires authorization, complete-policy, and route locks plus minimal selected routes: True.
- Target semantic runner requires the complete policy bundle: True.
- Automatic external authorization is GitHub-immutable-only; Zenodo/OSF are manual-only: True.
- Unauthorized receipt/lock/result artifacts present: none.
- Authorization rechecks a live external provider and downloaded archive bytes; provider/network failure is fail-closed.
- Every guarded entry rehashes the live master, code, and configuration manifests and every file they declare.
- The target scorer validates authorization and both freezes before opening a fold-scoped target reference ledger.
- A pooled reference-answer ledger is rejected by path and exact query-ID scope.

## Dangerous-token review

Matches below are guards, schemas, or explicitly gated post-outcome implementations; no matched data file was opened.

- `scripts/panel_c/analyze_primary_endpoint.py:29` — `"Bundle schema: {folds:[{fold_id,semantics,target_score_files:[...]}]}"`
- `scripts/panel_c/analyze_primary_endpoint.py:84` — `for path_text in spec["target_score_files"]:`
- `scripts/panel_c/audit_preoutcome_access.py:12` — `r"target_correct|target_score|target_label|heldout_correctness|"`
- `scripts/panel_c/audit_preoutcome_access.py:13` — `r"sealed_reference_answers|panel_c_results",`
- `scripts/panel_c/audit_preoutcome_access.py:89` — `root / "revision_outputs" / "panel_c_results.csv",`
- `scripts/panel_c/build_cohort.py:173` — `answer_path = private_dir / "sealed_reference_answers.jsonl"`
- `scripts/panel_c/build_cohort.py:361` — `"sealed_reference_answers": sha256_file(answer_path),`
- `scripts/panel_c/build_panel_c_preregistration_v3.py:50` — `"panel_c_results.csv",`
- `scripts/panel_c/build_panel_c_preregistration_v3.py:51` — `"target_scores.jsonl",`
- `scripts/panel_c/build_panel_c_preregistration_v3.py:52` — `"target_correctness.jsonl",`
- `src/karmavl/panel_c/common.py:15` — `r"(^|/)(panel_c_results|target_correctness|target_scores?|target_scored|"`
- `src/karmavl/panel_c/common.py:16` — `r"scored_target|target_outcomes?|heldout_correctness|target_candidate_outputs?|"`
