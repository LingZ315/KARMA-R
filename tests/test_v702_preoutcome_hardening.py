from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from karmavl.panel_c import external
from karmavl.panel_c.common import (
    outcome_named_paths,
    require_policy_bundle_freeze,
    require_target_route_bundle_freeze,
    sha256_file,
    verify_file_manifest,
)
from karmavl.panel_c.controls import fit_simple_controls
from karmavl.panel_c.nested import select_fold_b_star, select_fold_margin


def find_artifact_root(start: Path) -> Path:
    for candidate in (start.parent, *start.parents):
        if (candidate / "src/karmavl/panel_c/common.py").is_file() and (
            candidate / "scripts/panel_c"
        ).is_dir():
            return candidate
    raise RuntimeError("cannot resolve timestamp-artifact root")


ROOT = find_artifact_root(Path(__file__).resolve())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bind(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def github_receipt(archive: Path) -> dict:
    digest = sha256_file(archive)
    return {
        "provider": "github",
        "release_id": "12345",
        "persistent_url": "https://github.com/example/karma/releases/tag/v7.0.3",
        "repository": "example/karma",
        "tag": "v7.0.3",
        "commit_sha": "a" * 40,
        "published_at": "2026-08-31T00:00:00Z",
        "prerelease": False,
        "immutable": True,
        "asset_name": archive.name,
        "local_archive_sha256": digest,
        "remote_archive_sha256": digest,
        "verification_status": "PENDING_REMOTE_VERIFICATION",
        "target_execution_authorized": False,
    }


def github_release(archive: Path) -> dict:
    return {
        "id": 12345,
        "draft": False,
        "prerelease": False,
        "immutable": True,
        "published_at": "2026-08-31T00:00:00Z",
        "tag_name": "v7.0.3",
        "html_url": "https://github.com/example/karma/releases/tag/v7.0.3",
        "assets": [
            {
                "name": archive.name,
                "browser_download_url": "https://github.com/example/karma/releases/download/v7.0.3/archive.zip",
            }
        ],
    }


def verify_with_mocked_github(
    monkeypatch: pytest.MonkeyPatch,
    archive: Path,
    receipt_payload: dict,
    release_payload: dict,
    *,
    resolved_commit: str = "a" * 40,
    remote_hash: str | None = None,
) -> dict:
    receipt = archive.parent / "receipt.json"
    write_json(receipt, receipt_payload)
    monkeypatch.setattr(external, "_json", lambda _url: release_payload)
    monkeypatch.setattr(external, "_github_commit", lambda _repository, _tag: resolved_commit)
    monkeypatch.setattr(
        external,
        "_remote_sha256",
        lambda _url: (remote_hash or sha256_file(archive), archive.stat().st_size),
    )
    return external.verify_external_timestamp_receipt(receipt, archive)


def test_live_github_immutable_release_is_the_only_automatic_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "KARMA_R_PANEL_C_PREREGISTRATION_V3.zip"
    archive.write_bytes(b"frozen-v3")
    receipt = github_receipt(archive)
    result = verify_with_mocked_github(
        monkeypatch, archive, receipt, github_release(archive)
    )
    assert result["status"] == "VERIFIED_REMOTE_IMMUTABLE_ARCHIVE"
    assert result["immutable"] is True
    assert result["release_id"] == "12345"
    assert result["local_sha256"] == result["remote_sha256"]

    for provider in ("zenodo", "osf_registration"):
        manual = {**receipt, "provider": provider}
        with pytest.raises(PermissionError, match="EXTERNAL_TIMESTAMP_MANUAL_VERIFICATION_REQUIRED"):
            verify_with_mocked_github(monkeypatch, archive, manual, github_release(archive))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mutable", "mutable"),
        ("tag", "tag mismatch"),
        ("commit", "commit mismatch"),
        ("asset", "release asset"),
        ("hash", "remote archived bytes"),
    ],
)
def test_github_negative_cases_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    archive = tmp_path / "KARMA_R_PANEL_C_PREREGISTRATION_V3.zip"
    archive.write_bytes(b"frozen-v3")
    receipt = github_receipt(archive)
    release = github_release(archive)
    resolved = "a" * 40
    remote_hash = None
    if mutation == "mutable":
        release["immutable"] = False
    elif mutation == "tag":
        release["tag_name"] = "wrong-tag"
    elif mutation == "commit":
        resolved = "b" * 40
    elif mutation == "asset":
        release["assets"] = []
    elif mutation == "hash":
        remote_hash = "f" * 64
    with pytest.raises((PermissionError, FileNotFoundError), match=message):
        verify_with_mocked_github(
            monkeypatch,
            archive,
            receipt,
            release,
            resolved_commit=resolved,
            remote_hash=remote_hash,
        )


def test_no_receipt_code_drift_and_premature_target_output_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "KARMA_R_PANEL_C_PREREGISTRATION_V3.zip"
    archive.write_bytes(b"frozen")
    with pytest.raises(FileNotFoundError):
        external.verify_external_timestamp_receipt(tmp_path / "missing.json", archive)

    tracked = tmp_path / "tracked.py"
    tracked.write_text("before\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {"files": [{"path": "tracked.py", "bytes": tracked.stat().st_size, "sha256": sha256_file(tracked)}]},
    )
    tracked.write_text("after\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        verify_file_manifest(manifest, tmp_path)

    premature = tmp_path / "target_scores.jsonl"
    premature.write_text("opaque; test never parses this as a result\n", encoding="utf-8")
    assert outcome_named_paths(tmp_path)

    with pytest.raises(FileNotFoundError):
        require_policy_bundle_freeze(tmp_path / "PANEL_C_FOLD_POLICIES_FROZEN.lock")
    with pytest.raises(FileNotFoundError):
        require_target_route_bundle_freeze(tmp_path / "PANEL_C_TARGET_ROUTES_FROZEN.lock")


def policy_rows(kind: str) -> list[dict]:
    rows = []
    for margin in (0.0, 0.01):
        for index in range(4):
            if margin == 0.0:
                correct, cost = True, 10.0
            else:
                correct, cost = index < 3, 1.0
            rows.append(
                {
                    "fold_id": "fold_1",
                    "held_out_source": "source_a",
                    "role": "policy",
                    "source": "source_b",
                    "query_id": f"q{index}",
                    "margin": margin,
                    "correct": correct,
                    "serving_gpu_seconds": cost,
                    "realized_utility": float(correct) - 0.1 * cost,
                    "target_outcome": False,
                }
            )
    return rows


def test_gate3_and_gate4_h2_selectors_are_symmetric_and_distinct() -> None:
    rows = policy_rows("h2")
    accuracy = select_fold_margin(
        rows,
        fold_id="fold_1",
        held_out_source="source_a",
        margin_grid=(0.0, 0.01),
        fixed_margin_order=(0.0, 0.01),
        objective="accuracy",
        cost_coefficient=0.1,
    )
    utility = select_fold_margin(
        rows,
        fold_id="fold_1",
        held_out_source="source_a",
        margin_grid=(0.0, 0.01),
        fixed_margin_order=(0.0, 0.01),
        objective="utility",
        cost_coefficient=0.1,
    )
    assert accuracy["selected_margin"] == 0.0
    assert utility["selected_margin"] == 0.01
    assert accuracy["tie_break_order"][:2] == ["higher_accuracy", "higher_utility"]
    assert utility["tie_break_order"][:2] == ["higher_utility", "higher_accuracy"]


def test_gate3_and_gate4_simple_comparators_use_mirrored_rankings() -> None:
    rows = []
    for baseline in ("accurate_expensive", "useful_cheap"):
        for index in range(4):
            correct = baseline == "accurate_expensive" or index < 3
            cost = 10.0 if baseline == "accurate_expensive" else 1.0
            rows.append(
                {
                    "fold_id": "fold_1",
                    "role": "policy",
                    "source": "source_b",
                    "query_id": f"q{index}",
                    "baseline_id": baseline,
                    "correct": correct,
                    "serving_gpu_seconds": cost,
                    "target_outcome": False,
                }
            )
    order = ("accurate_expensive", "useful_cheap")
    accuracy = select_fold_b_star(
        rows,
        fold_id="fold_1",
        held_out_source="source_a",
        cost_coefficient=0.1,
        objective="accuracy",
        method_order=order,
    )
    utility = select_fold_b_star(
        rows,
        fold_id="fold_1",
        held_out_source="source_a",
        cost_coefficient=0.1,
        objective="utility",
        method_order=order,
    )
    assert accuracy["selected_B_f_star"] == "accurate_expensive"
    assert utility["selected_B_f_star"] == "useful_cheap"


def test_strong_simple_controls_share_h2_action_space_and_incumbent_is_eligible() -> None:
    semantics = {
        "q1": {"primary_class": "visual", "subtype": "none", "ambiguity": False},
        "q2": {"primary_class": "visual", "subtype": "none", "ambiguity": False},
    }
    scores = [
        {
            "query_id": query,
            "route_id": route,
            "correct": route == "incumbent",
            "generation_gpu_seconds": 1.0 if route == "incumbent" else 2.0,
            "target_outcome": False,
        }
        for query in semantics
        for route in ("incumbent", "candidate")
    ]
    profile = {
        route: {
            "global_accuracy": 0.9 if route == "incumbent" else 0.5,
            "mean_generation_gpu_seconds": 1.0 if route == "incumbent" else 2.0,
            "class_accuracy": {"visual": 0.9 if route == "incumbent" else 0.5},
            "class_support": {"visual": 100},
            "subtype_accuracy": {"none": 0.9 if route == "incumbent" else 0.5},
            "subtype_support": {"none": 100},
        }
        for route in ("incumbent", "candidate")
    }
    controls = fit_simple_controls(
        profile_rows=scores,
        calibration_rows=scores,
        semantics=semantics,
        profiles={"per_route": profile},
        incumbent_route="incumbent",
        candidate_routes=["candidate"],
        class_order=["visual"],
        subtype_order=["none"],
        minimum_support=1,
        cost_coefficient=0.01,
        logistic_l2=10.0,
        logistic_iterations=100,
    )
    expected = ["incumbent", "candidate"]
    assert controls["static_calibration_global_best"]["route_id"] == "incumbent"
    assert controls["logistic_raw"]["eligible_routes"] == expected
    assert controls["nearest_profile"]["eligible_routes"] == expected
    assert controls["matched_action_space"]["source_id_conditional_control"] is False


def build_chronology_fixture(tmp_path: Path) -> dict[str, list[Path] | Path]:
    common_files = {}
    for name in (
        "analysis_contract",
        "split_manifest",
        "code_manifest",
        "semantics",
        "profiles",
        "h1",
        "h15",
        "h2",
        "controls",
        "gate3",
        "gate4",
        "ba",
        "bu",
        "prereg",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_text(name, encoding="utf-8")
        common_files[name] = path

    fold_freezes = []
    bundle_folds = []
    required_bindings = {
        "analysis_contract": common_files["analysis_contract"],
        "split_manifest": common_files["split_manifest"],
        "code_manifest": common_files["code_manifest"],
        "semantic_features": common_files["semantics"],
        "candidate_profiles": common_files["profiles"],
        "h1_router": common_files["h1"],
        "h1_5_router": common_files["h15"],
        "h2_router": common_files["h2"],
        "simple_controls": common_files["controls"],
        "selected_h2_gate3_accuracy_policy": common_files["gate3"],
        "selected_h2_gate4_utility_policy": common_files["gate4"],
        "selected_B_A_star": common_files["ba"],
        "selected_B_U_star": common_files["bu"],
    }
    for index in range(5):
        fold_id = f"fold_{index}"
        path = tmp_path / f"{fold_id}.lock"
        payload = {
            "schema_version": 2,
            "artifact_role": "panel_c_fold_policy_freeze",
            "status": "FOLD_POLICY_FROZEN_PRE_TARGET_OUTCOME",
            "created_at_utc": "2026-01-01T00:00:00Z",
            "fold_id": fold_id,
            "held_out_source": f"source_{index}",
            "target_outcomes_accessed": False,
            "bindings": {name: bind(file) for name, file in required_bindings.items()},
        }
        write_json(path, payload)
        fold_freezes.append(path)
        bundle_folds.append(
            {
                "fold_id": fold_id,
                "held_out_source": f"source_{index}",
                "H1_policy": {},
                "H1_5_policy": {},
                "H2_gate3_accuracy_policy": {},
                "H2_gate4_utility_policy": {},
                "B_A_star_method": "a",
                "B_U_star_method": "u",
                "selected_margins": {},
                "selected_hyperparameters": {},
                "model_action_space_metadata": {},
                "fold_policy_freeze": str(path.resolve()),
                "fold_policy_freeze_sha256": sha256_file(path),
            }
        )
    bundle = tmp_path / "policy_bundle.json"
    write_json(
        bundle,
        {
            "artifact_role": "panel_c_complete_fold_policy_bundle",
            "status": "FROZEN_PRE_TARGET_INFERENCE",
            "target_outcomes_accessed": False,
            "folds": bundle_folds,
        },
    )
    policy_lock = tmp_path / "PANEL_C_FOLD_POLICIES_FROZEN.lock"
    write_json(
        policy_lock,
        {
            "artifact_role": "panel_c_fold_policy_bundle_freeze",
            "status": "ALL_FOLD_POLICIES_FROZEN_PRE_TARGET_INFERENCE",
            "created_at_utc": "2026-01-01T01:00:00Z",
            "target_outcomes_accessed": False,
            "bindings": {
                "policy_bundle": bind(bundle),
                "code_manifest": bind(common_files["code_manifest"]),
                "preregistration_archive": bind(common_files["prereg"]),
            },
        },
    )
    routes = tmp_path / "panel_c_target_routes_frozen.jsonl"
    route_rows = [
        {
            "fold_id": f"fold_{index}",
            "query_id": f"q{index}",
            "h2_gate3_accuracy_selected_route": "route",
            "b_A_star_selected_route": "route",
            "h2_gate4_utility_selected_route": "route",
            "b_U_star_selected_route": "route",
            "target_outcomes_accessed": False,
        }
        for index in range(5)
    ]
    routes.write_text("".join(json.dumps(row) + "\n" for row in route_rows), encoding="utf-8")
    route_lock = tmp_path / "PANEL_C_TARGET_ROUTES_FROZEN.lock"
    write_json(
        route_lock,
        {
            "artifact_role": "panel_c_target_route_bundle_freeze",
            "status": "ALL_TARGET_ROUTES_FROZEN_PRE_CANDIDATE_INFERENCE",
            "created_at_utc": "2026-01-01T03:00:00Z",
            "target_outcomes_accessed": False,
            "bindings": {
                "target_routes": bind(routes),
                "policy_bundle_lock": bind(policy_lock),
                "policy_bundle": bind(bundle),
                "preregistration_archive": bind(common_files["prereg"]),
                "code_manifest": bind(common_files["code_manifest"]),
            },
        },
    )
    semantic_receipts = []
    for index in range(5):
        output = tmp_path / f"semantic_{index}.jsonl"
        output.write_text("opaque semantics", encoding="utf-8")
        receipt = tmp_path / f"semantic_{index}.receipt.json"
        write_json(
            receipt,
            {
                "artifact_role": "panel_c_semantic_inference_execution_receipt",
                "status": "COMPLETED",
                "role": "target",
                "fold_id": f"fold_{index}",
                "started_at_utc": "2026-01-01T02:00:00Z",
                "completed_at_utc": "2026-01-01T02:30:00Z",
                "target_candidate_responses_generated": False,
                "bindings": {"output": bind(output), "policy_bundle_lock": bind(policy_lock)},
            },
        )
        semantic_receipts.append(receipt)
    candidate_output = tmp_path / "sealed_target_candidate_output.jsonl"
    candidate_output.write_text("opaque candidate responses", encoding="utf-8")
    candidate_receipt = tmp_path / "candidate.receipt.json"
    write_json(
        candidate_receipt,
        {
            "artifact_role": "panel_c_candidate_inference_execution_receipt",
            "status": "COMPLETED",
            "role": "target",
            "target_outcomes_accessed": False,
            "started_at_utc": "2026-01-01T04:00:00Z",
            "completed_at_utc": "2026-01-01T05:00:00Z",
            "bindings": {
                "output": bind(candidate_output),
                "policy_bundle_lock": bind(policy_lock),
                "target_routes": bind(routes),
                "target_route_freeze": bind(route_lock),
            },
        },
    )
    score_output = tmp_path / "target_scored.jsonl"
    score_output.write_text("opaque scores", encoding="utf-8")
    scoring_receipt = tmp_path / "score.receipt.json"
    write_json(
        scoring_receipt,
        {
            "artifact_role": "panel_c_target_scoring_execution_receipt",
            "status": "COMPLETED",
            "started_at_utc": "2026-01-01T06:00:00Z",
            "completed_at_utc": "2026-01-01T07:00:00Z",
            "bindings": {"predictions": bind(candidate_output), "output": bind(score_output)},
        },
    )
    summary = tmp_path / "primary_summary.json"
    summary.write_text("opaque summary", encoding="utf-8")
    analysis_receipt = tmp_path / "analysis.receipt.json"
    write_json(
        analysis_receipt,
        {
            "artifact_role": "panel_c_primary_analysis_execution_receipt",
            "status": "COMPLETED",
            "started_at_utc": "2026-01-01T08:00:00Z",
            "completed_at_utc": "2026-01-01T09:00:00Z",
            "bindings": {"primary_summary": bind(summary)},
        },
    )
    return {
        "bundle": bundle,
        "policy_lock": policy_lock,
        "routes": routes,
        "route_lock": route_lock,
        "semantic_receipts": semantic_receipts,
        "candidate_receipt": candidate_receipt,
        "scoring_receipt": scoring_receipt,
        "analysis_receipt": analysis_receipt,
    }


def run_chronology(monkeypatch: pytest.MonkeyPatch, fixture: dict, output: Path) -> None:
    script = ROOT / "scripts/panel_c/verify_panel_c_target_chronology.py"
    if not script.is_file():
        script = ROOT / "verify_panel_c_target_chronology.py"
    spec = importlib.util.spec_from_file_location("v702_chronology", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        str(script),
        "--policy-bundle",
        str(fixture["bundle"]),
        "--policy-bundle-lock",
        str(fixture["policy_lock"]),
        "--target-routes",
        str(fixture["routes"]),
        "--target-route-freeze",
        str(fixture["route_lock"]),
    ]
    for path in fixture["semantic_receipts"]:
        argv.extend(["--target-semantic-receipt", str(path)])
    argv.extend(
        [
            "--target-inference-receipt",
            str(fixture["candidate_receipt"]),
            "--target-scoring-receipt",
            str(fixture["scoring_receipt"]),
            "--analysis-receipt",
            str(fixture["analysis_receipt"]),
            "--output",
            str(output),
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)
    module.main()


def test_hash_receipt_chronology_passes_and_predating_candidate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_chronology_fixture(tmp_path)
    output = tmp_path / "chronology.json"
    run_chronology(monkeypatch, fixture, output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"

    receipt_path = fixture["candidate_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["started_at_utc"] = "2026-01-01T02:45:00Z"
    receipt["completed_at_utc"] = "2026-01-01T02:50:00Z"
    write_json(receipt_path, receipt)
    with pytest.raises(PermissionError, match="predates policy or route freeze"):
        run_chronology(monkeypatch, fixture, tmp_path / "chronology-fail.json")


def test_target_candidate_guard_contains_all_three_hard_locks_and_minimal_routes() -> None:
    text = (ROOT / "scripts/panel_c/run_candidate_inference.py").read_text(encoding="utf-8")
    assert "require_frozen_execution_state" in text
    assert "require_policy_bundle_freeze" in text
    assert "require_target_route_bundle_freeze" in text
    assert "exact minimal query set" in text
    assert "h2_gate3_accuracy_selected_route" in text
    assert "h2_gate4_utility_selected_route" in text


def test_no_real_target_or_external_authorization_artifacts_are_packaged() -> None:
    forbidden = [
        ROOT / "panel_c_target_routes_frozen.jsonl",
        ROOT / "PANEL_C_TARGET_ROUTES_FROZEN.lock",
        ROOT / "PANEL_C_FOLD_POLICIES_FROZEN.lock",
        ROOT / "PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock",
        ROOT / "PANEL_C_EXTERNAL_RELEASE_VERIFICATION.json",
        ROOT / "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.json",
        ROOT / "revision_outputs" / "panel_c_results.csv",
    ]
    assert not [path for path in forbidden if path.exists()]
    template = json.loads((ROOT / "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT_TEMPLATE.json").read_text())
    assert template["target_execution_authorized"] is False
    assert template["verification_status"] == "TEMPLATE_NOT_A_RECEIPT"
