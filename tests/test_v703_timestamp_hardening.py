from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from karmavl.panel_c.common import (
    FROZEN_EXECUTION_REQUIRED_PATHS,
    sha256_file,
    verify_frozen_execution_files,
    verify_preregistration_archive,
)


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


def entry(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def frozen_fixture(root: Path) -> tuple[Path, Path, Path]:
    for relative in sorted(FROZEN_EXECUTION_REQUIRED_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "path": relative,
            "semantic_model_revision": "a" * 40,
            "candidate_model_revision": "b" * 40,
            "dtype": "bfloat16",
            "device_map": "auto",
            "schema_version": 1,
            "margin_grid": [0.0, 0.01, 0.02],
            "tie_break_order": ["higher_accuracy", "higher_utility", "lower_margin"],
            "selector": "accuracy_or_utility_as_preregistered",
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    code_manifest = root / "PANEL_C_CODE_MANIFEST_V3.json"
    configuration_manifest = root / "PANEL_C_CONFIGURATION_MANIFEST_V3.json"
    code_paths = sorted(
        relative
        for relative in FROZEN_EXECUTION_REQUIRED_PATHS
        if relative.startswith(("src/", "scripts/"))
    )
    configuration_paths = sorted(FROZEN_EXECUTION_REQUIRED_PATHS - set(code_paths))
    write_json(code_manifest, {"files": [entry(root, relative) for relative in code_paths]})
    write_json(
        configuration_manifest,
        {"files": [entry(root, relative) for relative in configuration_paths]},
    )
    master_manifest = root / "PANEL_C_PREREGISTRATION_MANIFEST_V3.json"
    master_paths = sorted(
        [*FROZEN_EXECUTION_REQUIRED_PATHS, code_manifest.name, configuration_manifest.name]
    )
    write_json(master_manifest, {"files": [entry(root, relative) for relative in master_paths]})
    return master_manifest, code_manifest, configuration_manifest


@pytest.mark.parametrize(
    ("category", "relative"),
    [
        ("protocol", "PANEL_C_PROTOCOL.md"),
        ("configuration", "configs/panel_c_analysis_contract.json"),
        ("semantic_prompt", "configs/panel_c_explicit_source_metadata_blind_query_semantics_v1.md"),
        ("qwen_semantic_model_revision", "configs/panel_c_metadata_free_semantic_classifier.json"),
        ("semantic_output_schema_definition", "configs/panel_c_metadata_free_semantic_classifier.json"),
        ("semantic_dtype", "configs/panel_c_metadata_free_semantic_classifier.json"),
        ("semantic_device_map", "configs/panel_c_metadata_free_semantic_classifier.json"),
        ("candidate_model_revision", "configs/panel_c_model_pool.json"),
        ("receipt_schema", "configs/PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.schema.json"),
        ("split", "manifests/panel_c_public_split_id_manifest.json"),
        ("source", "src/karmavl/panel_c/common.py"),
        ("learner", "src/karmavl/panel_c/routing.py"),
        ("scorer", "src/karmavl/panel_c/scoring.py"),
        ("statistical_plan", "STATISTICAL_ANALYSIS_PLAN.md"),
        ("cost_plan", "COST_ACCOUNTING_PLAN.md"),
        ("gate_margin_grid", "configs/panel_c_analysis_contract.json"),
        ("gate_tie_break_order", "configs/panel_c_analysis_contract.json"),
        ("gate_accuracy_selector", "configs/panel_c_analysis_contract.json"),
        ("gate_utility_selector", "configs/panel_c_analysis_contract.json"),
    ],
)
def test_each_scientific_drift_category_fails_closed(
    tmp_path: Path, category: str, relative: str
) -> None:
    master, code, configuration = frozen_fixture(tmp_path)
    verified = verify_frozen_execution_files(
        execution_root=tmp_path,
        preregistration_manifest=master,
        code_manifest=code,
        configuration_manifest=configuration,
    )
    assert verified["status"] == "VERIFIED_LIVE_LOCAL_FROZEN_EXECUTION_STATE"
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + f"DRIFT:{category}".encode())
    with pytest.raises((ValueError, PermissionError), match="manifest mismatch|does not"):
        verify_frozen_execution_files(
            execution_root=tmp_path,
            preregistration_manifest=master,
            code_manifest=code,
            configuration_manifest=configuration,
        )


def test_v3_archive_verifier_supports_arbitrary_extraction_parent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    member = source / "README.md"
    member.write_text("frozen\n", encoding="utf-8")
    manifest = source / "PANEL_C_PREREGISTRATION_MANIFEST_V3.json"
    write_json(
        manifest,
        {
            "archive_top_level": "KARMA_R_PANEL_C_PREREGISTRATION_V3",
            "files": [entry(source, "README.md")],
        },
    )
    archive = tmp_path / "KARMA_R_PANEL_C_PREREGISTRATION_V3.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(member, "KARMA_R_PANEL_C_PREREGISTRATION_V3/README.md")
        bundle.write(
            manifest,
            "KARMA_R_PANEL_C_PREREGISTRATION_V3/PANEL_C_PREREGISTRATION_MANIFEST_V3.json",
        )
    assert len(verify_preregistration_archive(archive, manifest_path=manifest)) == 1


def test_every_result_determining_entry_rehashes_live_frozen_files() -> None:
    guarded = {
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
    for name in guarded:
        text = (ROOT / "scripts/panel_c" / name).read_text(encoding="utf-8")
        assert "require_frozen_execution_state" in text or "require_frozen_execution_from_args" in text


def test_author_facing_readiness_name_is_canonical() -> None:
    assert (ROOT / "READY_FOR_EXTERNAL_TIMESTAMP_BEFORE_PANEL_C_TARGET.md").is_file()
    assert not (ROOT / "READY_FOR_EXTERNAL_TIMESTAMP_BEFORE_PANEL_C_TARGET.txt").exists()
