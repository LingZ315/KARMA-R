#!/usr/bin/env python3
"""Run every pre-outcome verification for an extracted V3 timestamp artifact."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def artifact_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "PANEL_C_PREREGISTRATION_MANIFEST_V3.json").is_file() and (
            candidate / "scripts/panel_c"
        ).is_dir():
            return candidate
    raise RuntimeError("cannot resolve extracted V3 timestamp-artifact root")


ROOT = artifact_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from karmavl.panel_c.common import (  # noqa: E402
    outcome_named_paths,
    verify_file_manifest,
    verify_frozen_execution_files,
)


def run(label: str, command: list[str]) -> str:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), str(ROOT), current) if value
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(f"[{label}]")
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    return completed.stdout


def validate_layout() -> None:
    required_directories = ("runtime", "configs", "manifests", "src/karmavl", "scripts/panel_c", "tests", "verification")
    required_files = (
        "README.md",
        "PANEL_C_PROTOCOL.md",
        "HYPOTHESES.md",
        "PRIMARY_ENDPOINT.md",
        "STATISTICAL_ANALYSIS_PLAN.md",
        "COST_ACCOUNTING_PLAN.md",
        "DATASET_AND_SPLIT_SPEC.md",
        "MODEL_POOL.md",
        "HYPERPARAMETER_PROVENANCE.md",
        "ENVIRONMENT_LOCK.txt",
        "PANEL_C_PREREGISTRATION_MANIFEST_V3.json",
        "PANEL_C_CODE_MANIFEST_V3.json",
        "PANEL_C_CONFIGURATION_MANIFEST_V3.json",
        "runtime/qwen32b_execution_smoke_receipt.json",
        "runtime/qwen32b_execution_smoke_test.md",
        "runtime/qwen32b_execution_smoke_stdout.log",
    )
    missing = [name for name in required_directories if not (ROOT / name).is_dir()]
    missing.extend(name for name in required_files if not (ROOT / name).is_file())
    if missing:
        raise FileNotFoundError(f"canonical V3 layout is incomplete: {missing}")
    forbidden = outcome_named_paths(ROOT)
    if forbidden:
        raise PermissionError(
            "target-result-like artifacts are forbidden: "
            + json.dumps(forbidden, sort_keys=True)
        )
    print("[layout] PASS")


def verify_manifests() -> None:
    master = ROOT / "PANEL_C_PREREGISTRATION_MANIFEST_V3.json"
    code = ROOT / "PANEL_C_CODE_MANIFEST_V3.json"
    configuration = ROOT / "PANEL_C_CONFIGURATION_MANIFEST_V3.json"
    rows = verify_file_manifest(master, ROOT)
    verify_file_manifest(code, ROOT)
    verify_file_manifest(configuration, ROOT)
    live = verify_frozen_execution_files(
        execution_root=ROOT,
        preregistration_manifest=master,
        code_manifest=code,
        configuration_manifest=configuration,
    )
    print(f"[manifests] PASS ({len(rows)} master declarations; {live['code_files']} code; {live['configuration_files']} configuration)")


def verify_chronology_static() -> None:
    text = (ROOT / "scripts/panel_c/verify_panel_c_target_chronology.py").read_text(
        encoding="utf-8"
    )
    required = (
        "require_policy_bundle_freeze",
        "require_target_route_bundle_freeze",
        "target-semantic-receipt",
        "target-inference-receipt",
        "target-scoring-receipt",
        "analysis-receipt",
        "predates policy or route freeze",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"chronology verifier is incomplete: {missing}")
    print("[target chronology static] PASS")


def main() -> None:
    validate_layout()
    verify_manifests()
    with tempfile.TemporaryDirectory(prefix="karma_v3_selftest_") as directory:
        temporary = Path(directory)
        run(
            "nested LOSO",
            [
                sys.executable,
                "scripts/panel_c/verify_panel_c_nested_loso.py",
                "--split-manifest",
                "manifests/panel_c_public_split_id_manifest.json",
                "--derive-source-from-target-roles",
                "--analysis-contract",
                "configs/panel_c_analysis_contract.json",
                "--output",
                str(temporary / "nested_loso.json"),
            ],
        )
        run(
            "pre-outcome access audit",
            [
                sys.executable,
                "scripts/panel_c/audit_preoutcome_access.py",
                "--project-root",
                str(ROOT),
                "--output",
                str(temporary / "preoutcome_access.md"),
            ],
        )
    verify_chronology_static()
    run(
        "action-space tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_v702_preoutcome_hardening.py::test_strong_simple_controls_share_h2_action_space_and_incumbent_is_eligible",
        ],
    )
    run(
        "Gate-3/Gate-4 selector tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_v702_preoutcome_hardening.py::test_gate3_and_gate4_h2_selectors_are_symmetric_and_distinct",
            "tests/test_v702_preoutcome_hardening.py::test_gate3_and_gate4_simple_comparators_use_mirrored_rankings",
        ],
    )
    run("pytest", [sys.executable, "-m", "pytest", "-q"])
    print("TIMESTAMP ARTIFACT SELF-TEST: PASS")


if __name__ == "__main__":
    main()
