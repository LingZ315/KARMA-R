#!/usr/bin/env python3
"""Build, mirror, hash, extract, and self-test the canonical Panel-C V3 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_TOP = "KARMA_R_PANEL_C_PREREGISTRATION_V3"
ARCHIVE_NAME = f"{ARCHIVE_TOP}.zip"
MASTER_NAME = "PANEL_C_PREREGISTRATION_MANIFEST_V3.json"
CODE_NAME = "PANEL_C_CODE_MANIFEST_V3.json"
CONFIGURATION_NAME = "PANEL_C_CONFIGURATION_MANIFEST_V3.json"
GENERATED_AT_UTC = "2026-09-02T00:00:00Z"

SOURCE_ROOT_FILES = {
    "ACTION_SPACE_AND_SELECTOR_CONTRACT.md",
    "ACTION_SPACE_AUDIT.md",
    "COST_ACCOUNTING_PLAN.md",
    "DATASET_AND_SPLIT_SPEC.md",
    "ENVIRONMENT_LOCK.txt",
    "GITHUB_IMMUTABLE_RELEASE_INSTRUCTIONS.md",
    "HYPERPARAMETER_PROVENANCE.md",
    "HYPOTHESES.md",
    "MODEL_POOL.md",
    "PANEL_C_EXTERNAL_PREREGISTRATION_REQUIRED.md",
    "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT_TEMPLATE.json",
    "PANEL_C_PROTOCOL.md",
    "PRIMARY_ENDPOINT.md",
    "README.md",
    "STATISTICAL_ANALYSIS_PLAN.md",
}
EXCLUDED_SCRIPT_NAMES = {"build_preregistration_archive.py", "build_v702_submission_package.py"}
FORBIDDEN_REAL_ARTIFACTS = {
    "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.json",
    "PANEL_C_EXTERNAL_RELEASE_VERIFICATION.json",
    "PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock",
    "PANEL_C_FOLD_POLICIES_FROZEN.lock",
    "PANEL_C_TARGET_ROUTES_FROZEN.lock",
    "panel_c_target_routes_frozen.jsonl",
    "panel_c_results.csv",
    "target_scores.jsonl",
    "target_correctness.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_tree_files(source: Path, destination: Path, *, exclude: set[str] | None = None) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    excluded = exclude or set()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in excluded or "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc" or ".pytest_cache" in path.parts:
            continue
        copy_file(path, destination / path.relative_to(source))


def run(root: Path, label: str, command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root / "src"), str(root), current) if value
    )
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}):\n{result.stdout}")
    return result.stdout


def validate_no_forbidden_paths(root: Path) -> None:
    found = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_REAL_ARTIFACTS
    )
    if found:
        raise PermissionError(f"forbidden target/external artifacts found by path only: {found}")


def assemble(outer: Path, stage: Path) -> None:
    source = outer / "panel_c_preregistration"
    for name in sorted(SOURCE_ROOT_FILES):
        copy_file(source / name, stage / name)
    copy_file(
        outer / "READY_FOR_EXTERNAL_TIMESTAMP_BEFORE_PANEL_C_TARGET.md",
        stage / "READY_FOR_EXTERNAL_TIMESTAMP_BEFORE_PANEL_C_TARGET.md",
    )
    copy_file(outer / "pyproject.toml", stage / "pyproject.toml")
    copy_file(outer / "requirements-lock.txt", stage / "requirements-lock.txt")
    copy_tree_files(outer / "configs/kbs_v7_0_2", stage / "configs")
    for path in sorted((source / "manifests").rglob("*")):
        if not path.is_file() or path.name in {
            "panel_c_preoutcome_access_audit.md",
            "panel_c_fold_isolation_audit.json",
        }:
            continue
        copy_file(path, stage / "manifests" / path.relative_to(source / "manifests"))
    copy_tree_files(source / "runtime", stage / "runtime")

    copy_file(outer / "src/karmavl/__init__.py", stage / "src/karmavl/__init__.py")
    copy_tree_files(outer / "src/karmavl/panel_c", stage / "src/karmavl/panel_c")
    copy_tree_files(outer / "src/karmavl/kbs_replication_b", stage / "src/karmavl/kbs_replication_b")
    copy_tree_files(
        outer / "scripts/panel_c",
        stage / "scripts/panel_c",
        exclude=EXCLUDED_SCRIPT_NAMES,
    )
    copy_file(
        outer / "verify_panel_c_target_chronology.py",
        stage / "scripts/panel_c/verify_panel_c_target_chronology.py",
    )
    copy_file(
        outer / "tests/kbs_v7_0_1/test_preoutcome_unittest.py",
        stage / "tests/test_preoutcome_unittest.py",
    )
    copy_file(
        outer / "tests/kbs_v7_0_2/test_v702_preoutcome_hardening.py",
        stage / "tests/test_v702_preoutcome_hardening.py",
    )
    copy_file(
        outer / "tests/kbs_v7_0_3/test_v703_timestamp_hardening.py",
        stage / "tests/test_v703_timestamp_hardening.py",
    )
    copy_tree_files(outer / "verification", stage / "verification")
    validate_no_forbidden_paths(stage)


def generate_pre_manifest_audits(stage: Path) -> None:
    run(
        stage,
        "nested LOSO verifier",
        [
            sys.executable,
            "scripts/panel_c/verify_panel_c_nested_loso.py",
            "--split-manifest",
            "manifests/panel_c_public_split_id_manifest.json",
            "--derive-source-from-target-roles",
            "--analysis-contract",
            "configs/panel_c_analysis_contract.json",
            "--output",
            "manifests/panel_c_fold_isolation_audit.json",
        ],
    )
    run(
        stage,
        "pre-outcome access audit",
        [
            sys.executable,
            "scripts/panel_c/audit_preoutcome_access.py",
            "--project-root",
            str(stage),
            "--output",
            "manifests/panel_c_preoutcome_access_audit.md",
        ],
    )


def generate_manifests(stage: Path) -> tuple[dict, dict, dict]:
    code_paths = sorted(
        path
        for prefix in ("src", "scripts", "verification")
        for path in (stage / prefix).rglob("*.py")
        if path.is_file()
    )
    configuration_paths = sorted(
        {
            *[path for path in (stage / "configs").rglob("*") if path.is_file()],
            *[path for path in (stage / "manifests").rglob("*") if path.is_file()],
            *[path for path in (stage / "runtime").rglob("*") if path.is_file()],
            *[
                stage / name
                for name in SOURCE_ROOT_FILES
                if name not in {"README.md", "GITHUB_IMMUTABLE_RELEASE_INSTRUCTIONS.md"}
            ],
        }
    )
    code = {
        "schema_version": 3,
        "artifact_role": "panel_c_result_determining_code_manifest",
        "protocol_version": "1.0.2",
        "files": [entry(stage, path) for path in code_paths],
    }
    configuration = {
        "schema_version": 3,
        "artifact_role": "panel_c_result_determining_configuration_manifest",
        "protocol_version": "1.0.2",
        "files": [entry(stage, path) for path in configuration_paths],
    }
    write_json(stage / CODE_NAME, code)
    write_json(stage / CONFIGURATION_NAME, configuration)
    master_paths = sorted(
        path
        for path in stage.rglob("*")
        if path.is_file()
        and path.name != MASTER_NAME
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
    )
    master = {
        "schema_version": 3,
        "artifact_role": "panel_c_preregistration_file_manifest",
        "protocol_version": "1.0.2",
        "archive_top_level": ARCHIVE_TOP,
        "generated_at_utc": GENERATED_AT_UTC,
        "preoutcome_only": True,
        "target_execution_authorized": False,
        "manifest_self_excluded_to_avoid_recursive_hashing": True,
        "files": [entry(stage, path) for path in master_paths],
    }
    write_json(stage / MASTER_NAME, master)
    return master, code, configuration


def sync_mirror(outer: Path, stage: Path, master: dict) -> None:
    target = outer / "panel_c_preregistration"
    if target.resolve().parent != outer.resolve() or target.name != "panel_c_preregistration":
        raise PermissionError("unsafe mirror target")
    replacement = outer / ".panel_c_preregistration_v3_replacement"
    if replacement.exists():
        shutil.rmtree(replacement)
    replacement.mkdir()
    names = [row["path"] for row in master["files"]] + [MASTER_NAME]
    for relative in names:
        copy_file(stage / relative, replacement / relative)
    if target.exists():
        shutil.rmtree(target)
    replacement.rename(target)


def write_archive(stage: Path, archive: Path, master: dict) -> None:
    if archive.exists():
        archive.unlink()
    names = sorted([row["path"] for row in master["files"]] + [MASTER_NAME])
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for relative in names:
            info = zipfile.ZipInfo(f"{ARCHIVE_TOP}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(
                info,
                (stage / relative).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def update_author_handoff(outer: Path, digest: str, file_count: int) -> None:
    content = f"""# Panel-C external timestamp handoff

Current status: scientific design frozen; **target execution unauthorized**.

Artifact: `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip`

SHA-256: `{digest}`

Master-manifest declarations: {file_count}.

Publish these exact bytes as a GitHub Immutable Release. Then verify `immutable=true`, remote SHA-256 equals the local SHA-256, and the release ID, tag, resolved commit, asset name, and publication time all match before creating the authorization lock.

Do not run Panel-C target inference, scoring, correctness analysis, utility analysis, bootstrap, or oracle analysis before authorization succeeds.
"""
    (outer / "READY_FOR_EXTERNAL_TIMESTAMP_BEFORE_PANEL_C_TARGET.md").write_text(
        content, encoding="utf-8"
    )
    next_step = f"""# Timestamp next step

## Current status

Scientific design frozen; target execution unauthorized. `NOT YET EXTERNALLY TIMESTAMPED`.

## Artifact to archive

`KARMA_R_PANEL_C_PREREGISTRATION_V3.zip`

## Local SHA-256

`{digest}`

## Required external mechanism

Publish the exact ZIP bytes as a GitHub Immutable Release or another explicitly verified immutable mechanism. Do not rebuild or rename the archive after publication.

## After publishing

Run remote verification and confirm:

```text
immutable=true
remote SHA256 == local SHA256
tag/commit/release IDs match
```

## Only afterward

Create the target-execution authorization lock from the real receipt and live verification. Do not run Panel C target work before then.
"""
    (outer / "README_TIMESTAMP_NEXT_STEP.md").write_text(next_step, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    outer = args.outer_root.resolve()
    validate_no_forbidden_paths(outer)
    outer_pytest = run(outer, "outer pytest", [sys.executable, "-m", "pytest", "-q"])
    with tempfile.TemporaryDirectory(prefix="karma_panel_c_v3_build_") as directory:
        stage = Path(directory) / ARCHIVE_TOP
        stage.mkdir()
        assemble(outer, stage)
        generate_pre_manifest_audits(stage)
        master, code, configuration = generate_manifests(stage)
        stage_selftest = run(
            stage,
            "staged V3 self-test",
            [sys.executable, "verification/run_timestamp_artifact_selftest.py"],
        )
        sync_mirror(outer, stage, master)
        archive = outer / ARCHIVE_NAME
        write_archive(stage, archive, master)
        sidecar = outer / f"{ARCHIVE_NAME}.sha256"
        digest = sha256(archive)
        sidecar.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii")
        with tempfile.TemporaryDirectory(prefix="karma_panel_c_v3_extract_") as extract_directory:
            extract_root = Path(extract_directory)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extract_root)
            clean_root = extract_root / ARCHIVE_TOP
            clean_selftest = run(
                clean_root,
                "clean-extraction V3 self-test",
                [sys.executable, "verification/run_timestamp_artifact_selftest.py"],
            )
        update_author_handoff(outer, digest, len(master["files"]))
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive": ARCHIVE_NAME,
                "archive_sha256": digest,
                "master_files": len(master["files"]),
                "code_files": len(code["files"]),
                "configuration_files": len(configuration["files"]),
                "outer_pytest_summary": outer_pytest.strip().splitlines()[-1],
                "staged_selftest": "TIMESTAMP ARTIFACT SELF-TEST: PASS" in stage_selftest,
                "clean_selftest": "TIMESTAMP ARTIFACT SELF-TEST: PASS" in clean_selftest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
