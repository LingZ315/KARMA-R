#!/usr/bin/env python3
"""Fail-closed authorization guard for Panel-C target execution."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from karmavl.panel_c.common import (
    outcome_named_paths,
    sha256_file,
    verify_file_manifest,
    verify_preregistration_archive,
    write_json_new,
)
from karmavl.panel_c.external import verify_external_timestamp_receipt


def _quarantine(root: Path, rows: list[dict], destination: Path) -> list[dict]:
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise PermissionError("execution root cannot be a filesystem root")
    moved: list[dict] = []
    for row in rows:
        source = (resolved_root / row["path"]).resolve()
        if resolved_root not in source.parents:
            raise PermissionError("unsafe quarantine path")
        target = destination.resolve() / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({**row, "quarantined_to": str(target)})
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-root", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--configuration-manifest", type=Path, required=True)
    parser.add_argument("--environment-specification", type=Path, required=True)
    parser.add_argument("--external-receipt", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--quarantine-report", type=Path, required=True)
    parser.add_argument("--external-verification-output", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output_lock.exists() or args.external_verification_output.exists():
        raise FileExistsError("authorization outputs are immutable")
    prereg_verified = verify_file_manifest(args.preregistration_manifest, args.preregistration_root)
    verify_file_manifest(args.code_manifest, args.preregistration_root)
    verify_file_manifest(args.configuration_manifest, args.preregistration_root)
    relative_environment = args.environment_specification.resolve().relative_to(
        args.preregistration_root.resolve()
    ).as_posix()
    prereg_by_path = {row["path"]: row["sha256"] for row in prereg_verified}
    if prereg_by_path.get(relative_environment) != sha256_file(args.environment_specification):
        raise PermissionError("environment specification is not bound by the preregistration manifest")
    verify_preregistration_archive(
        args.preregistration_archive, manifest_path=args.preregistration_manifest
    )
    unexpected = outcome_named_paths(args.execution_root)
    if unexpected:
        moved = _quarantine(args.execution_root, unexpected, args.quarantine_root)
        write_json_new(
            args.quarantine_report,
            {
                "schema_version": 1,
                "artifact_role": "panel_c_unexpected_pre_authorization_artifact_quarantine",
                "status": "QUARANTINED_TARGET_EXECUTION_REMAINS_DISABLED",
                "files": moved,
            },
        )
        raise PermissionError("unexpected target-result-like artifacts were quarantined; authorization denied")
    remote = verify_external_timestamp_receipt(
        args.external_receipt, args.preregistration_archive
    )
    if remote.get("provider") != "github" or remote.get("immutable") is not True:
        raise PermissionError("automatic authorization requires a GitHub immutable release")
    write_json_new(args.external_verification_output, remote)
    receipt = json.loads(args.external_receipt.read_text(encoding="utf-8"))
    receipt_time = datetime.fromisoformat(str(receipt["published_at"]).replace("Z", "+00:00"))
    authorized_at = datetime.now(timezone.utc)
    if receipt_time.astimezone(timezone.utc) >= authorized_at:
        raise PermissionError("external timestamp must precede target authorization")
    lock = {
        "schema_version": 1,
        "artifact_role": "panel_c_target_execution_authorization",
        "status": "AUTHORIZED_REMOTE_TIMESTAMP_VERIFIED",
        "authorized_at_utc": authorized_at.isoformat().replace("+00:00", "Z"),
        "external_timestamp_utc": receipt_time.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "target_outcomes_present_before_authorization": False,
        "bindings": {
            "preregistration_archive": {
                "path": str(args.preregistration_archive.resolve()),
                "sha256": sha256_file(args.preregistration_archive),
            },
            "preregistration_manifest": {
                "path": str(args.preregistration_manifest.resolve()),
                "sha256": sha256_file(args.preregistration_manifest),
            },
            "code_manifest": {
                "path": str(args.code_manifest.resolve()),
                "sha256": sha256_file(args.code_manifest),
            },
            "configuration_manifest": {
                "path": str(args.configuration_manifest.resolve()),
                "sha256": sha256_file(args.configuration_manifest),
            },
            "environment_specification": {
                "path": str(args.environment_specification.resolve()),
                "sha256": sha256_file(args.environment_specification),
            },
            "external_receipt": {
                "path": str(args.external_receipt.resolve()),
                "sha256": sha256_file(args.external_receipt),
            },
            "external_verification": {
                "path": str(args.external_verification_output.resolve()),
                "sha256": sha256_file(args.external_verification_output),
            },
        },
    }
    write_json_new(args.output_lock, lock)
    print(json.dumps({"status": lock["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
