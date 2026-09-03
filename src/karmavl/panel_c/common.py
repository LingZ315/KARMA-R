"""Fail-closed I/O and integrity primitives for the frozen Panel-C workflow."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


OUTCOME_PATH_RE = re.compile(
    r"(^|/)(panel_c_results|target_correctness|target_scores?|target_scored|"
    r"scored_target|target_outcomes?|heldout_correctness|target_candidate_outputs?|"
    r"target_candidate_responses?|target_responses?|target_predictions?)(/|\.|$)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    material = "\n".join(values) + "\n"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def write_jsonl_new(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def verify_file_manifest(manifest_path: Path, root: Path) -> list[dict[str, Any]]:
    """Verify every declared member and reject undeclared hash drift."""

    manifest = load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty files list")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in files:
        relative = str(entry.get("path", "")).replace("\\", "/")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe manifest path: {relative!r}")
        if relative in seen:
            raise ValueError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != int(entry["bytes"]) or actual_hash != str(entry["sha256"]):
            raise ValueError(f"manifest mismatch: {relative}")
        verified.append({"path": relative, "bytes": actual_size, "sha256": actual_hash})
    return verified


def verify_preregistration_archive(
    archive: Path,
    *,
    manifest_path: Path,
    top_level: str | None = None,
) -> list[dict[str, Any]]:
    """Verify archive members directly against the external file manifest."""

    manifest = load_json(manifest_path)
    if top_level is None:
        top_level = str(manifest.get("archive_top_level", "panel_c_preregistration"))
    if not top_level or "/" in top_level or "\\" in top_level or top_level in {".", ".."}:
        raise ValueError(f"unsafe archive top-level directory: {top_level!r}")
    declared = {str(row["path"]).replace("\\", "/"): row for row in manifest["files"]}
    manifest_member = f"{top_level}/{manifest_path.name}"
    expected = {f"{top_level}/{name}" for name in declared} | {manifest_member}
    verified: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        names = {info.filename for info in bundle.infolist() if not info.is_dir()}
        if names != expected:
            raise ValueError(
                f"archive member set mismatch: missing={sorted(expected - names)}, "
                f"extra={sorted(names - expected)}"
            )
        archived_manifest = bundle.read(manifest_member)
        if hashlib.sha256(archived_manifest).hexdigest() != sha256_file(manifest_path):
            raise ValueError("archive contains a different preregistration manifest")
        for relative, entry in sorted(declared.items()):
            material = bundle.read(f"{top_level}/{relative}")
            actual = hashlib.sha256(material).hexdigest()
            if len(material) != int(entry["bytes"]) or actual != str(entry["sha256"]):
                raise ValueError(f"archive member hash mismatch: {relative}")
            verified.append({"path": relative, "bytes": len(material), "sha256": actual})
    return verified


FROZEN_EXECUTION_REQUIRED_PATHS = {
    "PANEL_C_PROTOCOL.md",
    "STATISTICAL_ANALYSIS_PLAN.md",
    "COST_ACCOUNTING_PLAN.md",
    "ENVIRONMENT_LOCK.txt",
    "configs/panel_c_analysis_contract.json",
    "configs/panel_c_explicit_source_metadata_blind_query_semantics_v1.md",
    "configs/PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.schema.json",
    "configs/panel_c_metadata_free_semantic_classifier.json",
    "configs/panel_c_model_pool.json",
    "manifests/panel_c_public_split_id_manifest.json",
    "scripts/panel_c/run_semantic_classifier.py",
    "scripts/panel_c/run_candidate_inference.py",
    "scripts/panel_c/score_candidate_outputs.py",
    "scripts/panel_c/fit_router.py",
    "scripts/panel_c/fit_simple_controls.py",
    "scripts/panel_c/select_fold_hyperparameters.py",
    "scripts/panel_c/select_fold_B_star.py",
    "scripts/panel_c/freeze_fold_policy.py",
    "scripts/panel_c/run_target_routing.py",
    "scripts/panel_c/analyze_primary_endpoint.py",
    "scripts/panel_c/bootstrap_primary_endpoint.py",
    "scripts/panel_c/calculate_deployment_utility.py",
    "src/karmavl/panel_c/common.py",
    "src/karmavl/panel_c/controls.py",
    "src/karmavl/panel_c/nested.py",
    "src/karmavl/panel_c/routing.py",
    "src/karmavl/panel_c/scoring.py",
    "src/karmavl/panel_c/statistics.py",
}


def _verified_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]).replace("\\", "/"): dict(row) for row in rows}


def verify_frozen_execution_files(
    *,
    execution_root: Path,
    preregistration_manifest: Path,
    code_manifest: Path,
    configuration_manifest: Path,
) -> dict[str, Any]:
    """Rehash the live scientific tree against all timestamped manifests.

    This is deliberately separate from authorization so it can be exercised by
    negative drift tests without fabricating an external release.  The guarded
    entry point below always performs both checks.
    """

    root = execution_root.resolve()
    for name, path in {
        "master manifest": preregistration_manifest,
        "code manifest": code_manifest,
        "configuration manifest": configuration_manifest,
    }.items():
        if path.resolve().parent != root:
            raise PermissionError(f"{name} must be located at the frozen execution root")

    master_rows = verify_file_manifest(preregistration_manifest, root)
    code_rows = verify_file_manifest(code_manifest, root)
    configuration_rows = verify_file_manifest(configuration_manifest, root)
    master = _verified_index(master_rows)
    code = _verified_index(code_rows)
    configuration = _verified_index(configuration_rows)

    for manifest_path in (code_manifest, configuration_manifest):
        relative = manifest_path.resolve().relative_to(root).as_posix()
        row = master.get(relative)
        if row is None or row["sha256"] != sha256_file(manifest_path):
            raise PermissionError(f"master manifest does not bind {relative}")

    for category, rows in (("code", code), ("configuration", configuration)):
        for relative, row in rows.items():
            master_row = master.get(relative)
            if master_row is None or master_row["sha256"] != row["sha256"]:
                raise PermissionError(
                    f"master manifest does not recursively bind {category} file: {relative}"
                )

    declared = set(master)
    missing = sorted(FROZEN_EXECUTION_REQUIRED_PATHS - declared)
    if missing:
        raise PermissionError(f"frozen execution manifest omits required scientific files: {missing}")
    if not FROZEN_EXECUTION_REQUIRED_PATHS <= set(code) | set(configuration):
        uncovered = sorted(FROZEN_EXECUTION_REQUIRED_PATHS - (set(code) | set(configuration)))
        raise PermissionError(f"code/configuration manifests leave frozen files uncovered: {uncovered}")

    return {
        "status": "VERIFIED_LIVE_LOCAL_FROZEN_EXECUTION_STATE",
        "master_files": len(master),
        "code_files": len(code),
        "configuration_files": len(configuration),
        "execution_root": str(root),
    }


def verify_bound_files(payload: dict[str, Any], *, anchor: Path) -> dict[str, str]:
    """Verify every ``{path, sha256}`` entry in a freeze artifact.

    Relative paths are resolved only against the directory containing the
    freeze artifact.  This deliberately avoids workspace-dependent fallback
    search paths.
    """

    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("freeze artifact contains no file bindings")
    verified: dict[str, str] = {}
    for name, row in bindings.items():
        if not isinstance(row, dict) or set(row) < {"path", "sha256"}:
            raise ValueError(f"invalid binding: {name}")
        bound = Path(str(row["path"]))
        path = bound if bound.is_absolute() else anchor / bound
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != str(row["sha256"]):
            raise ValueError(f"binding mismatch: {name}")
        verified[str(name)] = actual
    return verified


def outcome_named_paths(root: Path, *, allowed: Iterable[Path] = ()) -> list[dict[str, Any]]:
    """Return path metadata only; never open a possible outcome file."""

    allowed_resolved = {path.resolve() for path in allowed}
    found: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() in allowed_resolved:
            continue
        relative = path.relative_to(root).as_posix()
        if OUTCOME_PATH_RE.search(relative):
            found.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return sorted(found, key=lambda row: row["path"])


def require_authorization_lock(
    lock_path: Path,
    *,
    preregistration_archive: Path,
    preregistration_manifest: Path,
    code_manifest: Path,
) -> dict[str, Any]:
    """Verify cryptographic bindings and re-check the remote immutable asset.

    A locally typed JSON receipt or verification report is insufficient.  The
    provider API and archived asset are checked online on every guarded entry.
    Network/provider failure therefore leaves target execution disabled.
    """

    lock = load_json(lock_path)
    if lock.get("artifact_role") != "panel_c_target_execution_authorization":
        raise PermissionError("authorization lock has the wrong artifact role")
    if lock.get("status") != "AUTHORIZED_REMOTE_TIMESTAMP_VERIFIED":
        raise PermissionError("target execution is not authorized")
    authorized_at = datetime.fromisoformat(str(lock.get("authorized_at_utc", "")).replace("Z", "+00:00"))
    external_at = datetime.fromisoformat(str(lock.get("external_timestamp_utc", "")).replace("Z", "+00:00"))
    if authorized_at.tzinfo is None or external_at.tzinfo is None or external_at >= authorized_at:
        raise PermissionError("external timestamp does not precede authorization")
    if authorized_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise PermissionError("authorization time is in the future")
    bindings = lock.get("bindings", {})
    required_bindings = {
        "preregistration_archive",
        "preregistration_manifest",
        "code_manifest",
        "configuration_manifest",
        "environment_specification",
        "external_receipt",
        "external_verification",
    }
    if not required_bindings <= set(bindings):
        raise PermissionError("authorization lock is missing code/config/environment evidence")
    verify_bound_files(lock, anchor=lock_path.parent)
    expected = {
        "preregistration_archive": (preregistration_archive, sha256_file(preregistration_archive)),
        "preregistration_manifest": (preregistration_manifest, sha256_file(preregistration_manifest)),
        "code_manifest": (code_manifest, sha256_file(code_manifest)),
    }
    for name, (path, actual_hash) in expected.items():
        row = bindings.get(name, {})
        if row.get("sha256") != actual_hash:
            raise PermissionError(f"authorization binding failed for {name}: {path}")
    verification = bindings.get("external_verification", {})
    verification_path = Path(str(verification.get("path", "")))
    if not verification_path.is_absolute():
        verification_path = lock_path.parent / verification_path
    if not verification_path.is_file() or sha256_file(verification_path) != verification.get("sha256"):
        raise PermissionError("external verification report binding failed")
    report = load_json(verification_path)
    if report.get("status") != "VERIFIED_REMOTE_IMMUTABLE_ARCHIVE":
        raise PermissionError("external archive has not been remotely verified")
    if report.get("provider") != "github" or report.get("immutable") is not True:
        raise PermissionError("automatic target authorization requires a GitHub immutable release")
    receipt_binding = bindings.get("external_receipt", {})
    receipt_path = Path(str(receipt_binding.get("path", "")))
    if not receipt_path.is_absolute():
        receipt_path = lock_path.parent / receipt_path
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_binding.get("sha256"):
        raise PermissionError("external receipt binding failed")
    from .external import verify_external_timestamp_receipt

    live = verify_external_timestamp_receipt(receipt_path, preregistration_archive)
    if live.get("status") != "VERIFIED_REMOTE_IMMUTABLE_ARCHIVE":
        raise PermissionError("live remote archive verification failed")
    if live.get("immutable") is not True:
        raise PermissionError("live GitHub release is no longer reported immutable")
    stable_fields = (
        "provider",
        "release_id",
        "persistent_url",
        "remote_archive_sha256",
        "published_at",
        "tag",
        "commit_sha",
        "asset_name",
        "immutable",
    )
    if any(live.get(name) != report.get(name) for name in stable_fields):
        raise PermissionError("remote archive evidence changed after authorization")
    return lock


def require_frozen_execution_state(
    lock_path: Path,
    *,
    preregistration_archive: Path,
    preregistration_manifest: Path,
    code_manifest: Path,
) -> dict[str, Any]:
    """Require live remote authorization and byte-exact local frozen files.

    The configuration manifest is resolved from the current extracted
    preregistration root, never from a parent repository or an independently
    editable working tree.  Its hash must also equal the authorization binding.
    """

    lock = require_authorization_lock(
        lock_path,
        preregistration_archive=preregistration_archive,
        preregistration_manifest=preregistration_manifest,
        code_manifest=code_manifest,
    )
    root = preregistration_manifest.resolve().parent
    configuration_manifest = root / "PANEL_C_CONFIGURATION_MANIFEST_V3.json"
    if not configuration_manifest.is_file():
        raise FileNotFoundError(configuration_manifest)
    configuration_binding = lock.get("bindings", {}).get("configuration_manifest", {})
    if configuration_binding.get("sha256") != sha256_file(configuration_manifest):
        raise PermissionError("authorization binding failed for the live configuration manifest")
    local = verify_frozen_execution_files(
        execution_root=root,
        preregistration_manifest=preregistration_manifest,
        code_manifest=code_manifest,
        configuration_manifest=configuration_manifest,
    )
    return {**lock, "live_local_frozen_execution_state": local}


def add_frozen_execution_arguments(parser: Any, *, required: bool = True) -> None:
    """Add the uniform frozen-state arguments to a result-determining CLI."""

    parser.add_argument("--authorization-lock", type=Path, required=required)
    parser.add_argument("--preregistration-archive", type=Path, required=required)
    parser.add_argument("--preregistration-manifest", type=Path, required=required)
    parser.add_argument("--code-manifest", type=Path, required=required)


def require_frozen_execution_from_args(args: Any) -> dict[str, Any]:
    """Invoke the common guard for a CLI namespace."""

    return require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )


def require_fold_policy_freeze(
    path: Path,
    *,
    expected_fold_id: str | None = None,
) -> dict[str, Any]:
    """Require a hash-bound fold-local policy frozen before target scoring."""

    payload = load_json(path)
    if payload.get("artifact_role") != "panel_c_fold_policy_freeze":
        raise PermissionError("policy freeze has the wrong artifact role")
    if payload.get("status") != "FOLD_POLICY_FROZEN_PRE_TARGET_OUTCOME":
        raise PermissionError("fold policy is not frozen")
    if payload.get("target_outcomes_accessed") is not False:
        raise PermissionError("policy freeze does not assert pre-target construction")
    if expected_fold_id is not None and str(payload.get("fold_id")) != expected_fold_id:
        raise PermissionError("policy freeze belongs to another outer fold")
    if int(payload.get("schema_version", 1)) >= 2:
        required = {
            "analysis_contract",
            "split_manifest",
            "code_manifest",
            "semantic_features",
            "candidate_profiles",
            "h1_router",
            "h1_5_router",
            "h2_router",
            "simple_controls",
            "selected_h2_gate3_accuracy_policy",
            "selected_h2_gate4_utility_policy",
            "selected_B_A_star",
            "selected_B_U_star",
        }
    else:
        required = {
            "analysis_contract",
            "split_manifest",
            "code_manifest",
            "semantic_features",
            "candidate_profiles",
            "h2_router",
            "simple_controls",
            "selected_margin",
            "selected_B_f_star",
        }
    if not required <= set(payload.get("bindings", {})):
        raise PermissionError("policy freeze is missing a result-determining binding")
    verify_bound_files(payload, anchor=path.parent)
    return payload


def _bound_path(payload_path: Path, row: dict[str, Any]) -> Path:
    bound = Path(str(row.get("path", "")))
    return bound if bound.is_absolute() else payload_path.parent / bound


def parse_utc(value: Any, *, field: str) -> datetime:
    """Parse a timezone-aware ISO timestamp and normalize it to UTC."""

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid UTC timestamp in {field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timezone is required in {field}")
    return parsed.astimezone(timezone.utc)


def require_policy_bundle_freeze(
    lock_path: Path,
    *,
    bundle_path: Path | None = None,
    expected_fold_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the complete five-fold policy bundle and its immutable lock.

    The lock is deliberately distinct from the per-fold freezes: target
    semantic inference is not enabled until every outer-fold policy exists and
    the complete serialized bundle is hash-bound.
    """

    lock = load_json(lock_path)
    if lock.get("artifact_role") != "panel_c_fold_policy_bundle_freeze":
        raise PermissionError("policy-bundle lock has the wrong artifact role")
    if lock.get("status") != "ALL_FOLD_POLICIES_FROZEN_PRE_TARGET_INFERENCE":
        raise PermissionError("complete fold-policy bundle is not frozen")
    if lock.get("target_outcomes_accessed") is not False:
        raise PermissionError("policy-bundle lock does not assert outcome blindness")
    bindings = lock.get("bindings", {})
    if not {"policy_bundle", "code_manifest", "preregistration_archive"} <= set(bindings):
        raise PermissionError("policy-bundle lock is missing a required binding")
    verify_bound_files(lock, anchor=lock_path.parent)
    bound_bundle = _bound_path(lock_path, bindings["policy_bundle"])
    if bundle_path is not None and bound_bundle.resolve() != bundle_path.resolve():
        raise PermissionError("supplied policy bundle differs from the locked bundle")
    bundle = load_json(bound_bundle)
    if bundle.get("artifact_role") != "panel_c_complete_fold_policy_bundle":
        raise PermissionError("policy bundle has the wrong artifact role")
    if bundle.get("status") != "FROZEN_PRE_TARGET_INFERENCE":
        raise PermissionError("policy bundle is not frozen pre-target")
    if bundle.get("target_outcomes_accessed") is not False:
        raise PermissionError("policy bundle does not assert outcome blindness")
    folds = bundle.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise PermissionError("policy bundle must contain exactly five outer folds")
    ids = [str(row.get("fold_id")) for row in folds]
    if len(set(ids)) != 5:
        raise PermissionError("policy bundle contains duplicate outer folds")
    required_policy_fields = {
        "fold_id",
        "held_out_source",
        "H1_policy",
        "H1_5_policy",
        "H2_gate3_accuracy_policy",
        "H2_gate4_utility_policy",
        "B_A_star_method",
        "B_U_star_method",
        "selected_margins",
        "selected_hyperparameters",
        "model_action_space_metadata",
        "fold_policy_freeze",
        "fold_policy_freeze_sha256",
    }
    for row in folds:
        if not required_policy_fields <= set(row):
            raise PermissionError("policy bundle contains an incomplete fold policy")
        freeze_path = Path(str(row["fold_policy_freeze"]))
        if not freeze_path.is_absolute():
            freeze_path = bound_bundle.parent / freeze_path
        if sha256_file(freeze_path) != str(row["fold_policy_freeze_sha256"]):
            raise PermissionError("per-fold policy freeze hash drift")
        require_fold_policy_freeze(freeze_path, expected_fold_id=str(row["fold_id"]))
    if expected_fold_id is not None and expected_fold_id not in ids:
        raise PermissionError("requested fold is absent from the complete policy bundle")
    return lock, bundle


def require_target_route_bundle_freeze(
    lock_path: Path,
    *,
    routes_path: Path | None = None,
    policy_bundle_lock: Path | None = None,
    expected_fold_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the single all-fold target-route ledger and its pre-inference lock."""

    lock = load_json(lock_path)
    if lock.get("artifact_role") != "panel_c_target_route_bundle_freeze":
        raise PermissionError("target-route bundle lock has the wrong artifact role")
    if lock.get("status") != "ALL_TARGET_ROUTES_FROZEN_PRE_CANDIDATE_INFERENCE":
        raise PermissionError("complete target-route ledger is not frozen")
    if lock.get("target_outcomes_accessed") is not False:
        raise PermissionError("target-route bundle does not assert outcome blindness")
    bindings = lock.get("bindings", {})
    required = {
        "target_routes",
        "policy_bundle_lock",
        "policy_bundle",
        "preregistration_archive",
        "code_manifest",
    }
    if not required <= set(bindings):
        raise PermissionError("target-route bundle lock is missing a required binding")
    verify_bound_files(lock, anchor=lock_path.parent)
    bound_routes = _bound_path(lock_path, bindings["target_routes"])
    if routes_path is not None and bound_routes.resolve() != routes_path.resolve():
        raise PermissionError("supplied target routes differ from the frozen route ledger")
    bound_policy_lock = _bound_path(lock_path, bindings["policy_bundle_lock"])
    if policy_bundle_lock is not None and bound_policy_lock.resolve() != policy_bundle_lock.resolve():
        raise PermissionError("supplied policy lock differs from the route lock binding")
    require_policy_bundle_freeze(
        bound_policy_lock,
        bundle_path=_bound_path(lock_path, bindings["policy_bundle"]),
        expected_fold_id=expected_fold_id,
    )
    rows = load_jsonl(bound_routes)
    if not rows:
        raise PermissionError("frozen target-route ledger is empty")
    if len({str(row.get("query_id")) for row in rows}) != len(rows):
        raise PermissionError("frozen target-route ledger contains duplicate query IDs")
    response_keys = {
        "response",
        "answer",
        "correct",
        "correctness",
        "score",
        "outcome",
        "utility_outcome",
    }
    if any(response_keys & {str(key).casefold() for key in row} for row in rows):
        raise PermissionError("target-route ledger contains response/outcome information")
    if any(row.get("target_outcomes_accessed") is not False for row in rows):
        raise PermissionError("target-route row lacks an outcome-blind assertion")
    if expected_fold_id is not None and not any(
        str(row.get("fold_id")) == expected_fold_id for row in rows
    ):
        raise PermissionError("requested fold is absent from the frozen target routes")
    return lock, rows


def require_target_route_freeze(
    path: Path,
    *,
    expected_fold_id: str | None = None,
) -> dict[str, Any]:
    """Require deterministic target routes hash-frozen before outcomes open."""

    payload = load_json(path)
    if payload.get("artifact_role") != "panel_c_fold_target_route_freeze":
        raise PermissionError("target-route freeze has the wrong artifact role")
    if payload.get("status") != "TARGET_ROUTES_FROZEN_PRE_OUTCOME":
        raise PermissionError("target routes are not frozen")
    if payload.get("target_outcomes_accessed") is not False:
        raise PermissionError("target-route freeze is not pre-outcome")
    if expected_fold_id is not None and str(payload.get("fold_id")) != expected_fold_id:
        raise PermissionError("target-route freeze belongs to another fold")
    required = {"policy_freeze", "target_routes", "h2_target_features", "code_manifest"}
    if not required <= set(payload.get("bindings", {})):
        raise PermissionError("target-route freeze is incomplete")
    verify_bound_files(payload, anchor=path.parent)
    require_fold_policy_freeze(
        Path(str(payload["bindings"]["policy_freeze"]["path"])),
        expected_fold_id=expected_fold_id,
    )
    return payload


def require_analysis_input_freeze(path: Path, *, paired_rows: Path) -> dict[str, Any]:
    """Verify the immutable paired primary-analysis input created post-outcome."""

    payload = load_json(path)
    if payload.get("artifact_role") != "panel_c_primary_analysis_input_freeze":
        raise PermissionError("analysis input freeze has the wrong artifact role")
    if payload.get("status") != "PAIRED_PRIMARY_INPUT_FROZEN":
        raise PermissionError("paired primary input is not frozen")
    verify_bound_files(payload, anchor=path.parent)
    actual = sha256_file(paired_rows)
    accepted = {
        str(row["sha256"])
        for name, row in payload["bindings"].items()
        if name in {"paired_target_rows", "utility_paired_target_rows"}
    }
    if actual not in accepted:
        raise PermissionError("paired target input differs from its analysis freeze")
    return payload
