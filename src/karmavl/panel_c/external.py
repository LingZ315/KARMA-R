"""Remote verification for immutable Panel-C preregistration evidence."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import load_json, sha256_file


USER_AGENT = "KARMA-R-Panel-C-preregistration-verifier/1.0.2"
MAX_RESPONSE_BYTES = 20 * 1024 * 1024


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must normalize to UTC")
    return parsed


def _request(url: str, *, accept: str = "application/json") -> urllib.response.addinfourl:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise PermissionError("remote evidence must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return urllib.request.urlopen(request, timeout=60)  # noqa: S310 - hosts are provider constrained


def _json(url: str) -> dict[str, Any]:
    with _request(url) as response:
        material = response.read(MAX_RESPONSE_BYTES + 1)
    if len(material) > MAX_RESPONSE_BYTES:
        raise ValueError("provider response exceeds verifier limit")
    value = json.loads(material)
    if not isinstance(value, dict):
        raise ValueError("provider response is not a JSON object")
    return value


def _remote_sha256(url: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with _request(url, accept="application/octet-stream") as response:
        while block := response.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _same_timestamp(receipt_value: str, provider_value: str) -> bool:
    left = _utc(receipt_value)
    right = _utc(provider_value)
    return abs((left - right).total_seconds()) <= 1.0


def _github_commit(repository: str, tag: str) -> str:
    ref = _json(f"https://api.github.com/repos/{repository}/git/ref/tags/{urllib.parse.quote(tag, safe='')}")
    obj = ref.get("object", {})
    if obj.get("type") == "commit":
        return str(obj.get("sha"))
    if obj.get("type") == "tag":
        tag_object = _json(f"https://api.github.com/repos/{repository}/git/tags/{obj.get('sha')}")
        target = tag_object.get("object", {})
        if target.get("type") == "commit":
            return str(target.get("sha"))
    raise PermissionError("GitHub tag does not resolve to a commit")


def _verify_github(receipt: dict[str, Any], archive: Path) -> dict[str, Any]:
    repository = str(receipt.get("repository", ""))
    tag = str(receipt.get("tag", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not tag:
        raise ValueError("GitHub receipt requires repository owner/name and tag")
    if receipt.get("attestation_requested") is True:
        raise PermissionError(
            "attestation_requested=true but no independent attestation verifier is frozen; target remains disabled"
        )
    release = _json(
        f"https://api.github.com/repos/{repository}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    )
    if release.get("draft") is not False or not release.get("published_at"):
        raise PermissionError("GitHub release is draft or unpublished")
    if release.get("immutable") is not True:
        raise PermissionError("GitHub release is mutable; immutable must equal true")
    if str(receipt.get("release_id")) != str(release.get("id")):
        raise PermissionError("GitHub release ID mismatch")
    if str(release.get("tag_name", "")) != tag:
        raise PermissionError("GitHub release tag mismatch")
    if not isinstance(receipt.get("prerelease"), bool) or receipt["prerelease"] is not bool(
        release.get("prerelease")
    ):
        raise PermissionError("GitHub prerelease state mismatch")
    if receipt.get("immutable") is not True:
        raise PermissionError("receipt does not record immutable=true")
    persistent = str(receipt.get("persistent_url", ""))
    if persistent.rstrip("/") != str(release.get("html_url", "")).rstrip("/"):
        raise PermissionError("receipt URL does not match GitHub release")
    provider_timestamp = str(release["published_at"])
    if not _same_timestamp(str(receipt.get("published_at", "")), provider_timestamp):
        raise PermissionError("receipt timestamp does not match GitHub release")
    resolved_commit = _github_commit(repository, tag)
    if str(receipt.get("commit_sha", "")) != resolved_commit:
        raise PermissionError("GitHub immutable tag commit mismatch")
    if str(receipt.get("asset_name", "")) != archive.name:
        raise PermissionError("receipt asset name differs from the frozen archive")
    assets = [row for row in release.get("assets", []) if str(row.get("name")) == archive.name]
    if len(assets) != 1:
        raise FileNotFoundError("frozen archive is not a unique GitHub release asset")
    remote_hash, remote_size = _remote_sha256(str(assets[0]["browser_download_url"]))
    return {
        "provider": "github",
        "release_id": str(release["id"]),
        "persistent_url": persistent,
        "published_at": _utc(provider_timestamp).isoformat().replace("+00:00", "Z"),
        "immutable": True,
        "prerelease": bool(release.get("prerelease")),
        "asset_name": archive.name,
        "remote_archive_sha256": remote_hash,
        "remote_archive_bytes": remote_size,
        "repository": repository,
        "tag": tag,
        "commit_sha": resolved_commit,
        "attestation_verified": False,
        "attestation_note": "optional GitHub artifact attestation was not required by this receipt",
    }


def verify_external_timestamp_receipt(receipt_path: Path, archive: Path) -> dict[str, Any]:
    """Verify a receipt against a live provider API and downloaded asset."""

    receipt = load_json(receipt_path)
    local_hash = sha256_file(archive)
    if str(receipt.get("local_archive_sha256", "")) != local_hash:
        raise PermissionError("receipt local archive hash mismatch")
    if receipt.get("verification_status") != "PENDING_REMOTE_VERIFICATION":
        raise PermissionError("receipt is not a pending live-provider verification request")
    if receipt.get("target_execution_authorized") is not False:
        raise PermissionError("a receipt cannot self-authorize target execution")
    provider = str(receipt.get("provider", "")).casefold()
    try:
        if provider == "github":
            result = _verify_github(receipt, archive)
        elif provider in {"zenodo", "osf", "osf_registration"}:
            raise PermissionError(
                "EXTERNAL_TIMESTAMP_MANUAL_VERIFICATION_REQUIRED; target_execution_authorized=false"
            )
        else:
            raise PermissionError("unsupported external timestamp provider")
    except (urllib.error.URLError, TimeoutError) as error:
        raise PermissionError("remote verification unavailable; target remains disabled") from error
    if result["remote_archive_sha256"] != local_hash:
        raise PermissionError("remote archived bytes differ from the local frozen archive")
    if str(receipt.get("remote_archive_sha256", "")) != result["remote_archive_sha256"]:
        raise PermissionError("receipt remote archive hash mismatch")
    receipt_time = _utc(str(receipt.get("published_at", "")))
    if receipt_time >= datetime.now(timezone.utc):
        raise PermissionError("external timestamp is not earlier than verification")
    return {
        "schema_version": 1,
        "artifact_role": "panel_c_external_timestamp_remote_verification",
        "status": "VERIFIED_REMOTE_IMMUTABLE_ARCHIVE",
        "target_execution_authorized": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "local_archive_sha256": local_hash,
        "local_sha256": local_hash,
        "remote_sha256": result["remote_archive_sha256"],
        **result,
    }
