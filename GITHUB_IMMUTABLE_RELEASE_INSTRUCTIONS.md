# GitHub immutable-release instructions for Panel C

Status: **manual author action required; no external release or target authorization is claimed by this package**.

Use the exact frozen asset `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip`. Do not rebuild or rename it after computing its local SHA-256.

1. Create or update the intended GitHub repository and commit the frozen preregistration source.
2. Enable and use GitHub immutable releases for that repository, if the repository is eligible.
3. Create a new tag pointing to the exact frozen commit. Record the resolved 40-character commit SHA.
4. Create a release for that tag and attach the exact `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip` asset.
5. Publish the release. A draft creation time, repository creation time, or first-upload time is not the preregistration timestamp.
6. Query the live GitHub release API and confirm that `draft == false` and `immutable == true`. A missing, false, or non-Boolean `immutable` value is a hard failure.
7. Record the numeric release ID, exact tag, resolved commit SHA, `published_at`, and `prerelease` value.
8. Confirm that the release contains exactly one asset with the name `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip`.
9. Download that remote asset and calculate its SHA-256. It must equal the local archive SHA-256 exactly.
10. If a GitHub artifact attestation is available, verify it independently and retain the evidence; attestation is supplementary and does not replace `immutable == true` or the byte-for-byte hash check.
11. Copy `PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT_TEMPLATE.json` outside the frozen release, replace all null fields with real provider values, set `verification_status` to `PENDING_REMOTE_VERIFICATION`, and leave `target_execution_authorized` as `false`.
12. Run `scripts/verify_external_timestamp.py` against the real receipt and local archive. Save the successful output as `PANEL_C_EXTERNAL_RELEASE_VERIFICATION.json` outside the preregistration archive.
13. Run `scripts/authorize_panel_c_target_execution.py`. It rechecks the live GitHub API, redownloads the asset, verifies code/configuration/environment manifests, and refuses any premature target-result/output artifact.
14. Preserve the receipt, verification report, authorization lock, release URL, release ID, tag, commit SHA, `published_at`, and both SHA-256 values as the execution audit trail.

Non-GitHub mechanisms require explicit manual verification and cannot automatically create `PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock`; the automatic path remains a live-verified GitHub Immutable Release.
