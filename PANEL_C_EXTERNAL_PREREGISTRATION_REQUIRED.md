# Panel-C external immutable-timestamp handoff

Status: **READY FOR EXTERNAL IMMUTABLE TIMESTAMP; TARGET EXECUTION UNAUTHORIZED**.

The release receipt and authorization lock are intentionally absent. This package has not been externally archived and must not be represented as timestamped.

## Automatic authorization path

1. Publish the unchanged `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip` asset in a GitHub release.
2. Confirm through the live GitHub API that `draft == false` and `immutable == true`.
3. Record the exact numeric release ID, repository, tag, resolved 40-character commit SHA, `published_at`, `prerelease`, asset name, canonical URL, and local archive SHA-256.
4. Download the remote asset and independently recompute its SHA-256; it must equal the local archive hash.
5. Copy `PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT_TEMPLATE.json` outside the frozen archive and fill it only with the real live values.
6. Run `scripts/verify_external_timestamp.py`, then `scripts/authorize_panel_c_target_execution.py`.

Other immutable mechanisms require explicit manual verification and cannot automatically create `PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock`.

## Fail-closed boundary

Authorization also verifies the preregistration archive, code manifest, configuration manifest, environment binding, exact remote bytes, and absence of premature target artifacts. A mutable/missing release field, tag or commit mismatch, asset mismatch, hash drift, provider failure, code/configuration drift, or premature target output leaves execution disabled.

After authorization, complete all non-target fitting and freeze the five-fold policy bundle before target semantic inference. Freeze the all-fold target-route ledger before any target candidate response. Candidate generation accepts only the exact selected-route query subset. Outcome scoring and analysis remain later stages.

The immediate next action is immutable GitHub publication, not Panel-C target inference.
