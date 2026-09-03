# KARMA-R Panel C preregistration v1.0.2

This directory is the self-contained, public-safe, result-determining freeze for the prospective Panel-C evaluation. It contains the protocol, public split identifiers, model/configuration contracts, exact semantic prompt, implementation, environment metadata, manifests, authorization/chronology verifiers, and outcome-free tests. It contains no benchmark image or question text, source record identifier, reference answer, candidate response, correctness value, target route decision, Panel-C result, completed external receipt, external verification report, policy/route lock, or target authorization lock.

The primary Gate-3 comparison is fold-local accuracy-selected `H2_A` versus accuracy-selected `B_A*`. Gate 4 separately compares utility-selected `H2_U` with utility-selected `B_U*`. The strongest adaptive simple controls share H2's incumbent-plus-five-candidate action space. No explicit source/dataset identifier reaches a confirmatory router, although latent cues may remain in image/question content.

The pinned Qwen3-VL-32B classifier has passed a real synthetic-only four-RTX-5090 BF16 execution smoke test. This is runtime evidence, not construct accuracy or Panel-C outcome evidence. Query-time semantic cost uses the full concurrent `G × wall-time` rule.

Do not execute target code from this archive. The exact outer ZIP must first be published as a GitHub release whose live API reports `immutable=true`; its remote bytes, tag, commit, release ID, and `published_at` must then verify. Zenodo/OSF cannot automatically authorize. After authorization, all non-target fits and both gate selectors must be completed and the five-fold policy bundle locked before target semantics. Target routes must be separately frozen before any candidate response. Candidate inference then accepts only the exact selected query subset for each route.

Authoritative order:

1. Publish and live-verify the exact immutable `KARMA_R_PANEL_C_PREREGISTRATION_V3.zip`; create the authorization lock only after remote verification.
2. Run non-target semantic/candidate inference, scoring, fitting, and dual-gate selection.
3. Freeze every per-fold policy and the complete five-fold bundle.
4. Run target semantic inference.
5. Generate and freeze the single all-fold target-route ledger.
6. Run only selected target candidate inference and seal its receipts.
7. Score target outputs, run Gate-3 primary/bootstrap, then Gate-4 utility.
8. Only then run exploratory/oracle analyses.

Any missing lock, mutable release, hash/identity mismatch, unexpected premature target artifact, manifest drift, or chronology violation fails closed.
