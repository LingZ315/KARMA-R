# KARMA-R Panel C v1.0.2: final pre-outcome protocol

Evidence state: **locally frozen; external immutable timestamp absent; target execution unauthorized**.

This protocol is the sole authoritative confirmatory execution contract for Panel C. It preserves the v7.0.1 five-fold explicit-source-metadata-blind nested leave-one-source-out design and changes only pre-outcome runtime feasibility, chronology, immutable-release verification, dual-gate selector symmetry, and simple-control action-space comparability. No dataset, source split, candidate model, target outcome, or H2-favouring parameter was added.

## 1. Outcome-blind boundary

Before the externally verified release and subsequent locks, no process may generate or inspect a Panel-C target candidate response, reference answer, correctness, accuracy, utility, bootstrap statistic, or oracle matrix. A target-result-like artifact found before authorization is hashed as an opaque file, quarantined without reading its body, and causes authorization to fail. The V3 timestamp artifact contains none.

Five sources define five outer folds: IconQA (2,200 target rows), MME (515), TallyQA (2,200), VisOnlyQA (359), and WeMath2.0-Standard (2,200), totalling 7,474 unique targets. In fold `f`, the held-out source appears only in `target`; profile, calibration, and policy roles contain non-held-out sources. Query reuse in non-target roles across outer folds is permitted only when every result-determining fit and selection remains fold-local. No pooled cross-fold selector enters a confirmatory target policy.

## 2. Representation and model pool

No source name, dataset ID, benchmark name, source one-hot field, or reconstructed known-source label is supplied to a confirmatory router. Image/question content is not rewritten; latent source cues may therefore remain. Source labels exist only in orchestration metadata for fold construction, boundary verification, and source-level reporting.

The semantic feature generator is `Qwen/Qwen3-VL-32B-Instruct` at revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`. It receives one image and one question and returns exactly `primary_class`, `subtype`, `ambiguity`, and a rationale of at most 20 English words. The exact system prompt, user wrapper, seven classes, four subtypes, `none` rule, ambiguity schema, JSON parser, temperature 0, `do_sample=false`, `top_p=1`, 192-token maximum, seed rule, three identical attempts, and deterministic fallback are frozen. Individual target outputs may never be manually repaired.

Existing legitimate human-adjudicated annotations support the schema. No compatible retained mapping from those rows to the frozen classifier's exact image/question inputs is available in this package, so classifier-versus-human accuracy, macro-F1, confusion matrix, and per-class recall remain **PARTIAL / NOT COMPUTABLE**. No alignment is fabricated.

The route pool is the SmolVLM incumbent plus five candidates: Phi-4 Multimodal, Granite 4.0 3B Vision, Ovis2.5-9B, Qwen3-VL-4B-Instruct, and InternVL3.5-4B-HF. Repositories, revisions, licenses, prompts, precision, and terminal-failure handling remain as frozen in the model-pool file. No model may be added after the external timestamp.

## 3. Verified semantic runtime and cost

The semantic classifier uses BF16, no quantization, batch size one, SDPA, and Hugging Face `device_map="auto"` across exactly four visible NVIDIA GeForce RTX 5090 GPUs with a 30-GiB ceiling per visible device. The model must occupy all four devices or the runner fails. A real outcome-free synthetic smoke test on physical devices 0, 1, 4, and 5 loaded the pinned processor and model, completed multimodal generation in 3.307659 synchronized wall seconds, produced strictly valid semantic JSON without fallback, and used at most 18,402.32 MiB allocated on any device. Physical device indices are scheduler-assigned; the frozen requirement is four idle visible RTX 5090 devices.

For `G` concurrently occupied GPUs over synchronized generation time `t`, semantic cost is `G × t`. The smoke generation therefore cost 13.230635 GPU-seconds. For every first-use target query and method requiring semantic features:

`C_serve = C_semantic + C_routing + C_selected_model`.

For methods without semantic classification, `C_semantic=0`. The CPU NumPy router contributes zero GPU-seconds but retains measured onboarding/training records. Confirmatory utility uses the conservative first-use cost. A repeated identical-query cache hit may be reported only as a separately labelled post-confirmatory scenario.

## 4. H1, H1.5, and H2

H1 uses candidate-pooled profile features, H1.5 uses candidate-specific global profile evidence repeated across semantic slots, and H2 uses candidate-specific global plus matched class/subtype profile evidence. Profile estimation, support-25 fallback, beta smoothing, feature order, L2 bilinear learner, and sensitivity learners remain unchanged. All profile construction and fitting are fold-local and exclude the held-out source.

## 5. Gate 3 accuracy selection

For fold `f`, select the H2 margin on fold-local policy rows as:

`m_f,A* = argmax_m Acc_policy,f(H2_m)`.

The deterministic ranking tuple is: higher accuracy; higher utility; lower mean serving GPU-seconds; then fixed margin order `[0.0, 0.0025, 0.005, 0.01, 0.02]`. Define `H2_f,A = H2(m_f,A*)`.

For every simple control evaluated on the same policy queries, select:

`B_f,A* = argmax_B Acc_policy,f(B)`.

The mirrored ranking tuple is: higher accuracy; higher utility; lower mean serving GPU-seconds; then fixed method order `[incumbent_only, cheapest_candidate, static_calibration_global_best, static_class_conditional_best, logistic_raw, nearest_profile]`.

The query-weighted primary endpoint is:

`Delta_A(C) = sum_f sum_{i in T_f}[Y_i(H2_f,A)-Y_i(B_f,A*)] / sum_f |T_f|`.

Report the macro fold effect, median fold effect, positive/negative/zero fold counts, and individual fold estimates. Statistical positivity means the source-stratified paired-bootstrap 95% CI lower bound exceeds zero. Practical meaning separately requires a point estimate of at least +0.015. Both are required for the frozen combined actionability rule.

## 6. Gate 4 utility selection

Independently select:

`m_f,U* = argmax_m U_policy,f(H2_m)` and `B_f,U* = argmax_B U_policy,f(B)`.

H2 uses higher utility, higher accuracy, lower cost, then the same fixed margin order. The comparator uses higher utility, higher accuracy, lower cost, then the same fixed method order. Define `H2_f,U=H2(m_f,U*)` and compare `Delta_U(C)=U(H2_U)-U(B_U*)` at the frozen horizons and `lambda=0.01`.

Gate-3 and Gate-4 policies may legitimately differ. Their differences must be reported after evaluation. Neither policy may be substituted into the other gate after target results exist.

## 7. Matched simple controls

H2 can serve the incumbent or any candidate. Static-Global, Static-Class, Logistic-Raw, and Nearest-Profile use the same six-route action space. Static-Global may select the incumbent if it is best on fold-local calibration evidence. Incumbent-Only is deliberately restricted to the incumbent. Cheapest-Candidate is deliberately restricted to the five newcomers as a named cost reference and is not represented as a strongest adaptive matched-space control. No confirmatory source-conditional control is allowed.

Logistic-Raw uses the frozen 7-class, 4-subtype, and ambiguity indicators plus a six-route one-hot vector; inputs are unnormalized binary indicators. It uses binary calibration correctness labels, no class weighting, L2=10 on all non-intercept coefficients, an unpenalized intercept, deterministic full-batch Newton/IRLS, 100 maximum iterations, and convergence at maximum absolute step below `1e-10`.

Nearest-Profile represents each eligible route by its smoothed global accuracy, active primary-class accuracy, and active subtype accuracy, with class fallback when subtype support is insufficient. It minimizes unweighted Euclidean distance to `[1,1,1]`, then lower profile generation cost, then fixed route order. The incumbent uses exactly the same construction and smoothing as candidates.

## 8. Terminal failures

Semantic malformed JSON, unknown labels, incompatible class/subtype pairs, empty output, timeout, OOM, or inference exception trigger identical deterministic retries up to three total attempts. If still invalid, emit the frozen `general_visual_reasoning` / `none` / `ambiguity=true` fallback, record the terminal error, and retain all attempted GPU time. Candidate inference similarly makes three identical deterministic attempts; terminal failure emits an empty sealed response with `success=false`, retains attempted cost, stays in the denominator, and scores incorrect. No target row is manually repaired or excluded. No result-based early stopping is permitted.

## 9. External immutability

Automatic authorization supports only a GitHub release whose live API reports `draft=false` and `immutable=true`. The verifier records and checks the numeric release ID, exact tag, resolved 40-character commit SHA (including annotated-tag resolution), `published_at`, `prerelease`, release URL, exact unique asset name, local SHA-256, and SHA-256 of bytes redownloaded from the remote asset. A missing/false `immutable`, provider/network failure, hash drift, tag/commit/release mismatch, or missing asset fails closed. Optional artifact attestation may supplement but never replace these checks.

Zenodo and OSF are manual-only in v1.0.2. A URL, creation timestamp, or typed local JSON cannot authorize execution; the state remains `EXTERNAL_TIMESTAMP_MANUAL_VERIFICATION_REQUIRED` and `target_execution_authorized=false`.

The package contains only `PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT_TEMPLATE.json`. It contains nulls and `target_execution_authorized=false` and cannot authorize. `PANEL_C_EXTERNAL_RELEASE_VERIFICATION.json` and `PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock` may be created only from real live provider values after publication.

## 10. Authoritative execution chronology

No alternate confirmatory order is permitted:

A. Freeze protocol and result-determining code; build and hash the exact V3 archive.

B. Publish that exact archive as a GitHub immutable release.

C. Live-verify release state and remote bytes; create `PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock`.

D. For each outer fold: run non-target semantic inference; run non-target candidate inference; score profile/calibration/policy; construct profiles; fit H1/H1.5/H2 and simple controls; select `H2_A`, `H2_U`, `B_A*`, and `B_U*`; create the immutable per-fold policy freeze.

E. Serialize all five complete fold policies and create `PANEL_C_FOLD_POLICIES_FROZEN.lock`.

F. Run fold-scoped target semantic inference. It requires the authorization and complete policy locks.

G. Generate outcome-free target route decisions from only frozen semantic features, routers, policies, margins, and comparator mappings.

H. Merge the five ledgers as `panel_c_target_routes_frozen.jsonl`, hash it, and create `PANEL_C_TARGET_ROUTES_FROZEN.lock` binding the route hash, protocol archive hash, policy bundle hash, code manifest, and primary receipts.

I. Run only the target candidate query subsets selected by `H2_A`, `H2_U`, `B_A*`, or `B_U*`. Candidate inference requires authorization, complete-policy, and route locks and rejects any extra query/route pair.

J. Seal raw target candidate outputs and their hash-bound execution receipts.

K. Open only fold-scoped target answer ledgers and score the sealed selected-route outputs. Terminal failures score incorrect.

L. Assemble Gate-3 pairs and run the frozen primary endpoint, source-stratified paired bootstrap, and source summaries.

M. Assemble Gate-4 pairs and run the frozen deployment utility analysis.

N. Only after L and M may an oracle matrix or exploratory analysis be generated or opened.

Explicit hash-bound timestamps and receipts are primary chronology evidence. Filesystem mtimes are secondary only. `verify_panel_c_target_chronology.py` verifies that policies precede target semantics/candidate inference, routes precede candidate inference, scoring postdates its sealed candidate output, and primary analysis postdates all scoring.

## 11. Statistics and reporting

The primary endpoint uses 10,000 source-stratified paired percentile-bootstrap draws with NumPy PCG64 seed 2026082811 and 95% intervals. Resampling is within held-out source with identical paired indices. The query-weighted effect is primary; the macro fold effect, median, signs, and fold-specific estimates are secondary/descriptive. There is one primary endpoint; secondary intervals are not promoted after outcomes.

The complete target cohort is run unless an operational/safety failure interrupts execution. Apparent advantage, disadvantage, significance, or futility cannot stop the target cohort. Every interruption is documented.

## 12. Current state

The Qwen execution smoke test is PASS. Code and tests implement the policy/route locks, minimal-response guard, immutable-release verifier, mirrored Gate selectors, matched action spaces, and live local file rehashing. The V3 archive remains **NOT YET ARCHIVED** externally. No target candidate output existed or was accessed during this revision; no target correctness, accuracy, utility, or scientific parameter tuning occurred. The next action is to publish the exact frozen V3 preregistration ZIP as a verified GitHub immutable release before any Panel-C target inference.
