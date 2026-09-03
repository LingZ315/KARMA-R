# Panel-C statistical analysis plan

## Population and pairing

The fixed evaluation comprises five held-out sources and 7,474 unique target queries. Each query contributes one paired accuracy-selected `H2_A` versus accuracy-selected `B_A*` correctness difference in {-1,0,+1}. No post-inference scientific exclusion is allowed; terminal failures are incorrect and remain in the denominator.

## Primary calculation

The point estimate is the mean paired difference across all queries. The primary interval uses 10,000 source-stratified paired percentile-bootstrap replicates, NumPy PCG64 seed 2026082811, and confidence level 0.95. Within each held-out source, queries are sampled with replacement and identical indices are applied to `H2_A` and `B_A*`.

Statistical superiority requires the pooled interval's lower bound > 0. Practical superiority requires a point estimate >= 0.015. Both are required. There is one primary endpoint; other intervals are descriptive.

## Source summaries

Report pooled, macro, and median source effects; positive/negative/zero source counts; and source-specific descriptive intervals. All sources need not be positive. Five sources do not justify population inference over arbitrary future domains.

## Utility

At N in {1,000; 10,000; 100,000; 1,000,000}, compute the frozen normalized utility for separately utility-selected `H2_U` and `B_U*` using measured onboarding and method-specific serving GPU-seconds. No unmeasured timing value is substituted, and Gate-3 policies cannot replace Gate-4 policies.

## Chronology and integrity

Target correctness is opened only after live GitHub immutable-release authorization, the complete five-fold policy-bundle freeze, target semantic inference, and the all-fold target-route freeze. The analyzer assembles separate Gate-3 and Gate-4 paired rows from those frozen routes and exact scored route files, then hash-freezes both inputs. Hash-bound execution receipts are primary chronology evidence; filesystem mtimes are secondary.
