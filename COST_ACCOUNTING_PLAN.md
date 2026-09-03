# Panel-C cost-accounting plan

All costs are GPU-seconds. For method `m` and deployment horizon `N`:

`C_total,m(N) = C_onboard,m + N*C_serve,m`,

`U_m(N,lambda) = A_m - lambda*C_total,m(N)/N`, with `lambda=0.01`.

For query `q`, `C_serve(q)=C_feature-generation(q)+C_router(q)+C_selected-model(q)`. The feature term is zero only when the method does not run the frozen semantic classifier. The NumPy router runs on CPU and therefore adds zero GPU-seconds at serving time; its training time remains an onboarding record.

| Method | Qwen semantic pass per target? | Learned router? | Selected VLM? | Onboarding components |
|---|---:|---:|---:|---|
| Incumbent | No | No | Yes | None |
| Cheapest candidate | No | No | Yes | Profile cost measurement |
| Static calibration-global | No | No | Yes | Calibration inference |
| Static class-conditional | Yes | No | Yes | Non-target semantics + calibration |
| Logistic raw | Yes | Yes | Yes | Non-target semantics + calibration + training |
| Nearest profile | Yes | No (lookup) | Yes | Non-target semantics + profiling |
| H1 | Yes | Yes | Yes | Semantics + profile + calibration + policy + training |
| H1.5 | Yes | Yes | Yes | Same as H1 |
| H2 | Yes | Yes | Yes | Same as H1 |

Semantic timing is frozen to exactly four visible RTX 5090 GPUs, BF16, no quantization, `device_map="auto"`, a 30-GiB/device ceiling, batch size one, repository preprocessing at the pinned revision, one outcome-free warm-up, CUDA synchronization, and every attempted query in the denominator. If `G` devices are concurrently occupied for generation wall time `t`, cost is `G*t`; four-GPU inference is never recorded as one-GPU wall time. The synthetic execution test measured 3.307659 seconds and therefore 13.230635 GPU-seconds.

The confirmatory evaluation assumes first use: every target query incurs semantic generation for methods that require it. A repeated identical query could later produce a cache hit, but such a scenario is separately labelled and cannot replace confirmatory cost. Terminal semantic or candidate attempts retain their consumed GPU time. Candidate terminal failures stay in the denominator and score incorrect; no failure-cost value is invented or silently dropped.
