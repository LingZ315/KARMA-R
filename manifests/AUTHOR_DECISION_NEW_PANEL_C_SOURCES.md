# Author decision: optional new Panel-C sources

Status: **DECISION FROZEN — RETAIN CURRENT FIVE SOURCES**

| Current source | Target N | Used in an earlier KARMA-R panel? | Current use boundary |
|---|---:|---:|---|
| IconQA | 2,200 | Yes, Replication B | Local evaluation only; no raw redistribution |
| MME | 515 | Yes, Replication B | Local evaluation only; no raw redistribution |
| TallyQA | 2,200 | Yes, Replication B | Local evaluation only; no raw redistribution |
| VisOnlyQA | 359 | Yes, Replication B | Provider terms; retained card declared GPL-3.0 |
| WeMath2.0-Standard | 2,200 | Yes, Replication B | Local evaluation only; no raw redistribution |

The overlap with Replication B limits claims about entirely novel benchmark families but does not invalidate nested held-out-source evaluation. Each source is absent from its own fold's profile, calibration, and policy data.

## Locally implemented alternatives

| Candidate adapter | Eligible unseen exact-image groups after frozen filters | License/scoring compatibility | Outcome-independent assessment |
|---|---:|---|---|
| ChartMuseum | 120 | Card declared CC-BY-SA-4.0; frozen normalized/numeric scorer available | Below the predeclared minimum of 300 |
| TableVQA-Bench | 0 | Card declared CC-BY-4.0; frozen short-answer scorer available | No eligible historically unseen exact-image group |

No other locally frozen adapter had a documented outcome-independent eligibility audit suitable for silent addition. GPU-seconds are not estimated because no Panel-C timing run has occurred. For any new source with `N_new` rows, the incremental execution burden would be one Qwen semantic pass and six route passes per row, plus fold-local onboarding; it must be measured under the frozen timing method rather than guessed.

Decision: do not add a source before the v1.0.2 immutable timestamp. The scientific benefit of a genuinely new family would be stronger external-validity evidence, but neither implemented option meets the predeclared eligibility rule. A later addition must be chosen without H2 results and requires a new preregistration version and timestamp.
