# Panel-C dataset and split specification

## Fixed sources and targets

| Held-out source | Target N |
|---|---:|
| IconQA | 2,200 |
| MME | 515 |
| TallyQA | 2,200 |
| VisOnlyQA | 359 |
| WeMath2.0-Standard | 2,200 |

Pooled N is 7,474. Each source is target once. When source `s` is held out, none of its rows may enter profile, calibration, or policy. Other sources use their deterministic 20%/40%/40% profile/calibration/policy partitions. Cross-fold reuse is expected but may not influence the policy evaluated when that query's own source is held out.

The public ID manifest contains no question, answer, image, provider record ID, model output, or correctness. Private source identity exists only for orchestration and audit; it is excluded from router requests and feature matrices. No content is manually sanitized, so latent source cues may remain.

ChartMuseum (120 eligible rows) and TableVQA-Bench (0) did not meet the frozen minimum of 300 and are excluded without outcomes. Raw data remain under provider terms and are not redistributed.
