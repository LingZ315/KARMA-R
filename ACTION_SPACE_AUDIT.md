# Panel C action-space audit — V3 timestamp artifact (scientific protocol v1.0.2)

Status: **PASS**. This audit concerns only frozen method definitions and code. No Panel-C target response, answer, correctness, score, or utility value was opened.

The shared eligible action space is, in fixed order: `smolvlm_incumbent`, `phi4_mm`, `granite4_vision`, `ovis25_9b`, `qwen3vl_4b`, `internvl35_4b`. The incumbent is calibrated and profiled under the same fold-local outcome boundary as every candidate.

| Method | Incumbent allowed? | Candidate routes | Query-conditioned? | Semantic features? | Action-space matched to H2? |
| ------ | -----------------: | ---------------: | -----------------: | -----------------: | --------------------------: |
| H2 | Yes, as the deployment fallback/eligible action | All 5 | Yes | Yes | Reference space |
| Static-Global (`static_calibration_global_best`) | Yes | All 5 | No | No | Yes |
| Static-Class (`static_class_conditional_best`) | Yes | All 5 | Yes | Yes | Yes |
| Logistic-Raw | Yes | All 5 | Yes | Yes | Yes |
| Nearest-Profile | Yes | All 5 | Yes | Yes | Yes |
| Incumbent-Only | Yes, exclusively | 0 | No | No | No — deliberately defines the incumbent reference |
| Cheapest-Candidate | No | All 5 | No | No | No — deliberately defines a newcomer-only cost reference; it is not treated as a strongest adaptive matched-space control |

No confirmatory simple control conditions on source ID, dataset ID, benchmark name, or a reconstructed known-source label. A future predicted-source router, if ever run after confirmatory analysis, is exploratory by definition.

## Frozen Logistic-Raw training contract

- Inputs: seven primary-class indicators, four subtype indicators, one ambiguity indicator, and a six-route one-hot vector.
- Normalization: none; all design inputs are binary indicators.
- Labels: fold-local calibration binary correctness for every eligible query–route pair.
- Model: additive linear logistic regression with an intercept.
- Regularization: L2 = 10.0 on non-intercept coefficients; the intercept is unpenalized.
- Solver: deterministic full-batch Newton/IRLS with a direct linear solve.
- Stopping: maximum 100 iterations or maximum absolute Newton step below `1e-10`.
- Class weighting: none.
- Probability tie-break: fixed shared action-space order.
- Target-driven tuning: prohibited.

## Frozen Nearest-Profile contract

- Representation: `[global profile accuracy, active primary-class accuracy, active subtype accuracy]`; when the subtype is absent or under the frozen support threshold, the third component falls back to the active class accuracy.
- Incumbent rule: identical smoothing, support threshold, and representation construction as candidate routes.
- Distance: unweighted Euclidean distance to the ideal vector `[1, 1, 1]`.
- Tie-break: smaller distance, then lower profile mean generation GPU-seconds, then fixed shared action-space order.
- Candidate pool: incumbent plus all five candidates.
- Target-driven tuning: prohibited.

## Cost comparability

For H2 and every semantic-feature control, first-use serving cost is `semantic inference + routing + selected model`. Methods without semantic classification assign semantic cost zero. Confirmatory utility assumes first use; hypothetical repeated-query cache hits may only appear in a separately labelled post-confirmatory scenario.
