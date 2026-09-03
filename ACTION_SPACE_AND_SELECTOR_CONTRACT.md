# Action-space and dual-selector contract

Shared route order: incumbent `smolvlm_incumbent`, then `phi4_mm`, `granite4_vision`, `ovis25_9b`, `qwen3vl_4b`, and `internvl35_4b`.

H2, Static-Global, Static-Class, Logistic-Raw, and Nearest-Profile may choose any of these six routes. Incumbent-Only is the one-route reference. Cheapest-Candidate is a five-newcomer cost reference and is not classified as an adaptive matched-space comparator. No confirmatory method conditions on source/dataset ID.

Gate 3:

- H2 ranking: accuracy, utility, lower serving cost, fixed margin order.
- Simple-control ranking: accuracy, utility, lower serving cost, fixed method order.
- Target estimand: accuracy of `H2_A` minus accuracy of `B_A*`.

Gate 4:

- H2 ranking: utility, accuracy, lower serving cost, fixed margin order.
- Simple-control ranking: utility, accuracy, lower serving cost, fixed method order.
- Target estimand: utility of `H2_U` minus utility of `B_U*`.

The frozen margin order is `[0.0, 0.0025, 0.005, 0.01, 0.02]`. The frozen method order is `[incumbent_only, cheapest_candidate, static_calibration_global_best, static_class_conditional_best, logistic_raw, nearest_profile]`. Policy differences across gates are expected and must be reported; cross-gate substitution is prohibited.

Logistic-Raw and Nearest-Profile settings are specified in `HYPERPARAMETER_PROVENANCE.md`, the analysis contract, and executable `controls.py`. All target routes are computed only after the complete five-fold policy bundle is frozen.
