# Panel-C hyperparameter provenance

Protocol version: 1.0.2  
Status: **READY FOR EXTERNAL IMMUTABLE TIMESTAMP; NO PANEL-C TARGET INFERENCE OR OUTCOME ACCESS**

Every data-selected value is isolated within its outer held-out-source fold. No pooled cross-fold policy outcome may select a primary comparator or H2 parameter.

| Parameter | Value/range | Fixed or selected | Selection split | Fold-specific? | Target access? |
|---|---|---|---|---:|---:|
| H2 action space | Incumbent plus five frozen candidates | Fixed | None | No | No |
| Strong adaptive simple-control action space | Incumbent plus five frozen candidates | Fixed | None | No | No |
| L2 regularization | 10.0 | Fixed | None | No | No |
| L2 iterations | 100 | Fixed | None | No | No |
| L1 regularization / iterations | 1.0 / 100 | Fixed sensitivity | None | No | No |
| L3 hidden / epochs / learning rate / L2 / seed | 16 / 400 / 0.02 / 0.01 / 2026082810 | Fixed sensitivity | None | No | No |
| Logistic-raw solver / normalization / weights | Full-batch Newton-IRLS / none / no class weights | Fixed | None | No | No |
| Logistic-raw L2 / iterations / tolerance | 10.0 / 100 / 1e-10 maximum step | Fixed | None | No | No |
| Global smoothing prior | Beta(1,1) | Fixed | None | No | No |
| Conditional prior strength | 2 pseudo-observations | Fixed | None | No | No |
| Minimum class/subtype support | 25 | Fixed | None | No | No |
| Unsupported-cell fallback | Candidate-specific global profile | Fixed | Profile application | Yes, estimated per fold | No |
| Gate-3 H2 accuracy margin | {0, 0.0025, 0.005, 0.01, 0.02}; accuracy/utility/cost/fixed order | Selected | Fold-local policy | Yes | No |
| Gate-4 H2 utility margin | Same grid; utility/accuracy/cost/fixed order | Selected | Fold-local policy | Yes | No |
| Cheapest-candidate identity | All frozen candidates | Selected | Fold-local profile cost | Yes | No |
| Static global route | Incumbent plus all frozen candidates | Selected | Fold-local calibration | Yes | No |
| Static class mapping | Frozen classes; support fallback 25 | Selected | Fold-local calibration | Yes | No |
| Nearest-profile rule | Euclidean distance to [1,1,1], then cost and fixed route order | Fixed | None | No | No |
| Gate-3 `B_f,A*` | Six simple-control families; accuracy/utility/cost/fixed order | Selected | Fold-local policy | Yes | No |
| Gate-4 `B_f,U*` | Six simple-control families; utility/accuracy/cost/fixed order | Selected | Fold-local policy | Yes | No |
| Primary bootstrap | 10,000 draws; PCG64 seed 2026082811 | Fixed | None | No | No |
| Practical threshold | +1.5 percentage points | Fixed | None | No | No |
| Utility coefficient | 0.01 per GPU-second/query | Fixed | None | No | No |

The source/dataset identifier is available only to orchestration code that constructs folds and reports source-level summaries. It is excluded from every router feature matrix.
