# Panel C pre-outcome power and precision analysis

Status: **frozen locally; external timestamp pending; no Panel-C outcomes inspected**.

## Design target

The primary endpoint is the pooled paired difference `H2_f - B_f*` over 7,474 explicit-source-metadata-blind, held-out-source target queries, with `B_f*` selected independently from fold-local non-held-out policy rows. A practically meaningful action gain is fixed at **1.5 percentage points**. The confirmatory success rule additionally requires the two-sided 95% paired-bootstrap interval to exclude zero.

## Assumptions and calculation

For each query in fold `f`, let `D` be +1 when only `H2_f` is correct, -1 when only `B_f*` is correct, and 0 otherwise. If `d` is the fraction of discordant pairs and `delta=E[D]`, then `Var(D)=d-delta^2`. The normal-approximation requirement is

`N = ceil((z_(1-alpha/2)+z_power)^2 * (d-delta^2) / delta^2)`

with two-sided alpha=0.05 and power=0.80. At the planning value `d=0.20`, detecting 1.5 pp requires **6,969** paired queries. The frozen cohort of 7,474 therefore meets this planning target without outcome-dependent resizing.

At N=7,474, the approximate detectable differences are 1.02 pp for 10% discordance, 1.45 pp for 20%, and 1.77 pp for 30%.

## Sensitivity table

| Discordant fraction | Detect 1.0 pp | Detect 1.5 pp | Detect 2.0 pp |
|---:|---:|---:|---:|
| 10% | 7,842 | 3,481 | 1,955 |
| 20% | 15,690 | 6,969 | 3,917 |
| 30% | 23,539 | 10,458 | 5,879 |


## Interpretation limits

The calculation concerns the pooled paired endpoint and does not convert five source folds into a large source-level sample. Macro-source means, medians, signs, ranges, and heterogeneity are mandatory secondary summaries, but inference across only five sources is necessarily imprecise. The cohort size and source set may not be changed after the external timestamp in response to interim significance.
