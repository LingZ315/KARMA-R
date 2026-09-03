# Panel-C frozen hypotheses

## Confirmatory hypothesis H-C3

`Delta_A(C) = [sum_f sum_{i in T_f}{Y_i(H2_f,A)-Y_i(B_f,A*)}] / [sum_f |T_f|] > 0`,

where both `H2_f,A` and `B_f,A*` are accuracy-selected using only fold `f`'s non-held-out policy rows.

Statistical superiority: two-sided 95% source-stratified paired-bootstrap CI lower bound > 0.  
Practical superiority: point estimate >= 0.015.  
Confirmatory success: both conditions hold.

## Prespecified secondary hypotheses

- H-C1: pooled `H1.5-H1 > 0` (candidate-evidence value).
- H-C2: pooled `H2-H1.5 > 0` (representation value).
- H-C4: macro held-out-source `H2-H1.5 > 0` (transfer description).
- H-C5: cost-complete `U(H2_U;N,0.01)-U(B_U*;N,0.01) > 0` at at least one frozen horizon, with both sides utility-selected on fold-local policy rows.

Only H-C3 defines the primary actionability claim. Gate-4 selectors cannot be substituted into H-C3, and Gate-3 selectors cannot be substituted into H-C5. Secondary results cannot replace the primary result.
