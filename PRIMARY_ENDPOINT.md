# Panel C frozen Gate-3 and Gate-4 endpoints

For outer fold `f`, let `T_f` be its held-out-source target set. `H2_f,A` is selected on fold-local non-target policy rows by accuracy; `B_f,A*` is selected from the six simple controls by the same primary objective and mirrored deterministic tie-breaks.

The sole primary endpoint is query-weighted accuracy difference:

`Delta_A(C) = [sum_f sum_{i in T_f}{Y_i(H2_f,A)-Y_i(B_f,A*)}] / [sum_f |T_f|]`.

The target contains 7,474 unique queries. Each contributes exactly one paired Gate-3 contrast. The two-sided 95% source-stratified paired-bootstrap interval uses 10,000 draws and NumPy PCG64 seed 2026082811. Statistical positivity requires the lower bound to exceed zero; practical meaning separately requires the point estimate to be at least +0.015. The combined actionability rule requires both.

Required secondary summaries are the unweighted mean fold effect, median fold effect, number of positive/negative/zero folds, and every fold estimate. These do not replace the query-weighted primary endpoint.

Gate 4 is separately selected and evaluated:

`Delta_U(C) = U(H2_U;N,0.01)-U(B_U*;N,0.01)`.

`H2_U` and `B_U*` maximize fold-local policy utility with mirrored tie-breaks. They cannot be substituted into Gate 3. Likewise, `H2_A` and `B_A*` cannot be promoted into Gate 4 because target behaviour looks favourable. If the policies differ, the difference is reported as an expected consequence of different estimands.
