# Recheck of the three original tall logistic datasets

The completed native GPU implementation improves all three timings from the
earlier table. These measurements follow the full **1,118-test** pass and the
ill-conditioned, weighted and rank-deficient validation described in the
[solver report](rank-safe-newton-performance-2026-10-27.md).

## Complete fresh-fit timings

Apple M4 Max, MLX 0.32.2, scikit-learn 1.9.0, float32 input, seed 41,
`C=1`, `tol=1e-4`, `max_iter=200`, with an intercept. Each implementation
received one warmup and seven measured fits in the original rotating order.
No thread limit was imposed, matching the earlier screenshot's run. These
are medians in milliseconds; compilation is warmed, but validation, input
preparation, conversion, optimization and returned parameters are included
in every fresh fit. No fitted state or prepared input is reused.

| Dataset | Earlier screenshot's native time | Current mlx-learn | Stock sklearn, rerun | sklearn / current |
| --- | ---: | ---: | ---: | ---: |
| 500,000 × 32 | 15.437 | **13.864** | 72.671 | **5.24×** |
| 500,000 × 128 | 29.144 | **23.957** | 72.991 | **3.05×** |
| 60,000 × 1,024 | 25.709 | **13.191** | 34.775 | **2.64×** |

The earlier native values are historical observations, not a reexecution of
that old native implementation. Current times are 10.2%, 17.8% and 48.7%
lower, respectively, than those observations. Fresh CPU timings are shown
alongside the current GPU timings so the speedups do not mix run dates.

The earlier fused algorithm was also rerun, with current shared validation,
upload and centering. This is an algorithm control, not a frozen historical
checkout. Native selection is disabled consistently in both the estimator's
input-preparation stage and the solver. GPU execution is forced and audited
for both mlx-learn variants; stock sklearn is unpatched.

| Dataset | Fused control ms | Current ms | Fused / current | Current IQR ms | sklearn IQR ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500,000 × 32 | 83.967 | 13.864 | 6.06× | 2.030 | 1.564 |
| 500,000 × 128 | 93.330 | 23.957 | 3.90× | 1.204 | 1.070 |
| 60,000 × 1,024 | 30.294 | 13.191 | 2.30× | 0.855 | 0.405 |

All current native fits used seven accepted iterations. The first two datasets use
the existing narrow GPU route; the third uses the optimized wide native path.
The new 256-thread direction recurrence and guarded Newton refinement target
wider inputs and do not activate for these three feature counts. These numbers
therefore measure the cumulative implementation against the screenshot, not
the isolated effect of the latest Newton change.

## Accuracy and audit

Every one of the 63 measured fits was independently scored on original input
values with float64 arithmetic after all timing for its case. Additional
untimed float64 sklearn references used `tol=1e-10`, `max_iter=3000`; they
converged without warnings, with gradients below `4.37e-10`.

| Dataset | Current relative objective gap to tight reference | Current max gradient | Current prediction agreement with tight reference |
| --- | ---: | ---: | ---: |
| 500,000 × 32 | 2.15e-7 | 5.16e-5 | 99.9986% |
| 500,000 × 128 | 2.40e-7 | 5.37e-5 | 99.9982% |
| 60,000 × 1,024 | 1.96e-6 | 3.94e-5 | 99.9683% |

All variants passed the `1e-5` relative objective gate, requested `1e-4`
gradient tolerance, prediction agreement checks, warning checks and actual
backend/implementation checks. The current and stock models have different
achieved errors within those thresholds; passing them does not assert equal
coefficients or exact equal accuracy. Source hashes matched before and after
the run.

The first attempt completed the two narrow cases, then stopped because the
original fused-control harness did not disable the newer native-only input
statistics. The benchmark adapter was corrected and all three cases rerun.
The [incomplete artifact](../benchmarks/results/tall-logistic-recheck-incomplete-2026-10-27.json)
is retained; it is not used in the final timing table. No library change was
needed for this adapter correction.

## Reproduction

```bash
python -m benchmarks.tall_logistic_recheck --repeats 7 --warmup 1 \
  --output benchmarks/results/tall-logistic-recheck-2026-10-27.json
```

The [full results](../benchmarks/results/tall-logistic-recheck-2026-10-27.json)
include every timed sample, warmup, execution order, fit state, warning,
independent score, fixture hash, environment and source hash. The original
data generator and fit timer are reused from `benchmarks/logistic_native.py`.
The historical [native benchmark report](native-logistic-benchmarks.md) and
[original raw results](../benchmarks/results/native-logistic-2026-10-26.json)
remain unchanged.
