# Wide logistic regression: profiling and optimization — 2026-10-26

This follow-up targets the wide-data costs left by the
[previous native Metal solver](native-logistic-benchmarks.md). It preserves the
objective, weights, regularization, intercept handling, and requested gradient
tolerance. All measured library fits run on the GPU.

## What profiling found and what changed

Public input validation and repeated matrix reads dominated the remaining work.
The initial instrumented 60,000 × 1,024 float32 fit spent about 6.9 ms in public
validation, and each active optimizer step read the design once for line-search
losses and again for the accepted gradient. Float64 preparation added a conversion
and range check before that second validation scan.

The implementation now addresses both costs:

- A native Metal kernel computes the directional projection, candidate loss
  differences, and the first candidate's gradient while each row's features are
  still in registers. The acceptance kernel records the actual selected step.
  When the first candidate is accepted, its gradient is reused and the backward
  kernel skips the matrix read. A different accepted candidate uses the ordinary
  backward evaluation. Failed searches and continuation batches retain their
  existing behavior.
- Feature slices span multiple SIMD groups on wider designs to limit register
  pressure. Row batches accumulate partial gradients locally before writing them,
  reducing temporary buffers and final reduction work. Tiling was selected using
  warmed complete-fit comparisons, including output reductions.
- Float32 inputs get an owned GPU copy that checks finiteness in the same pass.
  Float64 inputs reuse their private checked float32 conversion, avoiding another
  CPU validation scan and upload copy. Caller-owned inputs remain isolated;
  read-only arrays, layouts, nonfinite values, and overflow retain their validation
  behavior. Older MLX versions without `asarray` keep the explicit-copy path.

No optimizer state, training arrays, or fitted coefficients are cached across
fits. The speculative gradient is an exact evaluation of a line-search candidate
at the existing float32 precision, not an approximation or relaxed stopping rule.
The independent direct objective/gradient check remains inside every complete
fit and can restart optimization when cached convergence is premature.

## Complete fresh-fit results

The 60,000 × 1,024 float32 fit improves from **26.082 ms to 16.702 ms**, compared with **35.182 ms for stock sklearn: 2.11×**, versus the preceding report’s 1.38×. Float64 on the same shape reaches **2.24×**. Gains vary by shape; the smaller 20,000 × 1,024 float32 case reaches **1.41×**.

The following panel uses the unchanged default `tol=1e-4`. The two rows marked † fail the stricter objective-parity gate described below and are retained as failures, not presented as quality-qualified wins.

| Input | Rows × features | Previous native ms | Optimized ms | sklearn ms | vs previous | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 9.388 | 10.549 | 11.953 | 0.89× | 1.13× |
| float64 | 20,000 × 1,024 | 17.016 | 15.611 | 19.814 | 1.09× | 1.27× |
| float64 | 60,000 × 1,024 | 43.545 | 25.150 | 56.458 | 1.73× | 2.24× |
| float64 | 40,000 × 2,048 | 56.556 | 36.407 | 68.989 | 1.55× | 1.89× |
| float64 | 20,000 × 4,096 † | 74.070 | 53.677 | 118.825 | 1.38× | 2.21× |
| float64 | 20,000 × 1,024 weighted | 20.022 | 13.026 | 21.654 | 1.54× | 1.66× |
| float64 | 500,000 × 32 | 23.624 | 16.622 | 79.481 | 1.42× | 4.78× |
| float32 | 20,000 × 512 | 6.679 | 4.341 | 8.017 | 1.54× | 1.85× |
| float32 | 20,000 × 1,024 | 11.139 | 9.087 | 12.780 | 1.23× | 1.41× |
| float32 | 60,000 × 1,024 | 26.082 | 16.702 | 35.182 | 1.56× | 2.11× |
| float32 | 40,000 × 2,048 | 34.987 | 22.133 | 41.430 | 1.58× | 1.87× |
| float32 | 20,000 × 4,096 † | 58.486 | 40.697 | 65.889 | 1.44× | 1.62× |
| float32 | 20,000 × 1,024 weighted | 12.480 | 9.088 | 13.998 | 1.37× | 1.54× |
| float32 | 500,000 × 32 | 18.947 | 17.954 | 75.599 | 1.06× | 4.21× |

Raw data: [default-tolerance JSON](../benchmarks/results/wide-logistic-2026-10-26.json). All 294 timed fits finished without warnings; all 98 optimized fits used native GPU execution and reported gradient convergence. Independent float64 gradients for the 14 optimized solutions are at most **9.9613e-5**. The final source-integrity audit passed.

Runtime variability is material on some smaller cases. The float64 20,000 × 512 optimized median is 10.549 ms, with a 5.048 ms IQR and samples spanning 5.517–10.853 ms; it regresses against the frozen baseline in this run. The 20,000 × 1,024 float32 IQR is 4.335 ms. The 60,000 × 1,024 float32 IQR is 1.031 ms. All samples and spreads are retained; there is no claim of universal improvement.

## Accuracy, including a pre-existing limitation

The optimization preserves the previous solver’s numerical result: the largest relative objective increase over the frozen implementation is below **1e-10** across the default panel. It does not loosen the objective comparison, gradient tolerance, or prediction checks.

At 20,000 × 4,096, however, both previous and optimized solvers stop after 14 iterations at essentially identical objectives: **0.187850374755** and **0.187850374774**, respectively, versus sklearn’s **0.187829218252**. Both native gradients are below `1e-4`, but the relative objective gap is **1.1264e-4**, exceeding the repository policy of `1e-5`. The full default-panel audit therefore correctly exits with failure. This is an existing stopping-path limitation exposed by the expanded workload, not a precision regression from fusion. The other 12 rows pass the objective gate.

The same 4,096-feature task was rerun with **`tol=1e-5` for all three implementations**, without changing production stopping rules or audit thresholds. Both input dtypes then pass every audit:

| Input | Rows × features | Previous native ms | Optimized ms | sklearn ms | vs previous | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 4,096 | 95.554 | 71.106 | 173.566 | 1.34× | 2.44× |
| float32 | 20,000 × 4,096 | 81.119 | 58.618 | 92.532 | 1.38× | 1.58× |

Independent float64 gradients are at most **9.8272e-6**, relative objective increases are below **1.861e-6**, and prediction agreement with sklearn is **99.98%**. All 42 timed fits completed without warnings. See the [stricter-tolerance JSON](../benchmarks/results/wide-logistic-2026-10-26-strict-4096.json). These numbers must not be mixed with default-tolerance timings.

## Remaining bottlenecks

The final instrumented profile compares the same frozen package with the optimized package. Values below are stage medians in milliseconds, shown as previous → optimized. Optimizer time includes deferred centering, initialization, and direct verification.

| Input / shape | Conversion + range check | Public validation | Owned upload + finite check, optimized | Optimizer inclusive |
|---|---:|---:|---:|---:|
| float32 / 20,000 × 1,024 | 0.00 → 0.00 | 2.46 → 0.06 | 1.18 | 7.79 → 6.06 |
| float32 / 60,000 × 1,024 | 0.00 → 0.00 | 6.81 → 0.07 | 3.72 | 14.30 → 11.21 |
| float32 / 40,000 × 2,048 | 0.00 → 0.00 | 9.75 → 0.09 | 5.20 | 21.94 → 17.11 |
| float64 / 20,000 × 1,024 | 3.57 → 3.56 | 2.46 → 0.08 | 0.06 | 7.01 → 6.92 |
| float64 / 60,000 × 1,024 | 10.75 → 10.72 | 6.90 → 0.10 | 0.07 | 15.84 → 14.35 |
| float64 / 40,000 × 2,048 | 14.38 → 14.37 | 9.58 → 0.11 | 0.09 | 21.68 → 18.86 |

The large duplicate CPU validation scan is gone. Float64 conversion remains substantial: roughly 10.7 ms at 60,000 × 1,024 and 14.4 ms at 40,000 × 2,048. Float32 still pays for an independent GPU copy, and the optimizer still reads the design during initialization, accepted steps, and final verification. GPU launch and scheduling costs limit smaller problems. An exploratory exact GPU float64 cast did not show a reliable benefit and was not shipped.

See the [stage-profile JSON](../benchmarks/results/wide-logistic-profile-2026-10-26.json). Profile intervals are diagnostic; the complete-fit table comes from separate uninstrumented runs.

## Measurement method

The benchmark loads the complete previous package from a
[source-only frozen archive](../benchmarks/reference/README.md). Its source hashes
match the preceding native benchmark. This preserves the old validation,
conversion, centering, optimizer, and convergence checks as a true end-to-end
baseline. Frozen Metal function names receive a separate runtime prefix during
import; verified source files and numerical expressions remain unchanged.

Stock sklearn is unpatched. Each timed call constructs a new estimator and starts
from zero. Validation, conversion, owned input preparation, centering, optimization,
direct convergence verification, and result construction are included. Data
generation and independent float64 quality calculations run outside the timer.

Parameters are `C=1`, `tol=1e-4`, `max_iter=200`, with a fitted intercept. Seven
rotated-order repetitions follow one warmup for each variant, shape, and input
dtype. MLX is synchronized at timing boundaries. These are warmed medians;
recorded first calls are not independent process-cold measurements. Threadpoolctl
requests one thread for recognized pools and may not control Apple Accelerate.

The full panel includes 512–4,096 features, weighted data, and a tall control.
Synthetic labels depend on the first feature plus noise; these are reproducible
performance cases rather than a survey of real-world data distributions.

Environment: Apple M4 Max, macOS 26.6.2, Python 3.13.3, MLX 0.32.2,
scikit-learn 1.9.0, NumPy 2.5.2, SciPy 1.18.1.

## Reproduction

```bash
python3 -m benchmarks.logistic_wide --profile full --dtype both \
  --repeats 7 --warmup 1 --threads 1 \
  --output benchmarks/results/wide-logistic-2026-10-26.json \
  --report bench/logistic-wide-final.md
python3 -m benchmarks.logistic_wide --profile full --case wide_20k_4096 \
  --dtype both --tol 1e-5 --repeats 7 --warmup 1 --threads 1 \
  --output benchmarks/results/wide-logistic-2026-10-26-strict-4096.json \
  --report bench/logistic-wide-strict-4096.md
python3 -m benchmarks.logistic_profile --dtype both --repeats 5 --threads 1 \
  --output benchmarks/results/wide-logistic-profile-2026-10-26.json
python3 -m pytest -q
python3 -m ruff check src tests benchmarks tools
python3 tools/compliance.py
```

The stage profiler adds nested timers without extra GPU synchronization. Its
optimizer interval includes deferred centering and initialization; stage medians
need not sum to total-fit medians. Use the uninstrumented comparison for speedup
claims. Both JSON outputs record raw samples, source hashes, and the reference
archive hash.

## Verification

The full suite passed: **514 tests, 41 expected warnings, 130.40 seconds**.
Coverage includes speculative and backtracked gradients, continuation searches,
large tuned tile configurations, independent float64 gradients, nonfinite tile
tails, readonly/strided input ownership, and benchmark failure persistence.
Ruff, repository compliance, wheel/sdist builds, and strict Twine checks passed.
The original L-BFGS baseline remains byte-for-byte unchanged.

## Scope and automatic dispatch

The new matrix-pass fusion covers native binary GPU fits with 512–4,096 features.
Other native shapes retain the existing kernels. Multinomial, CPU, and unsupported
native workspaces retain the generic fused solver. Runtime Metal compilation still
adds first-call cost, and float64 inputs still require CPU conversion to the
library's documented float32 representation.

Dispatch thresholds are unchanged: the wide route requires at least 20,000 rows
and 20,000,000 elements, while the tall route requires at least 250,000 rows and
8,000,000 elements. The 20,000 × 512 benchmark forces GPU execution and remains
below the automatic crossover. These are heuristics measured on one M4 Max,
not guarantees for every dataset or Apple chip.
