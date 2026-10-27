# Logistic input preparation and GPU scheduling — 2026-10-26

This follows the [wide-kernel optimization](wide-logistic-benchmarks.md), targeting
the conversion and submission costs remaining at 20,000 × 1,024. The comparison
uses a frozen copy of that complete previous implementation, including its
already optimized validation and wide matrix kernels.

## Changes retained

Wide GPU inputs now enter an owned centered buffer through two native Metal
passes. The first checks original-input finiteness and accumulates a shifted,
quarter-scaled mean. The second writes the actual centered values and, for native
binary fits, accumulates the initial gradient and curvature in the same pass.
The solver reuses those statistics with its existing normalization and scalar
initial metric. Computing them from centered values avoids cancellation from
subtracting large raw squared moments.

This removes a separate full-size copy for caller-owned float32 data and a later
initialization read. The exact applied center accompanies the buffer so that
intercept restoration stays consistent. Borrowed NumPy storage remains alive
through GPU evaluation and context cleanup, including failure paths. The model
retains no input buffers after fit.

Large native-endian, contiguous float64 inputs use ordinary NumPy conversion in
disjoint blocks with up to eight temporary CPU workers. Finiteness is checked
while each block is in cache. Every float32 output bit matches NumPy's cast;
custom underflow handling, smaller inputs, and other layouts retain serial
conversion. Workers join before return, including on exceptions. Centering and
optimization remain on the GPU, at the existing float32 precision.

The public estimator parameters and default tolerance are unchanged. No fitted
state or input arrays are cached between calls. Direct objective and gradient
verification remains inside each fit. The new centering accumulation can differ
by float32 rounding, so validation checks the resulting objective and gradient
rather than requiring every fitted coefficient to be bit-identical.

## Profiling and remaining costs

The diagnostic stage profile uses seven fresh fits after three warmups. CPU conversion/range-check medians fall from **3.531 to 1.678 ms** at 20,000 × 1,024 float64. The table also groups upload, centered preparation, solver, and cleanup intervals together, avoiding an apparent gain caused only by moving initialization outside the optimizer timer.

| Input / shape | Conversion + range check, previous → new ms | GPU preparation + solver + cleanup, previous → new ms |
|---|---:|---:|
| float32 / 20,000 × 1,024 | 0.001 → 0.001 | 7.185 → 6.762 |
| float32 / 60,000 × 1,024 | 0.002 → 0.002 | 14.708 → 12.019 |
| float32 / 40,000 × 2,048 | 0.002 → 0.002 | 21.312 → 17.758 |
| float64 / 20,000 × 1,024 | 3.531 → 1.678 | 6.544 → 6.461 |
| float64 / 60,000 × 1,024 | 10.799 → 4.428 | 13.949 → 12.511 |
| float64 / 40,000 × 2,048 | 14.352 → 5.746 | 19.300 → 17.921 |

These are diagnostic timings, not the source of the speedup ratios. GPU preparation still performs necessary full-matrix reads and an owned write. The solver still needs its iterative matrix passes and final independent verification. See the [complete stage profile](../benchmarks/results/overhead-logistic-profile-2026-10-26.json).


The large matrix pass dominates each optimizer step. Diagnostic component
measurements put it near 170 µs in an amortized GPU batch, while direction,
acceptance, individual reductions, inactive fallback, and tail operations each
take roughly 8–25 µs. These separately scheduled stage timings are not additive
and are not complete-fit speedups.

Compiling four-step chunks, replacing gradient finalization with another custom
kernel, and eight alternate thread-group layouts produced no dependable
complete-fit improvement, so they were not retained. Exact integer-based GPU
float64 casting, owned-copy alternatives, and smaller conversion worker pools
also failed to resolve the timing spikes.

There is a framework submission limit: MLX 0.32.2 defaults to 50 operations or
50 MiB per command buffer on Max/Ultra chips, while this float32 matrix alone is
78.125 MiB. A matrix-touching primitive can therefore trigger a commit even
inside a multi-step Python batch. Fusion can reduce those boundaries; the public
API has no per-fit threshold or queue-priority setting. The library does not
change process-wide MLX, Python, or BLAS scheduling settings.
[MLX command-buffer implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L477-L588).

## Complete-fit comparison

The final 20,000 × 1,024 float32 median is **7.647 ms versus 12.884 ms for sklearn: 1.68×**. Against the frozen previous implementation measured in the same run (8.414 ms), the gain is **1.10×**. This improves the targeted case, but does not make it a large GPU win. Float64 on that shape reaches **1.63× sklearn**, with a modest **1.05×** median gain over the frozen version.

The larger 60,000 × 1,024 float64 case reaches **3.15× sklearn**, and 40,000 × 2,048 reaches **2.83×**. Complete default-tolerance results follow; † marks the retained objective-parity failures discussed below.

| Input | Rows × features | Previous wide ms | Optimized ms | sklearn ms | vs previous | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 6.665 | 5.781 | 11.850 | 1.15× | 2.05× |
| float64 | 20,000 × 1,024 | 12.888 | 12.298 | 20.074 | 1.05× | 1.63× |
| float64 | 60,000 × 1,024 | 26.113 | 18.016 | 56.763 | 1.45× | 3.15× |
| float64 | 40,000 × 2,048 | 34.660 | 24.291 | 68.736 | 1.43× | 2.83× |
| float64 | 20,000 × 4,096 † | 53.868 | 43.835 | 121.366 | 1.23× | 2.77× |
| float64 | 20,000 × 1,024 weighted | 12.938 | 12.128 | 21.699 | 1.07× | 1.79× |
| float64 | 500,000 × 32 | 17.301 | 17.160 | 79.425 | 1.01× | 4.63× |
| float32 | 20,000 × 512 | 4.225 | 3.829 | 7.985 | 1.10× | 2.09× |
| float32 | 20,000 × 1,024 | 8.414 | 7.647 | 12.884 | 1.10× | 1.68× |
| float32 | 60,000 × 1,024 | 16.255 | 13.740 | 35.891 | 1.18× | 2.61× |
| float32 | 40,000 × 2,048 | 21.114 | 17.846 | 41.404 | 1.18× | 2.32× |
| float32 | 20,000 × 4,096 † | 40.900 | 37.279 | 65.093 | 1.10× | 1.75× |
| float32 | 20,000 × 1,024 weighted | 8.552 | 8.065 | 13.959 | 1.06× | 1.73× |
| float32 | 500,000 × 32 | 15.990 | 16.144 | 77.412 | 0.99× | 4.80× |

Raw results: [final 30-repeat panel](../benchmarks/results/overhead-logistic-2026-10-26-final.json). All rows are included, including the approximately 1% median regression on the float32 tall control. Its mean improves; this is small relative to its timing spread.

For the targeted 20,000 × 1,024 case, quartiles (NumPy percentiles) and means make the variability explicit:

| Input / implementation | Q1 ms | Median ms | Q3 ms | Mean ms |
|---|---:|---:|---:|---:|
| float64 / previous wide | 11.225 | 12.888 | 18.155 | 14.418 |
| float64 / optimized | 9.620 | 12.298 | 16.415 | 12.977 |
| float64 / sklearn | 19.779 | 20.074 | 20.642 | 20.408 |
| float32 / previous wide | 7.589 | 8.414 | 10.468 | 9.340 |
| float32 / optimized | 6.869 | 7.647 | 8.617 | 8.245 |
| float32 / sklearn | 12.781 | 12.884 | 13.048 | 12.925 |

The optimized implementation wins **25 of 30 paired repetitions in each dtype**. Float64 remains bimodal, so the ratio of medians and the ratio of means differ: the mean improvement over the previous version is 1.11× for float64 and 1.13× for float32. These are supplementary statistics; the headline ratios use the same median convention as preceding reports.

The [initial 15-repeat run](../benchmarks/results/overhead-logistic-2026-10-26.json) remains available. It measured a float64 median regression at the targeted shape despite faster isolated conversion, which prompted the additional ordering and scheduling investigation. The final panel was extended to five complete cycles of all six variant permutations; no samples were filtered or selected by speed.


## Accuracy and verification

All **480 optimized measured fits** across the default and strict panels used native Metal execution, reported gradient convergence, and emitted no warnings. Independent float64 gradients are below the requested tolerance in every case. Both final source-integrity audits pass.

The largest relative objective increase over the frozen wide implementation is below **5.5e-11** in the default panel. Thus the retained changes preserve its numerical quality; neither convergence tolerance nor objective/label checks were relaxed.

The expanded 4,096-feature case retains the **pre-existing default-tolerance limitation**: both old and new solvers satisfy `tol=1e-4`, but their relative objective gap to sklearn is about **1.1264e-4**, above the repository policy of `1e-5`. Both marked default rows therefore fail the quality audit, and the runner exits nonzero. The other 12 default rows pass.

At **`tol=1e-5` for all three implementations**, the 4,096-feature panel passes every audit:

| Input | Rows × features | Previous wide ms | Optimized ms | sklearn ms | vs previous | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 4,096 | 71.045 | 60.723 | 170.767 | 1.17× | 2.81× |
| float32 | 20,000 × 4,096 | 57.756 | 54.362 | 90.813 | 1.06× | 1.67× |

Independent gradients are at most **9.8272e-6**, the relative objective increase is below **1.861e-6**, and prediction agreement with sklearn is **99.98%**. See the [strict final panel](../benchmarks/results/overhead-logistic-2026-10-26-final-strict-4096.json). These timings are separate from the default-tolerance table.


The full suite passed: **583 tests, 41 warnings, 128.50 seconds**.
Ruff, repository compliance, wheel/sdist builds, and strict Twine checks passed.
The original L-BFGS baseline remains byte-for-byte unchanged.

New tests cover NumPy rounding ties, subnormals, NaN payloads, overflow,
thread cleanup, error callbacks, readonly/strided input ownership, nonfinite
tile tails, initialization equivalence on the actual centered values, complete
weighted fits, and GPU-input cleanup after solver failures. Benchmark tests also
verify isolated loading of lazy frozen kernels and persistence of failing rows.

## Method and reproduction

These are dense synthetic binary problems with seed 41: independent standard-normal
features, labels from `X[:, 0] + N(0, 0.5) > 0` (0.5 is the noise standard
deviation), and uniform sample weights in `[0.25, 3.0)` for the weighted case.
They isolate shape and input-preparation costs; these speedups do not establish
performance across real datasets or differently conditioned problems.

The [frozen reference](../benchmarks/reference/wide-logistic-2026-10-26.zip) contains
all 35 source files matching the previous wide benchmark. The loader validates
its archive and individual file hashes, then gives every reference Metal kernel
a separate namespace. The raw results record current source hashes and recheck
source integrity when the run ends.

Each timed call constructs a fresh estimator and fits from zero. Validation,
conversion, CPU worker creation/join, GPU preparation, optimization, direct
verification, cleanup, and output construction are included. Data generation
and independent float64 quality measurements are outside the timer. Stock
sklearn is unpatched; both library variants are forced onto their native GPU
solver, including cases below the normal automatic crossover.

Final runs use **30 repetitions after three warmups** per variant/case/dtype.
The six variant permutations each occur five times. These are warmed medians;
recorded first calls are not independent process-cold measurements. The initial
15-repeat run remains saved separately rather than being overwritten.
Benchmark stdout and stderr were redirected to log files during measurement.

The machine was not exclusively reserved: other CPU and graphics work remained
active. Fits can fall into substantially different timing modes, particularly
after sklearn. Additional 30-repeat comparisons of original centering, fused
centering, CPU worker counts, and GPU conversion did not eliminate this effect;
profiling located the variation across GPU evaluations, not in conversion or
the final ownership drain. The root cause is not established. Reported spreads
and all raw samples remain part of the evidence.

Environment: Apple M4 Max, macOS 26.6.2, Python 3.13.3, MLX 0.32.2,
scikit-learn 1.9.0, NumPy 2.5.2, SciPy 1.18.1. Threadpoolctl requests one thread
for recognized pools; it does not control Apple Accelerate or the explicit
NumPy input-conversion workers. Parameters: `C=1`, `max_iter=200`, fitted
intercept, and `tol=1e-4` except for the separately reported strict 4,096-feature
panel, where all three implementations use `tol=1e-5`.

```bash
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1 \
  --output benchmarks/results/overhead-logistic-2026-10-26-final.json \
  --report bench/overhead-logistic-final.md \
  > bench/overhead-logistic-final.log 2>&1
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip \
  --profile full --case wide_20k_4096 --dtype both --tol 1e-5 \
  --repeats 30 --warmup 3 --threads 1 \
  --output benchmarks/results/overhead-logistic-2026-10-26-final-strict-4096.json \
  --report bench/overhead-logistic-final-strict-4096.md \
  > bench/overhead-logistic-final-strict-4096.log 2>&1
python3 -m benchmarks.logistic_profile \
  --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip \
  --dtype both --repeats 7 --warmup 3 --threads 1 \
  --output benchmarks/results/overhead-logistic-profile-2026-10-26.json
python3 -m pytest -q
python3 -m ruff check src tests benchmarks tools
python3 tools/compliance.py
```

The profiler's context-entry interval includes centering and native initialization;
the preceding implementation performs those inside its optimizer interval.
Comparing optimizer timers alone would therefore overstate the gain. Its nested
timers add no extra GPU synchronization, and its results are diagnostic rather
than the source of the complete-fit speedup claims.
