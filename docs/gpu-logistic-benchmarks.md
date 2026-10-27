# Fused GPU logistic regression

The development version keeps coefficients, gradients and L-BFGS curvature
history on the selected MLX device. `mx.compile` fuses elementwise regions in
the direction recursion, trial/update steps, loss and gradient. Reductions and
matrix multiplications remain separate GPU kernels. Adaptive strong-Wolfe
decisions and final float64 accumulation of partial loss sums remain on the
host. This implementation does not use a CPU optimizer or SciPy optimization.

The two-loop recursion previously synchronized each dot product to Python,
including 21 dependent scalar reads with a full history of 10. The fused path
evaluates the direction and a small statistics vector together. Line-search
trials evaluate loss, gradient and directional derivative together, and accepted
steps evaluate curvature and convergence statistics together. Curvature
denominators stay on device to avoid rounding a host reciprocal to float32.

The objective remains sklearn's weighted L2/unpenalized logistic objective.
Signed binary logits preserve small positive-class losses and gradients;
explicitly shifted multiclass logits avoid cancellation from subtracting large
raw scores. Exact unit weights select the unweighted graph. An absent penalty
does not square coefficients, avoiding `inf * 0` on large finite coefficients.
Centering, float32 policy, convergence warnings and backend contracts remain.

## Complete-fit results

Measured 26 October 2026 on Apple M4 Max, 128 GiB; Python 3.13.3,
MLX 0.32.2, sklearn 1.9.0, NumPy 2.5.2 and SciPy 1.18.1. NumPy uses Apple
Accelerate; the detected OpenMP pool uses its default 16 threads. No thread limit
was requested. These measurements are specific to this machine and workload.

Seven repetitions after one warmup per variant/case; fresh models; rotated and
alternating execution order; synchronized complete fits, including validation,
input conversion and final model attributes. Scoring is outside the timer.
The legacy and fused MLX variants are explicitly forced to the GPU and assert
their actual backend. The original solver **and objective** are loaded from
revision `fb79a1b11dad401e82808885884fc633da9c3e86`; the shared original L-BFGS
source is verified byte-for-byte. The current estimator's unchanged numerical
validation surrounds both versions.

All cases use `C=1`, `tol=1e-4`, `max_iter=200`, seed 41. Multiclass uses five
classes and nonuniform sample weights. The ill-conditioned case introduces
correlated and unequally scaled features. Each dtype panel contains 168 timed
fits, including the separate default-patched variant: 336 timed fits total.

### Float64 public input — median milliseconds

| Samples × features | Case | Original MLX | Fused GPU | sklearn | Gain over original |
|---|---|---:|---:|---:|---:|
| 5,000 × 32 | binary | 23.24 | **8.97** | 1.40 | **2.59×** |
| 20,000 × 128 | binary | 26.52 | **11.86** | 5.24 | **2.24×** |
| 20,000 × 1,024 | binary | 68.92 | **54.70** | 20.62 | **1.26×** |
| 60,000 × 1,024 | binary | 130.48 | **116.89** | 56.28 | **1.12×** |
| 20,000 × 128 | weighted multiclass | 45.16 | **15.23** | 25.53 | **2.97×** |
| 5,000 × 32 | ill-conditioned binary | 64.63 | **17.63** | 3.49 | **3.67×** |

### Float32 public input — median milliseconds

| Samples × features | Case | Original MLX | Fused GPU | sklearn | Gain over original |
|---|---|---:|---:|---:|---:|
| 5,000 × 32 | binary | 22.22 | **8.99** | 1.67 | **2.47×** |
| 20,000 × 128 | binary | 31.52 | **16.81** | 4.09 | **1.87×** |
| 20,000 × 1,024 | binary | 60.40 | **39.05** | 13.85 | **1.55×** |
| 60,000 × 1,024 | binary | 70.22 | **54.11** | 35.80 | **1.30×** |
| 20,000 × 128 | weighted multiclass | 42.18 | **12.82** | 16.79 | **3.29×** |
| 5,000 × 32 | ill-conditioned binary | 67.62 | **17.83** | 3.54 | **3.79×** |

The fused GPU path wins against the original in every measured case. It also
beats sklearn on weighted multiclass, by 1.68× with float64 inputs and 1.31× with
float32 inputs. sklearn remains faster on all measured binary cases. These are
ratios of warmed medians, not statistical significance claims or universal
crossover estimates. The two dtype panels run sequentially; compare implementations
within each panel rather than attributing every between-panel difference to dtype.

## Quality and startup costs

Every timed fit completed without warnings. Both MLX implementations converged
by gradient tolerance with identical accepted iteration counts for all six
cases: 8, 8, 10, 8, 12 and 22. The maximum absolute difference between their
independently evaluated float64 objectives across both panels was **8.47e-11**.
The raw results also include float64 gradients, prediction agreement with
sklearn, coefficient differences and per-fit solver state; iteration counts
alone are not treated as proof of equal solution quality.

Compilation specializes on input shape/device and history length. First calls
are recorded separately; they include compilation/initialization where needed
and are **not** fresh-process cold-start measurements. For example, the first
float64 5k × 32 fused fit took 20.12 ms versus its 8.97 ms warmed median. For
60k × 1,024, the first fused fit took 144.00 ms versus 116.89 ms warmed. A
one-off fit should not be assumed to achieve the warmed number.

## Dispatch limitation

This change retains the existing crossover thresholds. In the separately timed
patched-default panel, the weighted multiclass workload still falls back to
sklearn even though the forced GPU wins. Wide binary cases select MLX even
though sklearn wins on this machine. The sample/feature thresholds need a broader,
class-aware calibration; these six cases do not justify a universal replacement.
Direct imports also use the internal MLX CPU path below those thresholds.
`device="gpu"` alone does not override the crossover; the diagnostic runner
uses the explicit private `force_backend="mlx"` override.

## Reproduction and evidence

Final validation on the environment above: **305 tests passed** (41 warnings,
including deprecations and deliberately truncated convergence cases) in 126.53 s.
Ruff, all mechanical compliance gates, source-hash verification against the
benchmarked implementation, source-distribution/wheel builds, and strict Twine
package validation passed. The wheel remains pure Python. This was a local full
suite run, not a rerun of every Python/sklearn combination in the CI matrix.

```sh
python3 -m benchmarks.logistic_gpu --profile full --dtype float64 \
  --output bench/gpu-float64.json --report bench/gpu-float64.md
python3 -m benchmarks.logistic_gpu --profile full --dtype float32 \
  --output bench/gpu-float32.json --report bench/gpu-float32.md
python3 -m pytest -q --durations=25
python3 -m ruff check src tests tools benchmarks
python3 tools/compliance.py
```

The benchmark requires a Git checkout containing the baseline revision. It
temporarily selects baseline code only inside its standalone process; production
code has no benchmark switches or mutable solver override.

- [Benchmark runner](../benchmarks/logistic_gpu.py)
- [Float64 raw samples, quality and environment](../benchmarks/results/gpu-logistic-2026-10-26-float64.json)
- [Float32 raw samples, quality and environment](../benchmarks/results/gpu-logistic-2026-10-26-float32.json)
- [Fused optimizer](../src/mlxlearn/_solvers/fused_lbfgs.py)
- [Fused objective](../src/mlxlearn/_solvers/logistic.py)
- [Optimizer contracts](../tests/unit/test_fused_lbfgs.py)
- [Objective precision regressions](../tests/unit/test_logistic_objective.py)

JSON includes raw repetitions, per-fit warnings and dispatch, source hashes,
baseline revision, software/hardware metadata, thread-pool configuration, first
calls and independent quality measurements.
