# Native Metal logistic regression — 2026-10-26

Binary logistic regression now uses handwritten Metal C++ GPU kernels. On the Apple M4 Max, the largest measured improvement over stock scikit-learn is **4.86×** for 500,000 rows × 32 features with float32 input; the same float64 case improves by **4.26×**. The native optimizer is up to **5.56×** faster than the previous fused implementation. Wide binary cases improve by 1.21–1.55× over scikit-learn. Small and ill-conditioned 5,000 × 32 cases still favor scikit-learn.

These are **warmed, complete fresh-fit medians**, including estimator construction, validation, conversion, centering, optimization, convergence verification, and output construction. They are not GPU-kernel-only timings. Every fit starts from zero.

## Implementation

- A native SIMD-group kernel maintains the L-BFGS ring history and computes directions without Python scalar transfers for each history dot product. The initial Hessian trace supplies a scalar step scale, preserving the row-space behavior of redundant designs.
- Direction projection and stable loss differences share a Metal kernel. Eight Armijo candidates run together; only an unsuccessful batch triggers the remaining search budget. Coefficients, cached logits, residuals, and acceptance state stay on the GPU.
- Tiled gradient reductions read adjacent features together. Frozen steps skip matrix reads. The host checks the updated gradient after at most four steps, preventing an unnecessary extra chunk after convergence.
- A separate native evaluation recomputes logits, loss, and gradient from the returned coefficients. It does not trust cached logits for convergence. Partial losses finish summation in host float64; an inconsistent cached convergence resets history and resumes within the remaining iteration budget.
- A shifted, scaled GPU mean replaces CPU centering. The returned intercept uses the actual float32 shift. Validation reuses a checked float32 cast for ordinary NumPy float64 input, and binary integer label encoding avoids sorting all rows.

The objective, regularization scaling, sample/class weights, unpenalized intercept, and gradient tolerance retain their existing meanings. Armijo replaces strong Wolfe only in the native path. CPU, multinomial, and unsupported native workspaces retain the generic fused solver. No CPU optimizer, SciPy runtime dependency, fitted-state cache, or compiled binary is added to the package; MLX compiles the Metal source at runtime.

## Complete-fit results

| Input | Case (rows × features) | Prior fused ms | Native/library ms | sklearn ms | vs prior | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | small: 5,000 × 32 | 6.218 | 2.000 | 1.462 | 3.11× | 0.73× |
| float64 | medium: 20,000 × 128 | 8.752 | 3.348 | 5.202 | 2.61× | 1.55× |
| float64 | wide: 20,000 × 1,024 | 23.930 | 14.430 | 19.873 | 1.66× | 1.38× |
| float64 | large: 60,000 × 1,024 | 50.502 | 36.091 | 55.883 | 1.40× | 1.55× |
| float64 | multiclass weighted: 20,000 × 128 | 15.627 | 14.097 | 26.122 | 1.11× | 1.85× |
| float64 | ill conditioned: 5,000 × 32 | 16.097 | 3.561 | 3.323 | 4.52× | 0.93× |
| float64 | boundary: 20,000 × 512 | 22.418 | 11.823 | 12.215 | 1.90× | 1.03× |
| float64 | tall crossover: 250,000 × 32 | 57.312 | 10.835 | 36.630 | 5.29× | 3.38× |
| float64 | tall: 500,000 × 32 | 90.547 | 18.619 | 79.355 | 4.86× | 4.26× |
| float64 | tall wide: 500,000 × 128 | 101.033 | 40.216 | 91.795 | 2.51× | 2.28× |
| float32 | small: 5,000 × 32 | 6.220 | 1.573 | 1.367 | 3.95× | 0.87× |
| float32 | medium: 20,000 × 128 | 7.964 | 2.491 | 3.941 | 3.20× | 1.58× |
| float32 | wide: 20,000 × 1,024 | 18.936 | 10.608 | 12.859 | 1.79× | 1.21× |
| float32 | large: 60,000 × 1,024 | 36.833 | 25.709 | 35.514 | 1.43× | 1.38× |
| float32 | multiclass weighted: 20,000 × 128 | 13.479 | 11.241 | 16.862 | 1.20× | 1.50× |
| float32 | ill conditioned: 5,000 × 32 | 16.695 | 3.494 | 3.322 | 4.78× | 0.95× |
| float32 | boundary: 20,000 × 512 | 11.468 | 5.700 | 7.841 | 2.01× | 1.38× |
| float32 | tall crossover: 250,000 × 32 | 49.269 | 10.150 | 34.506 | 4.85× | 3.40× |
| float32 | tall: 500,000 × 32 | 85.876 | 15.437 | 75.034 | 5.56× | 4.86× |
| float32 | tall wide: 500,000 × 128 | 93.154 | 29.144 | 75.323 | 3.20× | 2.58× |

The weighted five-class case uses the **same fused optimizer in both library columns**; its timing difference is not evidence of a native multiclass optimization. The previous-fused column shares the current validation, label encoding, and centering with the new column. It isolates the optimizer change rather than recreating every cost of an earlier checkout. The [earlier fusion report](gpu-logistic-benchmarks.md) records the preceding implementation separately.

For 500,000 × 32 float32 input, the native median is 15.437 ms, the minimum 15.050 ms, and the interquartile range 0.681 ms. Float64 medians are 18.619 ms native versus 79.355 ms sklearn. Raw timings and spreads for every case are in the [JSON results](../benchmarks/results/native-logistic-2026-10-26.json).

## Correctness and limitations

All 420 measured fits completed without warnings. Every library fit reported gradient convergence. For the new-library column, independent float64 evaluation on the original features found a maximum gradient of **9.432e-5** against `tol=1e-4`, maximum relative objective increase over sklearn of **1.731e-6**, and prediction agreement of at least **99.945%**. These checks run outside the timer; final native convergence certification runs inside it.

The native path supports binary GPU fits with at most 4,096 features, history size 1–32, and line-search budget 1–32. Its float32 counters also bound the supported iteration budget. Larger workspaces use the generic solver. Extreme-input, zero-budget, weighted, unpenalized, tail-block, rejected-step, history-rollover, and stale-convergence regressions have dedicated tests.

Compilation is not free. For example, the first recorded float64 5,000 × 32 call took 178.8 ms before the 2.000 ms warmed median. First calls for every case are recorded, but they are **not independent process-cold measurements**: earlier shapes and driver/compiler caches can already be warm. One-off small fits can therefore favor sklearn by much more than the median table suggests.

GPU launch overhead still limits small problems. Wide fits remain constrained by full-matrix traffic and public input preparation. Native kernels alone do not imply an order-of-magnitude end-to-end speedup.

## Automatic dispatch

Final verification: **386 tests passed**, with 41 expected warnings in the full
suite. Ruff, repository compliance, wheel/sdist build, and strict Twine checks
passed. Separate default-dispatch checks confirmed that direct and patched tall
fits actually enter the native GPU solver, while a small patched fit uses sklearn.

The existing wide threshold remains: at least 20,000 rows and 20,000,000 input elements. A measured tall route now also admits at least 250,000 rows and 8,000,000 elements. Thus 250,000 × 32 and 500,000 × 32 use GPU by default. The 20,000 × 512 case retains its previous route because the float64 win is marginal. Small patched sklearn estimators continue to use stock sklearn; direct mlxlearn estimators retain their existing MLX CPU route below crossover.

These thresholds are conservative shape heuristics measured on one M4 Max, not a guarantee for every dataset or Apple chip. The table forces GPU execution to expose both wins and losses rather than crediting a sklearn fallback as GPU speed.

## Reproduction

```bash
python -m benchmarks.logistic_native --profile full --dtype both \
  --repeats 7 --warmup 1 \
  --output benchmarks/results/native-logistic-2026-10-26.json \
  --report bench/native-logistic-report.md
python -m pytest -q
python tools/compliance.py
```

Environment: Apple M4 Max, macOS 26.6.2, MLX 0.32.2, sklearn 1.9.0, NumPy 2.5.2, SciPy 1.18.1. Parameters: `C=1`, `tol=1e-4`, `max_iter=200`, fitted intercept. Seven rotated-order repetitions follow one warmup per variant/case/dtype. Timing synchronizes MLX. Stock sklearn is unpatched. Threadpoolctl requests one thread for recognized pools; it may not control Apple Accelerate. The JSON contains environment details, source hashes, execution orders, first calls, every fit’s diagnostics, and independent quality measurements.
