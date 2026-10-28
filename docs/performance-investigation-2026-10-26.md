# Performance investigation — 26 October 2026

This records the investigation before the GPU-fusion implementation. The later
[GPU benchmark](gpu-logistic-benchmarks.md) measures the implemented change
against a frozen copy of the original solver **and** objective. The exploratory
script below now uses the current objective with the legacy optimizer; its saved
JSON remains the evidence for the historical prototype numbers.

mlxlearn has substantial optimization headroom, but the apparent gap to the other
projects combines implementation costs with unlike benchmarks. The most actionable
finding is the logistic optimizer architecture: a small experimental hybrid using
CPU L-BFGS-B and the existing MLX loss/gradient made forced-MLX estimator fits
**1.55–5.00× faster** on this machine. It still lost to stock sklearn on these cases.
SVC has a different bottleneck: at 4,000 rows, its Python/NumPy optimization loop
accounted for **94–96%** of two sampled fits after the Gram matrix was built.

This investigation adds a reproducible experiment and results; it does not change
the production library, dispatch thresholds, or historical benchmark records.

## Fresh measurements

Apple M4 Max, 128 GiB; macOS 26.6.2; Python 3.13.3; MLX 0.32.2;
sklearn 1.9.0; NumPy 2.5.2, built with Apple Accelerate. Library revision
`fb79a1b11dad401e82808885884fc633da9c3e86`. Experiments used fresh estimators,
one warmup per variant, seven timed repetitions with alternating order, and
synchronization before and after timing. These are small diagnostic experiments,
not a comprehensive performance guarantee or a rerun of the historical grid.

### Logistic regression: complete estimator fit, milliseconds

| Rows × features | Current forced MLX | Device-scalar recursion | Hybrid CPU optimizer / MLX objective | sklearn | Hybrid gain over current |
|---|---:|---:|---:|---:|---:|
| 5,000 × 32 | 20.52 | 12.86 | 4.11 | 1.48 | 5.00× |
| 20,000 × 128 | 25.14 | 16.75 | 9.13 | 5.29 | 2.75× |
| 20,000 × 1,024 | 68.31 | 56.05 | 44.14 | 20.07 | 1.55× |

Input was float64 Gaussian data with a noisy binary target, seed 41, `C=1`,
`tol=1e-4`, `max_iter=200`. The experiment explicitly overrides the crossover and
asserts the actual `mlx` fit backend. Merely setting `device="gpu"` does **not**
override the crossover: direct estimators can still select their CPU path.

The device-scalar prototype changes only the L-BFGS two-loop equations' execution:
intermediate dot products remain MLX scalars. On these three cases it retained
8/8/10 accepted iterations and changed the independently evaluated float64
objective by less than 3e-12. The hybrid uses SciPy L-BFGS-B for optimizer state,
the existing stable MLX objective and gradient, history 10, and `ftol=0` to avoid
declaring success based solely on relative function reduction. It took 8/8/8
iterations, agreed with sklearn's training predictions on all three cases, and
its float64 objectives differed from sklearn by less than 5e-9.

The standalone solver probe excludes estimator validation and shows 1.40–1.63×
gains for device scalars; the full-fit table above includes validation. Do not
confuse the two boundaries. The wide full fit has appreciable preprocessing and
validation cost, so solver-only gains shrink at the estimator boundary.

**Validation limit:** the hybrid passed all 40 existing logistic parity tests.
The device-scalar prototype passed 39/40: the ill-conditioned, equal-truncated-budget
test stopped on a line-search failure after 80 iterations rather than the expected
100, with gradient maximum about 3.24e-4. It is a promising experiment requiring
numerical work, not a production-ready patch. Neither prototype received the
entire estimator-compliance/invariants suite.

### SVC: complete forced-MLX fit and a separate stage profile

| Rows × features | mlxlearn median | sklearn median | Speedup | Gram build, sampled | CPU SMO, sampled |
|---|---:|---:|---:|---:|---:|
| 4,000 × 32 | 109.28 ms | 116.06 ms | 1.06× | 3.73 ms | 104.12 ms |
| 4,000 × 256 | 157.90 ms | 623.43 ms | 3.95× | 6.63 ms | 143.82 ms |

RBF, default cache and tolerance, seed 43. Stage timings are one additional
instrumented fit and should not be added to or substituted for the medians.
Both fits reported convergence, internal KKT violations below 1e-3, and identical
training predictions to sklearn. This is not an independent float64 KKT audit.

At the wider shape, eliminating Gram construction entirely would improve the
sampled total by only about 4.5%. Optimizing the solver is the consequential target.

### KNN: the baseline and thread policy matter

For 256 queries, k=10, exact brute-force search, float64 public inputs:

| Training shape | Threadpool setting | mlxlearn | sklearn | Speedup |
|---|---|---:|---:|---:|
| 8,000 × 32 | default | 2.41 ms | 2.50 ms | 1.04× |
| 8,000 × 32 | limit 1 | 2.66 ms | 4.54 ms | 1.71× |
| 8,000 × 32 | limit 4 | 2.56 ms | 1.91 ms | 0.75× |
| 32,000 × 768 | default | 6.69 ms | 19.06 ms | 2.85× |
| 32,000 × 768 | limit 1 | 7.05 ms | 34.79 ms | 4.93× |
| 32,000 × 768 | limit 4 | 11.25 ms | 23.53 ms | 2.09× |

Neighbor indices matched exactly; maximum distance error was 2.21e-6. Limits use
`threadpoolctl`, which detected sklearn's OpenMP pool (default 16). It did not
detect/control the Apple Accelerate BLAS pool; these labels do not mean all CPU
execution used exactly that many threads. This small sample also contains timing
variation and cannot establish a universal best thread setting.

## Why the headline comparisons exaggerate the gap

**Different denominators.** The saved remote comparison reports logistic fit at
22,500 × 64 as sklearn 154.56 ms, Intel 12.12 ms, cuML 8.99 ms: 12.75× and 17.20×.
The later Apple comparison at the same shape has sklearn at **5.12 ms**, about
30× faster than the remote baseline, and the earlier Apple accelerator at 7.39 ms.
The data seeds, builds and hardware differ, so this is not an absolute ranking;
it demonstrates why the speedup ratios cannot rank the accelerator implementations.
See the [remote CSV](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/remote_synthetic_summary_quick_svm10k_20260707_194304.csv:2)
and [Apple CSV](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/local_sklearn_mps_summary_quick_20260708_112832.csv:2).

**Reused fitting.** An earlier Apple logistic result reports 3.23× at 1.51 ms,
but records **zero optimizer iterations**. A subsequent run explicitly disables
`SKMPS_LOGREG_ENABLE_CONTINUATION_CACHE`, takes 11 iterations and 7.39 ms, and
reports 0.69×. The predecessor's continuation cache is on by default and reuses
theta across fresh estimators sharing X/y identities. Its keys even omit C.
These are strong signs that part of the apparent earlier advantage was reused
work, not faster fresh optimization. Sources: [earlier result, line 3](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/local_sklearn_mps_results_quick_20260708_112635.jsonl:3),
[cache-disabled run](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/local_sklearn_mps_results_quick_20260708_112832.jsonl:1),
[cache configuration](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/linear_model/logistic_regression.py:90),
[cache retrieval](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/linear_model/logistic_regression.py:1274).

**Fit versus inference.** The predecessor's saved 40k × 100 SVC notebook reports
2.47× fit, **52.09× predict**, and 2.86× combined. Comparing that prediction figure
to mlxlearn's fit figure manufactures a much larger gap. Native cached-row SMO
is explicitly recorded as active in this notebook. See [saved SVC notebook](/Users/tunguz/Programming/scikit_learn_mps/examples/notebooks_mps/09_svc_mps_M4.ipynb).

**Different timing boundaries and thread budgets.** The remote script uploads
CuPy inputs before timing and downloads predictions afterward. Its thread
environment variables do not control Intel's oneTBB pool, and it records estimator
module names rather than actual per-operation dispatch. It also uses float32
standardized classification data, k=5 classifier predictions, and different query
sizes. mlxlearn's standard runner uses float64 Gaussian data and k=10 neighbor
queries. Sources: [remote timing](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/remote_sklearn_accel_benchmark.py:164),
[output conversion](/Users/tunguz/Programming/sklearn_comparisons/benchmarks/remote_sklearn_accel_benchmark.py:292),
[Intel thread documentation](/Users/tunguz/Programming/sklearn_repos/scikit-learn-intelex/doc/sources/parallelism.rst:54).

The predecessor also caches completed KNN result prefixes across models and k
values for some uniform-weight predictions. That can legitimately accelerate an
explicit parameter sweep, but it is not a fresh-query speedup. mlxlearn should
retain estimator-owned immutable data and avoid restoring implicit identity caches.
See [prefix cache](/Users/tunguz/Programming/scikit_learn_mps/sklearnex/neighbors/common.py:55).

## Actual implementation bottlenecks and reference designs

### Logistic regression

[`_dot`](../src/mlxlearn/_solvers/lbfgs.py) returns a Python float from an MLX
reduction. With a full history of 10, `_two_loop` performs **21 dependent scalar
reads per direction**, before the line search, curvature updates and stopping
checks. In a sampled 20k × 128 fit, there were 95 `_dot` calls across eight
iterations, consuming 14.2 of 30.1 ms inclusively; `_two_loop` alone accounted for
10.7 ms. These nested durations must not be summed. Unified memory does not remove
the scheduling and synchronization needed to read unfinished GPU results.

The current docs focus on line-search synchronizations and miss this recursion
cost. Converting a float32 reduction to a Python float also does not restore
float64 reduction accuracy.

Intel's CPU path uses SciPy L-BFGS-B with oneDAL objective/gradient callbacks;
the predecessor uses SciPy plus MLX callbacks. Both demonstrate the usefulness
of keeping small optimizer vectors on CPU while accelerating bulk arithmetic.
See [Intel driver](/Users/tunguz/Programming/sklearn_repos/scikit-learn-intelex/daal4py/sklearn/linear_model/logistic_path.py:330)
and [predecessor hybrid](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/linear_model/logistic_regression.py:785).

For further gains, specialize the logistic line search. For fixed direction d,
compute logits Z and directional logits Xd once; candidate logits are Z+tXd.
The Wolfe directional derivative can be computed from residuals and Xd without
forming the whole gradient for every rejected step. Form the full gradient for
the accepted candidate. This preserves the objective and can permit bounded
batches of candidates. Keep mlxlearn's stable loss and blocked host float64
objective accumulation initially; do not transplant the predecessor's clipped
probability loss or silently relax convergence.

Compile array-only computation blocks before considering custom Metal.
[`mx.compile`](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
can fuse operations but cannot trace functions that evaluate arrays or extract
host values; adding a decorator around today's whole solver will not fix it.
cuML is not synchronization-free either: the inspected implementation transfers
loss scalars and uses scalar dots in its quasi-Newton loop. Its native execution
and workspace reuse are also relevant, not merely the algorithm name.

### SVC training and inference

The [current fit implementation](../src/mlxlearn/svm/_classes.py) materializes the
full float32 Gram when it fits the cache. With the default 200 MiB budget this
includes binary problems up to about 7,240 rows, unless another memory ceiling
intervenes. [`kernel_matrix`](../src/mlxlearn/_kernels/kernels.py) returns NumPy;
the entire SMO loop then uses Python/NumPy. It scans full arrays, creates temporary
vectors, and implements no shrinking despite accepting `shrinking=True`.

Thus the explanation in the existing benchmark documentation about per-iteration
GPU kernel-row dispatch does not describe its 4,000-row examples. Larger
cached-row problems do incur GPU synchronization on cache misses.

Near-term options are a compiled CPU SMO loop with reusable buffers and validated
shrinking/gradient reconstruction. The longer-term cuML lesson is **block working
sets**: many inner pair updates inside one device block, followed by batched
kernel evaluation and gradient updates. cuML still synchronizes per outer
iteration; the gain comes from amortizing that cost. Its documentation explicitly
describes [hierarchical SMO decomposition](https://docs.nvidia.com/cuml/26.08/api/generated/cuml.svm.SVC/).
Tune working-set sizes for Metal threadgroup limits rather than copying CUDA's
1,024-member choice. FP32 device gradients would also need validation against the
current FP64 CPU gradient/KKT behavior.

Inference offers a more contained improvement. Current `kernel_combine` recreates
support-vector tensors per call, transfers full kernel tiles to float64 NumPy,
and multiplies by weights on CPU. The predecessor retains support tensors and
reduces `K @ weights` on GPU before returning scores. Prototype persistent
estimator-owned tensors and device contraction, with precision checks around
decision boundaries. See [predecessor prediction](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/svm/svm.py:2639).
A future explicit model-conversion path could also combine CPU fitting with GPU
prediction; this needs a deliberate change to the current sticky-backend contract.

### The native-SMO no-go needs a new experiment

[`measure_native_core.py`](../tools/measure_native_core.py) tests 2k, 6k and 12k
rows. At the frozen predecessor revision `720053b`, default dispatch skips its
SMO below 12k, and exactly 12k uses the dense solver, which never invokes the
native cached-row core. Native execution starts in the branch **above** 12k.
The benchmark does not assert the backend or record the environment overrides.
See [minimum threshold](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/svm/svm.py:2349),
[dense/cached branch](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/svm/svm.py:2131),
and [native invocation](/Users/tunguz/Programming/scikit_learn_mps/mpsbackend/svm/svm.py:2046).

Consequently, absent undocumented overrides, the published toggle timings do
not measure native execution. The conclusion that its benefit disappears with
problem size is unsupported. The pure-Python packaging preference can remain a
product decision, but it should not be justified by that comparison. Rerun at
20k/40k with native execution asserted, then separately profile a compiled version
of **today's dense solver**. Neither gain is established by this investigation.

### Nearest neighbors

mlxlearn already improves on the predecessor: train-tile merging stays on GPU,
partition replaces repeated argmin, and candidates are refined on-device.
The next target is the dense distance tile and full-shaped partition index
buffer. Follow cuVS's distance/selection specialization: small-k top-k kernels,
bounded candidate merging, and separate paths for one query versus large batches.
Preserve centering, oversampling/refinement and distance/index tie ordering.
Fused paths also need shape-aware dispatch; cuVS itself documents poor behavior
for some [small-query configurations](https://github.com/NVIDIA/cuvs/issues/278).

MLX offers [JIT custom Metal kernels](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html),
so this does not inherently require platform-specific Python extension wheels.
Uniform-weight classifier voting could skip final distance output and eventually
run on GPU; this is secondary to search for small query batches. Approximate
neighbors should remain an explicitly separate API.

## Recommended sequence

1. **Repair measurement coverage.** Shared serialized datasets, recorded backend,
   dtype, BLAS/thread configuration, hardware, convergence/objectives, support-vector
   counts, and NumPy-to-NumPy versus resident-array timing. Include cold calls,
   fresh warmed fits, and explicit reuse workloads separately. Both current smoke
   logistic cases and both smoke SVC cases fall below the patched work thresholds,
   so they mostly test sklearn fallback. Add forced-backend panels. Record full
   shape/query/k/kernel conditions rather than reducing crossover to sample count.
2. **Productize the hybrid logistic experiment first.** Preserve stable numerics,
   explicit solver state and diagnostics. Compare more conditioning, weighting,
   multiclass and large-width cases; pass full invariants/compliance. Keep CPU
   dispatch where it wins. The existing wide threshold did not predict a win in
   this diagnostic run and deserves a broader refresh.
3. **Prototype SVC inference residency and contraction.** This directly targets
   the earlier project's large prediction gains with smaller scope than training.
4. **Reassess native CPU SMO and shrinking on measured hotspots.** Use backend
   assertions and matched convergence, not a toggle that may never be exercised.
5. **Build larger GPU kernels where evidence supports them.** Specialized logistic
   line search, SVC block decomposition, KNN fused selection. These are hypotheses
   requiring benchmarks; no additional speedup multiplier is established yet.

After these, expanding to dense Ridge/linear regression, PCA and KMeans provides
more matrix-heavy workloads than the present three algorithm families. Broader
accelerator headline ranges also reflect this portfolio difference. It is too
early to interpret the current results as an Apple hardware or MLX performance
ceiling.

## Reproduction and source scope

Run from the repository root:

```sh
python3 -m benchmarks.investigate_performance --output bench/investigation.json
python3 -m benchmarks.investigate_performance --section followup --output bench/investigation-forced.json
```

[Experiment source](../benchmarks/investigate_performance.py),
[solver/KNN raw samples](../benchmarks/results/investigation-2026-10-26/investigation-2026-10-26.json),
and [forced-estimator/SVC raw samples](../benchmarks/results/investigation-2026-10-26/investigation-forced-2026-10-26.json)
are retained. Timed sections contain no accuracy scoring. Profile sections run
separately. Prototype function replacements affect only the experiment process.

The user-named `scikit-learn-intelex` working tree had most tracked sources deleted.
It was left untouched. Source comparison used the populated sibling
`/Users/tunguz/Programming/sklearn_repos/scikit-learn-intelex` at `01d35892` and
`/Users/tunguz/Programming/sklearn_repos/cuml` at `9699777c5`. These checkouts need
not match the binaries in the historical reports. Intel and NVIDIA timings were
read from saved artifacts, not rerun remotely. Architectural ideas were studied;
no predecessor, Intel, or CUDA implementation was copied into production code.
