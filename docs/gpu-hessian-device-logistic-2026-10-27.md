# GPU Hessian-sketch and device logistic regression

This development report describes the GPU Hessian-sketch warm start, the
NumPy-free device-array interface, and the accuracy checks added after the
[GPU-native investigation](gpu-native-investigation-2026-10-27.md). Benchmark
results are recorded below. The full suite passed 767 tests with 43 warnings.

## Device-array interface

`mlxlearn_device` provides an experimental binary logistic fit whose import and
fit path do not import NumPy or scikit-learn:

```python
import mlx.core as mx
from mlxlearn_device import fit_logistic

X = mx.random.normal((20_000, 1_024)).astype(mx.float32)
y = (X[:, 0] > 0).astype(mx.int32)
model = fit_logistic(X, y, C=1.0, tol=1e-4, max_iter=200, sketch_size=4)
if model.converged:
    predictions = model.predict(X)
```

Inputs and returned coefficients, feature centers, intercept and predictions
are MLX arrays. The intercept is represented in centered coordinates to avoid
unnecessary cancellation on shifted inputs. The native optimizer, sketch,
compensated scoring and supported intercept refinement execute on the GPU;
small scalar controls and status cross to the host. The path currently supports
binary `0/1` targets, finite float32 features, a fitted intercept, positive L2
regularization, 512–4096 features, and the tested MLX 0.32.x Metal DLPack
bridge. See the API docstring for the exact supported dtypes and constraints.
An explicit status and `DeviceConvergenceWarning` report unsuccessful fits.
A NumPy/sklearn import-blocking subprocess tests this interface. The package's
existing distribution still declares NumPy and scikit-learn as dependencies for
the sklearn-compatible API; this runtime path does not remove those installation
requirements.

The final GPU check uses compensated float32 arithmetic on the original
float32 input. It has no general rounding-error or objective-optimality bound;
this is a numerical safeguard, not a proof that the reported model is optimal.
The device interface does not accept float64 training matrices. The existing
sklearn interface retains its selective original-input float64 CPU certificate
for risky inputs, rather than claiming that a float32 cast preserves arbitrary
float64 data.

## Hessian sketch

The binary sketch forms sample-space probes, computes directions in the row
space of centered `X`, and applies the weighted logistic Hessian at zero logits.
Finite positive-curvature pairs initialize the native L-BFGS ring; invalid
pairs are rejected. The search and convergence checks continue to use the
actual logistic objective rather than the sketch's quadratic approximation.
Seed histories, input and output buffers stay alive until all external Metal
commands complete, including failure paths. Seeded failure reports its spent
iteration budget instead of silently restarting the same full budget.

The device API enables four sketch pairs by default and supports
`sketch_size=0` for the unseeded control. In the sklearn-compatible path the
sketch is experimental and opt-in through the private tuning environment variable
`MLXLEARN_LBFGS_SKETCH_SIZE`, which must be set before importing mlxlearn, or
through the internal `tuning_context(lbfgs_sketch_size=4)`. It requires the
native prepared-input path. Default sklearn-compatible behavior remains
unseeded pending full-fit performance and correctness results.

## Convergence limitation

For the reproduced strongly scale-skewed binary inputs, an already-computed
initial weighted curvature diagonal triggers a stricter internal gradient
threshold. The optimizer, intercept refinement and original-input CPU check use
that same threshold. The requested public tolerance and accepted-iteration
budget are not relaxed. This fixes the two measured regressions, but a diagonal
spread is not a condition-number bound: correlated matrices can be ill
conditioned even when their diagonal scales are equal. Neither this heuristic
nor the wide progress rule guarantees objective parity with sklearn. Fits that
exhaust their budget or fail their numerical checks report nonconvergence.

## Validation and benchmark

The final full suite passed **795 tests with 43 warnings** using
`python3 -m pytest -q`. Focused tests also exercised the NumPy/sklearn import
block, native-buffer lifetime and failure paths, sketch curvature and row-space
behavior, bounded iterations, GPU loss/gradient scoring, zero weights and input
validation. These tests and the benchmark do not establish a universal error
bound for the device score.

The [final raw panel](../benchmarks/results/hessian-device-final-2026-10-27.json)
measured 12 fresh complete fits per variant after three warmups, rotating order,
with sklearn limited to one CPU thread. All input matrices were float32,
`tol=1e-4`, `max_iter=200`, and `C=1` except the exact duplicate-column case
(`C=1e6`). The host-input variants include validation, preparation, sketch
construction where applicable, optimization, verification and model export.
The device-interface inputs were already resident on the GPU; its one separately
measured input upload is **not included** in the device fit column. Independent
float64 objective, original-coordinate gradient, prediction and duplicate-column
null-space checks were run after timing every fit. Weighted device fits use
float32 device weights; checks include the original float64 weights and also
record scores for the represented weights. All 360 measured fits were scored;
the 216 corrected, sketched and device fits passed their quality gates, and the
51-file source/binary integrity audit passed. No measured fit emitted a warning.
The frozen baseline failed the weighted and wide conditioned objective gates,
and stock sklearn failed the shifted gradient check. This is a narrow synthetic panel on one
Apple-silicon machine, not a general performance claim.

| Dataset | Frozen solver ms | Corrected unseeded ms | Four-pair sketch ms | Device API ms | sklearn ms |
|---|---:|---:|---:|---:|---:|
| 20k × 1,024 | 6.97 | 6.94 | 8.40 | 10.62 | 12.36 |
| 20k × 4,096 | 42.56 | 42.81 | 52.03 | 56.01 | 65.22 |
| 20k × 1,024 weighted, ill-conditioned | 28.08 | 80.15 | 82.27 | 95.62 | 191.89 |
| 20k × 1,024 shifted | 13.41 | 13.67 | 13.50 | 10.85 | 17.92 |
| 20k × 1,024 duplicate columns | 8.82 | 8.80 | 9.94 | 11.26 | 14.51 |
| 20k × 4,096 ill-conditioned | 360.33 | 400.99 | 375.49 | 397.06 | 554.21 |

The one-shot upload observations ranged from 1.22 ms to 21.68 ms and are not a
repeated transfer benchmark. The unseeded corrected 4,096-feature ill-conditioned
fit ran the GPU-managed MLX loop directly; the other corrected host-input cases
used the native command loop. Every explicitly sketched host fit used four
accepted sketch pairs and the native loop; every device fit used four pairs
without an unsketched retry in this panel. `sketch_size=4` is slower than the
corrected unseeded path on the ordinary wide cases, but was about 6% faster on
the ill-conditioned 4,096-feature case. It is not a generally faster default.

The weighted ill-conditioned frozen fit was faster but had an objective about
**0.002903 relative above sklearn's**; the corrected unseeded fit was about
**7.80e-6 above**, and the sketch fit **2.69e-6 above**. Thus its 28.08 → 80.15 ms
change is a real cost of addressing that observed early stop. On the
ill-conditioned 4,096-feature fixture, the frozen objective was **1.981e-5 above
sklearn's**, while the corrected and sketch objectives were respectively
**4.50e-6** and **4.17e-6 below** it. These are one-sided comparisons: a lower
objective is not treated as an error. Both corrected and sketched paths retained
zero null-space coefficient ratio on the exact duplicate-column fixture.

On the shifted fixture, stock sklearn's raw-input run had an independent maximum
gradient of **0.351**, far above `1e-4`, despite emitting no warning. Its timing
is reported as observed, but it is not an accuracy-equivalent speed comparison.
The audit instead used a separate untimed, centered float64 sklearn reference
whose original-coordinate gradient was **6.14e-8**. Device and sketch predictions
agreed with that reference on **99.99%** of rows.

The [first audited panel](../benchmarks/results/hessian-device-panel-2026-10-27.json)
exposed a sketch objective regression and a max-iteration failure. It also
exposed an older fallback that spent a full native budget and then restarted
from zero while reporting only the second run. The final implementation preserves
accepted-iteration and evaluation counts across a checked reanchor, reports
budget exhaustion, uses the wide skewed unseeded GPU loop from the start, and
aligns seeded initial-metric calculation across the host and device interfaces.
The final panel was rerun after these changes; the first panel is retained as
historical evidence, not final performance data.

The final panel can be rerun in this workspace with:

```bash
python3 -m benchmarks.hessian_logistic --device \
  --sketch-size 4 --device-sketch-size 4 --repeats 12 --warmup 3 \
  --output benchmarks/results/hessian-device-final-2026-10-27.json
```

Its frozen Python baseline and manifest are in
[`benchmarks/reference/hessian-start-2026-10-27.zip`](../benchmarks/reference/hessian-start-2026-10-27.zip)
and its adjacent manifest. The exact baseline native binary is retained locally
at `bench/hessian-start/_metal_loop.abi3.so` and checked by hash; it is not
included in the source archive. The [previous source-only research archive](../benchmarks/reference/gpu-no-numpy-fit-prototype-2026-10-27.zip)
contains the earlier native source and build files, but rebuilding may not
produce the exact measured binary hash.
