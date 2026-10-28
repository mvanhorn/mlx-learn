# GPU-native logistic investigation — 2026-10-27

This investigation asks how much of logistic regression can run without NumPy
arithmetic or host-side numerical work, and which GPU algorithms might improve
complete-fit speed. It concerns the current dense binary and multinomial
logistic paths, rather than every estimator in the library. The experiments
below are research prototypes, not shipped solver replacements. Their timing is
not directly comparable to the end-to-end, 30-repeat benchmark in the
[previous report](convergence-accuracy-2026-10-27.md).

## NumPy and CPU boundaries in the current path

NumPy currently serves several distinct roles. Removing an `import` alone would
not remove copies, scheduling, precision loss, or CPU computation.

- Input preparation and validation convert or inspect host arrays; float64
  inputs can be converted and checked in blocks. See
  [`preparation.py`](../src/mlxlearn/_common/preparation.py) and
  [`_logistic.py`](../src/mlxlearn/linear_model/_logistic.py).
- Target labels, class and sample weights, and scikit-learn-compatible output
  attributes use NumPy. Arbitrary label types and scikit-learn interoperability
  are API boundary concerns, separate from the GPU optimizer. A new GPU-array
  API could avoid a NumPy output boundary, but would have a different contract.
- The native Metal loop still converts small metadata and parameter buffers at
  the Python boundary. Its main iterative binary optimizer runs on the GPU;
  see [`native_command.py`](../src/mlxlearn/_solvers/native_command.py) and
  [`_metal_loop.mm`](../src/mlxlearn/_solvers/_metal_loop.mm).
- For numerically risky cases, the current code scores the original input and
  exported parameters in float64 CPU blocks. Multiclass intercept refinement
  also solves a small Hessian system on the CPU. These are real host-side
  numerical computations, not merely NumPy array interoperability; see
  [`logistic.py`](../src/mlxlearn/_solvers/logistic.py).

A practical NumPy-light architecture would retain the existing scikit-learn
interface while handing contiguous buffers directly to Metal, doing conversion,
validation, weighting and numerical work on the GPU, and creating compatible
outputs at the boundary. A separate MLX or buffer-protocol interface could
avoid NumPy entirely for callers whose inputs and outputs satisfy that API.
Neither approach makes unsupported NumPy dtypes, strides, labels, or precision
requirements disappear; those cases need explicit handling or a fallback.

A device-array API alone would not make the current package NumPy-free. In a
fresh Python process, `import mlxlearn` loads NumPy; it does not load sklearn.
The [`_common` package initializer](../src/mlxlearn/_common/__init__.py) eagerly
imports `.arrays` and `.base`, which load NumPy. The package also declares both
NumPy and scikit-learn as mandatory dependencies in
[`pyproject.toml`](../pyproject.toml). A truly NumPy-free install and normal
package import would require changing these eager imports and dependency
packaging, or using a separate package or extension. Those are compatibility
and distribution changes in addition to GPU algorithm work.

## The original-input precision problem

The [official MLX dtype documentation](https://ml-explore.github.io/mlx/build/html/python/data_types.html)
states that float64 arrays are supported for CPU operations, while GPU use
raises an exception. The [conversion documentation](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
states that NumPy float64 arrays default-convert to MLX float32. Consequently,
a GPU check of only the converted float32 design cannot verify the model on an
arbitrary original float64 design. Large feature offsets also amplify tiny
intercept-gradient errors in the original feature coordinates.

Custom Metal code can instead read the original IEEE-754 bytes as integers and
emulate higher precision with several float32 components. That is software
arithmetic on the GPU, not native float64 support. It must preserve or bound
input-representation and arithmetic errors, including underflow, overflow,
subnormal values, signs, NaNs, infinities, and conversion or memory-layout
issues. Merely obtaining a small float32 gradient is insufficient for a
precision guarantee.

## Compensated GPU scoring prototype

The scratch prototype computes binary logits with compensated products and
sums, evaluates a range-reduced polynomial approximation to sigmoid, and
reduces feature and intercept gradients with float32-pair arithmetic. It is
intended to explore replacing the risky-input CPU score, **not** to loosen the
stopping criterion. The float64 experiment decodes input bytes into a pair of
float32 values in Metal rather than first rounding each value to float32.

On the earlier 20k × 1,024 float32 input shifted by 1,000, the prototype's
maximum absolute gradient difference from an independent float64 CPU score was
**6.2e-11** unweighted and **1.1e-8** weighted. The respective six-sample warm
GPU-kernel medians were **2.97 ms** and **1.13 ms**. The samples varied and
exclude GPU allocation and input transport; the weighted fixture also normalizes
weights on the host. These observations are not complete-fit speedups and do
not establish general accuracy.

An adversarial 20k × 1,024 float64 input with values `10000 + noise`, where the
noise has scale `1e-4`, demonstrates why the original bits matter. Scoring
after float32 downcasting had a **2.73** maximum gradient error; decoding the
original input bytes and using compensated arithmetic reduced the error to
**9.97e-8** on that example. The six-sample kernel median was **3.89 ms**,
again excluding input transport. This is a deliberately constructed numerical
stress case rather than a representative timing panel.

The [source-only prototype archive](../benchmarks/reference/gpu-compensated-certificate-prototype-2026-10-27.zip)
and [manifest](../benchmarks/reference/gpu-compensated-certificate-prototype-2026-10-27.manifest.json)
record reproduction commands and limitations; raw results are available for
[float32](../benchmarks/results/gpu-compensated-certificate-f32-2026-10-27.json)
and [adversarial float64](../benchmarks/results/gpu-compensated-certificate-f64-adversarial-2026-10-27.json).
The prototype does **not** yet meet the requested NumPy-free workflow:
fixture creation, byte-view transport, bias splitting, weight normalization,
and final objective reduction use NumPy. The decoder assumes little-endian,
ordinary finite normal inputs in a limited exponent range and is incorrect for
some other values. It handles binary logistic regression only. Its polynomial
and compensated reductions have no established global error bound, and the
input pair retains fewer than all 53 float64 significand bits in general.

A candidate production design would transport original bytes through a native
buffer interface, decode each binary64 significand into three exact float32
fragments, perform compensated accumulation on the GPU, and compute a
conservative error bound for the resulting gradient. It could report convergence
only when the estimated norm **plus its bound** is within tolerance. Cases
outside the supported exponent range or with a bound too large to certify would
need explicit nonconvergence or a separately justified fallback. Multinomial
loss, weights, arbitrary strides, and objective checks would require the same
care; none is validated by the present prototype.

## NumPy-free device-input proof of concept

A separate scratch binary optimizer accepts already resident MLX arrays and
runs without importing NumPy or scikit-learn. A smoke test blocks those imports
while loading the required kernel and extension files directly, bypassing the
current package import path. This proves a direct-file execution path is
possible, not that the full existing
library has become NumPy-free: the proof still has small CPU round trips and
omits the original-input full-precision certificate and fallback. It is not
production-ready.

A preliminary 12-fit warm resident-data panel produced the following medians.
These compare different API boundaries: the prototype already has input on the
GPU, while the current estimator and sklearn receive host arrays. Input upload
to the device prototype is excluded; its recorded upload samples are separate.
Last-fit independent checks passed on these fixtures, but sklearn's shifted fit
failed its own gradient check and cannot serve as a certified optimum there.
These timings are exploratory and do not establish a general speedup.

| float32 input | Device prototype ms | Current estimator ms | sklearn ms |
|---|---:|---:|---:|
| 20k × 1,024 | 15.46 | 9.91 | 13.42 |
| 20k × 1,024 weighted | 12.26 | 10.96 | 14.22 |
| 20k × 4,096 | 42.34 | 43.20 | 65.59 |
| 20k × 1,024 shifted | 10.86 | 15.03 | 20.50 |

The [source archive](../benchmarks/reference/gpu-no-numpy-fit-prototype-2026-10-27.zip),
[manifest](../benchmarks/reference/gpu-no-numpy-fit-prototype-2026-10-27.manifest.json)
and [raw results](../benchmarks/results/gpu-no-numpy-fit-prototype-2026-10-27.json)
include the import-blocking smoke test. The proof's limited numerical checks
prevent its apparent shifted-input timing from being interpreted as an
accuracy-equivalent speedup.

## Fusing the direct gradient check

The existing native loop already combines the projection and first-candidate
gradient on the optimization path. A separate scratch change fuses part of the
final float32 direct scoring, aiming to reduce matrix passes. In twelve
alternating complete fits per case, with each fit independently scored after
timing, it produced these preliminary medians:

| Case | Current ms | Fused direct check ms |
|---|---:|---:|
| 20k × 1,024 | 6.721 | 6.970 |
| 20k × 1,024 weighted | 7.435 | 7.427 |
| 20k × 4,096 | 43.288 | 42.496 |
| shifted 20k × 1,024 | 12.127 | 12.237 |
| strict `tol=1e-6` | 18.510 | 18.047 |

All 120 measured fits met the independent gradient check and emitted no
warnings, but these short runs show no consistent complete-fit gain. The fused
score is still float32 and does not replace original-input float64
certification. A 30-pair follow-up on four 20k × 4,096 cases, with seeds 41
and 83 and `C` values 1 and 10, measured **1.5–2.3%** lower median complete-fit
time for the fused version. All fits completed the native route without warnings,
with the same exported coefficients, passing independently scored gradients and
one-sided objective checks:

| 20k × 4,096 | Current ms | Fused ms | Paired wins |
|---|---:|---:|---:|
| seed 41, `C=1` | 42.864 | 41.942 | 26 / 30 |
| seed 83, `C=1` | 47.261 | 46.530 | 24 / 30 |
| seed 41, `C=10` | 43.435 | 42.582 | 28 / 30 |
| seed 83, `C=10` | 48.250 | 47.156 | 22 / 30 |

This is a small wide-case gain, not a general solver speedup; the 1,024-feature
and shifted cases did not show a consistent benefit. The [source archive](../benchmarks/reference/gpu-fused-direct-experiments-2026-10-27.zip),
[manifest](../benchmarks/reference/gpu-fused-direct-experiments-2026-10-27.manifest.json),
[initial panel](../benchmarks/results/gpu-fused-direct-full-fit-2026-10-27.json)
and [wide follow-up](../benchmarks/results/gpu-fused-direct-wide-followup-2026-10-27.json)
preserve these results. The benchmark helper scripts were captured after the
run and were not themselves included in its in-run source hash audit.

A subsequent stress panel found a **remaining accuracy limitation in the
existing solver as well as the fused prototype**. On an ill-conditioned 20k ×
4,096 problem at `C=1`, both stopped with an independently measured original
maximum gradient of about `6.85e-5` at `tol=1e-4`, but their objective was
**1.981e-5 relative above sklearn's**, exceeding the previous one-sided
`1e-5` comparison threshold. The gradient condition and current wide progress
heuristic therefore do not guarantee objective parity. On an extreme scaled fixture
with some zero sample weights, both hit `max_iter` and correctly warned, with an
independent maximum gradient around `0.0694`. Tight-tolerance, ordinary
weighted and shifted cases in this stress panel met the independent gradient
check; sklearn's shifted output did not provide a certified optimum. The [stress results](../benchmarks/results/gpu-fused-direct-wide-stress-2026-10-27.json)
and [source archive](../benchmarks/reference/gpu-fused-direct-stress-2026-10-27.zip)
preserve the observations.

An independent review identified a correctness blocker: the fused kernel sums
up to 64 float32 per-row losses in a lane, which can overflow even when each
loss and the existing smaller-block calculation remain finite. On one
[overflow probe](../benchmarks/results/gpu-fused-direct-overflow-probe-2026-10-27.json),
the existing direct loss was `9.99999968e36` while the fused result was
positive infinity, with 156 overflowing partial sums. Its changed gradient
reduction order can also move a result across the convergence threshold. A
nonfinite fallback and a conservative error or threshold-margin check would be
needed before considering production use; neither is established here.

## Diagonal initial metric for ill-conditioned data

A separate scratch experiment changes L-BFGS's initial inverse-Hessian metric
using clipped per-feature diagonal curvature estimates. It leaves the loss and
requested gradient tolerance unchanged. With a maximum metric ratio of eight,
the complete-fit results were promising on two synthetic 20k × 1,024
ill-conditioned cases:

| Case | Scalar metric | Clipped diagonal metric |
|---|---:|---:|
| Unweighted | 70.4 ms, 128 iterations | 17.1 ms, 26 iterations |
| Weighted | 29.7 ms, 46 iterations | 20.6 ms, 26 iterations |

These are only six interleaved timed fits after two warmups, and the run did
not instrument source integrity or solver route. Each timed result had an
independent original-input float64 objective and gradient check. A follow-up
with 12 balanced fits after three warmups, verified native-loop completion,
source and binary integrity checks, and independent scoring produced:

| Case | Scalar metric | Clipped diagonal metric | sklearn |
|---|---:|---:|---:|
| Unweighted ill-conditioned | 72.13 ms | 17.77 ms | 56.95 ms |
| Weighted ill-conditioned | 31.56 ms | 20.18 ms | 195.28 ms |

This remains a narrow synthetic study, not a broad production benchmark. On
the unweighted fixture, the diagonal
metric's objective was **1.34e-5 relative worse** than the scalar result even
though both were below sklearn's; so the speedup did not preserve the previous
objective at the `1e-5` comparison gate. On the weighted ill-conditioned
fixture, the scalar and clipped results were respectively **0.00290** and
**2.87e-6** relative above sklearn's objective.

More seriously, on two rank-deficient fixtures at `C=1e6`, the clipped metric
produced a null-space coefficient ratio of about **1.32**, compared with zero
for the scalar metric, while objective and gradient checks passed. This shows
that those checks alone do not protect model coefficients in poorly identified
problems. The method is **not safe as a global default**. Its promising
ill-conditioned speed merits further study with conditioning detection,
null-space/regularization checks, robust validation and a defensible fallback;
those safeguards and full production measurements are not established. The
[source archive](../benchmarks/reference/diagonal-metric-experiment-2026-10-27.zip),
[manifest](../benchmarks/reference/diagonal-metric-experiment-2026-10-27.manifest.json),
[initial panel](../benchmarks/results/diagonal-metric-experiment-2026-10-27.json)
and [sklearn comparison](../benchmarks/results/diagonal-metric-sklearn-2026-10-27.json)
retain the full data and reproduction details.

## Mixed-precision shadow solver

A scratch alternative converts the centered float32 design to float16 and runs
the native loop on that shadow copy. It then checks the final gradient and, for
wide inputs, the two-step objective-progress rule against the original
float32 design. The wide check retains three iterates; a rejected shadow
candidate restarts the unchanged float32 native solver. Conversion, checking
and fallback are included in the measured fit, and checkpoint differences were
crosschecked against a direct float32 evaluation after timing.

On 20k × 4,096, eight fits including cold runs measured **44.54 ms** for the
baseline versus **46.30 ms** for the shadow path in the initial panel; a later
fused-checkpoint version measured **45.95 ms** versus **48.54 ms**. All 32
shadow fits in the four-case fused panel passed the independent float64
quality checks and none fell back, but the wide case did not speed up. This is
not evidence of accuracy for arbitrary inputs, and the checks still include
small float64 CPU penalties. The candidate has not been added to the library.

The [source-only archive](../benchmarks/reference/fp16-shadow-experiment-2026-10-27.zip),
[manifest](../benchmarks/reference/fp16-shadow-experiment-2026-10-27.manifest.json),
[initial results](../benchmarks/results/fp16-shadow-experiment-2026-10-27.json)
and [fused-checkpoint results](../benchmarks/results/fp16-shadow-fused-checkpoint-2026-10-27.json)
preserve the experiments.

## Larger scalar L-BFGS history

A further scratch experiment kept the scalar initial metric and varied the
native L-BFGS history between 10, 20 and 32 pairs. It used three warmups and
eight balanced complete fits per variant, verified native-loop completion and
source/binary integrity, and independently scored each result in float64.
Selected warm medians were:

| Case | History 10 | History 20 | History 32 | sklearn |
|---|---:|---:|---:|---:|
| 20k × 1,024 ill-conditioned | 71.71 ms | 38.08 ms | 127.67 ms | 57.88 ms |
| 20k × 1,024 weighted ill-conditioned | 28.31 ms | 36.27 ms | 56.91 ms | 190.59 ms |
| 20k × 4,096 | 42.97 ms | 63.61 ms | 80.02 ms | 64.34 ms |

The 20-pair version is faster on the first fixture, but its objective is
about **0.00182 relative worse** than the 10-pair result, although it nearly
matches sklearn's. On the weighted fixture both histories remain about
**0.00290 relative above sklearn's objective**. The scalar metric preserved a
zero null-space coefficient ratio in both rank-deficient cases, but history 20
was slower there. The speed and quality changes therefore do not justify a
larger global default. The [source archive](../benchmarks/reference/history-metric-experiment-2026-10-27.zip),
[manifest](../benchmarks/reference/history-metric-experiment-2026-10-27.manifest.json)
and [per-fit results](../benchmarks/results/history-metric-experiment-2026-10-27.json)
record the experiment.

## Untested proposal: row-space Hessian sketch

One possible way to address poor conditioning without the diagonal metric's
null-space behavior is to keep a **scalar** L-BFGS initial metric and seed its
history with a small number of GPU-computed curvature pairs. For centered
binary data `Xc`, a random sample-space matrix `Ω` with `k` columns, sample
weights `sw` and L2 strength `λ`, construct

```text
S  = Xcᵀ Ω
D0 = diag(sw / 4)
Y  = Xcᵀ D0 (Xc S) / sum(sw) + λ S
```

Columns of `S` and `Y` lie in the row space of `Xc` in exact arithmetic;
combining them with a scalar initial metric preserves that space, unlike a
feature-wise Jacobi metric. An implementation must reject nonfinite or
nonpositive-curvature pairs, retain the existing Armijo and direct checks, and
handle the unpenalized intercept separately. Finite-precision drift and nearly
null directions still require direct testing; this is not a proof of the
behavior of a Metal implementation.

The computation needs three batched matrix traversals and additional work and
storage that grow with `k`. Random sketches can miss small-eigenvalue modes,
so any speed gain is **unproven** and could be overwhelmed by setup. A useful
experiment would compare `k=4, 8, 16` and the existing scalar baseline on
weighted, shifted, duplicate-column and near-null problems, with independent
objective, gradient and coefficient checks. It must include full-fit time,
peak memory, cold compilation and sklearn comparisons. The separate larger-history
experiment above does not establish that the proposed sketch will work.

## Further work

Any candidate must be tested on the complete fit, including preparation,
synchronization, conversion and result export, against the current native loop
and sklearn. The accuracy panel needs shifted, weighted, ill-conditioned,
unbalanced and wide examples, original float64 inputs, and independent objective
and original-coordinate gradient checks. A faster stopping rule or a
lower-precision input is not an improvement if it silently changes the problem
or reports false convergence.
