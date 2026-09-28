# The float32 policy

MLX computes in float32 on the GPU. scikit-learn computes in float64. That difference is
not a detail to be papered over with a loose tolerance — it decides which parity claims
mlxlearn can honestly make, and which `check_estimator` checks can pass.

This document is the decision, written down before the estimators were implemented, so
that "we xfail that one" is a policy and not a reaction to a red test.

## 1. The dtype contract

| Boundary | dtype |
|---|---|
| Input `X`, `y` | anything scikit-learn accepts; float64/int/bool are converted |
| Computation on the MLX device, GPU or CPU | **float32**, always |
| Host-side arithmetic around it | **float64**: the L-BFGS bookkeeping, the SMO dual variables and gradients, the contraction of SVC decision values, the rbf entries SVC recomputes directly (§2.4), neighbor votes, and all of `LogisticRegression`'s inference, which is scikit-learn's own NumPy code on `coef_` |
| Returned distances, probabilities, decision values | **float64** |
| Fitted attributes (`coef_`, `intercept_`, `dual_coef_`, `support_vectors_`) | **float64** |
| Returned indices, labels | `int64` / the input label dtype, matching scikit-learn |

Returning float64 from a float32 computation is not a claim of float64 accuracy. It is
there because scikit-learn's ecosystem — metrics, meta-estimators, `Pipeline` — is written
for float64, and handing it float32 causes dtype churn far from mlxlearn. The ancestor
leaked float32 out of `kneighbors` and every downstream tolerance had to absorb it. The
accuracy actually available is described below, not by the output dtype.

**Values float32 cannot hold are rejected, not converted.** A value that is finite in
float64 but overflows float32 (`|x| > ~3.4e38`) raises `UnsupportedInputError` with
reason `"float32-overflow"`; it is never silently turned into infinity. The check reads
the input the way scikit-learn is about to: float64 arrays, object arrays, and pandas
frames that mix float64 with bool or nullable columns (which NumPy would turn into object
arrays) are all range-checked. float32, integer and boolean data cannot reach the limit
and are not scanned.

What happens next depends on when it is found:

- **At `fit`** it is a capability mismatch. A directly imported estimator raises;
  under `patch_sklearn()` the fit falls back to scikit-learn's float64 path, subject to
  `fallback_policy`. A patched fit small enough to go to scikit-learn on size alone is
  decided from the input's shape, where it has one, before any scan — so it never
  reaches the check, and `fallback_policy="raise"` does not trip on it.
- **At inference** it is raised, in both layers. The backend is sticky: a model fitted on
  MLX has no scikit-learn state to hand the call to. The neighbor estimators and `SVC`
  raise for a query float32 cannot represent, and also for a query so far from the
  training data that the scale chosen at `fit` cannot hold it — for `SVC`, one whose inner
  product with a support vector float32 cannot hold (§2.4). `LogisticRegression`
  predicts with scikit-learn's float64 code and has no such limit.

**Values whose squares leave float32.** A value can fit in float32 while its square does
not: `|x| ≈ 1e19` squares to `1e38`, and a sum of a few of those is past float32's range;
`|x| ≈ 1e-20` squares into the subnormal range, which the GPU flushes to zero. A distance
or a Gram-form kernel on such data comes out infinite or zero, silently. mlxlearn scales
the data by an exact power of two before anything is squared — the neighbor index by the
training data's span around its mean, `SVC` by its kernel frame (§2.4), with the scale
folded back into `gamma`. Scaling by a power of two is exact, so data already in range is
unchanged bit for bit, and data around `1e19` and `1e-20` is served (both are tested).
What still does not fit raises `"float32-overflow"`: neighbor training data spanning more
than `FLOAT32_MAX / 8` from its mean, or spanning so far beyond the gaps between its rows
that no scale holds both (§2.1), and a linear-kernel `SVC` row whose squared norm
exceeds float32 — the linear kernel has no `gamma` to absorb a scale. mlxlearn also
defines a squared-range guard with a second reason slug, `"float32-underflow"`, for a
path that cannot rescale; every 0.1.x estimator rescales, so none of them raises it.

**Sparse pandas frames are sparse input.** A `DataFrame` whose *every* column has a
`SparseDtype` counts as sparse, by scikit-learn's own rule (`check_array` converts such
a frame to a sparse matrix): Layer 1 raises `SparseInputError`, and patched mode falls
back. A frame with only some sparse columns is dense input, as it is to scikit-learn.

**Devices.** `device` chooses where the work that dispatch sends to MLX runs: the GPU, or
MLX's CPU backend. Both run the same float32 code, so `device="cpu"` is not a route to
float64 — the float64 answer is stock scikit-learn's (§3.4). The two devices can round a
long sum in a different order, so results can differ in their last bits; for the neighbor
estimators that is visible only where a near-tie is decided (§3.1). A
model is served on the device it was fitted on, with one exception that re-decides
nothing: a GPU-fitted model loaded in a process with no usable GPU runs the same code on
MLX's CPU under `device="auto"` or `"cpu"`, recorded in diagnostics as
`"gpu-unavailable"`; under `device="gpu"` it raises `DeviceError`. Inference that is host
code on the fitted state — `LogisticRegression`'s float64 predictions, `radius_neighbors`
on an MLX-fitted `NearestNeighbors` — uses no device, and is unaffected.

## 2. Where float32 actually costs accuracy

### 2.1 Squared distances via the expanded Gram identity

‖q − t‖² = ‖q‖² + ‖t‖² − 2⟨q, t⟩ is exact in real arithmetic and hostile in float32. When
q and t are close, a small true value is obtained by subtracting two large nearly equal
ones, so the absolute error is on the order of `eps · ‖q‖²` ≈ `1.2e-7 · ‖q‖²`. For data
centered far from the origin this swamps the true separation entirely — differences
collapse to exactly equal, or to negative values that need clamping.

The identity is one matmul, so mlxlearn still uses it — but only to *select* candidates,
and then it proves the selection right:

1. **Translation and scaling.** Distances are translation-invariant, so the training mean
   is subtracted from the training and the query rows before the identity is applied. That
   moves ‖·‖² from "distance from the origin squared" down to "variance scale", where the
   identity is far better behaved. float64 input is translated by its float64 mean, with
   the rows and the mean each carried as a float32 value plus a float32 residual, so the
   translation is good to about 2⁻⁴⁸ of the coordinate and a large constant offset — Unix
   timestamps, projected coordinates — costs nothing: timestamps a few seconds apart
   certify on the first pass. (Rounding the mean to float32 first left up to 64 of offset
   near 1.7e9 in every row, and sent every such query to the second pass at 4–5× the
   time.) The translated rows are then scaled by a power of two, which is exact, so that
   data around 1e19, whose squares overflow float32, and data around 1e-20, whose squares
   underflow it (the GPU flushes those to zero), both come back into range. Only as far as
   that: ordinary data is not scaled at all, and a large span is scaled down just until its
   squares, with room for queries 2²⁰ times farther out, stop overflowing. Scaling down
   shrinks the gaps between neighbors too. Normalizing every span to 1, as mlxlearn once
   did, let a single unmasked fill value of 1e20 among unit-scale rows push their squared
   gaps below float32's normal range, and the GPU answered every query with zeros.
2. **Over-selection.** The identity selects `k + margin` candidates per query, the margin
   growing with `k` and with the number of features.
3. **Refinement.** The candidates' distances are recomputed directly as `Σ(t − q)²` from the
   *untranslated* coordinates. Translation rounds every coordinate at the precision of its
   distance from the mean, which for multi-modal or heavy-tailed data is far coarser than
   the input itself; subtracting the raw values rounds once, relative to the difference.
   float64 input keeps a float32 residual beside its float32 value, so the difference is
   taken to about 2⁻⁴⁸ of the coordinate. The refined values are what get ranked and
   reported.
4. **Certification.** For every query, a rigorous bound on the identity's float32 error —
   `γ_{d+2}·(‖q‖² + ‖t‖²)`, combined with the reverse triangle inequality so that one
   far-out training row cannot spoil it for everyone — must prove that no training row left
   out of the candidates can come within the k-th refined distance. The test is strict: a
   row that might merely *tie* the k-th is uncertain too, because §3.1 decides ties by
   index, and only a search that saw the tied rows knows their indices.
5. **Fallback.** A query the certificate rejects is searched again with at least 256 more
   candidates and, if that is rejected too, exactly: direct differences against every
   training row, on the device, in budgeted chunks, ranked by a composite
   `(distance, index)` key. Small-integer data — binary, one-hot, counts — makes the
   identity exact outright, and takes that exact search directly.

A fixed over-selection margin was not enough, and nothing checked it. Where the Gram noise
exceeds the gaps between neighbors — tight clusters, near-duplicates, one unscaled column
among unit-scale ones — it returned wrong neighbor sets without a word: 290 of 300 queries on
1000 clusters with a spread of 1e-3, 12% of queries on data with a single column spanning
0–1e4. The certificate turns each of those into a second pass rather than a wrong answer.

**What it costs.** On ordinary data — Gaussian, uniform, standardized features, normalized
embeddings, with or without a far-out training row — every query certifies on the first pass,
and the certificate is host arithmetic on one number per query. Where the old margin failed,
correctness costs the second pass. Measured on 50 000 training rows and 4096 queries (2048
for the last case), against the old (wrong) query time: 1.5× for 1000 clusters of spread
1e-3 at 32 features, 1.7× for `make_blobs(cluster_std=0.01)` at 16, 1.6× for the 0–1e4
column at 16, and 2.2× for two clusters at 0 and 1000 at 4 features. That last case needs
the exact search for every query — direct differences against every training row — so its
cost grows with the number of features: 2.8× at 16, 5× at 64. So does a tie at the k-th
distance, which the test above counts as uncertain: a training row repeated more than
about `k + 4` times sends every query whose k-th neighbor is one of its copies to the
wider pass, 1.5× the old time for 20 copies of each row at 16 features. The old answer
there was right on the GPU only because MLX's GPU partition happens to keep the lowest
indices among equal values; on the CPU it was not. Binary and small-integer data got
*faster*, because it no longer goes through the identity at all. A far-out row is not
ordinary once it moves the mean itself — a fill value of 1e10 among a few thousand
unit-scale rows does: every translated norm then carries the shift, the noise band
outgrows the gaps, and every query takes the exact search. When a call needed the wider
pass or the exact search, its diagnostics event says for how many rows.

What scaling cannot bring into range is refused (§1): refinement subtracts two translated
coordinates in float32, so training data spanning more than `FLOAT32_MAX / 8` (about
4.3e37) from its mean is a capability mismatch at `fit` — an `UnsupportedInputError`
directly imported, a fallback to scikit-learn under `patch_sklearn()` — and a query too far
from the training data for the scale chosen at `fit` is an error at inference. So is
training data whose span dwarfs the gaps between its rows — a fill value of 1e30, or the
NetCDF default of 9.97e36, left among unit-scale rows: no power of two keeps the span's
squares finite and the gaps' squares out of the underflow range (a scaled squared distance
below 2⁻⁹⁶) at once. When the span has to be scaled down, a sample of 64 training rows is
searched at `fit`, and if their own nearest neighbors come within 2¹⁰ of that limit, the
fit is refused the same way. A query that still lands nearer a training row than the limit
is an error at inference, never a distance of zero; an exact duplicate of a training row
is the one exception. Both devices refuse the same data, and within the limit both answer
it to float32 rounding.

**What remains.** Exact means exact with respect to the refined float32 distances. Two
training rows whose true distances from a query differ by less than float32 can resolve
(about 1e-7 relative) are ranked by their refined values rather than their true ones — the
storage limit of §3.4, not a selection error. `device="cpu"` runs this same float32
algorithm on MLX's CPU, so it is not a float64 escape hatch; the float64 answer is stock
scikit-learn's.

None of this applies to `radius_neighbors` and `radius_neighbors_graph`, which are
scikit-learn's own code run on the fitted state (§3.4).

### 2.2 Accumulation length

A `float32` sum over `n` terms carries relative error growing like `n · eps`. For
n = 10⁶ that is ~0.1 at the worst case and much smaller in practice, but it is why
mlxlearn does not claim `rtol=1e-7` parity on large reductions.

### 2.3 Iterative solvers

A float32-accumulated objective cannot resolve gradients much below `1e-6 · max|g₀|` — a
millionth of the largest gradient component at the starting point. `LogisticRegression`'s
L-BFGS evaluates the objective and its gradient in float32 on the device, and makes every
decision about steps and stopping in float64 on the host, so a rounding wobble in one inner
product cannot pass for convergence. Near the floor it does two things:

- **Approximate Wolfe acceptance** (Hager & Zhang, 2005). When a step's decrease is smaller
  than the objective's own resolution (`1e-7` relative), the value cannot certify it, but
  the slope still can: a step that meets the strong-Wolfe curvature condition is accepted
  as a descent step.
- **A floor stop.** When the objective can no longer resolve the progress the gradient
  predicts, the run stops with the stop reason `"float32-floor"` instead of wandering at
  noise level until `max_iter`.

So a `tol` below what float32 can resolve — `tol=0`, say — is honored as "run to the
float32 floor" rather than rejected, and `n_iter_` reports the steps actually accepted. A
floor stop with `max|g| ≤ 1e-6 · max|g₀|` has done everything float32 allows: it emits no
`ConvergenceWarning`, and is recorded on the fit's own diagnostics event (`fit_event_`),
whose `extra` gains `stop_reason="float32-floor"`, `n_iter`, `n_fev`, `grad_max_abs`, `tol`
and `at_resolution_floor`, and whose `detail` explains the stop. A floor stop above that
level is a stall. It warns, as a `max_iter` or line-search-failure stop does, and those are
recorded the same way.

### 2.4 Kernel evaluation (`SVC`)

LIBSVM computes each kernel entry in double and caches it rounded to C `float`. mlxlearn
computes the entries in float32 on the device, inner products included, and keeps the
dual variables, gradients and working-set arithmetic in float64, as LIBSVM does — on
NumPy 1.x as well as 2.x. Decision values contract the float32 kernel entries against
float64 coefficients in float64, so the float32 error that reaches the output is the
kernel's. Most of it is the rounding of inner products — of `⟨a, b⟩`, or of the rbf squared
distance `‖a‖² + ‖b‖² − 2⟨a, b⟩` (§2.1) — measured at up to about `7u·(‖a‖² + ‖b‖²)` on
either device, from 2 to 4 096 features. That error grows with the rows' magnitude, not
with their difference, while the model learns from their differences. The frame, the rbf
refinement and the resolution check below keep the one below the other; the self-entries
keep the solver consistent with what remains.

**The compute frame.** Before anything is rounded, the data is mapped exactly, in float64,
into a *compute frame*. For the rbf kernel it is translated by the training mean first,
so a large offset never reaches float32. For every kernel but the linear one it is then
scaled by the power of two that puts the largest mapped entry in `[0.5, 1)`, and the scale
is folded back into `gamma` (`γ′ = γ / s²`), so the kernel is unchanged. The scale is
exact: data scaled by any power of two fits bit-identically, and poly, sigmoid and linear
results for data already in range are bit-identical to unscaled evaluation. The rbf
translation is not a no-op against 0.1.0a2, which cast first and subtracted a float32
mean: the mean is now subtracted before rounding, which changes the float32 inputs
slightly and can move where inside `tol` the solver stops — measured up to 6e-4 in the
decision values on `N(3, 1)` data. Data around `1e±19` is served,
and so is a `gamma` outside float32's own range (`gamma="scale"` on data around `1e-20`
resolves past float32's largest value, and folds back to order one).

**Rows close together, far from the mean (rbf).** Centering keeps the identity's error
small only near the training mean. Two rows close to each other but far out — tight
clusters far apart with `gamma` matched to the clusters, or an explicit `gamma` on
unscaled multi-modal data — got kernel entries wrong by up to 0.58, and fits returned
decision values 3–6 away from scikit-learn's, silently, under `patch_sklearn()` too. So
every entry whose identity error, in the exponent, can exceed `1e-4`, and whose value,
computed or true, can exceed `1e-8`, is recomputed on the host from the float32 rows, in
float64 — exact, for the values the device holds (§3.4), to float32's own rounding.
Distance is at least the difference of the norms, so only rows in a band of similar norms
can qualify: a lone far-out row costs its own band, not a pass over the Gram, and on
ordinary data the test never opens — on standardized data at `gamma="scale"` the bound
stays near `1e-5` up to 4 096 features. Measured: two clusters 2 000 apart, decision values
within 1.1e-3 of scikit-learn's (6.1 before); 40 clusters at 2 400 × 512 on the GPU, within
3.1e-4 (3.5 before), in 0.14 s against 0.05 s unrefined and 0.39 s for scikit-learn. Where
every entry inside a cluster needs it — two clusters of 2 000 or 5 000 rows each — the fit
took 2.6–3× as long as unrefined, 1.4–1.7× scikit-learn's time. The same test runs on
queries at inference.

**The raw inner product (linear, poly).** These kernels are not translation-invariant, so
the frame cannot centre them: their float32 error scales with the offset, while the
curvature the solver learns from, `K_ii + K_jj − 2K_ij`, scales with the spread. On
`make_classification · 0.01 + 20` a cubic kernel near 6.5e7 came out wrong by 40 against
curvatures of 3–100. mlxlearn compares that error at the largest row norm with the mean
squared distance between rows — and, for poly, `coef0`'s own rounding with `gamma` times
it — and past `1e-3` refuses the fit with `UnsupportedInputError`, reason
`"float32-cancellation"`, which patched mode hands to scikit-learn. The ratio overstates the
measured error per median curvature about threefold. On the linear kernel, where
scikit-learn still converges on such data, the decision values moved by 1.4e-3 of their
range at a ratio of 1.7e-4, by 1.1e-2 at 1.7e-3, and by 6e-2, with 98.8% of labels
agreeing, at 1.5e-2.
`sigmoid` is not checked: on the same offsets it agreed with scikit-learn to 3e-4 or
better. The test is global, so structure that lives at a much finer scale than the data's
overall spread — tight clusters far from the origin — is not seen by it; standardize such
data before a poly kernel, or use scikit-learn.

**Self-entries.** The solver takes its curvature from the closed-form diagonal, in
float64, and its gradient updates from the rows. Every row carries that closed form,
rounded to float32, in its self-entry, so the two describe one quadratic. Left to the
float32 evaluation they disagreed — by 31 on that poly kernel, where each step then solved
one quadratic and applied another, and the fit circled its whole 10⁷-iteration budget:
376 s against scikit-learn's 0.001 s. On ordinary data the self-entries move by a rounding,
which can move where inside `tol` a fit stops: measured up to 2.3e-3 in the decision values
of in-range poly fits and 1.5e-2 in a linear one (whose dual is flat, §3.2), inside the §3
tolerance on either side.

What the frame cannot represent is a capability limit at `fit` — a `gamma` or `coef0` not
finite in float32 once folded (`UnsupportedParameterError`), a linear-kernel row whose
squared norm exceeds float32 (`UnsupportedInputError`) — and an error at inference, for a
query whose inner product with a support vector float32 cannot hold. A query far from the
training data is otherwise answered, as scikit-learn answers it: its rbf entries are
exactly 0 and its sigmoid entries ±1, so a CMIP `1e20` or netCDF `9.97e36` fill value in one
query row no longer fails the batch. A non-finite decision value is an error, never a
label.

**Shrinking.** `shrinking=True`, scikit-learn's default, runs LIBSVM's shrinking
heuristic: samples pinned at a bound are set aside between gradient reconstructions. The
optimum and the stopping test are unchanged — the test is evaluated over every sample
before the solver may stop — but, as in LIBSVM, the path to the optimum can change on
every kernel, and with it where inside `tol` the solver stops. It moves by the same order
as LIBSVM's own shrinking on the same data, more as `C` grows, and most on the flat duals
of the polynomial and linear kernels. Measured through `SVC`, LIBSVM's movement on the
same data in brackets: at 2 000 × 10 the decision values moved by up to 4.5e-3 (3.8e-3)
for poly at `C=1` and 2.3e-2 (3.2e-2) at `C=10`, while rbf fits were often bit-identical
and moved by at most 6.2e-4 (7.5e-4) at `C=10` and `C=100`; at 2 000 × 20, linear fits
moved by up to 6.3e-3 (1.6e-2) at `C=1` and 1.0e-2 (2.0e-2) at `C=10`. The number of
support vectors changed by one in some of those fits, and a training point whose decision
value was that close to zero could change label, as LIBSVM's did. The gradient it rebuilds
is summed in a fixed order, so shrinking adds no dependence of its own on `cache_size` or
on the NumPy/BLAS build. The tolerances in §3 hold either way.

**`cache_size`.** When the row cache can hold a whole one-vs-one subproblem, its Gram is
built in one blocked matmul; otherwise rows are evaluated on demand, and the two can round
differently. On the GPU they produced bit-identical rows at narrow widths (all four
kernels at 2 000 × 10), a property of MLX's matmul rather than a promise; at a thousand
features and more MLX picks another matmul kernel for some shapes — the Gram's, or a batch
of 33 rows, the prefetch's own — and at 2 048 × 2 048 and 3 000 × 4 096 `cache_size=5`
moved the decision values by up to 9.2e-4. On MLX's CPU device they accumulate in
different orders at any width: at 2 000 × 10 the decision values moved by up to 2.2e-3 and
the iteration count changed. Every fit is KKT-optimal and inside the §3 tolerance.
scikit-learn is bit-identical across `cache_size`; pin it for bit-reproducible fits, on
either device.

## 3. Tolerances the test suite uses

These are the numbers, and the reasoning for each. Tests do not invent their own.

**Precondition: both fits must have converged.** When either solver stops on `max_iter`
instead of its tolerance, the comparison is between two arbitrary intermediate iterates and
no tolerance can be stated for it. A randomized 48-configuration sweep at the default
`max_iter=100`, with feature scales varying over two orders of magnitude, produced apparent
violations up to `relJ = 1.8e-3` and `coef |Δ| = 0.145` — and on inspection *both* solvers
had hit the iteration limit and *both* had emitted `ConvergenceWarning`. Raising the budget
on the worst case: mlxlearn converged in **113** iterations, scikit-learn in **251**, and the
agreement fell inside the table below (`relJ = 2.3e-6`, `coef |Δ| = 1.5e-4`).

At the truncated budget mlxlearn's objective was the **lower** of the two. So the fair claim
is not "mlxlearn is within tolerance of scikit-learn at any budget" — it is "for converged
fits mlxlearn agrees to the tolerances below, and at equal truncated budgets its objective is
no worse." Both are tested.

| Quantity | Comparison against scikit-learn | Rationale |
|---|---|---|
| Neighbor **sets** (per row) | `assert_array_equal` on the sorted row — exact, except which of several rows tied at the k-th distance is returned | Which points are nearest is not a matter of opinion; which of *equally* near points makes the cut is — see §3.1 |
| Neighbor **index order** | exact where distances are strictly separated; set equality where they tie | See §3.1 — scikit-learn does not define an order for exact ties |
| Neighbor **distances** | `atol=1e-6, rtol=1e-6` | Refined float32, relative to the data scale |
| Classifier **predictions** | ≥ 99.5% label agreement | Labels are discrete; disagreements concentrate at the decision boundary. Measured: 0.9975 worst case over 40 randomized logistic configurations, 1.0 typical |
| Neighbor-vote **probabilities** | `atol=1e-5` | A sum of exact weights over an exact neighbor set |
| Linear-model **probabilities** | `atol=2e-3` on the fixed grid (n ≤ 5000, d ≤ 20) at default `tol`; on any converged problem at `tol=1e-6` | These inherit the coefficient error rather than being computed independently, so they cannot be tighter than the coefficients allow — and at default `tol` that error is where lbfgs stops, which a wide design sums over more features. Measured over the grid: 3e-4 typical, 1.34e-3 worst (multiclass with a 10× sample weight) at default `tol`; 6e-5–1.9e-4 at `tol=1e-6`. On 25 000 × 1 000, which the crossover sends to MLX: up to 4.9e-3 at default `tol`, where stock lbfgs and stock newton-cg differ from each other by as much, and 7.2e-5 at `tol=1e-6`. The objective assertion is what judges such a problem at default `tol` |
| Linear **coefficients** | `atol=2e-3, rtol=5e-3` | Dominated by *where the solver stops*, not by arithmetic — and on a rank-deficient design only the penalty pins them down at all. Measured: 5e-4 typical, 1.15e-3 worst over the fixed grid |
| **Objective value** at the solution | `rtol=1e-5` | The objective is far better conditioned than its argument — this is the assertion that actually catches a wrong solver. Measured: worst relative difference 2.9e-7 across every non-separable logistic case |
| SVM **decision values** (PSD kernels) | `atol=2e-2, rtol=5e-3` | Dominated by the SMO stopping point, not by arithmetic: both solvers stop when the maximal KKT violation drops below the shared `tol` (default `1e-3`), and the dual variables still have freedom there. Measured worst 1.56e-2 on a ±4.5 value range (linear, C=10); rbf cases sit at 5e-7–7e-4 |
| SVM **decision values** (multiclass `kernel="sigmoid"`) | not asserted — see §3.2.1 | The sigmoid kernel is indefinite, so the dual is non-convex and two correct solvers reach different local optima. Measured 0.63 on the worst one-vs-one pair, and it does not shrink with `tol`. Binary sigmoid cases do meet the PSD tolerance above and are asserted at it |
| SVM **support-vector sets** | ≤ 5% symmetric difference | Measured: counts agree essentially exactly (1311 vs 1311, 827 vs 827; worst disagreement one point out of 548). Which points sit exactly on the margin is genuinely ill-conditioned, so exact set equality is not something either solver promises |
| SVM **KKT conditions** | margins within `2·tol` at mlxlearn's own solution | The assertion that does not appeal to scikit-learn at all, and the one that catches a solver agreeing with LIBSVM for the wrong reason. The stopping rule bounds every margin by `tol` in exact arithmetic; the second `tol` is the float32 kernel's. Measured worst 4.9e-4 (CPU device), 5.2e-4 (GPU), 7.9e-4 over a 72-configuration sweep |

**Coefficients are the weak assertion; objective value is the strong one.** For an
ill-conditioned problem two solvers can reach very different coefficients that are equally
optimal. A test that only compares `coef_` therefore fails for a correct implementation and
passes for a subtly wrong one. Every parity test that compares coefficients also compares
the objective.

**Objective comparisons are one-sided where mlxlearn can legitimately win.** On data with a
large constant offset mlxlearn reaches a *lower* objective than scikit-learn — 0.6104452
against 0.6104529 at an offset of 1e5, converging in 113 iterations against 175 — because
it centers the design and scikit-learn does not. A symmetric tolerance would fail mlxlearn
for being better, so those assertions are `j_mlxlearn ≤ j_sklearn · (1 + rtol)`.

### 3.1 Neighbor ties: mlxlearn is stricter than scikit-learn

mlxlearn returns the first `k` training rows under `(distance, training index)` — a total
order over the *whole* training set, not merely an ordering of whichever rows it returns.
Rows tied at the k-th distance are decided by index too, so `kneighbors(k)` is always a
prefix of `kneighbors(k + m)`, and no answer depends on block sizes. That includes
`kneighbors()` with no query, which is the same order with each row itself left out: a
row with more than `k` exact duplicates before it gets the first `k` of them. "Distance" is the
refined float32 value of §2.1. Where that value is exact — integer, binary and one-hot
features, halves and quarters — the answer is also identical on the GPU and the CPU;
elsewhere the two devices may round a long sum differently, and so break a near-tie
differently. (Until the certificate existed, none of this held on tie-heavy data: a tie
group larger than the over-selection margin was cut arbitrarily, and the cut moved with
the device, the block size and `k`.)

scikit-learn's brute-force path partitions and then sorts the candidates with a
non-stable sort, so it has **no defined order among exactly equal distances** — neither
within a returned row nor at the k-th boundary. Measured on 50 000 points at coordinate
scale 10⁴ with 200 queries and k=7: neighbor sets agreed on every row, distances agreed
exactly (`max |Δ| = 0`), and 26 of 1400 cells differed — every one of them a swap between two
neighbors at an identical distance.

So the parity assertion is: **distances are exact; sets are exact except for which of
several rows tied at the k-th distance is returned; index order is exact wherever distances
are strictly separated.** Asserting elementwise index equality would demand that mlxlearn
reproduce an ordering scikit-learn does not itself guarantee across releases.

Ties *within* the returned set are invisible downstream: predictions and probabilities
aggregate over the set. Ties *at the boundary* are not. scikit-learn keeps an arbitrary
subset of the tied rows and mlxlearn the lowest-indexed ones, so on tie-heavy data a vote
can legitimately differ between the two; mlxlearn's is the reproducible one. The parity
tests assert predictions elementwise on continuous data, where boundary ties do not occur.
A vote tied between classes goes to the class that sorts first in `classes_`, as it does
in scikit-learn.

### 3.2 SVC: the dual is flat, so compare the primal

For the rbf and sigmoid kernels, tightening `tol` from `1e-3` to `1e-5` shrinks the
decision-value gap against scikit-learn a hundredfold (8.97e-3 → 1.30e-5 and
6.33e-4 → 7.51e-6). That is the signature of a stopping-point difference: the underlying
solutions agree, the solvers just stopped at different moments.

For the **linear** kernel the gap does not shrink at all — 1.56e-2 at `tol=1e-3`, 1.71e-2 at
`tol=1e-5`. With n ≫ d the Gram matrix is rank-deficient, so the dual has a flat optimum:
many α vectors map to essentially the same primal solution, and the KKT rule both solvers
stop on measures *dual* violation.

The question "did mlxlearn solve the problem?" is therefore answered by the primal
objective `½‖w‖² + C·Σ hinge`, which is what both are minimizing. Measured: agreement to
**5.2e-07 relative**, worst 5.9e-06 at C = 10. Comparing decision values alone would report
a problem the objective shows does not exist — the same trap as comparing `coef_` for a
linear model.

### 3.2.1 The sigmoid kernel is indefinite, so its dual is not convex

`tanh(γ⟨x, y⟩ + r)` is not positive semi-definite for most parameter choices. The SVM dual
is then a **non-convex** quadratic program with multiple local optima, and two correct
solvers can land on genuinely different solutions — not different points on one flat
optimum, as with the rank-deficient linear case, but different optima.

Measured on the test's 5-class sigmoid problem (2000 rows, 15 features): the worst
one-vs-one pair's decision values differ by 0.63, and the gap does **not** shrink with
`tol`. The disagreement is isolated to that pair, whose dual objective differs by
`rtol = 2.3e-04` — and there mlxlearn's objective is the **higher** of the two
(−581.536821 against −581.671780). Neither solver is wrong: both solutions satisfy the KKT
conditions to within `tol`. On a 1500-row variant of the same data the order reverses, and
mlxlearn's is the lower. Which local optimum a solver reaches depends on the data.

So for multiclass `kernel="sigmoid"` the decision-value tolerance in the table above does
not apply and is not asserted. What is asserted is that mlxlearn's solution is a KKT point
of every pair (margins within `2·tol`), that each pair's dual objective is within `rtol=1e-3`
of LIBSVM's in either direction, and that predictions agree. The objective bound pins the
behavior on that dataset; for a non-convex problem it is not a guarantee either solver
could make in general. LIBSVM makes the same caveat about indefinite kernels; mlxlearn
states it rather than tuning a tolerance until the test goes green.

### 3.3 Where the objective assertion does not apply

On **perfectly separable** data at large `C` the minimizer is at infinity: the objective
approaches zero and both solvers stop on an absolute gradient tolerance somewhere along the
way. A relative comparison of two nearly-zero numbers is then meaningless. Measured at
n = 2000, d = 10, `C = 1e6`: scikit-learn reaches 2.45e-05, mlxlearn 2.73e-05 — a relative
difference of 0.114 from an absolute gap of 2.8e-06, with **100% label agreement** and
`predict_proba` differing by 1.8e-05. At `C = 1` on the same data mlxlearn's objective is
*lower* than scikit-learn's.

So: objective parity is asserted at `rtol=1e-5` on non-separable data, and on separable
data the assertions are the absolute gap, the predictions, and the probabilities. Both
solvers must terminate and produce finite coefficients — that part is never relaxed.

### 3.4 A limit that is about storage, not arithmetic

Some adversarial cases cannot be rescued by any kernel. Take 120 000 points spaced `1e-8`
apart and shift them to coordinate `1e4`: float32 has a spacing of about `1e-3` there, so
handed over as float32 the *input* has collapsed to three distinct values before a single
distance is computed. mlxlearn returns the correct answer for the data it was actually
given — verified against a float64 reference computed on the float32-rounded data — but it
cannot return the answer for values the input never held.

Handed over as float64, a constant offset like that one is no longer thrown away first.
Wherever the model is invariant to it, it is removed in float64, before anything is
rounded to float32:

- The **neighbor estimators** translate by the float64 training mean and keep a float32
  residual beside each float32 value, the mean's included, so the translation and the
  refined differences are taken to about 2⁻⁴⁸ of the coordinate (§2.1). 5 000 Unix timestamps within one hour, which collapse to 29
  distinct values when cast first, return their exact neighbors.
- **`LogisticRegression`** centers float64 input in float64 when it fits an intercept, and
  rounds only the centered values; the intercept is un-centered with exactly the mean that
  was subtracted. Without an intercept there is nothing to absorb the shift, so the design
  is used as given.
- **`SVC`** translates rbf input by the float64 training mean and scales every non-linear
  kernel's input by a power of two (§2.4).

What stays bounded by float32 is the *spread* of the data around that offset — 24 bits of
it — and any input that arrives as float32. The exception is `radius_neighbors` and
`radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`, which are scikit-learn's own
code run on the fitted state: they compute against the index's float32 training copy —
the input rounded as given, neither translated nor refined by its residual — with the
query cast to float32 to match. That is what keeps them at stock speed, and it bounds
their precision by the data's raw magnitude, not its spread: Unix timestamps round to
128-second steps there and UTM northings to half-metre ones, so points within about that
step of the radius can land on the wrong side of it. Subtract such an offset before
fitting, or use stock scikit-learn. A sparse query that scikit-learn's brute force answers
on an MLX-fitted model — to the radius methods in both layers, and to `kneighbors` and the
predictions built on it under `patch_sklearn()` — is cast to float32 and computed against
the same copy. A float64 query float32 cannot hold is therefore an `UnsupportedInputError`,
as it is for `kneighbors`.

The rule this yields: mlxlearn's accuracy claims are relative to the float32 representation
of the input after its exact float64 translation and scaling — for float32 input, the input
itself. When the data's own dynamic range exceeds that, the answer is scikit-learn itself —
stock, unpatched scikit-learn (import it before `patch_sklearn()`, or call
`unpatch_sklearn()`). `device="cpu"` is not a way to get it: it runs the same float32
algorithm on MLX's CPU backend, with the same rounding. Under `patch_sklearn()`, problems
below the measured crossover are handed to scikit-learn automatically, on either device.

## 4. `check_estimator` policy

mlxlearn runs scikit-learn's full estimator check suite against **both** layers — the
strict Layer 1 classes and the patched Layer 2 classes — because the two advertise
different tags and a tag that lies is worse than a missing feature.

**The mlxlearn-specific xfail list is empty.** Every suite passes: five estimators in each
of the two layers, each run twice — at the default configuration, and again with the MLX
GPU path forced, because the check data is smaller than every crossover and would
otherwise never meet MLX-fitted state behind the wide Layer 2 tags.
`check_param_validation`, which `check_estimator` does not run, is run separately for every
estimator in both layers, at the default configuration and with mlxlearn's own `fit` path
forced.

Two kinds of exemption exist and are kept strictly apart in
`tests/test_sklearn_compliance.py`:

**Inherited failures.** `check_sample_weight_equivalence_on_dense_data` and
`..._on_sparse_data` are declared by scikit-learn as expected failures for *its own* `SVC`
and `LogisticRegression` — "sample_weight is not equivalent to removing/repeating
samples." An mlxlearn estimator that subclasses one of them is not expected to pass a check
its base class does not. A test re-derives this list from scikit-learn's own source so the
exemption cannot outlive the behavior that justified it. mlxlearn's equivalent property is
asserted at float32 tolerances in `tests/parity/`.

**Rule-based xfails.** Checks scikit-learn passes and mlxlearn does not. Each must name a
rule below, and the list is asserted so a new one cannot be added silently to turn a red
test green.

**Rule A — float64 exactness.** A check that asserts float64-level agreement between two
computations of the same quantity may be xfailed. It is *not* xfailed if a
float32-appropriate tolerance would pass; in that case mlxlearn's implementation is fixed
instead.

**Rule B — dtype preservation.** A check asserting that float32 input yields float32 fitted
attributes is xfailed: mlxlearn deliberately returns float64 attributes (§1).

**Rule C — deliberate determinism (reserved).** No 0.1.x estimator draws random numbers,
so the `deterministic` and `random_state` options have no observable effect today and no
check is xfailed under this rule. It is reserved for the estimators that will: a check
asserting that `random_state=None` produces varying results may be xfailed under
`deterministic=True` (`_common/rng.py` resolves `None` to a fixed seed there), provided it
passes under `deterministic=False`.

**Never xfailed, under any circumstances:**

- `check_estimators_pickle` and every clone / `get_params` / `set_params` check
- `check_fit2d_predict1d`, `check_estimators_unfitted`, `n_features_in_` checks
- `check_classifiers_train`, `check_regressors_train` — an estimator that cannot learn is
  broken, not imprecise
- Any check about raising on invalid input

The live list lives in `tests/test_sklearn_compliance.py::EXPECTED_XFAILS`, with each entry
carrying its rule letter and a one-line justification. It is currently empty, and the goal
is to keep it that way.

## 5. What mlxlearn will not do

- **Silently downcast and claim parity.** Every deviation in this document is tested, not
  assumed.
- **Turn a value into infinity and carry on.** Values, squares and queries float32 cannot
  hold are rescaled exactly or refused (§1). The one infinity carried is exact where it
  lands: an `SVC` query so far out that its rbf squared distance overflows gets the kernel
  value `exp(−inf) = 0`, which is its value in float64 too, and only where the nearest
  support vector is past float32's `exp` (§2.4). A non-finite result is an error, never an
  answer.
- **Loosen a tolerance to make a test pass.** Tolerances come from this document. A failure
  at these tolerances is a bug in mlxlearn.
- **Offer a float64 GPU path.** MLX does not have one, and mlxlearn's CPU path is the same
  float32 computation on another device. Requesting float64 accuracy means requesting
  scikit-learn: `patch_sklearn()` users get it for problems below the crossover, and
  otherwise by using stock scikit-learn for that estimator.
