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
| Internal computation | **float32**, always |
| Returned distances, probabilities, decision values | **float64**, upcast from the float32 computation |
| Fitted attributes (`coef_`, `dual_coef_`, `intercept_`, `centroids`) | **float64**, upcast |
| Returned indices, labels | `int64` / the input label dtype, matching scikit-learn |

Returning float64 from a float32 computation is not a claim of float64 accuracy. It is
there because scikit-learn's ecosystem — metrics, meta-estimators, `Pipeline` — is written
for float64, and handing it float32 causes dtype churn far from mlxlearn. The ancestor
leaked float32 out of `kneighbors` and every downstream tolerance had to absorb it. The
accuracy actually available is described below, not by the output dtype.

Values that are finite in float64 but overflow float32 (`|x| > ~3.4e38`) are **rejected**
with `UnsupportedInputError`, not silently converted to infinity.

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
   identity is far better behaved. float64 input is translated in float64, before anything
   is rounded to float32, so a large constant offset — Unix timestamps, projected
   coordinates — costs nothing. The translated rows are then scaled by a power of two, which
   is exact, so that data around 1e19, whose squares overflow float32, and data around
   1e-20, whose squares underflow it (the GPU flushes those to zero), both come back into
   range.
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
for the last case): 1.5× the old (wrong) query time for 1000 clusters of spread 1e-3, 1.7×
for `make_blobs(cluster_std=0.01)`, 1.6× for the 0–1e4 column, and 2.3× for two clusters at 0
and 1000, which needs the exact search for every query. Binary and small-integer data got
*faster*, because it no longer goes through the identity at all.

**What remains.** Exact means exact with respect to the refined float32 distances. Two
training rows whose true distances from a query differ by less than float32 can resolve
(about 1e-7 relative) are ranked by their refined values rather than their true ones — the
storage limit of §3.4, not a selection error. `device="cpu"` runs this same float32
algorithm on MLX's CPU, so it is not a float64 escape hatch; the float64 answer is stock
scikit-learn's.

### 2.2 Accumulation length

A `float32` sum over `n` terms carries relative error growing like `n · eps`. For
n = 10⁶ that is ~0.1 at the worst case and much smaller in practice, but it is why
mlxlearn does not claim `rtol=1e-7` parity on large reductions.

### 2.3 Iterative solvers

Gradient norms below ~`1e-6 · ‖initial gradient‖` are not resolvable in float32.
`tol` values smaller than that are honored as "run to the float32 floor" rather than
rejected, and `n_iter_` reports what actually happened. An estimator that stops early
because the objective stopped moving records it in diagnostics.

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
| Linear-model **probabilities** | `atol=2e-3` | These inherit the coefficient error rather than being computed independently, so they cannot be tighter than the coefficients allow. Measured: 3e-4 typical, 1.34e-3 worst (multiclass with a 10× sample weight) at default `tol`; 6e-5–1.9e-4 at `tol=1e-6` |
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
prefix of `kneighbors(k + m)`, and no answer depends on block sizes. "Distance" is the
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
the *input* collapses to three distinct values before a single distance is computed.
mlxlearn returns the correct answer for the data it was actually given — verified against a
float64 reference computed on the float32-rounded data — but it cannot return the answer
for the float64 data it never saw.

The rule this yields: mlxlearn's accuracy claims are always relative to the float32
representation of the input. When the input's own dynamic range exceeds float32, the answer
is scikit-learn itself — stock, unpatched scikit-learn (import it before `patch_sklearn()`,
or call `unpatch_sklearn()`). `device="cpu"` is not a way to get it: it runs the same float32
algorithm on MLX's CPU backend, with the same rounding. Under `patch_sklearn()`, problems
below the measured crossover are handed to scikit-learn automatically, on either device.

## 4. `check_estimator` policy

mlxlearn runs scikit-learn's full estimator check suite against **both** layers — the
strict Layer 1 classes and the patched Layer 2 classes — because the two advertise
different tags and a tag that lies is worse than a missing feature.

**As of 0.1.0a2 the mlxlearn-specific xfail list is empty.** All ten suites pass.

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
- **Loosen a tolerance to make a test pass.** Tolerances come from this document. A failure
  at these tolerances is a bug in mlxlearn.
- **Offer a float64 GPU path.** MLX does not have one, and mlxlearn's CPU path is the same
  float32 computation on another device. Requesting float64 accuracy means requesting
  scikit-learn: `patch_sklearn()` users get it for problems below the crossover, and
  otherwise by using stock scikit-learn for that estimator.
