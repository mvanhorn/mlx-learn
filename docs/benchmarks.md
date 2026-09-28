# Benchmarks

Every number here was measured. Reproduce the tables with:

```bash
python -m benchmarks.run --profile published --report bench/published.md
```

The `published` profile holds exactly these cases, with the settings they were measured
under (Layer 1 classes forced onto the MLX device, 25 iterations). `--profile smoke` is the
quick check and `--profile full` the regression grid; see
[`development.md`](development.md#benchmarks).

**These numbers were measured on 0.1.0a2.** Several changes since then — listed under
Unreleased in `CHANGELOG.md` — move them: neighbor queries and index builds, the
logistic solver's per-iteration cost, SVC fits past the kernel cache, and the patched
handoff overhead of all three. They are re-measured on fixed hardware before the next
release, and the thresholds with them.

<!-- perf-phase: re-measure every table and crossover; knn_tree_density, knn_min_work and
the logreg thresholds were flagged for re-measurement by the groups that changed them.
Integration smoke run (shared M4 Max, 7 iterations): the gate flagged knn/kneighbors
1000x32 and 8000x32 at q=256 (0.77x, 0.67x) and logreg/fit 20000x1024 (0.46x) on the MLX
path. The KNN flags are the harness's interleaving: the GPU idles while the stock call
runs, and 0.1.0a2 measures the same way (0.64-0.76x at 8000x32 interleaved; both trees
are ~2.5x stock back to back). logreg 20000x1024 is 1.8x faster than 0.1.0a2 but still
~0.5x stock on this easy data (10 iterations), so logreg_min_samples/logreg_min_work send
it to MLX too early. Handoff flags there (logreg 5000x32, knn 50000x3, 0.82x) did not
reproduce at 41-101 interleaved reps (1.02-1.05x). -->

## Method

**Synchronized.** MLX is lazy, so a benchmark that stops the clock after issuing work
measures how long it took to *schedule* — a spectacular and meaningless number. Every
timing forces evaluation first.

**Interleaved.** mlxlearn and scikit-learn are timed A, B, A, B rather than all of one then
all of the other, so a thermal ramp or a background process degrades both sides equally
instead of whichever ran second.

**Median with its spread, and best case too.** At small sizes scikit-learn's timings are
heavy-tailed: thread-pool scheduling and garbage collection produce occasional outliers
several times the median. Measured on a 500 × 8 neighbor query, scikit-learn's median was
26.3 ms and its own minimum 1.55 ms — the same case reads as 17× on medians and 1.2× on
minima. Both are true; publishing only the first would not be. A crossover requires a win
on **both**.

**Significance.** A difference counts only when the two medians are three standard errors
apart, each standard error being that *of the median* — estimated from the interquartile
range as `1.2533 · (IQR / 1.349) / √n` — and combined in quadrature. Not the raw spread,
which for a heavy-tailed distribution declared a measured 10× "within noise"; and a real
interquartile range, not the full min–max range, so one outlier cannot hide a regression.
Fewer than five samples on either side is never significant. The `z = 3` rather than 1.96
is deliberate: the standard errors come from a handful of samples, and a nightly run gates
well over a hundred cases.

**Patched dispatch is what is gated.** The performance claim is about the classes
`patch_sklearn()` installs, and those hand sub-crossover problems to scikit-learn outright.
The `smoke` and `full` profiles measure them by default, record which backend each case
actually ran on, and the gate judges those decisions against the configured `Tuning`
rather than a crossover inferred from the run it is judging. `--layer direct` measures the
Layer 1 classes, which never dispatch to scikit-learn and use their own CPU path instead —
useful for development, misleading as a headline. The tables below are the exception, and
say so: they force the Layer 1 classes onto the MLX device to isolate the kernels from the
dispatch decision. In the `full` profile the stock neighbors baseline is scikit-learn's
default `algorithm="auto"`, which builds a KD-tree at up to 15 features; the neighbor
table below compares against its brute-force path.

**Fixed hardware.** Absolute timings only compare across runs on the same machine. Shared
CI runners do correctness only; the nightly job pins itself to fixed self-hosted Apple
silicon.

Environment for the numbers below: Apple M4 Max, 128 GiB, macOS 25.2.0, Python 3.13.3,
MLX 0.31.2, scikit-learn 1.7.2, NumPy 2.5.1.

## Measured crossovers, and the thresholds they set

| Algorithm / operation | Crossover | Best speedup | Threshold in `Tuning` |
|---|---|---|---|
| `knn/kneighbors` | ~250–1 000 samples | 17.4× | `knn_min_train_samples = 256`, `knn_min_work = 2048` |
| `knn/fit` | never | — | accepted regression, see below |
| `svc/fit` | ~2 000 samples **and** ~10⁶ elements; never for `kernel="linear"` | 2.93× | `svc_min_samples = 2048`, `svc_min_work = 1_000_000` |
| `logreg/fit` | ~20 000 samples **and** ~2·10⁷ elements | 1.34× | `logreg_min_samples = 20_000`, `logreg_min_work = 20_000_000` |
| `knn`, patched, `algorithm="auto"`, ≤ 15 features | stock's KD-tree wins once `n ≳ 512 · 2^d` | — | `knn_tree_density = 512` |
| `svc/fit`, more than two classes | the mean one-vs-one pair, `(2n/K) · d`, must clear its own floor | — | `svc_min_pair_work = 250_000` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

## Neighbors — the strong result

Query speedups against stock scikit-learn's brute-force path, forced onto the MLX device to
isolate the kernel from the dispatch decision, k = 10, 256 queries, 25 iterations:

| samples × features | mlxlearn (ms) | scikit-learn (ms) | median | best |
|---|---|---|---|---|
| 250 × 8 | 1.48 | 4.26 | 2.87× | 0.92× |
| 500 × 64 | 1.78 | 29.86 | 16.79× | 3.30× |
| 1 000 × 768 | 2.43 | 25.54 | 10.50× | 2.86× |
| 4 000 × 64 | 2.46 | 26.95 | 10.96× | 4.80× |
| 8 000 × 768 | 5.13 | 44.39 | 8.66× | 4.91× |
| 32 000 × 768 | 15.32 | 124.73 | 8.14× | 5.55× |
| 128 000 × 768 | 30.74 | 440.69 | 14.34× | — |

The MLX path wins at every size tested from 250 rows upward, on both statistics except the
smallest case. The initial threshold of 4 096 was a placeholder that gave most of this away;
it is now 256.

### KNN `fit` is slower, and that is structural

0.17×–0.36×. mlxlearn's `fit` uploads the training matrix, its transpose, and its row norms
to the device; scikit-learn's brute-force `fit` stores a reference and does no work at all.

<!-- perf-phase: stale cause. The index layout changed: the raw training rows are uploaded
once and the host's _fit_X is a read-only view of that device copy, beside the translated,
scaled copy the search runs on (plus a float32 residual for float64 input); no full-size host
copies or temporaries are allocated, and the float32 range check no longer allocates.
Measured fit peak was ~2.1x the float32 data on the device (was ~2.0x on device plus ~3x on
host). Re-measure and rewrite this section and benchmarks/gate.py's ACCEPTED_REGRESSIONS
text. -->

It is listed in `benchmarks/gate.py::ACCEPTED_REGRESSIONS` rather than left to slip through
the "no crossover measured, so not gated" branch — which is the most comfortable place for
a real regression to hide. The gate fails on any *unlisted* operation that regresses
significantly with no crossover.

The trade is worth taking whenever the model is queried at all. At 50 000 × 32: fit costs
5.05 ms against 0.88 ms, and the first query alone saves 30.8 ms.

## SVC — width decides, not sample count

The first grid was damning. (It is historical: the `published` profile reproduces the slice
below, not the whole grid.) Over 64 configurations — n ∈ {1 000, 2 500}, d ∈ {10, 20}, 2 and
4 classes, four kernels, C ∈ {1, 10} — **mlxlearn was slower in 58 of 64, worst 0.013×**. A
representative slice at 2 500 × 20, four classes:

| kernel | C | mlxlearn (s) | scikit-learn (s) | ratio |
|---|---|---|---|---|
| linear | 1 | 6.87 | 0.24 | 0.04× |
| linear | 10 | 42.82 | 0.68 | **0.02×** |
| rbf | 1 | 0.110 | 0.044 | 0.40× |
| rbf | 10 | 0.319 | 0.053 | 0.17× |
| poly | 10 | 0.394 | 0.060 | 0.15× |
| sigmoid | 1 | 0.054 | 0.054 | 1.00× |

But that grid was too narrow to see the real variable. Holding the sample count fixed and
widening the data:

| case | mlxlearn (s) | scikit-learn (s) | ratio |
|---|---|---|---|
| rbf, 4 000 × 32 | 0.16 | 0.07 | 0.47× |
| rbf, 4 000 × **256** | 0.20 | 0.57 | **2.93×** |
| linear, 4 000 × 32 | 45.56 | 1.59 | 0.03× |
| linear, 4 000 × 256 | 178.81 | 11.45 | 0.06× |

Same `n`, 6× the speedup swing. SMO is sequential — each iteration picks a working pair
from the current gradient and cannot start before the previous one finished — so MLX can
accelerate the kernel *rows* an iteration needs but not the iterations themselves, and
LIBSVM's inner loop over a cache-resident row is hard to beat per iteration. What decides
the outcome is therefore whether a kernel row is expensive enough to be worth dispatching,
and that is a question about the number of features.

The linear kernel loses at every width, because its rows are the cheapest of any kernel and
per-iteration dispatch is nearly all of the cost. It also degrades with C, which increases
the iteration count.

A sample-count threshold cannot express any of this — which is exactly why the original
`svc_min_samples = 2048`, with no work floor, routed every one of those measured losses onto
the MLX path.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check. mlxlearn
solves the right problem; it is slow getting there.

Consequences, applied rather than merely noted:

- The crossover is on **work**: `svc_min_samples = 2048` *and* `svc_min_work = 1_000_000`
  elements, both required. Checked against the measurements — 2 500 × 20 (5·10⁴),
  4 000 × 32 (1.3·10⁵) and 1 000 × 20 (2·10⁴) all fall below and go to scikit-learn;
  4 000 × 256 (1.0·10⁶) clears it and keeps the 2.93×.
- **`kernel="linear"` never takes the MLX path**, at any size.
- **Many classes are a pile of small problems.** One-vs-one solves `K(K−1)/2` subproblems
  of about `2n/K` rows each, and each pays its own Gram and its own loop, so a problem whose
  total clears both floors can still be slow. The mean pair's work, `(2n/K) · d`, must also
  clear `svc_min_pair_work`; for two classes that is the whole problem, so binary routing
  is unchanged.
- Under `patch_sklearn()` this is invisible: those problems go to scikit-learn. A directly
  imported `mlxlearn.svm.SVC` is Layer 1 and by design never dispatches to scikit-learn, so
  it will use its own CPU path and be slower. That is the documented Layer 1 contract, and
  it is a real reason to prefer the patched entry point for SVC in 0.1.0a2.

<!-- perf-phase: partly stale. Past the kernel-cache Gram cutoff (~7 240 rows per pair at
cache_size=200) SVC now prefetches kernel rows in batches and implements LIBSVM's
shrinking, on by default (shrinking=True). Preliminary A/B through SVC, off vs on: rbf
1.02-1.37x, poly 0.97x, linear C=1 1.58x, linear C=10 0.72x (MLX CPU) / 0.91x (GPU), where
LIBSVM's own shrinking doubles the iteration count too. Re-measure at 8 000-32 000 rows and
revisit the plan below. -->

The useful region is narrower than the plan assumed: wide data on a non-linear kernel. The
fix that would widen it is not a threshold — it is making an iteration cheap enough to be
worth dispatching, which means keeping the gradient and the working-set selection resident
on the device across iterations instead of crossing the boundary each time. That is 0.2.x
work, and it is the same underlying problem as the logistic solver's below.

## Logistic regression — an honest disappointment

**mlxlearn's L-BFGS is 10–50× slower than scikit-learn's below 1 024 features.** Measured on
the MLX device, `fit` with `max_iter=200`:

| samples × features | mlxlearn (ms) | scikit-learn (ms) | speedup |
|---|---|---|---|
| 1 000 × 16 | 83.5 | 2.8 | 0.03× |
| 5 000 × 128 | 120.7 | 13.0 | 0.11× |
| 20 000 × 16 | 950.8 | 16.4 | 0.02× |
| 60 000 × 128 | 148.0 | 109.6 | 0.74× |
| 20 000 × 1 024 | 245.8 | 309.0 | 1.26× |
| 60 000 × 1 024 | 497.1 | 665.8 | 1.34× |

<!-- perf-phase: misdiagnosed; rewrite. The per-iteration cost came from 12-33 blocking
host synchronizations per iteration in the L-BFGS bookkeeping (inner products, curvature
test, max|g|), not from line-search evaluations, which averaged about 1.1 per iteration.
That bookkeeping is now host NumPy float64 with one synchronization per evaluation; drop the
"on-device Wolfe / batched candidates" plan and re-measure the table and the crossover. -->

The solver is not doing extra work — its iteration counts match scikit-learn's almost
exactly (17 against 17, 20 against 19). It pays **per-iteration overhead**: at 20 000 × 16 an
L-BFGS iteration costs about 5.9 ms while the matmul inside it costs 163 µs. The line search
evaluates the objective several times per step, each evaluation forces a host
synchronization, and the fixed cost of crossing that boundary dominates until the arithmetic
per iteration becomes large.

So the crossover is set high, and both conditions must hold: MLX is used only for problems
that are tall *and* wide. Everything else goes to scikit-learn, which is genuinely faster.
Under `patch_sklearn()` the measured result is 0.79×–1.01×, both within noise, because
dispatch is doing its job.

**This is an optimization target, not a permanent property.** What would move it: evaluating
the Wolfe conditions on device so an accepted step costs one synchronization instead of
several, and batching the line-search candidate evaluations. Neither is in 0.1.0a2.

## Reading these numbers honestly

- Below a crossover, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name.
