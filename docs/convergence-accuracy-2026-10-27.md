# Logistic convergence and wide-objective investigation — 2026-10-27

This document records the convergence changes, tests, and complete-fit
benchmarks from the current development work. The revised two-step rule has
passed the full suite and targeted and full benchmark audits. Both the revised and earlier rules increased
time on the tested 4,096-feature fits relative to the frozen implementation.
The [previous native-loop report](native-loop-optimization-2026-10-26.md)
describes the implementation and measurements before these changes.

## Frozen comparison and scope

The [frozen starting package](../benchmarks/reference/certified-start-2026-10-27.zip)
and [manifest](../benchmarks/reference/certified-start-2026-10-27.manifest.json)
record the complete Python package before this round. The
[separate native source archive](../benchmarks/reference/certified-start-native-2026-10-27.zip)
and [manifest](../benchmarks/reference/certified-start-native-2026-10-27.manifest.json)
record the extension and build sources needed to rebuild its command loop; no
compiled binary is distributed in that archive. Direct comparisons must verify
the baseline source and the loaded native binary, and report any fallback.

The problematic fixtures are dense synthetic binary classification data. The
unweighted wide fixture has 20,000 rows, 4,096 float32 features, seed 41,
`C=1`, a fitted intercept and `tol=1e-4`. The shifted production fixture has
20,000 rows, 1,024 float32 features and a shift of 1,000. These examples expose
specific failures; they do not establish performance or accuracy for every
feature distribution, regularization strength, or tolerance.

## Convergence in the original feature coordinates

When an intercept is fitted, the solver works with centered features
`X_centered = X - mean` and exports `b_original = b_centered - mean · w`.
For the corresponding loss gradients, the chain rule gives

```text
g_original,w = g_centered,w + mean * g_centered,b
g_original,b = g_centered,b
```

The implementation uses this transformation when checking convergence; the
optimizer's centered objective, gradients, search directions and intercept
restoration remain unchanged. Without an intercept, no centering transformation
is applied. The same identity applies separately to each class in multinomial
fits. A direct device objective and transformed-gradient evaluation checks the
returned parameters using the represented float32 data.

An additional, risk-triggered certificate can score the exported model on the
caller's original input in bounded float64 CPU blocks. The trigger considers the
feature shift, the margin to tolerance, and a 128-element sample of float64-to-
float32 conversion error. This is a heuristic, not a global rounding-error bound:
it can miss unsampled input conversion differences and skips fits without an
intercept. When the certificate is selected, its original-coordinate gradient
must meet the requested tolerance before convergence is reported. Ordinary fits
that do not trigger it avoid the full CPU score; risky fits may pay for one or
more extra scoring passes.

For a stalled centered fit with a remaining iteration budget, the development
solver can try a small number of intercept refinements. Binary refinement below
2,048 features uses GPU-computed intercept gradient and curvature, caps each
proposed bias change, and tests a numerically stable *loss difference* before
accepting it. It is disabled for wider binary designs: a bias-only loss change
cannot certify the separate full-gradient progress condition. Multinomial
refinement fixes the unidentifiable common-logit shift and solves a small
principal Hessian block on the CPU for at most 64 classes; its statistics and
loss-difference work remain on the selected device. The direct objective and
original-coordinate gradient must still certify convergence. These refinements
respect the remaining iteration and line-search budgets; if the solver cannot
meet the requested tolerance it must report nonconvergence rather than relabel a
stalled fit as converged.

The [final offset diagnostic](../benchmarks/results/certified-offset-final-2026-10-27.json)
compared the frozen solver with the corrected implementation. On the same
20,000 × 1,024 shifted input, the frozen solver reported a centered gradient of
about **8.87e-5**, yet the independent original-coordinate maximum gradient was
**0.0209156** at `tol=1e-4`. The corrected implementation reported a direct
original-coordinate gradient of **5.04448e-5**, with objective **0.2913232946**
versus **0.2913261490** for the frozen solver. A smaller 1,027 × 520 fixture
likewise improved its independent gradient from **0.0122121** to **4.01544e-5**.
The final run emitted no duplicate-class warning. The earlier
[intermediate diagnostic](../benchmarks/results/certified-offset-2026-10-27.json)
did emit one because both native binaries defined the same private helper class.
On the difficult shifted 20k fixture, sklearn's independent gradient was
**0.356854** and objective **0.693146**: it did not meet the gradient tolerance,
so its output is not a certified optimum. These examples are not a general
convergence guarantee.

## Wide objective-progress heuristic: earlier single-step rule

On the frozen 4,096-feature fixture, the optimizer stopped at iteration 14 with
an independent maximum gradient below `1e-4`, but its objective exceeded
sklearn's by about **1.126e-4 relative**. The sklearn fit had a *larger* maximum
gradient while reaching a lower objective. Thus matching a maximum-coordinate
gradient threshold alone does not ensure equal objective quality when many
coordinates contribute.

The earlier native binary candidate added a **relative accepted-loss progress
heuristic** to the requested gradient check. For designs with at
least 2,048 features, it also requires the magnitude of the last accepted
objective decrease to be no more than `tol * max(abs(objective), 1e-12)`.
The accepted loss difference is already evaluated by the batched Armijo search;
tracking it and the running objective in device metadata avoids a new full
matrix pass. A directly recomputed objective and original-coordinate gradient
remain authoritative at the final check. Zero-budget and initial-stationary
cases cannot have a previous accepted decrease; their direct gradient check keeps
its existing role. If cached and direct checks disagree, the solver reanchors
or reports nonconvergence rather than trusting the cached status.

This was an additional *heuristic*, not a proven objective-optimality or
sklearn-parity bound. In particular, a tiny accepted step after difficult
backtracking can produce a small decrease even when a substantial objective gap
remains. The progress condition is not currently applied to the generic fused
solver or to native designs below 2,048 features. It tightens the stopping rule;
it does not relax the user's gradient tolerance or change the L2 objective. At
2,048 or more features, the float64 certificate does not promote a previously
nonconverged state, preserving the native guard even though the generic solver
does not itself apply that guard. Additional iterations may cost time, and the
final performance impact must be measured rather than assumed away.

The [frozen per-iteration trace](../benchmarks/results/wide-objective-step-progress-2026-10-27.json)
records the objective, maximum and L2 gradient norms, accepted-step relative
change and an L-BFGS directional proxy after each step. On that fixture the last
relative decreases at iterations 14, 15 and 16 were approximately **2.48e-4**,
**1.22e-4**, and **9.33e-5**. The independent objectives were approximately
**0.18785037**, **0.18782743**, and **0.18780990**, compared with sklearn's
**0.18782922** at its default tolerance. By contrast, on the 1,024-feature
unweighted and weighted traces, the last relative decreases at their previous
stopping points were **4.7e-6** and **1.3e-5**. The trace therefore predicts no
extra accepted step in these particular cases, but this is not a general speed
guarantee.

The observed half-slope Newton-decrement proxy at iteration 14 was approximately
**8.87e-5 relative**, already below `1e-4`; using it alone would not have
prevented the observed early stop. The trace and
[per-budget objective record](../benchmarks/results/wide-objective-iteration-sweep-2026-10-27.json)
are exploratory, not source-and-binary-audited final performance results.
Budget-truncated diagnostic fits deliberately record convergence warnings; no
such rows were filtered. The [source-only scripts](../benchmarks/reference/wide-objective-diagnostics-2026-10-27.zip)
and their adjacent manifest preserve the diagnostic method and limitations.

A subsequent counterexample demonstrates the single-step rule's limitation.
On the same 20k × 4,096 fixture with **`C=10`**, it stopped at iteration 15
with an independent gradient below `1e-4`, yet its objective exceeded sklearn's
by about **1.739e-4 relative**, failing the one-sided `1e-5` threshold. An
experimental rule averaging the last two absolute accepted decreases took
17 iterations and produced an objective about **2.490e-5 relative lower** than
sklearn's. Its six-repeat warm median was **43.06 ms**, versus **39.19 ms** for
the single-step rule. The average passed the objective comparison on ten
diagnostic cases, but remains a heuristic and the six-repeat measurements are
not a substitute for final source-and-binary-audited tests. See the
[ten-case evaluation](../benchmarks/results/wide-progress-average-evaluation-2026-10-27.json)
and its [source archive](../benchmarks/reference/wide-progress-average-evaluation-2026-10-27.zip).

The revised native rule uses the average of the last two accepted loss
decreases for this wide-model progress check. Averaging can reduce
the influence of a single small accepted step, but it still cannot bound the
remaining objective gap. It may require more optimizer iterations; the extra
cost observed in the diagnostic must not be interpreted as a no-slowdown result.

An independent 64-case shifted/scaled stress run with weighted and unweighted
binary and three-class problems found no false convergence after selective
original-input certification. Four highly shifted three-class cases explicitly
reported nonconvergence, with independently measured maximum gradients between
approximately `1.116e-4` and `1.435e-4` at `tol=1e-4`. They remain limitations,
not successful fits; other distributions and rounding patterns may behave
differently, particularly when the certificate heuristic does not trigger. The
[stress results](../benchmarks/results/shifted-stress-2026-10-27.json),
[source archive](../benchmarks/reference/shifted-stress-2026-10-27.zip) and
[manifest](../benchmarks/reference/shifted-stress-2026-10-27.manifest.json)
preserve the cases and method.

## Unweighted optimization experiments

Three scratch approaches were measured in balanced variant order with independent
float64 checks after timing. None is retained in production:

- A screened single-pass preparation skips centering for inputs judged close to
  centered. On the tested 20k × 1,024 float32 case, its median was **6.564 ms**
  versus **6.706 ms** for the current path, with 16 of 24 paired wins. The
  screening rule does not establish numerical behavior across arbitrary input
  distributions, and the improvement was not sufficiently dependable.
- A first-candidate-only native search has to restart the established solver if
  that candidate is rejected. On the stricter 20k × 4,096 case, every tested
  candidate required fallback, taking **80.10 ms** versus **55.28 ms** for the
  current path. Retaining the multi-candidate continuation is important.
- Trying an initial step of two, with the existing Armijo search, measured
  **7.20 ms** versus **6.83 ms** on 20k × 1,024 float32 and **47.32 ms** versus
  **43.07 ms** on the default 4,096-feature case. It helped one weighted fixture
  but increased that fixture's relative objective gap versus sklearn from about
  **−4.1e-7** to **7.5e-6**. Both meet the `1e-5` audit threshold, but the
  candidate did not preserve the previous numerical result or improve the
  unweighted workloads consistently.

The raw [single-pass](../benchmarks/results/unweighted-onepass-2026-10-27.json),
[first-candidate](../benchmarks/results/unweighted-one-candidate-2026-10-27.json),
and [initial-step](../benchmarks/results/unweighted-alpha2-2026-10-27.json)
results retain all samples and fit/quality records. The
[source-only preparation and line-search archive](../benchmarks/reference/unweighted-first-step-experiments-2026-10-27.zip)
and [initial-step archive](../benchmarks/reference/unweighted-alpha2-experiment-2026-10-27.zip)
include reproduction notes. The alpha-step run loaded two extension binaries
with duplicate private Objective-C helper names and emitted a runtime warning;
its timing and quality observations should therefore be treated as exploratory.

Further first-step scaling experiments did not produce a safe, dependable gain.
At 20k × 4,096, a scale of 0.5 saved one step for seed 41, but its objective
was **1.81e-5 relative worse** than the two-step baseline; it saved no steps
on held-out seed 83. On 20k × 1,024, scale 1.25 increased the objective by
**0.00138 relative** on an unbalanced fixture and was slower on an
ill-conditioned fixture. These candidates were not retained. See the
[seed-41](../benchmarks/results/wide-first-step-scales-seed41-2026-10-27.json),
[seed-83](../benchmarks/results/wide-first-step-scales-seed83-2026-10-27.json),
[stress](../benchmarks/results/unweighted-first-step-stress-2026-10-27.json),
and [held-out](../benchmarks/results/unweighted-first-step-heldout-2026-10-27.json)
records and [source archive](../benchmarks/reference/wide-first-step-scale-experiment-2026-10-27.zip).

## Revised-rule verification and targeted benchmark

The revised implementation passed the full suite: **700 passed, 43 warnings**,
in **129.76 seconds**. Focused tests also cover averaging and a single accepted
step. The source-and-binary integrity audit and all **180 per-fit independent
float64 quality audits** passed in the [targeted panel](../benchmarks/results/certified-average-targeted-2026-10-27.json).
Each frozen and current fit completed its respective native command loop. The
panel uses 30 fresh fits per variant after three warmups and the same environment,
timing boundaries and one-sided objective threshold described below.

| Input | Rows × features | Frozen ms | Revised ms | sklearn ms | vs frozen | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 1,024 | 9.701 | 10.494 | 19.344 | 0.92× | 1.84× |
| float64 | 20,000 × 4,096 | 43.959 | 49.623 | 120.833 | 0.89× | 2.44× |
| float64 | 20,000 × 1,024 weighted | 11.504 | 12.190 | 21.624 | 0.94× | 1.77× |
| float32 | 20,000 × 1,024 | 6.541 | 6.734 | 12.624 | 0.97× | 1.87× |
| float32 | 20,000 × 4,096 | 37.205 | 42.435 | 64.518 | 0.88× | 1.52× |
| float32 | 20,000 × 1,024 weighted | 7.288 | 8.276 | 13.891 | 0.88× | 1.68× |

For 4,096 features the revised rule is **14.1% slower** than the frozen
baseline in float32 and **12.9% slower** in float64. On the float32 target
20k × 1,024 it is **3.0% slower** than the frozen baseline. The weighted
float32 median is **13.6% slower** than frozen; the timings show substantial
spread and should not be assumed to generalize. The no-speed-loss goal is not
met. All six measured shapes still beat sklearn by the median ratios above.

On the float32 4,096-feature case, the independently evaluated objective is
**0.1878071636** versus sklearn's **0.1878292179**, a relative gap of
**−1.1742e-4**; the frozen value is **+1.1264e-4** relative to sklearn. The
audit accepts lower objective values and limits only relative increases to
`1e-5`; it does not require absolute agreement. The revised objective therefore
passes that one-sided criterion but is not within `1e-5` in absolute difference.
The broader panel and offset diagnostic are described below.

The [full revised panel](../benchmarks/results/certified-average-full-2026-10-27.json)
contains 14 cases with 30 fresh fits and three warmups per variant. Source and
binary integrity passed for **86 files** and every row passed the one-sided
objective, gradient and prediction checks. Each of the 420 revised fit records
was checked for convergence, route and warnings; independent float64 quality
was computed for the last fit in each row. The frozen and revised wide fits used
their respective native loops, and the tall-control fits used the existing MLX
path. Timings are medians on the same shared machine.

| Input | Rows × features | Frozen ms | Revised ms | sklearn ms | vs frozen | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 6.064 | 6.452 | 11.799 | 0.94× | 1.83× |
| float64 | 20,000 × 1,024 | 10.662 | 11.224 | 20.248 | 0.95× | 1.80× |
| float64 | 60,000 × 1,024 | 19.440 | 20.156 | 56.215 | 0.96× | 2.79× |
| float64 | 40,000 × 2,048 | 26.334 | 27.014 | 68.754 | 0.97× | 2.55× |
| float64 | 20,000 × 4,096 | 44.815 | 50.256 | 121.080 | 0.89× | 2.41× |
| float64 | 20,000 × 1,024 weighted | 12.687 | 12.420 | 21.916 | 1.02× | 1.76× |
| float64 | 500,000 × 32 | 17.804 | 18.802 | 79.614 | 0.95× | 4.23× |
| float32 | 20,000 × 512 | 3.601 | 3.714 | 7.826 | 0.97× | 2.11× |
| float32 | 20,000 × 1,024 | 7.677 | 7.962 | 13.004 | 0.96× | 1.63× |
| float32 | 60,000 × 1,024 | 15.258 | 14.966 | 35.448 | 1.02× | 2.37× |
| float32 | 40,000 × 2,048 | 20.508 | 20.364 | 41.435 | 1.01× | 2.03× |
| float32 | 20,000 × 4,096 | 37.823 | 43.296 | 64.931 | 0.87× | 1.50× |
| float32 | 20,000 × 1,024 weighted | 8.162 | 8.052 | 14.147 | 1.01× | 1.76× |
| float32 | 500,000 × 32 | 15.384 | 15.658 | 76.379 | 0.98× | 4.88× |

The full panel's 4,096-feature median slowed by **14.5%** for float32 and
**12.1%** for float64 relative to the frozen solver, consistent with the
targeted panel's measured cost. In the float32 20k × 1,024 full-panel case the
median is **1.63× sklearn**, versus **1.87×** in the targeted panel; this
variation illustrates why the two panels and their raw samples are retained.

## Pre-update verification and benchmarks

Before the revised progress rule, the full suite passed: **698 tests**, with
**43 warnings**, in 127.91 seconds. These results must not be interpreted as
validation of a later source version.
The warnings include expected fallback, convergence and optional-check warnings;
the raw test log is kept with the local benchmark logs. The targeted and full
benchmark source/binary integrity audits both passed, with **86 files checked**
per panel. All 1,800 measured fits across both panels emitted no warnings.

The [targeted panel](../benchmarks/results/certified-targeted-2026-10-27.json)
uses 30 fresh fits after three warmups per variant for 20k × 1,024, its weighted
variant, and 20k × 4,096, with both float32 and float64 inputs. Each of the
**180 current fits** was independently scored in float64 after timing. Every
targeted fit passed the original-coordinate gradient, objective and prediction
checks, and every frozen and current fit actually completed its native command
loop. The baseline was loaded from the frozen package and its separately
hash-verified baseline extension; it did not silently fall back.

| Input | Rows × features | Frozen ms | Current ms | sklearn ms | vs frozen | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 1,024 | 9.785 | 9.600 | 19.274 | 1.02× | 2.01× |
| float64 | 20,000 × 4,096 | 43.960 | 47.783 | 120.780 | 0.92× | 2.53× |
| float64 | 20,000 × 1,024 weighted | 10.843 | 11.572 | 21.633 | 0.94× | 1.87× |
| float32 | 20,000 × 1,024 | 6.866 | 7.051 | 12.654 | 0.97× | 1.79× |
| float32 | 20,000 × 4,096 | 37.416 | 41.441 | 64.418 | 0.90× | 1.55× |
| float32 | 20,000 × 1,024 weighted | 7.881 | 7.959 | 14.138 | 0.99× | 1.78× |

The targeted float32 20k × 1,024 median was **7.051 ms**, compared with
**6.866 ms** for the frozen starting solver and **12.654 ms** for sklearn.
That is **1.79× sklearn** but about **2.7% slower** than the starting solver.
The weighted float32 case was also about 1% slower. These are workload-specific
warm medians on a shared machine, not a general claim of unchanged speed.

At 20k × 4,096, the additional convergence work cost about **10.8%** in
float32 (37.416 → 41.441 ms) and **8.7%** in float64 (43.960 → 47.783 ms)
in the targeted panel. The repeated full panel below measured about **10.4%**
and **9.3%** slowdowns respectively. Thus the goal of fixing the gap with no
speed loss was **not met** on these fixtures.

The stronger stopping rule removed the prior *positive* objective excess in
this fixture. At default `tol=1e-4`, float32 independent objective values were
**0.1878503745** for the frozen solver, **0.1878099011** for the current solver,
and **0.1878292179** for sklearn. The current value is approximately **1.0284e-4
relative lower** than sklearn, while the frozen value was **1.1264e-4 relative
higher**. The benchmark policy is **one-sided**: it rejects an objective increase
above `1e-5`, rather than requiring the absolute difference to be below `1e-5`.
The present pass therefore does not establish symmetric objective agreement or
a universal objective-optimality bound. The current solution used 16 accepted
steps versus 14 for the frozen solver and 15 for sklearn on this fixture.

The [full panel](../benchmarks/results/certified-full-2026-10-27.json) uses the
same 30 repetitions and three warmups over every listed case. All 14 rows pass
the unchanged one-sided objective, gradient and prediction audit. Its 420 current
fit records were checked for convergence, route and warnings; independent
float64 quality was computed for the last fit in each row. All 360 wide fits
in both MLX variants used their respective native loops. The 60 tall-control
fits in each MLX variant used the existing MLX Metal path.

| Input | Rows × features | Frozen ms | Current ms | sklearn ms | vs frozen | vs sklearn |
|---|---|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 5.164 | 5.097 | 11.571 | 1.01× | 2.27× |
| float64 | 20,000 × 1,024 | 12.034 | 10.767 | 19.348 | 1.12× | 1.80× |
| float64 | 60,000 × 1,024 | 19.579 | 19.786 | 56.119 | 0.99× | 2.84× |
| float64 | 40,000 × 2,048 | 26.124 | 27.132 | 68.363 | 0.96× | 2.52× |
| float64 | 20,000 × 4,096 | 44.251 | 48.369 | 119.734 | 0.91× | 2.48× |
| float64 | 20,000 × 1,024 weighted | 11.113 | 11.245 | 22.079 | 0.99× | 1.96× |
| float64 | 500,000 × 32 | 16.292 | 16.034 | 78.600 | 1.02× | 4.90× |
| float32 | 20,000 × 512 | 3.553 | 3.739 | 7.681 | 0.95× | 2.05× |
| float32 | 20,000 × 1,024 | 7.927 | 7.728 | 12.964 | 1.03× | 1.68× |
| float32 | 60,000 × 1,024 | 15.228 | 14.598 | 35.016 | 1.04× | 2.40× |
| float32 | 40,000 × 2,048 | 19.796 | 21.183 | 40.963 | 0.93× | 1.93× |
| float32 | 20,000 × 4,096 | 37.746 | 41.662 | 65.207 | 0.91× | 1.57× |
| float32 | 20,000 × 1,024 weighted | 7.331 | 7.429 | 13.865 | 0.99× | 1.87× |
| float32 | 500,000 × 32 | 14.933 | 15.060 | 75.184 | 0.99× | 4.99× |

At the targeted sizes, 25th/50th/75th percentile ranges and paired wins show
the variation behind the median ratios:

| Input / shape | Frozen Q1 / median / Q3 ms | Current Q1 / median / Q3 ms | Current paired wins |
|---|---:|---:|---:|
| float64 / 20,000 × 1,024 | 8.491 / 9.785 / 11.914 | 8.980 / 9.600 / 14.093 | 12/30 |
| float64 / 20,000 × 4,096 | 43.181 / 43.960 / 46.456 | 46.976 / 47.783 / 49.542 | 3/30 |
| float32 / 20,000 × 1,024 | 6.406 / 6.866 / 7.581 | 6.567 / 7.051 / 8.078 | 10/30 |
| float32 / 20,000 × 4,096 | 36.938 / 37.416 / 37.990 | 40.839 / 41.441 / 42.084 | 0/30 |

The environment was Apple M4 Max, macOS 26.6.2, Python 3.13.3, MLX 0.32.2,
NumPy 2.5.2, SciPy 1.18.1 and scikit-learn 1.9.0. Synthetic binary data uses
seed 41 and the same settings (`C=1`, `max_iter=200`, intercept and `tol=1e-4`)
for every variant. All six variant orderings appeared five times. Threadpoolctl
requested one thread for recognized pools; it does not govern Apple Accelerate
or explicit conversion workers. Data generation and independent scoring were
outside the timer; estimator construction, preparation, device synchronization,
optimization and direct certification were inside. The machine was not exclusive,
and first-use shader compilation is not represented by these warmed medians.

To rerun on the recorded machine with the separately preserved, hash-matching
baseline extension, use:

```bash
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/certified-start-2026-10-27.zip \
  --reference-extension bench/certified-start/_metal_loop.abi3.so \
  --profile full --case wide_20k_1024 --case wide_weighted --case wide_20k_4096 \
  --dtype both --repeats 30 --warmup 3 --threads 1 \
  --require-native-loop --require-reference-native-loop --audit-every-fit \
  --output bench/certified-targeted.json --report bench/certified-targeted.md
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/certified-start-2026-10-27.zip \
  --reference-extension bench/certified-start/_metal_loop.abi3.so \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1 \
  --output bench/certified-full.json --report bench/certified-full.md
python3 -m pytest -q
```

The baseline binary under `bench/` is a local build artifact and is not tracked
in the source-only archives. The runner deliberately rejects a baseline binary
whose SHA-256 differs from the recorded manifest. Rebuilding the provided native
source on another toolchain or in a different build location may produce a
different digest; in that case capture a new verified baseline and manifest
rather than substituting it into these records. The hash and source archives
make the original comparison auditable; they do not guarantee a bit-identical
binary can be recreated on another machine.
