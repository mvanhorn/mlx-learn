# Cooperative L-BFGS and guarded Newton refinement

The implemented solver nearly halves ordinary 20,000 × 4,096 fit time:
**30.24 ms**, versus **58.89 ms** for the frozen implementation and
**65.52 ms** for default-tolerance sklearn on the measured Apple M4 Max.
Weighted ordinary fits fall from **75.67 to 37.52 ms**. The guarded Newton
route closes the reproduced ill-conditioned and weighted objective gaps on
two seeds, and the general rank-deficiency panel passes its objective and
coefficient checks. Speedups are not universal; scaled rank-deficient and
rare-class cases retain limitations described below.

After completing validation, the three earlier screenshot datasets were
rerun with their original fixtures and settings. Current complete fits take
**13.86 ms** on 500,000 × 32, **23.96 ms** on 500,000 × 128, and
**13.19 ms** on 60,000 × 1,024, respectively **5.24×**, **3.05×** and
**2.64×** faster than freshly timed stock sklearn. All measured fits passed
independent accuracy checks. The [three-dataset report](tall-logistic-recheck-2026-10-27.md)
contains variability, historical comparisons and the full protocol.

## What changed

The earlier wide L-BFGS direction kernel assigned the complete recurrence to
one 32-thread SIMD group. At 4,096 features each lane retained a long private
vector and repeatedly traversed the history. In an isolated ten-pair screen,
that direction stage took about 2.37 ms, while the fused matrix pass took
about 0.86 ms. Optimizing line-search candidate counts or matrix dot-product
arithmetic did not remove that bottleneck.

Wide directions now use 256 threads, with SIMD reductions combined through a
small shared workspace. The scalar initial metric, L-BFGS ring ordering,
curvature acceptance and line search remain the same. Inputs below 2,048
features retain 32 threads. Both the native command loop and the MLX step path
use the new recurrence. Native dispatch checks the compiled pipeline's thread
and shared-memory limits, as well as the device limits. The NumPy-free device
path uses the same source generator. Older optional extensions that hardcode
a 32-thread launch receive a matching 32-thread kernel; the updated extension
uses the cooperative width.

For prepared binary fits with an unregularized intercept, the solver selects
Newton refinement when all of the following apply:

- At least 4,096 rows and between 2,048 and 4,096 features.
- The initial regularized Hessian diagonal has a maximum more than 100 times
  its positive minimum.
- The regularization scale `lambda = 1 / (C * sum(sample_weight))` is at least
  `1e-7`; unweighted fits use the row count as the weight sum.
- The compatible native Metal CG extension is available.

The selected path starts with at most 32 accepted L-BFGS updates, then uses
inexact Newton steps within the original remaining iteration budget. This
selection is a workload heuristic, not a condition-number estimate or an
accuracy guarantee. Other inputs retain L-BFGS, including the weakly
regularized rank-deficient regime that exposed the previous CG failure.

Newton systems use a one-pass Hessian-vector product in a native C++/Metal CG
loop. Up to eight inner products are submitted per command-buffer batch,
removing Python/MLX scheduling between those inner iterations. Inner solves
stop at a relative residual of 10% early or 1% near the solution, and truncate
on nonpositive or numerically unresolved curvature. They do not attempt to
resolve directions below the float32 operator's useful precision. No diagonal
preconditioner or coefficient projection changes the requested objective.

The outer line search evaluates the actual represented float32 parameter
step through a stable Armijo loss difference. Convergence requires a numerical
primal/dual-gap diagnostic in addition to the compensated gradient check on
original inputs. The diagnostic balances the class residual masses for the
unregularized intercept and includes a numerical reserve. It can detect slow
correlated directions missed by a maximum-gradient threshold alone. It is
**not a formal floating-point objective-error certificate**. Neither it nor
CG truncation proves rank safety for arbitrary inputs; independent objective,
gradient and general rank-deficiency tests remain necessary.

The private Newton, CG and duality modules operate on MLX arrays without
NumPy. The sklearn-compatible public API retains its existing host validation
and conversion dependencies. The distribution still depends on NumPy and
scikit-learn.

## Cooperative direction screen

The paired screen used Apple M4 Max, float32 inputs, `tol=1e-4`, `max_iter=200`,
a requested thread-pool limit of one, two warmups and five timed repetitions per variant. These
are complete fresh fits, in milliseconds, against the pre-change recurrence;
they are not timings of the direction kernel alone.

| Seed | Previous 32-thread recurrence | Cooperative 256-thread recurrence | Previous / new |
| --- | ---: | ---: | ---: |
| 41 | 58.861 | 31.320 | 1.88× |
| 83 | 58.146 | 31.366 | 1.85× |

Both variants used 24 accepted iterations without warnings. Relative objective
differences from the previous result were below `6.4e-11`. A 512-thread trial
reached about 30 ms, but the implementation uses the more conservative
256-thread width. A one-candidate line search failed accuracy/convergence on
one seed and was not adopted. The source-math alternatives provided no
meaningful full-fit gain.

The [raw paired fits](../benchmarks/results/direction-cooperative-ordinary-2026-10-27.json),
[component profile](../benchmarks/results/direction-cooperative-profile-2026-10-27.json)
and [experiment archive](../benchmarks/reference/cooperative-direction-experiment-2026-10-27.zip)
retain timings, numerical scores and source hashes. The isolated recurrence
check covered 48 dimension/history/ring combinations against an independent
float64 calculation; maximum relative vector error was `1.31e-7` and every
direction was descending.

## Final standard benchmark panel

The final panel used Apple M4 Max, MLX 0.32.2 and scikit-learn 1.9.0, with
float32 inputs, `tol=1e-4`, `max_iter=200`, a requested thread-pool limit of one, three warmups and
12 timed repetitions. All cases have 20,000 rows. Timings are complete fresh
fits in milliseconds, with a rotated variant order. Current and frozen are
the unseeded sklearn-compatible GPU path; sketch uses four initial history
pairs. Resident-device input upload is excluded from its fit time and is
recorded separately in the raw results.

The harness requests `threadpoolctl(limits=1)`. On this installation its
inventory reports sklearn's OpenMP runtime, but does not expose the NumPy/SciPy
Apple Accelerate BLAS pool. Thus the request does **not** establish single-threaded
BLAS execution. The raw artifacts' shorthand "one BLAS thread" describes this
request, not a verified Accelerate thread count. This qualification also applies
to the other limited-thread panels in this report.

| Case / features | Frozen | Current | Sketch | sklearn | Resident device |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide 1,024 | 7.304 | 7.160 | 8.385 | 12.473 | 10.788 |
| Wide 4,096 | 58.891 | 30.243 | 33.240 | 65.517 | 34.422 |
| Weighted ill-conditioned 1,024 | 75.789 | 75.421 | 76.881 | 196.063 | 86.058 |
| Shifted 1,024 | 13.583 | 13.422 | 13.378 | 17.424 | 13.692 |
| Paired rank-deficient 1,024 | 9.082 | 8.879 | 9.550 | 14.635 | 11.572 |
| Ill-conditioned 4,096 | 407.395 | 254.859 | 213.682 | 590.800 | 223.610 |
| Weighted 1,024 | 9.233 | 9.151 | 10.583 | 13.873 | 11.581 |
| Weighted 4,096 | 75.671 | 37.523 | 41.424 | 80.060 | 47.703 |
| Weighted unbalanced 1,024 | 24.549 | 24.355 | 25.519 | 24.069 | 27.989 |
| Weighted rare-class 1,024 | 25.955 | 26.273 | 29.021 | 24.858 | 30.232 |
| Weighted effective-unbalanced 1,024 | 12.030 | 11.885 | 12.963 | 18.765 | 14.431 |

All current quality and previous-result regression audits passed, and source
integrity passed. Where the default sklearn fit was not a usable reference,
the benchmark used an independently scored centered float64 reference. The
ordinary wide improvement is **1.95× over the frozen implementation** and
**2.17× over default-tolerance sklearn**; the weighted ordinary improvement
is **2.02× over frozen** and **2.13× over sklearn**. The existing narrow
rare-class workload remains slower than sklearn, and the sketch does not
uniformly improve total fit time.

The frozen and current solvers use their respective captured native binaries.
The refreshed panel includes the old-extension compatibility change, whose
32-thread fallback is required when an older binary is installed. The
[final standard panel](../benchmarks/results/rank-safe-final-2026-10-27.json)
contains every sample, independent score, source hash and warning. The earlier
[pre-compatibility panel](../benchmarks/results/rank-safe-before-compat-panel-2026-10-27.json)
is retained rather than overwritten. Passing the standard panel alone is not
proof of the ill-conditioned objective fix; the tight references below are
the relevant evidence.

## Ill-conditioned objective checks

Dedicated seed-41 and seed-83 checks used one warmup and three timed fits per
variant. The maximum allowed relative objective gap was `1e-5`, scored on the
original inputs using float64 arithmetic outside the fit timer. All current
unseeded, sketch and resident-device fits passed that objective threshold:

| Case | Seed | Current gap | Sketch gap | Device gap | Reference max gradient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ill-conditioned | 41 | 1.75e-8 | 2.54e-8 | 2.54e-8 | 6.06e-8 |
| Ill-conditioned | 83 | 2.80e-8 | 2.79e-8 | 2.79e-8 | 5.93e-8 |
| Weighted ill-conditioned | 41 | 1.74e-7 | 1.73e-9 | 1.73e-9 | 2.05e-8 |
| Weighted ill-conditioned | 83 | 2.04e-9 | 2.05e-9 | 2.05e-9 | 1.09e-13 |

The unseeded seed-41 fits use 35 accepted updates: 32 L-BFGS warm-start steps
and three Newton steps. The previous unseeded relative gaps on these fixtures
were approximately `1.25e-5` and `3.48e-5`, respectively. The new fits also
pass the original-coordinate gradient checks, without convergence warnings.
The [seed-41 tight panel](../benchmarks/results/rank-safe-tight-seed41-2026-10-27.json)
and the unweighted row of the [seed-83 tight panel](../benchmarks/results/rank-safe-tight-seed83-2026-10-27.json)
retain the independently checked L-BFGS references.

Weighted seed 83 required a different independent reference. The tight
float64 L-BFGS attempt reported no warning but left an original-coordinate
gradient of `1.63e-7`, above the required `1e-7`, so its objective was rejected
as a reference. A centered float64 sklearn `newton-cg` fit (`tol=1e-10`,
`max_iter=300`) completed in 14 iterations with gradient `1.09e-13` and no
warning. The new GPU fits pass against that reference. The
[replacement-reference panel](../benchmarks/results/rank-safe-weighted83-newton-reference-2026-10-27.json)
still has an overall failure flag because the **frozen sketch baseline** fails
its convergence audit; the current quality audit passes. Its failure must not
be described as a current solver failure or silently discarded. The earlier
unusable-reference results are retained in the seed-83 artifact.

Default-tolerance sklearn is not equally accurate on every row. For example,
on unweighted seed 83 it finishes in about 197 ms versus the GPU's 216 ms,
but has a relative objective gap of about `6.22e-3` to the tight reference.
That runtime cannot establish a speed comparison at the achieved GPU accuracy.

## Shifted and rank-deficient checks

The extended panel used one warmup, three timed fits, and independently scored
centered float64 sklearn Newton-CG references. Its source integrity, current
quality checks, prediction checks and previous-result audits all passed.
All three cases have 20,000 rows and 4,096 features.

| Case | Frozen | Current | Sketch | sklearn default | Resident device |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted | 84.499 | 49.494 | 51.040 | 60.930 | 39.437 |
| Weighted shifted | 113.673 | 60.204 | 64.170 | 67.037 | 92.973 |
| Paired rank-deficient, C=1e6 | 42.012 | 23.262 | 26.310 | 52.435 | 30.079 |

Current relative objective gaps are below `4.49e-7` in these cases; paired
null-space ratios are zero. The default-tolerance sklearn shifted fits were
not usable accuracy references, so the independent references are essential.
The [extended panel](../benchmarks/results/rank-safe-extended-2026-10-27.json)
contains their scores and diagnostics.

Paired duplicates alone cannot expose the earlier CG failure. A separate
20,000 × 4,096 panel uses an exact **three-column linear dependency**, with
weighted and unweighted cases under both weak and ordinary regularization.
The weakly regularized cases retain L-BFGS; scale-skewed `C=1` cases exercise
the Newton route. Each uses one warmup and three timed fits. All current
variants pass the independently checked `1e-5` objective-gap and `2e-5`
coefficient null-space-ratio gates.

| General rank case | Current ms | Sketch ms | Device ms | sklearn ms | Current objective gap | Current null ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weak regularization | 23.564 | 29.104 | 28.804 | 49.048 | 3.20e-07 | 1.53e-08 |
| Weak, weighted | 25.151 | 32.331 | 32.200 | 53.883 | 4.54e-07 | 3.68e-08 |
| Scaled, C=1 | 629.902 | 635.415 | 628.692 | 689.061 | 3.85e-10 | 8.49e-07 |
| Scaled, weighted, C=1 | 725.664 | 717.932 | 699.791 | 684.685 | 2.50e-10 | 6.34e-07 |

Across all current variants, the maximum coefficient null ratio is `4.43e-6`,
far below the approximately `0.9997` failure from the previous oversolved-CG
prototype. This is evidence on these tested dependencies, not a general
floating-point rank-invariance proof.

The scaled weighted case is **slower than default sklearn** in this panel.
Both scaled sklearn fits exhaust `max_iter=200` with convergence warnings;
their objective gaps are about `4.87e-4` and `1.09e-3`, versus below `4e-10`
for the current GPU variants. The new solver therefore preserves much better
accuracy on these cases, but does not deliver a uniform wall-clock win. The
[general-rank panel](../benchmarks/results/rank-safe-general-rank-2026-10-27.json)
retains all warnings, independent references, null ratios and timings.

## Regularization controls

Additional 20,000 × 4,096 ill-conditioned controls tested `C=0.1` and
`C=10`, with and without sample weights, on seed 41. Each GPU variant had one
warmup and two timed fits; every returned fit passed the objective, original
float64 gradient, convergence and prediction audits. All four independent
centered float64 sklearn Newton-CG references (`tol=1e-10`, `max_iter=300`)
completed without warnings and had original-coordinate gradients below
`1e-10`. Source integrity passed.

| Case | C | Current ms | Sketch ms | Resident device ms |
| --- | ---: | ---: | ---: | ---: |
| Ill-conditioned | 0.1 | 156.425 | 166.918 | 153.043 |
| Weighted ill-conditioned | 0.1 | 148.813 | 180.606 | 163.425 |
| Ill-conditioned | 10 | 811.653 | 838.356 | 832.636 |
| Weighted ill-conditioned | 10 | 1247.672 | 1175.467 | 1197.333 |

Across all variants, the largest relative objective gap was `4.83e-8` and
the largest original-coordinate gradient was `3.06e-6`. The `C=10` controls
remain accurate but are substantially slower than the `C=1` cases: they
need 40–45 total accepted outer steps including the 32-step warm start,
versus 34 steps at `C=0.1`. These two-repeat controls establish additional
accuracy coverage, not stable performance rankings. These four ill-conditioned
controls include no repeated stock sklearn timing; reference durations are
recorded outside candidate fit timers. Resident-device upload is excluded and
recorded separately.

The [raw regularization controls](../benchmarks/results/rank-safe-regularization-controls-2026-10-27.json)
and [exact-source archive](../benchmarks/reference/rank-safe-regularization-controls-2026-10-27.zip)
retain all measured samples, warnings, reference scores and source hashes.

A separate `C=10` panel applies the exact three-column dependency to scaled
20,000 × 4,096 data, with and without weights. With one warmup and two timed
fits, all three GPU paths passed the objective, original-gradient, prediction
and coefficient-null-space audits. Both independent centered float64 sklearn
Newton-CG references completed without warnings, with maximum gradients
`1.47e-11` and `1.82e-11`. Source integrity passed.

| C=10 general-rank case | Current ms | Sketch ms | Device ms | sklearn default ms | Current objective gap | Current null ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scaled | 746.984 | 748.554 | 737.880 | 714.948 | 4.07e-11 | 5.80e-7 |
| Scaled, weighted | 812.281 | 814.721 | 802.198 | 706.984 | 3.41e-11 | 1.17e-6 |

Across the current variants, the largest coefficient null ratio is `2.43e-6`,
and all predictions agree with the tight reference. The fits use 39–40 total
accepted updates. These GPU fits are **slower than default sklearn**; both
stock fits stop at 200 iterations with convergence warnings and objective gaps
of `5.66e-4` and `1.19e-3`. The panel establishes additional accuracy and rank
coverage, not an equal-accuracy speed win. Resident-device upload is excluded.
The [Newton-reference C=10 rank panel](../benchmarks/results/rank-safe-C10-rank-newton-reference-2026-10-27.json)
contains every score, warning and measured sample.

The [earlier C=10 L-BFGS-reference panel](../benchmarks/results/rank-safe-C10-rank-controls-2026-10-27.json)
is retained with its failed overall audit: its unweighted reference reported
no warning but had gradient `1.47e-7`, exceeding the `1e-7` reference threshold.
The weighted reference was usable. Replacing the independent reference does
not erase that failed attempt or change the GPU acceptance thresholds.

## Ordinary wide fits at a common objective gate

The three-seed accuracy-gated panel used exact float64 widenings of the same
float32 values, so every variant received identical input values. It measured
12 repetitions after three warmups, with the same requested thread-pool limit. The GPU retained
`tol=1e-4`, `max_iter=200`; timed sklearn used `tol=1e-7`, `max_iter=2000`.
Every fit passed the shared `1e-5` relative objective threshold against an
independent centered float64 reference. Source integrity passed.

| Seed | Current ms | Sketch ms | Resident device ms | sklearn ms | sklearn / current |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 44.572 | 43.042 | 36.870 | 271.508 | 6.09× |
| 83 | 46.388 | 42.121 | 38.597 | 288.491 | 6.22× |
| 2027 | 45.507 | 42.313 | 37.657 | 287.231 | 6.31× |

Passing one threshold does not mean equal achieved errors. The current
unseeded relative gaps are `1.33e-6`–`2.43e-6`; sklearn's are
`8.24e-11`–`3.63e-10`. Resident-device upload is excluded and reported
separately. These float64/tighter-sklearn timings must not be substituted for
the default-tolerance float32 comparison. The [accuracy-gated panel](../benchmarks/results/rank-safe-accuracy-matched-2026-10-27.json)
contains all scores and measured samples.

## Validation status and retained failures

The final full-suite run passed **1,118 tests**, with 43 warnings, in
139.38 seconds. This includes optional-extension compatibility and additional
rank regressions. Ruff, mechanical compliance checks and `git diff --check`
passed; the original `lbfgs.py` remains unchanged.
Wheel and source distributions built successfully and passed `twine check`.
The built wheel's native `run_cg` entry point imports successfully; the source
distribution includes the native CG header and Objective-C++ source.
The [validation manifest](../benchmarks/results/rank-safe-validation-2026-10-27.json)
records final checks and verifies every final panel's source hashes against
the delivered files. The [reproduction archive](../benchmarks/reference/rank-safe-final-2026-10-27.zip)
retains source, tests, benchmark drivers, measured artifacts and validation logs.

The cooperative-direction checks include independent float64 recurrence
comparisons, history insertion and wraparound, odd dimensions and a general
three-column null-space relation. Additional CPU contracts verify matching
host/device source generation and correct 32-thread launches with older
optional binaries. The earlier 20 targeted tests and six new compatibility
tests are included in the final full-suite count.

The initial [integrated screen](../benchmarks/results/rank-safe-integrated-screen-2026-10-27.json)
failed a legacy benchmark assertion that expected the bounded native warm
start itself to converge. Hybrid instrumentation now records the warm start
and subsequent Newton work explicitly; the original failed artifact is
preserved. The weighted seed-83 unusable-reference attempts and the frozen
sketch failure are also retained, as described above.

The preceding [Hessian-vector investigation](ill-conditioned-hvp-investigation-2026-10-27.md)
remains the record of rejected oversolved CG and mixed-precision attempts.
The present implementation changes neither the requested logistic objective
nor the weak-regularization route to force a favorable benchmark. Numerical
dual diagnostics, inner truncation and the workload-selection heuristic are
empirical safeguards, not universal accuracy guarantees. Reported wins apply
to the measured inputs; shifted, scaled rank-deficient and narrow imbalanced
cases should be read with their actual reference errors and warnings.
