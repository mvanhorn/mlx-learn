# Weighted accuracy and wide GPU investigation

Weighted binary GPU fits now require both the existing original-coordinate
gradient check and a two-step relative accepted-loss progress check. The latter
uses 0.025 times the effective internal tolerance and requires two accepted
steps unless the gradient is exactly zero. The managed and native Metal paths
and the resident-device API apply the check, retain the iteration budget, and
report an unresolved fit rather than promoting a failed line search. A device
intercept polish may run only after the selected candidate has passed its own
progress check, which is checked again against the polished objective.

The check is an empirical safeguard, not a bound on the remaining objective
error. On unbalanced data, individual gradient components can fall below the
requested tolerance long before the objective settles; small recent progress
can still be misleading on other, very flat problems. The generic GPU binary
fallback does not expose accepted-loss progress and uses a tighter internal
gradient threshold for weighted inputs. The narrow CPU solver retains its prior
behavior. Exact unit weights use the existing unweighted path.

The managed native fallback also now preserves a failed line-search status when
the subsequently checked gradient happens to be small. An accepted and certified
intercept refinement can still repair a stalled shifted fit. In a weighted shifted
smoke case, the device sketch and unsketched paths converged without warnings
with original-coordinate maximum gradients of 6.94e-7 and 1.28e-6, respectively.
The earlier ill-conditioned seed-41 device sketch still reaches the default
200-iteration limit and reports nonconvergence.

The [frozen starting source](../benchmarks/reference/weighted-wide-start-2026-10-27.zip)
and [source snapshot](../benchmarks/reference/weighted-wide-start-sources-2026-10-27.zip)
are hashed in the benchmark. The starting native extension is kept locally at
`bench/weighted-wide-start/_metal_loop.abi3.so`; a fresh checkout needs a matching
build for the same comparison. Tight references are centered, float64 sklearn
fits scored on the original data; validation checks their original-coordinate
gradient and records warnings.

## Complete-fit timings and accuracy

On an M4 Max, the following are medians of 12 fresh fits after three warmups,
in rotating order, with sklearn limited to one BLAS thread. Host timings include
validation and input preparation. Device timings use resident MLX arrays and
exclude upload, which is recorded separately. The data have 20,000 rows and
float32 host inputs; sklearn uses its default `tol=1e-4`, `max_iter=200`.

| Dataset | Previous GPU | Current GPU | Current sketch | sklearn | Device sketch |
|---|---:|---:|---:|---:|---:|
| Wide, 1,024 | 7.245 | 7.121 | 8.763 | 12.731 | 10.563 |
| Wide, 4,096 | 42.400 | 42.623 | 51.166 | 65.011 | 56.333 |
| Weighted ill-conditioned, 1,024 | 75.490 | 75.185 | 78.403 | 193.405 | 84.493 |
| Shifted, 1,024 | 13.541 | 13.535 | 13.276 | 17.413 | 12.902 |
| Rank-deficient, 1,024 | 9.150 | 9.063 | 10.197 | 15.074 | 11.518 |
| Ill-conditioned, 4,096 | 401.314 | 402.739 | 372.978 | 561.677 | 391.598 |
| Weighted, 1,024 | 7.510 | 8.868 | 10.383 | 13.084 | 11.654 |
| Weighted, 4,096 | 57.253 | 75.493 | 82.013 | 78.419 | 96.534 |
| Weighted 10% positive, 1,024 | 16.255 | 24.182 | 25.596 | 24.096 | 28.981 |
| Weighted 2% positive, 1,024 | 11.678 | 25.711 | 28.865 | 23.720 | 30.299 |
| Effectively imbalanced weights, 1,024 | 8.668 | 11.824 | 13.496 | 18.650 | 14.264 |

These times are not all accuracy-equivalent. In the 10%-positive and 2%-positive
rows, the default sklearn objectives were 1.09% and 2.40% above the tight
reference. The previous GPU objectives were 2.69% and 3.03% above it, while
the current GPU gaps were 4.01e-6 and 2.98e-6. On the three imbalance fixtures
and seeds 41, 83, and 2027, all current host, sketch and device results met the
`1e-5` objective gate; current unsketched host gaps ranged from 7.93e-8 to
5.63e-6. The accuracy safeguard increases the accepted work and is slower on
these cases; for example, the weighted 4,096-feature fit increased from 23 to
31 accepted iterations. The shifted stock sklearn row fails the original-gradient
check and is not accuracy-equivalent; its audited reference is a tighter centered
fit. Other rows are checked against default sklearn and the frozen prior fit,
not against a general proof of optimality.

The [main panel](../benchmarks/results/weighted-wide-final-2026-10-27.json),
[seed-83 holdouts](../benchmarks/results/weighted-wide-heldout-seed83-2026-10-27.json),
and [seed-2027 holdouts](../benchmarks/results/weighted-wide-heldout-seed2027-2026-10-27.json)
record every timing, independent score, warning and source hash. Tight-reference
experiments also found a pre-existing ordinary unweighted 4,096-feature gap:
the starting unsketched solver was about 8.19e-5 above the tight objective.
The adaptive candidate improved that particular gap but did not meet the
`1e-5` gate, and its sketch worsened; it was not promoted.

To compare fits that both meet the tight objective gate, a separate panel gave
both solvers the same contiguous float64 originals (exact widenings of the
generated float32 values). The timed GPU fit used `tol=1e-4`; stock sklearn used
`tol=1e-7`, selected because `1e-6` missed the gate on one rare-class seed.
Neither timed fit receives pre-centered data. A separate centered float64
sklearn fit at `tol=1e-10` provides the untimed reference. Every measured fit
passed the objective, original-gradient, prediction, and convergence checks.

| Dataset, 20,000 × 1,024 | Seed | GPU (ms) | sklearn (ms) | sklearn / GPU |
|---|---:|---:|---:|---:|
| Weighted 10% positive | 41 | 34.434 | 104.793 | 3.04× |
| Weighted 10% positive | 83 | 34.574 | 115.831 | 3.35× |
| Weighted 10% positive | 2027 | 33.543 | 95.839 | 2.86× |
| Weighted 2% positive | 41 | 32.178 | 93.603 | 2.91× |
| Weighted 2% positive | 83 | 34.820 | 94.191 | 2.71× |
| Weighted 2% positive | 2027 | 33.186 | 91.769 | 2.77× |
| Effectively imbalanced weights | 41 | 18.352 | 52.203 | 2.84× |
| Effectively imbalanced weights | 83 | 20.271 | 49.290 | 2.43× |
| Effectively imbalanced weights | 2027 | 19.512 | 46.931 | 2.41× |

These are 12-fit medians after three warmups, with one BLAS thread, and include
each solver's input preparation. The GPU converts its float64 input during
the timed fit. The [raw accuracy-matched panel](../benchmarks/results/weighted-wide-tight-sklearn-2026-10-27.json)
records every fit and the validated reference. It is specific to these fixtures
and does not establish a speedup for arbitrary float64-only values.

Several wide GPU prototypes were rejected. A conditional backward kernel and
alternative tilings did not produce a reliable speedup across held-out inputs;
one tiling also missed the objective gate. An adaptive line search with two
fused candidate gradients improved some unbalanced weighted fits but slowed an
effectively imbalanced case by about 30% and produced a device convergence
failure on an ill-conditioned case. The [adaptive experiment archive](../benchmarks/reference/adaptive-dual-gradient-2026-10-27.zip)
and [geometry archive](../benchmarks/reference/wide-geometry-experiment-2026-10-27.zip)
retain raw timings, errors, and reference limitations. No faster wide algorithm
was promoted in this investigation.

The complete test suite passed: 839 tests and 43 warnings. It includes weighted
objective regressions, matched host/device specializations, budget limits,
failed line-search handling and progress-preserving device intercept polish.
