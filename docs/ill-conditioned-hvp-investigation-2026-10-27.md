# Ill-conditioned wide solver investigation

The production logistic optimizer is unchanged in this investigation. In
particular, the existing stopping checks still leave a known objective gap on
some strongly correlated 4,096-feature inputs. The work added the private,
NumPy-free `mlxlearn_device._hessian_vector` building block and additional
benchmark fixtures; the building block is not wired into a solver.

The private component computes an exact-form logistic Hessian-vector product
using a one-pass Metal kernel or a same-device, two-pass MLX fallback. A
prepared wrapper retains the selected device and a row-contiguous design so
the design is not implicitly copied at every product. Fused execution requires
explicit, verified limits for the exact compiled kernel pipeline and device;
unknown, invalid or insufficient capabilities use the fallback. The component
caps threadgroup memory and partial workspace, handles zero-weight rows before
their dot products, propagates invalid curvature, and uses a stable curvature
expression for saturated logits. Its output is not an objective certificate.
There is currently no production capability query or optimizer caller, so the
default path remains the two-pass fallback.

An isolated microbenchmark of the earlier one-pass prototype on 20,000 × 4,096
inputs measured approximately 0.87 ms per product versus 2.67 ms for the
two-pass product, with independent float64 comparisons. A subsequent earlier
version of the private component measured approximately 1.23 versus 3.05 ms.
These are HVP microbenchmarks, not end-to-end solver measurements, and they do
not establish the performance of the hardened final wrapper on arbitrary
devices. The [prototype archive](../benchmarks/reference/fused-hvp-cg-prototype-2026-10-27.zip)
and [component validation](../benchmarks/reference/private-hvp-component-2026-10-27.zip)
retain source, hashes, limits and the numerical tests.

An exact-form Newton-CG correction using the fused prototype reduced the
reproduced ill-conditioned objectives to near a tight centered float64
reference on several cases. Its online Armijo and original-gradient checks,
plus fresh loss-progress checks after reanchoring, added approximately
120 ms to a converged 380–420 ms fit in an initial warm panel. That experiment
is not a production implementation: it uses a managed host path and does not
cover all device, budget, intercept, or finite-precision conditions. Early
correction appeared faster on one seed but failed and exhausted the budget on
another. All measured costs and warnings are retained in the
[continuation archive](../benchmarks/reference/newton-cg-continuation-2026-10-27.zip).

More decisively, a separate exact three-column dependency at high `C` exposed
catastrophic finite-precision null-space drift. A full-float32 CG correction
changed the coefficient null-space ratio from approximately `6.61e-8` to
`0.999664` while its objective, gradient and progress checks passed. A
float16-storage iterative-refinement variant also introduced a large null-space
component and was slower in its first ill-conditioned screen. Neither is
safe to enable. The [rank and mixed-precision archive](../benchmarks/reference/newton-cg-rank-and-mixed-precision-2026-10-27.zip)
preserves both failures; a single-product rank test does not establish safety
for repeated CG recurrences.

Other prototypes did not yield a reliable wide speedup. A 512-thread Metal
geometry was slower than the current geometry on two ordinary 4,096-feature
seeds. An 8,000-row warm start raised the full-data starting objective and
increased warm 20,000 × 4,096 fit time from about 60 ms to 103–114 ms. The
[geometry](../benchmarks/reference/large-threadgroup-experiment-2026-10-27.zip)
and [sample-growth](../benchmarks/reference/sample-growth-warm-start-2026-10-27.zip)
archives contain their methods, scores, timings and limitations.

## Validation and default-reference benchmark

The complete test suite passed: **941 passed, 43 warnings**. The private HVP
component was tested separately on CPU and GPU, including odd dimensions,
weights, saturated logits, invalid capability reports and an exact
three-column dependency. Those product tests do not address the CG drift above.

The following complete-fit panel ran on an Apple M4 Max with MLX 0.32.2 and
scikit-learn 1.9.0. Inputs were float32, `tol=1e-4`, `max_iter=200`, with one
BLAS thread, three warmups and 12 timed repetitions. Values are median
milliseconds; current and frozen are the unseeded sklearn-compatible GPU path,
sketch is the current four-pair seeded path, and device is the resident-device
path (its input upload is reported separately in the raw data).

| Case | Frozen | Current | Sketch | sklearn | Device |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide 20,000 × 1,024 | 7.623 | 7.462 | 8.701 | 13.100 | 10.998 |
| Wide 20,000 × 4,096 | 58.641 | 58.879 | 63.072 | 64.403 | 64.982 |
| Weighted ill-conditioned 20,000 × 1,024 | 75.188 | 75.650 | 76.641 | 192.712 | 88.379 |
| Shifted 20,000 × 1,024 | 13.003 | 13.177 | 12.780 | 17.192 | 12.988 |
| Rank-deficient 20,000 × 1,024 | 8.733 | 8.801 | 9.460 | 14.414 | 11.556 |
| Ill-conditioned 20,000 × 4,096 | 397.570 | 397.442 | 368.700 | 554.729 | 400.289 |
| Weighted 20,000 × 1,024 | 8.714 | 8.656 | 10.030 | 12.913 | 11.562 |
| Weighted 20,000 × 4,096 | 74.671 | 74.461 | 80.775 | 77.717 | 95.917 |
| Weighted unbalanced 20,000 × 1,024 | 24.298 | 23.551 | 25.487 | 24.126 | 29.268 |
| Weighted rare 20,000 × 1,024 | 24.912 | 25.013 | 27.483 | 22.801 | 30.764 |
| Weighted effective unbalanced 20,000 × 1,024 | 11.406 | 11.247 | 12.467 | 18.355 | 13.994 |

This default panel's quality checks passed against the default-tolerance
scikit-learn fit. That is not a tight-reference test and does not establish
that the ill-conditioned objective gap is fixed. No optimizer code changed in
this investigation; differences between the frozen and current times are
measurement variation, not an improvement attributable to the HVP component.
The frozen and current paths used the same native extension file, whose hash
matches the frozen extension, avoiding duplicate Objective-C runtime classes.
The [raw panel](../benchmarks/results/ill-solver-final-2026-10-27.json) records
all samples, input details, source hashes and per-fit quality checks.

## Tight-reference objective checks

Two separately reproduced ill-conditioned 20,000 × 4,096 cases were also
checked against independently fitted centered float64 scikit-learn references.
The independent scores evaluate the returned parameters in the original
coordinates. Both references completed without warnings; their original-coordinate
maximum absolute gradients were below `1e-7`. The tested quality gate requires
the relative objective gap to be at most `1e-5`.

| Seed | Reference objective | Reference max gradient | Current gap | Sketch gap | Device gap | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 41 | 0.26137497946 | 6.06e-8 | 1.24736e-5 | 1.28065e-5 | 1.14164e-5 | All fail |
| 83 | 0.26318263678 | 5.93e-8 | 1.26270e-5 | 1.03665e-5 | 1.02480e-5 | All fail |

The same gaps appeared in all three timed fits of each variant. These results
explicitly demonstrate why passing against the ordinary sklearn reference in
the preceding default panel does not resolve the accuracy question. The
separate [seed 41](../benchmarks/results/ill-solver-tight-seed41-2026-10-27.json)
and [seed 83](../benchmarks/results/ill-solver-tight-seed83-2026-10-27.json)
artifacts include reference scores, per-fit checks, timings, warnings and
source hashes. Each used one warmup and three timed fits.

## Extended 4,096-feature panel

Four additional 20,000-row cases used the same GPU and sklearn settings,
one warmup and three timed fits, and an independently fitted centered float64
reference. Source integrity passed; the overall quality audit failed because
of the weighted ill-conditioned and shifted cases described below. Median
times in milliseconds and outcomes are:

| Case | Current | Sketch | sklearn | Device | Tight-reference audit |
| --- | ---: | ---: | ---: | ---: | --- |
| Weighted ill-conditioned | 418.162 | 400.927 | 635.602 | 451.808 | Fail: objective gaps |
| Shifted | 87.345 | 89.927 | 62.664 | 72.813 | Fail: reference unusable; device predictions flagged |
| Weighted shifted | 120.495 | 122.875 | 68.025 | 119.522 | Pass |
| Paired rank-deficient | 43.024 | 47.988 | 52.209 | 53.113 | Pass |

For the weighted ill-conditioned case, the reference had objective
`0.24989498816`, maximum original-coordinate gradient `2.05e-8`, and no
warnings. Current, sketch and device relative objective gaps were respectively
`3.47971e-5`, `3.62493e-5` and `3.49471e-5`, failing the `1e-5` gate on all
three fits. The default-tolerance sklearn fit itself had a `4.80720e-5` gap
to the independent reference.

The shifted case's independently scored reference had gradient `3.29e-7`,
above the required `1e-7` reference threshold, despite reporting no warnings;
its objective therefore cannot be used to establish parity. The audit also
flagged actual device predictions against the audited reference for all three
fits. For the weighted shifted case, the independently scored reference had
gradient `5.37e-8` and no warnings; all GPU variants passed the quality audit,
although the ordinary sklearn fit was not itself a usable reference. The
paired rank-deficient case passed with an independent reference gradient of
`4.54e-9` and no warnings. Its paired-column fixture does not establish
general rank safety: the separate three-column CG failure described above is
a counterexample. Full per-fit scores, reference diagnostics and timings are
in the [extended panel](../benchmarks/results/ill-solver-extended-2026-10-27.json).

## Ordinary wide fits at a shared objective gate

An additional three-seed panel checks ordinary 20,000 × 4,096 inputs at `C=1`
against a shared `1e-5` relative objective-gap gate using an independently
fitted centered float64 reference. Each input is an exact float64 widening of
float32 values, so all paths receive the same values. The GPU paths use
`tol=1e-4`; timed sklearn uses `tol=1e-7`, `max_iter=2000`, and one BLAS
thread. The panel uses three warmups and 12 timed repetitions; median fit
times are in milliseconds. Resident-device input upload is excluded from the
device fit time and is measured separately in the raw data.

| Seed | Current | Sketch | Device | sklearn | sklearn / current |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 70.935 | 69.985 | 64.588 | 271.619 | 3.83× |
| 83 | 73.697 | 69.522 | 71.042 | 276.560 | 3.75× |
| 2027 | 69.597 | 70.303 | 68.592 | 285.344 | 4.10× |

All four variants passed the shared gate on all fits and source integrity
passed. The threshold is a common upper bound, not equality of achieved
accuracy: current relative objective gaps were about `1.33e-6`, `2.43e-6` and
`1.61e-6`, whereas sklearn's were about `1.68e-10`, `3.63e-10` and
`8.24e-11`. The tight centered references were warning-free with maximum
gradients below `2e-9`. These timings use a tighter sklearn tolerance and
float64 inputs, and should not be compared directly with the default-tolerance
float32 panel above. They also do not demonstrate an improvement from this
investigation, whose optimizer is unchanged. The [accuracy-matched panel](../benchmarks/results/ill-solver-matched-2026-10-27.json)
contains every sample, error and independent reference score.
