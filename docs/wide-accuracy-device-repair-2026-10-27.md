# Wide accuracy and device-sketch convergence

The ordinary wide binary solver now uses a stricter relative two-step
loss-progress check for 2,048 or more features, alongside its existing
original-coordinate gradient check. This change applies to the native Metal,
managed GPU, and resident-device paths. It does not change the logistic
objective, requested iteration budget, or public gradient tolerance. The
progress condition is a heuristic rather than an objective-error bound.

The resident-device solver now requests a 5% tighter native stopping tolerance
for seeded attempts. It still checks the returned parameters with the same
independent compensated score, uses the original tolerance for unsketched
retries, and does not exceed the requested iteration budget. The tighter
native threshold is clamped at the smallest normal float32 value. On the
reproduced weighted, ill-conditioned 20,000 × 1,024 fixture, the seeded fit
converges in 153 iterations without a restart; the earlier fit exhausted 200
iterations after its independent check triggered a restart. The independent
original-coordinate gradient is approximately 1.64e-5, below the requested
1e-4. This is evidence for the fixture, not a guarantee for all inputs.

The wide progress safeguard closes the reproduced ordinary 4,096-feature
gap, but it does not resolve every ill-conditioned case. On two strongly
correlated 4,096-feature fixtures with independently checked centered float64
references, the returned objective remains approximately 1.04e-5–1.28e-5
relative above the reference. The loss-progress condition can be small while
slow directions retain objective error. In a separate sweep, tightening the
gradient threshold fourfold still missed the `1e-5` objective gate; tenfold
tightening exhausted the 200-step budget or hit a failed line search on some
routes. The [ill-conditioned sweep](../benchmarks/results/wide-ill-gradient-sweep-2026-10-27.json)
retains the reference gradients, warnings, and all measured outcomes.

A class-prior initialization prototype fused its starting statistics into the
existing GPU preparation pass, avoiding the synchronization hang encountered
with an earlier extra-matmul prototype. It produced small and inconsistent
speed changes across unbalanced cases, regressed one rare-class case, and its
shifted and ill-conditioned references did not satisfy the strict reference
gate. It was not promoted. The [fused-prior archive](../benchmarks/reference/fused-prior-bias-experiment-2026-10-27.zip)
and [screen results](../benchmarks/results/fused-prior-bias-screen-2026-10-27.json)
record its code, measurements and reference limitations. No substantially
faster general wide algorithm was adopted in this investigation.

The new timing panels use pure cyclic ordering. With 12 measurements this
places every variant in each timing position equally often even after warmup.
Historical panels used an alternating reversal that did not balance all timing
positions; their recorded measurements remain historical and are not repeated
as the new panel's results.

## Complete-fit comparison

On an M4 Max, the table gives median milliseconds for 12 measured fresh fits
after three warmups. Stock sklearn uses one BLAS thread, `tol=1e-4` and the
same float32 host input. GPU host times include validation, input preparation
and output export. Device times start with resident MLX arrays; upload is
measured separately. The frozen package and native binary are hashed before
and after timing.

| Dataset, 20,000 rows | Previous host | Current host | Current sketch | sklearn | Device sketch |
|---|---:|---:|---:|---:|---:|
| Ordinary, 1,024 features | 7.204 | 7.090 | 8.362 | 12.780 | 10.738 |
| Ordinary, 4,096 features | 41.952 | 58.753 | 63.310 | 66.991 | 64.524 |
| Weighted ill-conditioned, 1,024, seed 83 | 74.382 | 74.941 | 76.108 | 192.343 | 86.055 |
| Shifted, 1,024 | 15.111 | 15.307 | 15.524 | 17.604 | 14.867 |
| Rank-deficient, 1,024 | 12.523 | 11.580 | 11.986 | 15.851 | 15.371 |
| Ill-conditioned, 4,096, seed 41 | 401.751 | 401.056 | 372.242 | 556.234 | 400.616 |
| Weighted, 4,096 | 74.884 | 75.187 | 81.621 | 79.355 | 95.751 |

The ordinary 4,096 fix adds iterations and increases runtime against the
previous, less accurate host result. The shifted sklearn timing fails its
original-coordinate gradient check and is not accuracy-equivalent. The
ill-conditioned row passes comparisons against default sklearn, but **fails**
the stricter reference gate described below. The full table, including
weighted and unbalanced cases, is in the [raw panel](../benchmarks/results/wide-repair-final-2026-10-27.json).

On the formerly failing device-sketch fixture, all 12 measured fits converged
in 153 iterations without fallback, with maximum original-coordinate gradient
1.64e-5. The resident-device median was 87.731 ms; upload was measured
separately. The previous device path's 200-iteration warning is recorded in
the [device experiment](../benchmarks/reference/device-sketch-margin-experiment-2026-10-27.zip),
and the new measurements are in the [seed-41 result](../benchmarks/results/wide-repair-device-seed41-2026-10-27.json).

## Accuracy-matched ordinary wide comparison

These timings use identical original float64 inputs, exactly widened from the
float32 fixtures, and include host input preparation. GPU fits use `tol=1e-4`;
sklearn uses the stricter `tol=1e-7` and one BLAS thread. Every fit passes an
objective gap of at most `1e-5`, original-gradient, prediction and convergence
checks against a separately centered, tightly converged float64 reference.
The resident-device column excludes the separately measured upload.

| Seed and C, 20,000 × 4,096 | Host GPU (ms) | GPU sketch (ms) | Device sketch (ms) | sklearn (ms) |
|---|---:|---:|---:|---:|
| 41, C=1 | 69.504 | 69.430 | 64.206 | 270.157 |
| 83, C=1 | 68.795 | 68.832 | 71.159 | 280.239 |
| 2027, C=1 | 68.813 | 69.602 | 68.204 | 284.975 |
| 41, C=0.1 | 61.472 | 61.960 | 60.566 | 271.083 |
| 41, C=10 | 68.670 | 70.390 | 69.069 | 271.723 |

On the three C=1 seeds, current host objective gaps ranged from 1.33e-6 to
2.43e-6, while sklearn's stricter tolerance reached still smaller gaps. This
comparison uses a common accuracy gate, not identical tolerances or identical
objective values. The results do not cover arbitrary float64-only inputs or
claim a general speedup. See the [seed panel](../benchmarks/results/wide-repair-matched-2026-10-27.json)
and [C panel](../benchmarks/results/wide-repair-matched-C-2026-10-27.json).

For the strongly correlated 4,096 fixtures, the independent tighter-reference
checks deliberately fail the `1e-5` objective gate: current host/sketch/device
gaps are 1.25e-5/1.28e-5/1.14e-5 for seed 41 and
1.26e-5/1.04e-5/1.02e-5 for seed 83. Both reference gradients are below 1e-7,
with no reference warnings. Their [seed-41](../benchmarks/results/wide-repair-ill-tight-seed41-2026-10-27.json)
and [seed-83](../benchmarks/results/wide-repair-ill-tight-seed83-2026-10-27.json)
results preserve those failures; their nonzero exit statuses are expected.

An additional small CPU proof of a sampled block preconditioner showed why it
was not promoted: it reduced iterations on correlated data but returned an
original-coordinate gradient above tolerance, and on exact rank deficiency it
introduced a large null-space component. This was not a GPU performance test;
its [source and results](../benchmarks/reference/block-preconditioner-cpu-demonstration-2026-10-27.zip)
are preserved for future work.

The complete test suite passed: 860 tests and 43 warnings.
