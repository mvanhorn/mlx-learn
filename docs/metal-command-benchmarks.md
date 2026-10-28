# Experimental native Metal command loop — 2026-10-26

A native Objective-C++/Metal command-loop prototype bypasses MLX's per-operation
submission for batches of logistic L-BFGS steps. At the time of this experiment
it was not enabled in the library: the gains were mixed and exceptional paths
needed buffer-lifetime safety work. A later implementation addresses ownership
and scheduling; see the [current results](native-loop-optimization-2026-10-26.md).
The results below measure the earlier prototype, not the current implementation.

The [final prototype source archive](../benchmarks/reference/metal-command-final-prototype-2026-10-26.zip)
contains the source, build configuration, benchmark runner, frozen comparison,
and tests used for this panel. Its [manifest](../benchmarks/reference/metal-command-final-prototype-2026-10-26.manifest.json)
records source and archive hashes and the SHA-256 of the compiled extension that
was measured; compiled binaries are not included. The earlier
[exploratory prototype archive](../benchmarks/reference/metal-command-prototype-2026-10-26.zip)
and its [manifest](../benchmarks/reference/metal-command-prototype-2026-10-26.manifest.json)
contain preceding probes and are documented in the [exploratory notes](metal-command-prototype-2026-10-26.md).

The prototype consumes evaluated MLX Metal buffers through DLPack and encodes up
to eight optimizer steps in each command buffer, reusing the existing Metal
kernel expressions. It synchronizes before consuming data on the new queue and
checks the returned objective and gradient directly. If its bounded fast path
cannot certify convergence, it restarts the existing solver. Exceptional Metal
submission and completion paths, including keeping borrowed buffers alive until
GPU work is definitely finished, require further review and testing before
production use.

## Retained library: final sklearn benchmark

After restoring the existing solver, the full panel was rerun with 30 repetitions
and three warmups. The frozen baseline and current package source were
byte-identical. The following medians are from the current package and stock
sklearn; † marks the two pre-existing default-tolerance quality failures.

| Input | Rows × features | Library ms | sklearn ms | vs sklearn |
|---|---|---:|---:|---:|
| float64 | 20,000 × 512 | 5.382 | 11.930 | 2.22× |
| float64 | 20,000 × 1,024 | 12.476 | 19.812 | 1.59× |
| float64 | 60,000 × 1,024 | 18.670 | 56.166 | 3.01× |
| float64 | 40,000 × 2,048 | 24.408 | 69.155 | 2.83× |
| float64 | 20,000 × 4,096 † | 43.477 | 119.843 | 2.76× |
| float64 | 20,000 × 1,024 weighted | 11.262 | 21.585 | 1.92× |
| float64 | 500,000 × 32 | 17.049 | 79.714 | 4.68× |
| float32 | 20,000 × 512 | 3.977 | 7.965 | 2.00× |
| float32 | 20,000 × 1,024 | 7.245 | 12.997 | 1.79× |
| float32 | 60,000 × 1,024 | 14.036 | 35.878 | 2.56× |
| float32 | 40,000 × 2,048 | 18.701 | 41.172 | 2.20× |
| float32 | 20,000 × 4,096 † | 37.079 | 64.739 | 1.75× |
| float32 | 20,000 × 1,024 weighted | 8.161 | 14.069 | 1.72× |
| float32 | 500,000 × 32 | 14.641 | 75.604 | 5.16× |

The [full current-library results](../benchmarks/results/metal-command-current-2026-10-26-full.json)
retain all samples and both failing rows; the source-integrity audit passed.
At a shared `tol=1e-5`, both 4,096-feature cases passed the
[strict current-library audit](../benchmarks/results/metal-command-current-2026-10-26-strict-4096.json).
The full suite on the retained library passed **589 tests, with 41 warnings**.

As the source is unchanged, the difference between this run and earlier timing
ratios is measurement variation, not a new library speedup.

## Prototype complete-fit results

On the targeted 20,000 × 1,024 float32 case, the prototype measured **7.203 ms**
versus **7.567 ms** for the frozen existing implementation, a **1.05×** gain in
the same run. Stock scikit-learn measured **13.118 ms**, or **1.82×** the
prototype time. This is a small gain relative to the spread. The float64 target
case reached **1.02×** the existing implementation and **1.29×** sklearn in this
run. These values should not be compared across runs without accounting for their
variance.

Medians in milliseconds are below. The previous implementation was measured in
the same interleaved run; † marks default-tolerance objective parity failures.
The final column counts actual completed native loops among 30 prototype fits.

| Input | Rows × features | Previous ms | Prototype ms | sklearn ms | vs previous | vs sklearn | Loop completions |
|---|---|---:|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 5.984 | 5.467 | 11.921 | 1.09× | 2.18× | 30/30 |
| float64 | 20,000 × 1,024 | 16.168 | 15.788 | 20.388 | 1.02× | 1.29× | 30/30 |
| float64 | 60,000 × 1,024 | 21.640 | 21.318 | 57.537 | 1.02× | 2.70× | 30/30 |
| float64 | 40,000 × 2,048 | 23.494 | 24.466 | 68.753 | 0.96× | 2.81× | 30/30 |
| float64 | 20,000 × 4,096 † | 45.230 | 43.752 | 121.092 | 1.03× | 2.77× | 30/30 |
| float64 | 20,000 × 1,024 weighted | 12.416 | 13.389 | 22.229 | 0.93× | 1.66× | 30/30 |
| float64 | 500,000 × 32 | 19.203 | 19.109 | 80.604 | 1.00× | 4.22× | 0/30 |
| float32 | 20,000 × 512 | 7.621 | 5.000 | 8.132 | 1.52× | 1.63× | 30/30 |
| float32 | 20,000 × 1,024 | 7.567 | 7.203 | 13.118 | 1.05× | 1.82× | 30/30 |
| float32 | 60,000 × 1,024 | 14.538 | 14.575 | 36.026 | 1.00× | 2.47× | 30/30 |
| float32 | 40,000 × 2,048 | 20.558 | 20.288 | 42.154 | 1.01× | 2.08× | 30/30 |
| float32 | 20,000 × 4,096 † | 37.570 | 36.830 | 65.712 | 1.02× | 1.78× | 30/30 |
| float32 | 20,000 × 1,024 weighted | 10.325 | 10.849 | 14.567 | 0.95× | 1.34× | 30/30 |
| float32 | 500,000 × 32 | 15.279 | 15.939 | 76.683 | 0.96× | 4.81× | 0/30 |

The tall control is outside the prototype's wide-input fast path and uses the
existing MLX Metal solver. All 360 wide fits in this panel completed the native
loop; the 60 tall fits used the existing solver. The weighted and some larger
cases regress against the previous implementation. The float32 20k × 512 median
shows a larger isolated gain, but the mixed results and variability do not
establish a general large speedup.

For 20,000 × 1,024, the individual timing distributions were:

| Input / implementation | Q1 ms | Median ms | Q3 ms | Mean ms |
|---|---:|---:|---:|---:|
| float64 / previous | 15.499 | 16.168 | 16.431 | 15.076 |
| float64 / prototype | 10.837 | 15.788 | 16.379 | 14.149 |
| float64 / sklearn | 19.970 | 20.388 | 20.806 | 20.602 |
| float32 / previous | 6.808 | 7.567 | 8.884 | 8.730 |
| float32 / prototype | 6.634 | 7.203 | 8.178 | 8.019 |
| float32 / sklearn | 12.987 | 13.118 | 13.359 | 13.165 |

The prototype won 21 of 30 paired float64 repetitions and 23 of 30 float32
repetitions. Quartiles are NumPy's 25th/75th percentiles; the headline ratios
use medians. All samples, orderings and fit records are in the
[default-tolerance raw results](../benchmarks/results/metal-command-2026-10-26-full.json).

## Combined-reduction experiment

The prototype's fused loss and gradient reduction was also tried in the existing
MLX solver. All eight quality audits passed, but the 18-repeat full-fit results
were mixed: the 20k × 1,024 float32 median changed from 6.727 to 6.492 ms,
while the candidate won only 4 of 18 paired repetitions and its mean regressed.
The float64 target median changed from 9.270 to 9.903 ms, and the weighted
float64 median from 10.774 to 14.338 ms. The candidate was not retained.

The [raw results](../benchmarks/results/combined-reduction-experiment-2026-10-26.json)
and [source-only experiment archive](../benchmarks/reference/combined-reduction-experiment-2026-10-26.zip)
retain the evidence and reproduction instructions. The adjacent manifest notes
which helper files were not included in the original measurement-time hash set.

## Accuracy and test status

All 420 optimized fits in the default panel reported gradient convergence without
warnings. The independent float64 gradient check on the last fit in each row
met `tol=1e-4`; it is outside the timed region. The independent objective of
the last fit differs from the frozen implementation by less than 1.4e-10
relatively in each row. This compares numerical quality on these datasets; it
is not a guarantee for all inputs.

The **default panel fails its quality audit** for both 4,096-feature rows:
the prototype's relative objective increase versus sklearn is approximately
1.1264e-4, above the 1e-5 policy threshold. The frozen previous implementation
has essentially the same pre-existing gap. The other 12 rows pass all recorded
quality checks. No failed timing row was removed.

At a shared `tol=1e-5`, the separate 4,096-feature panel passes the full audit;
all 60 prototype fits completed the native loop:

| Input | Previous ms | Prototype ms | sklearn ms | vs previous | vs sklearn |
|---|---:|---:|---:|---:|---:|
| float64 / 20,000 × 4,096 | 60.997 | 58.522 | 171.050 | 1.04× | 2.92× |
| float32 / 20,000 × 4,096 | 54.388 | 52.533 | 90.791 | 1.04× | 1.73× |

The largest independent gradient is 9.8271e-6, the objective gap is below
1.861e-6, and prediction agreement is 99.98%. The
[strict raw results](../benchmarks/results/metal-command-2026-10-26-strict-4096.json)
include every sample and both quality/source audits.

The full prototype suite ran **614 tests successfully, with 41 warnings**, before
the final edge-case fix. After that fix, **25 focused tests passed**; a full suite
was not rerun on that exact final source. Both benchmark source-integrity audits
passed, covering 84 recorded source, reference and binary entries per run. The
benchmarks record the loaded extension's binary hash, not just its source hash.

## Method and reproduction

The machine was an Apple M4 Max with macOS 26.6.2, Python 3.13.3, MLX 0.32.2,
NumPy 2.5.2, SciPy 1.18.1 and scikit-learn 1.9.0. Inputs are dense synthetic
binary problems with seed 41, independent normal features, labels determined by
`X[:, 0] + N(0, 0.5) > 0` (0.5 is the noise standard deviation), and weights
uniform on `[0.25, 3.0)` in the weighted case. Results do not establish
performance on differently conditioned or real datasets.

Each sample is a fresh, synchronized fit with validation, input preparation,
solver execution, direct convergence verification and output construction
included. Data generation and independent float64 quality checks are outside
the timer. The panel uses three warmups and 30 repetitions per variant and row;
all six variant orderings occur five times. Threadpoolctl requests one thread
for recognized pools, but does not control Apple Accelerate or the library's
explicit conversion workers. These are warm measurements on a non-exclusive
machine, not independent process-cold timings.

To repeat the retained library comparison from the repository checkout:

```bash
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1 \
  --output bench/metal-command-current-full.json --report bench/metal-command-current-full.md
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --case wide_20k_4096 --dtype both --tol 1e-5 \
  --repeats 30 --warmup 3 --threads 1 \
  --output bench/metal-command-current-strict-4096.json --report bench/metal-command-current-strict-4096.md
```

To reproduce the experimental panel, extract the final source archive into a
separate directory on a compatible Apple Silicon machine with the dependencies
and macOS Objective-C++ toolchain installed. This does not modify the library
checkout. The default command exits nonzero because the two marked objective
checks fail; the second command tests the strict panel. The benchmark records
`git_head`, so the commands initialize a local empty commit in the extracted
directory; source integrity is verified separately using file hashes.

```bash
experiment_dir=$(mktemp -d)
unzip -q benchmarks/reference/metal-command-final-prototype-2026-10-26.zip -d "$experiment_dir"
cd "$experiment_dir"
git init -q
git -c user.name='Benchmark' -c user.email='benchmark@example.invalid' commit --allow-empty -qm 'Benchmark snapshot'
python3 setup.py build_ext --inplace --force
PYTHONPATH="$PWD/src" python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1 \
  --output bench/metal-command-full.json --report bench/metal-command-full.md
PYTHONPATH="$PWD/src" python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --case wide_20k_4096 --dtype both --tol 1e-5 \
  --require-native-loop --repeats 30 --warmup 3 --threads 1 \
  --output bench/metal-command-strict-4096.json --report bench/metal-command-strict-4096.md
```
