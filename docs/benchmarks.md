# Benchmarks

Every number here was measured, on one machine, with nothing else benchmarking beside it:
an Apple M4 Max (16 cores, 12 performance and 4 efficiency; 128 GiB), macOS 26.6.2, MLX
0.32.2. Stock scikit-learn is 1.9.0 on Python 3.13.3 with NumPy 2.5.2 and SciPy 1.18.1;
every dispatch threshold was also checked against scikit-learn 1.7.2 on Python 3.10.19
with NumPy 2.2.6 and SciPy 1.15.3, the other end of the supported range. Through the
threshold measurements a stuck system process (a `ReportCrash` agent) held about 90% of
one core; running stock scikit-learn's neighbor search on 16, 15 or 12 OpenMP threads
moved its timings by less than their own spread. It had exited before the `published`
and `smoke` runs the tables come from.

Reproduce the tables with:

```bash
python -m benchmarks.run --profile published --report bench/published.md
python -m benchmarks.run --profile smoke --report bench/smoke.md
```

`published` holds the kernel tables below, with the settings they were measured under
(the Layer 1 classes forced onto the MLX device, 25 iterations); `smoke` measures patched
dispatch on both sides of every threshold. The grids behind the thresholds were measured
with the same harness (`benchmarks._harness.compare`) on the patched classes with their
thresholds lifted; the two profiles hold representative cases from both sides of each.
`--profile full` is the regression grid; see
[`development.md`](development.md#benchmarks).

## Method

**Synchronized.** MLX is lazy, so a benchmark that stops the clock after issuing work
measures how long it took to *schedule* — a spectacular and meaningless number. Every
timing forces evaluation first.

**Interleaved.** mlxlearn and scikit-learn are timed in pairs rather than all of one then
all of the other, so a thermal ramp or a background process degrades both sides equally
instead of whichever ran second. Within a pair the two sides take turns going first —
A B, B A, A B — and every timed call gets the same preparation, whichever backend it runs
on: a garbage collection before it and a synchronization inside its window. A harness
that always timed mlxlearn first, straight after the collection, and made only that side
pay the synchronization read identical stock work in both slots as 0.77–0.85× and failed
the gate in 18–19 of 20 trials.

**An idle GPU is part of the cost.** The collection before every timed call takes about
10 ms here, and the GPU idles through it. An MLX call made after that pays to bring the
GPU back up: a 256-query neighbor search on 32 768 × 16 took 1.3 ms back to back and
4.9 ms after a collection; 16 384 rows, 0.9 against 3.0 ms; 131 072 rows, 4.2 against
5.1 ms. A program that calls mlxlearn once after doing other work pays the same, so the
harness keeps it, and the thresholds below are set against it. A tight loop of calls does
not pay it, and runs faster than these tables say.

**Median with its spread, and best case too.** Both the median and the best-case ratio are
reported, because at small sizes a median alone can flatter either side; a crossover
requires a win on **both**.

**Significance.** A difference counts only when the two medians are three standard errors
apart, each standard error being that *of the median* — estimated from the interquartile
range as `1.2533 · (IQR / 1.349) / √n` — and combined in quadrature. Not the raw spread,
which for a heavy-tailed distribution declared a measured 10× "within noise"; and a real
interquartile range, not the full min–max range, so one outlier cannot hide a regression.
Fewer than five samples on either side is never significant. The `z = 3` rather than 1.96
is deliberate: the standard errors come from a handful of samples, and it keeps the
false-alarm rate per case near 1%. That rate is per case, and a nightly run gates well over
a hundred: simulated with 137 cases of identical work, 73–79% of nights would fail on one
measurement each. So the runner re-measures any case the gate would fail, over three
times the iterations, and records that as the result's `confirmation`; the gate fails the
case only if the loss reproduces there, which brings the simulated share of failing nights
to 0.4–0.7%. The report's `re-measured` column shows the confirmation.

**Patched dispatch is what is gated.** The performance claim is about the classes
`patch_sklearn()` installs, and those hand work to scikit-learn outright where it is
faster. The `smoke` and `full` profiles measure them by default, record which backend
each case actually ran on, and the gate judges those decisions against the configured
`Tuning` rather than a crossover inferred from the run it is judging. `--layer direct`
measures the Layer 1 classes, which never dispatch to scikit-learn and use their own CPU
path instead — useful for development, misleading as a headline. The kernel tables below
are the exception, and say so: they force the Layer 1 classes onto the MLX device, so
that both sides of every threshold are on the record; their `dispatch` column says what
patched dispatch would have done.

**Fixed hardware.** Absolute timings only compare across runs on the same machine. Shared
CI runners do correctness only; the nightly job pins itself to fixed self-hosted Apple
silicon.

## How patched dispatch decides

A threshold sends a problem to MLX only where MLX won, or tied within noise, against both
scikit-learn versions, in every configuration measured at that shape: every query batch
size for the neighbor estimators, every kernel and C for `SVC`, binary and multinomial
targets for `LogisticRegression`. A fit cannot see how many queries will follow it or how
many iterations it will take, so where the answer turns on those, the threshold takes the
side that does not lose. Everything else goes to stock scikit-learn.

| Estimator | Patched dispatch uses MLX when — all required | `Tuning` |
|---|---|---|
| neighbor estimators | at least 4 096 rows, 4 194 304 elements (rows × features) and 16 features | `knn_min_train_samples = 4096`, `knn_min_work = 4_194_304`, `knn_min_features = 16` |
| `LogisticRegression` | at least 60 000 rows and 100 000 element-ops (rows × features × weight vectors); binary: also at least 100 000 rows and at most 32 features | `logreg_min_samples = 60_000`, `logreg_min_work = 100_000`, `logreg_binary_min_samples = 100_000`, `logreg_binary_max_features = 32` |
| `SVC` | a non-linear kernel, at least 128 features, 1 024 rows and 250 000 elements; with more than two classes, a mean one-vs-one pair `(2n/K) · d` of at least 250 000 | `svc_min_features = 128`, `svc_min_samples = 1024`, `svc_min_work = 250_000`, `svc_min_pair_work = 250_000` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

### Patched dispatch against stock: the gate

The `smoke` profile measures the classes `patch_sklearn()` installs on both sides of every
threshold. Run under both scikit-learn versions, 7 interleaved pairs each, it passed the
performance gate on both (`~` within noise, **bold** a significant loss):

| case | patched dispatch | mlxlearn / stock, ms (1.9.0) | 1.9.0 | 1.7.2 |
|---|---|---|---|---|
| `knn/fit/1000x32/q256` | scikit-learn | 0.24 / 0.18 | **0.76×** | **0.80×** |
| `knn/kneighbors/1000x32/q256` | scikit-learn | 1.23 / 1.34 | 1.08×~ | 0.96×~ |
| `knn/fit/131072x32/q256` | MLX | 9.31 / 0.73 | **0.08×** | **0.07×** |
| `knn/kneighbors/131072x32/q256` | MLX | 6.41 / 9.59 | 1.50×~ | 1.70× |
| `knn/kneighbors/524288x8/q256` | scikit-learn | 34.78 / 35.22 | 1.01×~ | 0.98×~ |
| `knn/kneighbors/50000x3/q256` | scikit-learn | 0.98 / 1.01 | 1.04×~ | 0.95×~ |
| `logreg/fit/5000x32` | scikit-learn | 1.58 / 1.49 | 0.94×~ | 0.94×~ |
| `logreg/fit/200000x128` | scikit-learn | 37.37 / 37.25 | 1.00×~ | 1.00×~ |
| `logreg/fit/200000x16` | MLX | 16.14 / 27.96 | 1.73× | 1.77× |
| `logreg/fit/60000x64/n_classes=5` | MLX | 26.11 / 37.04 | 1.42× | 1.51× |
| `svc/fit/1000x32` | scikit-learn | 9.11 / 9.04 | 0.99×~ | 0.99× |
| `svc/fit/4000x64` | scikit-learn | 182.56 / 182.60 | 1.00×~ | 1.01×~ |
| `svc/fit/4000x256` | MLX | 161.69 / 606.82 | 3.75× | 3.42× |
| `svc/fit/2048x512/n_classes=4` | MLX | 29.77 / 61.88 | 2.08× | 1.89× |
| `svc/fit/2048x512/n_classes=16` | scikit-learn | 228.97 / 229.02 | 1.00×~ | 1.00×~ |

A handoff costs something of its own, about 0.05 ms for a neighbor fit: a quarter of a
stock brute-force fit at 1 000 × 32, which stores a reference and does nothing else. So
patched `knn/fit` below the floors reads 0.76×–0.80×, one more instance of the accepted
`knn/fit` regression below; the logistic and SVC handoffs were within noise of stock, or
within 1% of it.

## Neighbors

### Where MLX wins, and where it cannot

`kneighbors` on a fitted model against stock scikit-learn's brute force — which is what
`algorithm="auto"` uses from 16 features — k = 10, Gaussian data, by query batch size.
Each cell is stock time over MLX time, so above 1 MLX is faster: scikit-learn 1.9.0 /
1.7.2, median of 9 interleaved pairs. **Bold** is a significant loss by the gate's test
(three standard errors, and more than 5%); `~` is within noise. The last column is where
patched dispatch sends the fit.

| rows × features | 1 query | 16 | 256 | 4096 | fit | dispatch |
|---|---|---|---|---|---|---|
| 2 048 × 16 | 1.16 / **0.85** | 1.12~ / **0.86** | 1.14 / 0.91~ | **0.43** / **0.36** | **0.21** / **0.12** | scikit-learn |
| 4 096 × 16 | 1.10~ / **0.85** | 1.24 / **0.88** | 1.14 / **0.91** | 1.23 / 1.09 | **0.20** / **0.16** | scikit-learn |
| 32 768 × 16 | 1.12~ / **0.90** | 1.31 / **0.87** | **0.74** / **0.69** | 2.35 / 1.55 | **0.12** / **0.09** | scikit-learn |
| 65 536 × 16 | 1.19 / 1.10 | 1.07~ / 1.04~ | 0.71~ / 0.73~ | 2.15 / 1.71 | **0.10** / **0.08** | scikit-learn |
| 131 072 × 16 | 1.37 / 1.18 | 1.10~ / 1.13 | 1.60 / 1.20~ | 1.87 / 1.51 | **0.08** / **0.06** | scikit-learn |
| 2 048 × 64 | 1.10~ / 0.92~ | 0.99~ / **0.87** | 1.07~ / 1.08 | **0.42** / **0.42** | **0.19** / **0.16** | scikit-learn |
| 4 096 × 64 | 1.02~ / **0.90** | 1.17~ / 0.95~ | 1.32 / 1.30 | 1.27 / 1.68 | **0.16** / **0.13** | scikit-learn |
| 32 768 × 64 | 1.19~ / 1.07~ | 1.22~ / 1.11 | 1.03~ / **0.91** | 3.08 / 2.24 | **0.10** / **0.07** | scikit-learn |
| 65 536 × 64 | 1.31 / 1.26 | 1.49 / 1.17 | 1.05~ / 1.59 | 3.00 / 2.34 | **0.09** / **0.07** | MLX |
| 131 072 × 64 | 1.80 / 1.43 | 1.10~ / 1.19~ | 2.04 / 1.60~ | 3.31 / 2.70 | **0.08** / **0.06** | MLX |
| 2 048 × 768 | 1.10~ / 1.00~ | 1.10~ / 1.00~ | 1.35 / 1.30~ | **0.67** / **0.61** | **0.11** / **0.08** | scikit-learn |
| 4 096 × 768 | 1.17 / 1.18 | 1.32 / 1.29 | 1.62 / 1.39~ | 1.81 / 1.74 | **0.10** / **0.07** | scikit-learn |
| 32 768 × 768 | 2.11 / 2.05 | 1.96 / 1.96 | 2.13 / 2.15 | 5.90 / 5.01 | **0.08** / **0.06** | MLX |
| 65 536 × 768 | 3.77 / 2.60 | 2.43 / 2.50 | 3.68 / 2.58 | 6.35 / 6.42 | **0.08** / **0.06** | MLX |

Three things decide it, and a fit can see only two of them.

- **Rows.** Below 4 096 training rows a 4096-query batch runs at 0.25×–0.8× of stock at
  every width measured, from 2 to 768 features. Even against 256 rows MLX spends 1–4 µs a
  query row (4–18 ms for the batch), and stock brute force over so small an index costs
  less (1.4–4.5 ms).
- **Elements.** Between 4 096 rows and about 4·10⁶ elements some batch size ties or loses
  on one version or the other: 256 queries at 32 768 × 16 ran 0.74× and 0.69×, and single
  rows at 4 096 × 16 0.85× against 1.7.2. The idle-GPU cost above is most of it; it is
  largest on calls that take a few milliseconds.
- **The query batch.** Above both floors every batch won or tied on both versions: single
  rows 1.2×–3.8×, 16 queries 1.0×–2.6×, 256 queries 1.05×–4.0×, 4096 queries 1.4×–7.4×.
  It is the one of the three a fit cannot see.

Queries win; the index build does not. Stock brute force stores a reference, while
mlxlearn uploads the training rows and builds the translated, scaled copy the search runs
on, with its row norms and, for float64 input, a float32 residual: 0.06×–0.09× of the
stock fit above the floors, 8–110 ms against 0.5–8 ms. It is repaid after about 300–900
queries, in batches of 256 or 4 096 (up to 5 300 at the region's edge), which is why
`knn/fit` is listed in `benchmarks/gate.py::ACCEPTED_REGRESSIONS` rather than left to slip
through the "no crossover measured, so not gated" branch. A model fitted to answer a
handful of queries is faster on stock scikit-learn.

In memory, a float64 index holds 3.0× the float32 data on the device — rows, residual and
the translated copy — and 3.5–4.2× at the peak of the build, measured at 50 000 × 32 and
128 000 × 768.

### Narrow data goes to scikit-learn, whatever its size

With at most 15 features, scikit-learn's default `algorithm="auto"` builds a KD-tree. A
single-row query on the tree costs 0.3–1.2 ms; on MLX, 1.0–2.3 ms before any arithmetic.
That is not the idle GPU: called back to back, a single row took the tree 0.16 ms and MLX
0.73 ms at 262 144 × 8, and 0.35 against 0.56 ms at 65 536 × 12.
Against the tree, k = 10, scikit-learn 1.9.0, Gaussian data, 9 pairs; the fit column is
the tree's build time over MLX's index build:

| rows × features | 1 query | 16 | 256 | 4096 | fit |
|---|---|---|---|---|---|
| 4 096 × 2 | **0.26** | **0.26** | **0.31** | **0.31** | **0.60** |
| 65 536 × 2 | **0.29** | **0.23** | **0.10** | **0.14** | 3.59 |
| 1 048 576 × 3 | **0.16** | **0.09** | **0.05** | **0.03** | 16.09 |
| 65 536 × 4 | **0.27** | **0.25** | **0.37** | **0.50** | 4.77 |
| 1 048 576 × 5 | **0.17** | **0.15** | **0.18** | **0.16** | 17.57 |
| 65 536 × 6 | **0.31** | **0.57** | 0.61~ | 2.00 | 6.37 |
| 262 144 × 6 | **0.30** | **0.34** | 0.90~ | **0.72** | 11.03 |
| 4 096 × 8 | **0.33** | **0.45** | 2.28 | 4.47 | 0.75~ |
| 65 536 × 8 | **0.34** | **0.83** | 1.48 | 4.74 | 6.28 |
| 262 144 × 8 | **0.43** | **0.68** | 2.64 | 2.48 | 10.69 |
| 1 048 576 × 8 | **0.26** | 0.79~ | 1.36 | 1.34 | 19.88 |
| 4 096 × 12 | **0.31** | 0.75~ | 4.97 | 10.50 | 0.82~ |
| 262 144 × 12 | 0.89~ | 4.49 | 11.52 | 21.61 | 9.57 |
| 1 048 576 × 12 | **0.46** | 4.15 | 12.53 | 14.95 | 16.71 |
| 16 384 × 14 | **0.46** | 2.31 | 10.95 | 27.22 | 2.23 |
| 262 144 × 14 | 1.85 | 10.75 | 27.85 | 47.51 | 10.25 |
| 65 536 × 15 | 1.24 | 5.79 | 14.91 | — | 5.22 |

The same, on data of low intrinsic dimension — three latent dimensions embedded in 8 or
12 features, plus 5% noise (7 pairs):

| rows × features | 1 query | 16 | 256 | 4096 | fit |
|---|---|---|---|---|---|
| 65 536 × 8 | **0.25** | — | **0.38** | **0.46** | 4.86 |
| 262 144 × 8 | **0.21** | — | **0.28** | **0.20** | 9.87 |
| 65 536 × 12 | **0.24** | — | **0.46** | **0.71** | 4.91 |
| 262 144 × 12 | **0.19** | — | **0.40** | **0.27** | 8.36 |

The tree wins the single row at every size measured up to 12 features, and at 2–5
features every batch as well; spot checks against 1.7.2 at 16 384 and 262 144 rows × 8,
12 and 14 features agreed (single rows 0.25×–0.79× up to 12 features). Gaussian data is
where a KD-tree does worst. On data of low intrinsic dimension, the other extreme, the
tree won every batch. Patched dispatch therefore hands every fit with fewer than 16
features to scikit-learn: to its KD-tree where it would build one (reason `"tree-wins"`),
and to its brute force otherwise (`algorithm="brute"`, a metric the tree does not take,
`n_neighbors` of half the rows or more), which at 2–8 features also answers 16-query
batches faster, 0.78×–0.86× for MLX up to 524 288 rows.

What that gives up is real. On Gaussian data MLX answered 256-query batches 1.2×–30×
faster than the tree from 8 features and 1 024 rows up, built its index 5×–20× faster
from 65 536 rows, and at 14–15 features and 65 536 rows or more won or tied single rows
too (1.03×–4.0×). A workload known to query in large batches can import
`mlxlearn.neighbors` directly, which always searches on MLX. The previous rule,
`n ≥ 512 · 2^d`, was calibrated against the 0.1.0a2 scan: the tree wins it cited at
200 000 × 8 and 1 000 000 × 8–10 do not reproduce against the current one, and it never
looked at single rows.

### The published neighbor cases

The `published` profile's neighbor cases, reproducible with the command at the top: the
Layer 1 classes forced onto the MLX device against stock brute force, k = 10, 25
interleaved pairs, scikit-learn 1.9.0. `dispatch` is where patched dispatch sends the
fit; cases it hands to scikit-learn are on the record as evidence, not claims.

| samples × features | queries | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|---|
| 1 000 × 32 | 256 | 1.27 | 1.26 | 0.99×~ | 1.05× | scikit-learn |
| 4 096 × 64 | 256 | 1.80 | 2.01 | 1.12×~ | 0.92× | scikit-learn |
| 32 768 × 16 | 256 | 4.79 | 3.39 | **0.71×** | 0.65× | scikit-learn |
| 50 000 × 32 | 256 | 9.23 | 4.60 | **0.50×** | 1.12× | scikit-learn |
| 8 192 × 768 | 256 | 3.96 | 6.77 | 1.71× | 1.46× | MLX |
| 131 072 × 128 | 256 | 7.54 | 19.21 | 2.55× | 2.71× | MLX |
| 128 000 × 768 | 256 | 13.48 | 63.31 | 4.70× | 7.00× | MLX |
| 1 000 × 32 | 1 | 0.99 | 0.84 | 0.85×~ | 0.96× | scikit-learn |
| 1 000 × 32 | 4096 | 6.30 | 2.33 | **0.37×** | 0.35× | scikit-learn |
| 8 192 × 768 | 1 | 1.14 | 1.44 | 1.26× | 1.36× | MLX |
| 8 192 × 768 | 4096 | 34.49 | 94.20 | 2.73× | 3.16× | MLX |
| 131 072 × 128 | 1 | 1.18 | 2.31 | 1.96× | 1.84× | MLX |
| 131 072 × 128 | 4096 | 86.50 | 328.10 | 3.79× | 4.11× | MLX |

| samples × features | mlxlearn fit (ms) | scikit-learn fit (ms) | ratio |
|---|---|---|---|
| 1 000 × 32 | 1.06 | 0.20 | **0.19×** |
| 4 096 × 64 | 1.53 | 0.24 | **0.16×** |
| 32 768 × 16 | 2.23 | 0.26 | **0.12×** |
| 50 000 × 32 | 4.15 | 0.39 | **0.09×** |
| 8 192 × 768 | 10.91 | 0.97 | **0.09×** |
| 131 072 × 128 | 28.12 | 2.22 | **0.08×** |
| 128 000 × 768 | 149.00 | 12.15 | **0.08×** |

## SVC — width decides

Binary fits on the benchmark data (`y = x₀ + noise > 0`). Each cell is stock time over MLX
time, scikit-learn 1.9.0 / 1.7.2 (— where a configuration was measured on one version
only), median of 5–7 interleaved pairs; bold and `~` as above. The sigmoid kernel at
C = 1, not shown, won everywhere it was measured, 1.2×–2.0×, 16 features included.

| rows × features | rbf, C=1 | rbf, C=10 | poly, C=1 | poly, C=10 |
|---|---|---|---|---|
| 4 000 × 16 | 1.12 / 1.00~ | **0.39** / **0.36** | **0.63** / **0.57** | **0.16** / **0.15** |
| 4 000 × 32 | 1.24 / 1.12 | **0.54** / **0.49** | **0.87** / **0.78** | **0.56** / **0.51** |
| 4 000 × 64 | 1.60 / 1.45 | **0.81** / **0.74** | 1.18 / 1.05 | **0.85** / **0.77** |
| 4 000 × 128 | 2.30 / 2.10 | 1.40 / 1.24 | 1.74 / 1.54 | 1.31 / 1.19 |
| 4 000 × 256 | 3.73 / 3.44 | 2.48 / 2.24 | 2.89 / 2.56 | 2.44 / 2.18 |
| 1 000 × 128 | 1.10 / 1.02 | — / — | — / — | — / — |
| 1 000 × 256 | 1.87 / 1.75 | — / — | — / — | — / — |
| 1 000 × 512 | 3.10 / 2.85 | 1.94 / 1.75 | 2.68 / 2.38 | 2.16 / 1.91 |
| 2 000 × 256 | 2.79 / 2.57 | 1.81 / 1.62 | 2.30 / 2.04 | 1.76 / 1.57 |
| 8 000 × 32 | 1.24 / 1.15 | **0.53** / **0.49** | **0.84** / **0.76** | **0.39** / **0.36** |
| 8 000 × 128 | 2.58 / 2.44 | 1.55 / 1.43 | 1.87 / 1.71 | 1.57 / 1.48 |
| 8 000 × 256 | 4.30 / 4.13 | 3.08 / 2.90 | 3.26 / 3.06 | 3.77 / 3.58 |
| 16 000 × 8 | **0.85** / **0.74** | — / — | — / — | — / — |
| 16 000 × 64 | — / — | 1.75 / — | **0.55** / — | — / — |
| 16 000 × 128 | — / — | 3.66 / — | — / — | — / — |
| 16 000 × 256 | 8.33 / — | — / — | — / — | — / — |
| 32 000 × 256 | 16.62 / — | — / — | — / — | — / — |

SMO is sequential, and its iteration counts match LIBSVM's to within a few percent, so the
comparison is per iteration. At 4 000 rows and C = 10, mlxlearn spent 27–41 µs an
iteration from 16 to 256 features — nearly flat, because up to the kernel cache's
capacity (about 7 240 rows per one-vs-one pair at `cache_size=200`) a pair's whole Gram is
one MLX matmul and the loop then runs over it in NumPy — while LIBSVM spent 11 µs at 16
features and 103 µs at 256, because every kernel row it misses costs `O(n·d)`. They cross
near 100 features. The kernel and C change how many iterations there are and how often
LIBSVM misses, not where the crossing lies: from 128 features every kernel and C measured
won on both versions (1.2×–4.3× up to 8 000 rows above the work floor; 1 000 × 128, below
it, by only 1.02×–1.10×), while below 128 features rbf and poly at C = 10 lost everywhere.
The rule does not look at the kernel or C, so it takes the width at which all of them
win. Past the kernel cache, where rows are fetched on demand, narrow data lost
at C = 1 too (16 000 × 2–8: 0.56×–0.85×), and wide data won by more: 16 000 × 256, 8.3×;
32 000 × 256, 16.6× (6.5 s against 107 s).

A sample count or a work floor cannot see any of this. The previous floors, 2 048 rows and
10⁶ elements, could not tell 16 000 × 64 from 4 000 × 256. At 16 000 × 64, which those
floors sent to MLX, the poly kernel at C = 1 ran 0.55× while rbf at C = 10 ran 1.75×; at
16 000 × 128, rbf at C = 10 ran 3.66×.

**Many classes.** One-vs-one solves `K(K−1)/2` subproblems of about `2n/K` rows, each with
its own Gram and its own loop, so the mean pair's work must clear the same 250 000
elements the whole problem must. Balanced, well-separated blobs (the benchmark's
multiclass data), rbf, C = 1, same notation:

| rows × features | K = 4 | K = 8 | K = 16 | K = 32 | K = 64 |
|---|---|---|---|---|---|
| 4 000 × 32 | **0.35** / **0.32** | **0.25** / **0.23** | **0.18** / **0.15** | **0.11** / **0.09** | — / — |
| 8 000 × 32 | **0.39** / **0.34** | **0.34** / **0.31** | **0.25** / **0.23** | **0.18** / **0.15** | — / — |
| 8 000 × 256 | 1.88 / 1.71 | 1.69 / 1.47 | 1.23 / 1.10 | **0.77** / **0.70** | — / — |
| 2 048 × 512 | 2.03 / 1.80 | 1.36 / 1.21 | **0.82** / **0.73** | **0.47** / **0.38** | **0.24** / **0.20** |

Pairs of 512 rows × 512 features (2 048 rows, K = 8) and of 1 000 × 256 (8 000 rows,
K = 16) won; 256 × 512 and 500 × 256 lost. At 32 features, where each pair takes a
hundred-odd iterations, every K lost, as the width floor already says.

**The linear kernel** never takes the MLX path. Forced onto MLX it ran 0.09× at
2 000 × 20 and 0.04× at 2 000 × 256 (46 s against 1.8 s): its iterations are the
cheapest of any kernel's, so per-iteration overhead is nearly the whole cost, and no
width makes up for it.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check.

### The published SVC cases

Forced onto MLX, 25 interleaved pairs (5 for the linear kernel), scikit-learn 1.9.0:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 4 000 × 32 | 92.07 | 120.46 | 1.31× | 1.29× | scikit-learn |
| 4 000 × 64 | 116.46 | 188.05 | 1.61× | 1.61× | scikit-learn |
| 4 000 × 256 | 168.82 | 628.02 | 3.72× | 3.78× | MLX |
| 1 000 × 512 | 24.18 | 75.84 | 3.14× | 3.16× | scikit-learn |
| 4 000 × 64, C = 10.0 | 329.19 | 274.41 | **0.83×** | 0.84× | scikit-learn |
| 4 000 × 64, C = 10.0, kernel = poly | 313.29 | 270.34 | **0.86×** | 0.86× | scikit-learn |
| 4 000 × 256, C = 10.0 | 302.83 | 744.12 | 2.46× | 2.48× | MLX |
| 4 000 × 256, C = 10.0, kernel = poly | 276.22 | 663.94 | 2.40× | 2.44× | MLX |
| 2 000 × 20, kernel = linear (5 pairs) | 448.47 | 39.59 | **0.09×** | 0.09× | scikit-learn |
| 2 000 × 256, kernel = linear (5 pairs) | 46 122.1 | 1 771.97 | **0.04×** | 0.04× | scikit-learn |
| 2 048 × 512, K = 4 | 30.24 | 62.25 | 2.06× | 2.10× | MLX |
| 2 048 × 512, K = 16 | 281.52 | 234.22 | **0.83×** | 0.84× | scikit-learn |
| 2 500 × 20, K = 4 | 15.39 | 4.23 | **0.27×** | 0.28× | scikit-learn |

The forced wins at 4 000 × 32–64 and 1 000 × 512 that patched dispatch leaves on LIBSVM
are the price of rules that cannot see C or the iteration count, and of a sample floor
with a margin: the same widths at C = 10 lost.

## Logistic regression — narrow and tall, or many classes

`fit` with `max_iter=200`. Binary targets are the benchmark data; multinomial ones are
overlapping Gaussian blobs, one per class, whose separation does not grow with the width.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2, median of 9 interleaved
pairs; bold and `~` as above.

| binary: rows | 4 features | 16 features | 32 features | 64 features | 128 features | 256 features | 1024 features |
|---|---|---|---|---|---|---|---|
| 5 000 | **0.21** / **0.28** | **0.39** / **0.32** | **0.37** / **0.32** | — / — | — / — | — / — | — / — |
| 20 000 | **0.52** / **0.56** | **0.40** / **0.55** | **0.55** / **0.60** | — / — | **0.67** / **0.48** | **0.67** / **0.56** | **0.52** / **0.46** |
| 50 000 | 1.45 / 1.37 | **0.86** / 0.93~ | **0.93** / **0.68** | **0.92** / **0.78** | **0.86** / **0.80** | **0.74** / **0.70** | **0.60** / **0.54** |
| 100 000 | 1.67 / 1.71 | 1.33 / 1.46 | 1.17 / 1.17 | 0.96 / **0.92** | **0.92** / **0.86** | **0.81** / **0.74** | **0.73** / **0.66** |
| 200 000 | 2.12 / 2.18 | 1.76 / 1.84 | 1.33 / 1.31 | 1.04~ / 0.96 | **0.88** / **0.81** | **0.84** / **0.79** | **0.73** / **0.67** |
| 1 000 000 | — / — | 2.37 / 2.51 | — / — | 1.33 / 1.27 | 1.01~ / **0.94** | **0.85** / **0.79** | — / — |

| multinomial: rows | 3 classes × 16 | 3 classes × 64 | 3 classes × 512 | 5 classes × 64 | 10 classes × 16 | 10 classes × 128 |
|---|---|---|---|---|---|---|
| 5 000 | **0.38** / **0.37** | **0.35** / **0.48** | 1.01~ / **0.90** | **0.44** / **0.46** | **0.66** / 0.67~ | **0.59** / **0.58** |
| 20 000 | 0.95 / 0.91~ | **0.75** / 0.70~ | 1.11 / 1.05~ | 0.94~ / **0.90** | 1.83 / 1.75 | 1.26 / 1.19 |
| 60 000 | 1.27 / 1.27~ | 1.90 / 1.82 | 1.71 / 1.58 | 2.15 / 2.24 | 3.33 / 3.38 | 2.07 / 2.06 |
| 200 000 | 3.46 / 3.54 | 2.45 / 2.34 | 2.16 / 1.94 | 2.89 / 2.99 | 7.10 / 6.95 | 3.43 / 3.27 |

The L-BFGS iteration counts match scikit-learn's (7–10 against 7–9 on the binary data), so
what differs is the cost of an iteration. The solver's bookkeeping runs in float64 on the
host, one device synchronization per objective evaluation, and 0.1.0a2's per-iteration
overhead — 12–33 blocking host synchronizations per iteration in that bookkeeping, not
line-search evaluations, which average about 1.1 per iteration — is gone.

What an MLX fit spends its time on now is mostly one pass over `X`. Profiled at
100 000 × 1 024 (binary), a 127 ms fit spent 64 ms making the float32 copy of `X` and its
transpose on the device and 20 ms in the float32 range check, and 19 ms in its seven
L-BFGS iterations; scikit-learn's whole fit took 92 ms. At 1 000 000 × 16 a fit took
59 ms, of which 15 ms preparing `X` and 18 ms iterating, against 145 ms for
scikit-learn, whose per-sample loss dominates at that width. So a binary fit wins where
its iterations save more than one pass over `X` costs — tall, narrow data — and a
multinomial fit, which does a matrix product per class in every iteration, wins almost
everywhere past the fixed costs.

- **Binary, narrow and tall: faster.** From 100 000 rows at 4–32 features, 1.2×–2.7× on
  both versions; at 50 000 rows only 4–8 features win.
- **Binary and wide: slower, at every size tried.** At 256 and 1 024 features, 0.46×–0.85×
  up to 1 000 000 × 256; at 64–128 features, parity only past a million rows (1.01×–1.33×
  at 1 000 000, 0.94× for 1.7.2 at 128).
- **Multinomial: faster from 60 000 rows at every width,** 1.3×–7.3× at 3, 5 and 10
  classes and 16–512 features on both versions (one tie within noise, 3 × 16 against
  1.7.2), and 1.8×–12.7× at 1–8 features. At 20 000 rows it depends on the classes and
  the width (0.70×–1.8×); at 5 000 it loses.

The 0.1.0a2 rule — 20 000 rows and 2·10⁷ elements, both required, no width limit — sent
20 000 × 1 024 (0.52×) and 200 000 × 128 (0.88×) to MLX and kept 1 000 000 × 16 (2.4×)
from it.

### The published logistic cases

Forced onto MLX, 25 interleaved pairs, scikit-learn 1.9.0; the multinomial targets are
the benchmark's well-separated blobs:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 20 000 × 16 | 4.83 | 3.51 | **0.73×** | 0.73× | scikit-learn |
| 200 000 × 16 | 16.33 | 26.23 | 1.61× | 1.67× | MLX |
| 1 000 000 × 16 | 59.67 | 145.63 | 2.44× | 2.66× | MLX |
| 200 000 × 128 | 43.41 | 37.92 | **0.87×** | 0.89× | scikit-learn |
| 20 000 × 1024 | 38.90 | 20.04 | **0.52×** | 0.53× | scikit-learn |
| 100 000 × 1024 | 129.75 | 92.84 | **0.72×** | 0.74× | scikit-learn |
| 20 000 × 64, K = 5 | 13.44 | 11.64 | **0.87×** | 0.89× | scikit-learn |
| 60 000 × 64, K = 5 | 23.44 | 53.01 | 2.26× | 3.72× | MLX |
| 200 000 × 64, K = 5 | 47.12 | 121.23 | 2.57× | 2.96× | MLX |

## Reading these numbers honestly

- Below a threshold, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path. The kernel tables show those cases so that the thresholds can be
  checked, not as claims.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name.
- The neighbor numbers are for k = 10 on Gaussian data; `SVC`'s for the data above at C of
  1 and 10. Neither k nor C is part of any rule.
