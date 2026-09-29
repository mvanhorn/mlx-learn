# Benchmarks

Every number here was measured, on one machine, with nothing else benchmarking beside it:
an Apple M4 Max (16 cores, 12 performance and 4 efficiency; 128 GiB), macOS 26.6.2, MLX
0.32.2. Stock scikit-learn is 1.9.0 on Python 3.13.3 with NumPy 2.5.2 and SciPy 1.18.1;
every dispatch threshold was also checked against scikit-learn 1.7.2 on Python 3.10.19
with NumPy 2.2.6 and SciPy 1.15.3, the other end of the supported range. The runners log
the load average and the busiest processes before every grid; beside the benchmark itself
the busiest was the window server, at about half of one core.

Two tools produce the tables here, and the text before each says which:

```bash
# The calibration grids behind every dispatch threshold, once per scikit-learn version,
# then side by side:
python -m benchmarks.calibrate list
python -m benchmarks.calibrate run knn --output bench/calibration/knn-1.9.0.jsonl
python -m benchmarks.calibrate table knn bench/calibration/knn-*.jsonl
# ...or re-render every calibration table in this page from the committed results:
python -m benchmarks.calibrate update docs/benchmarks.md benchmarks/results/calibration/*.jsonl

# Patched dispatch against stock on both sides of every threshold (the gate), and the
# kernel cases published below:
python -m benchmarks.run --profile smoke --report bench/smoke.md
python -m benchmarks.run --profile published --report bench/published.md
```

A **calibration table** is introduced by its grid's name, and its source carries the
command that renders it (`python -m benchmarks.calibrate table <grid>`, in an HTML
comment). Every range that the README, the CHANGELOG, the private `Tuning` values and the
dispatch docstrings quote is read off one of them, by grid name. Its `dispatch` column is computed from the rules as they stand, so a table can be
checked against the thresholds it justifies. The raw results the tables were rendered from
are in [`benchmarks/results/calibration/`](../benchmarks/results/calibration/).
`smoke` measures what `patch_sklearn()` users get, on both sides of every threshold;
`published` forces the Layer 1 classes onto the MLX device and reports kernel timings.
`--profile full` is the regression grid; see [`development.md`](development.md#benchmarks).
The release-over-release timings in the CHANGELOG come from `benchmarks/ab.py`, which runs
two source trees in alternating fresh processes; its results against 0.1.0a2 are
[`benchmarks/results/ab-0.1.0a2.jsonl`](../benchmarks/results/ab-0.1.0a2.jsonl).

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

**An idle GPU is part of the cost.** The collection before every timed call idles the
GPU, and an MLX call made after that pays to bring it back up: it runs its first few
milliseconds of work at a lower clock. A 64-query neighbor search on 262 144 × 16 took
3.2 ms back to back and 5.5–8.3 ms after the GPU had idled for 10 ms — through a
collection, a sleep or busy host code; 32 queries, 2.0 against 5.0–5.1 ms; 256 queries,
8.6 against 9.1–9.4 ms (scikit-learn 1.9.0, median of 11). So the cost
is largest, relatively, for calls of a few milliseconds — which for the neighbor search
is a batch of tens to hundreds of queries against a mid-sized index, exactly where a
stock search costs about as much. A program that calls mlxlearn after doing other work
pays it, so the harness keeps it, and the thresholds below are set against it. A tight
loop of calls does not pay it, and runs faster than these tables say.

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
path instead — useful for development, misleading as a headline. The calibration grids
force the patched classes onto their MLX path, which is what patched dispatch runs when a
threshold sends a problem there, so both sides of every threshold are on the record; the
published kernel tables force the Layer 1 classes. Both say what patched dispatch would
have done.

**Fixed hardware.** Absolute timings only compare across runs on the same machine. Shared
CI runners do correctness only; the nightly job pins itself to fixed self-hosted Apple
silicon.

## How patched dispatch decides

A threshold sends a problem to MLX only where MLX won, or tied within noise, against both
scikit-learn versions, in every configuration measured at that shape: every query batch
size from 1 to 4 096, on every law of data measured, for the neighbor estimators; every
kernel, `gamma`, degree, `coef0` and C from 0.1 to 100 for `SVC`; binary and multinomial
targets, with and without signal, for `LogisticRegression`. A fit cannot see how many
queries will follow it or how many iterations it will take, so where the answer turns on
those, the threshold takes the side that does not lose. Everything else goes to stock
scikit-learn. Two rules read more than the shape: a neighbor fit probes its own rows once
its index is built, and `SVC` reads `gamma` against the data's variance.

| Estimator | Patched dispatch uses MLX when — all required | `Tuning` |
|---|---|---|
| neighbor estimators | at least 64 features, 4 096 rows and 6 000 000 elements (rows × features); with fewer than 256 features, at least 1 000 000 rows; and data the first search pass can certify — at most 8 of 64 training rows probed at fit uncertified, and, below 256 features, not integer-valued | `knn_min_features = 64`, `knn_min_train_samples = 4096`, `knn_min_work = 6_000_000`, `knn_wide_features = 256`, `knn_narrow_min_samples = 1_000_000`, `knn_probe_max_uncertified = 0.125` |
| `LogisticRegression` | at least 60 000 rows and 100 000 element-ops (rows × features × weight vectors); binary: also at least 200 000 rows and at most 64 features | `logreg_min_samples = 60_000`, `logreg_min_work = 100_000`, `logreg_binary_min_samples = 200_000`, `logreg_binary_max_features = 64` |
| `SVC` | the rbf or poly kernel, at least 128 features, 1 024 rows and 384 000 elements; with more than two classes, a mean one-vs-one pair `(2n/K) · d` of at least 384 000; rbf: `gamma` at least 0.8 × scikit-learn's `"scale"` value, or C at most 1; poly: degree 3 or more and `coef0 = 0` | `svc_min_features = 128`, `svc_min_samples = 1024`, `svc_min_work = 384_000`, `svc_min_pair_work = 384_000`, `svc_min_gamma_ratio = 0.8`, `svc_low_gamma_max_c = 1.0`, `svc_poly_min_degree = 3` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

### Patched dispatch against stock: the gate

The `smoke` profile measures the classes `patch_sklearn()` installs on both sides of every
threshold. Run under both scikit-learn versions, 7 interleaved pairs each, it passed the
performance gate on both (`~` within noise, **bold** a significant loss):

| case | patched dispatch | mlxlearn / stock, ms (1.9.0) | 1.9.0 | 1.7.2 |
|---|---|---|---|---|
| `knn/fit/1000x32/q256` | scikit-learn | 0.20 / 0.16 | 0.81×~ | **0.70×** |
| `knn/kneighbors/1000x32/q256` | scikit-learn | 1.21 / 1.21 | 1.00×~ | 0.95×~ |
| `knn/fit/65536x256/q256` | MLX | 27.82 / 2.11 | **0.08×** | **0.06×** |
| `knn/kneighbors/65536x256/q256` | MLX | 11.28 / 19.87 | 1.76×~ | 3.45× |
| `knn/kneighbors/262144x64/q256` | scikit-learn | 22.01 / 24.65 | 1.12×~ | 1.02×~ |
| `knn/kneighbors/1048576x64/q256` | MLX | 46.12 / 81.44 | 1.77× | 1.96× |
| `knn/kneighbors/1048576x16/q256` | scikit-learn | 45.57 / 45.31 | 0.99×~ | 1.03×~ |
| `knn/kneighbors/1048576x8/q256` | scikit-learn | 41.96 / 42.25 | 1.01×~ | 1.01×~ |
| `knn/kneighbors/50000x3/q256` | scikit-learn | 0.89 / 0.87 | 0.97×~ | 0.98×~ |
| `knn/kneighbors/24000x256/q256/data=dups` | scikit-learn | 9.91 / 10.06 | 1.02×~ | 1.07×~ |
| `knn/kneighbors/1048576x64/q256/data=binary` | scikit-learn | 71.77 / 74.28 | 1.04×~ | 0.99×~ |
| `logreg/fit/5000x32` | scikit-learn | 1.66 / 1.62 | 0.98×~ | 0.94×~ |
| `logreg/fit/200000x128` | scikit-learn | 37.47 / 37.20 | 0.99×~ | 0.99×~ |
| `logreg/fit/100000x16` | scikit-learn | 13.98 / 13.75 | 0.98× | 0.98×~ |
| `logreg/fit/200000x16` | MLX | 12.25 / 26.23 | 2.14× | 1.99× |
| `logreg/fit/60000x64/n_classes=5` | MLX | 17.98 / 37.04 | 2.06× | 2.09× |
| `svc/fit/1000x32` | scikit-learn | 8.57 / 8.52 | 0.99×~ | 0.99×~ |
| `svc/fit/4000x64` | scikit-learn | 182.53 / 182.48 | 1.00×~ | 1.00×~ |
| `svc/fit/4000x256` | MLX | 159.61 / 595.45 | 3.73× | 3.47× |
| `svc/fit/2048x512/n_classes=4` | MLX | 31.70 / 64.34 | 2.03× | 1.89× |
| `svc/fit/2048x512/n_classes=16` | scikit-learn | 229.55 / 229.63 | 1.00×~ | 1.00×~ |
| `svc/fit/4000x256/C=100.0/gamma=0.000390625` | scikit-learn | 931.40 / 930.89 | 1.00×~ | 1.00×~ |
| `svc/fit/3000x128/C=10.0/degree=2/kernel=poly` | scikit-learn | 258.62 / 258.42 | 1.00×~ | 1.00×~ |
| `svc/predict/4000x256/q1` | MLX (host) | 0.24 / 0.38 | 1.56× | 1.23× |
| `svc/predict/4000x256/q1000` | MLX | 2.49 / 251.91 | 101.22× | 85.30× |

A handoff costs something of its own, about 0.05 ms for a neighbor fit: a third to a half
of a stock brute-force fit at 1 000 × 32 (0.20 against 0.16 ms on 1.9.0, 0.19 against
0.13 on 1.7.2), which stores a reference and does nothing else. So patched `knn/fit` below
the floors reads 0.70×–0.81×, one more instance of the accepted `knn/fit` regression
below. The logistic and SVC handoffs were within noise of stock or within its 5%:
`logreg/fit/100000x16`, a binary fit below the binary row floor, read 0.98× on both
versions. The two data handoffs — duplicated rows at 24 000 × 256 and binary features at
1 048 576 × 64, which the size rules alone would send to MLX — answered their queries
within noise of stock. `svc/predict/…/q1` is
answered on the host, and the 1 000-row batch on the device.

## Neighbors

Every neighbor table here is `kneighbors` on a fitted model, k = 10, Gaussian data,
against stock scikit-learn — its brute force from 16 features, which is what
`algorithm="auto"` uses there, and its KD-tree below. Each cell is stock time over MLX
time, so above 1 MLX is faster. **Bold** is a loss by the gate's test (three standard
errors, and more than 5% slower); `~` is within noise. The fit column is the index build
against scikit-learn's fit, which for brute force stores a reference.

### Where MLX wins, and where it cannot

A fit cannot see how it will be queried, so the grids time twelve batch sizes, from 1 to
4 096 queries, against every index. Short of a million rows, mid-size batches lose below
256 features. Grid `knn`: 16 to 256 features, 12 000 to 1.6 million rows, both
versions; each cell is the worse version's ratio, a loss on either version shown as a
loss:

<!-- python -m benchmarks.calibrate table knn: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120 000 × 16 | 0.92~ | 1.17~ | 1.26 | 0.96~ | **0.81** | **0.62** | **0.58** | 0.67~ | 1.01~ | 1.18 | 1.21 | 1.12 | **0.07** | scikit-learn |
| 180 000 × 16 | 1.10~ | 1.24 | 1.33 | 1.01~ | **0.69** | **0.55** | **0.46** | **0.70** | 1.05~ | 1.16 | 1.17 | 0.93~ | **0.07** | scikit-learn |
| 262 144 × 16 | 1.27~ | 1.74 | 1.64 | 1.25 | 0.92~ | **0.64** | **0.63** | 1.29~ | 1.46 | 1.54 | 1.55 | 1.37 | **0.07** | scikit-learn |
| 400 000 × 16 | 1.32~ | 1.86 | 1.55 | 1.16 | **0.73** | **0.49** | 1.03~ | 1.20 | 1.14 | 1.11 | 1.02~ | 0.81~ | **0.06** | scikit-learn |
| 600 000 × 16 | 1.33~ | 2.15 | 1.72 | 1.21 | **0.77** | 0.74~ | 1.24 | 1.20 | 1.16 | 1.16 | 1.01~ | 0.97~ | **0.05** | scikit-learn |
| 1 000 000 × 16 | 1.65 | 2.59 | 1.90 | 1.17 | 1.21~ | 1.45 | 1.31 | 1.25 | 1.23 | 1.10~ | 1.08~ | 1.17~ | **0.05** | scikit-learn |
| 1 600 000 × 16 | 1.92 | 2.97 | 2.25 | 1.95 | 1.94 | 1.63 | 1.58 | 1.55 | 1.48 | 1.33 | 1.42 | 1.17~ | **0.05** | scikit-learn |
| 80 000 × 24 | 1.09~ | 1.12~ | 1.02~ | 0.97~ | **0.84** | **0.67** | **0.53** | **0.46** | **0.64** | 1.25 | 1.37 | 1.43 | **0.08** | scikit-learn |
| 120 000 × 24 | 1.17 | 1.29 | 1.21 | 0.90~ | **0.73** | **0.62** | **0.47** | **0.44** | 1.26 | 1.28 | 1.38 | 1.20 | **0.07** | scikit-learn |
| 180 000 × 24 | 1.12~ | 1.22~ | 1.18 | 1.01~ | **0.75** | **0.61** | **0.49** | **0.62** | 1.21 | 1.27 | 1.40 | 1.02~ | **0.07** | scikit-learn |
| 262 144 × 24 | 1.32~ | 1.60 | 1.40 | 1.23 | 0.89~ | **0.68** | 0.65~ | 1.59 | 1.65 | 1.77 | 1.91 | 1.48 | **0.07** | scikit-learn |
| 400 000 × 24 | 1.32 | 1.66 | 1.39 | 1.15 | **0.80** | **0.55** | 1.20 | 1.28 | 1.27 | 1.32 | 1.05~ | 0.91~ | **0.06** | scikit-learn |
| 600 000 × 24 | 1.05~ | 1.89 | 1.52 | 1.08 | **0.70** | 0.97~ | 1.28 | 1.28 | 1.28 | 1.28 | 1.03~ | 1.02~ | **0.06** | scikit-learn |
| 1 000 000 × 24 | 1.82 | 2.08 | 1.62 | 1.12 | 1.22~ | 1.55 | 1.38 | 1.39 | 1.37 | 1.17 | 1.12 | 1.19~ | **0.05** | scikit-learn |
| 1 600 000 × 24 | 1.90 | 2.32 | 1.80 | 2.05 | 1.34 | 1.63 | 1.67 | 1.69 | 1.68 | 1.36 | 1.50 | — | **0.05** | scikit-learn |
| 80 000 × 32 | 1.01~ | 1.05~ | 1.09~ | 0.96~ | **0.84** | **0.70** | **0.60** | **0.47** | 0.91~ | 1.25 | 1.39 | 1.67 | **0.08** | scikit-learn |
| 120 000 × 32 | 1.16~ | 1.23 | 1.06~ | 0.93~ | **0.86** | **0.62** | **0.52** | 0.67~ | 1.30 | 1.35 | 1.50 | 1.37 | **0.07** | scikit-learn |
| 180 000 × 32 | 1.11~ | 1.38 | 1.23 | 1.05~ | **0.76** | **0.57** | 0.82~ | 0.84~ | 1.14 | 1.33 | 1.52 | 1.13 | **0.07** | scikit-learn |
| 262 144 × 32 | 1.10~ | 1.56 | 1.39 | 1.19 | 0.94~ | **0.76** | 0.86~ | 1.55 | 1.65 | 1.87 | 2.09 | 1.66 | **0.06** | scikit-learn |
| 400 000 × 32 | 1.14~ | 1.63 | 1.46 | 1.12 | **0.75** | 0.78~ | 0.91~ | 1.34 | 1.36 | 1.45 | 1.16 | 1.00~ | **0.06** | scikit-learn |
| 600 000 × 32 | 1.29~ | 1.96 | 1.54 | 1.12 | 1.19~ | 1.32~ | 1.47 | 1.40 | 1.39 | 1.39 | 1.19 | 1.14~ | **0.06** | scikit-learn |
| 1 000 000 × 32 | 1.77 | 2.17 | 1.59 | 1.13 | 1.44 | 1.56 | 1.46 | 1.44 | 1.43 | 1.18 | 1.20 | — | **0.06** | scikit-learn |
| 1 600 000 × 32 | 2.10 | 2.30 | 1.82 | 1.75 | 1.88 | 1.84 | 2.25 | 1.97 | 1.86 | 1.50 | 1.63 | — | **0.05** | scikit-learn |
| 50 000 × 48 | 0.86~ | 1.05~ | 1.04~ | 0.98~ | 0.80~ | **0.78** | **0.66** | **0.58** | 0.84~ | 0.97~ | 1.47 | 1.95 | **0.08** | scikit-learn |
| 80 000 × 48 | 1.09~ | 1.04~ | 1.10 | 1.03~ | **0.85** | **0.72** | **0.63** | **0.55** | 0.99~ | 1.24~ | 1.72 | 1.91 | **0.07** | scikit-learn |
| 120 000 × 48 | 1.11~ | 1.18 | 1.10~ | 1.02~ | 0.85~ | **0.66** | **0.62** | 0.89~ | 1.04~ | 1.61 | 1.85 | 1.72 | **0.06** | scikit-learn |
| 180 000 × 48 | 1.26 | 1.35 | 1.17 | 1.09~ | **0.81** | **0.72** | 0.99~ | 1.03~ | 1.46 | 1.72 | 1.81 | 1.48 | **0.06** | scikit-learn |
| 262 144 × 48 | 1.41 | 1.53 | 1.44 | 1.25 | 0.98~ | **0.77** | 0.70~ | 1.69 | 1.84 | 2.18 | 2.54 | 2.20 | **0.06** | scikit-learn |
| 400 000 × 48 | 1.61 | 1.60 | 1.36 | 1.04~ | **0.82** | **0.62** | 1.55 | 1.64 | 1.77 | 1.93 | 1.67 | 1.43 | **0.06** | scikit-learn |
| 600 000 × 48 | 1.91 | 1.86 | 1.51 | 1.09 | 0.79~ | **0.90** | 1.63 | 1.58 | 1.76 | 1.73 | 1.46 | 1.45~ | **0.06** | scikit-learn |
| 1 000 000 × 48 | 2.21 | 1.95 | 1.45 | 1.11~ | 1.15~ | 1.99 | 2.07 | 1.88 | 1.75 | 1.44 | 1.49 | — | **0.06** | scikit-learn |
| 1 600 000 × 48 | 2.57 | 2.07 | 1.63 | 1.89 | 2.61 | 2.08 | 2.56 | 2.29 | 2.06 | 1.78 | 1.78 | — | **0.05** | scikit-learn |
| 33 000 × 64 | 0.92~ | 0.96~ | 0.92~ | **0.78** | 0.89~ | **0.85** | **0.76** | **0.70** | **0.54** | 0.55~ | 1.68 | 2.25 | **0.09** | scikit-learn |
| 50 000 × 64 | 1.13 | 1.15~ | 1.01~ | 0.90~ | **0.90** | 0.86~ | **0.72** | **0.63** | 0.80~ | 1.17~ | 1.52 | 2.50 | **0.08** | scikit-learn |
| 80 000 × 64 | 1.10~ | 1.23 | 1.02~ | 0.99~ | 0.89~ | **0.72** | **0.67** | **0.57** | 1.04~ | 1.95 | 2.01 | 2.46 | **0.07** | scikit-learn |
| 120 000 × 64 | 1.25~ | 1.21 | 1.11 | 1.05~ | **0.86** | **0.72** | **0.66** | 0.91~ | 1.50 | 1.95 | 2.39 | 2.18 | **0.07** | scikit-learn |
| 180 000 × 64 | 1.37~ | 1.43 | 1.22 | 1.04~ | 0.87~ | **0.72** | 0.65~ | 1.34 | 1.68 | 2.01 | 2.33 | 1.97 | **0.06** | scikit-learn |
| 262 144 × 64 | 1.53~ | 1.43 | 1.24 | 1.23 | 0.94~ | **0.89** | **0.78** | 1.52 | 2.11 | 2.66 | 3.09 | 2.86 | **0.06** | scikit-learn |
| 400 000 × 64 | 1.71 | 1.65 | 1.40 | 1.03~ | **0.85** | 0.75~ | 1.65 | 1.83 | 2.18 | 2.45 | 2.08 | 1.72 | **0.06** | scikit-learn |
| 600 000 × 64 | 2.06 | 1.61 | 1.37 | 1.13 | 1.15~ | 1.70 | 1.76 | 1.95 | 2.10 | 2.14 | 1.87 | — | **0.06** | scikit-learn |
| 1 000 000 × 64 | 2.37 | 1.75 | 1.44 | 1.38~ | 2.27 | 2.57 | 2.22 | 2.13 | 2.24 | 1.71 | 1.79 | — | **0.06** | MLX |
| 1 600 000 × 64 | 2.70 | 1.91 | 2.52 | 1.45 | 2.83 | 2.66 | 2.70 | 2.69 | 2.24 | 2.09 | 2.03 | — | **0.06** | MLX |
| 20 000 × 96 | 0.83~ | 0.93~ | 1.01~ | 0.98~ | 0.93~ | 1.02~ | **0.90** | **0.81** | 0.82~ | **0.65** | 1.14~ | 1.83 | **0.11** | scikit-learn |
| 33 000 × 96 | 0.99~ | 1.08~ | 1.06~ | 0.98~ | 0.96~ | 0.91~ | 0.84~ | **0.76** | **0.60** | 1.07~ | 1.82 | 2.84 | **0.08** | scikit-learn |
| 50 000 × 96 | 1.15~ | 1.16 | 1.08~ | 0.93~ | 1.01~ | 0.91~ | 0.87~ | **0.71** | 0.65~ | 1.53 | 2.39 | 3.47 | **0.07** | scikit-learn |
| 80 000 × 96 | 1.35 | 1.17~ | 1.01~ | 1.00~ | 1.00~ | 0.87~ | 0.78~ | 0.65~ | 1.21~ | 1.78 | 2.46 | 3.33 | **0.07** | scikit-learn |
| 120 000 × 96 | 1.43 | 1.40 | 1.26 | 1.13 | 0.95~ | **0.83** | **0.73** | 0.93~ | 1.96 | 2.84 | 3.10 | 2.95 | **0.06** | scikit-learn |
| 180 000 × 96 | 1.66 | 1.38 | 1.26 | 1.02~ | 0.95~ | **0.80** | **0.74** | 1.33~ | 2.36 | 2.59 | 2.89 | 2.67 | **0.07** | scikit-learn |
| 262 144 × 96 | 1.79 | 1.40 | 1.40 | 1.27 | 1.09~ | 0.98~ | 1.11~ | 2.39 | 3.21 | 3.04 | 3.61 | 3.64 | **0.06** | scikit-learn |
| 400 000 × 96 | 2.03 | 1.52 | 1.35 | 1.18 | **0.90** | 1.05~ | 1.79 | 2.87 | 2.81 | 3.02 | 2.53 | — | **0.06** | scikit-learn |
| 600 000 × 96 | 2.35 | 1.72 | 1.45 | 1.13 | 1.00~ | 1.66 | 2.72 | 2.73 | 2.82 | 2.70 | 2.49 | — | **0.06** | scikit-learn |
| 1 000 000 × 96 | 2.34 | 1.65 | 1.50 | 1.91 | 1.61 | 2.64 | 3.04 | 2.95 | 2.68 | 2.27 | 2.24 | — | **0.06** | MLX |
| 20 000 × 128 | 1.03~ | 1.03~ | 1.11~ | 0.95~ | 0.93~ | 0.96~ | 0.93~ | 0.86~ | **0.75** | 1.09~ | 2.23 | 2.49 | **0.08** | scikit-learn |
| 33 000 × 128 | 1.05~ | 1.14~ | 1.13~ | 1.08~ | 0.94~ | 0.96~ | 0.98~ | **0.80** | **0.79** | 1.08~ | 1.95 | 3.75 | **0.08** | scikit-learn |
| 50 000 × 128 | 1.21~ | 1.16~ | 1.11 | 1.03~ | 1.00~ | 0.94~ | 0.91~ | **0.77** | 1.01~ | 2.34 | 3.44 | 3.99 | **0.08** | scikit-learn |
| 80 000 × 128 | 1.38 | 1.15 | 1.14 | 1.11~ | 0.96~ | **0.91** | **0.80** | **0.88** | 1.11~ | 2.88 | 3.35 | 3.49 | **0.07** | scikit-learn |
| 120 000 × 128 | 1.52 | 1.39 | 1.29 | 1.23 | 1.09~ | 0.95~ | **0.80** | 1.27~ | 2.43 | 3.11 | 3.40 | 3.39 | **0.07** | scikit-learn |
| 180 000 × 128 | 1.74 | 1.62 | 1.28 | 1.16 | 1.02~ | **0.90** | 0.97~ | 1.90 | 2.69 | 2.94 | 3.31 | 3.20 | **0.06** | scikit-learn |
| 262 144 × 128 | 2.18 | 1.49 | 1.46 | 1.29 | 1.15 | 1.19 | 1.47 | 2.18 | 2.71 | 3.89 | 3.92 | — | **0.06** | scikit-learn |
| 400 000 × 128 | 2.26 | 1.59 | 1.40~ | 1.20 | 0.99~ | 1.16~ | 2.77 | 3.19 | 3.20 | 3.40 | 3.10 | — | **0.06** | scikit-learn |
| 600 000 × 128 | 2.62 | 1.63 | 1.51 | 1.30 | 1.89 | 1.50~ | 3.20 | 2.72 | 3.29 | 3.41 | 2.91 | — | **0.06** | scikit-learn |
| 1 000 000 × 128 | 3.46 | 1.87 | 1.63 | 1.77 | 2.69 | 3.03 | 3.33 | 3.35 | 3.29 | 2.78 | — | — | **0.06** | MLX |
| 12 000 × 256 | 1.06~ | 1.08~ | 1.06~ | 1.02~ | 1.03~ | 1.05~ | 1.18 | 1.08~ | 1.25 | 1.48 | 1.77 | 2.69 | **0.08** | scikit-learn |
| 20 000 × 256 | 1.10~ | 1.20 | 1.10~ | 1.03~ | 1.16 | 1.11~ | 1.21~ | 1.10~ | 1.28~ | 1.65 | 2.23 | 3.67 | **0.08** | scikit-learn |
| 33 000 × 256 | 1.27 | 1.21 | 1.16~ | 1.29 | 1.12~ | 1.15 | 1.20~ | 1.39 | 1.50 | 2.43 | 3.83 | 4.94 | **0.07** | MLX |
| 50 000 × 256 | 1.52 | 1.38 | 1.34 | 1.22 | 1.18 | 1.20 | 1.34 | 1.42 | 1.75 | 3.10 | 4.08 | 4.58 | **0.07** | MLX |
| 80 000 × 256 | 1.95 | 1.39 | 1.44 | 1.35 | 1.26 | 1.20 | 1.61 | 1.39 | 2.22 | 3.38 | 3.49 | 4.29 | **0.07** | MLX |
| 120 000 × 256 | 1.97 | 1.58 | 1.51 | 1.45 | 1.25 | 1.52 | 1.58 | 2.42 | 2.77 | 4.45 | 4.19 | — | **0.06** | MLX |
| 180 000 × 256 | 2.40 | 1.74 | 1.64 | 1.39 | 1.41 | 1.62 | 2.24 | 2.99 | 3.36 | 4.51 | 4.32 | — | **0.06** | MLX |
| 262 144 × 256 | 3.15 | 2.05 | 1.91 | 1.68 | 1.88 | 1.84 | 2.32 | 3.94 | 4.10 | 5.35 | 5.52 | — | **0.06** | MLX |
| 400 000 × 256 | 3.53 | 2.02 | 2.06 | 1.70 | 1.92 | 2.36 | 2.69 | 4.49 | 4.34 | 4.89 | 4.59 | — | **0.06** | MLX |

Three things decide it, and a fit can see only two of them.

- **Width.** Stock brute force pays for every feature of every training row on every
  query; MLX's search costs about the same per query at 16 features as at 256, and what
  it pays on top is fixed — a launch, and the wake-up of an idled GPU. At 256 features
  the stock search is expensive enough to cover that from 12 000 rows on; at 16–128
  features it is not, until the index is large.
- **Rows.** Below 256 features, some batch of 8–512 queries lost on one version or both at
  42 of the 50 sizes measured below 1 000 000 rows, down to 0.44× — still at 400 000 rows
  at 64 and 96 features. From 1 000 000 rows this table shows every batch winning or
  tying, but on Gaussian data only; the narrow rows did not hold up on other data, below.
  Large batches rarely lose below a million rows: 1 024 queries or more did so nowhere in
  this table (up to 4.3×), which is why a workload known to query in large batches should
  import `mlxlearn.neighbors` directly.
- **The query batch.** It is the one of the three a fit cannot see, and the rules are set
  so that it does not matter: every shape patched dispatch sends to MLX won or tied at
  every batch size, on both versions.

What the million-row rule gives up is mostly the idle GPU's doing: called back to back, 64
queries against 262 144 × 16 took MLX 3.2 ms, and stock brute force about 5 ms. Answering
mid-size batches somewhere other than the fitted backend would re-decide the backend at
inference, which the sticky-backend contract rules out; a search that keeps the GPU from
idling, or hides its wake-up, is future work.

At a million rows the narrow widths were measured again, on Gaussian data and on two other
laws the fit-time probe (below) passes — 100 clusters of std 1, and uniform — with
everything else idle. Grid `knn-narrow`, k = 10:

<!-- python -m benchmarks.calibrate table knn-narrow: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | 1 query | 4 queries | 16 queries | 64 queries | 256 queries | 1024 queries | probe | dispatch |
|---|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, gauss | 1.53 / 2.06 | 1.93 / 1.91 | 0.75~ / 1.60~ | 1.07~ / 1.52~ | 0.88~ / 0.93~ | 0.98~ / **0.63** | 0/64 | scikit-learn |
| 1 000 000 × 16, blobs | 1.60 / 2.03 | 1.80 / 1.93 | 0.76~ / 0.78~ | 1.01~ / 1.19~ | 0.97~ / 0.91~ | 1.08 / **0.62** | 0/64 | scikit-learn |
| 1 000 000 × 16, uniform | 1.64 / 2.07 | 2.01 / 1.86 | **0.73** / 1.63~ | 1.06~ / 0.96~ | 1.11~ / 0.96~ | **0.69** / **0.66** | 0/64 | scikit-learn |
| 1 000 000 × 32, gauss | 1.80 / 2.52 | 1.73 / 1.69 | **0.78** / 0.89~ | 1.37~ / 1.33~ | 1.15~ / 1.05~ | 1.15 / **0.73** | 0/64 | scikit-learn |
| 1 000 000 × 32, blobs | 1.81 / 2.44 | 1.68 / 1.72 | 1.61~ / 0.80~ | 1.06~ / 1.25~ | 1.14~ / 1.09~ | 1.18 / **0.79** | 0/64 | scikit-learn |
| 1 000 000 × 32, uniform | 1.87 / 2.50 | 1.70 / 1.66 | 0.92~ / 1.50~ | 1.37~ / 1.32~ | 1.26 / 1.07~ | **0.79** / **0.74** | 0/64 | scikit-learn |
| 1 600 000 × 32, gauss | 2.10 / 3.02 | 1.87 / 1.83 | 2.35 / 1.18~ | 2.02 / 2.01 | 1.48 / 1.56 | 1.55 / 0.93~ | 0/64 | scikit-learn |
| 1 600 000 × 32, blobs | 2.05 / 3.24 | 1.77 / 1.89 | 2.28 / 2.08 | 1.41 / 2.02 | 1.52 / 1.39 | 1.16~ / 0.94~ | 0/64 | scikit-learn |
| 1 600 000 × 32, uniform | 2.07 / 3.07 | 1.97 / 1.82 | 2.17~ / 1.10~ | 1.65 / 1.93 | 1.66 / 1.51 | 1.01~ / 0.95~ | 0/64 | scikit-learn |
| 1 000 000 × 48, gauss | 2.21 / 2.79 | 1.57 / 1.51 | **0.84** / 0.91~ | 1.81~ / 1.69~ | 1.26 / 1.32 | 1.48 / 0.90~ | 0/64 | scikit-learn |
| 1 000 000 × 48, blobs | 2.06 / 2.82 | 1.49 / 1.58 | 0.83~ / 0.94~ | 1.39~ / 1.93 | 1.36 / 1.25 | 1.28 / 0.92~ | 0/64 | scikit-learn |
| 1 000 000 × 48, uniform | 2.19 / 2.90 | 1.53 / 1.60 | 0.94~ / 1.81~ | 2.01 / 1.98 | 1.40 / 1.35 | 0.95~ / 0.96~ | 0/64 | scikit-learn |
| 1 000 000 × 64, gauss | 2.44 / 3.08 | 1.55 / 1.50 | 2.12 / 0.94~ | 1.94 / 2.26 | 1.64 / 1.59 | 1.81 / 1.21~ | 0/64 | MLX |
| 1 000 000 × 64, blobs | 2.42 / 3.01 | 1.54 / 1.46 | 1.12~ / 1.08~ | 1.67~ / 2.03 | 1.60 / 1.55 | 1.50 / 1.30 | 0/64 | MLX |
| 1 000 000 × 64, uniform | 2.43 / 2.90 | 1.58 / 1.58 | 1.09~ / 1.07~ | 1.56 / 2.10 | 1.85 / 1.74 | 1.13~ / 1.30 | 0/64 | MLX |
| 1 600 000 × 64, gauss | 2.70 / 2.69 | 1.72 / 1.77 | 2.56 / 2.87 | 1.71 / 1.82 | 1.84 / 2.08 | 1.70 / 1.30 | 0/64 | MLX |
| 1 600 000 × 64, blobs | 2.75 / 3.04 | 1.71 / 1.69 | 2.97 / 2.78 | 1.87 / 1.80 | 2.11 / 2.03 | 1.37 / 1.34 | 0/64 | MLX |
| 1 600 000 × 64, uniform | 2.76 / 2.69 | 1.95 / 1.76 | 1.49 / 2.87 | 1.80 / 1.60 | 2.22 / 2.06 | 1.37 / 1.32 | 0/64 | MLX |
| 1 000 000 × 96, gauss | 2.66 / 2.80 | 1.52 / 1.77 | 1.14~ / 2.66 | 1.65~ / 1.88 | 2.41 / 2.57 | 2.83 / 2.08 | 0/64 | MLX |
| 1 000 000 × 96, blobs | 2.64 / 2.66 | 1.52 / 1.58 | 1.26~ / 1.50~ | 1.58~ / 1.82 | 2.43 / 2.61 | 2.38 / 2.20 | 0/64 | MLX |
| 1 000 000 × 96, uniform | 2.56 / 2.59 | 1.60 / 1.53 | 1.25~ / 2.45 | 1.91 / 1.66 | 2.75 / 2.45 | 2.09 / 2.34 | 0/64 | MLX |
| 1 000 000 × 128, gauss | 3.58 / 3.12 | 1.66 / 1.71 | 2.64 / 2.80 | 2.03 / 2.35 | 2.93 / 2.93 | — | 0/64 | MLX |
| 1 000 000 × 128, blobs | 3.20 / 3.43 | 1.58 / 1.70 | 2.60 / 1.61~ | 1.81 / 2.38 | 2.86 / 2.99 | — | 0/64 | MLX |
| 1 000 000 × 128, uniform | 3.27 / 3.43 | 1.70 / 1.80 | 3.12 / 3.01 | 2.25 / 2.27 | 3.16 / 2.89 | — | 0/64 | MLX |

At 16–48 features some batch lost on one version or both at every width but 1 600 000 ×
32 — 16 queries at 0.73×–0.84×, where a call made after the GPU has idled costs MLX about
10–12 ms against 8–10 ms for stock brute force, and 1 024 at 0.62×–0.79× — while the `knn`
table had shown the same shapes winning. From 64 features every batch won or tied on both
versions (0.94×~ at worst, 16 queries at 1 000 000 × 64 on 1.7.2). So patched dispatch
hands every fit with fewer than 64 features to scikit-learn, whatever its rows
(`knn_min_features = 64`): to its KD-tree up to 15 features, to its brute force from 16.

No rule reads k. The tables above are k = 10; at the narrow corner the rules now keep,
grid `knn-k` times k = 1 and 50 — at 1 000 000 × 16 both had lost a 1 024-query batch
in one run and tied in another:

<!-- python -m benchmarks.calibrate table knn-k: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, k | 1 query | 16 queries | 64 queries | 256 queries | 1024 queries | dispatch |
|---|---|---|---|---|---|---|
| 1 000 000 × 64, k = 1 | 2.41 / 2.35 | 1.16~ / 1.23~ | 1.92 / 1.89 | 1.49 / 1.55 | 1.70 / 1.75 | MLX |
| 1 000 000 × 64, k = 50 | 2.19 / 2.85 | 1.95~ / 0.93~ | 1.49~ / 1.74~ | 1.44 / 1.21~ | 1.48 / 1.44 | MLX |
| 1 600 000 × 64, k = 1 | 2.71 / 2.68 | 2.55 / 1.62~ | 1.70 / 1.56 | 1.82 / 1.75 | 2.01 / 1.86 | MLX |
| 1 600 000 × 64, k = 50 | 2.67 / 2.48 | 2.29 / 2.15 | 1.95 / 1.43 | 1.52 / 1.50 | 1.62 / 1.24 | MLX |
| 1 000 000 × 96, k = 1 | 2.74 / 2.62 | 2.25 / 2.28 | 1.93 / 1.83 | 2.40 / 2.37 | 2.88 / 2.97 | MLX |
| 1 000 000 × 96, k = 50 | 2.59 / 2.46 | 1.08~ / 2.09 | 1.47~ / 1.54~ | 2.11 / 2.16 | 2.47 / 2.36 | MLX |
| 1 000 000 × 128, k = 1 | 3.16 / 3.26 | 2.51 / 2.63 | 2.17 / 1.93 | 2.93 / 2.94 | — | MLX |
| 1 000 000 × 128, k = 50 | 3.11 / 2.93 | 2.36 / 2.28 | 1.75 / 1.99 | 2.50 / 2.31 | — | MLX |

Every batch won or tied on both versions, 0.93×~ at worst (16 queries, k = 50,
1 000 000 × 64 on 1.7.2).

Wide data needs rows too, but far fewer. Grid `knn-rows`: every batch size again, at 256
to 1 024 features, 2 048 to 131 072 rows; same notation:

<!-- python -m benchmarks.calibrate table knn-rows: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 048 × 256 | **0.74** | **0.73** | **0.70** | **0.68** | **0.70** | **0.81** | **0.87** | 0.92~ | 0.89~ | 1.23~ | 1.26~ | **0.68** | **0.09** | scikit-learn |
| 4 096 × 256 | **0.76** | **0.76** | **0.70** | **0.80** | **0.82** | **0.83** | 0.98~ | 1.08~ | 1.18 | 1.52 | 1.72 | 1.59 | **0.10** | scikit-learn |
| 8 192 × 256 | **0.82** | 0.92~ | **0.86** | **0.85** | 0.95~ | 0.99~ | 1.11~ | 1.16~ | 1.37 | 1.70 | 1.93 | 2.17 | **0.08** | scikit-learn |
| 16 384 × 256 | 0.92~ | 1.02~ | 1.01~ | 0.98~ | 1.05~ | 1.15 | 1.20 | 1.37 | 1.51 | 1.60 | 2.49 | 3.70 | **0.07** | scikit-learn |
| 32 768 × 256 | 1.21 | 1.23 | 1.19 | 1.16 | 1.22 | 1.26 | 1.34 | 1.67 | 1.76 | 2.02 | 2.85 | 4.82 | **0.07** | MLX |
| 65 536 × 256 | 1.66 | 1.40 | 1.38 | 1.23 | 1.31 | 1.49 | 1.90 | 1.98 | 1.80 | 2.54 | 3.71 | 6.20 | **0.07** | MLX |
| 131 072 × 256 | 2.21 | 1.62 | 1.54 | 1.52 | 1.45~ | 1.84 | 1.97 | 2.26 | 2.91 | 3.79 | 5.24 | — | **0.06** | MLX |
| 2 048 × 384 | **0.71** | **0.77** | **0.70** | **0.71** | **0.78** | **0.85** | **0.82** | 0.90~ | 1.10~ | 1.35~ | 1.59 | **0.44** | **0.10** | scikit-learn |
| 4 096 × 384 | **0.79** | **0.82** | **0.78** | **0.77** | **0.83** | 0.94~ | 0.98~ | 1.15~ | 1.12~ | 1.27 | 1.44 | 1.50 | **0.09** | scikit-learn |
| 8 192 × 384 | 0.91~ | 0.92~ | 0.95~ | 0.96~ | 1.00~ | 1.04~ | 1.16~ | 1.39~ | 1.36 | 1.38 | 1.53 | 1.52 | **0.08** | scikit-learn |
| 16 384 × 384 | 1.02~ | 1.13 | 1.14 | 1.08~ | 1.16~ | 1.25 | 1.45 | 1.83 | 1.33 | 1.46 | 1.96 | 2.77 | **0.07** | MLX |
| 32 768 × 384 | 1.38 | 1.40 | 1.37 | 1.31 | 1.41 | 1.43 | 1.84 | 2.10 | 1.64 | 1.71 | 2.43 | 3.28 | **0.06** | MLX |
| 65 536 × 384 | 1.84 | 1.66 | 1.51 | 1.49 | 1.58 | 1.92 | 2.31 | 2.12 | 1.69 | 2.26 | 3.33 | 4.62 | **0.07** | MLX |
| 131 072 × 384 | 2.61 | 2.01 | 1.84 | 1.86 | 2.04 | 2.25 | 2.17 | 3.12 | 2.06 | 3.12 | 4.24 | — | **0.06** | MLX |
| 2 048 × 512 | **0.74** | **0.76** | **0.73** | **0.78** | **0.79** | **0.79** | **0.78** | **0.72** | 0.94~ | 1.30~ | 1.58 | **0.55** | **0.09** | scikit-learn |
| 4 096 × 512 | **0.79** | 0.86~ | **0.81** | **0.82** | 0.92~ | 1.00~ | 0.93~ | 1.06~ | 1.19 | 1.19~ | 1.40 | 1.46 | **0.08** | scikit-learn |
| 8 192 × 512 | 0.90~ | 1.11 | 1.02~ | 1.05~ | 1.08 | 1.19 | 1.27 | 1.52 | 1.44 | 1.40 | 1.81 | 1.81 | **0.07** | scikit-learn |
| 16 384 × 512 | 1.16 | 1.35 | 1.31 | 1.34 | 1.23~ | 1.30 | 1.81 | 2.27 | 1.51 | 1.89 | 2.13 | 3.03 | **0.07** | MLX |
| 32 768 × 512 | 1.60 | 1.58 | 1.55 | 1.43 | 1.50 | 1.60 | 2.23 | 2.27 | 1.74 | 2.81 | 3.31 | 4.55 | **0.07** | MLX |
| 65 536 × 512 | 2.24 | 1.97 | 1.77 | 1.79 | 2.05 | 2.32 | 2.89 | 2.69 | 2.06 | 3.06 | 3.76 | — | **0.06** | MLX |
| 131 072 × 512 | 2.93 | 2.36 | 2.46 | 2.10 | 2.29 | 2.50 | 3.11 | 3.71 | 2.97 | 3.98 | 5.59 | — | **0.06** | MLX |
| 2 048 × 768 | **0.82** | **0.73** | **0.73** | **0.79** | **0.81** | **0.79** | **0.81** | 1.03~ | 1.15~ | 1.38~ | 1.23~ | **0.60** | **0.09** | scikit-learn |
| 4 096 × 768 | 0.89~ | **0.95** | **0.88** | 0.93~ | 0.98~ | 1.15~ | 1.17 | 1.21~ | 1.26 | 1.22~ | 1.43 | 1.64 | **0.08** | scikit-learn |
| 8 192 × 768 | 1.10 | 1.17 | 1.04~ | 1.15 | 1.18 | 1.26 | 1.41 | 1.38 | 1.43 | 1.93 | 2.12 | 2.56 | **0.07** | MLX |
| 16 384 × 768 | 1.43 | 1.40 | 1.38 | 1.37 | 1.57 | 1.60 | 2.18 | 1.65 | 1.79 | 2.07 | 2.80 | 3.89 | **0.07** | MLX |
| 32 768 × 768 | 1.95 | 1.87 | 1.79 | 1.70 | 1.83 | 2.07 | 3.09 | 2.11 | 2.04 | 2.87 | 3.42 | 5.63 | **0.06** | MLX |
| 65 536 × 768 | 2.63 | 2.34 | 2.20 | 2.01 | 2.35 | 2.33 | 3.11 | 2.59 | 2.81 | 4.10 | 5.19 | — | **0.06** | MLX |
| 131 072 × 768 | 3.43 | 2.17 | 2.30 | 1.99 | 2.02 | 2.99 | 3.68 | 3.03 | 3.91 | 5.45 | 6.93 | — | **0.06** | MLX |
| 2 048 × 1024 | **0.82** | **0.79** | **0.76** | **0.82** | **0.82** | **0.76** | **0.75** | 0.95~ | 1.16~ | 1.15~ | 1.21~ | **0.60** | **0.08** | scikit-learn |
| 4 096 × 1024 | 0.94~ | 1.04~ | 1.01~ | 1.00~ | 1.15 | 1.11~ | 0.97~ | 1.20~ | 1.31 | 1.46 | 1.66 | 1.66 | **0.07** | scikit-learn |
| 8 192 × 1024 | 1.20 | 1.39 | 1.37 | 1.31 | 1.29 | 1.51 | 1.42 | 1.56 | 1.73 | 1.91 | 2.08 | 2.93 | **0.07** | MLX |
| 16 384 × 1024 | 1.69 | 1.70 | 1.79 | 1.57 | 1.35 | 2.16 | 2.73 | 2.13 | 2.08 | 2.43 | 3.40 | 4.37 | **0.06** | MLX |
| 32 768 × 1024 | 2.44 | 2.35 | 2.00 | 2.37 | 2.12 | 2.37 | 3.44 | 2.73 | 2.48 | 3.20 | 4.03 | — | **0.06** | MLX |
| 65 536 × 1024 | 3.36 | 2.49 | 2.66 | 2.55 | 2.89 | 3.24 | 4.33 | 2.80 | 3.52 | 4.63 | 6.37 | — | **0.06** | MLX |
| 131 072 × 1024 | 5.15 | 2.88 | 2.79 | 2.31 | 2.42 | 2.59 | 4.07 | 4.47 | 5.23 | 7.02 | — | — | **0.06** | MLX |

Below 4 096 rows a 4 096-query batch lost at every width (0.44×–0.79× at 2 048 rows): MLX
spends a few microseconds a query row however small the index, and stock brute force over
so small an index costs less. Between 4 096 rows and 6 million elements small batches
lost — 1 to 32 queries, 0.70×–0.95× at 4 096 and 8 192 rows — and at 4 194 304 elements,
where the previous work floor sat, single rows only tied (0.90×–0.99×). From 6 million
elements every batch won or tied on both versions.

Where patched dispatch uses MLX, then — tall data from 64 features, wide data from 256,
on data the fit-time probe passes — the grids above and below (`knn`, `knn-rows`,
`knn-narrow`, `knn-data`) answered single rows 1.0×–5.7× faster than stock, 16 queries
0.94×–3.2× (a tie at worst), 256 queries 1.2×–7.7× and 4 096 queries 2.6×–6.2×, on both
versions, and every batch size between those won or tied too.

Queries win; the index build does not. Stock brute force stores a reference, while
mlxlearn uploads the training rows and builds the translated, scaled copy the search runs
on, with its row norms and, for float64 input, a float32 residual: 0.05×–0.09× of the
stock fit where patched dispatch uses MLX (11–256 ms against 1–16 ms, the fit columns
above). It is repaid after about 180–950 queries in batches of 256 or more, which is why
`knn/fit` is listed in `benchmarks/gate.py::ACCEPTED_REGRESSIONS` rather than left to slip
through the "no crossover measured, so not gated" branch. A model fitted to answer a
handful of queries is faster on stock scikit-learn.

In memory, a float64 index holds 3.0× the float32 data on the device — rows, residual and
the translated copy — and 3.5–4.2× at the peak of the build, measured at 50 000 × 32 and
128 000 × 768.

### Data the size rules cannot see

The tables above are Gaussian data, on which the search's first float32 pass certifies
every query row ([`fp32_policy.md`](fp32_policy.md) §2.1). A row it cannot certify is
searched again, with a much wider candidate set and then exactly, and that costs more than
the certified search saves. Grid `knn-data` times the same batches on other data at shapes
the rules above send to MLX, k = 10, both versions. `probe` is what a patched fit finds
when it searches 64 of its own rows with the first pass alone: how many it could not
certify, or `integer` for data on the exact integer path, which has nothing to certify:

<!-- python -m benchmarks.calibrate table knn-data: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | 1 query | 16 queries | 64 queries | 256 queries | 1024 queries | probe | dispatch |
|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, gauss | 1.48 / 1.96 | 1.52~ / 0.80~ | 1.19~ / 1.18~ | 0.91~ / 0.89~ | 0.91~ / **0.68** | 0/64 | scikit-learn |
| 1 000 000 × 16, blobs | 1.53 / 1.98 | **0.71** / **0.76** | 1.09~ / 1.14~ | 0.94~ / 0.94~ | **0.84** / **0.80** | 0/64 | scikit-learn |
| 1 000 000 × 16, uniform | 1.86 / 1.99 | 0.72~ / 1.30~ | 1.26~ / 1.06~ | 0.93~ / 0.92~ | **0.84** / **0.76** | 0/64 | scikit-learn |
| 1 000 000 × 16, unscaled | **0.03** / **0.03** | **0.10** / **0.09** | **0.09** / **0.09** | **0.07** / **0.07** | **0.07** / **0.07** | 61/64 | scikit-learn |
| 1 000 000 × 16, dups | **0.82** / 1.10~ | 0.76~ / 0.77~ | **0.55** / **0.53** | **0.40** / **0.38** | **0.41** / **0.25** | 61/64 | scikit-learn |
| 1 000 000 × 16, tight-blobs | 1.41 / 2.00 | 0.62~ / 1.08~ | 0.89~ / 0.98~ | **0.76** / 0.97~ | **0.88** / **0.59** | 4/64 | scikit-learn |
| 1 000 000 × 16, binary | 1.47 / 1.41 | 0.77~ / 1.21~ | 0.79~ / 0.83~ | **0.64** / **0.74** | **0.71** / **0.48** | integer | scikit-learn |
| 1 000 000 × 16, counts | 1.48 / 1.54 | 0.78~ / 1.22 | 0.86~ / 0.81~ | **0.70** / **0.73** | **0.62** / **0.50** | integer | scikit-learn |
| 1 000 000 × 32, gauss | 1.78 / 2.43 | 0.89~ / 0.90~ | 1.23~ / 1.00~ | 1.03~ / 1.10~ | 0.94~ / **0.81** | 0/64 | scikit-learn |
| 1 000 000 × 32, blobs | 1.85 / 2.37 | 0.81~ / 0.83~ | 1.44~ / 1.28~ | 1.23 / 1.09~ | 0.85~ / 0.87~ | 0/64 | scikit-learn |
| 1 000 000 × 32, uniform | 1.91 / 2.36 | 0.91~ / 1.35~ | 1.34~ / 0.96~ | 1.20 / 1.11~ | 0.80~ / **0.80** | 0/64 | scikit-learn |
| 1 000 000 × 32, unscaled | **0.04** / **0.04** | **0.10** / **0.10** | **0.10** / **0.09** | **0.08** / **0.07** | **0.08** / **0.07** | 60/64 | scikit-learn |
| 1 000 000 × 32, dups | 0.99~ / 1.34 | 0.81~ / 0.81~ | 0.67~ / 0.64~ | **0.46** / **0.42** | **0.35** / **0.28** | 63/64 | scikit-learn |
| 1 000 000 × 32, tight-blobs | 1.79 / 2.32 | 1.03~ / 0.75~ | 0.96~ / 0.81~ | **0.86** / **0.80** | **0.43** / **0.46** | 13/64 | scikit-learn |
| 1 000 000 × 32, binary | 1.73 / 1.76 | 1.43~ / 1.38~ | 1.07~ / 1.03~ | 0.91~ / 0.96~ | **0.57** / **0.60** | integer | scikit-learn |
| 1 000 000 × 32, counts | 1.71 / 1.84 | 1.38~ / 1.31~ | 1.03~ / 0.99~ | 1.01~ / **0.89** | **0.55** / **0.58** | integer | scikit-learn |
| 24 000 × 256, gauss | 1.01~ / 1.04~ | 1.10 / 1.09 | 1.14 / 1.14 | 1.19~ / 1.63 | 2.47 / 2.61 | 0/64 | MLX |
| 24 000 × 256, blobs | 1.05~ / 1.04~ | 1.11~ / 1.08 | 1.15 / 1.20 | 1.39 / 1.61 | 2.45 / 2.54 | 0/64 | MLX |
| 24 000 × 256, uniform | 1.03~ / 1.03~ | 1.09 / 1.02~ | 1.17 / 1.14 | 1.49 / 1.64 | 2.63 / 2.64 | 0/64 | MLX |
| 24 000 × 256, unscaled | **0.57** / **0.59** | **0.55** / **0.54** | **0.49** / **0.43** | **0.52** / **0.57** | 0.82~ / **0.71** | 60/64 | scikit-learn |
| 24 000 × 256, dups | **0.62** / **0.59** | **0.52** / **0.54** | **0.48** / **0.49** | 0.79~ / **0.58** | 0.99~ / 0.99~ | 61/64 | scikit-learn |
| 24 000 × 256, tight-blobs | **0.59** / **0.67** | **0.54** / **0.52** | **0.46** / **0.43** | 0.67~ / 0.56~ | 0.80~ / 0.85~ | 64/64 | scikit-learn |
| 24 000 × 256, binary | 1.30 / 1.19 | 1.15 / 1.15 | 0.97~ / 1.18 | 1.52 / 1.51 | 2.63 / 2.58 | integer | MLX |
| 24 000 × 256, counts | 1.09~ / 1.15 | 1.10~ / 1.10 | 1.15 / 1.28 | 1.49 / 1.65 | 2.51 / 2.70 | integer | MLX |
| 65 536 × 256, gauss | 1.68 / 1.52 | 1.20 / 1.32 | 1.77 / 2.00 | 2.08 / 2.30 | 4.60 / 4.68 | 0/64 | MLX |
| 65 536 × 256, blobs | 1.52 / 1.54 | 1.28 / 1.32 | 1.70 / 2.04 | 2.00 / 1.88 | 4.34 / 4.09 | 0/64 | MLX |
| 65 536 × 256, uniform | 1.65 / 1.73 | 1.31 / 1.46 | 1.86 / 1.77 | 2.08 / 2.39 | 3.91 / 4.85 | 0/64 | MLX |
| 65 536 × 256, unscaled | **0.24** / **0.25** | **0.16** / **0.17** | **0.24** / **0.32** | **0.17** / **0.29** | **0.35** / **0.26** | 63/64 | scikit-learn |
| 65 536 × 256, dups | 0.97~ / **0.87** | **0.70** / **0.71** | 0.89~ / **0.80** | 1.01~ / 1.57 | 1.73 / 1.78 | 62/64 | scikit-learn |
| 65 536 × 256, tight-blobs | 0.99~ / **0.87** | **0.70** / **0.59** | **0.75** / **0.80** | 1.06~ / 1.12~ | 1.54 / 1.30 | 64/64 | scikit-learn |
| 65 536 × 256, binary | 1.71 / 1.71 | 1.39 / 1.58 | 1.62 / 1.69 | 1.73~ / 1.91 | 3.48 / 3.83 | integer | MLX |
| 65 536 × 256, counts | 1.66 / 2.46 | 1.23 / 1.46 | 1.81 / 1.75 | 1.96 / 2.11 | 3.51 / 3.63 | integer | MLX |
| 8 192 × 1024, gauss | 1.24 / 1.27 | 1.24 / 1.31 | 1.61 / 1.57 | 1.60 / 1.69 | 2.08 / 2.19 | 0/64 | MLX |
| 8 192 × 1024, blobs | 1.20 / 1.19 | 1.32 / 1.22 | 1.04~ / 1.02~ | 1.42 / 1.57 | 2.04 / 1.92 | 3/64 | MLX |
| 8 192 × 1024, uniform | 1.18 / 1.21 | 1.38 / 1.32 | 1.47 / 1.76 | 1.56 / 1.83 | 2.18 / 2.16 | 0/64 | MLX |
| 8 192 × 1024, unscaled | **0.75** / **0.66** | **0.70** / **0.60** | **0.66** / **0.72** | **0.44** / **0.60** | **0.63** / **0.60** | 59/64 | scikit-learn |
| 8 192 × 1024, dups | **0.72** / **0.74** | **0.71** / **0.72** | **0.72** / **0.65** | **0.49** / **0.54** | **0.72** / **0.71** | 62/64 | scikit-learn |
| 8 192 × 1024, tight-blobs | **0.74** / **0.71** | **0.70** / **0.62** | **0.64** / **0.69** | **0.56** / **0.47** | **0.59** / **0.61** | 64/64 | scikit-learn |
| 8 192 × 1024, binary | 1.28 / 1.34 | 1.20 / 1.19 | 1.52~ / 1.36~ | 1.53 / 1.71 | 1.84 / 2.18 | integer | MLX |
| 8 192 × 1024, counts | 1.32 / 1.59 | 1.23 / 1.36 | 1.53 / 1.42 | 1.91 / 1.50 | 2.11 / 2.34 | integer | MLX |

`gauss`, `blobs` (100 clusters of std 1 around centres of std 3) and `uniform` are data the
first pass certifies. `unscaled` is one column uniform on 0–10 000 beside standardized
ones; `dups` repeats every row about 20 times; `tight-blobs` is 100 clusters of std 0.1;
`binary` is 0/1 features at 20%, `counts` Poisson(2) counts.

- **The probe separates them.** Every shape of the three data the first pass cannot
  certify lost on some batch, on both versions, down to 0.03× (a single row at
  1 000 000 × 16, unscaled), and their probes found 59–64 of 64 rows uncertified — but
  only 4 and 13 of 64 for tight clusters at a million rows and 16 or 32 features, where
  each cluster holds 10 000 rows. The width rule (below) now hands those shapes to
  scikit-learn anyway, and from 64 features the same three laws left 50–64 of 64
  uncertified (probed untimed at 1 000 000 and 1 600 000 × 64, 1 000 000 × 128, and
  262 144 and 400 000 × 256). Certified data left at most 3 of 64 uncertified. So
  patched mode probes every fit the size rules would send to MLX and hands it to
  scikit-learn when more than an eighth of the rows probed could not be certified
  (`knn_probe_max_uncertified = 0.125`), recorded as `"data-probe"` with the count on the
  event.
- **Integer data** is searched on the exact integer path, which kept up with stock brute
  force from 256 features (0.97×~ at worst) but, at a million rows and 16 or 32 features,
  lost batches of 256 and 1 024 queries (0.48×–0.88×). Below `knn_wide_features` it goes
  to scikit-learn too, with the same reason — conservatively at 64–255 features, which
  this grid did not measure on MLX.

The probe costs the fit one first pass for 64 queries — on certified data, the 64-query
column's MLX time, 2–15 ms here — on top of the index build (the `knn/fit` accepted
regression), and a fit
it hands over has paid for both before scikit-learn's own. A directly imported estimator
does not probe: it searches on MLX, certified, whatever the data.

### Narrow data goes to scikit-learn, whatever its size

With at most 15 features, scikit-learn's default `algorithm="auto"` builds a KD-tree.
Grid `knn-tree`, against the tree on Gaussian data — the fit column is the tree's build time over MLX's
index build, and the single-row column times a different row in every pair, since what
one row costs a tree depends on where it lands:

<!-- python -m benchmarks.calibrate table knn-tree: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|
| 4 096 × 2 | **0.27** / **0.19** | **0.27** / **0.22** | **0.32** / **0.28** | **0.31** / 0.52~ | **0.48** / **0.48** | scikit-learn |
| 65 536 × 2 | **0.26** / **0.21** | **0.22** / **0.18** | 0.14~ / 0.14~ | **0.14** / **0.14** | 3.31 / 3.43 | scikit-learn |
| 1 048 576 × 3 | **0.18** / **0.13** | **0.05** / **0.07** | **0.04** / **0.04** | **0.03** / **0.03** | 14.66 / 13.62 | scikit-learn |
| 65 536 × 4 | **0.27** / **0.21** | **0.28** / **0.23** | 0.42~ / 0.31~ | **0.49** / **0.50** | 3.92 / 3.81 | scikit-learn |
| 1 048 576 × 5 | **0.18** / **0.18** | **0.17** / **0.15** | **0.17** / **0.17** | **0.15** / **0.13** | 17.82 / 14.55 | scikit-learn |
| 65 536 × 6 | **0.32** / **0.24** | **0.53** / **0.52** | 1.00~ / 0.73~ | 1.98 / 1.97 | 6.16 / 5.31 | scikit-learn |
| 262 144 × 6 | **0.32** / **0.22** | **0.36** / **0.32** | 0.97~ / **0.83** | **0.74** / **0.64** | 11.05 / 9.26 | scikit-learn |
| 4 096 × 8 | **0.33** / **0.24** | **0.45** / **0.40** | 2.15 / 2.11 | 4.40 / 4.49 | **0.65** / **0.61** | scikit-learn |
| 65 536 × 8 | **0.37** / **0.29** | **0.85** / **0.79** | 1.48 / 1.45~ | 4.42 / 4.24 | 5.85 / 5.18 | scikit-learn |
| 262 144 × 8 | **0.43** / **0.29** | 0.98~ / **0.87** | 2.34 / 2.23 | 2.41 / 2.15 | 11.87 / 9.22 | scikit-learn |
| 1 048 576 × 8 | **0.27** / **0.23** | **0.41** / **0.74** | 1.32 / 0.98~ | 1.29 / 1.17 | 19.15 / 15.18 | scikit-learn |
| 4 096 × 12 | **0.33** / **0.24** | **0.73** / **0.63** | 4.74 / 4.40 | 10.39 / 10.54 | **0.68** / **0.64** | scikit-learn |
| 65 536 × 12 | **0.67** / **0.61** | 2.93 / 2.65 | 7.51 / 7.19 | 24.65 / 22.78 | 5.16 / 4.60 | scikit-learn |
| 262 144 × 12 | 0.88~ / 0.89~ | 5.08 / 4.59 | 12.73 / 11.94 | 20.97 / 21.89 | 9.43 / 7.91 | scikit-learn |
| 1 048 576 × 12 | 1.12~ / 1.13~ | 3.22 / 4.38 | 11.77 / 12.44 | 14.89 / 14.10 | 17.06 / 14.57 | scikit-learn |
| 16 384 × 14 | **0.46** / **0.44** | 1.99 / 2.10 | 12.52 / 11.62 | 28.70 / 25.48 | 2.30 / 1.87 | scikit-learn |
| 262 144 × 14 | 1.47~ / 1.51~ | 9.47 / 7.79 | 26.29 / 21.06 | 47.59 / 38.76 | 10.58 / 7.81 | scikit-learn |
| 65 536 × 15 | 0.97~ / 1.02~ | 5.18 / 5.44 | 15.34 / 14.91 | 47.23 / 43.45 | 5.35 / 4.69 | scikit-learn |

Grid `knn-tree-lowrank`: the same, on data of low intrinsic dimension — three latent
dimensions embedded in 8 or 12 features, plus 5% noise:

<!-- python -m benchmarks.calibrate table knn-tree-lowrank: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|
| 65 536 × 8 | **0.28** / **0.23** | **0.26** / **0.24** | **0.36** / **0.42** | **0.49** / **0.47** | 5.29 / 4.70 | scikit-learn |
| 262 144 × 8 | **0.19** / **0.19** | **0.13** / **0.12** | **0.21** / **0.22** | **0.15** / **0.15** | 10.17 / 9.96 | scikit-learn |
| 65 536 × 12 | **0.25** / **0.23** | **0.28** / **0.26** | **0.50** / **0.48** | **0.67** / **0.68** | 4.93 / 4.53 | scikit-learn |
| 262 144 × 12 | **0.19** / **0.18** | **0.16** / **0.15** | **0.32** / **0.32** | **0.25** / **0.25** | 8.96 / 7.34 | scikit-learn |

A single row is where the tree wins: faster than an MLX call at every size measured
up to 8 features (0.13×–0.43× for MLX, both versions), and at 12 features up to
65 536 rows (0.24×–0.67×); at 2–5 features it won every batch. Single rows tie at
262 144 × 12 and from there up, and cost a tree very differently from row to row — the
`~` there is that spread, not a small sample. Gaussian data is where a KD-tree does
worst. On data of low intrinsic dimension, the other extreme, the tree won every batch.

Where scikit-learn would not build a tree — `algorithm="brute"`, a metric the tree does
not take, `n_neighbors` of half the rows or more — its brute force answers small batches
on narrow data faster too (grid `knn-narrow-brute`):

<!-- python -m benchmarks.calibrate table knn-narrow-brute: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | dispatch |
|---|---|---|---|---|---|
| 8 192 × 2 | **0.79** / 0.83~ | **0.74** / **0.79** | **0.76** / 0.92~ | **0.79** / 0.91~ | scikit-learn |
| 131 072 × 2 | **0.67** / 0.90~ | **0.66** / **0.66** | 1.01~ / 1.16 | 1.37 / 1.53 | scikit-learn |
| 524 288 × 2 | 0.85~ / 1.15~ | **0.59** / **0.72** | 1.27 / 1.41 | 1.32 / 0.85~ | scikit-learn |
| 2 097 152 × 2 | **0.90** / 1.11~ | **0.90** / 1.03~ | **0.75** / **0.39** | **0.81** / **0.67** | scikit-learn |
| 8 192 × 8 | **0.74** / 1.00~ | 0.83~ / **0.88** | 0.97~ / 0.90~ | 0.93~ / 0.84~ | scikit-learn |
| 131 072 × 8 | **0.89** / 0.87~ | **0.70** / **0.71** | 1.14~ / 0.77~ | 1.57 / 1.73 | scikit-learn |
| 524 288 × 8 | 1.21 / 1.58 | **0.61** / 0.90~ | 1.46~ / 1.65 | 1.48 / 1.82 | scikit-learn |
| 2 097 152 × 8 | 1.98 / 2.62 | 1.18~ / 1.61 | 1.35 / 1.56 | 0.99~ / 0.95~ | scikit-learn |

Patched dispatch therefore hands every fit with at most 15 features to scikit-learn: to
its KD-tree where it would build one (reason `"tree-wins"`), and to its brute force
otherwise — as it does every fit with fewer than 64 features (`knn-narrow`, above).

What that gives up is real. On Gaussian data MLX answered 256-query batches
1.3×–26× faster than the tree from 8 features up (one tie, 1 048 576 × 8 on 1.7.2),
16-query batches 2.6×–9.5× faster from 12 features and 65 536 rows, and built its
index 3×–19× faster from 65 536 rows. A workload known to query in batches can
import `mlxlearn.neighbors` directly, which always searches on MLX.

### The published neighbor cases

The `published` profile's neighbor cases, reproducible with the command at the top: the
Layer 1 classes forced onto the MLX device against stock brute force, k = 10, 25
interleaved pairs, scikit-learn 1.9.0. `dispatch` is where patched dispatch sends the
fit; cases it hands to scikit-learn are on the record as evidence, not claims.

| samples × features | queries | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|---|
| 1 000 × 32 | 256 | 1.32 | 1.35 | 1.03×~ | 1.02× | scikit-learn |
| 4 096 × 64 | 256 | 1.78 | 1.91 | 1.08×~ | 1.09× | scikit-learn |
| 32 768 × 16 | 256 | 4.81 | 3.34 | **0.70×** | 1.28× | scikit-learn |
| 50 000 × 32 | 256 | 6.19 | 4.60 | 0.74×~ | 1.08× | scikit-learn |
| 8 192 × 768 | 256 | 4.08 | 6.67 | 1.64× | 1.36× | MLX |
| 131 072 × 128 | 256 | 8.08 | 20.63 | 2.55× | 2.41× | scikit-learn |
| 128 000 × 768 | 256 | 12.01 | 63.74 | 5.31× | 6.70× | MLX |
| 1 000 000 × 16 | 256 | 42.69 | 45.13 | 1.06× | 1.02× | scikit-learn |
| 1 000 000 × 64 | 256 | 52.30 | 73.38 | 1.40× | 1.49× | MLX |
| 1 000 × 32 | 1 | 1.02 | 0.92 | **0.90×** | 1.22× | scikit-learn |
| 1 000 × 32 | 64 | 1.10 | 1.10 | 1.00×~ | 1.02× | scikit-learn |
| 1 000 × 32 | 4096 | 6.39 | 2.42 | **0.38×** | 0.34× | scikit-learn |
| 8 192 × 768 | 1 | 0.92 | 1.45 | 1.58× | 1.45× | MLX |
| 8 192 × 768 | 64 | 1.93 | 3.17 | 1.64× | 1.45× | MLX |
| 8 192 × 768 | 4096 | 37.07 | 94.98 | 2.56× | 2.98× | MLX |
| 131 072 × 128 | 1 | 1.42 | 2.50 | 1.76× | 1.84× | scikit-learn |
| 131 072 × 128 | 64 | 5.70 | 5.95 | 1.05×~ | 1.00× | scikit-learn |
| 131 072 × 128 | 4096 | 85.23 | 342.36 | 4.02× | 4.14× | scikit-learn |
| 1 000 000 × 16 | 1 | 1.90 | 3.08 | 1.62× | 1.76× | scikit-learn |
| 1 000 000 × 16 | 64 | 11.36 | 14.82 | 1.30× | 1.24× | scikit-learn |
| 1 000 000 × 16 | 4096 | 702.62 | 829.28 | 1.18× | 1.14× | scikit-learn |
| 1 000 000 × 64 | 1 | 2.26 | 5.17 | 2.29× | 2.37× | MLX |
| 1 000 000 × 64 | 64 | 14.63 | 25.43 | 1.74× | 1.83× | MLX |
| 1 000 000 × 64 | 4096 | 732.02 | 1346.33 | 1.84× | 1.61× | MLX |

| samples × features | mlxlearn fit (ms) | scikit-learn fit (ms) | ratio | dispatch |
|---|---|---|---|---|
| 1 000 × 32 | 1.04 | 0.20 | **0.20×** | scikit-learn |
| 4 096 × 64 | 1.65 | 0.25 | **0.15×** | scikit-learn |
| 32 768 × 16 | 2.30 | 0.28 | **0.12×** | scikit-learn |
| 50 000 × 32 | 4.14 | 0.42 | **0.10×** | scikit-learn |
| 8 192 × 768 | 10.44 | 0.99 | **0.10×** | MLX |
| 131 072 × 128 | 26.45 | 2.25 | **0.09×** | scikit-learn |
| 128 000 × 768 | 136.91 | 12.23 | **0.09×** | MLX |
| 1 000 000 × 16 | 33.07 | 2.14 | **0.06×** | scikit-learn |
| 1 000 000 × 64 | 104.11 | 7.69 | **0.07×** | MLX |

131 072 × 128, which the previous floors sent to MLX, won single rows and large batches
here, but 64 queries only tied (1.05×~), and at 120 000 × 128 in the `knn` table 64
queries lost (0.80×): patched dispatch now hands it to scikit-learn. So it does
1 000 000 × 16, which won every batch timed here (1.06×–1.62×) but lost batches of 16
and 1 024 queries in the `knn-narrow` table; at the same rows, 1 000 000 × 64 is the
narrowest shape it sends to MLX (1.40×–2.29× here).

## SVC — width decides

Grid `svc`: binary fits on the benchmark data (`y = x₀ + noise > 0`), against LIBSVM.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2 (— where a configuration
was not measured), median of 5 interleaved pairs; bold and `~` as above. C = 1 and 10
everywhere, 0.1 and 100 around the floors:

<!-- python -m benchmarks.calibrate table svc: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | rbf, C = 0.1 | rbf, C = 1 | rbf, C = 10 | rbf, C = 100 | poly, C = 0.1 | poly, C = 1 | poly, C = 10 | poly, C = 100 | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 4 000 × 16 | — | 1.12 / 1.00~ | **0.39** / **0.35** | — | — | **0.63** / **0.58** | **0.16** / **0.15** | — | scikit-learn |
| 4 000 × 32 | — | 1.23 / 1.11 | **0.53** / **0.49** | — | — | **0.86** / **0.79** | **0.55** / **0.50** | — | scikit-learn |
| 8 000 × 32 | — | 1.23 / 1.16 | **0.52** / **0.50** | — | — | **0.83** / **0.78** | **0.39** / **0.37** | — | scikit-learn |
| 4 000 × 64 | 2.64 / 2.39 | 1.57 / 1.45 | **0.80** / **0.73** | **0.79** / **0.74** | 2.22 / 1.98 | 1.17 / 1.06 | **0.83** / **0.77** | **0.85** / **0.77** | scikit-learn |
| 1 000 × 128 | 2.18 / 1.96 | 1.09 / 1.00~ | **0.74** / **0.67** | **0.75** / **0.68** | 1.70 / 1.54 | **0.90** / **0.81** | **0.68** / **0.61** | **0.68** / **0.61** | scikit-learn |
| 1 500 × 128 | — | 1.38 / 1.29 | **0.90** / **0.83** | — | — | 1.06 / 1.00~ | **0.87** / **0.79** | — | scikit-learn |
| 2 000 × 128 | 3.09 / 2.78 | 1.65 / 1.51 | 1.04 / 0.95 | 1.05~ / **0.95** | 2.89 / 2.62 | 1.29 / 1.17 | 1.01~ / **0.92** | 1.01 / **0.90** | scikit-learn |
| 2 500 × 128 | 3.72 / 3.35 | 1.81 / 1.66 | 1.13 / 1.02~ | 1.13 / 1.04 | 2.85 / 2.51 | 1.41 / 1.27 | 1.11 / 1.03 | 1.08 / 1.02 | scikit-learn |
| 3 000 × 128 | 3.96 / 3.55 | 2.00 / 1.90 | 1.25 / 1.13 | 1.23 / 1.11 | 3.06 / 2.78 | 1.53 / 1.36 | 1.12 / 1.04~ | 1.14 / 1.04 | MLX |
| 4 000 × 128 | 4.06 / 3.66 | 2.27 / 2.09 | 1.36 / 1.26 | 1.38 / 1.26 | 3.98 / 3.59 | 1.69 / 1.55 | 1.28 / 1.19 | 1.26 / 1.19 | MLX |
| 8 000 × 128 | — | 2.55 / 2.42 | 1.54 / 1.43 | — | — | 1.83 / 1.70 | 1.54 / 1.46 | — | MLX |
| 1 000 × 256 | 3.64 / 3.24 | 1.88 / 1.74 | 1.20 / 1.09 | 1.20 / 1.08 | 2.44 / 2.18 | 1.58 / 1.40 | 1.15 / 1.04 | 1.16 / 1.04 | scikit-learn |
| 1 500 × 256 | 4.98 / 4.55 | 2.41 / 2.21 | 1.50 / 1.41 | 1.53 / 1.42 | 3.78 / 3.42 | 1.85 / 1.67 | 1.53 / 1.36 | 1.53 / 1.37 | MLX |
| 2 000 × 256 | 5.38 / 4.88 | 2.73 / 2.52 | 1.77 / 1.63 | 1.77 / 1.59 | 3.99 / 3.56 | 2.25 / 2.01 | 1.73 / 1.57 | 1.73 / 1.56 | MLX |
| 4 000 × 256 | 7.09 / 6.48 | 3.68 / 3.43 | 2.44 / 2.23 | 2.43 / 2.27 | 6.32 / 5.80 | 2.82 / 2.58 | 2.42 / 2.24 | 2.42 / 2.21 | MLX |
| 8 000 × 256 | — | 4.38 / 4.05 | 3.16 / 2.83 | — | — | 3.28 / 3.00 | 3.83 / 3.52 | — | MLX |
| 1 000 × 512 | 5.42 / 4.98 | 3.08 / 2.79 | 1.91 / 1.74 | 1.91 / 1.74 | 4.39 / 3.91 | 2.66 / 2.38 | 2.12 / 1.91 | 2.13 / 1.91 | scikit-learn |
| 2 000 × 512 | 8.38 / 7.49 | 4.22 / 3.84 | 2.73 / 2.49 | 2.70 / 2.50 | 7.02 / 6.15 | 3.75 / 3.40 | 3.00 / 2.68 | 3.00 / 2.65 | MLX |

SMO is sequential, and its iteration counts match LIBSVM's to within a few percent, so the
comparison is per iteration. At 4 000 rows and C = 10, mlxlearn spent 27–41 µs an
iteration from 16 to 256 features — nearly flat, because up to the kernel cache's
capacity (about 7 240 rows per one-vs-one pair at `cache_size=200`) a pair's whole Gram is
one MLX matmul and the loop then runs over it in NumPy — while LIBSVM spent 11 µs at 16
features and 103 µs at 256, because every kernel row it misses costs `O(n·d)`; once its
cache holds the rows an iteration revisits, a few µs. At the default `gamma` and `degree`
they cross near 100 features: below 128 features rbf and poly lost at C = 10 and 100
everywhere (0.15×–0.85×), and poly at C = 1 up to 32 features; from 128 features they
won at every C — once the fit had rows. At 128 features 1 000–2 000 rows still lost at
C = 10 and above on one version or both (0.61×–0.95×; 2 000 × 128 on 1.7.2 only), and
from 3 000 rows every kernel and C won or tied on both.

The size rules take the width, rows and work at which both kernels win at every C: 128
features, 1 024 rows and 384 000 elements — the corner at 3 000 × 128, which also keeps
1 000 × 256 (a 1.04× win at worst) and lets 1 500 × 256 and 1 024 × 512 through. Above all
three floors every fit measured at the default kernel parameters won or tied on both
versions, 1.04×–8.4× (1.04×–4.4× at C of 1 and 10, up to 8 000 rows).

**The kernel parameters move the crossing.** A kernel close to the linear one takes many
iterations per row, over rows LIBSVM has already cached — the linear kernel's own case —
and at those the iteration count, not the width, decides. Grid `svc-params`: the same
binary fits at shapes the floors send to MLX, rbf from 0.03 to 10 times scikit-learn's
`"scale"` value of `gamma` (`1 / (d · Var X)`; the default is 1), poly at degree 2 to 4 and
`coef0` 0 and 1, C = 1 to 100:

<!-- python -m benchmarks.calibrate table svc-params: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, kernel | C = 1 | C = 10 | C = 100 | dispatch |
|---|---|---|---|---|
| 3 000 × 128, rbf, gamma = 0.03 × scale | 3.92 / 3.53 | 1.95 / 1.75 | **0.61** / **0.54** | MLX at C = 1 |
| 3 000 × 128, rbf, gamma = 0.1 × scale | 3.27 / 2.93 | 1.14 / 1.02 | **0.33** / **0.30** | MLX at C = 1 |
| 3 000 × 128, rbf, gamma = 0.3 × scale | 2.63 / 2.37 | **0.67** / **0.61** | **0.61** / **0.56** | MLX at C = 1 |
| 3 000 × 128, rbf, gamma = 0.5 × scale | 2.37 / 2.13 | **0.85** / **0.77** | **0.84** / **0.76** | MLX at C = 1 |
| 3 000 × 128, rbf, gamma = 0.8 × scale | 2.13 / 1.93 | 1.10 / 1.01 | 1.10 / 1.01 | MLX |
| 3 000 × 128, rbf, gamma = 3 × scale | 1.70 / 1.53 | 1.65 / 1.48 | 1.65 / 1.47 | MLX |
| 3 000 × 128, rbf, gamma = 10 × scale | 1.73 / 1.56 | 1.52 / 1.37 | 1.52 / 1.37 | MLX |
| 3 000 × 128, poly, degree 2, coef0 = 0 | 1.84 / 1.66 | **0.63** / **0.57** | **0.63** / **0.57** | scikit-learn |
| 3 000 × 128, poly, degree 2, coef0 = 1 | **0.76** / **0.70** | **0.58** / **0.53** | **0.58** / **0.53** | scikit-learn |
| 3 000 × 128, poly, degree 3, coef0 = 1 | **0.89** / **0.81** | **0.89** / **0.81** | **0.89** / **0.81** | scikit-learn |
| 3 000 × 128, poly, degree 4, coef0 = 0 | 1.83 / 1.66 | 1.16 / 1.04 | 1.16 / 1.04 | MLX |
| 8 000 × 128, rbf, gamma = 0.03 × scale | 3.31 / 3.10 | 2.47 / 2.32 | 1.07 / 0.99 | MLX at C = 1 |
| 8 000 × 128, rbf, gamma = 0.1 × scale | 2.96 / 2.80 | 1.78 / 1.67 | **0.50** / **0.46** | MLX at C = 1 |
| 8 000 × 128, rbf, gamma = 0.3 × scale | 2.78 / 2.64 | 1.04 / 0.97 | **0.86** / **0.81** | MLX at C = 1 |
| 8 000 × 128, rbf, gamma = 0.5 × scale | 2.86 / 2.71 | 1.06 / 0.99 | 0.98 / **0.92** | MLX at C = 1 |
| 8 000 × 128, rbf, gamma = 0.8 × scale | 2.67 / 2.53 | 1.33 / 1.26 | 1.33 / 1.26 | MLX |
| 8 000 × 128, rbf, gamma = 3 × scale | 2.42 / 2.29 | 2.79 / 2.63 | 2.79 / 2.64 | MLX |
| 8 000 × 128, rbf, gamma = 10 × scale | 2.35 / 2.22 | 2.12 / 2.00 | 2.12 / 2.00 | MLX |
| 8 000 × 128, poly, degree 2, coef0 = 0 | 2.07 / 1.94 | **0.54** / **0.50** | **0.46** / **0.43** | scikit-learn |
| 8 000 × 128, poly, degree 2, coef0 = 1 | 1.13 / 1.04 | **0.51** / **0.48** | **0.51** / **0.47** | scikit-learn |
| 8 000 × 128, poly, degree 3, coef0 = 1 | 1.01 / 0.96 | 1.00~ / **0.94** | 1.00~ / **0.94** | scikit-learn |
| 8 000 × 128, poly, degree 4, coef0 = 0 | 1.65 / 1.56 | 1.07 / 1.03 | 1.05 / 1.01 | MLX |
| 1 500 × 256, rbf, gamma = 0.03 × scale | 4.91 / 4.50 | 2.07 / 1.88 | **0.41** / **0.37** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.1 × scale | 4.26 / 3.85 | **0.95** / **0.84** | **0.42** / **0.38** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.3 × scale | 2.97 / 2.70 | **0.88** / **0.81** | **0.81** / **0.74** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.5 × scale | 2.65 / 2.40 | 1.12 / 0.99~ | 1.11 / 1.00~ | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.8 × scale | 2.45 / 2.21 | 1.43 / 1.27 | 1.42 / 1.28 | MLX |
| 1 500 × 256, rbf, gamma = 3 × scale | 2.14 / 1.92 | 1.95 / 1.75 | 1.94 / 1.73 | MLX |
| 1 500 × 256, rbf, gamma = 10 × scale | 1.55 / 1.40 | 1.41 / 1.24 | 1.38 / 1.24 | MLX |
| 1 500 × 256, poly, degree 2, coef0 = 0 | 2.18 / 1.95 | 1.49 / 1.32 | 1.50 / 1.33 | scikit-learn |
| 1 500 × 256, poly, degree 2, coef0 = 1 | **0.94** / **0.85** | **0.83** / **0.74** | **0.83** / **0.74** | scikit-learn |
| 1 500 × 256, poly, degree 3, coef0 = 1 | 1.18 / 1.06 | 1.18 / 1.06 | 1.18 / 1.06 | scikit-learn |
| 1 500 × 256, poly, degree 4, coef0 = 0 | 2.00 / 1.77 | 1.45 / 1.28 | 1.44 / 1.29 | MLX |
| 4 000 × 256, rbf, gamma = 0.03 × scale | 7.26 / 6.54 | 3.79 / 3.44 | 1.07 / 0.97 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.1 × scale | 5.92 / 5.37 | 2.06 / 1.88 | **0.68** / **0.62** | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.3 × scale | 4.83 / 4.39 | 1.25 / 1.15 | 1.23 / 1.13 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.5 × scale | 4.25 / 3.90 | 1.70 / 1.57 | 1.70 / 1.57 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.8 × scale | 3.89 / 3.57 | 2.24 / 2.06 | 2.23 / 2.05 | MLX |
| 4 000 × 256, rbf, gamma = 3 × scale | 3.30 / 3.02 | 3.00 / 2.75 | 3.01 / 2.76 | MLX |
| 4 000 × 256, rbf, gamma = 10 × scale | 2.81 / 2.58 | 2.34 / 2.13 | 2.33 / 2.13 | MLX |
| 4 000 × 256, poly, degree 2, coef0 = 0 | 3.73 / 3.39 | 1.77 / 1.62 | 1.77 / 1.62 | scikit-learn |
| 4 000 × 256, poly, degree 2, coef0 = 1 | 1.42 / 1.30 | 1.24 / 1.14 | 1.24 / 1.14 | scikit-learn |
| 4 000 × 256, poly, degree 3, coef0 = 1 | 1.82 / 1.67 | 1.82 / 1.67 | 1.82 / 1.66 | scikit-learn |
| 4 000 × 256, poly, degree 4, coef0 = 0 | 3.26 / 2.98 | 2.34 / 2.13 | 2.34 / 2.11 | MLX |
| 1 100 × 512, rbf, gamma = 0.03 × scale | 5.58 / 5.04 | 2.91 / 2.70 | **0.45** / **0.41** | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.1 × scale | 5.39 / 5.00 | 1.08 / 1.00~ | **0.58** / **0.52** | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.3 × scale | 4.51 / 4.11 | 1.08 / 0.98 | 1.07 / 0.97 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.5 × scale | 3.75 / 3.40 | 1.41 / 1.27 | 1.40 / 1.26 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.8 × scale | 3.29 / 3.02 | 1.81 / 1.63 | 1.79 / 1.64 | MLX |
| 1 100 × 512, rbf, gamma = 3 × scale | 1.98 / 1.83 | 1.58 / 1.44 | 1.59 / 1.44 | MLX |
| 1 100 × 512, rbf, gamma = 10 × scale | 1.85 / 1.72 | 1.31 / 1.20 | 1.31 / 1.21 | MLX |
| 1 100 × 512, poly, degree 2, coef0 = 0 | 2.28 / 2.07 | 1.94 / 1.74 | 1.93 / 1.74 | scikit-learn |
| 1 100 × 512, poly, degree 2, coef0 = 1 | 1.23 / 1.12 | 1.17 / 1.07 | 1.16 / 1.07 | scikit-learn |
| 1 100 × 512, poly, degree 3, coef0 = 1 | 1.62 / 1.47 | 1.61 / 1.47 | 1.62 / 1.47 | scikit-learn |
| 1 100 × 512, poly, degree 4, coef0 = 0 | 2.40 / 2.17 | 1.98 / 1.78 | 1.98 / 1.78 | MLX |
| 4 000 × 512, rbf, gamma = 0.03 × scale | 11.15 / 10.36 | 5.90 / 5.43 | 1.32 / 1.22 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.1 × scale | 9.75 / 9.10 | 2.85 / 2.64 | 1.25 / 1.15 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.3 × scale | 7.28 / 6.85 | 2.28 / 2.11 | 2.04 / 1.90 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.5 × scale | 6.57 / 6.19 | 2.74 / 2.56 | 2.78 / 2.58 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.8 × scale | 6.12 / 5.72 | 3.55 / 3.31 | 3.57 / 3.31 | MLX |
| 4 000 × 512, rbf, gamma = 3 × scale | 4.45 / 4.17 | 4.61 / 4.27 | 4.57 / 4.27 | MLX |
| 4 000 × 512, rbf, gamma = 10 × scale | 4.19 / 3.90 | 3.16 / 2.92 | 3.14 / 2.93 | MLX |
| 4 000 × 512, poly, degree 2, coef0 = 0 | 6.04 / 5.62 | 4.21 / 3.90 | 4.24 / 3.88 | scikit-learn |
| 4 000 × 512, poly, degree 2, coef0 = 1 | 2.62 / 2.44 | 2.20 / 2.07 | 2.21 / 2.06 | scikit-learn |
| 4 000 × 512, poly, degree 3, coef0 = 1 | 3.17 / 2.96 | 3.17 / 2.94 | 3.18 / 2.95 | scikit-learn |
| 4 000 × 512, poly, degree 4, coef0 = 0 | 5.35 / 4.97 | 4.44 / 4.08 | 4.42 / 4.09 | MLX |

- **rbf, `gamma` below the default: loses at large C.** From 0.03 to 0.5 times `"scale"`
  it lost at C = 10 or 100 at five of the six shapes, down to 0.30× (0.1 × scale,
  3 000 × 128, C = 100: 28 409 SMO iterations against LIBSVM's 28 343, three times as
  long). At 0.8 it won or tied everywhere, and from 1 to 10 it won. At C = 1 every
  `gamma` below the default won at every shape, 2.1×–11×: too few iterations to lose.
- **poly, degree 2 or `coef0` ≠ 0: loses.** Degree 2 lost at C = 10 and 100 at
  128 features (0.43×–0.63×), and `coef0 = 1` at degree 2 or 3 lost at 128 features
  (down to 0.47×) and, at degree 2, at 1 500 × 256 (0.74×–0.94×). Degree 3 and 4 at
  `coef0 = 0` won or tied everywhere; for that kernel `gamma` is the same problem as the
  default with C scaled by `(gamma / scale)^degree`, so the C column covers it.

So patched dispatch also hands an rbf fit to LIBSVM when `gamma` is below 0.8 × `"scale"`
and C is above 1 (`svc_min_gamma_ratio`, `svc_low_gamma_max_c`), and a poly fit whose
degree is below 3 or whose `coef0` is not 0 (`svc_poly_min_degree`), whatever the size —
a grid search over C and `gamma` has its near-linear corner answered by LIBSVM and the
rest by MLX. The rule gives up the wins these kernels took at wide shapes (at 4 000 × 512
every configuration here won). A directly imported class keeps them on MLX.

Narrow data loses at C = 1 too. At 2–8 features rbf lost on 1.7.2 at every size measured,
2 000 to 16 000 rows (0.50×–0.91×), and on 1.9.0 everywhere but a tie at 4 000 rows —
most past the kernel cache, where rows are fetched on demand (grid `svc-narrow`):

<!-- python -m benchmarks.calibrate table svc-narrow: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | rbf, C = 1 | dispatch |
|---|---|---|
| 2 000 × 2 | **0.64** / **0.56** | scikit-learn |
| 4 000 × 2 | 1.00~ / **0.88** | scikit-learn |
| 8 000 × 2 | **0.79** / **0.72** | scikit-learn |
| 16 000 × 2 | **0.54** / **0.50** | scikit-learn |
| 2 000 × 4 | **0.67** / **0.58** | scikit-learn |
| 4 000 × 4 | 1.02 / **0.89** | scikit-learn |
| 8 000 × 4 | **0.88** / **0.78** | scikit-learn |
| 16 000 × 4 | **0.68** / **0.61** | scikit-learn |
| 2 000 × 8 | **0.65** / **0.56** | scikit-learn |
| 4 000 × 8 | 1.02 / **0.89** | scikit-learn |
| 8 000 × 8 | 0.99~ / **0.91** | scikit-learn |
| 16 000 × 8 | **0.84** / **0.77** | scikit-learn |

Wide data past the cache won by more — rbf, C = 1, scikit-learn 1.9.0, three pairs each
(a stock fit at 32 000 × 256 takes nearly two minutes), too few for the significance test
to judge, so the ratio alone is the evidence (grid `svc-large`):

<!-- python -m benchmarks.calibrate table svc-large: scikit-learn 1.9.0 -->
| rows × features | rbf, C = 1 | dispatch |
|---|---|---|
| 16 000 × 256 | 10.52~ | MLX |
| 32 000 × 256 | 20.21~ | MLX |

At 32 000 × 256 the MLX fit took 6.9 s against 139 s, over the same number of SMO
iterations to within 0.1% (22 646 against LIBSVM's 22 667).

**The sigmoid kernel** never takes the MLX path. It won at C = 0.1 everywhere, and lost
at C = 100 from 64 features and at C = 10 from 128 (bar one win, at 8 000 × 256) — down to
0.06× — and at C = 1 at 2 000 × 128 and 1 000 × 256; the rules cannot see C (grid
`svc-sigmoid`):

<!-- python -m benchmarks.calibrate table svc-sigmoid: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | sigmoid, C = 0.1 | sigmoid, C = 1 | sigmoid, C = 10 | sigmoid, C = 100 | dispatch |
|---|---|---|---|---|---|
| 4 000 × 16 | 2.65 / 2.39 | 1.96 / 1.80 | 0.98~ / **0.89** | 1.08 / 0.97 | scikit-learn |
| 4 000 × 64 | 3.68 / 3.29 | 2.62 / 2.34 | 1.08 / 0.98 | **0.58** / **0.52** | scikit-learn |
| 2 000 × 128 | 3.56 / 3.20 | **0.93** / **0.84** | **0.53** / **0.48** | **0.12** / **0.11** | scikit-learn |
| 4 000 × 128 | 4.44 / 3.96 | 1.41 / 1.27 | **0.60** / **0.55** | **0.43** / **0.39** | scikit-learn |
| 1 000 × 256 | 3.63 / 3.25 | **0.44** / **0.40** | **0.17** / **0.15** | **0.07** / **0.06** | scikit-learn |
| 4 000 × 256 | 6.82 / 6.14 | 1.36 / 1.24 | **0.49** / **0.45** | **0.37** / **0.33** | scikit-learn |
| 8 000 × 256 | 5.51 / 5.12 | 2.91 / 2.71 | 1.29 / 1.19 | **0.61** / **0.57** | scikit-learn |
| 2 000 × 512 | 7.91 / 7.12 | 1.68 / 1.54 | **0.28** / **0.25** | **0.17** / **0.15** | scikit-learn |

**The linear kernel** never takes the MLX path either. Its iterations are the cheapest of
any kernel's, so per-iteration overhead is nearly the whole cost, and no width makes up
for it (grid `svc-linear`):

<!-- python -m benchmarks.calibrate table svc-linear: scikit-learn 1.9.0 -->
| rows × features | linear, C = 1 | dispatch |
|---|---|---|
| 2 000 × 20 | **0.10** | scikit-learn |
| 2 000 × 256 | **0.07** | scikit-learn |

**Many classes.** One-vs-one solves `K(K−1)/2` subproblems of about `2n/K` rows, each with
its own Gram and its own loop, so the mean pair's work, `(2n/K)·d`, must clear the binary
work floor itself, 384 000 elements. Balanced, well-separated blobs (the benchmark's
multiclass data), rbf, C = 1, same notation (grid `svc-multi`):

<!-- python -m benchmarks.calibrate table svc-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | K = 4 | K = 8 | K = 16 | K = 32 | K = 64 | dispatch |
|---|---|---|---|---|---|---|
| 4 000 × 32 | **0.36** / **0.32** | **0.25** / **0.22** | **0.18** / **0.16** | **0.10** / **0.09** | — | scikit-learn |
| 8 000 × 32 | **0.39** / **0.38** | **0.32** / **0.31** | **0.24** / **0.23** | **0.17** / **0.16** | — | scikit-learn |
| 8 000 × 256 | 1.85 / 1.71 | 1.65 / 1.55 | 1.21 / 1.17 | **0.77** / **0.70** | — | MLX at K = 4, K = 8 |
| 2 048 × 512 | 2.07 / 1.92 | 1.39 / 1.28 | **0.81** / **0.73** | **0.47** / **0.40** | **0.23** / **0.22** | MLX at K = 4 |

Pairs of 256 × 512 and 500 × 256 lost, 0.73×–0.81× and 0.70×–0.77×. At 32 features,
where each pair takes a hundred-odd iterations, every K lost, as the width floor already
says. On overlapping classes — the binary target's score cut into K quantile bands — at
C = 1 to 100, grid `svc-multi-overlap`:

<!-- python -m benchmarks.calibrate table svc-multi-overlap: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, classes | C = 1 | C = 10 | C = 100 | dispatch |
|---|---|---|---|---|
| 3 000 × 128, K = 3 (pairs of 2 000) | 1.65 / 1.50 | 1.00~ / **0.92** | 1.00~ / **0.92** | scikit-learn |
| 4 000 × 128, K = 4 (pairs of 2 000) | 1.61 / 1.49 | 1.00~ / **0.92** | 1.00 / **0.91** | scikit-learn |
| 4 500 × 128, K = 3 (pairs of 3 000) | 1.94 / 1.79 | 1.18 / 1.09 | 1.18 / 1.09 | MLX |
| 6 000 × 128, K = 4 (pairs of 3 000) | 1.95 / 1.79 | 1.13 / 1.04 | 1.14 / 1.04 | MLX |
| 7 500 × 128, K = 5 (pairs of 3 000) | 1.96 / 1.80 | 1.12 / 1.03 | 1.12 / 1.03 | MLX |
| 15 000 × 128, K = 10 (pairs of 3 000) | 2.00 / 1.82 | 1.10 / 1.01 | 1.10 / 1.01 | MLX |
| 2 000 × 256, K = 4 (pairs of 1 000) | 2.00 / 1.83 | 1.21 / 1.09 | 1.22 / 1.09 | scikit-learn |
| 3 000 × 256, K = 4 (pairs of 1 500) | 2.54 / 2.35 | 1.52 / 1.39 | 1.52 / 1.39 | MLX |
| 1 536 × 512, K = 3 (pairs of 1 024) | 3.48 / 3.19 | 2.01 / 1.82 | 2.00 / 1.82 | MLX |
| 2 048 × 512, K = 4 (pairs of 1 024) | 3.52 / 3.20 | 2.04 / 1.85 | 2.03 / 1.86 | MLX |

Pairs of 2 000 × 128, the corner the binary work floor excludes, lost at C = 10 and 100 on
1.7.2 (0.91×–0.92×) and tied on 1.9.0; every pair from 3 000 × 128 up won or tied on both
(1.01×–3.5×). The floor at the binary work floor gives up pairs of 512 × 512 (2 048 rows,
K = 8; 1.28×–1.39× above) and of 1 000 × 256 (1.09×–2.0×), which won; the floor of
250 000 it replaced let the losing pairs of 2 000 × 128 through.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check.

### Inference: a few rows on the host

A model fitted on MLX used to answer every `predict` on the fitted device, and for a
batch of a few rows what the device charges is the call, not the work: 0.64–0.98 ms after
the GPU has idled, against 0.22–0.82 ms for LIBSVM's whole single-row predict (0.19–0.35
ms on the host, below). With the host threshold at 0 — every batch on the GPU — a single
row lost to LIBSVM at five of the six fits below on one version or both (0.27×–0.78×),
and two to four rows at four of them; only at 6 000 × 512, where LIBSVM's own predict is
slowest, did the device tie or win. So a batch of at most 2²⁵ rows ×
support vectors × features is evaluated on the host, in NumPy float32, with the device
tiles' own arithmetic (`svc_host_inference_work`); larger batches stay on the device.
Grid `svc-predict`: `predict` on an rbf model (C = 1) fitted on MLX against one fitted by
LIBSVM, 15 interleaved pairs; `dispatch` says which batch sizes the host answered:

<!-- python -m benchmarks.calibrate table svc-predict: scikit-learn 1.9.0 / 1.7.2 -->
| fit (support vectors) | 1 row | 2 rows | 4 rows | 8 rows | 16 rows | 32 rows | 64 rows | 128 rows | 256 rows | 1024 rows | 4096 rows | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 000 × 128 (2 169 SVs) | 1.13~ / 1.13 | 1.66 / 1.60 | 2.61 / 2.66 | 3.95 / 4.20 | 6.39 / 7.06 | 10.22 / 10.36 | 14.24 / 14.55 | 13.38 / 13.85 | 23.49 / 23.02 | 67.08 / 60.06 | 124.26 / 93.15 | host to 64 rows, MLX above |
| 1 500 × 256 (1 357 SVs) | 1.06~ / 1.21 | 1.42 / 1.51 | 2.37 / 2.56 | 3.80 / 4.04 | 6.15 / 6.76 | 10.60 / 10.80 | 14.80 / 15.72 | 12.81 / 12.15 | 21.25 / 21.86 | 59.01 / 55.10 | 102.15 / 85.24 | host to 64 rows, MLX above |
| 4 000 × 256 (3 115 SVs) | 1.62 / 1.66 | 2.06 / 2.21 | 3.69 / 3.90 | 5.99 / 6.59 | 10.34 / 11.17 | 16.46 / 16.51 | 13.87 / 14.56 | 24.66 / 25.32 | 43.45 / 40.45 | 101.83 / 85.11 | 151.61 / 98.77 | host to 32 rows, MLX above |
| 8 000 × 256, K = 4 (7 648 SVs) | 2.23 / 2.40 | 2.83 / 2.94 | 5.08 / 5.03 | 9.04 / 9.29 | 14.97 / 14.59 | 15.64 / 16.41 | 29.00 / 28.76 | 50.85 / 47.96 | 77.53 / 68.47 | 128.65 / 102.05 | 131.61 / 105.48 | host to 16 rows, MLX above |
| 1 100 × 512 (1 089 SVs) | 1.19 / 1.27 | 1.54 / 1.75 | 2.72 / 2.80 | 4.33 / 4.83 | 7.98 / 8.54 | 13.97 / 14.18 | 8.34 / 8.18 | 14.84 / 14.80 | 25.53 / 25.38 | 65.09 / 61.91 | 104.54 / 89.22 | host to 32 rows, MLX above |
| 6 000 × 512 (4 961 SVs) | 2.53 / 2.68 | 2.88 / 3.01 | 5.50 / 5.63 | 9.78 / 10.71 | 8.41 / 9.30 | 16.88 / 17.18 | 32.26 / 31.94 | 55.41 / 55.21 | 88.84 / 83.58 | 150.16 / 131.45 | 163.70 / 135.68 | host to 8 rows, MLX above |

Every batch size won or tied on both versions: single rows 1.06×–2.7×, and from 8 rows
up 3.8×–164×. The host stayed faster than a GPU that has idled up to about 5·10⁷ of that
work at 128 features, and further out at wider data, so the threshold sits below the
crossover at every shape; the dip where a shape crosses it (64 rows at 1 100 × 512,
14× → 8×) is the device's wake-up coming back, not a loss.

### The published SVC cases

Forced onto MLX, 25 interleaved pairs (5 for the linear kernel), scikit-learn 1.9.0:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 4 000 × 32 | 92.63 | 124.57 | 1.34× | 1.31× | scikit-learn |
| 4 000 × 64 | 115.29 | 181.83 | 1.58× | 1.59× | scikit-learn |
| 4 000 × 256 | 165.93 | 616.98 | 3.72× | 3.79× | MLX |
| 1 000 × 512 | 24.92 | 76.15 | 3.06× | 3.06× | scikit-learn |
| 4 000 × 64, C = 10 | 333.22 | 278.91 | **0.84×** | 0.84× | scikit-learn |
| 4 000 × 64, C = 10, kernel = poly | 315.80 | 276.13 | **0.87×** | 0.86× | scikit-learn |
| 2 000 × 128, C = 10 | 104.99 | 111.20 | 1.06× | 1.04× | scikit-learn |
| 2 000 × 128, C = 10, kernel = poly | 96.65 | 98.65 | 1.02× | 1.03× | scikit-learn |
| 4 000 × 128, C = 10 | 299.11 | 421.73 | 1.41× | 1.41× | MLX |
| 4 000 × 128, C = 10, kernel = poly | 299.77 | 395.03 | 1.32× | 1.32× | MLX |
| 4 000 × 256, C = 10 | 294.43 | 735.17 | 2.50× | 2.50× | MLX |
| 4 000 × 256, C = 10, kernel = poly | 285.66 | 655.52 | 2.29× | 2.32× | MLX |
| 4 000 × 256, C = 10, kernel = sigmoid | 647.17 | 426.00 | **0.66×** | 0.66× | scikit-learn |
| 2 000 × 20, kernel = linear (5 pairs) | 552.00 | 37.80 | **0.07×** | 0.07× | scikit-learn |
| 2 000 × 256, kernel = linear (5 pairs) | 21512.67 | 1059.74 | **0.05×** | 0.05× | scikit-learn |
| 2 048 × 512, K = 4 | 29.11 | 59.15 | 2.03× | 2.06× | MLX |
| 2 048 × 512, K = 16 | 283.88 | 226.48 | **0.80×** | 0.80× | scikit-learn |
| 2 500 × 20, K = 4 | 14.88 | 3.35 | **0.23×** | 0.22× | scikit-learn |

The forced wins below the floors that patched dispatch leaves on LIBSVM are the price of
rules that cannot see C or the iteration count: the same shapes at C = 10 lost. 2 000 × 128
at C = 10 won by 1.02×–1.06× here and lost on scikit-learn 1.7.2 (the `svc` table), which
is why the work floor sits above it. Every case here that patched dispatch sends to MLX
won, 1.32×–3.72×.

## Logistic regression — tall, or many classes

`fit` with `max_iter=200`. Binary targets are the benchmark data; multinomial ones are
overlapping Gaussian blobs, one per class, whose separation does not grow with the width.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2, median of 9 interleaved
pairs; bold and `~` as above. Grids `logreg-binary` and `logreg-multi`:

<!-- python -m benchmarks.calibrate table logreg-binary: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 1 feature | 4 features | 16 features | 32 features | 64 features | 128 features | 256 features | 1024 features | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | **0.26** / **0.24** | **0.23** / **0.28** | **0.32** / **0.28** | **0.29** / **0.29** | — | — | — | — | scikit-learn |
| 20 000 | **0.48** / **0.55** | **0.45** / **0.52** | **0.42** / 0.54~ | **0.43** / **0.49** | **0.59** / **0.50** | 0.74~ / **0.55** | **0.83** / **0.65** | **0.87** / **0.83** | scikit-learn |
| 50 000 | 1.30 / 1.03~ | 1.27 / 1.10~ | 1.02~ / 0.93~ | 1.01~ / 0.81~ | 1.11 / 0.97~ | 1.15 / 1.08 | 1.25 / 1.09 | 1.11 / 1.03 | scikit-learn |
| 100 000 | 2.22 / 2.10 | 1.82 / 1.71 | 1.64 / 1.52 | 1.65 / 1.51 | 1.48 / 1.35 | 1.45 / 1.35 | 1.35 / 1.30 | 1.45 / 1.39 | scikit-learn |
| 200 000 | 3.19 / 2.98 | 2.39 / 2.20 | 2.32 / 2.12 | 2.00 / 1.88 | 1.79 / 1.63 | 1.65 / 1.55 | 1.63 / 1.58 | 1.58 / 1.53 | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| 1 000 000 | 6.17 / 5.54 | 4.39 / 3.77 | 4.39 / 3.96 | 3.82 / 3.55 | 2.85 / 2.63 | 2.30 / 2.14 | 1.95 / 1.87 | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |

<!-- python -m benchmarks.calibrate table logreg-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 3 classes × 1 | 3 classes × 4 | 3 classes × 16 | 3 classes × 64 | 3 classes × 512 | 5 classes × 64 | 10 classes × 16 | 10 classes × 128 | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | 0.64~ / 0.78~ | **0.56** / **0.55** | **0.41** / **0.45** | **0.37** / **0.39** | 1.23 / 1.14 | **0.46** / **0.48** | 0.81~ / 0.91~ | **0.66** / **0.62** | scikit-learn |
| 20 000 | 0.78~ / 1.15~ | 0.99~ / 0.98~ | 1.22 / 1.18 | **0.86** / 0.79~ | 1.80 / 1.73 | 1.11 / 1.06 | 2.34 / 2.29 | 1.44 / 1.39 | scikit-learn |
| 60 000 | 1.88 / 1.87 | 3.38 / 3.49 | 1.52 / 1.50 | 2.13 / 2.17 | 2.79 / 2.93 | 2.76 / 2.53 | 3.34 / 3.04 | 2.33 / 2.32 | MLX |
| 200 000 | 2.67 / 2.67 | 3.43 / 3.21 | 4.76 / 4.60 | 3.77 / 3.78 | 3.96 / 3.81 | 4.46 / 4.28 | 8.58 / 8.42 | 5.23 / 5.24 | MLX |

What an MLX fit pays once is a pass over `X` — the column means, the centered float32
copy and the float32 range check, each split by rows across up to eight threads — its
upload, and a device round trip per objective evaluation; what it saves is per
iteration, the loss and its gradient on the GPU. The iteration counts follow
scikit-learn's: no fit patched dispatch sends to MLX took more than one iteration more
than scikit-learn's in any table here, and multinomial fits on overlapping blobs often
took fewer (22 against 36 at 60 000 × 4). The first step is L-BFGS-B's own, a unit move
along the first gradient, and the line search takes an interpolated step anywhere past
1% of its bracket rather than bisecting until the step clears 10%.

- **Binary: faster from 100 000 rows at every width up to 1 024,** 1.30×–6.2× on both
  versions; at 50 000 rows it depends on the width, and below it loses.
- **Multinomial: faster from 60 000 rows at every width,** 1.50×–8.6× at 3, 5 and 10
  classes and 1–512 features on both versions. At 20 000 rows it depends on the classes
  and the width (0.78×–2.3×); at 5 000 it loses or ties at every width but 512.

A fit that converges in one or two iterations is mostly the fixed cost, and both of those
targets hide one. Classes that are well separated — `benchmarks/run.py`'s multiclass
data, whose centres move apart with the width — converge in one scikit-learn iteration
from 64 features on; grid `logreg-separable`:

<!-- python -m benchmarks.calibrate table logreg-separable: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 3 classes × 16 | 3 classes × 64 | 3 classes × 128 | 3 classes × 256 | 3 classes × 512 | 5 classes × 64 | 5 classes × 256 | 10 classes × 128 | 10 classes × 512 | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 000 | 1.54 / 1.49 | 1.20 / 1.14 | 1.36 / 1.30 | 1.02~ / 1.00~ | 1.11 / 1.09 | 2.35 / 2.31 | 1.08 / 1.06 | 2.29 / 2.25 | 1.22 / 1.21 | MLX |
| 200 000 | 4.72 / 4.74 | 2.64 / 2.67 | 1.25 / 1.22 | 1.20 / 1.18 | 1.35 / 1.29 | 4.15 / 4.00 | 1.29 / 1.28 | 4.52 / 4.37 | 1.51 / 1.50 | MLX |

Labels that carry no signal converge in one or two, and noisy ones in about four; grid
`logreg-labels` fits both, random labels and `x₀ + N(0, 3²) > 0`:

<!-- python -m benchmarks.calibrate table logreg-labels: scikit-learn 1.9.0 / 1.7.2 -->
| labels, rows | 1 feature | 4 features | 16 features | 32 features | 64 features | 256 features | 3 classes × 16 | 3 classes × 64 | 3 classes × 256 | 10 classes × 64 | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noisy, 100 000 | 1.71 / 1.16 | 1.40 / 1.07~ | 1.25 / **0.88** | 1.36 / 1.03~ | 1.24 / 1.08~ | 1.01~ / 0.97 | — | — | — | — | scikit-learn |
| noisy, 200 000 | 2.71 / 2.37 | 1.89 / 1.58 | 1.93 / 1.53 | 1.74 / 1.55 | 1.43 / 1.25 | 1.20 / 1.13 | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| noisy, 500 000 | 3.91 / 3.43 | 2.36 / 2.07 | 2.46 / 2.17 | 2.16 / 1.94 | 1.69 / 1.55 | 1.37 / 1.29 | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| noisy, 1 000 000 | 4.52 / 3.97 | 2.95 / 2.61 | 2.77 / 2.46 | 2.42 / 2.16 | 1.94 / 1.73 | 1.46 / 1.37 | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| random, 60 000 | — | — | — | — | — | — | 1.22 / 1.18 | 1.71 / 1.66 | 1.42 / 1.38 | 1.76 / 1.70 | MLX |
| random, 100 000 | 1.66 / 1.01~ | **0.81** / **0.61** | 1.17 / **0.77** | 1.16 / **0.82** | 1.01~ / **0.76** | **0.90** / **0.84** | — | — | — | — | scikit-learn |
| random, 200 000 | 1.67 / 1.15 | 1.18 / 0.97~ | 1.36 / 1.12 | 1.45 / 1.27 | 1.18 / 1.03 | 1.03 / 0.96 | 2.40 / 2.45 | 2.10 / 1.98 | 2.02 / 1.98 | 2.25 / 2.26 | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features, 3 classes × 16, 3 classes × 64, 3 classes × 256, 10 classes × 64 |
| random, 500 000 | 1.84 / 1.47 | 2.00 / 1.70 | 1.78 / 1.53 | 1.58 / 1.35 | 1.26 / 1.09 | 1.17 / 1.09 | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| random, 1 000 000 | 2.27 / 1.76 | 1.94 / 1.57 | 1.89 / 1.58 | 1.59 / 1.35 | 1.41 / 1.21 | 1.06 / 0.97 | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |

Separable classes tied at worst (1.00×~ at 60 000 × 256, three classes), so the
multinomial floor stands. Labels with little or no signal lost at 100 000 rows — on
1.7.2 at 4–256 features (0.61×–0.88×), on 1.9.0 at 4 and 256 (0.81×, 0.90×) — and random
ones at 256 features stood at 0.96×–1.17× from 200 000 rows, too close to call. So a binary fit needs 200 000 rows and at most 64 features, where every label set
measured won or tied (0.97× at worst); the price is the binary wins between 100 000 and
200 000 rows, and beyond 64 features (1.30×–2.3× in the `logreg-binary` table).

The 0.1.0a2 rule — 20 000 rows and 2·10⁷ elements, both required, no width limit — sent
20 000 × 1 024 (0.83×–0.87× even now) to MLX and kept 1 000 000 × 16 (4.0×–4.4×) from it.
Before the host passes were threaded and the first step matched L-BFGS-B's, binary fits
lost at 64 features and more up to 200 000 rows, and separable multinomial fits lost
0.31×–0.46× at 256 and 512 features, where MLX took 11–15 iterations to get where
scikit-learn's one did.

### The published logistic cases

Forced onto MLX, 25 interleaved pairs, scikit-learn 1.9.0; the multinomial targets are
the benchmark's well-separated blobs:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 20 000 × 16 | 7.07 | 3.08 | **0.44×** | 0.66× | scikit-learn |
| 200 000 × 16 | 11.29 | 25.46 | 2.26× | 2.34× | MLX |
| 1 000 000 × 16 | 31.81 | 142.24 | 4.47× | 5.21× | MLX |
| 200 000 × 128 | 22.59 | 37.01 | 1.64× | 1.70× | scikit-learn |
| 20 000 × 1024 | 21.68 | 19.24 | **0.89×** | 0.90× | scikit-learn |
| 100 000 × 1024 | 62.14 | 90.76 | 1.46× | 1.49× | scikit-learn |
| 20 000 × 64, K = 5 | 10.59 | 12.09 | 1.14× | 1.17× | scikit-learn |
| 60 000 × 64, K = 5 | 18.67 | 52.77 | 2.83× | 5.27× | MLX |
| 200 000 × 64, K = 5 | 31.37 | 108.77 | 3.47× | 4.67× | MLX |

## Reading these numbers honestly

- Below a threshold, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path. The calibration and kernel tables show those cases so that the
  thresholds can be checked, not as claims.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name; a patched
  fit the data probe hands to scikit-learn pays for the index and the probe first.
- The rules were measured on synthetic data, and data unlike it can land elsewhere. The
  neighbor tables are k = 10, apart from `knn-k`, on Gaussian data and the laws of
  `knn-data` and `knn-narrow`; the probe reads the one property of the data that was
  seen to decide, but no set of laws is all data. `SVC`'s are the binary target above,
  overlapping and separated classes, C from 0.1 to 100, and `gamma`, `degree` and
  `coef0` as `svc-params` varies them; C beyond 100 at the default kernel was not
  tabulated. `LogisticRegression`'s are labels with a clear signal, a weak one and none.
  k is part of no rule.
