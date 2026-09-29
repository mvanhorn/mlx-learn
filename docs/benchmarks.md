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
size from 1 to 4 096 for the neighbor estimators, every kernel and C from 0.1 to 100 for
`SVC`, binary and multinomial targets for `LogisticRegression`. A fit cannot see how many
queries will follow it or how many iterations it will take, so where the answer turns on
those, the threshold takes the side that does not lose. Everything else goes to stock
scikit-learn.

| Estimator | Patched dispatch uses MLX when — all required | `Tuning` |
|---|---|---|
| neighbor estimators | at least 16 features, 4 096 rows and 6 000 000 elements (rows × features); and with fewer than 256 features, at least 1 000 000 rows | `knn_min_features = 16`, `knn_min_train_samples = 4096`, `knn_min_work = 6_000_000`, `knn_wide_features = 256`, `knn_narrow_min_samples = 1_000_000` |
| `LogisticRegression` | at least 60 000 rows and 100 000 element-ops (rows × features × weight vectors); binary: also at least 100 000 rows and at most 32 features | `logreg_min_samples = 60_000`, `logreg_min_work = 100_000`, `logreg_binary_min_samples = 100_000`, `logreg_binary_max_features = 32` |
| `SVC` | the rbf or poly kernel, at least 128 features, 1 024 rows and 384 000 elements; with more than two classes, a mean one-vs-one pair `(2n/K) · d` of at least 250 000 | `svc_min_features = 128`, `svc_min_samples = 1024`, `svc_min_work = 384_000`, `svc_min_pair_work = 250_000` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

### Patched dispatch against stock: the gate

The `smoke` profile measures the classes `patch_sklearn()` installs on both sides of every
threshold. Run under both scikit-learn versions, 7 interleaved pairs each, it passed the
performance gate on both (`~` within noise, **bold** a significant loss):

| case | patched dispatch | mlxlearn / stock, ms (1.9.0) | 1.9.0 | 1.7.2 |
|---|---|---|---|---|
| `knn/fit/1000x32/q256` | scikit-learn | 0.32 / 0.23 | **0.73×** | **0.77×** |
| `knn/kneighbors/1000x32/q256` | scikit-learn | 1.42 / 1.50 | 1.06×~ | 1.02×~ |
| `knn/fit/65536x256/q256` | MLX | 27.57 / 2.30 | **0.08×** | **0.07×** |
| `knn/kneighbors/65536x256/q256` | MLX | 6.73 / 18.43 | 2.74× | 2.98× |
| `knn/kneighbors/262144x32/q256` | scikit-learn | 17.50 / 16.28 | 0.93×~ | 1.04×~ |
| `knn/kneighbors/1048576x16/q256` | MLX | 36.51 / 48.59 | 1.33× | 1.23× |
| `knn/kneighbors/1048576x8/q256` | scikit-learn | 48.45 / 48.40 | 1.00×~ | 1.00×~ |
| `knn/kneighbors/50000x3/q256` | scikit-learn | 0.95 / 0.92 | 0.97×~ | 1.02×~ |
| `logreg/fit/5000x32` | scikit-learn | 1.66 / 1.59 | 0.96×~ | 0.95×~ |
| `logreg/fit/200000x128` | scikit-learn | 39.23 / 39.18 | 1.00×~ | 0.99×~ |
| `logreg/fit/200000x16` | MLX | 16.82 / 29.30 | 1.74× | 1.74× |
| `logreg/fit/60000x64/n_classes=5` | MLX | 26.76 / 46.41 | 1.73× | 1.80× |
| `svc/fit/1000x32` | scikit-learn | 8.80 / 8.81 | 1.00×~ | 1.00×~ |
| `svc/fit/4000x64` | scikit-learn | 186.93 / 186.08 | 1.00×~ | 1.00×~ |
| `svc/fit/4000x256` | MLX | 163.03 / 615.06 | 3.77× | 3.51× |
| `svc/fit/2048x512/n_classes=4` | MLX | 30.77 / 61.53 | 2.00× | 1.86× |
| `svc/fit/2048x512/n_classes=16` | scikit-learn | 238.36 / 238.62 | 1.00×~ | 1.00×~ |

A handoff costs something of its own, about 0.1 ms for a neighbor fit: a third of a stock
brute-force fit at 1 000 × 32 (0.32 against 0.23 ms), which stores a reference and does
nothing else. So patched `knn/fit` below the floors reads 0.73×–0.77×, one more instance
of the accepted `knn/fit` regression below; the logistic and SVC handoffs were within
noise of stock.

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
| 1 000 000 × 16 | 1.65 | 2.59 | 1.90 | 1.17 | 1.21~ | 1.45 | 1.31 | 1.25 | 1.23 | 1.10~ | 1.08~ | 1.17~ | **0.05** | MLX |
| 1 600 000 × 16 | 1.92 | 2.97 | 2.25 | 1.95 | 1.94 | 1.63 | 1.58 | 1.55 | 1.48 | 1.33 | 1.42 | 1.17~ | **0.05** | MLX |
| 80 000 × 24 | 1.09~ | 1.12~ | 1.02~ | 0.97~ | **0.84** | **0.67** | **0.53** | **0.46** | **0.64** | 1.25 | 1.37 | 1.43 | **0.08** | scikit-learn |
| 120 000 × 24 | 1.17 | 1.29 | 1.21 | 0.90~ | **0.73** | **0.62** | **0.47** | **0.44** | 1.26 | 1.28 | 1.38 | 1.20 | **0.07** | scikit-learn |
| 180 000 × 24 | 1.12~ | 1.22~ | 1.18 | 1.01~ | **0.75** | **0.61** | **0.49** | **0.62** | 1.21 | 1.27 | 1.40 | 1.02~ | **0.07** | scikit-learn |
| 262 144 × 24 | 1.32~ | 1.60 | 1.40 | 1.23 | 0.89~ | **0.68** | 0.65~ | 1.59 | 1.65 | 1.77 | 1.91 | 1.48 | **0.07** | scikit-learn |
| 400 000 × 24 | 1.32 | 1.66 | 1.39 | 1.15 | **0.80** | **0.55** | 1.20 | 1.28 | 1.27 | 1.32 | 1.05~ | 0.91~ | **0.06** | scikit-learn |
| 600 000 × 24 | 1.05~ | 1.89 | 1.52 | 1.08 | **0.70** | 0.97~ | 1.28 | 1.28 | 1.28 | 1.28 | 1.03~ | 1.02~ | **0.06** | scikit-learn |
| 1 000 000 × 24 | 1.82 | 2.08 | 1.62 | 1.12 | 1.22~ | 1.55 | 1.38 | 1.39 | 1.37 | 1.17 | 1.12 | 1.19~ | **0.05** | MLX |
| 1 600 000 × 24 | 1.90 | 2.32 | 1.80 | 2.05 | 1.34 | 1.63 | 1.67 | 1.69 | 1.68 | 1.36 | 1.50 | — | **0.05** | MLX |
| 80 000 × 32 | 1.01~ | 1.05~ | 1.09~ | 0.96~ | **0.84** | **0.70** | **0.60** | **0.47** | 0.91~ | 1.25 | 1.39 | 1.67 | **0.08** | scikit-learn |
| 120 000 × 32 | 1.16~ | 1.23 | 1.06~ | 0.93~ | **0.86** | **0.62** | **0.52** | 0.67~ | 1.30 | 1.35 | 1.50 | 1.37 | **0.07** | scikit-learn |
| 180 000 × 32 | 1.11~ | 1.38 | 1.23 | 1.05~ | **0.76** | **0.57** | 0.82~ | 0.84~ | 1.14 | 1.33 | 1.52 | 1.13 | **0.07** | scikit-learn |
| 262 144 × 32 | 1.10~ | 1.56 | 1.39 | 1.19 | 0.94~ | **0.76** | 0.86~ | 1.55 | 1.65 | 1.87 | 2.09 | 1.66 | **0.06** | scikit-learn |
| 400 000 × 32 | 1.14~ | 1.63 | 1.46 | 1.12 | **0.75** | 0.78~ | 0.91~ | 1.34 | 1.36 | 1.45 | 1.16 | 1.00~ | **0.06** | scikit-learn |
| 600 000 × 32 | 1.29~ | 1.96 | 1.54 | 1.12 | 1.19~ | 1.32~ | 1.47 | 1.40 | 1.39 | 1.39 | 1.19 | 1.14~ | **0.06** | scikit-learn |
| 1 000 000 × 32 | 1.77 | 2.17 | 1.59 | 1.13 | 1.44 | 1.56 | 1.46 | 1.44 | 1.43 | 1.18 | 1.20 | — | **0.06** | MLX |
| 1 600 000 × 32 | 2.10 | 2.30 | 1.82 | 1.75 | 1.88 | 1.84 | 2.25 | 1.97 | 1.86 | 1.50 | 1.63 | — | **0.05** | MLX |
| 50 000 × 48 | 0.86~ | 1.05~ | 1.04~ | 0.98~ | 0.80~ | **0.78** | **0.66** | **0.58** | 0.84~ | 0.97~ | 1.47 | 1.95 | **0.08** | scikit-learn |
| 80 000 × 48 | 1.09~ | 1.04~ | 1.10 | 1.03~ | **0.85** | **0.72** | **0.63** | **0.55** | 0.99~ | 1.24~ | 1.72 | 1.91 | **0.07** | scikit-learn |
| 120 000 × 48 | 1.11~ | 1.18 | 1.10~ | 1.02~ | 0.85~ | **0.66** | **0.62** | 0.89~ | 1.04~ | 1.61 | 1.85 | 1.72 | **0.06** | scikit-learn |
| 180 000 × 48 | 1.26 | 1.35 | 1.17 | 1.09~ | **0.81** | **0.72** | 0.99~ | 1.03~ | 1.46 | 1.72 | 1.81 | 1.48 | **0.06** | scikit-learn |
| 262 144 × 48 | 1.41 | 1.53 | 1.44 | 1.25 | 0.98~ | **0.77** | 0.70~ | 1.69 | 1.84 | 2.18 | 2.54 | 2.20 | **0.06** | scikit-learn |
| 400 000 × 48 | 1.61 | 1.60 | 1.36 | 1.04~ | **0.82** | **0.62** | 1.55 | 1.64 | 1.77 | 1.93 | 1.67 | 1.43 | **0.06** | scikit-learn |
| 600 000 × 48 | 1.91 | 1.86 | 1.51 | 1.09 | 0.79~ | **0.90** | 1.63 | 1.58 | 1.76 | 1.73 | 1.46 | 1.45~ | **0.06** | scikit-learn |
| 1 000 000 × 48 | 2.21 | 1.95 | 1.45 | 1.11~ | 1.15~ | 1.99 | 2.07 | 1.88 | 1.75 | 1.44 | 1.49 | — | **0.06** | MLX |
| 1 600 000 × 48 | 2.57 | 2.07 | 1.63 | 1.89 | 2.61 | 2.08 | 2.56 | 2.29 | 2.06 | 1.78 | 1.78 | — | **0.05** | MLX |
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
  42 of the 50 sizes measured below 1 000 000 rows, down to 0.44× — still at 600 000 rows
  at 16, 24 and 48 features. From 1 000 000 rows every batch of 1–4 096 queries won or
  tied on both versions, the worst batch 1.08×–1.9×. Large batches do not lose there:
  1 024 queries or more never did (up to 4.3×), which is why a workload known to query in
  large batches should import `mlxlearn.neighbors` directly.
- **The query batch.** It is the one of the three a fit cannot see, and the rules are set
  so that it does not matter: every shape patched dispatch sends to MLX won or tied at
  every batch size, on both versions.

What the million-row rule gives up is mostly the idle GPU's doing: called back to back, 64
queries against 262 144 × 16 took MLX 3.2 ms, and stock brute force about 5 ms. Answering
mid-size batches somewhere other than the fitted backend would re-decide the backend at
inference, which the sticky-backend contract rules out; a search that keeps the GPU from
idling, or hides its wake-up, is future work.

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

Where patched dispatch uses MLX, then — tall data from 16 features, wide data from 256 —
the two grids above answered single rows 1.0×–5.7× faster than stock, 16 queries
1.1×–3.2×, 256 queries 1.2×–7.7× and 4 096 queries 1.2×–6.2×, on both versions, and every
batch size between those won or tied too.

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

Patched dispatch therefore hands every fit with fewer than 16 features to scikit-learn:
to its KD-tree where it would build one (reason `"tree-wins"`), and to its brute force
otherwise.

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
| 1 000 × 32 | 256 | 1.34 | 1.37 | 1.02×~ | 1.17× | scikit-learn |
| 4 096 × 64 | 256 | 1.82 | 2.06 | 1.13× | 1.08× | scikit-learn |
| 32 768 × 16 | 256 | 5.63 | 3.39 | **0.60×** | 0.65× | scikit-learn |
| 50 000 × 32 | 256 | 6.10 | 4.73 | 0.78×~ | 1.34× | scikit-learn |
| 8 192 × 768 | 256 | 4.68 | 6.48 | 1.38× | 1.56× | MLX |
| 131 072 × 128 | 256 | 8.82 | 19.87 | 2.25× | 2.48× | scikit-learn |
| 128 000 × 768 | 256 | 14.20 | 68.81 | 4.85× | 6.90× | MLX |
| 1 000 000 × 16 | 256 | 44.26 | 47.64 | 1.08× | 1.03× | MLX |
| 1 000 × 32 | 1 | 1.03 | 0.92 | 0.89×~ | 1.08× | scikit-learn |
| 1 000 × 32 | 64 | 1.14 | 1.05 | 0.92×~ | 0.99× | scikit-learn |
| 1 000 × 32 | 4096 | 7.41 | 2.41 | **0.33×** | 0.35× | scikit-learn |
| 8 192 × 768 | 1 | 1.21 | 1.53 | 1.27× | 1.32× | MLX |
| 8 192 × 768 | 64 | 2.00 | 3.16 | 1.58× | 1.49× | MLX |
| 8 192 × 768 | 4096 | 37.52 | 97.78 | 2.61× | 3.48× | MLX |
| 131 072 × 128 | 1 | 1.50 | 2.41 | 1.61× | 1.68× | scikit-learn |
| 131 072 × 128 | 64 | 6.56 | 6.19 | 0.94×~ | 1.92× | scikit-learn |
| 131 072 × 128 | 4096 | 89.62 | 348.81 | 3.89× | 4.27× | scikit-learn |
| 1 000 000 × 16 | 1 | 1.95 | 3.39 | 1.74× | 1.83× | MLX |
| 1 000 000 × 16 | 64 | 11.98 | 16.13 | 1.35× | 1.31× | MLX |
| 1 000 000 × 16 | 4096 | 799.91 | 902.18 | 1.13× | 1.07× | MLX |

| samples × features | mlxlearn fit (ms) | scikit-learn fit (ms) | ratio | dispatch |
|---|---|---|---|---|
| 1 000 × 32 | 1.06 | 0.21 | **0.20×** | scikit-learn |
| 4 096 × 64 | 1.62 | 0.27 | **0.16×** | scikit-learn |
| 32 768 × 16 | 2.37 | 0.28 | **0.12×** | scikit-learn |
| 50 000 × 32 | 4.27 | 0.42 | **0.10×** | scikit-learn |
| 8 192 × 768 | 11.23 | 1.02 | **0.09×** | MLX |
| 131 072 × 128 | 28.53 | 2.27 | **0.08×** | scikit-learn |
| 128 000 × 768 | 152.96 | 12.39 | **0.08×** | MLX |
| 1 000 000 × 16 | 36.21 | 2.17 | **0.06×** | MLX |

131 072 × 128, which the previous floors sent to MLX, won single rows and large batches
here, but 64 queries only tied, and at 120 000 × 128 in the `knn` table 64 queries lost
(0.80×): patched dispatch now hands it to scikit-learn.

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
features and 103 µs at 256, because every kernel row it misses costs `O(n·d)`. They cross
near 100 features. The kernel and C change how many iterations there are and how often
LIBSVM misses, not where the crossing lies: below 128 features rbf and poly lost at C = 10
and 100 everywhere (0.15×–0.85×), and poly at C = 1 up to 32 features; from 128 features
they won at every C — once the fit had rows. At 128 features 1 000–2 000 rows still lost
at C = 10 and above on one version or both (0.61×–0.95×; 2 000 × 128 on 1.7.2 only), and
from 3 000 rows every kernel and C won or tied on both. C = 10 and C = 100 behave alike;
C = 0.1, which needs fewer iterations of LIBSVM, favours MLX everywhere.

The rules look at neither the kernel nor C, so they take the width, rows and work at
which all of them win: 128 features, 1 024 rows and 384 000 elements — the corner at
3 000 × 128, which also keeps 1 000 × 256 (a 1.04× win at worst) and lets 1 500 × 256
and 1 024 × 512 through. Above all three floors every fit measured won or tied on both
versions, 1.04×–8.4× (1.04×–4.4× at C of 1 and 10, up to 8 000 rows).

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
its own Gram and its own loop, so the mean pair's work must clear 250 000 elements.
Balanced, well-separated blobs (the benchmark's multiclass data), rbf, C = 1, same
notation (grid `svc-multi`):

<!-- python -m benchmarks.calibrate table svc-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | K = 4 | K = 8 | K = 16 | K = 32 | K = 64 | dispatch |
|---|---|---|---|---|---|---|
| 4 000 × 32 | **0.36** / **0.32** | **0.25** / **0.22** | **0.18** / **0.16** | **0.10** / **0.09** | — | scikit-learn |
| 8 000 × 32 | **0.39** / **0.38** | **0.32** / **0.31** | **0.24** / **0.23** | **0.17** / **0.16** | — | scikit-learn |
| 8 000 × 256 | 1.85 / 1.71 | 1.65 / 1.55 | 1.21 / 1.17 | **0.77** / **0.70** | — | MLX at K = 4, K = 8, K = 16 |
| 2 048 × 512 | 2.07 / 1.92 | 1.39 / 1.28 | **0.81** / **0.73** | **0.47** / **0.40** | **0.23** / **0.22** | MLX at K = 4, K = 8 |

Pairs of 512 rows × 512 features (2 048 rows, K = 8) and of 1 000 × 256 (8 000 rows,
K = 16) won, 1.28×–1.39× and 1.17×–1.21×; 256 × 512 and 500 × 256 lost, 0.73×–0.81× and
0.70×–0.77×. At 32 features, where each pair takes a hundred-odd iterations, every K
lost, as the width floor already says.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check.

### The published SVC cases

Forced onto MLX, 25 interleaved pairs (5 for the linear kernel), scikit-learn 1.9.0:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 4 000 × 32 | 94.47 | 122.63 | 1.30× | 1.29× | scikit-learn |
| 4 000 × 64 | 118.90 | 183.81 | 1.55× | 1.56× | scikit-learn |
| 4 000 × 256 | 169.13 | 621.12 | 3.67× | 3.72× | MLX |
| 1 000 × 512 | 25.42 | 77.03 | 3.03× | 3.05× | scikit-learn |
| 4 000 × 64, C = 10 | 336.19 | 277.28 | **0.82×** | 0.83× | scikit-learn |
| 4 000 × 64, C = 10, kernel = poly | 322.74 | 273.39 | **0.85×** | 0.86× | scikit-learn |
| 2 000 × 128, C = 10 | 107.01 | 110.29 | 1.03× | 1.03× | scikit-learn |
| 2 000 × 128, C = 10, kernel = poly | 98.34 | 100.13 | 1.02× | 1.03× | scikit-learn |
| 4 000 × 128, C = 10 | 304.12 | 423.14 | 1.39× | 1.41× | MLX |
| 4 000 × 128, C = 10, kernel = poly | 306.98 | 399.83 | 1.30× | 1.30× | MLX |
| 4 000 × 256, C = 10 | 300.85 | 743.24 | 2.47× | 2.47× | MLX |
| 4 000 × 256, C = 10, kernel = poly | 291.07 | 667.33 | 2.29× | 2.30× | MLX |
| 4 000 × 256, C = 10, kernel = sigmoid | 656.18 | 433.11 | **0.66×** | 0.66× | scikit-learn |
| 2 000 × 20, kernel = linear (5 pairs) | 560.46 | 37.90 | **0.07×** | 0.07× | scikit-learn |
| 2 000 × 256, kernel = linear (5 pairs) | 22417.65 | 1087.00 | **0.05×** | 0.05× | scikit-learn |
| 2 048 × 512, K = 4 | 30.33 | 61.55 | 2.03× | 2.09× | MLX |
| 2 048 × 512, K = 16 | 289.51 | 234.25 | **0.81×** | 0.82× | scikit-learn |
| 2 500 × 20, K = 4 | 14.77 | 3.53 | **0.24×** | 0.25× | scikit-learn |

The forced wins below the floors that patched dispatch leaves on LIBSVM are the price of
rules that cannot see C or the iteration count: the same shapes at C = 10 lost. 2 000 × 128
at C = 10 won by 1.02×–1.03× here and lost on scikit-learn 1.7.2 (the `svc` table), which
is why the work floor sits above it.

## Logistic regression — narrow and tall, or many classes

`fit` with `max_iter=200`. Binary targets are the benchmark data; multinomial ones are
overlapping Gaussian blobs, one per class, whose separation does not grow with the width.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2, median of 9 interleaved
pairs; bold and `~` as above. Grids `logreg-binary` and `logreg-multi`:

<!-- python -m benchmarks.calibrate table logreg-binary: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 1 feature | 4 features | 16 features | 32 features | 64 features | 128 features | 256 features | 1024 features | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | **0.31** / **0.26** | **0.30** / **0.26** | **0.37** / **0.35** | **0.38** / **0.35** | — | — | — | — | scikit-learn |
| 20 000 | **0.71** / **0.63** | **0.62** / **0.59** | **0.67** / **0.57** | **0.62** / **0.57** | **0.59** / **0.54** | **0.67** / **0.58** | **0.64** / **0.55** | **0.56** / **0.45** | scikit-learn |
| 50 000 | 1.31 / 1.22~ | 1.40 / 1.36 | 0.93~ / **0.94** | **0.88** / 0.85~ | **0.89** / **0.81** | **0.87** / **0.74** | **0.76** / **0.67** | **0.58** / **0.52** | scikit-learn |
| 100 000 | 1.89 / 1.70 | 1.58 / 1.51 | 1.38 / 1.36 | 1.20 / 1.10 | **0.94** / **0.88** | **0.88** / **0.82** | **0.76** / **0.72** | **0.70** / **0.64** | MLX at 1 feature, 4 features, 16 features, 32 features |
| 200 000 | 2.33 / 2.46 | 2.06 / 2.07 | 1.74 / 1.71 | 1.27 / 1.24 | 1.00~ / **0.94** | **0.86** / **0.80** | **0.83** / **0.77** | **0.72** / **0.67** | MLX at 1 feature, 4 features, 16 features, 32 features |
| 1 000 000 | 3.78 / 4.59 | 2.43 / 2.59 | 2.37 / 2.47 | 1.86 / 1.88 | 1.29 / 1.27 | 1.02 / **0.93** | **0.85** / **0.78** | — | MLX at 1 feature, 4 features, 16 features, 32 features |

<!-- python -m benchmarks.calibrate table logreg-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 3 classes × 1 | 3 classes × 4 | 3 classes × 16 | 3 classes × 64 | 3 classes × 512 | 5 classes × 64 | 10 classes × 16 | 10 classes × 128 | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | **0.69** / 0.67~ | **0.54** / **0.51** | **0.43** / **0.42** | **0.39** / **0.35** | 0.96~ / **0.89** | **0.46** / **0.46** | **0.89** / 0.90~ | **0.53** / **0.55** | scikit-learn |
| 20 000 | 0.69~ / 0.52~ | 0.89~ / **0.84** | 1.02~ / 0.95~ | **0.74** / 0.74~ | 1.27 / 1.18~ | 0.96~ / 1.01~ | 2.01 / 1.86 | 1.22 / 1.20 | scikit-learn |
| 60 000 | 1.61 / 1.50 | 3.12 / 2.60 | 1.31 / 1.73 | 1.64 / 1.80 | 1.74 / 1.60 | 2.26 / 2.93 | 2.94 / 3.08 | 2.04 / 2.05 | MLX |
| 200 000 | 2.18 / 2.12 | 3.19 / 3.08 | 4.00 / 3.73 | 2.73 / 2.47 | 1.98 / 1.88 | 3.16 / 3.26 | 7.03 / 7.03 | 3.45 / 3.29 | MLX |

The L-BFGS iteration counts match scikit-learn's, so what differs is the cost of an
iteration. The solver's bookkeeping runs in float64 on the host, one device
synchronization per objective evaluation, and 0.1.0a2's per-iteration overhead — 12–33
blocking host synchronizations per iteration in that bookkeeping, not line-search
evaluations, which average about 1.1 per iteration — is gone.

What an MLX fit spends its time on now is mostly one pass over `X`. Profiled at
100 000 × 1 024 (binary), a 127 ms fit spent 64 ms making the float32 copy of `X` and its
transpose on the device and 20 ms in the float32 range check, and 19 ms in its seven
L-BFGS iterations; scikit-learn's whole fit took 92 ms. At 1 000 000 × 16 a fit took
59 ms, of which 15 ms preparing `X` and 18 ms iterating, against 145 ms for
scikit-learn, whose per-sample loss dominates at that width. So a binary fit wins where
its iterations save more than one pass over `X` costs — tall, narrow data — and a
multinomial fit, which does a matrix product per class in every iteration, wins almost
everywhere past the fixed costs.

- **Binary, narrow and tall: faster.** From 100 000 rows at 1–32 features, 1.10×–4.6× on
  both versions; at 50 000 rows only 1–4 features win or tie.
- **Binary and wide: slower, at every size tried.** At 256 and 1 024 features,
  0.45×–0.85× up to 1 000 000 × 256; at 64–128 features, parity only at a million rows
  (0.93×–1.29×).
- **Multinomial: faster from 60 000 rows at every width,** 1.31×–7.0× at 3, 5 and 10
  classes and 1–512 features on both versions. At 20 000 rows it depends on the classes
  and the width (0.52×–2.0×); at 5 000 it loses (0.35×–0.96×).

The 0.1.0a2 rule — 20 000 rows and 2·10⁷ elements, both required, no width limit — sent
20 000 × 1 024 (0.45×–0.56×) and 200 000 × 128 (0.80×–0.86×) to MLX and kept
1 000 000 × 16 (2.37×–2.47×) from it.

### The published logistic cases

Forced onto MLX, 25 interleaved pairs, scikit-learn 1.9.0; the multinomial targets are
the benchmark's well-separated blobs:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 20 000 × 16 | 5.21 | 3.31 | **0.63×** | 0.71× | scikit-learn |
| 200 000 × 16 | 16.46 | 27.00 | 1.64× | 1.74× | MLX |
| 1 000 000 × 16 | 59.84 | 148.31 | 2.48× | 2.67× | MLX |
| 200 000 × 128 | 44.38 | 38.94 | **0.88×** | 0.92× | scikit-learn |
| 20 000 × 1024 | 40.13 | 20.51 | **0.51×** | 0.53× | scikit-learn |
| 100 000 × 1024 | 133.34 | 94.20 | **0.71×** | 0.73× | scikit-learn |
| 20 000 × 64, K = 5 | 13.87 | 12.75 | **0.92×** | 1.52× | scikit-learn |
| 60 000 × 64, K = 5 | 25.20 | 54.50 | 2.16× | 3.72× | MLX |
| 200 000 × 64, K = 5 | 44.88 | 111.79 | 2.49× | 2.90× | MLX |

## Reading these numbers honestly

- Below a threshold, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path. The calibration and kernel tables show those cases so that the
  thresholds can be checked, not as claims.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name.
- The neighbor numbers are for k = 10 on Gaussian data; `SVC`'s for the data above at C
  from 0.1 to 100. Neither k nor C is part of any rule.
