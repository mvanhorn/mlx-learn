# Native Metal command-loop optimization — 2026-10-26

This continues the [previous command-loop experiment](metal-command-benchmarks.md).
The development implementation submits bounded groups of binary logistic L-BFGS
steps directly to Metal, while retaining the existing MLX preparation and direct
objective/gradient verification. It targets the 20,000 × 1,024 and weighted cases;
the report separates final source-audited comparisons from earlier exploratory
measurements.

## Execution and ownership

For eligible inputs, the solver evaluates and synchronizes the MLX producer before
exporting its buffers through DLPack. The native loop uses one per-fit aligned
scratch arena, reuses the existing Metal kernel expressions, and keeps its optimizer
state and history on the GPU. Losses and the first candidate gradient are computed
together. Pipeline objects are cached by specialized source with a bounded cache;
training inputs and fitted states are not cached between fits.

The loop encodes at most eight initial steps in its first command buffer, checks
convergence, then encodes at most one further step and subsequently bounded groups
of four. This avoids seven redundant inactive rounds when a weighted fit converges
on its ninth accepted step. A command-buffer boundary may occur sooner if the
iteration budget is smaller. The selected model must still pass the direct
objective and gradient check with the requested tolerance. A rejected initial
search batch, insufficient iteration budget or failed direct certification returns
to the established MLX solver, restarting from its initial state; this preserves
its result contract but can repeat work.

The optional extension is built for Apple Silicon with the CPython stable ABI and
Metal/Foundation frameworks. The library retains its existing solver when the
extension is absent, when there is no eligible prepared input, or when the supported
DLPack format is unavailable. Currently the command loop is limited to prepared
binary intercept fits with 512–4,096 features, history size 10 and a search budget
of at least eight. Normal estimator preparation supplies such inputs for matrices
with at least 4,194,304 elements. The command-loop path is enabled only for stable
MLX 0.32.x releases: MLX 0.29–0.31 can report a Metal device while exporting a
host pointer, so those versions and unknown, prerelease or later versions use the
existing solver. This is a conservative format gate, not a change to the package's
MLX dependency range. The distinction is visible in the
[MLX 0.31 DLPack export](https://github.com/ml-explore/mlx/blob/v0.31.0/python/src/convert.cpp#L103-L160)
and [MLX 0.32 export](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/convert.cpp#L327-L349).

Each submission registers the six exported DLPack capsules **and** the six source
MLX arrays. The module-instance registry publishes a strong reference to the command
buffer before committing it. Normal completion releases the owners under the GIL;
if wait or status queries fail, the registry keeps them until a later call can
observe terminal completion. Python decrefs occur outside the registry lock, and
Metal completion handlers do not invoke Python. A registry belongs to one Python
module instance, preventing a later interpreter from reaping another interpreter's
owners. If the module is torn down while completion remains unobservable, the
whole registry, including the command buffer and owners, is intentionally retained
until process exit. A permanently unobservable submission can therefore retain
memory; this conservative exceptional behavior avoids allowing in-flight storage
to be reused.

## Latest targeted complete-fit comparison

After the packaging metadata changed, the targeted panel was rerun with **30
interleaved fresh fits** per variant after three warmups. All cases use 20,000 ×
1,024 inputs and `tol=1e-4`. Speedup columns are ratios of medians; means and
paired wins retain evidence of timing variation.

| Input | Previous median ms | Native median ms | sklearn median ms | vs previous | vs sklearn | Previous mean ms | Native mean ms | Native paired wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float64, unweighted | 9.108 | 9.716 | 19.714 | 0.94× | 2.03× | 11.374 | 11.726 | 16/30 |
| float64, weighted | 14.174 | 13.805 | 21.941 | 1.03× | 1.59× | 14.807 | 13.783 | 23/30 |
| float32, unweighted | 6.925 | 6.555 | 12.829 | 1.06× | 1.96× | 6.942 | 7.047 | 17/30 |
| float32, weighted | 7.806 | 6.921 | 13.944 | 1.13× | 2.01× | 8.671 | 7.496 | 29/30 |

The float32 unweighted median improved about **5.6%** in this panel, while its
mean regressed about **1.5%**; float64 unweighted regressed on both medians and
means. The weighted float32 improvement was more consistent, with 29 of 30
paired wins. These are synthetic-data results on a shared machine, so the earlier
panel below remains part of the evidence rather than being overwritten.

All **120/120** optimized fits completed the native loop and passed independent
float64 checks without warnings. The maximum independent gradient was below
**7.337e-5**, the largest relative objective increase versus sklearn below
**9.959e-7**, and prediction agreement at least **99.965%**. The audit passed
all 84 source, reference and loaded-extension-binary integrity checks. The
[latest targeted raw results](../benchmarks/results/native-loop-packaged-targeted-2026-10-26.json)
contain every timing, order and per-fit score.

## Earlier targeted complete-fit comparison

The final targeted panel measured 30 interleaved fresh fits per variant after
three warmups, with the complete frozen previous package and unpatched sklearn.
All cases use 20,000 × 1,024 inputs and `tol=1e-4`. The speedup columns are
ratios of medians; means and paired wins expose additional timing variability.

| Input | Previous median ms | Native median ms | sklearn median ms | vs previous | vs sklearn | Previous mean ms | Native mean ms | Native paired wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float64, unweighted | 9.593 | 8.747 | 19.860 | 1.10× | 2.27× | 11.628 | 10.758 | 19/30 |
| float64, weighted | 12.494 | 10.810 | 21.729 | 1.16× | 2.01× | 13.959 | 11.716 | 25/30 |
| float32, unweighted | 6.668 | 6.639 | 12.829 | 1.00× | 1.93× | 7.031 | 6.991 | 19/30 |
| float32, weighted | 8.948 | 7.980 | 14.130 | 1.12× | 1.77× | 10.688 | 9.497 | 26/30 |

The unweighted float32 median gain over the frozen previous implementation is
only about 0.4%; the weighted cases show larger gains in this run. All **120/120**
optimized fits completed the native loop and passed the per-fit independent
float64 quality audit; no optimized fit warned. Across those fits the maximum
independent gradient was below **7.337e-5**, the largest relative objective
increase versus sklearn below **9.959e-7**, and prediction agreement at least
**99.965%**. The source and loaded-extension-binary integrity audit passed.
See the [targeted raw results](../benchmarks/results/native-loop-final-targeted-2026-10-26.json)
for every fit, ordering and score. Results were obtained on a shared M4 Max and
are synthetic-workload measurements, not a general speed guarantee.

The full suite passed: **646 tests, 41 warnings, 129.45 seconds**.

## Latest full-panel comparison

After the packaging metadata update, the full panel was rerun with 30
interleaved fits per variant after three warmups. Its default-tolerance audit
**fails** only on the two marked 4,096-feature objective-parity rows; no
results were filtered. The source and loaded-binary integrity audit passed
all 84 entries.

| Input | Rows × features | Previous ms | Native ms | sklearn ms | vs previous | vs sklearn | Native loops |
|---|---|---:|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 6.760 | 5.489 | 11.964 | 1.23× | 2.18× | 30/30 |
| float64 | 20,000 × 1,024 | 9.777 | 10.345 | 19.932 | 0.95× | 1.93× | 30/30 |
| float64 | 60,000 × 1,024 | 18.545 | 19.259 | 56.267 | 0.96× | 2.92× | 30/30 |
| float64 | 40,000 × 2,048 | 25.540 | 23.854 | 69.283 | 1.07× | 2.90× | 30/30 |
| float64 | 20,000 × 4,096 † | 44.023 | 43.967 | 120.232 | 1.00× | 2.73× | 30/30 |
| float64 | 20,000 × 1,024 weighted | 12.319 | 12.101 | 21.775 | 1.02× | 1.80× | 30/30 |
| float64 | 500,000 × 32 | 16.299 | 17.719 | 80.018 | 0.92× | 4.52× | 0/30 |
| float32 | 20,000 × 512 | 4.338 | 4.833 | 8.015 | 0.90× | 1.66× | 30/30 |
| float32 | 20,000 × 1,024 | 7.649 | 7.541 | 12.995 | 1.01× | 1.72× | 30/30 |
| float32 | 60,000 × 1,024 | 13.477 | 14.665 | 35.922 | 0.92× | 2.45× | 30/30 |
| float32 | 40,000 × 2,048 | 19.030 | 19.407 | 41.630 | 0.98× | 2.15× | 30/30 |
| float32 | 20,000 × 4,096 † | 37.405 | 37.228 | 65.039 | 1.00× | 1.75× | 30/30 |
| float32 | 20,000 × 1,024 weighted | 8.142 | 7.278 | 14.114 | 1.12× | 1.94× | 30/30 |
| float32 | 500,000 × 32 | 14.232 | 14.603 | 76.133 | 0.97× | 5.21× | 0/30 |

All **360 wide fits** completed the command loop, while 60 tall-control fits
used the existing MLX solver. Every optimized fit reported gradient convergence
without warnings. Independent float64 quality was checked on the last fit in
each full-panel row; the latest targeted and strict panels check every optimized
fit. See the [latest full raw results](../benchmarks/results/native-loop-packaged-full-2026-10-26.json)
for every measured fit, ordering, route and audit result.

The latest full-panel medians regressed on float32 **60k × 1,024** from
**13.477 to 14.665 ms** and on **40k × 2,048** from **19.030 to 19.407 ms**.
At **20k × 512**, the float32 median rose from **4.338 to 4.833 ms**, although
the mean fell from **5.899 to 5.572 ms**, reflecting its wide timing spread.
The unweighted 20k × 1,024 float64 median also regressed, from **9.777 to
10.345 ms**. That case is particularly variable: its native median was
**9.716 ms** in the latest targeted panel and **16.015 ms** in the earlier
full panel. Full-panel interquartile ranges and means for the target were:

| Input / implementation | Q1 ms | Median ms | Q3 ms | Mean ms |
|---|---:|---:|---:|---:|
| float64 / previous | 8.714 | 9.777 | 12.833 | 11.081 |
| float64 / native | 8.727 | 10.345 | 15.301 | 11.714 |
| float64 / sklearn | 19.748 | 19.932 | 20.077 | 20.151 |
| float32 / previous | 6.776 | 7.649 | 8.435 | 8.150 |
| float32 / native | 6.905 | 7.541 | 8.502 | 7.843 |
| float32 / sklearn | 12.846 | 12.995 | 13.543 | 13.188 |

In the latest full panel, both the previous and native solvers took seven
accepted steps in every unweighted 60k × 1,024 and 40k × 2,048 fit. A separate
fixed-eight versus adaptive-scheduling diagnostic checked 32 fits per schedule,
shape and dtype; both schedules reported seven iterations and produced bit-identical
coefficient and intercept fingerprints. Both therefore encoded the same first
eight-step chunk. The adaptive schedule's subsequent one- and four-step chunks
cannot explain these larger-case regressions; their cause remains unresolved.
The diagnostic's timing order was confounded, so its timings are not used as
evidence of a speed difference. Its source and binary hashes were captured
after the run rather than audited throughout. See the [raw diagnostic](../benchmarks/results/native-loop-large-chunk-ab-2026-10-26.json)
and [source archive](../benchmarks/reference/native-loop-large-chunk-ab-2026-10-26.zip).

The full and targeted panels are separate balanced runs on a non-exclusive
machine. Their full distributions and earlier panels remain available; these
speedups should not be treated as invariant across workloads or runs.

The latest strict 4,096-feature panel used `tol=1e-5` for all three solvers
and passed its per-fit and source audits:

| Input | Previous ms | Native ms | sklearn ms | vs previous | vs sklearn |
|---|---:|---:|---:|---:|---:|
| float64 / 20,000 × 4,096 | 60.645 | 60.552 | 171.780 | 1.00× | 2.84× |
| float32 / 20,000 × 4,096 | 54.148 | 53.759 | 91.117 | 1.01× | 1.69× |

All **60/60** strict optimized fits completed the native loop and passed
independent float64 checks: maximum gradient **9.828e-6**, maximum relative
objective increase versus sklearn **1.861e-6**, and prediction agreement at
least **99.98%**. See the [latest strict raw results](../benchmarks/results/native-loop-packaged-strict-4096-2026-10-26.json).

## Earlier full-panel comparison

Before the packaging metadata update, this full panel used 30 interleaved fits
per variant after three warmups.
Its default-tolerance quality audit **fails** only on the two marked
4,096-feature objective-parity rows discussed below; no rows were removed.
The source and loaded-binary integrity audit passed for all 84 entries.

| Input | Rows × features | Previous ms | Native ms | sklearn ms | vs previous | vs sklearn | Native loops |
|---|---|---:|---:|---:|---:|---:|---:|
| float64 | 20,000 × 512 | 9.287 | 8.965 | 12.198 | 1.04× | 1.36× | 30/30 |
| float64 | 20,000 × 1,024 | 12.196 | 16.015 | 19.968 | 0.76× | 1.25× | 30/30 |
| float64 | 60,000 × 1,024 | 18.452 | 18.854 | 56.432 | 0.98× | 2.99× | 30/30 |
| float64 | 40,000 × 2,048 | 24.522 | 25.037 | 69.360 | 0.98× | 2.77× | 30/30 |
| float64 | 20,000 × 4,096 † | 44.619 | 43.948 | 120.507 | 1.02× | 2.74× | 30/30 |
| float64 | 20,000 × 1,024 weighted | 14.716 | 10.736 | 21.916 | 1.37× | 2.04× | 30/30 |
| float64 | 500,000 × 32 | 16.459 | 16.250 | 79.663 | 1.01× | 4.90× | 0/30 |
| float32 | 20,000 × 512 | 3.769 | 3.514 | 7.907 | 1.07× | 2.25× | 30/30 |
| float32 | 20,000 × 1,024 | 7.586 | 7.305 | 12.987 | 1.04× | 1.78× | 30/30 |
| float32 | 60,000 × 1,024 | 14.011 | 14.060 | 35.923 | 1.00× | 2.56× | 30/30 |
| float32 | 40,000 × 2,048 | 19.456 | 18.664 | 41.564 | 1.04× | 2.23× | 30/30 |
| float32 | 20,000 × 4,096 † | 37.374 | 37.501 | 65.133 | 1.00× | 1.74× | 30/30 |
| float32 | 20,000 × 1,024 weighted | 8.040 | 7.754 | 13.993 | 1.04× | 1.80× | 30/30 |
| float32 | 500,000 × 32 | 14.754 | 15.413 | 75.956 | 0.96× | 4.93× | 0/30 |

All **360 wide fits** completed the command loop; the 60 tall-control fits
used the existing MLX solver. Every optimized fit reported gradient convergence
without warnings. Independent float64 quality was checked on the last fit in
each full-panel row; the separately reported targeted and strict panels check
every optimized fit. The [full raw results](../benchmarks/results/native-loop-final-full-2026-10-26.json)
retain all timings, orderings, route details, scores and failures.

Timing variation is substantial. In the full panel the unweighted 20k × 1,024
float64 native median was **16.015 ms**, versus **12.196 ms** for the previous
solver, a regression; the separate targeted panel measured **8.747 ms** versus
**9.593 ms**. Their interquartile ranges and means in the full panel were:

| Input / implementation | Q1 ms | Median ms | Q3 ms | Mean ms |
|---|---:|---:|---:|---:|
| float64 / previous | 9.631 | 12.196 | 16.784 | 13.294 |
| float64 / native | 10.905 | 16.015 | 16.740 | 14.153 |
| float64 / sklearn | 19.781 | 19.968 | 20.357 | 20.782 |
| float32 / previous | 6.831 | 7.586 | 8.922 | 8.543 |
| float32 / native | 6.544 | 7.305 | 8.843 | 8.412 |
| float32 / sklearn | 12.818 | 12.987 | 13.301 | 13.241 |

These differences on a non-exclusive machine limit the precision of the
speedup claims; the targeted and full panels must not be merged or selectively
filtered. The same-run comparisons and all individual samples are provided.

The strict 4,096-feature panel used `tol=1e-5` for all three solvers and
passed its full per-fit and source audits:

| Input | Previous ms | Native ms | sklearn ms | vs previous | vs sklearn |
|---|---:|---:|---:|---:|---:|
| float64 / 20,000 × 4,096 | 60.783 | 60.230 | 171.817 | 1.01× | 2.85× |
| float32 / 20,000 × 4,096 | 54.210 | 53.710 | 90.982 | 1.01× | 1.69× |

All **60/60** strict optimized fits completed the command loop and passed
independent float64 checks: maximum gradient **9.828e-6**, maximum relative
objective increase versus sklearn **1.861e-6**, and prediction agreement
at least **99.98%**. See the [strict raw results](../benchmarks/results/native-loop-final-strict-4096-2026-10-26.json).

## Exploratory scheduling and cold-fit measurements

The [scheduling experiment](../benchmarks/results/native-loop-adaptive-schedule-2026-10-26.json)
used 32 balanced, interleaved warmed fits per variant on synthetic 20k × 1,024
inputs. The fixed eight-step loop (`mode0`) and the adaptive eight/one/four loop
(`mode1`) produced these medians in milliseconds:

| Input | Fixed eight steps | Adaptive | Frozen previous | sklearn |
|---|---:|---:|---:|---:|
| float32, unweighted | 6.007 | 6.035 | 6.449 | 13.009 |
| float32, weighted | 7.758 | 6.749 | 7.692 | 14.479 |
| float64, unweighted | 8.229 | 8.280 | 8.441 | 20.315 |
| float64, weighted | 10.555 | 9.219 | 10.477 | 22.447 |

These exploratory figures show why the weighted scheduling change was selected;
they are not final library results. The experiment used scratch code and did not
record the loaded extension binary hash, so source/binary correspondence at
measurement time was not independently verified. The full sample order, fit
metadata and final-fit numerical scores are retained in the JSON. The
[source-only experiment archive](../benchmarks/reference/native-loop-scheduling-experiments-2026-10-26.zip)
and its adjacent manifest document the source variants and reproduction method.

Three fresh-process first-fit probes per variant included shader compilation but
started after package import, data creation and GPU synchronization. Their
unweighted float32 medians were **59.519 ms** for the adaptive prototype and
**30.761 ms** for the frozen previous solver; weighted medians were **61.389 ms**
and **35.026 ms**, respectively. The calls were not interleaved, the sample is
small, and this does not measure full process startup. It does show a meaningful
cold-start cost that warm-fit results do not capture. Raw samples are in the
[first-fit record](../benchmarks/results/native-loop-first-fit-2026-10-26.jsonl).

A native initial-statistics reduction and removal of the extra synchronization
after MLX evaluation were also tried; their complete-fit results were mixed and
neither change was retained. The explicit producer synchronization remains in
the current development implementation. See the
[initialization](../benchmarks/results/native-loop-native-initialization-2026-10-26.json)
and [synchronization](../benchmarks/results/native-loop-eval-synchronization-2026-10-26.json)
records and the experiment archive for details.

## Accuracy scope

The direct objective and centered gradient check remains mandatory before a
native result is returned. The benchmark's independent checks compute the
original-coordinate objective and gradient in float64 outside the timed region,
and compare predictions with stock sklearn; the optional `--audit-every-fit`
checks every optimized fit rather than only the last. No tolerance or accuracy
threshold is relaxed.

The previously measured 4,096-feature default-tolerance panel has an objective
gap above the benchmark's `1e-5` relative threshold for both the earlier solver
and the earlier command-loop prototype. The previous strict panel passes with
`tol=1e-5` for all implementations. Failed rows remain in the raw output.

An offset design with features shifted by 1,000 exposes a **pre-existing
original-coordinate convergence limitation**. On the same 20,000 × 1,024 input
at the normal estimator thresholds, the frozen previous solver reported a
centered-coordinate maximum gradient of **8.87370e-5**, and the new command loop
reported **8.87368e-5**, both below `tol=1e-4`. The independent float64 maximum
gradients in the original coordinates were **0.0209144** and **0.0209156**,
respectively. Their independent objective values differed by less than
**2.5e-11** in absolute terms. The raw-feature gradient includes the feature
mean multiplied by the intercept gradient, so convergence after centering does
not imply the same absolute gradient threshold in the original coordinates.

This probe does not establish prediction parity with sklearn: both MLX models'
float64-logit predictions agreed with sklearn on **50.01%** of rows. Sklearn's
objective was approximately **0.6931**, versus **0.2913** for both MLX models;
its own independent original-coordinate gradient was approximately **0.3569**.
These numbers describe this difficult offset fixture and do not establish which
predictions are correct. The earlier smaller edge probe, which forced preparation
below its usual threshold, likewise showed original-coordinate gradients above
tolerance. The [same-data baseline comparison](../benchmarks/results/native-loop-offset-baseline-2026-10-26.json)
records both cases, independent scores, routes and input hashes; its
[probe source](../benchmarks/offset_logistic_probe.py) is available separately.
This is a diagnostic comparison, not a passing benchmark quality gate.

## Reproduction

On Apple Silicon with the compatible MLX environment and the macOS compiler/SDK,
build the optional extension in the checkout before running the benchmark:

```bash
python3 setup.py build_ext --inplace --force
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --case wide_20k_1024 --case wide_weighted --dtype both \
  --repeats 30 --warmup 3 --threads 1 --require-native-loop --audit-every-fit \
  --output bench/native-loop-targeted.json --report bench/native-loop-targeted.md
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1 \
  --output bench/native-loop-full.json --report bench/native-loop-full.md
python3 -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --profile full --case wide_20k_4096 --dtype both --tol 1e-5 \
  --repeats 30 --warmup 3 --threads 1 --require-native-loop --audit-every-fit \
  --output bench/native-loop-strict-4096.json --report bench/native-loop-strict-4096.md
python3 -m benchmarks.offset_logistic_probe \
  --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
  --output bench/native-loop-offset-baseline.json
python3 -m pytest -q
```

The default full panel is expected to exit nonzero if the known 4,096-feature
objective gap remains. The strict panel applies the same `tol=1e-5` to the
frozen previous solver, native loop and sklearn. Use `--require-native-loop` on
eligible targeted cases to ensure fallback cannot be reported as a command-loop
speedup; the full panel includes a tall control outside the native loop's scope.
Benchmark data generation and independent accuracy scoring occur outside the
timers; estimator construction, preprocessing, optimization, verification and
synchronization are included. The machine, thread-pool limits, source hashes,
compiled extension hash, execution routes and all measured samples are recorded
in JSON.
