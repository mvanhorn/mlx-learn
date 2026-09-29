# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Mostly correctness. Exact neighbor search is certified rather than trusted to a fixed
margin; float64 input keeps the offsets a float32 cast used to round away; values float32
cannot hold are rescaled exactly or refused; and every fit and every public inference call
records exactly one diagnostics event. The changes existing code is most likely to notice
come first under **Changed**.

### Fixed

#### Neighbors

- Exact nearest-neighbor queries (`NearestNeighbors`, `KNeighborsClassifier`,
  `KNeighborsRegressor`) could silently return wrong neighbor sets on clustered,
  near-duplicate, multi-modal or unscaled data, on the CPU device too: a fixed float32
  over-selection margin was never checked. Every query is now certified by a rigorous
  float32 error bound; rows the bound cannot certify get a wider second pass, then an
  exact on-device search ([`docs/fp32_policy.md`](docs/fp32_policy.md) §2.1).
- Neighbor distances were refined from mean-translated float32 coordinates, which lost
  precision for data far from its own mean. Refinement now uses the input's own
  coordinates.
- float64 input was cast to float32 before centering, so large offsets such as Unix
  timestamps or projected coordinates were quantized away. The index now translates by the
  float64 mean — carried, like each row, as a float32 value plus a float32 residual — so
  data whose spread is comparable to a float32 rounding of its mean (timestamps a few
  seconds apart) certifies on the first pass, where a float32-rounded mean sent every
  query to the second. Such data now costs what data without the offset does: 37 ms for
  4096 queries against 50 000 rows (two features, spread 5, around 1.7e9 or around 0;
  see **Performance** for how it was measured).
- Rows tied at the k-th distance now follow the documented `(distance, index)` order:
  `kneighbors(k)` is a prefix of `kneighbors(k + m)`, results no longer depend on block
  sizes, and GPU and CPU agree wherever distances are exact (binary, integer and one-hot
  features). `kneighbors()` with no query follows the same order with each row left out,
  so a row with more than `k` exact duplicates before it gets the first `k` of them, and
  `predict(None)` does not change with `k`. 0.1.0a1 described this order, but on
  tie-heavy data it did not hold.
- Data whose squares overflow or underflow float32 (`|x|` around 1e19 or 1e-20) no longer
  yields infinite or zero distances and arbitrary neighbors; it is scaled by an exact
  power of two. Training data spanning more than float32's largest value / 8 from its
  mean (a single 1e38, say), or so wide beside the gaps between its rows that no scale
  holds both (a 1e30 fill value among unit-scale rows), raises `UnsupportedInputError`
  (`"float32-overflow"`) at `fit` directly imported and falls back to scikit-learn under
  `patch_sklearn()`. Queries still out of range raise `UnsupportedInputError`, and so does
  a query nearer a training row than the scale chosen at `fit` resolves, instead of getting
  a distance of zero; an exact duplicate of a training row is exempt.
- One unmasked fill value among unit-scale rows, such as CMIP's 1e20, made the other rows'
  neighbor distances zero and their neighbor sets wrong, on both devices and without an
  error. The index scales a large span down only as far as float32 needs — ordinary data
  is not scaled at all — so such a value is now served on both devices, through the exact
  search of [`docs/fp32_policy.md`](docs/fp32_policy.md) §2.1, and one float32 cannot
  resolve beside the gaps between the other rows (1e30, or NetCDF's default 9.97e36) is
  refused at `fit` as above.
- Concurrent first queries on an unpickled neighbors model could segfault or raise
  `BackendExecutionError`. The device state is now built, evaluated and published once,
  under a per-model lock.
- Constructor parameters are validated on every path, so `n_neighbors=0` and invalid
  `weights`, `algorithm` or `p` raise scikit-learn's `InvalidParameterError`. Before, they
  silently "fitted" and predicted `classes_[0]` or NaN, in both layers.
- `KNeighborsRegressor` with a column-vector `y` of shape `(n, 1)` predicts
  `(n_queries, 1)` without a `DataConversionWarning`, as scikit-learn does. Before, the
  shape under `patch_sklearn()` depended on the size of the data.
- `KNeighborsRegressor` copies `y`: writing into a float64 `y` after `fit` no longer
  changes the predictions.
- String arrays in `X`, in queries and, on the MLX path, in `KNeighborsRegressor`'s `y` are
  rejected with scikit-learn's error instead of being parsed as numbers (`"1.5"` as 1.5).
  Under `patch_sklearn()`, whether string targets worked used to depend on the size of the
  data.
- `kneighbors` reports the `n_neighbors` bound error with scikit-learn's full message
  (`n_samples = …`), and only after validating the query.
- Under `patch_sklearn()`, a model fitted on MLX answers a sparse query to `kneighbors`,
  `kneighbors_graph`, `predict` or `predict_proba` with scikit-learn's brute-force search
  instead of raising `SparseInputError`. The query is cast to float32 to match the
  training copy, so the search runs at stock speed and its distances carry that copy's
  float32 rounding; a sparse value float32 cannot hold raises `UnsupportedInputError`, as
  a dense one does. This is recorded as a fallback and honors `fallback_policy`.
  `radius_neighbors` and `radius_neighbors_graph` take a sparse query in both layers as
  ordinary calls.
- Under `patch_sklearn()`, a prebuilt `KDTree`, `BallTree` or fitted estimator passed as
  `X` falls back to scikit-learn instead of raising `TypeError`.
- `effective_metric_` is set as scikit-learn sets it (`metric="l2"` stays `"l2"`).

#### LogisticRegression

- Features around 1e8 and above produced the null model (`coef_ = 0`, `n_iter_ = 0`): the
  line search had an absolute 1e-12 step floor. The floor is now relative, the
  curvature-pair test is scale-invariant, and a diagonal preconditioner built from the
  column scales handles features in large units and mixed-scale designs.
- A small `tol` (1e-6, 1e-8), or features scaled by 100 or more, ended in
  `line-search-failure` and a `ConvergenceWarning` where scikit-learn converged. The line
  search now accepts approximate-Wolfe steps. A `tol` below the float32 floor (`tol=0`,
  say) stops with `"float32-floor"`, recorded in the diagnostics rather than warned about
  ([`docs/fp32_policy.md`](docs/fp32_policy.md) §2.3). Only a `max_iter` stop advises
  increasing `max_iter`.
- float64 input is centered in float64 before it is rounded to float32, so large offsets
  such as timestamps no longer lose precision. The intercept is un-centered with exactly
  the mean that was subtracted.
- Sample weights of about 1e36 and above no longer overflow float32 into a zero model; they
  are rescaled by an exact power of two.
- Negative `sample_weight` under `patch_sklearn()` now falls back to scikit-learn above
  the crossover as well, instead of raising.
- A `LogisticRegression` built by assigning `coef_`, `intercept_` and `classes_` now
  predicts, in both layers.
- Regression targets get scikit-learn's exact "Unknown label type" message.
- `ConvergenceWarning` messages start with scikit-learn's `"lbfgs failed to converge after
  N iteration(s)"`, so message filters and `pytest.warns(match=...)` written for
  scikit-learn match on the MLX path too.
- A `ConvergenceWarning` that a warnings filter escalates to an error leaves the estimator
  unfitted, with no `fit_event_`; the diagnostics record keeps that fit's decision and
  its stop. Before, it stayed fitted and was recorded as a completed fit.
- `MLXLEARN_FORCE_BACKEND=mlx` on a host without a usable GPU fits on MLX's CPU, as the
  neighbor estimators and `SVC` do. Before, it raised a `DeviceError` claiming
  `device="gpu"` had been requested. An explicit `device="gpu"` still raises.
- An exception raised by a scikit-learn 1.9 callback is re-raised as itself, with its own
  `__cause__` and `__context__`, as scikit-learn re-raises it, not reported as a
  `BackendExecutionError`; the failed fit's device copy of `X` is not kept alive in a
  reference cycle until the next garbage collection.
- Under `patch_sklearn()`, where a fit ran no longer depends on how its labels are
  spelled. The size rules counted string and object labels — a pandas `Series` of class
  names, a categorical — as binary, so a multinomial fit that integer labels sent to MLX
  went to scikit-learn under the binary rule. Labels are now counted whatever their type,
  binary ones in two vectorized passes without a sort. `SVC`'s many-class rule reads them
  the same way before validation, where it used to wait for it.

#### SVC

- `gamma="scale"` falls back to `gamma=1` only for an exactly-zero variance, as
  scikit-learn does. It did so for any variance up to 1e-12, which gave a model at chance
  on data with features around 1e-6 or smaller.
- Kernel inputs are mapped exactly in float64 before the float32 cast: the rbf training
  mean is subtracted first, so large offsets are no longer rounded away, and an exact
  power-of-two scale is folded into `gamma`, so data around 1e±19, and `gamma` values
  outside float32's range, match scikit-learn instead of returning a NaN model. The
  power-of-two scale is exact, so it leaves results for data already in range
  bit-identical; the rbf translation does not, because the mean is now subtracted before
  rounding, and moved in-range rbf decision values by up to 6e-4 against 0.1.0a2 (on
  `N(3, 1)` data), within the solver's `tol`.
- rbf entries between rows close to each other but far from the training mean came out of
  float32's Gram identity wrong by up to 0.58: tight clusters far apart with `gamma`
  matched to them, or an explicit `gamma` on unscaled multi-modal data. Fits returned
  decision values 3–6 away from scikit-learn's, silently, under `patch_sklearn()` too.
  Those entries are now recomputed on the host in float64 — in the Gram, in the rows
  evaluated on demand and at inference — so two clusters 2 000 apart match scikit-learn to
  1.1e-3 and 40 clusters at 2 400 × 512 to 3.1e-4. Ordinary data never takes that path
  ([`docs/fp32_policy.md`](docs/fp32_policy.md) §2.4).
- The SMO solver takes its curvature from the closed-form kernel diagonal; every kernel
  row it reads now carries that value in its self-entry, so its steps and its gradient
  updates describe one quadratic. A poly fit on offset data used to circle its whole
  10⁷-iteration budget: 376 s against scikit-learn's 0.001 s. In-range fits can move
  within `tol`: up to 2.3e-3 in poly decision values measured, and 1.5e-2 in a linear one.
- `linear` and `poly` fits on data whose variation float32's inner product cannot resolve —
  a large offset with a small spread, or rows that all round to the same nonzero row —
  raise `UnsupportedInputError` (reason `"float32-cancellation"`), which `patch_sklearn()`
  hands to scikit-learn. Such fits used to return models up to 13 points less accurate,
  silently.
- A class whose every sample has zero effective weight raises `UnsupportedInputError`
  (reason `"empty-class"`, still a `ValueError`) before any device work, so
  `patch_sklearn()` hands the fit to scikit-learn, which fits the remaining classes. It
  was a plain `ValueError` in both layers.
- A fit whose float32 kernel overflowed returned a model with a NaN intercept and
  `fit_status_=0`. It now raises scikit-learn's "dual coefficients or intercepts are not
  finite" `ValueError` as soon as a one-vs-one pair stops on a non-finite value. The SMO
  solver no longer reports convergence when non-finite gradients leave it no working
  pair, checks every selection and its final gradient and dual variables, and no longer
  spins to the iteration budget on an infinite kernel diagonal.
- These inputs now raise a `CapabilityError` at `fit`, which `patch_sklearn()` hands to
  scikit-learn: a `gamma` or `coef0` float32 cannot represent, a linear-kernel inner
  product that overflows float32, and float64 or object-dtype input past 3.4e38.
- `predict` and `decision_function` raise `UnsupportedInputError` for queries float32
  cannot represent, instead of returning NaN decision values and silently flipped labels.
  A query that float32 holds but that lies far outside the training data — a CMIP `1e20`
  or netCDF `9.97e36` fill value — is still answered as scikit-learn answers it, on the
  rbf, sigmoid and linear kernels; only one whose inner product with a support vector
  float32 cannot hold raises.
- `SVC(kernel="linear", gamma=0.0)` fits, as in scikit-learn.
- Negative `sample_weight` values are treated as zero weights, exactly as LIBSVM treats
  them, so patched `SVC` no longer raises where stock scikit-learn fits.
- A sparse query on a dense-fitted `SVC` raises scikit-learn's `ValueError` instead of a
  `TypeError`.
- The `max_iter=-1` `ConvergenceWarning` names the budget mlxlearn resolved `-1` to,
  instead of the contradictory "max_iter=-1".
- `SVC(max_iter=0)` records the KKT violation at its initial iterate in its private
  diagnostics, instead of an unmeasured 0.0.
- The SMO gradient update ran in float32 on NumPy 1.x (value-based casting), losing
  precision against the float64 gradient. It is float64 on every NumPy, and bit-identical
  on NumPy 2.
- Internal kernel-row gathers are bounds-checked; an MLX gather reads past the end
  silently.

#### All estimators

- `mx.array`, torch tensors and other non-NumPy array-likes crashed every estimator with
  `AttributeError`, in both layers. The float32 range check now reads them the way
  scikit-learn does.
- float64 values beyond float32's range are caught in object-dtype arrays and in pandas
  frames that mix dtypes (float64 with bool or nullable columns, for example): Layer 1
  raises `UnsupportedInputError` and `patch_sklearn()` falls back. Before, KNN and
  `LogisticRegression` raised a plain `ValueError`, and `SVC` silently fitted a
  NaN-intercept model.
- A pandas `DataFrame` whose every column is sparse is detected as sparse input: Layer 1
  raises `SparseInputError`, and patched mode falls back instead of raising
  scikit-learn's `TypeError`.
- mlxlearn's exceptions survive pickling. `cross_val_score(..., n_jobs=2,
  error_score="raise")` surfaces `UnsupportedParameterError` instead of
  `BrokenProcessPool`.
- Diagnostics record exactly one event per fit, matching `execution_backend_`: the
  phantom `fit -> mlx` event on below-crossover fits is gone, and patched sub-crossover
  and `tree-wins` fits record one event instead of two; a `LogisticRegression` fit that
  stops without converging records its stop on that event (see **Added**). The event is
  recorded when the fit decides, and a later decision in the same fit takes its place: a
  fit that fails after deciding is on the record once — `get_last_backend_event()`
  describes it, not an older call — and leaves the estimator unfitted. A patched
  estimator's fallback warning comes with the decision, before scikit-learn does the
  work. One public inference call records one event, named after the method called —
  `predict`, `predict_proba`, `predict_log_proba`, and scikit-learn's inherited `score`,
  `kneighbors_graph` and `radius_neighbors_graph` included, which were recorded under the
  inner method's name — whichever backend fitted the model, through one mechanism shared
  by every estimator; a patched `SVC(probability=True)`, always fitted by scikit-learn,
  used to record none for `predict_proba` and `predict_log_proba`. A call on one
  estimator made from inside another's, such as a `weights=` callable that queries a
  second model, records its own. A patched neighbor estimator's sparse-query fallback is
  that call's one event. A fit forced to scikit-learn with `MLXLEARN_FORCE_BACKEND=sklearn`
  is recorded with `fallback=True`.
- A refit that falls back to scikit-learn, or fails, no longer keeps the previous fit's
  MLX state — the neighbor index and its training copy, `SVC`'s pair weights and
  device-resident support vectors — alive in memory and in pickles.
- Inference after a refit that failed raises scikit-learn's `NotFittedError`, as it does
  on an estimator never fitted, instead of a bare `MLXLearnError`. It is not an
  `MLXLearnError`, so code that caught `MLXLearnError` there should catch
  `NotFittedError`.
- A model fitted on the MLX GPU and loaded on a host without a usable GPU predicts on
  MLX's CPU device under `device="auto"` or `"cpu"`, recorded once per call as
  `"gpu-unavailable"` (with the fitted backend, `"mlx"`), instead of raising
  `DeviceError`; kernel and distance blocks are then budgeted for the CPU, and the rbf
  tile is not compiled there. An explicit `device="gpu"` still raises, with an accurate
  message. Inference that runs on the host needs no device, so it is recorded as
  `"sticky"` and works under `device="gpu"` too: `LogisticRegression`'s, and
  `radius_neighbors` and `radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`
  (`extra["compute"] = "scikit-learn"`).
- Patched estimators honor the size crossover under `device="cpu"` and on hosts without a
  GPU: small problems go to scikit-learn instead of the float32 MLX-CPU path.
- Layer 1 estimators no longer advertise `array_api_support`. Under
  `array_api_dispatch=True`, patched `LogisticRegression` hands the fit to scikit-learn on
  scikit-learn 1.9, where the stock class advertises array-API support.
- An internal helper, `as_float32(copy=True)`, returned an alias for `np.matrix`,
  `np.memmap`, pandas and `__array__` inputs.

#### Runtime: configuration, devices, diagnostics

- `set_config` now changes the process-wide default. Worker threads used by
  scikit-learn's `n_jobs` (joblib's threading backend) see it, so
  `fallback_policy="raise"`, `device` and `diagnostics` no longer silently stop applying
  there. `config_context` stays scoped: it is a context variable, so it follows `asyncio`
  but not plain threads. Process workers start from the new `MLXLEARN_<OPTION>`
  environment defaults (see **Added**).
- Concurrent mlxlearn work on different MLX devices no longer runs on the wrong device or
  leaves the application's default device changed. MLX's default device is
  process-global; mlxlearn now serializes its switches and leaves a matching default
  untouched.
- In a process forked after MLX or scikit-learn was loaded, mlxlearn raises `DeviceError`
  (Layer 2 falls back, reason `"forked-process"`) instead of the child crashing with
  SIGSEGV.
- Fallback warnings point at the caller's line, so module-based warning filters match —
  on scikit-learn 1.9 a patched `LogisticRegression`'s too, rather than scikit-learn's
  callback decorator. A warning hidden by an `"ignore"` filter no longer uses up its
  once-per-reason slot, and an `"error"` filter raises on every fallback.
- With `device="gpu"` on a host with no usable GPU, a directly imported `SVC` raised
  `DeviceError` on problems below its crossover, and on every `kernel="linear"` fit, even
  though it runs those on MLX's CPU. Every estimator did the same for input with no shape
  before validation, such as a list. The configured device no longer overrides size
  dispatch anywhere, as documented.
- `mlx_guard` reports MLX's own `ValueError`/`TypeError` as `BackendExecutionError`, and
  describes Metal allocation failures as memory limits. The doubled period in its message
  is gone.
- `MLXLEARN_FORCE_BACKEND` and the other tuning values are validated: `gpu` and mixed case
  are rejected, and so are zero block sizes. Tuning is read from the environment once per
  process, not once per thread, and `diagnostics_capacity` no longer leaks from tests.
- `Config` accepts NumPy booleans and integers, rejects out-of-range and boolean seeds, and
  names the bad value in its error.
- Messages about sparse input, float32 range and other capability limits are worded for
  the layer that raised them, so a patched user is no longer told to run under
  `patch_sklearn()`. A patched fallback warning states the mismatch and then "mlxlearn
  handed this fit to scikit-learn."; under `fallback_policy="raise"` the error states it
  and then "fallback_policy='raise' is set, so mlxlearn did not hand this fit to
  scikit-learn…". A float32 range error at inference, which neither layer can fall back
  on, advises no fallback.

#### Patching

- Under patch-first, scikit-learn's own users of the patched names no longer pick up the
  mlxlearn classes: `DBSCAN`, `OPTICS`, `HDBSCAN`, `MeanShift`, `SpectralClustering`,
  `TSNE`, `Isomap`, `LocallyLinearEmbedding`, `SpectralEmbedding`, `LabelPropagation`,
  `mutual_info_*`, and `StackingClassifier`'s default final estimator.
  `mutual_info_classif` was about 24% low under patch-first, because the Kraskov estimator
  needs float64-exact k-th distances. `patch_sklearn()` now imports those modules before
  rebinding.
- `unpatch_sklearn()` also restores any `sklearn.*` module attribute still bound to an
  mlxlearn class, so the module attributes of scikit-learn modules first imported while
  patched are restored too. Containers such a module filled with classes as it was
  imported are not, such as the tables of scikit-learn's test instance generator: import
  `sklearn.utils.estimator_checks` before patching to run the estimator checks after
  unpatching.
- `patch_sklearn()` and `unpatch_sklearn()` complete under warnings-as-errors. The tamper
  warning is issued after every estimator has been handled, and it no longer leaves later
  estimators patched, or a stale entry behind that makes every retry fail.
- The patch and unpatch warnings point at the caller's line. The unpatch tamper warning
  used to point at the caller's caller, or `<sys>:0` at top level.
- Re-patching after `sklearn.svm` was reloaded, or restored by hand, no longer refuses
  with a misleading "third-party patch" warning.
- If mlxlearn's patch bookkeeping is lost (a module reload, `%autoreload`), mlxlearn no
  longer records its own class as the "original"; unpatching restores the stock class.
- `help()` and IDE documentation for patched classes show the full parameter
  documentation again, with the patched-variant note. Before, only the note was shown.
  The note no longer calls a problem below the crossover a capability mismatch: the
  patched class hands it to scikit-learn by design, and the directly imported class runs
  it on MLX.

#### Tooling, CI and benchmarks

- `tools/compliance.py --json` printed invalid JSON; stdout is now the JSON document alone.
- The benchmark "IQR" was the full min–max range at the default 7 iterations, so one
  outlier could hide a 2× regression. The gate also excused MLX-path regressions using a
  crossover derived from the run it was judging.
- The benchmark harness penalized mlxlearn: `compare()` timed the mlxlearn side first in
  every pair, straight after `gc.collect()`, and only that side paid
  `mx.synchronize()`. Identical work in both slots read 0.77–0.85× and failed the
  performance gate in almost every trial. Every timed call is now collected and
  synchronized the same way, and the two sides take turns going first; identical work
  reads 1.00×.

#### Documentation

- 0.1.0a1 said `LogisticRegression.multi_class` was removed in scikit-learn 1.9. It was
  removed in 1.8, the release that also made `penalty` a deprecated sentinel;
  `SVC.probability` became one in 1.9. [`docs/support_matrix.md`](docs/support_matrix.md)
  now tabulates all three per version.
- 0.1.0a1 attributed `LogisticRegression`'s per-iteration cost to one host
  synchronization per line-search evaluation. The cost was 12–33 blocking
  synchronizations per iteration in the L-BFGS bookkeeping; the line search averaged about
  1.1 evaluations per iteration. The fix is under **Performance**, and
  [`docs/benchmarks.md`](docs/benchmarks.md) describes the cost as it now is.
- [`docs/benchmarks.md`](docs/benchmarks.md) is rendered from the calibration grids (see
  **Added**), and every range the README, this changelog, `Tuning` and the dispatch
  docstrings quote cites one of its tables by grid name. Ranges it used to quote did not
  all hold: that every batch of 1–4 096 queries won above the neighbor floors (batches of
  8–512 lost, at sizes the calibration had not timed); that a KD-tree answers a single
  row faster than MLX up to 12 features (it holds up to 8, and at 12 up to 65 536 rows —
  one fixed query row had stood for what a row costs the tree); that the sigmoid kernel
  won at C = 10; an index-build range its own table contradicted; and lower bounds that
  left out one version's worse figure.
- An independent re-measurement found more that did not hold outside the data the grids
  used: that every neighbor shape patched dispatch routes to MLX wins or ties at every
  batch size (on Gaussian data only; duplicated rows, tight clusters, an unscaled column
  and narrow integer data lost, and so, at a million rows and 16–48 features, did some
  batches of clustered and uniform data — table `knn-data`, `knn-narrow`); that the SVC
  kernel and C change the iteration count but not where the width crossover lies, that
  C = 10 and C = 100 behave alike, and that from 128 features `SVC` wins at every C (all
  three at the default `gamma` and `degree` only — table `svc-params`); that the L-BFGS
  iteration counts match scikit-learn's and that multinomial fits are faster from 60 000
  rows at every width (not on well-separated classes, where scikit-learn stops after one
  iteration — table `logreg-separable`); and that binary fits win from 100 000 rows (not
  on labels without signal — table `logreg-labels`). Each is corrected, the rules that
  depended on it are changed (see **Changed**), and the grids that show it are
  committed.
- A second independent re-measurement found more: that the fit-time probe hands off the
  data its first pass cannot certify (rows repeated about twice, clusters of std 0.4–0.5
  and one column on 100 times the scale of the rest left a few of 64 probed rows
  uncertified, passed the probe, and lost mid-size batches near the size floors — the
  cost is per batch, not per row; table `knn-structured`); that k did not matter (k = 100
  and 200 lost at a million rows and at 2 048–4 096 features, wider than any grid had
  gone; table `knn-k`); that `SVC` wins from 128 features (not on labels with little or
  no signal, nor with `gamma` far above `"scale"` at C above 1; table `svc-hard`); that
  multinomial `LogisticRegression` wins from 60 000 rows at every width (not at one
  feature on labels without signal; table `logreg-labels`); and that single-row
  `LogisticRegression.predict` runs at 0.94×–1.02× of stock (see **Performance** for what
  it measures). Each is corrected, with the rule, grid and table under **Changed** and
  **Added**, and what each new rule gives up is stated beside it.
- 0.1.0a1 listed cold-start measurements among the benchmark outputs. The harness never
  implemented them; see **Removed**.
- The README's links were relative and 404ed on the PyPI page; they are absolute now. It
  also documented the private `_execution_backend_` instead of the public
  `execution_backend_` property.
- `SVC`'s `shrinking` was documented as accepted and ignored; it is now honored (see
  **Changed**). `max_iter=-1` is documented as a deliberate cap, and the `fit_status_`
  and crossover explanations are corrected.
- [`docs/fp32_policy.md`](docs/fp32_policy.md) no longer offers `device="cpu"` as a
  float64 escape hatch: it is the same float32 algorithm on MLX's CPU. It now describes
  what the SVC sigmoid and KKT tests actually assert.
- [`docs/fp32_policy.md`](docs/fp32_policy.md) §3 gave `LogisticRegression`'s probability
  tolerance, `atol=2e-3` at default `tol`, without saying it was measured on a fixed grid
  (n ≤ 5 000, d ≤ 20). At default `tol` a wide problem stops elsewhere: at
  25 000 × 1 000, which 0.1.0a2's crossover sent to MLX, the probabilities differed by up to
  4.9e-3, as much as stock lbfgs and stock newton-cg differ from each other. Such a
  problem is judged on the objective, and on probabilities at `tol=1e-6` (7.2e-5).

### Changed

#### Behavior

- **`SVC` honors `shrinking`.** It was accepted and ignored; the SMO solver now runs
  LIBSVM's shrinking heuristic when `shrinking=True`, which is scikit-learn's default, so
  default fits change path. The optimum and the stopping test are the same — the test is
  evaluated over every sample before the solver may stop — but, as in LIBSVM, where inside
  `tol` the solver stops can differ, on every kernel and by the same order as LIBSVM's
  own shrinking ([`docs/fp32_policy.md`](docs/fp32_policy.md) §2.4). The gradient of the
  samples set aside is rebuilt one row at a time in index order, as LIBSVM rebuilds it, so
  a fit does not depend on how `cache_size` batches that work or on the NumPy/BLAS build.
  `shrinking=False` never sets a sample aside.
- **Multi-output targets are accelerated** in `KNeighborsClassifier` and
  `KNeighborsRegressor`: the neighbor search does not depend on `y`, and votes and averages
  are taken per output. The Layer 1 tags now advertise multi-output (and, for the
  classifier, multi-label) targets.
- **Patched dispatch thresholds re-measured, rules reshaped, and some that read more than
  the shape.** Measured against scikit-learn 1.9.0 and 1.7.2 on an M4 Max, on grids
  `python -m benchmarks.calibrate` regenerates ([`docs/benchmarks.md`](docs/benchmarks.md)
  has every table, named by grid), each threshold now sends a problem to MLX only where
  MLX won or tied against both, in every configuration measured at that shape — for the
  neighbor estimators, every batch of 1 to 4 096 queries, k from 1 to 200 and data with
  and without structure; for `SVC`, the kernel parameters, labels with and without
  signal and data of lower intrinsic dimension too; for `LogisticRegression`, labels with
  and without signal, `tol` and `max_iter`. Where a fit cannot see what decides — how it
  will be queried, how many rows become support vectors, how many iterations its data
  take — the rules take the side that does not lose, and give up wins to do it. Where a
  corner lost, a rule gives up the region around it rather than grow another rule per
  corner: patched `SVC` sends only rbf to MLX, and a patched fit's `tol` may be no looser,
  and `SVC`'s `cache_size` no smaller, than scikit-learn's defaults, where everything
  else was measured.
  - Neighbor estimators need at least 4 096 rows **and** 6 000 000 elements (was 256 rows
    **or** 2 048 elements; table `knn-rows`): below 4 096 rows a 4096-query batch ran
    0.44×–0.79× of stock at every width from 256 to 1 024, and below 6 million elements
    batches of 1–32 queries lost at 4 096–8 192 rows (0.70×–0.95×).
  - Patched mode hands every neighbor fit with fewer than 256 features and fewer than
    1 000 000 rows to scikit-learn (`knn_wide_features`, `knn_narrow_min_samples`;
    `MLXLEARN_KNN_WIDE_FEATURES`, `MLXLEARN_KNN_NARROW_MIN_SAMPLES`; table `knn`). Every
    MLX call in the benchmark follows an idle GPU, as one made after other work does, and
    that costs a call of a few milliseconds most: at 16–128 features a batch of 8–512
    queries lost on one version or both at 42 of the 50 sizes measured below a million
    rows, down to 0.44× — mostly at batch sizes the previous calibration, which timed 1,
    16, 256 and 4 096 queries, never tried. From a million rows at 64 features and more,
    and at 256 features from 12 000 rows, every batch won or tied on both versions. A
    directly imported class searches on MLX regardless.
  - Patched mode hands every neighbor fit with 256 features or more and fewer than 8 192
    rows to scikit-learn (`knn_wide_min_samples`, `MLXLEARN_KNN_WIDE_MIN_SAMPLES`; table
    `knn-batches`, every few batch sizes from 8 to 64 queries at 4 096–12 000 rows and
    2 048–4 096 features). Right after a stock query, whose OpenMP threads are still
    spinning, an MLX call of some batch sizes — about 256 KiB of float32 queries — ran up
    to twice as slow as its neighbours, and on so short a training set stock brute force
    answers such a batch about as fast: at 4 096–7 500 rows batches of 16–32 queries lost
    (0.77×–0.90×) or only tied, including 32 queries at 4 096 × 2 048, which the `knn-k`
    table had shown winning in one long run and lost 0.83×–0.86× in fresh processes. From
    8 192 rows every batch won or tied (1.20× at worst). This gives up the wins at
    4 096–7 500 rows and 2 048–4 096 features (1.02×–3.0×), and every fit of 6 000–8 191
    rows at 1 000–2 047 features, which no grid measured. A directly imported class
    searches on the GPU from 4 096 rows as before.
  - Patched mode hands every neighbor fit with fewer than 64 features to scikit-learn, at
    any size: to its KD-tree where `algorithm="auto"` would build one (diagnostics reason
    `"tree-wins"`), to its brute force otherwise (`knn_min_features`,
    `MLXLEARN_KNN_MIN_FEATURES`; tables `knn-tree`, `knn-tree-lowrank`,
    `knn-narrow-brute`, `knn-narrow`). A KD-tree answered single rows faster than an MLX
    call at every size measured up to 8 features, and at 12 features up to 65 536 rows;
    at 2–5 features it won every batch. Before, a patched fit and 1 000-row predict at
    200 000 × 3 ran at 0.40× of stock (see **Performance**). At 16–48 features, even at a
    million rows, a batch of 16 queries lost on one version or both (0.73×–0.84×), and
    one of 1 024 (0.62×–0.79×), on Gaussian, clustered and uniform data alike.
  - **Patched neighbor fits read `n_neighbors`.** The size rules were measured at k = 10,
    and MLX pays per query for every neighbor it keeps where stock brute force barely
    does (table `knn-k`, k = 1–200, up to 4 096 features): at k = 100 a 1 024-query
    batch lost at 1 000 000 × 64 (0.80×) and batches of 32–4 096 queries at 4 096–6 000
    rows and 2 048–4 096 features (0.61×–0.89×), and k = 200 lost down to 0.33×; k = 40–50
    only tied with fewer than 150 rows per neighbor at those widths, which the row floors
    no longer admit. A patched fit now goes to scikit-learn above 50 neighbors
    (`knn_max_neighbors`, `MLXLEARN_KNN_MAX_NEIGHBORS`; reason `"below-crossover"`), and so
    does a query on an MLX-fitted model that asks for more — `kneighbors(n_neighbors=...)`,
    or `n_neighbors` raised with `set_params` — which is answered by scikit-learn's brute
    force over a float64 copy of the training rows (the float32 rows plus their
    residuals, translated by their mean as the MLX search is, so timestamps keep their
    precision), built on the first such query, kept with the model and not pickled; its
    event says `"below-crossover"`. Table `knn-query-k` (21 pairs a point; float64 and
    float32 at 8 192 × 1 024 and 2 048, 65 536 × 256 and 1 000 000 × 64): on MLX such a
    query costs what a fit with that k does, and 200 neighbors at 8 192 × 2 048 lost
    (0.73×–0.86×), as did 100–200 at a million rows and 1 024 queries (0.77×–0.86×).
    Handed over, float64 queries kept up with a stock model (0.88×–1.09×, one significant
    loss in 160, 0.88×); float32 queries at a million rows lost by a few percent
    (0.93×–0.95×), stock searching float32 rows and the copy being float64 — a known
    limit. The first such query builds the copy, `8 · n_samples · n_features` bytes:
    2–29 ms more at the three smaller shapes and 25–85 ms at a million × 64. This gives up
    k = 100–200 at 96–128 features and a million rows (1.2×–3.0×) and k = 100 at
    8 192 × 1 024 and 100 000 × 256 (1.1×–4.3×), and MLX's wins at the k asked of a query
    (1.08×–4.5×).
  - **Patched neighbor fits probe their data.** The size rules were calibrated on
    Gaussian data, where the first float32 search pass certifies every query row. A row
    it cannot certify is searched again, and its whole batch waits for it. On rows
    repeated about 20 times, tight clusters, or one column in raw units beside
    standardized ones, it certified almost none, and queries at shapes the rules send to
    MLX ran as slow as 0.17× of stock brute force (table `knn-data`); on rows repeated
    about twice, one column on 100 times the scale of the rest or clusters of std
    0.4–0.5, a few rows in a hundred went uncertified, enough to put one in most batches
    of 16–64 queries, and some batch lost at 30 of the 40 shapes and laws measured on
    and near the size floors, down to 0.36× (table `knn-structured`). A patched fit now
    searches 256 of its own rows with the first pass alone and hands the fit to
    scikit-learn if any of them could not be certified — as few as 2 of 256 lost — or
    when the data are integer-valued below 256 features, whose exact path lost batches
    of 256 and 1 024 queries at a million rows (`knn_probe_rows`,
    `knn_probe_max_uncertified`; `MLXLEARN_KNN_PROBE_ROWS`,
    `MLXLEARN_KNN_PROBE_MAX_UNCERTIFIED`; diagnostics reason `"data-probe"`, with the
    probe's count on the fit's event). Gaussian and uniform data left none uncertified at
    every shape measured below 2 048 features. The probe costs a patched fit one
    256-query first pass: timed alone, at k = 5, 2.1–6.8 ms back to back and 3.8–9.7 ms
    right after the GPU has idled at 8 192–65 536 rows and 256–4 096 features, and
    43–62 ms at a million rows and more (medians of 9, both versions). A directly imported
    class does not probe and searches on MLX regardless.
    This gives up those laws' wins where a second search was affordable (7 of the 40
    won or tied every batch, up to 4.9×).
  - `LogisticRegression` needs at least 60 000 rows (was 20 000 rows and 2·10⁷
    element-ops), a multinomial fit also at least two features
    (`logreg_multinomial_min_features`), and a binary fit at least 200 000 rows and at
    most 64 features
    (`logreg_binary_min_samples`, `logreg_binary_max_features`; tables `logreg-binary`,
    `logreg-labels`, `logreg-multi`, `logreg-separable`). A fit that converges in an
    iteration or two is mostly MLX's fixed cost, and labels that carry no signal do: at
    100 000 rows random and noisy binary labels lost (0.61×–0.90×, most on scikit-learn
    1.7.2), and at 256 features stood at 0.96×–1.17× from 200 000 rows. One feature costs
    MLX's multinomial path about 10 ms at 60 000 rows whatever the class count, and there
    three to nine random classes lost or tied (0.70×–1.17×; six met a work floor of
    360 000 rows × features × classes that this replaces, and lost 0.93×–0.94× on 1.9.0
    in runs outside the grid),
    while from two features every multinomial fit won or tied. The binary fits of the
    benchmark's own target now win from 100 000 rows at every width up to 1 024
    (1.30×–6.2×, see **Performance**), so the limits give up wins, not losses: those, and
    one-feature multinomial fits that won (1.15×–2.7×).
  - **Patched `LogisticRegression` reads `tol` and `max_iter`.** A looser `tol` or a
    smaller `max_iter` stops a fit within an iteration or two, where it is mostly MLX's
    fixed cost (table `logreg-params`, 21 pairs a point: at 9 its ratios near 1 did not
    reproduce, and a second full run found the same losses, within 0.03). On random
    labels `tol` of 3e-3 and 0.01 lost at 200 000 × 16 and 64 on both versions
    (0.50×–0.82×), and 5e-4 and 1e-3 at 200 000 × 64 on 1.7.2 (0.90×–0.93×);
    `max_iter = 1` lost at 200 000 × 64 and at 60 000 × 16 with three classes
    (0.76×–0.94×). At the default `tol` and at 1e-5, and from `max_iter = 3`, every fit
    won. A patched fit whose `tol` is above scikit-learn's default, 1e-4 — the `tol`
    every size rule was measured at — or whose `max_iter` is below 3 goes to
    scikit-learn (`logreg_min_max_iter`, `MLXLEARN_LOGREG_MIN_MAX_ITER`; the `tol` rule
    has no knob; reason `"below-crossover"`), decided before any scan of `X`. This gives
    up every looser `tol` that won (1.05×–3.5×) and `max_iter = 2` (1.05×–2.4×).
  - `SVC` needs the rbf or poly kernel (patched mode: rbf alone, below), at least 256
    features (`svc_min_features`), 1 024 rows and 384 000 elements (was any kernel but
    linear, 2 048 rows and 10⁶
    elements, with no width floor; tables `svc`, `svc-hard`). Below 128 features rbf and
    poly fits at C = 10 and 100 are slower on MLX, and narrow fits past the kernel cache
    are at C = 1 too. At 128 features the benchmark's target won from 3 000 rows, but on
    labels with little or no signal — where nearly every row becomes a support vector,
    which a fit cannot know before it runs — 3 000–6 000 rows lost at C = 10 and 100
    (0.83×–0.94×) and one-vs-one pairs of 3 000 × 128 lost on both versions
    (0.81×–0.91×); from 256 features every label law on Gaussian data won at every C
    (1.21×–5.9×). This
    gives up everything at 128–255 features that won, 1.1×–4.1× on the benchmark's
    target. The sigmoid kernel now stays on LIBSVM at every size, as the linear kernel
    does: it lost at C = 10 and 100 nearly everywhere, down to 0.06× (table
    `svc-sigmoid`).
  - **Patched `SVC` reads C.** The floors were set on isotropic Gaussian data, whose
    intrinsic dimension is its width; on data of lower intrinsic dimension C multiplies
    the SMO iterations a fit takes per row at any width, and LIBSVM, which caches the rows
    they revisit, wins them back (table `svc-structured`: a few latent directions plus
    noise, singular values 1/i, smoothed, Student-t(2) and column-scaled features at
    1 024–4 000 × 256–512, and scikit-learn's digits upsampled to 256–784 pixels). At
    C = 10 rbf and poly fits lost at 1 024–1 797 rows on both versions, down to 0.54× and
    0.26× (the digits at 256 and 400 pixels 0.57×–0.81×), and at C = 100, in runs outside
    the grid, down to 0.28×;
    at C ≤ 1 every one of them won, 1.06×–9.2×, at every `gamma` from 0.03 to 100 ×
    `"scale"`, with `tol = 1e-5`, with balanced class weights and at 8 000 rows. A
    patched rbf fit now needs a mean box — C times the mean class and sample weight, C
    itself without weights, with balanced class weights and with sample weights
    normalized to mean 1 — of at most 1 (`svc_max_c`, `MLXLEARN_SVC_MAX_C`; reason
    `"below-crossover"`), compared with a relative tolerance of 1e-9, since weights that
    average to one do so only up to rounding (balanced weights at 2% positives average
    to 1.0000000000000002, which a strict comparison sent to LIBSVM), and read from the
    parameters before any scan of `X` unless other weights need the validated targets.
    That subsumes the rbf `gamma` bounds this release had added above C = 1 (0.8–30 ×
    `"scale"`), which are gone. This gives up every rbf and poly fit above C = 1 that
    won: the benchmark's target at C = 10 and 100 from 256 features (1.36×–3.8×), labels
    with little or no signal (1.10×–2.7×), overlapping one-vs-one classes (1.39×–2.0×),
    `gamma` of 0.8–10 × `"scale"` (1.20×–4.6×) and the structured data at 2 000–4 000
    rows (1.04×–2.3×). Patched only; a directly imported class keeps these fits on MLX.
  - **Patched `SVC` sends only the rbf kernel to MLX.** The poly kernel lost even at
    C ≤ 1 on data the Gaussian grids did not hold (table `svc-poly`): at C = 1 on
    scikit-learn's digits upsampled to 256–784 pixels and on five latent directions with
    random labels (0.46×–0.73×), at C = 0.5 on the digits (0.61×–0.87×), and at `gamma`
    1.25–1.5 × `"scale"` (0.46×–0.89×), where rbf on the same data won everywhere
    (1.70×–6.3×); degree 2 and a nonzero `coef0` had lost already (table `svc-params`).
    Poly now joins linear and sigmoid on LIBSVM at every size, and the degree and
    `coef0` rule this release had added for it is gone. This gives up every poly fit
    that rule sent to MLX and won, 1.07×–7.0× across the grids. A directly imported
    class keeps poly on its size rules, and on the GPU above them.
  - **Patched `SVC` reads `cache_size`.** Every grid ran at scikit-learn's default cache,
    200 MB; with less, mlxlearn evaluates the kernel rows it cannot hold on demand, and
    at 1 MB five latent directions at 4 000 × 256 lost on 1.7.2 (0.92×; table
    `svc-cache`). A patched fit with a `cache_size` below the default goes to LIBSVM
    (reason `"below-crossover"`; no knob: the rule reads the default), giving up 3–100 MB
    where they won (1.03×–4.6×); at the default and at 1 000 MB every fit won
    (3.4×–6.6×).
  - Many-class `SVC`: the mean one-vs-one pair must clear the binary work floor,
    384 000 elements (`svc_min_pair_work`, was 250 000). Pairs of 2 000 × 128, which the
    old floor admitted, lost on overlapping classes at C = 10 and 100 on scikit-learn
    1.7.2 (0.91×–0.92×, table `svc-multi-overlap`), and pairs of 3 000 × 128 on random
    labels (0.81×–0.91×, table `svc-hard`), below the width floor now; pairs of
    1 500 × 256 and 1 024 × 512 won on both kinds of labels (1.23×–4.2×).
- **Patched fits below the crossover are decided from `X.shape`**, before any data scan or
  copy, for all three estimator families, and record a single event. A consequence: a
  sub-crossover fit whose only mismatch is data-dependent — values past 3.4e38, or for
  `LogisticRegression` a multi-label `y` or negative weights — is recorded as
  `"below-crossover"` and no longer raises under `fallback_policy="raise"`.
  `MLXLEARN_FORCE_BACKEND=sklearn` outranks those checks too, in every estimator: a forced
  patched fit is recorded as `"forced"`. The three estimators decide through one shared
  dispatch rather than their own copies of its order.
- **`set_config` is process-wide** and `config_context` is context-local (see **Fixed**).
- Patched `SVC` routes one-vs-one problems with many small classes to scikit-learn: the
  mean pair's work, `(2n/K) · n_features`, must also clear a new private
  `svc_min_pair_work` threshold (384 000, `MLXLEARN_SVC_MIN_PAIR_WORK`). The class
  count comes from the raw labels, so such a handoff, too, is decided before any scan or
  copy of `X`.
- A callable `weights` in `KNeighborsClassifier` and `KNeighborsRegressor` is served on the
  MLX path and applied to the returned distances on the host, as scikit-learn applies it.
  Before, Layer 1 raised `UnsupportedParameterError`, Layer 2 fell back to scikit-learn,
  and one set with `set_params` after an MLX fit made `predict` raise "weights not
  recognized".
- Neighbor training sets of more than 2³² rows, whose indices the exact search cannot
  address, raise `UnsupportedInputError` at `fit`, which `patch_sklearn()` hands to
  scikit-learn.
- Whether a logistic fit is binary is read from the raw labels in two vectorized passes,
  not a sort, and only when it can change the decision.
- `metric_params={"p": ...}` with `metric="minkowski"` is accepted and resolved as
  scikit-learn resolves it, with its `SyntaxWarning`. Other `metric_params` still raise.
- The neighbor estimators validate `X` and queries as float64 or float32, keeping float64
  for the index to translate, instead of casting to float32 first.
- `radius_neighbors` and `radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`
  compute with a float32 query (see **Performance**), so a float64 query float32 cannot
  hold raises `UnsupportedInputError` (reason `"float32-overflow"`, a `ValueError`), as
  `kneighbors` does. 0.1.0a2 upcast it and answered in float64.
- On the GPU at narrow widths, kernel rows evaluated on demand are bit-identical to the
  rows of the materialized Gram, so `cache_size` crossing the Gram boundary no longer
  changes the fitted model there (measured on all four kernels at 2 000 × 10). At a
  thousand features and more MLX can pick another matmul kernel for the Gram's shape or
  for a batch of rows, and `cache_size=5` moved decision values by up to 9.2e-4 at
  2 048 × 2 048; MLX's CPU device moves them at any width, by up to 2.2e-3 at 2 000 × 10.
  Pin `cache_size` for bit-reproducible fits, on either device.
- `mlx_guard` lets only mlxlearn's own errors, `MemoryError` and escalated warnings pass
  through. A deliberate input error detected inside guarded code — `SVC`'s non-finite
  model — is an `InvalidInputError`, which is a `ValueError`, so scikit-learn-facing
  behavior is unchanged.
- `import mlxlearn; mlxlearn.svm.SVC` works: the estimator subpackages load lazily on first
  access, and `import mlxlearn` still imports neither scikit-learn nor MLX.
- An `mx.array` converted to NumPy through mlxlearn's array helpers is a read-only view.

#### Memory

- The neighbor query's memory budget now bounds its whole working set — tile, selection
  temporaries and refinement gather — instead of the distance tile alone.
- A fitted neighbors model keeps one private copy of the training rows, shared read-only
  between host and device (`_fit_X` is a view of it), beside the translated copy the search
  runs on; fitting no longer allocates full-size host copies or temporaries. float64 input
  additionally keeps a float32 residual. A model loaded with `pickle` or `joblib`, or
  deep-copied, returns to that one copy on its first query, rather than keeping a second
  full copy of its training rows that doubled the file when it was saved again.
- The fitted `SVC` keeps its support vectors mapped into the compute frame and resident on
  the device, built lazily on the first prediction and dropped from pickles.

#### Packaging, CI and benchmarks

- The MLX floor is raised to 0.31.2: concurrent fits abort the process on earlier
  releases.
- The version is defined once, in `src/mlxlearn/__init__.py`; `pyproject.toml` reads it.
- `pytest` deselects tests marked `slow` by default (`pytest -m slow`, `pytest -m ""`).
- Without a working GPU, tests that need one are skipped with a reason.
  `MLXLEARN_REQUIRE_GPU=1`, set in CI, makes a missing GPU stop the run instead. Tests
  that assert the MLX path ran are marked `gpu`, since without a GPU nothing raises —
  auto-selection picks the CPU — and a slow tooling test runs the suite on a simulated
  GPU-less host to hold every test to that.
- pandas is in the `test` extra (floor 2.2.3, installed exactly by the minimum-versions
  job), so the pandas input-handling tests run on every CI leg. The nightly job also
  installs torch for the torch container cases.
- Benchmark significance requires the medians to be three standard errors apart, each
  estimated from a real interquartile range, with at least five samples on each side.
- The performance gate controls its false-failure rate per run, not only per case: the
  runner re-measures any case the gate would fail over three times the iterations, records
  it as that result's `confirmation`, and the gate fails the case only if the loss
  reproduces. In a simulation of 137 gated cases with no real change, the share of
  failing nights drops from 73–79% to under 1%.
- The performance gate judges the backend each case actually ran on under the configured
  `Tuning`. A backend forced with `--force-backend` or `MLXLEARN_FORCE_BACKEND` is
  recorded as forced, so the gate does not blame `Tuning` for it, and results record the
  effective `Tuning` and every `MLXLEARN_*` variable.
- The stock KNN baseline in the benchmark grid is scikit-learn's default
  `algorithm="auto"`.
- ruff is pinned, in the dev extra, and CI lints with that pin.

#### Documentation

- The README states precisely what `unpatch_sklearn()` restores, that scikit-learn's
  internal users of the patched names are deliberately not accelerated, where each
  configuration setting reaches (threads, `asyncio`, process workers), what `device`
  selects, the error taxonomy, and the two cases inference runs off its fitted device.
- [`docs/fp32_policy.md`](docs/fp32_policy.md) describes the neighbor certificate and tie
  contract and what its second pass costs, float64 translation in each estimator, `SVC`'s
  compute frame, rbf refinement, resolution check, self-entries, shrinking and
  `cache_size`, the L-BFGS float32 floor, and range checking at inference. `deterministic`
  and `random_state` are documented as having no observable effect in 0.1.x, and Rule C
  as reserved.
- `radius_neighbors` and `radius_neighbors_graph` are documented as rounding float64
  input at its raw magnitude: unlike `kneighbors`, they do not translate it by its mean, so
  a large constant offset costs them precision (Unix timestamps round to 128-second steps).
  The README, the `NearestNeighbors` docstring, `docs/fp32_policy.md` §3.4 and the support
  matrix say so, and that the radius methods take sparse queries in both layers. The
  `kneighbors` docstring says `metric="sqeuclidean"` returns squared distances.
- [`docs/support_matrix.md`](docs/support_matrix.md) lists what each estimator falls back
  for, which problems patched dispatch hands to scikit-learn by size, and the deliberate
  differences from scikit-learn.
- The README and [`docs/benchmarks.md`](docs/benchmarks.md) have a section, *Known limits
  of patched dispatch*: the data laws and hyperparameters the committed grids cover; why
  a fit on data outside them can still be slower than stock — MLX pays more per call and
  per solver iteration, and support-vector counts, intrinsic dimension and convergence
  are not known before a fit runs; how to keep a workload on scikit-learn
  (`patch_sklearn([...])`, `unpatch_sklearn()`, `MLXLEARN_FORCE_BACKEND=sklearn`); and
  how to report a case.
- [`docs/development.md`](docs/development.md) documents the fit pattern, how a fit's
  decisions become its one event, and the rules against unevaluated `mx.array` state and
  plain `ValueError`s inside `mlx_guard`.

### Added

- `fit_event_` on every fitted estimator: the diagnostics event its fit committed. It
  survives the ring buffer's eviction and pickling, and is recorded even with
  `diagnostics=False`.
- `InvalidInputError`, a `ValueError` for invalid input detected inside mlxlearn's own
  code, in `mlxlearn.exceptions` and exported from `mlxlearn`. `SparseInputError` is
  exported from `mlxlearn` too.
- Environment defaults for every public option, read by each new process:
  `MLXLEARN_DEVICE`, `MLXLEARN_FALLBACK_POLICY`, `MLXLEARN_OUTPUT_TYPE`,
  `MLXLEARN_DETERMINISTIC`, `MLXLEARN_RANDOM_STATE`, `MLXLEARN_DIAGNOSTICS`.
- `mlxlearn.reset_fallback_warnings()`, which re-arms fallback warnings without dropping
  the recorded events.
- `CapabilityError` and its subclasses take `advice=`, the remedy for a directly imported
  estimator, appended to the message; `.summary` is the message without it and `.advice`
  the remedy. Both survive pickling.
- MLX's buffer cache is released after a top-level mlxlearn operation that grew it by more
  than 1/16 of physical memory. Private knob: `MLXLEARN_MLX_CACHE_RELEASE_FRACTION`; `0`
  disables it.
- `metric="sqeuclidean"` on the MLX path of the neighbor estimators.
- scikit-learn 1.9 callbacks (`set_callbacks`) are driven on `LogisticRegression`'s MLX
  path with scikit-learn's own lbfgs task tree: the fit, plus one subtask per iteration.
- scikit-learn's fit-time deprecation warnings are emitted on the MLX path too:
  `LogisticRegression`'s `penalty`, `l1_ratio=None`, `n_jobs` and, on 1.7, `multi_class`
  (in the installed patch release's own words: 1.7.0 and 1.7.1 say "will be removed in
  1.7");
  `SVC`'s `probability` on 1.9.
- Diagnostics:
  - a `LogisticRegression` fit that stops without converging records the stop on its one
    fit event: `fit_event_.extra` gains `stop_reason`, `n_iter`, `n_fev`,
    `grad_max_abs`, `tol` and `at_resolution_floor`, and `fit_event_.detail` the
    explanation;
  - new reasons `"tree-wins"`, `"prebuilt-index"`, `"gpu-unavailable"`,
    `"forked-process"`, `"array-api-dispatch"` and, for a `LogisticRegression` predicting
    from assigned attributes, `"external-state"`;
  - inference events carry the fit's reason as `extra["fit_reason"]`;
    `LogisticRegression`'s carry `extra["compute"] = "numpy-float64"`, since it predicts
    with scikit-learn's float64 code on every backend;
  - a neighbor query that needed the certificate's second pass or its exact search
    reports how many rows did in its event's `extra` (`search="certificate-fallback"`,
    `widened_rows`, `exact_rows`).
- Private tuning knobs `knn_min_features`, `knn_wide_features`,
  `knn_narrow_min_samples`, `knn_wide_min_samples`, `knn_max_neighbors`,
  `knn_probe_rows`, `knn_probe_max_uncertified`, `logreg_multinomial_min_features`,
  `logreg_binary_min_samples`, `logreg_binary_max_features`, `logreg_min_max_iter`,
  `svc_min_features`, `svc_min_pair_work`, `svc_max_c` and `mlx_cache_release_fraction`,
  each with its `MLXLEARN_*` variable. The rules on `LogisticRegression`'s `tol`, `SVC`'s
  kernel and `SVC`'s `cache_size` read scikit-learn's own defaults and have no knob.
- Internal validation and array helpers: `validate_float32_input`,
  `require_nonnegative_weights`, `finite_range` and `ValueRange`, and
  `check_squared_float32_range`, a guard for data whose squares overflow or underflow
  float32 (reasons `"float32-overflow"` and `"float32-underflow"`).
- Tests:
  - direct unit tests for L-BFGS, the SVM kernels (against `sklearn.metrics.pairwise`) and
    the SMO solver (against LIBSVM's dual objective and an independent KKT check);
  - merge-gate tests for every estimator in both layers: `joblib` round trips (including
    memory-mapped), process-parallel cross-validation, `Pipeline` inside `GridSearchCV`,
    and sticky-backend inference;
  - the `check_estimator` suites for both layers also run with the MLX GPU path forced,
    and `check_param_validation` runs for every estimator in both layers;
  - runtime tests for configuration scope across threads and processes, the device gate,
    fork safety, `mlx_guard` and buffer-cache release;
  - an event-invariant suite over every estimator, layer and fit path — selected, below
    the crossover, forced, capability fallback, `tree-wins`, `SVC`'s per-pair gate — that
    asserts one fit event matching `execution_backend_` and `fit_event_`, and one event per
    public inference call named after it;
  - stronger logistic parity: objective-level unpenalized parity, every capability gate
    isolated, forced-GPU parity for weighted and no-intercept fits, and an independent
    check of the parity oracle.
- Python 3.14 support (classifier and CI leg). That leg is unpinned: it tests the newest
  scikit-learn the declared range admits, so a pull request that raises the cap, a
  Dependabot one included, runs the suite on the new release.
- A minimum-versions CI leg that installs every declared floor exactly, a weekly job
  against upstream pre-releases, and Dependabot configuration.
- The `layering` compliance gate: no scikit-learn, user-configuration or estimator imports
  in `_solvers`/`_kernels`, and no global RNG seeding anywhere. The kernels and the
  neighbor index read no configuration indirectly either: the estimator resolves the MLX
  device, a GPU-fitted model's on a GPU-less host included, and passes it in (an internal
  `device=` keyword).
- `tools/check_environment.py`, which fails a CI job whose versions or device are not what
  it claims.
- The sdist carries the test suite, `docs/`, `CHANGELOG.md` and `ACKNOWLEDGMENTS.md`.
- CI runs doctests, the benchmark harness and gate at one iteration, and the slow tests in
  a job of their own.
- The benchmark `published` profile, which reproduces `docs/benchmarks.md`, and a
  `--force-backend` flag. A case may carry its own iteration count, which caps
  `--iterations` and never raises it. New benchmark cases: low-dimensional KNN
  (`d` = 2, 3), large query batches (10 000 and 50 000), and many-class SVC.
- `python -m benchmarks.calibrate`, the calibration grids behind every dispatch
  threshold: `run` measures a named grid (the dense neighbor grid times twelve batch
  sizes, 1 to 4 096 queries, on every index), resuming where an interrupted run stopped;
  `table` renders it for one or more scikit-learn versions side by side, with where
  patched dispatch now sends each row; `update` re-renders every such table in
  `docs/benchmarks.md`. The raw results those tables come from are committed in
  `benchmarks/results/calibration/`, and every threshold, docstring and headline range
  cites its table by grid name. Grids vary more than the shape: the data (`knn-data`,
  `knn-structured`, `knn-narrow`, with a column for what the fit-time probe found), k
  (`knn-k`, 1–200, up to 4 096 features), `gamma`, `degree` and `coef0` (`svc-params`,
  `svc-hard`), labels without signal (`svc-hard`, `logreg-labels`), overlapping classes
  (`svc-multi-overlap`), inference batch sizes (`svc-predict`), well-separated classes
  (`logreg-separable`), the kernel on data unlike the Gaussian grids (`svc-poly`),
  `cache_size` (`svc-cache`), `tol` and `max_iter` (`logreg-params`), and the k a query
  asks for, in float64 and float32, with the first such query timed apart
  (`knn-query-k`). Grids whose ratios sit near 1 (`logreg-params`, `knn-query-k`) take
  21 interleaved pairs a point: at 9 they did not reproduce. The harness's `compare`
  takes an `mlx_setup` run before every mlxlearn call, outside the timed window, to
  time a first call as often as a comparison needs.
- `benchmarks/ab.py`, the release-over-release A/B behind **Performance**: two source
  trees in alternating fresh processes. Its results against 0.1.0a2 are
  `benchmarks/results/ab-0.1.0a2.jsonl`.
- README notes on fork safety, MLX's buffer cache, and pickles of patched models needing
  mlxlearn to load.
- A `Documentation` project URL.

### Removed

- The unused `NotSupportedByBackend` alias in `mlxlearn.exceptions`.
  `MLXLearnPerformanceWarning` and `UnsupportedEstimatorStateError` stay, documented as
  reserved: nothing in 0.1.x emits or raises them.
- The internal helpers `check_finite_float32` and `class_weight_vector`, which nothing
  called any more.
- The benchmark harness's cold-start measurement, which 0.1.0a1 documented
  ("cold-start / fit / predict reported separately") but never implemented.
- `wheel` from the build-system requirements, and the `MLXLEARN_CI` variable from CI,
  which nothing read.

### Security

- Workflow actions are pinned by commit SHA, workflows run with least-privilege token
  permissions, and every job has a timeout.
- The release workflow published without running the tests, skipped its tag check on a
  manual dispatch, and never checked the wheel it described. It now runs the whole CI
  workflow on the tagged commit and publishes the distributions that run's build job
  built and checked — a pure-Python wheel that installs cleanly and whose metadata agrees
  with `mlxlearn.__version__`, and an sdist whose own suite collects against it — rather
  than building them a second time. A `version` job requires the wheel's filename and
  metadata, the sdist's filename and the tag to agree. It publishes only on a `v*` tag
  push, through PyPI Trusted Publishing from an environment that requires approval. A
  manual run is a dry run.
- The license gate walks the full installed runtime dependency closure and parses SPDX
  expressions. It used to check direct dependencies only, by substring, and accepted, for
  example, "Proprietary; unlicensed".

### Performance

Measured on an Apple M4 Max (16 cores, 128 GiB), macOS 26.6.2, with MLX 0.32.2, Python
3.13.3, NumPy 2.5.2 and scikit-learn 1.9.0, and nothing else benchmarking: fresh
processes alternating between 0.1.0a2 and this release, three of each (two for the
slowest SVC fits), at least 7 timed calls per process after warm-up (3 for those fits),
each call after a garbage collection as [`docs/benchmarks.md`](docs/benchmarks.md)
describes; medians. Against 0.1.0a2 unless an entry names another baseline. The tool is
`benchmarks/ab.py` and every raw timing is in `benchmarks/results/ab-0.1.0a2.jsonl`,
named by case (`nnq:100000:32:20000:5` is 20 000 queries, k = 5, against 100 000 × 32).
Where patched dispatch now sends each problem is under **Changed**.

- Neighbor queries on the GPU use two-stage segmented selection and fused tile
  expressions: 100 000 × 32 with 20 000 queries went from 953 to 337 ms, 200 000 × 64
  with 4096 queries from 446 to 153 ms, and `k=200` (100 000 × 32, 4096 queries) from
  237 to 149 ms. The tile expressions are compiled on the GPU only; MLX's CPU device runs
  the same expressions eagerly, so a first fit on a cold kernel cache does not wait on
  MLX's C compiler.
- Binary and small-integer data takes an exact integer path: 100 000 binary rows × 64
  with 4096 queries, 197 → 112 ms.
- The index build is slower than 0.1.0a2 at every size measured — 1 000 × 32 on the GPU
  0.79 → 1.05 ms, 50 000 × 32 3.3 → 4.1 ms, 2 000 000 × 64 207 → 222 ms, and 20 000 × 32
  on MLX's CPU device 1.7 → 2.5 ms. It does more than 0.1.0a2's did: the span check, the
  certificate's sorted norms, the integer-path check and, for float64 input, a float32
  residual. The residual and the translated copy stay on the device, so a float64 index
  holds 3.0× the float32 data there (3.5–4.2× at the peak of the build); 0.1.0a2 held one
  copy, 1.0×, on the device and its others on the host.
- Peak device memory per neighbor query stays near the budget instead of reaching about
  7× it: 7 427 → 896 MiB at 50 000 × 768 with `k=1000` and 4096 queries.
- Certification costs a second pass where the first pass cannot prove its answer — where
  the old margin was wrong, and at exact ties at the k-th distance. On the cases in
  [`docs/fp32_policy.md`](docs/fp32_policy.md) §2.1, 50 000 rows, 4096 queries, `k=10`:
  1.6–3.0× the old query time at 4–32 features, and 5.3× for two clusters at 64
  features, where every query takes the exact search. No second pass on ordinary data:
  Gaussian 50 000 × 16 went from 85 to 37 ms.
- Patched low-dimensional KNN handed to the KD-tree: 200 000 × 3, fit and 1 000-row
  predict, went from 0.40× to 1.00× of stock.
- `radius_neighbors` on an MLX-fitted `NearestNeighbors` hands scikit-learn a float32
  query: 28.3 → 6.7 ms for 1 000 queries against 20 000 × 16.
- Neighbor votes use a single `np.bincount`, bitwise identical. End to end, with the
  faster search, `KNeighborsClassifier.predict` on 20 000 queries against 100 000 × 16 and
  50 classes went from 997 to 381 ms.
- `LogisticRegression`'s L-BFGS bookkeeping (two-loop recursion, inner products,
  curvature test, `max|g|`) runs in NumPy float64 on the host, so each objective
  evaluation costs one device synchronization instead of 12–33 blocking round trips per
  iteration, and the binary gradient on the GPU uses a contiguous `Xᵀ`. Fits forced onto
  the GPU are 1.9–4.2× faster with iteration counts unchanged: 20 000 × 16 20.0 → 4.8 ms,
  60 000 × 128 46.7 → 16.1 ms, 20 000 × 1 024 71.5 → 38.6 ms, 2 000 000 × 16 362 → 110 ms.
- `SVC` fits no longer fall off a cliff once a one-vs-one subproblem exceeds the
  kernel-cache Gram cutoff (about 7 240 rows at the default `cache_size=200`): on a cache
  miss, kernel rows are fetched many per GPU round trip, together with the rows closest to
  becoming the next working pair, and shrinking (below) is on. rbf fits went from 1.17 to
  0.45 s at 8 000 × 128, 1.95 to 0.83 s at 12 000 × 128, 2.60 to 1.23 s at 16 000 × 64 and
  4.81 to 1.88 s at 16 000 × 256, with results unchanged.
- Shrinking, now honored and on by default: measured through `SVC`, off against on, rbf
  1.00–1.20× (1.15× at 8 000 × 64, 1.20× at 16 000 × 32, 1.00× at 4 000 × 256) and poly
  1.00× at 8 000 × 64. It can cost on the linear kernel, as LIBSVM's own shrinking does: at
  2 000 × 20, `C=1` is 1.27× faster but `C=10` runs at 0.89× on the GPU. Patched dispatch
  never sends the linear kernel to MLX.
- `LogisticRegression` on MLX pays much less before its first iteration. The column
  means, the centered float32 copy of `X` and the float32 range check are split by rows
  across up to eight threads (`_common/host.py`; blocks fixed by the shape, so results
  are reproducible), three to six times faster than one NumPy call each at 1 000 000 × 32;
  the first L-BFGS step is L-BFGS-B's own, a unit move, so a fit scikit-learn finishes in
  one iteration takes one here too instead of 11–15; the line search accepts an
  interpolated step anywhere past 1% of its bracket instead of bisecting to 10%, which
  took a fit on labels without signal from six objective evaluations to four; and
  numeric labels are encoded by a binary search of the classes instead of a sort. Against
  stock, in the calibration tables (before → after, scikit-learn 1.9.0): binary
  100 000 × 1 024 0.70× → 1.45×, 1 000 000 × 16 2.37× → 4.39×, 200 000 × 128 0.86× →
  1.65×; multinomial on well-separated classes at 60 000 rows and 256 or 512 features,
  three to ten classes, 0.31×–0.46× → 1.00×–1.22×.
  Iteration counts moved on the pinned problems of `tests/unit/test_lbfgs.py`: two or
  three more on the binary ones (scikit-learn takes more still), one fewer or one more on
  the multinomial ones, and a lower objective on every one.
- `SVC` inference on a batch of a few rows — up to 2²⁵ rows × support vectors × features
  — runs on the host in NumPy float32, with the device's arithmetic: `predict` on one row
  of a model fitted on MLX went from 0.64–0.98 ms, most of it waking an idle GPU, to
  0.19–0.35 ms, against 0.22–0.82 ms for LIBSVM (table `svc-predict`).
- Every public inference call costs less in the wrapper: an estimator fitted by mlxlearn
  skips scikit-learn's `check_is_fitted` (which builds the estimator's tags, twice for a
  `predict` that reaches `decision_function`), the call's diagnostics event is kept as
  its fields until someone reads it, and `LogisticRegression` records it inline. What
  is left, measured on `LogisticRegression.predict` and `predict_proba` of 1 and 16 rows
  — host NumPy on every backend, 80–135 µs of stock scikit-learn's own work — is 0.9–1.7
  µs over stock in a tight loop, on both versions (1.4–2.7 µs before the inline record).
  Timed as the benchmark times it, each call after a garbage collection, which leaves
  the caches cold, it is more and noisier: 0.85×–1.04× of stock on 1.9.0 and
  0.90×–0.98× on 1.7.2 (up to 20 µs over; 41 interleaved pairs, two runs, a
  significant difference in 9 and 5 of the 24 cases). It is a fixed cost per call, a few
  Python calls, so it matters only where stock's own call is this short.
- `SVC` inference: the rbf kernel tile is one fused GPU kernel, bit-identical to the
  unfused evaluation, and `decision_function`/`predict` overlap the device with the
  float64 host contraction. On the GPU at 4 000 × 256, `predict` on 1 row went from 1.09
  to 0.83 ms, on 16 rows from 1.18 to 0.91 ms and on 4 000 rows from 7.4 to 6.4 ms.
- The `SVC` rbf entries recomputed in float64 (see **Fixed**) cost nothing on ordinary
  data, which never opens that path; data where every entry within a cluster needs it
  pays for the recomputation.
- Many-class `SVC` under `patch_sklearn()`: at 2048 × 512, `K=64` went from 0.25× to 1.00×
  of stock, and `K=8` stays on MLX at 1.40×.
- Patched fits below the crossover pay less before handing the work to scikit-learn: at
  200 × 5, a patched fit took 1.11× the stock fit's time for KNN (was 1.35×), 1.09× for
  `LogisticRegression` (was 1.22×) and 1.11× for `SVC` (was 1.36×).
- The float32 range check is 3.8× faster and allocates no temporaries on finite input:
  19 ms instead of 73 ms at 128 000 × 768, with peak extra memory near 0 instead of
  1.6 GiB. A `NearestNeighbors` fit on that float64 input still takes as long as before
  (143 → 150 ms): the rest of the index build does more (above).
- Costs: `patch_sklearn()` with every estimator imports five scikit-learn packages that
  bind the estimators at import time, so that they keep the stock classes. In a process
  that has already imported the estimator modules that takes 121 ms on Python 3.13 with
  scikit-learn 1.9.0, and 100–102 ms on Python 3.10 with scikit-learn 1.7.2 and 1.7.0;
  from a fresh process, importing mlxlearn and patching everything took 937 ms against
  827 ms for patching only `SVC`, `KNeighborsClassifier` and `KNeighborsRegressor`, which
  costs 18 ms more than importing those three scikit-learn modules alone (fresh
  processes, operating-system file cache warm, median of 21). `unpatch_sklearn()`'s sweep
  takes about 1 ms. Where the buffer-cache release triggers — machines with less memory —
  repeated same-shape large queries pay reallocation: a repeated 150 000 × 64 query with
  6 000 queries took 168 ms with the release forced on every call against 162 ms without.

## [0.1.0a2] — 2026-09-02

### Changed

- Renamed the source repository from `tabulai/mlxlearn` to `tabulai/mlx-learn` and
  updated package metadata, development instructions, and release automation to use the
  canonical URL.
- Updated the release metadata and runtime version for the repository-name migration.
- Corrected the CI support matrix so Python 3.10 is tested with its compatible
  scikit-learn 1.7.x release line; scikit-learn 1.8+ requires Python 3.11 or newer.

## [0.1.0a1] — 2026-09-02

First alpha. Published for external testers before the second wave of estimators; the API
is not yet stable.

### Added

- **Layer 1 estimators** (`import mlxlearn.…`), MLX-accelerated with NumPy in and NumPy out:
  - `mlxlearn.neighbors.NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` —
    exact brute-force neighbors with blocked distance evaluation, `mx.argpartition` top-k
    selection, deterministic `(distance, index)` tie-breaking, and candidate-distance
    refinement that avoids fp32 cancellation in the expanded Gram formula.
  - `mlxlearn.linear_model.LogisticRegression` — L2 / unpenalized, binary and multinomial,
    L-BFGS on the exact scikit-learn objective. The design is centered before solving
    (exact when there is an intercept to absorb the shift), which is what makes the fit
    survive features carrying a large constant offset — and makes it converge in fewer
    iterations than scikit-learn on such data.
  - `mlxlearn.svm.SVC` — exact SMO with a memory-bounded kernel cache and one-vs-one
    multiclass. No approximations ship under a scikit-learn estimator name.
- **Layer 2 patching** (`mlxlearn.patch_sklearn` / `unpatch_sklearn`) — idempotent,
  fully reversible, module-attribute based, with a registry of Layer 1 estimators.
- **Sticky backend invariant** — the backend is selected during `fit`, recorded on
  `_execution_backend_`, and reused by every inference call on that fitted model.
- **Fallback taxonomy** — capability mismatches fall back under
  `fallback_policy="warn" | "raise" | "silent"`; unexpected MLX runtime failures raise by
  default and are never silently rerun on scikit-learn. Anything scikit-learn *can* serve
  is in the capability family so the fallback actually works — including float64 values
  that overflow float32 and negative sample weights, both of which scikit-learn accepts.
- **Diagnostics** — `get_backend_diagnostics()`, `get_last_backend_event()`,
  `clear_backend_diagnostics()`.
- **Configuration** — `set_config` / `get_config` / `config_context` over `device`,
  `fallback_policy`, `output_type`, `deterministic`, `random_state`, `diagnostics`;
  private tuning via `MLXLEARN_*` environment variables.
- **Benchmarks** — `python -m benchmarks.run`, `mx.eval`-synchronized, warmup plus median
  of ≥5, cold-start / fit / predict reported separately, per-algorithm size grids, and
  measured crossovers wired into the dispatch thresholds.
- **Compliance gates in CI** — Intel/sklearnex branding grep, private `sklearn._*` import
  check (quarantined to `patching/`), header/provenance audit, dependency license scan.

### Measured

Full numbers and method in [`docs/benchmarks.md`](docs/benchmarks.md); tolerances and
parity claims in [`docs/fp32_policy.md`](docs/fp32_policy.md).

- **Neighbor queries: 2.9×–17× against stock scikit-learn**, from ~250 samples upward. The
  crossover was measured at ~250–1 000 samples, not the 4 096 originally assumed.
- **Neighbor `fit`: 0.17×–0.36×**, and knowingly so — mlxlearn uploads to the device where
  scikit-learn stores a reference. Listed by name in `benchmarks/gate.py` rather than
  hidden behind "no crossover measured".
- **`LogisticRegression` is 10–50× slower than scikit-learn below 1 024 features.** Its
  iteration count matches scikit-learn's, but each iteration pays a host synchronization
  per line-search evaluation: 5.9 ms per iteration where the matmul inside it costs 163 µs.
  The crossover is set high accordingly (20 000 samples **and** 2·10⁷ elements), so patched
  dispatch measures 0.79×–1.01×, both within noise. This is an optimization target, not a
  property of the design.
- **`SVC` is decided by width, not sample count.** 2.93× at 4 000 × 256 (rbf), 0.47× at the
  same sample count and 32 features, and slower in 58 of 64 configurations on the initial
  narrow grid — worst 0.013×. SMO is sequential, so MLX can accelerate the kernel rows an
  iteration needs but not the iterations; whether that pays depends on how expensive a row
  is. The crossover is therefore on work (`svc_min_samples` **and** `svc_min_work`), and
  `kernel="linear"` never takes the MLX path at any size. Its primal objective agrees with
  LIBSVM to 5.2e-07 relative and the KKT conditions hold on an independent check — this is
  a dispatch story, not a correctness one.
- **All ten scikit-learn estimator check suites pass with zero mlxlearn-specific xfails.**
  Two checks are exempt only because scikit-learn declares them as expected failures for
  its own `SVC` and `LogisticRegression`; a test re-derives that list from scikit-learn so
  the exemption cannot outlive its justification.

### Compatibility

- Tested against scikit-learn **1.7.2 and 1.9.0**; 289 tests pass on both. Three parameters
  scikit-learn is deprecating mid-range — `LogisticRegression.multi_class` (removed in 1.9),
  `LogisticRegression.penalty` (now a sentinel, with `l1_ratio` carrying the meaning) and
  `SVC.probability` (now a sentinel) — are handled by resolving what the installed version
  actually means rather than reading the raw attribute. See
  [`docs/support_matrix.md`](docs/support_matrix.md).

### Notes

- Ships as a pure-Python wheel. The compiled-extension decision is recorded with
  measurements in [`phase0/cython_decision.md`](phase0/cython_decision.md): the compiled
  SMO core was worth 1.26× at 2 000 samples, 1.16× at 6 000, and 1.00× at 12 000 — the
  advantage vanishes exactly where performance matters. Revisit when `TSNE` lands in 0.2.
- Requires macOS on Apple silicon. Behavior on other platforms is untested and unsupported.
- Not reviewed by a lawyer.

### Deliberately not carried over from the ancestor

Each of these is now covered by a regression test:

- `exclude_self` compared each row against its **block-local** index, so `kneighbors(None)`
  returned a row as its own nearest neighbor for any training set larger than one query
  block (1 wrong row in 1 500 with default blocking).
- The GPU top-k returned `uint32` indices, so padding a short trailing tile with `-1` raised
  `Converting -1 to uint32 would result in overflow` — presented to users as an unexplained
  fallback, because the dispatcher caught every exception.
- `metric_params` was accepted and silently ignored, returning unweighted Euclidean results
  for a weighted-Minkowski request.
- Module-level caches keyed on `id()` or a weakref to a caller's array, so a mutated or
  freed-and-reallocated array produced predictions computed on the wrong data.
- A `fit` continuation cache that warm-started from a previous unrelated fit of the same
  shape, making `fit` a non-pure function of its arguments.
- `except Exception:` around the accelerated path, rerunning silently on scikit-learn.

[Unreleased]: https://github.com/tabulai/mlx-learn/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/tabulai/mlx-learn/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/tabulai/mlx-learn/releases/tag/v0.1.0a1
