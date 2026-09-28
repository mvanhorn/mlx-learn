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
  float64 mean and keeps a float32 residual beside each float32 value.
- Rows tied at the k-th distance now follow the documented `(distance, index)` order:
  `kneighbors(k)` is a prefix of `kneighbors(k + m)`, results no longer depend on block
  sizes, and GPU and CPU agree wherever distances are exact (binary, integer and one-hot
  features). 0.1.0a1 described this order, but on tie-heavy data it did not hold.
- Data whose squares overflow or underflow float32 (`|x|` around 1e19 or 1e-20) no longer
  yields infinite or zero distances and arbitrary neighbors; it is scaled by an exact
  power of two. Training data spanning more than float32's largest value / 8 from its
  mean (a single 1e38, say) raises `UnsupportedInputError` at `fit` directly imported and
  falls back to scikit-learn under `patch_sklearn()`. Queries still out of range raise
  `UnsupportedInputError`.
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
- String arrays are rejected with scikit-learn's error instead of being parsed as numbers.
- `kneighbors` reports the `n_neighbors` bound error with scikit-learn's full message
  (`n_samples = …`), and only after validating the query.
- Under `patch_sklearn()`, a model fitted on MLX answers sparse queries with
  scikit-learn's brute-force search instead of raising `SparseInputError`. This is
  recorded as a fallback and honors `fallback_policy`.
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
- An exception raised by a scikit-learn 1.9 callback is re-raised as itself, not reported
  as a `BackendExecutionError`.

#### SVC

- `gamma="scale"` falls back to `gamma=1` only for an exactly-zero variance, as
  scikit-learn does. It did so for any variance up to 1e-12, which gave a model at chance
  on data with features around 1e-6 or smaller.
- Kernel inputs are mapped exactly in float64 before the float32 cast: the rbf training
  mean is subtracted first, so large offsets are no longer rounded away, and an exact
  power-of-two scale is folded into `gamma`, so data around 1e±19, and `gamma` values
  outside float32's range, match scikit-learn instead of returning a NaN model. Results
  are bit-identical for data already in range.
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
- `SVC(kernel="linear", gamma=0.0)` fits, as in scikit-learn.
- Negative `sample_weight` values are treated as zero weights, exactly as LIBSVM treats
  them, so patched `SVC` no longer raises where stock scikit-learn fits.
- A sparse query on a dense-fitted `SVC` raises scikit-learn's `ValueError` instead of a
  `TypeError`.
- The `max_iter=-1` `ConvergenceWarning` names the budget mlxlearn resolved `-1` to,
  instead of the contradictory "max_iter=-1".
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
  and `tree-wins` fits record one event instead of two. One public inference call
  records one event, named after the method called — `predict`, `predict_proba`,
  `predict_log_proba`, and scikit-learn's inherited `score`, `kneighbors_graph` and
  `radius_neighbors_graph` included, which were recorded under the inner method's name —
  whichever backend fitted the model. A patched neighbor estimator's sparse-query
  fallback is that call's one event. A fit forced to scikit-learn with
  `MLXLEARN_FORCE_BACKEND=sklearn` is recorded with `fallback=True`.
- A refit that falls back to scikit-learn, or fails, no longer keeps the previous fit's
  MLX state — the neighbor index and its training copy, `SVC`'s pair weights and
  device-resident support vectors — alive in memory and in pickles.
- Inference after a refit that failed raises `NotFittedError` (an `IncompleteFitError`,
  which is also an `MLXLearnError`) instead of a bare `MLXLearnError`.
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
- Two internal helpers: `as_float32(copy=True)` returned an alias for `np.matrix`,
  `np.memmap`, pandas and `__array__` inputs, and `class_weight_vector` treated weights
  merely close to 1 as unweighted.

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
- Fallback warnings point at the caller's line, so module-based warning filters match. A
  warning hidden by an `"ignore"` filter no longer uses up its once-per-reason slot, and an
  `"error"` filter raises on every fallback.
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
  mlxlearn class, so scikit-learn modules first imported while patched are restored too.
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

#### Tooling, CI and benchmarks

- `tools/compliance.py --json` printed invalid JSON; stdout is now the JSON document alone.
- The benchmark "IQR" was the full min–max range at the default 7 iterations, so one
  outlier could hide a 2× regression. The gate also excused MLX-path regressions using a
  crossover derived from the run it was judging.

#### Documentation

- 0.1.0a1 said `LogisticRegression.multi_class` was removed in scikit-learn 1.9. It was
  removed in 1.8, the release that also made `penalty` a deprecated sentinel;
  `SVC.probability` became one in 1.9. [`docs/support_matrix.md`](docs/support_matrix.md)
  now tabulates all three per version.
- 0.1.0a1 attributed `LogisticRegression`'s per-iteration cost to one host
  synchronization per line-search evaluation. The cost was 12–33 blocking
  synchronizations per iteration in the L-BFGS bookkeeping; the line search averaged about
  1.1 evaluations per iteration. The fix is under **Performance**; the benchmark write-up
  is marked for rewrite when the numbers are re-measured.
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

### Changed

#### Behavior

- **`SVC` honors `shrinking`.** It was accepted and ignored; the SMO solver now runs
  LIBSVM's shrinking heuristic when `shrinking=True`, which is scikit-learn's default, so
  default fits change path. The optimum and the stopping test are the same — the test is
  evaluated over every sample before the solver may stop — but, as in LIBSVM, where inside
  `tol` the solver stops can differ ([`docs/fp32_policy.md`](docs/fp32_policy.md) §2.4).
  `shrinking=False` runs the plain loop.
- **Multi-output targets are accelerated** in `KNeighborsClassifier` and
  `KNeighborsRegressor`: the neighbor search does not depend on `y`, and votes and averages
  are taken per output. The Layer 1 tags now advertise multi-output (and, for the
  classifier, multi-label) targets.
- **Patched low-dimensional KNN goes to scikit-learn's KD-tree.** Under `patch_sklearn()`,
  fits with `algorithm="auto"` where scikit-learn would build a KD-tree are handed to it
  when `n_samples >= 512 * 2**n_features` (diagnostics reason `"tree-wins"`, private
  tuning knob `MLXLEARN_KNN_TREE_DENSITY`). Before, low-dimensional data such as
  200 000 × 3 ran up to 100× slower than stock. A directly imported class still searches
  by brute force.
- **Patched fits below the crossover are decided from `X.shape`**, before any data scan or
  copy, for all three estimator families, and record a single event. A consequence: a
  sub-crossover fit whose only mismatch is data-dependent — values past 3.4e38, or for
  `LogisticRegression` a multi-label `y` or negative weights — is recorded as
  `"below-crossover"` and no longer raises under `fallback_policy="raise"`.
- **`set_config` is process-wide** and `config_context` is context-local (see **Fixed**).
- Patched `SVC` routes one-vs-one problems with many small classes to scikit-learn: the
  mean pair's work, `(2n/K) · n_features`, must also clear a new private
  `svc_min_pair_work` threshold (default 250 000, `MLXLEARN_SVC_MIN_PAIR_WORK`).
- The logistic crossover's work proxy counts one weight vector per class
  (`n × d × n_targets`). The threshold values are unchanged.
- `metric_params={"p": ...}` with `metric="minkowski"` is accepted and resolved as
  scikit-learn resolves it, with its `SyntaxWarning`. Other `metric_params` still raise.
- The neighbor estimators validate `X` and queries as float64 or float32, keeping float64
  for the index to translate, instead of casting to float32 first.
- `radius_neighbors` and `radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`
  compute with a float32 query (see **Performance**), so a float64 query float32 cannot
  hold raises `UnsupportedInputError` (reason `"float32-overflow"`, a `ValueError`), as
  `kneighbors` does. 0.1.0a2 upcast it and answered in float64.
- On the GPU, kernel rows evaluated on demand are bit-identical to the rows of the
  materialized Gram, so `cache_size` crossing the Gram boundary no longer changes the
  fitted model there (measured on all four kernels; MLX's CPU device still moves decision
  values by up to 2.2e-3 at 2 000 × 10).
- `mlx_guard` lets only mlxlearn's own errors, `MemoryError` and escalated warnings pass
  through. A deliberate input error detected inside guarded code — `SVC`'s non-finite
  model, a class whose every sample has zero weight — is an `InvalidInputError`, which is
  a `ValueError`, so scikit-learn-facing behavior is unchanged.
- `import mlxlearn; mlxlearn.svm.SVC` works: the estimator subpackages load lazily on first
  access, and `import mlxlearn` still imports neither scikit-learn nor MLX.
- An `mx.array` converted to NumPy through mlxlearn's array helpers is a read-only view,
  and `check_finite_float32` now actually checks float32 data after a cast.

#### Memory

- The neighbor query's memory budget now bounds its whole working set — tile, selection
  temporaries and refinement gather — instead of the distance tile alone.
- A fitted neighbors model keeps one private copy of the training rows, shared read-only
  between host and device (`_fit_X` is a view of it), beside the translated copy the search
  runs on; fitting no longer allocates full-size host copies or temporaries. float64 input
  additionally keeps a float32 residual.
- The fitted `SVC` keeps its support vectors mapped into the compute frame and resident on
  the device, built lazily on the first prediction and dropped from pickles.

#### Packaging, CI and benchmarks

- The MLX floor is raised to 0.31.2: concurrent fits abort the process on earlier
  releases.
- The version is defined once, in `src/mlxlearn/__init__.py`; `pyproject.toml` reads it.
- `pytest` deselects tests marked `slow` by default (`pytest -m slow`, `pytest -m ""`).
- Without a working GPU, tests that need one are skipped with a reason.
  `MLXLEARN_REQUIRE_GPU=1`, set in CI, makes a missing GPU stop the run instead.
- Benchmark significance requires the medians to be three standard errors apart, each
  estimated from a real interquartile range, with at least five samples on each side.
- The performance gate judges the backend each case actually ran on under the configured
  `Tuning`.
- The stock KNN baseline in the benchmark grid is scikit-learn's default
  `algorithm="auto"`.
- ruff is pinned, in the dev extra, and CI lints with that pin.

#### Documentation

- The README states precisely what `unpatch_sklearn()` restores, that scikit-learn's
  internal users of the patched names are deliberately not accelerated, where each
  configuration setting reaches (threads, `asyncio`, process workers), what `device`
  selects, the error taxonomy, and the two cases inference runs off its fitted device.
- [`docs/fp32_policy.md`](docs/fp32_policy.md) describes the neighbor certificate and tie
  contract, float64 translation in each estimator, `SVC`'s compute frame and shrinking,
  the L-BFGS float32 floor, and range checking at inference. `deterministic` and
  `random_state` are documented as having no observable effect in 0.1.x, and Rule C as
  reserved.
- [`docs/support_matrix.md`](docs/support_matrix.md) lists what each estimator falls back
  for, which problems patched dispatch hands to scikit-learn by size, and the deliberate
  differences from scikit-learn.
- [`docs/development.md`](docs/development.md) documents the fit pattern, the staged fit
  event, and the rules against unevaluated `mx.array` state and plain `ValueError`s inside
  `mlx_guard`.

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
  `LogisticRegression`'s `penalty`, `l1_ratio=None`, `n_jobs` and, on 1.7, `multi_class`;
  `SVC`'s `probability` on 1.9.
- Diagnostics:
  - non-converged `LogisticRegression` fits are recorded as `"lbfgs"` events with the stop
    reason, `n_iter`, `n_fev`, `grad_max_abs`, `tol` and `at_resolution_floor`;
  - new reasons `"tree-wins"`, `"prebuilt-index"`, `"gpu-unavailable"`,
    `"forked-process"`, `"array-api-dispatch"` and, for a `LogisticRegression` predicting
    from assigned attributes, `"external-state"`;
  - inference events carry the fit's reason as `extra["fit_reason"]`;
    `LogisticRegression`'s carry `extra["compute"] = "numpy-float64"`, since it predicts
    with scikit-learn's float64 code on every backend;
  - a neighbor query that needed the certificate's second pass or its exact search
    reports how many rows did in its event's `extra` (`search="certificate-fallback"`,
    `widened_rows`, `exact_rows`).
- Private tuning knobs `knn_tree_density`, `svc_min_pair_work` and
  `mlx_cache_release_fraction`, each with its `MLXLEARN_*` variable.
- Internal validation and array helpers: `validate_float32_input`,
  `require_nonnegative_weights`, `check_sample_weight(allow_negative=...)`,
  `finite_range` and `ValueRange`, and `check_squared_float32_range`, a guard for data
  whose squares overflow or underflow float32 (reasons `"float32-overflow"` and
  `"float32-underflow"`).
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
- Python 3.14 support (classifier and CI leg).
- A minimum-versions CI leg that installs every declared floor exactly, a weekly job
  against upstream pre-releases, and Dependabot configuration.
- The `layering` compliance gate: no scikit-learn, user-configuration or estimator imports
  in `_solvers`/`_kernels`, and no global RNG seeding anywhere.
- `tools/check_environment.py`, which fails a CI job whose versions or device are not what
  it claims.
- The sdist carries the test suite, `docs/`, `CHANGELOG.md` and `ACKNOWLEDGMENTS.md`.
- CI runs doctests, the benchmark harness and gate at one iteration, and the slow tests in
  a job of their own.
- The benchmark `published` profile, which reproduces `docs/benchmarks.md`, and a
  `--force-backend` flag. New benchmark cases: low-dimensional KNN (`d` = 2, 3), large
  query batches (10 000 and 50 000), and many-class SVC.
- README notes on fork safety, MLX's buffer cache, and pickles of patched models needing
  mlxlearn to load.
- A `Documentation` project URL.

### Removed

- The unused `NotSupportedByBackend` alias in `mlxlearn.exceptions`.
  `MLXLearnPerformanceWarning` and `UnsupportedEstimatorStateError` stay, documented as
  reserved: nothing in 0.1.x emits or raises them.
- The benchmark harness's cold-start measurement, which 0.1.0a1 documented
  ("cold-start / fit / predict reported separately") but never implemented.
- `wheel` from the build-system requirements, and the `MLXLEARN_CI` variable from CI,
  which nothing read.

### Security

- Workflow actions are pinned by commit SHA, workflows run with least-privilege token
  permissions, and every job has a timeout.
- The release workflow published without running the tests, skipped its tag check on a
  manual dispatch, and never checked the wheel it described. It now runs the whole CI
  workflow on the tagged commit, checks that the wheel is pure-Python and installs
  cleanly, requires the tag, the wheel metadata and `mlxlearn.__version__` to agree, and
  publishes only on a `v*` tag push, through PyPI Trusted Publishing from an environment
  that requires approval. A manual run is a dry run.
- The license gate walks the full installed runtime dependency closure and parses SPDX
  expressions. It used to check direct dependencies only, by substring, and accepted, for
  example, "Proprietary; unlicensed".

### Performance

Every number here is an interleaved A/B measurement by the change's author on a shared
M4 Max, against the 0.1.0a2 code. They are preliminary; the benchmarks and crossovers are
re-measured on fixed hardware before release.

- Neighbor queries on the GPU use two-stage segmented selection and fused tile
  expressions: large queries are 2.0–2.4× faster — 100 000 × 32 with 20 000 queries from
  956 to 403 ms, 200 000 × 64 with 4096 queries from 485 to 205 ms, and `k=200` from 246
  to 154 ms. (preliminary; re-measured before release)
- Binary and small-integer data takes an exact integer path, about 1.7× faster; index
  build is about 2× faster. (preliminary; re-measured before release)
- Peak device memory per neighbor query stays near the budget instead of reaching about
  7× it: 7 409 MiB → 762 MiB at 50 000 × 768 with `k=1000`. (preliminary; re-measured
  before release)
- Certification costs a second pass only where the old margin was wrong: 1.5–2.3× the old
  (wrong) query time on the pathological cases in
  [`docs/fp32_policy.md`](docs/fp32_policy.md) §2.1, and no second pass on ordinary data.
  (preliminary; re-measured before release)
- Patched low-dimensional KNN handed to the KD-tree: 200 000 × 3 fit and predict went from
  0.12× to 0.99–1.00× of stock. (preliminary; re-measured before release)
- `radius_neighbors` and `radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`
  hand scikit-learn a float32 query and run at stock speed; they took about 5× as long.
  (preliminary; re-measured before release)
- Neighbor votes use a single `np.bincount`: 2–3× faster and bitwise identical.
  (preliminary; re-measured before release)
- `LogisticRegression`'s L-BFGS bookkeeping (two-loop recursion, inner products,
  curvature test, `max|g|`) runs in NumPy float64 on the host, so each objective
  evaluation costs one device synchronization instead of 12–33 blocking round trips per
  iteration. Forced-GPU fits are 1.7–8× faster with iteration counts unchanged, and the
  solver alone 2–11× faster. The binary gradient on the GPU uses a contiguous `Xᵀ`,
  avoiding MLX's slow vector–matrix path. (preliminary; re-measured before release)
- `SVC` fits no longer fall off a cliff once a one-vs-one subproblem exceeds the
  kernel-cache Gram cutoff (about 7 240 rows at the default `cache_size=200`): on a cache
  miss, kernel rows are fetched many per GPU round trip, together with the rows closest to
  becoming the next working pair. rbf fits at 8 000 × 128 are 2.4× faster, 12 000 × 128
  1.9×, and 16 000 × 64/256 1.5–1.8×, with results unchanged. (preliminary; re-measured
  before release)
- Shrinking, now honored and on by default, is a further 1.2–1.8× on those fits in the
  solver's own A/B; with prefetch, 2.0–3.0× at 8 000–16 000 rows and 1.5× at 32 000 × 64.
  Measured through `SVC`, off against on: rbf 1.02–1.37× (1.17× at 8 000 × 64, 1.37× at
  16 000 × 32, 1.02× at 4 000 × 256), poly 0.97×. It can cost on the linear kernel, as
  LIBSVM's own shrinking does: at 2 000 × 20, `C=1` is 1.58× faster, but `C=10` runs at
  0.72× on MLX's CPU and 0.91× on the GPU, where LIBSVM's iterations double too (442 000
  against 223 000). Patched dispatch never sends the linear kernel to MLX.
  (preliminary; re-measured before release)
- `SVC` inference: the rbf kernel tile is one fused GPU kernel, bit-identical to the
  unfused evaluation, and `decision_function`/`predict` overlap the device with the
  float64 host contraction, in one tile loop shared with the kernel helpers. GPU predict
  is 1.2–1.5× faster and CPU predict 1.0–1.16×, with decision values bit-identical; with
  the support vectors kept on the device, single-row `predict` at 4 000 × 256 went from
  0.50 to 0.25 ms and 16 rows from 0.62 to 0.35 ms. (preliminary; re-measured before
  release)
- Many-class `SVC` under `patch_sklearn()`: at 2048 × 512, `K=64` went from 0.23× to 1.00×
  of stock, and `K=8` stays on MLX at about 1.6×. (preliminary; re-measured before
  release)
- Patched fits below the crossover pay less before handing the work to scikit-learn:
  KNN 1.16× of the stock fit instead of 1.52×, `LogisticRegression` 1.00–1.08× instead of
  1.2–1.6×, `SVC` 1.12× instead of 1.47× at 200 × 5. (preliminary; re-measured before
  release)
- The float32 range check is about 3.8× faster and allocates no temporaries on finite
  input: 19 ms instead of 72 ms at 128 000 × 768, with peak extra memory near 0 instead
  of 1.6 GiB. `NearestNeighbors` fit on float64 input at 128 000 × 768 went from 169 ms to
  98 ms. (preliminary; re-measured before release)
- Costs: `patch_sklearn()` with every estimator takes about 90 ms longer on a cold start
  because of the consumer imports (patching only `SVC`, `KNeighborsClassifier` or
  `KNeighborsRegressor` costs nothing extra), and `unpatch_sklearn()`'s sweep about 1 ms.
  Where the buffer-cache release triggers — machines with less memory — repeated
  same-shape large queries pay reallocation: a repeated 150 000 × 64 query with 6 000
  queries measured 470 → 618 ms with the release forced on every call.
  (preliminary; re-measured before release)

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
