# Development

```bash
git clone https://github.com/tabulai/mlx-learn
cd mlx-learn
pip install -e ".[dev]"
pytest
```

`pytest` runs the fast suite. Tests marked `slow` are deselected by default (`addopts` in
`pyproject.toml`); `pytest -m slow` runs just those, and `pytest -m ""` runs everything. CI
runs the slow tests in a job of their own on every PR, so deselecting them locally never
means nobody runs them.

## Layout

```
src/mlxlearn/
├── _common/     configuration, device, arrays, validation, RNG, diagnostics, dispatch
├── _kernels/    MLX array operations; knows nothing about estimators
├── _solvers/    typed config in, typed state out; never sees an estimator or scikit-learn
├── neighbors/ linear_model/ svm/   Layer 1 estimators
└── patching/    Layer 2 — the only package allowed near scikit-learn's namespace
tests/  benchmarks/  docs/  tools/  phase0/
```

The layering is enforced, not merely described:

- A `_solvers` or `_kernels` module that imports `sklearn`, reads the user's configuration
  (`get_config`, `set_config`, `config_context`), or imports an estimator package fails the
  `layering` gate in `tools/compliance.py`. Solvers take arrays and a frozen config
  dataclass and return a frozen state dataclass, so the numerics can be tested without
  constructing an estimator. `_kernels` may read `get_tuning()` for block sizes, which are
  private typed knobs, not user configuration. The gate is static, so an indirect read —
  `use_device(None)` resolving the configured device — is still on the reviewer: solvers
  pass their config's device explicitly.
- A `sklearn._*` import outside `patching/` fails CI. Inside `patching/` it must be on an
  explicit allowlist in `tools/compliance.py`, which is currently empty.
- Layer 1 raises on a capability mismatch. Layer 2 falls back. The whole difference is
  `_mlxlearn_allow_fallback`, and any behavior that cannot be expressed as flipping that
  flag is a design error.

## Adding an estimator

**Subclass the scikit-learn estimator.** Never redefine `__init__`. Subclassing inherits
the exact constructor signature, `get_params`/`set_params`, `_parameter_constraints`, and
the estimator tags, so `clone`, `Pipeline`, `GridSearchCV` and `isinstance` keep working —
and `super().fit(...)` becomes an exact scikit-learn fallback with no extra machinery.

The shape of a `fit`:

```python
class MLXThingMixin(BackendMixin):
    # Fitted attributes only the MLX path creates. _clear_backend deletes them, so a
    # refit that ends on scikit-learn (or fails) does not keep the old device state
    # alive in memory and in pickles.
    _mlxlearn_mlx_attrs = ("_device_state_",)


def fit(self, X, y):
    self._validate_params()   # scikit-learn's constraints: overriding fit bypasses its decorator
    self._clear_backend()     # forget the last fit; the estimator is unfitted until _set_backend
    backend = self._select_backend(
        "fit",
        shape=_safe_shape(X),                          # the raw shape, before any conversion
        crossover=self._below_crossover,               # (n_samples, n_features) -> (below, detail)
        capability_check=lambda: self._check_params(X),   # parameters and sparsity: cheap
        data_check=lambda: check_float32_range(X),        # O(n·d) scans: skipped on a handoff
    )
    if backend == "sklearn":                           # only reachable from Layer 2
        super().fit(X, y)
        self._set_backend("sklearn")
        return self
    X, y = validate_data(self, X, y, ...)
    with mlx_guard(type(self).__name__, "fit"):
        state = solve(X, y, ...)                       # locals only, evaluated before return
    self.coef_ = state.coef                            # publish
    self._set_backend(backend)                         # the commit point: the one fit event
    return self
```

`_select_backend` decides in a fixed order: the capability check, then
`MLXLEARN_FORCE_BACKEND`, then size, then the data checks, then the device. The order is
the point. "We cannot do this at all" outranks "this is small", so a capability mismatch
is always reported as one. And when patched mode is about to hand a fit to scikit-learn
by size, the decision comes from `X.shape` alone, before any validation, scan or copy —
scikit-learn validates the data itself, and scanning it first made patched fits slower
than the stock ones they replaced. Input with no shape before validation (a list) needs
the size decision made again once it has one. The three 0.1.0 estimators implement this
ordering in their own `fit` rather than through the `crossover=`/`data_check=` keywords,
which a new estimator should use.

`_select_backend` may run several times in one fit; it only *stages* its decision.
`_set_backend(backend)`, called exactly once at the end of every successful fit, is the
single commit point: it records the one fit event, with its warning if it is a fallback,
and keeps it on the estimator as `fit_event_`. A fit that raises before it leaves the
estimator unfitted, and inference then raises `NotFittedError`.

And the shape of every inference method:

```python
def predict(self, X):
    check_is_fitted(self)
    if self._fitted_backend("predict") == "sklearn":
        self._note_inference("predict")
        return super().predict(X)
    ...                                                # _note_inference where the work is routed
```

`BackendMixin` wraps every public inference method a subclass defines or inherits as a
plain function (`predict`, `predict_proba`, `decision_function`, `kneighbors`, and
scikit-learn's own `score`, `kneighbors_graph`, `radius_neighbors_graph`, …), so one
public call records one event, named after the method the caller invoked, however many of
the others it reaches internally. `available_if` descriptors are left alone, because
wrapping one would make the method exist on estimators that do not have it; a class whose
`available_if` method must record its call redeclares it as
`available_if(check)(_public_call(name, fn))`, as `SVC` does for `predict_proba` and
`predict_log_proba`. An inference path that records something other than a sticky event —
a fallback at inference — claims the call's one event with `_claim_call_event`.

Inference never re-decides the backend. That is the sticky invariant, and it is what stops
an MLX-fitted model from wandering into scikit-learn's inference path holding state that
path cannot read. There is one carve-out, and it is Layer 2's: a sparse query to
`kneighbors`, `kneighbors_graph`, `predict` or `predict_proba` of a neighbor estimator
fitted on MLX is answered by scikit-learn's brute-force search, which reads the fitted
state the MLX path populates for it (`_fit_X`, `_fit_method="brute"`,
`effective_metric_`), and is recorded as a fallback. The query is cast to `_fit_X`'s
float32 first: scikit-learn's search needs the two to share a dtype, and a float64 query
fell to a slower pairwise path that cannot compute `"sqeuclidean"` on sparse input. A
GPU-fitted model running on MLX's CPU in a process without a GPU (`inference_device`) is
not a carve-out: it is the same code on another MLX device, recorded as
`"gpu-unavailable"`. Inference that is host code on the fitted state
(`LogisticRegression`'s, `NearestNeighbors.radius_neighbors` and
`radius_neighbors_graph`) uses no device: `_note_inference(..., host=True)` records it as
`"sticky"` and resolves no device, so `device="gpu"` has nothing to refuse. The radius
methods take a sparse query that way too, in both layers — scikit-learn's code serves
it — so it is an ordinary sticky call, not a fallback, and `fallback_policy` does not
apply to it.

Wrap every MLX block in `mlx_guard(...)` so an unexpected failure becomes
`BackendExecutionError` rather than a silent fallback.

### The gates, per estimator, before merge

1. scikit-learn's estimator checks — `check_estimator(on_fail=None)` over both layers, at
   the default configuration and with the MLX GPU path forced, plus
   `check_param_validation` — in `tests/test_sklearn_compliance.py`. Any xfail is
   justified under a named rule in [`fp32_policy.md`](fp32_policy.md) §4 and listed in
   `EXPECTED_XFAILS`.
2. `clone`, `get_params`/`set_params` round-trip, `Pipeline`, `GridSearchCV`,
   cross-validation, `pickle`, `joblib`, refit, `n_features_in_`. `tests/unit/test_gates.py`
   runs `joblib` (memory-mapped too), process-parallel cross-validation and a `Pipeline`
   inside `GridSearchCV` for every registered estimator in both layers; the check suite
   covers clone, parameters, refit and `n_features_in_`, and
   `tests/unit/test_invariants.py` pickling and fit purity.
3. Numerical parity against stock scikit-learn at the tolerances in `fp32_policy.md` §3 —
   including the objective value, not only the coefficients. Two correct solvers land in
   different places on an ill-conditioned problem, so a coefficient-only test fails for a
   correct implementation and passes for a subtly wrong one. `tests/parity/`.
4. Precise errors on the Layer 1 path; recorded fallbacks on the Layer 2 path — exactly one
   fit event per fit, matching `execution_backend_` and `fit_event_`, and one event per
   public inference call, named after it (`tests/unit/test_runtime.py`, and
   `tests/unit/test_event_invariants.py` over every fit path of every estimator in both
   layers).
5. Sticky-backend tests: inference reuses whatever `fit` chose
   (`tests/unit/test_gates.py`; `tests/unit/test_event_invariants.py` for the event each
   inference call records, GPU-less hosts included).
6. A benchmark entry in `benchmarks/run.py` with a measured crossover wired into
   `_common/config.py`'s `Tuning`.

An estimator is registered in `patching/_registry.py` only after all six pass. Registering
early means silently swapping something untested into someone's `sklearn` namespace.

## Things that are not allowed

**No module-level mutable state, and no cache keyed on data.** The ancestor kept weakrefs
to the last training and query matrices so it could skip re-uploading them, plus `id()`-keyed
LRU caches in the estimator layer. CPython recycles `id()` after a garbage collection, so a
freed array whose address was reused produced predictions computed on the wrong data,
silently. Everything cached lives on the fitted estimator and dies with it.

**No aliasing of a caller's array.** `fit` copies. An estimator that holds a view of the
caller's buffer changes its own predictions when the caller mutates their array.

**No broad `except Exception` around accelerated work.** The ancestor's dispatcher caught
everything and reran on scikit-learn, so a `ValueError: Converting -1 to uint32 would result
in overflow` from a padding bug reached users as a slightly slow but correct answer. That
bug survived for exactly as long as the `except` did.

**No plain `ValueError` or `TypeError` for bad input inside `mlx_guard`.** The guard lets
only the `MLXLearnError` family, `MemoryError` and escalated warnings through; everything
else — including the `ValueError`s and `TypeError`s MLX raises for its own shape and
argument errors — becomes `BackendExecutionError`. A deliberate input error detected
inside a guarded block is therefore an `InvalidInputError`, which is a `ValueError`, so
scikit-learn-facing behavior is unchanged. A capability limit discovered there is a
`CapabilityError` as usual, but `_select_backend` has already returned, so nothing turns
it into a fallback: find it before, or route it through `_handle_capability_error` by
hand, as `SVC` does for its data-dependent limits and the neighbor estimators for a
training span their index cannot hold.

**No `patch_sklearn()` advice in a capability message.** A `CapabilityError`'s message says
what cannot be served; the remedy for a directly imported estimator ("run under
`patch_sklearn()`", "compute these with scikit-learn") goes in `advice=`. Layer 1 raises
both; a patched estimator, already under `patch_sklearn()`, warns and records `.summary`
followed by what it did. An error at inference, where neither layer can fall back, advises
no fallback.

**No unevaluated `mx.array` on shared state.** MLX is lazy: an `mx.array` is a graph until
something evaluates it. One published on an estimator or an index that other threads read
— a device-resident training copy, a prepared operand — gets evaluated by whichever thread
touches it first, concurrently with the others. Concurrent first queries on an unpickled
neighbors model could segfault while two threads built the same device state, and the
rule has since caught a lazy transpose published on an index. Build device state into
locals, `mx.eval` it, and publish it with one assignment, under a lock if it is built
lazily, so a reader sees either nothing or the finished state.

**No data scan before a by-design handoff.** A patched fit headed for scikit-learn on size
alone must cost what scikit-learn costs. Decide from the shape, and keep the O(n·d)
capability checks — float32 range, target type, weight signs — in `data_check`, which
does not run on that path.

**No approximation under a scikit-learn estimator name.** If an estimator cannot implement
the objective its name promises, it does not ship under that name.

**No silently ignored parameters.** The ancestor accepted `metric_params` and computed
unweighted Euclidean distances anyway. A wrong answer delivered quickly is the worst
outcome available.

**No `mx.random.seed`.** It is process-global and belongs to the application, not to a
library it imported. Use explicit keys through `_common/rng.py`. The `layering` gate rejects
any `*.random.seed` — MLX's, NumPy's or the standard library's — anywhere in the package.

## Compliance

```bash
python tools/compliance.py                     # all gates
python tools/compliance.py licenses --json     # some gates; stdout is pure JSON
```

Six mechanical gates — branding, private imports, upstream headers, dependency licenses,
layout, and layering — run on every push. The license gate walks the whole runtime
dependency closure as installed, not only the direct dependencies, and parses SPDX
expressions rather than matching substrings; a package whose metadata cannot be read
mechanically needs a reviewed entry in `LICENSE_OVERRIDES`, with a reason.

## Testing

- **GPU.** Tests that need a working Metal device either carry `@pytest.mark.gpu` or reach
  the GPU implicitly and fail with `DeviceError` without one. On a machine with no GPU,
  both are skipped with a reason rather than failing by the dozen. The implicit skip only
  catches a `DeviceError` that reaches pytest unwrapped — not one inside `GridSearchCV`'s
  "all fits failed" — and without a GPU most MLX work raises nothing: auto-selection picks
  the CPU, and a forced MLX fit runs on MLX's CPU device. So a test that asserts the MLX
  path ran must carry the marker. The slow test `tests/tooling/test_gpu_policy.py` runs the
  default suite on a simulated GPU-less host to hold every test to that. CI sets
  `MLXLEARN_REQUIRE_GPU=1`, under which a missing GPU stops the session instead, so a
  runner that loses Metal cannot skip its way to green. The policy is `tests/_gpu.py`.
- **pandas** is a test dependency (`pandas>=2.2.3` in the `test` and `dev` extras, the floor
  installed exactly by the minimum-versions job), so the tests of how mlxlearn reads data
  frames run on every CI leg. torch is not: the torch container cases in
  `tests/unit/test_validation.py` run only where it is installed — the nightly job, and
  local runs.
- **Doctests** run as their own CI step: `pytest --doctest-modules src/mlxlearn`.
- **`tests/tooling/`** tests the repository's tooling — this file's gates, the benchmark
  harness and gate, the workflows' pins — and is left out of the sdist with the tooling.
- **`tools/check_environment.py`** prints the versions and device a CI job is testing and
  fails if they are not what the job claims: unmet requirements, a stale install whose
  metadata disagrees with `mlxlearn.__version__`, or (`--constraints`) a pinned floor that
  did not install exactly.

## Benchmarks

```bash
python -m benchmarks.run --profile smoke
python -m benchmarks.run --profile full --output bench/results.json --report bench/report.md
python -m benchmarks.run --profile published --report bench/published.md
python -m benchmarks.gate --results bench/results.json
```

`smoke` is small enough for a laptop and still clears every `Tuning` threshold once;
`full` is the regression grid; `published` holds the cases `docs/benchmarks.md` reports,
with the settings they were measured under (Layer 1, forced onto MLX).

Each result records the backend the mlxlearn side ran on, and the gate judges those
decisions — the configured `Tuning` — rather than a crossover inferred from the run it is
judging: a problem `Tuning` sends to MLX that loses, and a patched handoff whose own
overhead loses, both fail. A backend forced with `--force-backend` or with
`MLXLEARN_FORCE_BACKEND` is recorded as forced, the same way, so the gate does not charge
it to `Tuning`; the environment record includes the effective `Tuning` and every
`MLXLEARN_*` variable. A difference counts only when the medians are three standard
errors apart, each estimated from a real interquartile range. That bounds the false-alarm
rate per case; a run gates over a hundred cases, so the runner re-measures any case the
gate would fail over three times the iterations, records it as the result's
`confirmation`, and the gate fails the case only if the loss reproduces there. A case's
own iteration count caps `--iterations`; it never raises it.

Shared CI runners do correctness only: CI runs the smoke profile with one iteration, which
checks the harness, the gate, and the result schema, and can never be significant.
Absolute timings come from the nightly job on fixed self-hosted Apple hardware, because a
timing gate on a shared, throttled, heterogeneous runner measures the runner. Its schedule
waits for the repository variable `MLXLEARN_BENCH_RUNNER` to be `true`, which says the
`mlxlearn-bench` runner exists; a manual dispatch always runs.

Everything is `mx.eval`-synchronized — MLX is lazy, so an unsynchronized benchmark times
how long it took to *schedule* the work — warmed up, reported as a median with its spread,
and compared against scikit-learn in interleaved pairs so a thermal ramp degrades both
sides equally rather than whichever ran second. Within a pair the two sides take turns
going first, and every timed call gets the same preparation, whichever backend it runs
on: a garbage collection before it and a synchronization inside its window.

## Releasing

1. Every gate green; `python -m benchmarks.run --profile full` re-run and
   `docs/benchmarks.md` refreshed.
2. `CHANGELOG.md` updated.
3. Version bumped in `src/mlxlearn/__init__.py` — the only place it lives. `pyproject.toml`
   reads it (`dynamic = ["version"]`), statically, so a build never imports MLX. After a
   bump, reinstall (`pip install -e .`): an editable install's metadata is frozen at
   install time and otherwise keeps reporting the old version.
4. README links stay absolute (`https://github.com/tabulai/mlx-learn/...`): the README is
   also the PyPI page, where a relative link is a 404. `tests/tooling/` checks this.
5. Tag `vX.Y.Z`. The release workflow runs the whole CI workflow on the tagged commit and
   does not build anything itself: CI's build job builds the `dist` artifact and checks
   it — a `py3-none-any` wheel that installs into a clean environment with metadata that
   agrees with `mlxlearn.__version__`, an sdist that carries the test suite, and that
   suite collecting against the installed wheel. The release workflow's `version` job
   then requires the wheel's filename, its metadata, the sdist's filename and the tag to
   agree. Only then does it publish exactly those files, through PyPI Trusted Publishing
   from a `pypi` environment that requires manual approval. No API token is stored in this
   repository. Publishing takes both a `v*` tag push and that approval; a manual run of
   the workflow is a dry run that never publishes.
