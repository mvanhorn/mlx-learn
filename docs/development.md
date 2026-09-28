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
def fit(self, X, y):
    if self._fit_something(X, y) == "sklearn":   # only reachable from Layer 2
        super().fit(X, y)
        self._set_backend("sklearn")
        return self
    ...                                          # MLX path
    return self
```

and of every inference method:

```python
def predict(self, X):
    check_is_fitted(self)
    if self._fitted_backend("predict") == "sklearn":
        self._note_inference("predict")
        return super().predict(X)
    ...
```

Inference never re-decides the backend. That is the sticky invariant, and it is what stops
an MLX-fitted model from wandering into scikit-learn's inference path holding state that
path cannot read.

Wrap every MLX block in `mlx_guard(...)` so an unexpected failure becomes
`BackendExecutionError` rather than a silent fallback.

### The gates, per estimator, before merge

1. `parametrize_with_checks` from scikit-learn, with any xfail justified under a named rule
   in [`fp32_policy.md`](fp32_policy.md) §4 and listed in `EXPECTED_XFAILS`.
2. `clone`, `get_params`/`set_params` round-trip, `Pipeline`, `GridSearchCV`,
   cross-validation, `pickle`, `joblib`, refit, `n_features_in_`.
3. Numerical parity against stock scikit-learn at the tolerances in `fp32_policy.md` §3 —
   including the objective value, not only the coefficients. Two correct solvers land in
   different places on an ill-conditioned problem, so a coefficient-only test fails for a
   correct implementation and passes for a subtly wrong one.
4. Precise errors on the Layer 1 path; recorded fallbacks on the Layer 2 path.
5. Sticky-backend tests.
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
  both are skipped with a reason rather than failing by the dozen. CI sets
  `MLXLEARN_REQUIRE_GPU=1`, under which a missing GPU stops the session instead, so a
  runner that loses Metal cannot skip its way to green. The policy is `tests/_gpu.py`.
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
overhead loses, both fail. A difference counts only when the medians are three standard
errors apart, each estimated from a real interquartile range.

Shared CI runners do correctness only: CI runs the smoke profile with one iteration, which
checks the harness, the gate, and the result schema, and can never be significant.
Absolute timings come from the nightly job on fixed self-hosted Apple hardware, because a
timing gate on a shared, throttled, heterogeneous runner measures the runner. Its schedule
waits for the repository variable `MLXLEARN_BENCH_RUNNER` to be `true`, which says the
`mlxlearn-bench` runner exists; a manual dispatch always runs.

Everything is `mx.eval`-synchronized — MLX is lazy, so an unsynchronized benchmark times
how long it took to *schedule* the work — warmed up, reported as a median with its spread,
and compared against scikit-learn in interleaved pairs so a thermal ramp degrades both
sides equally rather than whichever ran second.

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
5. Tag `vX.Y.Z`. The release workflow runs the whole CI workflow on the tagged commit,
   builds, checks that the wheel is `py3-none-any`, installs it into a clean environment,
   and requires the wheel's filename, its metadata, `mlxlearn.__version__` and the tag to
   agree. Only then does it publish, through PyPI Trusted Publishing from a `pypi`
   environment that requires manual approval. No API token is stored in this repository.
   Publishing takes both a `v*` tag push and that approval; a manual run of the workflow
   is a dry run that never publishes.
