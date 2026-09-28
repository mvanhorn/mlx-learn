# mlxlearn

**Classical machine learning on Apple silicon, accelerated with [MLX](https://github.com/ml-explore/mlx), with a scikit-learn compatible API.**

> mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.

`mlxlearn` has two layers.

**Layer 1 — the library.** Explicit imports, typed solver configuration and state, strict
errors, eager `fit`, no mutation of scikit-learn, no data-keyed caches:

```python
from mlxlearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

**Layer 2 — the patching shell.** A thin adapter that points scikit-learn's own names at
Layer 1 estimators, so existing code — including code an LLM generated against the
scikit-learn API — runs accelerated without being edited:

```python
from mlxlearn import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC          # now resolves to the accelerated class
```

The numerical library is fully usable and fully tested with the patching mechanism absent.

## Status

**`0.1.0a2` — alpha.** The API may change. See [CHANGELOG.md](https://github.com/tabulai/mlx-learn/blob/main/CHANGELOG.md).

| Estimator | Status |
|---|---|
| `mlxlearn.neighbors.NearestNeighbors` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsClassifier` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsRegressor` | 0.1.0 |
| `mlxlearn.linear_model.LogisticRegression` | 0.1.0 |
| `mlxlearn.svm.SVC` (exact) | 0.1.0 |
| `LinearRegression`, `Ridge`, `PCA`, `KMeans`, `DBSCAN`, `TSNE` | planned, 0.2.x |
| `SVR`, `NuSVR`, `NuSVC` | deferred — will not ship until they implement the true objective |
| tree ensembles | out of scope |

## Requirements

- macOS on Apple silicon (arm64)
- Python 3.10 – 3.14
- scikit-learn ≥ 1.7, MLX ≥ 0.31.2

Python 3.10 is supported with scikit-learn 1.7.x; scikit-learn 1.8+ requires Python ≥ 3.11.
CI tests only valid pairs, plus every declared minimum version at once — see
[`docs/support_matrix.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/support_matrix.md).

## Install

```bash
pip install mlx-learn
```

`0.1.0a2` ships as a pure-Python wheel — no compiler, no build step, no ABI matrix.

## Import-order semantics

Patching replaces attributes **on scikit-learn modules**. Attribute access
(`sklearn.svm.SVC`) therefore resolves to the mlxlearn class whether `import sklearn`
happened before or after `patch_sklearn()`.

Symbols captured *before* patching are local bindings and cannot be rebound:

```python
from sklearn.svm import SVC     # binds the stock class into your namespace
patch_sklearn()                 # cannot reach back and change SVC
SVC()                           # still the stock class
```

That is Python name binding, not a bug in mlxlearn. **The supported pattern is patch-first.**
`patch_sklearn()` is idempotent, and `unpatch_sklearn()` restores every scikit-learn module
attribute mlxlearn changed.

Only the public names are patched. scikit-learn's own users of them — `DBSCAN`, `OPTICS`,
`TSNE`, `Isomap`, `mutual_info_classif`, `StackingClassifier` and the like — keep the stock
classes whatever the import order: those algorithms have not been through mlxlearn's gates,
and some depend on float64-exact distances. Names bound *outside* scikit-learn while patched
(your own `from sklearn.svm import SVC`, or another library doing the same) are ordinary
bindings and keep the mlxlearn class after `unpatch_sklearn()`.

## When mlxlearn falls back

Layer 1 raises a precise error when it cannot honor a request. Layer 2 falls back to stock
scikit-learn instead — but only for *capability* mismatches (sparse input, an unsupported
parameter, an unsupported dtype), never to hide a bug:

| Situation | Direct import | Patched |
|---|---|---|
| Capability mismatch | precise `MLXLearnError` | falls back, warns once per class, records diagnostics |
| Problem below the measured crossover | internal CPU path | dispatches to scikit-learn |
| Invalid user input | sklearn-equivalent validation error | scikit-learn's own exception, unmasked |
| Unexpected MLX runtime failure | raises | **raises** — never silently rerun on scikit-learn |

Set the policy with `fallback_policy="warn" | "raise" | "silent"`. Diagnostics are recorded
in every mode:

```python
import mlxlearn

with mlxlearn.config_context(fallback_policy="raise"):
    ...                                     # any capability fallback becomes an error

mlxlearn.get_last_backend_event()           # what the last fit/predict actually did
mlxlearn.get_backend_diagnostics()          # everything recorded this session
```

**Sticky backend.** The backend is chosen during `fit` and exposed as
`estimator.execution_backend_` (`"mlx"`, `"cpu"`, or `"sklearn"`; like any fitted attribute,
it raises `AttributeError` before `fit`). Every subsequent
`predict` / `predict_proba` / `transform` / `kneighbors` uses the backend the model was
fitted with. A model fitted on MLX can never wander into scikit-learn inference with
incompatible state. A new `fit` clears state and may pick a different backend.

## Configuration

The public surface is deliberately small:

```python
mlxlearn.set_config(
    device="auto",              # "auto" | "gpu" | "cpu"
    fallback_policy="warn",     # "warn" | "raise" | "silent"
    output_type="numpy",        # 0.1.0: NumPy in, NumPy out
    deterministic=True,
    random_state=0,
    diagnostics=True,
)
```

Crossover thresholds, block sizes, and solver tuning are private, typed, and overridable
only through `MLXLEARN_*` environment variables for CI and debugging. They are not part of
the public API.

## Processes, memory, and pickles

**Do not `fork` after using mlxlearn.** MLX, like Metal underneath it, is not fork-safe.
Once a process has loaded MLX or scikit-learn's estimators — the first `fit` or `predict`
does, and so does `patch_sklearn()` — a child it creates with a bare `fork()` crashes on
its first MLX call, `device="cpu"` included, or hangs inside a `multiprocessing` pool. The
safe patterns are the defaults: Python's `spawn` start method (the macOS default) and
`forkserver`, and scikit-learn's `n_jobs`, whose `loky` workers are spawned. What breaks
is choosing `fork` explicitly — `multiprocessing.get_context("fork")`, `os.fork()`,
Celery's prefork pool (use `solo`, `threads`, or spawned workers), or gunicorn `--preload`
of an app that has already fitted or loaded a model (load it in each worker instead).

**Release MLX's buffer cache in long-running processes.** MLX keeps freed device buffers in
a process-wide cache for reuse, and by default lets that cache grow to nearly the whole
memory limit. The cache belongs to the process, not to a model, so deleting a model does
not shrink it, and a service or a notebook that queries differently sized data can hold
several GiB of idle memory. Call `mlx.core.clear_cache()` after large jobs, or bound the
cache once at startup with `mlx.core.set_cache_limit(n_bytes)`. Both act on the whole
process, including any other MLX work in it, which pays in re-allocation, not correctness.

**Pickles need mlxlearn to load.** Every model mlxlearn fits is an mlxlearn object — under
`patch_sklearn()` too, where its repr shows the scikit-learn name — and a `pickle` or
`joblib` file of it names mlxlearn's classes. Loading one therefore needs mlxlearn (and so
MLX) installed, even when `execution_backend_` is `"sklearn"`. For an artifact that stock
scikit-learn can load anywhere, call `unpatch_sklearn()` and construct the final model
after it.

## Performance

"Never slower than scikit-learn" is the design goal. The shipped *gate* is measurable: on
each benchmarked workload class, patched dispatch shows no statistically significant
regression against stock scikit-learn. Crossover points are measured per algorithm and per
operation, published in [`docs/benchmarks.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/benchmarks.md), and wired into the
dispatch thresholds — small problems are handed to scikit-learn on purpose.

What that means in practice for 0.1.0a2, on an M4 Max:

| | |
|---|---|
| **Neighbor queries** | **2.9×–17×** from ~250 samples up. This is the reason to use mlxlearn. |
| Neighbor `fit` | 0.17×–0.36×. mlxlearn uploads to the device; scikit-learn stores a reference. The first query repays it several times over. |
| `SVC` | **2.93× on wide data** (4 000 × 256, rbf), but slower than LIBSVM on narrow data — 0.47× at 4 000 × 32, worse below. `kernel="linear"` never uses MLX. Width decides, and patched dispatch routes accordingly. |
| `LogisticRegression` | **Slower than scikit-learn below 1 024 features**, by a lot. The crossover is set high so patched dispatch hands those to scikit-learn; see [`docs/benchmarks.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/benchmarks.md) for why and what would change it. |

Reproduce them from a clone of the repository (the benchmarks are not in the wheel):

```bash
python -m benchmarks.run --profile published --report bench/published.md
```

## Precision

MLX computes in float32 on the GPU. mlxlearn's parity tests are written to float32
tolerances, and the cases where float32 makes strict scikit-learn equivalence impossible
are enumerated — not waved away — in [`docs/fp32_policy.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/fp32_policy.md).

## Development

```bash
pip install -e ".[dev]"
pytest              # the fast suite; tests marked slow are deselected by default
pytest -m slow      # just those; `pytest -m ""` runs everything
```

Contributor rules, the estimator gate checklist, and the compliance checks are in
[`docs/development.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/development.md).

## Provenance

mlxlearn succeeds a private research fork of an Intel-maintained scikit-learn accelerator.
That history is not carried here: this repository was bootstrapped from an audited
allowlist, and no source file was copied. The audit, the recorded behavioral baseline of
the ancestor, and the authorship attestation are in [`phase0/`](https://github.com/tabulai/mlx-learn/tree/main/phase0).
Full lineage, disclaimers, and third-party notices are in
[`ACKNOWLEDGMENTS.md`](https://github.com/tabulai/mlx-learn/blob/main/ACKNOWLEDGMENTS.md) and
[`THIRD_PARTY_NOTICES.md`](https://github.com/tabulai/mlx-learn/blob/main/THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](https://github.com/tabulai/mlx-learn/blob/main/LICENSE).
