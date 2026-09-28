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
attribute mlxlearn changed. It does not rewrite containers that a scikit-learn module
first imported while patched filled with classes, such as the tables of scikit-learn's
test instance generator: import `sklearn.utils.estimator_checks` before patching if you
will run scikit-learn's estimator checks after unpatching.

Only the public names are patched. scikit-learn's own users of them — `DBSCAN`, `OPTICS`,
`TSNE`, `Isomap`, `mutual_info_classif`, `StackingClassifier` and the like — keep the stock
classes whatever the import order: those algorithms have not been through mlxlearn's gates,
and some depend on float64-exact distances. Names bound *outside* scikit-learn while patched
(your own `from sklearn.svm import SVC`, or another library doing the same) are ordinary
bindings and keep the mlxlearn class after `unpatch_sklearn()`.

## When mlxlearn falls back

Layer 1 raises a precise error when it cannot honor a request. Layer 2 falls back to stock
scikit-learn instead — but only for *capability* mismatches (sparse input, an unsupported
parameter, a value float32 cannot hold), never to hide a bug:

| Situation | Direct import | Patched |
|---|---|---|
| Capability mismatch | `CapabilityError`: `UnsupportedParameterError`, `UnsupportedInputError` or `SparseInputError` | falls back, warns once per class and reason, records diagnostics |
| Problem below the measured crossover | runs on MLX's CPU device | dispatches to scikit-learn |
| Invalid user input | scikit-learn's validation error, or `InvalidInputError` where mlxlearn detects it itself | the same, unmasked |
| `device="gpu"` with no usable GPU, on a problem above the crossover | `DeviceError` | `DeviceError` |
| Unexpected MLX runtime failure | `BackendExecutionError` | **`BackendExecutionError`** — never silently rerun on scikit-learn |

Every capability, device and backend error mlxlearn raises is an `MLXLearnError`, and the
classes above are all importable from `mlxlearn`. Argument errors from `set_config`,
`config_context`, `patch_sklearn` and the `MLXLEARN_*` environment variables, and input
errors that reproduce scikit-learn's own, are plain `ValueError`s and `TypeError`s, as in
scikit-learn. A `CapabilityError` keeps the mismatch (`.summary`) apart from the remedy
for a directly imported estimator (`.advice`); a patched estimator's warning reports the
mismatch and what it did about it. The input errors are also
`ValueError`s (`SparseInputError` a `TypeError`, as scikit-learn raises for sparse
input), so code written against scikit-learn's error contract keeps working, and all of
them survive pickling, so a failure in a `joblib` worker arrives intact rather than as a
`BrokenProcessPool`. Inference on an estimator whose last `fit` did not complete raises
scikit-learn's `NotFittedError`.

Set the policy with `fallback_policy="warn" | "raise" | "silent"`. Diagnostics are recorded
in every mode:

```python
import mlxlearn

with mlxlearn.config_context(fallback_policy="raise"):
    ...                                     # any capability fallback becomes an error

mlxlearn.get_last_backend_event()           # what the last fit/predict actually did
mlxlearn.get_backend_diagnostics()          # the recent record, this process (a ring buffer)
mlxlearn.reset_fallback_warnings()          # warn again for reasons already warned about
```

**Sticky backend.** The backend is chosen during `fit` and exposed as
`estimator.execution_backend_` (`"mlx"`, `"cpu"`, or `"sklearn"`; like any fitted attribute,
it raises `AttributeError` before `fit`). The decision itself — reason, detail, shape — is
`estimator.fit_event_`, kept with the model: it outlives the diagnostics ring buffer, is
recorded even with `diagnostics=False`, and pickles with the estimator, so a model fitted
in a worker process still explains itself. Every subsequent
`predict` / `predict_proba` / `decision_function` / `kneighbors` uses the backend the model
was fitted with, and each public call records one diagnostics event. A model fitted on MLX
can never wander into scikit-learn inference with incompatible state. A new `fit` clears
state and may pick a different backend.

Two cases run somewhere other than the fitted device, and neither re-decides anything.
Under `patch_sklearn()`, a sparse query to `kneighbors`, `kneighbors_graph`, `predict` or
`predict_proba` of a neighbor estimator fitted on MLX is answered by scikit-learn's
brute-force search, which reads the MLX-fitted state directly; the query is cast to
float32 to match the training copy, so its distances carry that copy's rounding, and the
call is recorded, and warned about, as a fallback. And a model fitted on the MLX GPU, then
loaded in a process with no usable GPU, runs the same float32 code on MLX's CPU, recorded
as `"gpu-unavailable"` — unless `device="gpu"` is set, which raises `DeviceError`.
Inference that never touches a device is unaffected by either: `LogisticRegression`
predicts with scikit-learn's float64 NumPy code on every backend, and `radius_neighbors`
and `radius_neighbors_graph` on an MLX-fitted `NearestNeighbors` are scikit-learn's own
search on the fitted state — sparse queries included, in both layers — so they record
`"sticky"`, not a fallback, and run under `device="gpu"` on any host.

## Configuration

The public surface is deliberately small:

```python
mlxlearn.set_config(
    device="auto",              # "auto" | "gpu" | "cpu"
    fallback_policy="warn",     # "warn" | "raise" | "silent"
    output_type="numpy",        # 0.1.0: NumPy in, NumPy out
    deterministic=True,         # no observable effect in 0.1.x: every solver is deterministic
    random_state=None,          # no observable effect in 0.1.x: nothing draws random numbers
    diagnostics=True,
)
```

**Where a setting reaches.** `set_config` changes the process-wide default, which every
thread sees — including the worker threads scikit-learn's `n_jobs` runs on under joblib's
threading backend. `config_context` overrides the options it names for the current thread,
and for `asyncio` tasks started from it, which inherit it as they inherit any context
variable; other threads, joblib's included, do not see it. A separate process — joblib's
default `loky` workers, `multiprocessing` — imports mlxlearn afresh and starts from the
environment: `MLXLEARN_DEVICE`, `MLXLEARN_FALLBACK_POLICY`, `MLXLEARN_OUTPUT_TYPE`,
`MLXLEARN_DETERMINISTIC`, `MLXLEARN_RANDOM_STATE` and `MLXLEARN_DIAGNOSTICS`, one per
option. Nothing a parent sets at runtime reaches a worker that is already running, so set
those before the workers start. To assert that nothing falls back under `n_jobs`,
`warnings.filterwarnings("error", category=MLXLearnFallbackWarning)` with
`error_score="raise"` also works, because scikit-learn forwards warning filters to its
workers; the warning comes with the decision, so it raises before scikit-learn runs the
fit. Diagnostics are per process too: events recorded in a worker stay there, while a
model fitted in one carries its `fit_event_` back.

**What `device` means.** It picks the MLX device for work that dispatch sends to MLX, and
nothing else: it never overrides the size crossover and never selects scikit-learn.
`"auto"` uses the GPU when a working Metal device exists and MLX's CPU otherwise. `"gpu"`
raises `DeviceError` rather than substitute the CPU. `"cpu"` runs the same float32
algorithm on MLX's CPU backend — not scikit-learn's float64 one — and the fit records its
backend as `"cpu"`. Inference follows the backend recorded at `fit`, not the current
setting.

Crossover thresholds, block sizes, and solver tuning are private and typed, overridable
through their own `MLXLEARN_*` environment variables (`MLXLEARN_KNN_MIN_TRAIN_SAMPLES`,
say) for CI and debugging. They are not part of the public API.

## Processes, memory, and pickles

**Do not `fork` after using mlxlearn.** MLX, like Metal underneath it, is not fork-safe.
Once a process has loaded MLX or scikit-learn's estimators — the first `fit` or `predict`
does, and so does `patch_sklearn()` — MLX in a child it creates with a bare `fork()`
crashes on its first operation, `device="cpu"` included, or hangs inside a
`multiprocessing` pool. mlxlearn detects such a child and refuses rather than crash: a
directly imported estimator raises `DeviceError`, and a patched one hands the fit to
scikit-learn. It cannot detect every unsafe fork, so the rule stands. The safe patterns
are the defaults: Python's `spawn` start method (the macOS default) and `forkserver`, and
scikit-learn's `n_jobs`, whose `loky` workers are spawned. What breaks is choosing `fork`
explicitly — `multiprocessing.get_context("fork")`, `os.fork()`, Celery's prefork pool
(use `solo`, `threads`, or spawned workers), or gunicorn `--preload` of an app that has
already fitted or loaded a model (load it in each worker instead).

**MLX's buffer cache belongs to the process.** MLX keeps freed device buffers in a
process-wide cache for reuse, and by default lets that cache grow to nearly the whole
memory limit; deleting a model does not shrink it. mlxlearn bounds its own share: each
neighbor query's working set is held to a memory budget, and after any mlxlearn `fit` or
query that grew the cache by more than 1/16 of physical memory, mlxlearn calls
`mlx.core.clear_cache()`, which frees idle buffers and changes no setting. Growth below
that is kept for reuse, so a service or a notebook that queries differently sized data
can still hold idle memory. Call `mlx.core.clear_cache()` after large jobs, or bound the
cache once at startup with `mlx.core.set_cache_limit(n_bytes)`; both are the
application's choice, and both act on the whole process, including any other MLX work in
it, which pays in re-allocation, not correctness.

**Pickles need mlxlearn to load.** Every model mlxlearn fits is an mlxlearn object — under
`patch_sklearn()` too, where its repr shows the scikit-learn name — and a `pickle` or
`joblib` file of it names mlxlearn's classes. Loading one therefore needs mlxlearn (and so
MLX) installed, even when `execution_backend_` is `"sklearn"`. For an artifact that stock
scikit-learn can load anywhere, call `unpatch_sklearn()` and construct the final model
after it.

## Performance

"Never slower than scikit-learn" is the design goal. The shipped *gate* is measurable: on
each benchmarked workload, patched dispatch shows no statistically significant regression
against stock scikit-learn. Crossover points are measured per estimator, against both ends
of the supported scikit-learn range, published in [`docs/benchmarks.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/benchmarks.md), and wired into the
dispatch thresholds: patched dispatch uses MLX only where it measured faster or even, and
hands everything else to scikit-learn on purpose.

Measured on an M4 Max (MLX 0.32.2, scikit-learn 1.9.0; thresholds checked against 1.7.2
too), for the problems patched dispatch sends to MLX:

| | |
|---|---|
| **Neighbor queries** | **1.2×–3.8× for single rows, 1.4×–7.4× for 4096-query batches**, against scikit-learn's brute-force search, from 16 features, 4 096 training rows and about 4 million elements up — wide data such as embeddings. Below that stock scikit-learn is faster for some query batch a fit cannot foresee, and with at most 15 features its KD-tree answers a single row in 0.3–1.2 ms against 1–2 ms for an MLX call; patched dispatch hands all of those fits to it. |
| Neighbor `fit` | 0.06×–0.09× of stock brute force, which stores a reference where mlxlearn uploads and prepares the index. Repaid after roughly 300–900 queries. |
| `SVC` | **1.2×–4.3× from 128 features** up to 8 000 rows (rbf, poly and sigmoid, C of 1 and 10), **8×–17×** at 16 000–32 000 × 256. Narrower data is slower on MLX at C = 10 or past the kernel cache, so patched dispatch keeps it on LIBSVM; `kernel="linear"` never leaves it. |
| `LogisticRegression` | **Multinomial: 1.3×–13× from 60 000 rows. Binary: 1.1×–4.5× on tall, narrow data** (100 000 rows or more, at most 32 features). Binary fits on wider data are slower on MLX — 0.46×–0.85× at 256–1 024 features — and go to scikit-learn. |

Reproduce them from a clone of the repository (the benchmarks are not in the wheel):

```bash
python -m benchmarks.run --profile published --report bench/published.md
```

## Precision

MLX computes in float32 on the GPU. mlxlearn's parity tests are written to float32
tolerances, and the cases where float32 makes strict scikit-learn equivalence impossible
are enumerated — not waved away — in [`docs/fp32_policy.md`](https://github.com/tabulai/mlx-learn/blob/main/docs/fp32_policy.md).
A large constant offset in float64 input (timestamps, projected coordinates) costs no
precision where the model is invariant to it — the neighbor estimators' `kneighbors` and
the predictions built on it, `LogisticRegression` with an intercept, `SVC`'s rbf kernel —
because it is removed in float64 before anything is rounded to float32. `radius_neighbors`
and `radius_neighbors_graph` are the exception: they run scikit-learn's search on the
float32 training copy as given, so data with an offset rounds at the offset's magnitude
(Unix timestamps to 128-second steps); subtract it before fitting, or use stock
scikit-learn. Values
float32 cannot hold — and neighbor training data whose span it cannot resolve beside the
gaps between rows, such as an unmasked `1e30` fill value — are rejected with
`UnsupportedInputError`, never turned into infinities or zero distances. Exact neighbor
searches are certified against float32 rounding, not trusted to a fixed margin.

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
