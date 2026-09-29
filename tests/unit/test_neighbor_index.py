"""``NeighborIndex`` mechanics: publication, memory, devices, pickling, selection.

The numerical claims live in ``tests/parity/test_neighbors_fp32.py``. These pin the
machinery around them — the parts that fail as crashes, blow-ups and silent
aliasing rather than as a wrong distance.
"""

from __future__ import annotations

import copy
import pickle
import sys
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._kernels import distance
from mlxlearn._kernels.distance import NeighborIndex, build_index, resolve_blocks
from mlxlearn.exceptions import UnsupportedInputError
from mlxlearn.neighbors import KNeighborsRegressor, NearestNeighbors
from tests._helpers import join_or_fail

MiB = 1024 * 1024


# ----------------------------------------------------------------------------------
# publication
# ----------------------------------------------------------------------------------


def test_concurrent_first_queries_on_an_unpickled_index(rng, monkeypatch):
    """A burst of threads right after unpickling — joblib.load, then a thread pool.

    The device arrays are rebuilt lazily on the first query. They used to be
    published one at a time and still unevaluated, so a second thread could pass
    the "already built" check and read ``None``, or evaluate a lazy graph that
    belonged to another thread's stream: a segfault or "There is no Stream(gpu, N)
    in current thread". The build is slowed down here to hold that window open.
    """
    X = rng.normal(size=(3000, 16)).astype(np.float32)
    Q = rng.normal(size=(64, 16)).astype(np.float32)
    expected_dist, expected_idx = build_index(X, backend="mlx").query(Q, 5)
    restored = pickle.loads(pickle.dumps(build_index(X, backend="mlx")))

    builds = []
    real_build = distance._build_state

    def slow_build(*args):
        builds.append(threading.get_ident())
        time.sleep(0.05)
        return real_build(*args)

    monkeypatch.setattr(distance, "_build_state", slow_build)

    n_threads = 12
    barrier = threading.Barrier(n_threads, timeout=60)
    results: list = [None] * n_threads
    errors: list[BaseException] = []

    def worker(slot):
        try:
            barrier.wait()
            results[slot] = restored.query(Q, 5)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        join_or_fail(threads)
    finally:
        sys.setswitchinterval(interval)

    assert not errors, errors
    assert len(builds) == 1, "every thread uploaded its own copy of the training set"
    for dist, idx in results:
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_array_equal(dist, expected_dist)


def test_the_lock_is_not_pickled_and_is_fresh_after_loading(rng):
    index = build_index(rng.normal(size=(50, 3)).astype(np.float32), backend="cpu")
    restored = pickle.loads(pickle.dumps(index))
    assert restored._lock is not index._lock
    assert restored._lock.acquire(blocking=False)
    restored._lock.release()


# ----------------------------------------------------------------------------------
# no aliasing, one copy
# ----------------------------------------------------------------------------------


def test_the_index_owns_a_read_only_copy(rng):
    """Never the caller's buffer; and the host view is the device buffer itself."""
    X = rng.normal(size=(300, 4)).astype(np.float32)
    Q = rng.normal(size=(5, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")
    before = index.query(Q, 3)

    assert not np.shares_memory(index.X, X)
    assert not index.X.flags.writeable
    X[:] = 0.0
    after = index.query(Q, 3)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[0], after[0])


def test_querying_with_the_index_own_rows_matches_a_copy_of_them(rng):
    """The self-query path reads the device rows directly; it must agree with the
    ordinary path given the same values."""
    X = rng.normal(size=(700, 6))
    index = build_index(X, backend="mlx")
    direct = index.query(index.X, 4, exclude_self=True)
    copied = index.query(np.array(X), 4, exclude_self=True)
    np.testing.assert_array_equal(direct[1], copied[1])
    np.testing.assert_allclose(direct[0], copied[0], rtol=1e-6)


# ----------------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------------


def test_the_distance_tile_is_budgeted_as_a_working_set():
    """The budget bounds the whole working set, not the tile alone."""
    budget = 64 * MiB
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        plan = resolve_blocks(4096, 262_144, backend="mlx", k_search=14)
    assert plan.query_rows * plan.train_cols * 4 * distance._TILE_WORKSET_FACTOR <= budget


@pytest.mark.parametrize("backend", ["mlx", "cpu"])
def test_query_peak_memory_stays_near_the_budget(backend, rng):
    """Large ``k`` on wide data: the refinement gather used to be unbudgeted.

    Before, the peak was about 7× the budget from the tile alone, and the
    ``(rows, k, d)`` gather came on top — 7.4 GB at k=1000, d=768. Here it was
    20× the budget.
    """
    budget = 16 * MiB
    X = rng.normal(size=(8000, 256)).astype(np.float32)
    Q = rng.normal(size=(512, 256)).astype(np.float32)
    with tuning_context(
        distance_block_bytes=budget,
        distance_block_bytes_max=budget,
        cpu_distance_block_bytes=budget,
    ):
        index = build_index(X, backend=backend)
        index.query(Q[:8], 200)
        mx.eval(mx.zeros(1))
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        index.query(Q, 200)
        peak = mx.get_peak_memory() - base
    assert peak <= 2 * budget, f"peak {peak / MiB:.0f} MiB against a {budget / MiB:.0f} MiB budget"


# ----------------------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------------------


def test_a_gpu_built_index_runs_where_it_is_told(rng, monkeypatch):
    """A model fitted on a Mac, unpickled in a process without Metal.

    MLX's CPU runs the same float32 code. Whether to use it is the estimator's
    decision, from the configured device (``inference_device``); the index runs on
    the device it is handed and reads no configuration. It used to read it itself,
    below the layer allowed to.
    """
    X = rng.normal(size=(2000, 8)).astype(np.float32)
    Q = rng.normal(size=(40, 8)).astype(np.float32)
    expected_dist, expected_idx = build_index(X, backend="cpu").query(Q, 5)

    def no_configuration():
        raise AssertionError("the index read the user's configuration")

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    monkeypatch.setattr("mlxlearn._common.device.get_config", no_configuration)
    fitted = build_index(X, backend="mlx", device="cpu")
    for index in (fitted, pickle.loads(pickle.dumps(fitted))):
        dist, idx = index.query(Q, 5, device="cpu")
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_array_equal(dist, expected_dist)


def test_the_cpu_device_does_not_compile(rng, monkeypatch):
    """Only the GPU fuses the elementwise pieces.

    On MLX's CPU each compiled graph shells out to a C compiler, once per MLX
    version and ``$TMPDIR``, and runs no faster: a 200-row fit and predict — the
    Layer 1 path for every problem below the crossover — took 3-4 s on a fresh
    machine, against 0.1 s for 0.1.0a2.
    """
    compiled = []
    real_compile = mx.compile

    def recording_compile(fn, *args, **kwargs):
        compiled.append(fn.__name__)
        return real_compile(fn, *args, **kwargs)

    monkeypatch.setattr(mx, "compile", recording_compile)
    distance._kernels.cache_clear()
    try:
        X = rng.normal(size=(200, 5))
        for data in (X.astype(np.float32), X):  # the float64 residual path too
            index = build_index(data, backend="cpu")
            index.query(data[:7] + 0.1, 3)
            index.query(index.X, 3, exclude_self=True)
    finally:
        distance._kernels.cache_clear()
    assert compiled == []


# ----------------------------------------------------------------------------------
# pickling
# ----------------------------------------------------------------------------------


def test_a_float64_index_round_trips_exactly(rng):
    """The residual is part of the state; without it a reloaded index would answer
    differently from the one that was saved."""
    X = 1.7e9 + rng.uniform(0, 3600, size=(3000, 2))
    Q = 1.7e9 + rng.uniform(0, 3600, size=(40, 2))
    index = build_index(X, backend="mlx")
    assert index.X_lo is not None

    restored = pickle.loads(pickle.dumps(index))
    for got, want in zip(restored.query(Q, 5), index.query(Q, 5), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "reload",
    [lambda model: pickle.loads(pickle.dumps(model)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
@pytest.mark.parametrize("estimator", [NearestNeighbors, KNeighborsRegressor])
def test_a_reloaded_model_keeps_one_copy_of_its_training_rows(estimator, reload, rng):
    """A fitted model's ``_fit_X`` is the index's own host view, and stays that way.

    Reloaded, the index rebuilds its device copy on the first query and points its
    ``X`` at it; ``_fit_X`` stayed on the loaded array, so the model held one more
    full copy of the training set than a fresh fit, and saving it again doubled the
    file.
    """
    X = rng.normal(size=(3000, 16)).astype(np.float32)
    model = estimator(n_neighbors=3).fit(X, X[:, 0])
    assert model.execution_backend_ in ("mlx", "cpu")
    fresh = len(pickle.dumps(model))

    loaded = reload(model)
    loaded.kneighbors(X[:5])
    assert loaded._fit_X is loaded._index_.X
    assert len(pickle.dumps(loaded)) < 1.05 * fresh


def test_a_pickle_from_before_the_residual_still_loads(rng):
    X = rng.normal(size=(500, 4)).astype(np.float32)
    Q = rng.normal(size=(10, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")

    old = NeighborIndex.__new__(NeighborIndex)
    old.__setstate__({"X": np.array(index.X), "center": index.center, "backend": "cpu"})
    for got, want in zip(old.query(Q, 3), index.query(Q, 3), strict=True):
        np.testing.assert_array_equal(got, want)


# ----------------------------------------------------------------------------------
# row limits
# ----------------------------------------------------------------------------------


def test_exact_search_keys_address_rows_past_int32():
    """The exact search packs each training index into the low 32 bits of its key.

    The indices were an ``arange`` starting at the tile's first row, which MLX 0.31
    — the supported floor — rejects with a TypeError from row ``2**31`` on.
    """
    vals = mx.array([[1.0, 0.5, 2.0]], dtype=mx.float32)
    for t_start in (0, 2**31 + 5, 2**32 - 3):
        keys = np.asarray(distance._composite_keys(mx, vals, t_start))
        np.testing.assert_array_equal(keys & 0xFFFFFFFF, [t_start + np.arange(3)])
        values = (keys >> np.uint64(32)).astype(np.uint32).view(np.float32)
        np.testing.assert_array_equal(values, [[1.0, 0.5, 2.0]])


def test_more_rows_than_a_key_can_address_are_refused():
    """Past ``2**32`` rows the index would come back truncated, and the distance
    corrupted by its high bits: refused at fit instead, before anything is copied."""
    X = np.lib.stride_tricks.as_strided(
        np.zeros(1, np.float32), shape=(2**32 + 1, 1), strides=(0, 4)
    )
    with pytest.raises(UnsupportedInputError, match="rows"):
        build_index(X, backend="cpu")


# ----------------------------------------------------------------------------------
# two-stage selection
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2048, 2049, 2048 + 511, 5003, 9001])
@pytest.mark.parametrize("take", [1, 14, 128, 129, 270])
def test_segmented_selection_is_exact(width, take, rng):
    """Ragged tails, takes either side of a segment-width step, and heavy ties."""
    vals = rng.random((7, width)).astype(np.float32)
    vals[:, ::3] = vals[:, :1]
    device = mx.gpu if gpu_available() else mx.cpu
    with mx.stream(device):
        idx, got = distance._select_smallest(mx, mx.array(vals), take, segmented=True)
        mx.eval(idx, got)
    idx = np.asarray(idx).astype(np.int64)
    got = np.asarray(got)

    np.testing.assert_array_equal(np.sort(got, axis=1), np.sort(vals, axis=1)[:, :take])
    np.testing.assert_array_equal(np.take_along_axis(vals, idx, axis=1), got)
    assert np.all(idx < width)
    assert all(len(set(row)) == take for row in idx.tolist())


# ----------------------------------------------------------------------------------
# the fit-time probe
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["mlx", "cpu"] if gpu_available() else ["cpu"])
def test_the_probe_counts_what_the_first_pass_cannot_certify(backend, rng):
    """Gaussian rows certify on the first pass; rows repeated 20 times cannot, since
    their k-th neighbor ties with the copies left out; integer rows take the exact
    path and are reported as such, unsearched."""
    X = rng.normal(size=(6000, 32))
    found = build_index(X, backend=backend).probe(5)
    assert found == {"rows": 64, "uncertified": 0, "integer": False}

    duplicated = rng.normal(size=(300, 32))[rng.integers(0, 300, size=6000)]
    found = build_index(duplicated, backend=backend).probe(5)
    assert found["rows"] == 64 and found["uncertified"] > 48 and not found["integer"]

    counts = rng.poisson(2.0, size=(6000, 32)).astype(np.float64)
    assert build_index(counts, backend=backend).probe(5) == {
        "rows": 0, "uncertified": 0, "integer": True,
    }


def test_the_probe_agrees_with_the_query_it_predicts(rng):
    """The probe is the query's own first pass: the rows it counts are the rows a query
    of the same training rows searches again."""
    X = rng.normal(size=(5000, 24))
    X[:, 0] = rng.uniform(0.0, 1e4, size=5000)
    X = X.astype(np.float32)
    index = build_index(X, backend="cpu")
    found = index.probe(10)
    rows = np.unique(np.linspace(0, 4999, 64).astype(np.intp))
    stats: dict = {}
    index.query(X[rows], 10, stats=stats)
    assert found["uncertified"] == stats["widened_rows"] > 0


def test_the_probe_searches_as_many_rows_as_it_is_asked_to(rng):
    """Patched dispatch probes ``knn_probe_rows`` rows (256), enough to see a rate of
    uncertified rows near 1%, which 64 rows miss about half the time; fewer when the
    index has fewer rows."""
    X = rng.normal(size=(3000, 16))
    X[:, 0] *= 100.0
    index = build_index(X, backend="cpu")
    wide = index.probe(10, rows=256)
    assert wide["rows"] == 256 and not wide["integer"]
    rows = np.unique(np.linspace(0, 2999, 256).astype(np.intp))
    stats: dict = {}
    index.query(X[rows], 10, stats=stats)
    assert wide["uncertified"] == stats["widened_rows"]
    assert build_index(X[:100], backend="cpu").probe(10, rows=256)["rows"] == 100


def test_the_probe_of_a_tiny_index_is_the_exact_answer(rng):
    """With every row a candidate there is nothing to certify."""
    X = rng.normal(size=(12, 3))
    assert build_index(X, backend="cpu").probe(10) == {
        "rows": 12, "uncertified": 0, "integer": False,
    }
