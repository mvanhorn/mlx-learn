"""``NeighborIndex`` mechanics: publication, memory, devices, pickling, selection.

The numerical claims live in ``tests/parity/test_neighbors_fp32.py``. These pin the
machinery around them — the parts that fail as crashes, blow-ups and silent
aliasing rather than as a wrong distance.
"""

from __future__ import annotations

import pickle
import sys
import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.config import config_context, tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._kernels import distance
from mlxlearn._kernels.distance import NeighborIndex, build_index, resolve_blocks
from mlxlearn.exceptions import DeviceError

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
    barrier = threading.Barrier(n_threads)
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
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
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


def test_query_peak_memory_stays_near_the_budget(rng):
    """Large ``k`` on wide data: the refinement gather used to be unbudgeted.

    Before, the peak was about 7× the budget from the tile alone, and the
    ``(rows, k, d)`` gather came on top — 7.4 GB at k=1000, d=768.
    """
    budget = 16 * MiB
    X = rng.normal(size=(8000, 256)).astype(np.float32)
    Q = rng.normal(size=(512, 256)).astype(np.float32)
    with tuning_context(distance_block_bytes=budget, distance_block_bytes_max=budget):
        index = build_index(X, backend="mlx")
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


def test_a_gpu_built_index_runs_on_the_cpu_where_there_is_no_gpu(rng, monkeypatch):
    """A model fitted on a Mac, unpickled in a process without Metal.

    MLX's CPU runs the same float32 code, so ``device="auto"`` and ``"cpu"`` must
    serve it; an explicit ``device="gpu"`` still cannot be honored and raises.
    It used to raise under every setting, advising ``device='auto'``, which was
    already set.
    """
    X = rng.normal(size=(2000, 8)).astype(np.float32)
    Q = rng.normal(size=(40, 8)).astype(np.float32)
    fitted = build_index(X, backend="mlx")
    expected_dist, expected_idx = fitted.query(Q, 5)
    payload = pickle.dumps(fitted)

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    for device in ("auto", "cpu"):
        with config_context(device=device):
            dist, idx = pickle.loads(payload).query(Q, 5)
        np.testing.assert_array_equal(idx, expected_idx)
        np.testing.assert_allclose(dist, expected_dist, rtol=1e-6)

    with config_context(device="gpu"), pytest.raises(DeviceError):
        pickle.loads(payload).query(Q, 5)


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


def test_a_pickle_from_before_the_residual_still_loads(rng):
    X = rng.normal(size=(500, 4)).astype(np.float32)
    Q = rng.normal(size=(10, 4)).astype(np.float32)
    index = build_index(X, backend="cpu")

    old = NeighborIndex.__new__(NeighborIndex)
    old.__setstate__({"X": np.array(index.X), "center": index.center, "backend": "cpu"})
    for got, want in zip(old.query(Q, 3), index.query(Q, 3), strict=True):
        np.testing.assert_array_equal(got, want)


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
