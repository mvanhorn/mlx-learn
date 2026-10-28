"""Checked CPU input conversion before the GPU logistic solver runs."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_CAST_BLOCK_SIZE = 1 << 20
_PARALLEL_CAST_MIN_SIZE = 4 << 20


def cast_float32_checked(X):
    """Convert to contiguous float32 and report whether every value is finite.

    Large native-endian, contiguous float64 arrays use up to eight workers for
    the existing CPU input conversion. Each worker uses NumPy's ordinary cast
    on disjoint blocks, then checks those blocks while they are still in cache.
    Other inputs retain the serial NumPy conversion. This does not move any
    solver computation off the GPU.

    Converting float64 or wider input owns new storage. The executor lives for
    one call and all workers join before returning, including on exceptions;
    no input buffers or worker pools are cached. A nonfinite result is reported
    to the caller, which can distinguish input NaN/Inf from conversion overflow.
    """
    errors = np.geterr()
    parallel = (
        type(X) is np.ndarray
        and X.dtype == np.dtype(np.float64)
        and X.flags.c_contiguous
        and X.size >= _PARALLEL_CAST_MIN_SIZE
        # Custom underflow handling retains its single-threaded callback,
        # logging, warning and exception behavior, including callback order.
        and errors["under"] == "ignore"
    )
    workers = min(8, os.cpu_count() or 1, X.size // _CAST_BLOCK_SIZE) if parallel else 1
    if workers <= 1:
        with np.errstate(over="ignore", invalid="ignore"):
            prepared = np.asarray(X, dtype=np.float32, order="C")
        return prepared, bool(np.isfinite(prepared).all())

    prepared = np.empty(X.shape, dtype=np.float32, order="C")
    source, destination = X.reshape(-1), prepared.reshape(-1)
    errors.update(over="ignore", invalid="ignore")
    error_callback = np.geterrcall()

    def convert_chunk(start, stop):
        finite = True
        # NumPy's error settings are thread-local. Preserve the caller's
        # underflow policy and callback, suppressing only the existing cast
        # overflow/invalid warnings. errstate restores each worker's settings.
        with np.errstate(call=error_callback, **errors):
            for block_start in range(start, stop, _CAST_BLOCK_SIZE):
                block_stop = min(block_start + _CAST_BLOCK_SIZE, stop)
                target = destination[block_start:block_stop]
                np.copyto(target, source[block_start:block_stop], casting="unsafe")
                finite = bool(np.isfinite(target).all()) and finite
        return finite

    chunk_size = (X.size + workers - 1) // workers
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mlxlearn-input") as pool:
        pending = [
            pool.submit(convert_chunk, start, min(start + chunk_size, X.size))
            for start in range(0, X.size, chunk_size)
        ]
        # Gather every result before reducing; an exception also joins workers
        # through the executor's context manager before the buffer can escape.
        flags = [future.result() for future in pending]
    return prepared, all(flags)
