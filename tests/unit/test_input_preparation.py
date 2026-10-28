"""Threaded input conversion retains NumPy rounding, ownership and error policy."""

from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest

from mlxlearn._common import preparation


@pytest.fixture
def small_parallel_blocks(monkeypatch):
    # Exercise real workers and partial blocks without allocating benchmark data.
    monkeypatch.setattr(preparation, "_CAST_BLOCK_SIZE", 257)
    monkeypatch.setattr(preparation, "_PARALLEL_CAST_MIN_SIZE", 1024)
    monkeypatch.setattr(preparation.os, "cpu_count", lambda: 8)


def _numpy_bits(X):
    with np.errstate(over="ignore", invalid="ignore"):
        return X.astype(np.float32).view(np.uint32)


def test_random_float64_bits_match_numpy_exactly(small_parallel_blocks):
    rng = np.random.default_rng(723)
    X = rng.integers(0, 2**64 - 1, 131_073, dtype=np.uint64).view(np.float64)
    before = X.view(np.uint64).copy()
    prepared, finite = preparation.cast_float32_checked(X)
    np.testing.assert_array_equal(prepared.view(np.uint32), _numpy_bits(X))
    np.testing.assert_array_equal(X.view(np.uint64), before)
    assert finite is False


def test_rounding_ties_subnormals_signed_zero_and_nan_payloads(small_parallel_blocks):
    values = [0.0, -0.0, np.inf, -np.inf, np.nan]
    for exponent in range(-150, 128):
        for significand in (1.0, 1.0 + 2**-23, 2.0 - 2**-23):
            value = np.ldexp(significand, exponent)
            halfway = value + np.ldexp(1.0, exponent - 24)
            for item in (value, halfway, np.nextafter(halfway, -np.inf),
                         np.nextafter(halfway, np.inf)):
                values.extend((item, -item))
    X = np.array(values, dtype=np.float64)
    # Include signaling NaNs and payloads too small to survive truncation.
    payloads = np.array([0x7FF0000000000001, 0xFFF123456789ABCD], dtype=np.uint64)
    X = np.concatenate((X, payloads.view(np.float64)))
    prepared, finite = preparation.cast_float32_checked(X)
    np.testing.assert_array_equal(prepared.view(np.uint32), _numpy_bits(X))
    assert finite is False


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 1e300, -1e300])
def test_nonfinite_or_overflow_in_final_partial_block_is_reported(
    small_parallel_blocks, bad_value
):
    X = np.ones(4099, dtype=np.float64)
    X[-1] = bad_value
    with np.errstate(over="raise", invalid="raise"):
        prepared, finite = preparation.cast_float32_checked(X)
    assert finite is False
    np.testing.assert_array_equal(prepared.view(np.uint32), _numpy_bits(X))


def test_parallel_readonly_input_has_disjoint_owned_output(small_parallel_blocks):
    X = np.linspace(-3, 3, 4099)
    X.setflags(write=False)
    expected = X.astype(np.float32)
    prepared, finite = preparation.cast_float32_checked(X)
    assert finite is True
    assert prepared.flags.c_contiguous
    assert prepared.flags.writeable
    assert not X.flags.writeable
    assert not np.shares_memory(prepared, X)
    np.testing.assert_array_equal(prepared, expected)
    prepared.fill(0)
    assert np.count_nonzero(X) == len(X) - 1


@pytest.mark.parametrize("layout", ["small", "fortran", "strided", "big-endian", "scalar", "empty"])
def test_ineligible_inputs_keep_serial_conversion(monkeypatch, small_parallel_blocks, layout):
    X = np.linspace(-2, 2, 4096).reshape(64, 64)
    if layout == "small":
        X = X[:2].copy()
    elif layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = X[:, ::2]
    elif layout == "big-endian":
        X = X.astype(">f8")
    elif layout == "scalar":
        X = np.array(0.1, dtype=np.float64)
    else:
        X = X[:0]

    def unexpected_pool(*args, **kwargs):
        pytest.fail("this input must retain the serial NumPy conversion")

    monkeypatch.setattr(preparation, "ThreadPoolExecutor", unexpected_pool)
    prepared, finite = preparation.cast_float32_checked(X)
    assert prepared.shape == X.shape
    assert prepared.flags.c_contiguous
    assert not np.shares_memory(prepared, X)
    assert finite is True
    np.testing.assert_array_equal(prepared.view(np.uint32), _numpy_bits(X))


@pytest.mark.parametrize("cpu_count", [None, 1, 2, 32])
def test_worker_count_is_bounded_and_single_core_uses_serial(
    monkeypatch, small_parallel_blocks, cpu_count
):
    original = preparation.ThreadPoolExecutor
    observed = []

    def record_pool(*args, **kwargs):
        observed.append(kwargs["max_workers"])
        return original(*args, **kwargs)

    monkeypatch.setattr(preparation.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(preparation, "ThreadPoolExecutor", record_pool)
    prepared, finite = preparation.cast_float32_checked(np.ones(4099))
    assert finite is True
    np.testing.assert_array_equal(prepared, np.ones(4099, dtype=np.float32))
    assert observed == ([] if cpu_count in (None, 1) else [min(cpu_count, 8)])


@pytest.mark.parametrize("parallel", [False, True])
def test_underflow_raise_policy_is_preserved_and_workers_join(small_parallel_blocks, parallel):
    X = np.ones(4099 if parallel else 17)
    X[-1] = 1e-320
    with np.errstate(under="raise"):
        before = np.geterr()
        with pytest.raises(FloatingPointError, match="underflow"):
            preparation.cast_float32_checked(X)
        assert np.geterr() == before
    assert not any(t.name.startswith("mlxlearn-input") for t in threading.enumerate())


@pytest.mark.parametrize("policy", ["ignore", "warn", "call", "log", "print"])
def test_underflow_policies_and_callbacks_are_preserved(
    monkeypatch, small_parallel_blocks, policy, capfd
):
    X = np.ones(4099)
    X[-1] = 1e-320
    events = []

    class Log:
        def write(self, text):
            events.append(text)

    callback = Log() if policy == "log" else lambda message, flag: events.append((message, flag))
    if policy != "ignore":
        def unexpected_pool(*args, **kwargs):
            pytest.fail("custom underflow handling must retain serial callback semantics")

        monkeypatch.setattr(preparation, "ThreadPoolExecutor", unexpected_pool)
    with np.errstate(under=policy, call=callback):
        before = np.geterr()
        if policy == "warn":
            with pytest.warns(RuntimeWarning, match="underflow"):
                prepared, finite = preparation.cast_float32_checked(X)
        else:
            prepared, finite = preparation.cast_float32_checked(X)
        assert np.geterr() == before
        assert np.geterrcall() is callback
    assert finite is True
    assert prepared[-1] == 0
    if policy in ("call", "log"):
        assert events and "underflow" in str(events)
    if policy == "print":
        assert "underflow" in capfd.readouterr().err


def test_worker_exception_joins_other_workers(monkeypatch, small_parallel_blocks):
    original_copy = np.copyto

    def fail_first_chunk(destination, source, **kwargs):
        if source[0] == 0:
            raise RuntimeError("conversion worker failed")
        original_copy(destination, source, **kwargs)

    monkeypatch.setattr(np, "copyto", fail_first_chunk)
    before = np.geterr()
    with pytest.raises(RuntimeError, match="conversion worker failed"):
        preparation.cast_float32_checked(np.arange(4099, dtype=np.float64))
    assert np.geterr() == before
    assert not any(t.name.startswith("mlxlearn-input") for t in threading.enumerate())


def test_multiple_calls_retain_no_sources_or_worker_threads(small_parallel_blocks):
    references = []
    for offset in range(4):
        X = np.arange(4099, dtype=np.float64) + offset
        references.append(weakref.ref(X))
        prepared, finite = preparation.cast_float32_checked(X)
        assert finite is True
        np.testing.assert_array_equal(prepared, X.astype(np.float32))
        del X, prepared
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not any(t.name.startswith("mlxlearn-input") for t in threading.enumerate())
