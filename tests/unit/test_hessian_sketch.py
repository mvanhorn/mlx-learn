"""Correctness of row-space Hessian-sketch initialization and native bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available


def _sketch_module():
    path = Path(__file__).parents[2] / "src/mlxlearn_device/hessian_sketch.py"
    spec = importlib.util.spec_from_file_location("_sketch_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_hessian_pairs_match_binary_zero_logit_hessian(weighted, fit_intercept):
    """Every accepted pair implements the full weighted Hessian, including bias."""
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(96, 12)).astype(np.float32)
    raw -= raw.mean(axis=0)
    weights = rng.uniform(0.1, 2.0, size=len(raw)).astype(np.float32) if weighted else None
    sw_sum = float(weights.astype(np.float64).sum()) if weighted else float(len(raw))
    l2 = 0.03
    p = raw.shape[1] + int(fit_intercept)
    with mx.stream(mx.cpu):
        X = mx.array(raw)
        sw = mx.array(weights) if weighted else None
        h0 = mx.full((p,), 2.0, dtype=mx.float32)
        hs, hy, hc, count = _sketch_module().build_sketch_history(
            X, sw, sw_sum, l2, fit_intercept, h0,
            history_size=6, sketch_size=4, seed=5,
        )
        hs, hy, hc = np.asarray(hs), np.asarray(hy), np.asarray(hc)
    assert count == 4
    assert hs.shape == hy.shape == (6, p)
    for s, observed, coefficients in zip(hs[:count], hy[:count], hc[:count], strict=True):
        assert not fit_intercept or s[-1] == 0.0
        projection = raw.astype(np.float64) @ s[:raw.shape[1]].astype(np.float64)
        if weighted:
            projection *= weights.astype(np.float64)
        projection /= 4.0 * sw_sum
        expected = raw.astype(np.float64).T @ projection + l2 * s[:raw.shape[1]]
        if fit_intercept:
            expected = np.append(expected, np.sum(projection))
        np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=1e-7)
        sy = float(s.astype(np.float64) @ observed.astype(np.float64))
        yy = 2.0 * float(observed.astype(np.float64) @ observed.astype(np.float64))
        np.testing.assert_allclose(coefficients, [1.0 / sy, sy / yy], rtol=3e-5)
    assert np.all(hs[count:] == 0.0)
    assert np.all(hy[count:] == 0.0)
    assert np.all(hc[count:] == 0.0)


def test_rank_deficient_sketch_remains_in_feature_row_space():
    rng = np.random.default_rng(83)
    raw = rng.normal(size=(64, 5)).astype(np.float32)
    ratio = np.array([2.0, 0.5, 2.0, 0.5, 2.0], dtype=np.float32)
    X = np.concatenate([raw, raw * ratio], axis=1)
    with mx.stream(mx.cpu):
        hs, hy, _, count = _sketch_module().build_sketch_history(
            mx.array(X), None, len(X), 1e-8, True, mx.ones((11,)),
            history_size=8, sketch_size=4,
        )
        hs, hy = np.asarray(hs), np.asarray(hy)
    assert count == 4
    np.testing.assert_allclose(hs[:count, 5:10], hs[:count, :5] * ratio, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(hy[:count, 5:10], hy[:count, :5] * ratio, rtol=3e-6, atol=3e-7)


def test_zero_design_and_zero_sketch_leave_unseeded_history():
    with mx.stream(mx.cpu):
        X = mx.zeros((16, 5), dtype=mx.float32)
        h0 = mx.ones((6,), dtype=mx.float32)
        for size in (0, 4):
            hs, hy, hc, count = _sketch_module().build_sketch_history(
                X, None, 16.0, 0.1, True, h0, history_size=5, sketch_size=size,
            )
            assert count == 0
            assert not bool(mx.any(hs)) and not bool(mx.any(hy)) and not bool(mx.any(hc))


@pytest.mark.parametrize("sw_sum", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_weight_sum_is_rejected(sw_sum):
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="sw_sum"):
            _sketch_module().build_sketch_history(
                mx.zeros((5, 4)), None, sw_sum, 0.1, True, mx.ones((5,))
            )


@pytest.mark.gpu
@pytest.mark.parametrize("weight_mode", ["none", "ones", "nonuniform"])
def test_native_seed_history_and_device_output(weight_mode):
    """Host and device output use the same weighted specialization and budget."""
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    from mlxlearn._kernels.logistic_input import prepare_centered
    from mlxlearn._solvers import logistic, native_command, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSConfig

    if not native_command._supports_metal_dlpack() or native_command._metal_loop is None:
        pytest.skip("requires the supported native Metal loop")
    with mx.stream(mx.gpu):
        n, d = 4_096, 512
        X = mx.random.normal((n, d), key=mx.random.key(123))
        y = (X[:, 0] > 0.1).astype(mx.float32)
        sw = (mx.where(mx.arange(n) % 2 == 0, mx.array(0.5), mx.array(1.5))
              if weight_mode == "nonuniform" else mx.ones((n,), dtype=mx.float32))
        sw_argument = None if weight_mode == "none" else sw
        weighted = sw_argument is not None
        centered, center, finite, parts = prepare_centered(X, y, sw_argument)
        reg = mx.concatenate([mx.full((d,), 1.0 / n), mx.zeros((1,))])
        g, h0 = native_logistic.initial_from_parts(parts, reg, mx.array(float(n)))
        hs, hy, hc, count = _sketch_module().build_sketch_history(
            centered, sw_argument, float(n), 1.0 / n, True, h0, history_size=10, sketch_size=4
        )
        assert bool(finite) and count > 0
        fun = logistic._native_binary_objective(centered, y, sw_argument, float(n), 1.0 / n, True)
        state = native_command.run_native_loop(
            centered, y, sw_argument, float(n), 1.0 / n, True,
            LBFGSConfig(max_iter=100, tol=1e-4), fun,
            initial_parts=parts, convergence_center=center, initial_history=(hs, hy, hc, count),
        )
        assert state is not None and state.converged
        values = (centered, y, sw, g, h0, reg)
        output = mx.zeros((d + 1,), dtype=mx.float32)
        mx.eval(*values, center, output, hs, hy, hc)
        mx.synchronize()
        caps = tuple(value.__dlpack__() for value in values)
        seed_caps = tuple(value.__dlpack__() for value in (hs, hy, hc))
        device_caps = (center.__dlpack__(), output.__dlpack__())
        owners = (caps, *values, seed_caps, hs, hy, hc, device_caps, center, output)
        groups, fg, rows = 2, 1, 16
        blocks = (n + (groups // fg) * rows - 1) // ((groups // fg) * rows)
        source = native_command._source(n, d, d + 1, 10, 8, blocks, groups, rows, fg, weighted)
        weighted_argument = (1,) if weighted else ()
        raw, iterations, status, *_ = native_command._metal_loop.run(
            source, caps, n, d, d + 1, 10, blocks, groups, 100, float(n), 1e-4,
            1e-4, 1e-12, owners, b"", seed_caps, count, device_caps, *weighted_argument,
        )
        assert raw is None and status == 1
        assert 0 <= iterations == state.n_iter <= 100
        np.testing.assert_allclose(np.asarray(output), np.asarray(state.x).ravel(), rtol=0, atol=1e-6)

        def run_with_buffers(center_value, output_value):
            mx.eval(center_value, output_value)
            bad_caps = (center_value.__dlpack__(), output_value.__dlpack__())
            bad_owners = (caps, *values, seed_caps, hs, hy, hc, bad_caps, center_value, output_value)
            return native_command._metal_loop.run(
                source, caps, n, d, d + 1, 10, blocks, groups, 100, float(n), 1e-4,
                1e-4, 1e-12, bad_owners, b"", seed_caps, count, bad_caps, *weighted_argument,
            )

        with pytest.raises(RuntimeError, match="contiguous"):
            run_with_buffers(mx.zeros((d - 1,), dtype=mx.float32), output)
        with pytest.raises(RuntimeError, match="float32 Metal"):
            run_with_buffers(center, mx.zeros((d + 1,), dtype=mx.int32))
        with pytest.raises(RuntimeError, match="must not alias"):
            run_with_buffers(center, reg)


@pytest.mark.parametrize(
    ("status", "n_iter", "value", "gradient", "reason"),
    [
        (0, 5, 0.2, 2e-4, "max_iter"),
        (2, 2, 0.2, 2e-4, "line-search-failure"),
        (2, 2, 0.2, 0.0, "line-search-failure"),
        (3, 2, 0.2, 2e-4, "non-finite-objective"),
        (1, 2, float("nan"), 2e-4, "non-finite-objective"),
        (1, 2, 0.2, 2e-4, "line-search-failure"),
    ],
)
def test_seeded_failure_does_not_restart_full_iteration_budget(
    monkeypatch, status, n_iter, value, gradient, reason
):
    """A tried sketch yields a scored failure instead of requesting a restart."""
    from types import SimpleNamespace

    from mlxlearn._solvers import native_command
    from mlxlearn._solvers.lbfgs import LBFGSConfig

    with mx.stream(mx.cpu):
        n, d, m = 5, 512, 10
        X = mx.ones((n, d), dtype=mx.float32)
        y = mx.array([0, 1, 0, 1, 0], dtype=mx.float32)
        parts = mx.zeros((2, 1, d + 1), dtype=mx.float32)
        hs = mx.zeros((m, d + 1), dtype=mx.float32)
        hy = mx.zeros((m, d + 1), dtype=mx.float32)
        hc = mx.zeros((m, 2), dtype=mx.float32)
        calls = []

        def run(*args):
            calls.append(args)
            return np.zeros(d + 1, dtype=np.float32).tobytes(), n_iter, status, 16, 0.0, 0.0

        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "fake Metal source")
        monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
        def fun(theta):
            return value, mx.full(theta.shape, gradient, dtype=mx.float32)
        state = native_command.run_native_loop(
            X, y, None, float(n), 0.1, True, LBFGSConfig(max_iter=5, tol=1e-4), fun,
            initial_parts=parts, initial_history=(hs, hy, hc, 2),
        )
    assert len(calls) == 1
    assert len(calls[0]) == 17 and calls[0][16] == 2
    assert len(calls[0][13]) == 11 and calls[0][13][7] is calls[0][15]
    assert state is not None and not state.converged
    assert state.n_iter == n_iter and state.stop_reason == reason


def test_extended_seed_and_device_owners_survive_failed_native_wait():
    """A failed wait retains extra seed and output owners until GPU completion."""
    import gc
    import weakref

    from mlxlearn._solvers import native_command

    loop = native_command._metal_loop
    if loop is None:
        pytest.skip("requires the optional native extension")

    class Owner:
        pass

    originals = [Owner() for _ in range(16)]
    refs = [weakref.ref(owner) for owner in originals]
    owners = tuple(originals)
    before = loop._test_retained_count()
    token = loop._test_begin_retention(owners)
    try:
        message = loop._test_fail_waits(token, 2, 1)
        assert "failed to confirm completion" in message
        del owners, originals
        gc.collect()
        assert all(reference() is not None for reference in refs)
        assert loop._test_retained_count() == before + 1
        loop._test_set_terminal(token)
        loop._reap_completed()
        gc.collect()
        assert all(reference() is None for reference in refs)
        assert loop._test_retained_count() == before
    finally:
        loop._test_set_terminal(token)
        loop._reap_completed()


def test_empty_sketch_attempt_does_not_request_a_fresh_iteration_budget(monkeypatch):
    from types import SimpleNamespace

    from mlxlearn._solvers import native_command
    from mlxlearn._solvers.lbfgs import LBFGSConfig

    with mx.stream(mx.cpu):
        n, d, m = 5, 512, 10
        X = mx.ones((n, d), dtype=mx.float32)
        y = mx.array([0, 1, 0, 1, 0], dtype=mx.float32)
        parts = mx.zeros((2, 1, d + 1), dtype=mx.float32)
        history = (
            mx.zeros((m, d + 1), dtype=mx.float32),
            mx.zeros((m, d + 1), dtype=mx.float32),
            mx.zeros((m, 2), dtype=mx.float32),
            0,
        )
        calls = []

        def run(*args):
            calls.append(args)
            return np.zeros(d + 1, dtype=np.float32).tobytes(), 5, 0, 16, 0.0, 0.0

        def direct(theta):
            return 0.2, mx.full(theta.shape, 2e-4, dtype=mx.float32)

        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "fake Metal source")
        monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
        state = native_command.run_native_loop(
            X, y, None, float(n), 0.1, True, LBFGSConfig(max_iter=5, tol=1e-4), direct,
            initial_parts=parts, initial_history=history,
        )
    assert len(calls) == 1 and len(calls[0]) == 15  # no invalid zero-length seed bundle
    assert not state.converged and state.stop_reason == "max_iter" and state.n_iter == 5


@pytest.mark.gpu
def test_seeded_native_fit_preserves_rank_deficient_row_space():
    """Near-zero regularization must not create Jacobi-style null coefficients."""
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    from mlxlearn._kernels.logistic_input import prepare_centered
    from mlxlearn._solvers import logistic, native_command, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSConfig

    if not native_command._supports_metal_dlpack() or native_command._metal_loop is None:
        pytest.skip("requires the supported native Metal loop")
    with mx.stream(mx.gpu):
        n, d = 4_096, 512
        left = mx.random.normal((n, d // 2), key=mx.random.key(61))
        ratio = mx.tile(mx.array([2.0, 0.5], dtype=mx.float32), (d // 4,))
        X = mx.concatenate([left, left * ratio], axis=1)
        y = (left[:, 0] + 0.2 * left[:, 1] > 0).astype(mx.float32)
        sw = mx.ones((n,), dtype=mx.float32)
        X, center, finite, parts = prepare_centered(X, y, sw)
        l2 = 1.0 / (1e6 * n)
        reg = mx.concatenate([mx.full((d,), l2), mx.zeros((1,))])
        _, h0 = native_logistic.initial_from_parts(parts, reg, mx.array(float(n)))
        history = _sketch_module().build_sketch_history(
            X, sw, float(n), l2, True, h0, history_size=10, sketch_size=4
        )
        assert bool(finite) and history[3] > 0
        fun = logistic._native_binary_objective(X, y, sw, float(n), l2, True)
        config = LBFGSConfig(max_iter=200, tol=1e-4)
        for initial_history in (None, history):
            state = native_command.run_native_loop(
                X, y, sw, float(n), l2, True, config, fun, initial_parts=parts,
                convergence_center=center, initial_history=initial_history,
            )
            assert state is not None and state.converged
            weights = np.asarray(state.x).ravel()[:d]
            scale = np.asarray(ratio)
            residual = np.linalg.norm(weights[d // 2:] - scale * weights[:d // 2])
            assert residual / max(np.linalg.norm(weights), 1e-30) < 1e-5
            assert state.grad_max_abs <= config.tol


def test_sketch_bridge_and_shared_helper_have_identical_metric_bits(monkeypatch):
    """The seeded recurrence must use exactly the h0 used to build its pairs."""
    from types import SimpleNamespace

    from mlxlearn._solvers import native_command, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSConfig
    from mlxlearn_device import hessian_sketch

    rng = np.random.default_rng(77)
    n, d, m = 7, 512, 10
    with mx.stream(mx.cpu):
        X, y = mx.ones((n, d), mx.float32), mx.zeros((n,), mx.float32)
        parts = mx.array(rng.uniform(0.01, 2.0, size=(2, 1, d + 1)).astype(np.float32))
        reg = mx.concatenate([mx.full((d,), 0.1), mx.zeros((1,))])
        expected_g, expected_h0 = hessian_sketch.initial_metric_from_parts(parts, reg, mx.array(float(n)))
        history = (mx.zeros((m, d + 1)), mx.zeros((m, d + 1)), mx.zeros((m, 2)), 2)
        received = []
        compiled_calls = []

        def compiled(*args):
            compiled_calls.append(1)
            return mx.zeros((d + 1,)), mx.full((d + 1,), 3.0)

        def run(*args):
            owners = args[13]
            received.append((np.asarray(owners[4]).copy(), np.asarray(owners[5]).copy()))
            return np.zeros(d + 1, dtype=np.float32).tobytes(), 2, 1, 16, 0.0, 0.0

        def direct(theta):
            return 0.2, mx.zeros(theta.shape)

        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "fake Metal source")
        monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
        monkeypatch.setattr(native_logistic, "initial_from_parts", compiled)
        config = LBFGSConfig(max_iter=5)
        native_command.run_native_loop(X, y, None, float(n), 0.1, True, config, direct,
                                       initial_parts=parts, initial_history=history)
        native_command.run_native_loop(X, y, None, float(n), 0.1, True, config, direct,
                                       initial_parts=parts)
    np.testing.assert_array_equal(received[0][0].view(np.uint32), np.asarray(expected_g).view(np.uint32))
    np.testing.assert_array_equal(received[0][1].view(np.uint32), np.asarray(expected_h0).view(np.uint32))
    assert compiled_calls == [1]
    np.testing.assert_array_equal(received[1][1], np.full(d + 1, 3.0, dtype=np.float32))
