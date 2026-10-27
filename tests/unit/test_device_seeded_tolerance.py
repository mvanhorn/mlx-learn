"""Seeded native stopping leaves room for independent device certification."""
from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits

from mlxlearn._common.device import gpu_available
from mlxlearn_device import DeviceConvergenceWarning, _logistic, fit_logistic


def test_seeded_native_tolerance_rounds_down_and_stays_normal():
    minimum = 1.1754943508222875e-38
    for requested in (minimum, 1e-37, 2.5e-5, 1.0):
        certificate_tol = _logistic._device_tol(requested)
        native_tol = _logistic._seeded_native_tol(certificate_tol)
        assert minimum <= native_tol <= certificate_tol
        if certificate_tol * 0.95 >= minimum:
            assert native_tol <= certificate_tol * 0.95
    assert _logistic._seeded_native_tol(minimum) == minimum


@pytest.mark.gpu
def test_seeded_attempt_reserves_margin_but_retry_keeps_certificate_tol(monkeypatch):
    if not gpu_available() or _logistic.metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    calls = []

    def fake_run(*args):
        calls.append(args)
        return None, 3 if len(calls) == 1 else 2, 1 if len(calls) == 1 else 0, 0, 0.0, 0.0

    def fake_score(*args):
        # First attempt fails the unchanged 0.95 safety margin. The retry is
        # above an accidentally double-tightened margin but below the real one.
        value = 2.4e-5 if len(calls) == 1 else 2.34e-5
        return mx.array(0.2), mx.array(value), mx.zeros(513), mx.zeros(513), mx.array(0.0)

    def fake_sketch(*args, **kwargs):
        return mx.zeros((10, 513)), mx.zeros((10, 513)), mx.zeros((10, 2)), 1

    monkeypatch.setattr(_logistic.metal_loop, "run", fake_run)
    monkeypatch.setattr(_logistic, "score_original", fake_score)
    monkeypatch.setattr(_logistic, "build_sketch_history", fake_sketch)
    with mx.stream(mx.gpu), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = mx.reshape(mx.arange(4 * 512, dtype=mx.float32), (4, 512)) / 2048
        y = mx.array([0, 1, 0, 1])
        model = fit_logistic(X, y, max_iter=10, sketch_size=1)
    certificate_tol = _logistic._device_tol(2.5e-5)
    assert len(calls) == 2 and calls[0][8] == 10 and calls[1][8] == 7
    assert calls[0][10] == _logistic._seeded_native_tol(certificate_tol)
    assert calls[1][10] == certificate_tol
    assert calls[0][16] == 1 and calls[1][15] is None and calls[1][16] == 0
    assert model.used_unsketched_fallback_ and model.converged and model.n_iter == 5
    assert not caught


@pytest.mark.gpu
@pytest.mark.parametrize("sketch_size", [0, 1])
def test_unseeded_and_invalid_sketch_keep_native_tolerance(monkeypatch, sketch_size):
    if not gpu_available() or _logistic.metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    calls = []

    def fake_run(*args):
        calls.append(args)
        return None, 1, 2, 0, 0.0, 0.0

    def fake_sketch(*args, **kwargs):
        return mx.zeros((10, 513)), mx.zeros((10, 513)), mx.zeros((10, 2)), 0

    monkeypatch.setattr(_logistic.metal_loop, "run", fake_run)
    monkeypatch.setattr(_logistic, "build_sketch_history", fake_sketch)
    monkeypatch.setattr(_logistic, "score_original", lambda *args: (
        mx.array(0.2), mx.array(1e-6), mx.zeros(513), mx.zeros(513), mx.array(0.0)
    ))
    with mx.stream(mx.gpu), pytest.warns(DeviceConvergenceWarning):
        X = mx.reshape(mx.arange(4 * 512, dtype=mx.float32), (4, 512)) / 2048
        model = fit_logistic(X, mx.array([0, 1, 0, 1]), max_iter=1, sketch_size=sketch_size)
    assert len(calls) == 1 and calls[0][10] == _logistic._device_tol(2.5e-5)
    assert calls[0][15] is None and calls[0][16] == 0
    assert not model.converged and model.stop_reason == "line-search-failure"
    assert model.n_iter == 1 and not model.used_unsketched_fallback_


@pytest.mark.gpu
def test_ill_conditioned_weighted_sketch_passes_independent_certificate():
    if not gpu_available() or _logistic.metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    rng = np.random.default_rng(41)
    n, d = 20_000, 1_024
    X = rng.standard_normal((n, d), dtype=np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    X[:, d // 2:] = X[:, :d // 2] + np.float32(0.01) * X[:, d // 2:]
    X *= np.logspace(-1, 1, d, dtype=np.float32)
    weights = rng.uniform(0.25, 3.0, n).astype(np.float32)
    with threadpool_limits(limits=1), mx.stream(mx.gpu), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit_logistic(mx.array(X), mx.array(y), sample_weight=mx.array(weights), max_iter=200, sketch_size=4)
        unseeded = fit_logistic(mx.array(X), mx.array(y), sample_weight=mx.array(weights), max_iter=200, sketch_size=0)
        assert not caught and model.converged and unseeded.converged
        assert not model.used_unsketched_fallback_ and model.n_iter <= 200

        def score(fit):
            coefficient = np.asarray(fit.coef_, dtype=np.float64).reshape(-1)
            center = np.asarray(fit.center_, dtype=np.float64)
            intercept = float(fit.centered_intercept_) - coefficient @ center
            gradient = np.zeros(d, dtype=np.float64)
            bias = loss = 0.0
            for start in range(0, n, 256):
                end = min(n, start + 256)
                block = X[start:end].astype(np.float64)
                logits = block @ coefficient + intercept
                residual = (expit(logits) - y[start:end]) * weights[start:end]
                gradient += block.T @ residual
                bias += residual.sum()
                loss += np.dot(weights[start:end], np.logaddexp(0.0, logits) - y[start:end] * logits)
            total = weights.sum(dtype=np.float64)
            objective = loss / total + np.dot(coefficient, coefficient) / (2 * total)
            gradient = gradient / total + coefficient / total
            return objective, max(np.max(np.abs(gradient)), abs(bias / total))

        objective, gradient = score(model)
        unseeded_objective, unseeded_gradient = score(unseeded)
    assert gradient <= 2.5e-5 * 0.95 and unseeded_gradient <= 1e-4
    assert objective <= unseeded_objective * (1 + 1e-5)
