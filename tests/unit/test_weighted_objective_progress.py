"""Weighted binary convergence should track objective progress, not just max gradient."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import native_command
from mlxlearn.linear_model import LogisticRegression


def _score(model, X, y, weights):
    coefficient = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    if hasattr(model, "centered_intercept_"):
        intercept = float(model.centered_intercept_) - coefficient @ np.asarray(model.center_, dtype=np.float64)
    else:
        intercept = float(model.intercept_[0])
    denominator = weights.sum(dtype=np.float64)
    loss = 0.0
    gradient = np.zeros(X.shape[1], dtype=np.float64)
    bias_gradient = 0.0
    for first in range(0, len(y), 256):
        last = min(first + 256, len(y))
        block = np.asarray(X[first:last], dtype=np.float64)
        logits = block @ coefficient + intercept
        residual = (expit(logits) - y[first:last]) * weights[first:last]
        loss += np.dot(weights[first:last], np.logaddexp(0, logits) - y[first:last] * logits)
        gradient += block.T @ residual
        bias_gradient += residual.sum()
    loss = loss / denominator + np.dot(coefficient, coefficient) / (2 * denominator)
    gradient = gradient / denominator + coefficient / denominator
    return loss, max(np.max(np.abs(gradient)), abs(bias_gradient / denominator))


@pytest.fixture(scope="module", params=[(41, 0.90), (83, 0.90), (2027, 0.90), (2027, 0.98)])
def unbalanced_weighted(request):
    seed, quantile = request.param
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((20_000, 1024), dtype=np.float32)
    rng.normal(scale=0.5, size=len(X))  # Match the established fixture's RNG order.
    y = (X[:, 0] > np.quantile(X[:, 0], quantile)).astype(np.int32)
    weights = rng.uniform(0.25, 3.0, len(X))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        center = X.mean(axis=0, dtype=np.float64)
        centered = X.astype(np.float64) - center
        reference = SkLogisticRegression(C=1, tol=1e-10, max_iter=2000).fit(centered, y, sample_weight=weights)
        reference = SimpleNamespace(coef_=reference.coef_, intercept_=reference.intercept_ - reference.coef_ @ center)
        reference_score, reference_gradient = _score(reference, X, y, weights)
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert reference_gradient < 1e-7
    return X, y, weights, reference_score


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["host", "sketch", "device", "device_unsketched"])
def test_weighted_unbalanced_objective_matches_tight_reference(unbalanced_weighted, route):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    X, y, weights, reference = unbalanced_weighted
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if route in ("host", "sketch"):
            with tuning_context(force_backend="mlx", lbfgs_sketch_size=4 if route == "sketch" else 0):
                model = LogisticRegression(C=1, tol=1e-4, max_iter=200).fit(X, y, sample_weight=weights)
        else:
            from mlxlearn_device import fit_logistic

            model = fit_logistic(mx.array(X), mx.array(y), sample_weight=mx.array(weights, dtype=mx.float32),
                                 C=1, tol=1e-4, max_iter=200, sketch_size=4 if route == "device" else 0)
            assert model.converged
        objective, gradient = _score(model, X, y, weights)
    assert not caught
    assert gradient <= 1e-4
    assert (objective - reference) / abs(reference) <= 1e-5


def test_zero_budget_weighted_requires_exact_stationarity():
    from mlxlearn._solvers.lbfgs import LBFGSConfig
    from mlxlearn._solvers.native_logistic import minimize_binary

    with mx.stream(mx.cpu):
        X = mx.ones((2, 2), dtype=mx.float32)
        y = mx.array([0, 1], dtype=mx.float32)
        weights = mx.array([0.5, 1.5], dtype=mx.float32)
        config = LBFGSConfig(max_iter=0, tol=1e-4)
        def almost_stationary(theta):
            return 0.69, mx.full(theta.shape, 1e-6, dtype=mx.float32)

        def stationary(theta):
            return 0.69, mx.zeros(theta.shape, dtype=mx.float32)
        result = minimize_binary(X, y, weights, 2.0, 0.1, True, config, almost_stationary)
        optimum = minimize_binary(X, y, weights, 2.0, 0.1, True, config, stationary)
    assert result.n_iter == 0 and not result.converged and result.stop_reason == "max_iter"
    assert optimum.n_iter == 0 and optimum.converged


@pytest.mark.gpu
@pytest.mark.parametrize("sketch_size", [0, 4])
def test_device_exact_unit_weights_use_unweighted_path(sketch_size):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    from mlxlearn_device import fit_logistic

    rng = np.random.default_rng(17)
    X = rng.standard_normal((2048, 512), dtype=np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(np.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_device, y_device = mx.array(X), mx.array(y)
        ordinary = fit_logistic(X_device, y_device, sketch_size=sketch_size)
        ones = fit_logistic(X_device, y_device, sample_weight=mx.ones(len(y)), sketch_size=sketch_size)
    assert not caught
    assert ordinary.converged and ones.converged and ordinary.n_iter == ones.n_iter
    np.testing.assert_array_equal(np.asarray(ordinary.coef_), np.asarray(ones.coef_))
    np.testing.assert_array_equal(np.asarray(ordinary.centered_intercept_), np.asarray(ones.centered_intercept_))
