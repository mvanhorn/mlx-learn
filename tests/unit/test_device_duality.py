"""Independent checks for the private float32 logistic dual-gap diagnostic."""

from __future__ import annotations

import ast
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit, xlogy

from mlxlearn_device._duality import _scaled_bernoulli_kl, logistic_duality_gap


def _oracle(X, y, theta, weights, denominator, l2):
    X, theta = np.asarray(X, np.float64), np.asarray(theta, np.float64)
    a = np.asarray(weights, np.float64) / denominator
    d = X.shape[1]
    z = X @ theta[:d]
    intercept = theta.size == d + 1
    if intercept:
        z += theta[d]
    u = (1 - 2 * y) * z
    q = expit(u)
    beta = a * q
    if intercept:
        m0, m1 = beta[y == 0].sum(), beta[y == 1].sum()
        target = min(m0, m1)
        beta *= np.where(y == 0, target / m0 if m0 else 1, target / m1 if m1 else 1)
    residual = (1 - 2 * y) * beta
    primal = np.dot(a, np.logaddexp(0, u)) + 0.5 * l2 * np.dot(theta[:d], theta[:d])
    feasible_q = np.divide(beta, a, out=np.zeros_like(beta), where=a > 0)
    entropy = xlogy(feasible_q, feasible_q) + xlogy(1 - feasible_q, 1 - feasible_q)
    dual = -np.dot(a, entropy) - np.dot(X.T @ residual, X.T @ residual) / (2 * l2)
    return primal - dual, primal


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_duality_matches_independent_primal_minus_dual(intercept, weighted):
    rng = np.random.default_rng(913)
    X = rng.normal(size=(63, 7)).astype(np.float32)
    y = rng.integers(0, 2, len(X), dtype=np.int32)
    theta = rng.normal(size=7 + intercept).astype(np.float32)
    weights = rng.uniform(0.1, 3, len(X)).astype(np.float32) if weighted else np.ones(len(X), np.float32)
    weights[::13] = 0
    denominator = float(weights.sum(dtype=np.float64))
    expected, objective = _oracle(X, y, theta, weights, denominator, 0.15)
    with mx.stream(mx.cpu):
        result = logistic_duality_gap(mx.array(X), mx.array(y), mx.array(theta),
                                     mx.array(weights), denominator, 0.15, device=mx.cpu)
        mx.eval(result.gap, result.valid, result.objective, result.feasibility_error)
    assert bool(result.valid)
    assert float(result.gap) == pytest.approx(expected, rel=3e-6, abs=3e-7)
    assert float(result.objective) == pytest.approx(objective, rel=3e-6, abs=3e-7)
    assert float(result.feasibility_error) < 1e-7


def test_balanced_optimum_has_zero_gap_with_explicit_reserve():
    with mx.stream(mx.cpu):
        result = logistic_duality_gap(mx.zeros((8, 3)), mx.array([0, 1] * 4),
                                     mx.zeros((4,)), mx.ones((8,)), 8.0, 0.1, device=mx.cpu)
        mx.eval(result.gap, result.roundoff_margin, result.relative_gap, result.valid)
    assert bool(result.valid)
    assert float(result.gap) == 0
    assert 0 < float(result.relative_gap) < 4e-6
    assert float(result.roundoff_margin) > 0


def test_gap_detects_nullspace_error_despite_small_componentwise_gradient():
    # Exact duplicated columns make this coefficient vector a null direction.
    # Its logistic loss is optimal, but its L2 penalty is not.
    with mx.stream(mx.cpu):
        X = mx.ones((8, 2))
        theta = mx.array([1.0, -1.0, 0.0])
        result = logistic_duality_gap(X, mx.array([0, 1] * 4), theta,
                                     mx.ones((8,)), 8.0, 1e-5, device=mx.cpu)
        mx.eval(result.gap, result.relative_gap, result.valid)
    assert bool(result.valid)
    assert float(result.gap) == pytest.approx(1e-5, rel=2e-6)
    assert float(result.relative_gap) > 1e-5
    np.testing.assert_array_equal(np.array(theta), [1.0, -1.0, 0.0])


@pytest.mark.parametrize("u", [-100.0, -25.0, -1.0, 0.0, 1.0, 25.0, 100.0])
@pytest.mark.parametrize("scale", [0.0, 1e-20, 0.1, 0.5, 0.9999, 1.0])
def test_stable_scaled_kl_handles_saturated_logits(u, scale):
    q = expit(u)
    s = float(np.float32(scale))
    qs = s * q
    expected = (xlogy(qs, qs) + xlogy(1 - qs, 1 - qs)
                + qs * np.logaddexp(0, -u) + (1 - qs) * np.logaddexp(0, u))
    with mx.stream(mx.cpu):
        value = _scaled_bernoulli_kl(mx.array([u]), mx.array([s]))
        mx.eval(value)
    assert np.isfinite(float(value[0]))
    assert float(value[0]) == pytest.approx(expected, abs=2e-7, rel=3e-5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_active_weights_invalidate_diagnostic(bad):
    with mx.stream(mx.cpu):
        result = logistic_duality_gap(mx.zeros((2, 2)), mx.array([0, 1]), mx.zeros((3,)),
                                     mx.array([1.0, bad]), 2.0, 0.1, device=mx.cpu)
        mx.eval(result.valid, result.gap)
    assert not bool(result.valid)
    assert np.isinf(float(result.gap))


def test_zero_weight_ignores_unused_overflowed_logit():
    with mx.stream(mx.cpu):
        result = logistic_duality_gap(mx.zeros((3, 2)), mx.array([0, 1, 0]), mx.zeros((3,)),
                                     mx.array([1.0, 1.0, 0.0]), 2.0, 0.1, device=mx.cpu,
                                     logits=mx.array([0.0, 0.0, float("inf")]))
        mx.eval(result.valid, result.gap)
    assert bool(result.valid)
    assert float(result.gap) == 0


def test_no_numpy_or_sklearn_dependency():
    import mlxlearn_device._duality as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert all(not name.startswith(("numpy", "sklearn")) for name in imports if name)
