"""Objective arithmetic and budgeting for the private inexact Newton driver."""
from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _newton


def test_represented_loss_difference_matches_original_float64_objective():
    rng = np.random.default_rng(927)
    X = rng.normal(size=(97, 9)).astype(np.float32)
    y = rng.integers(0, 2, 97).astype(np.float32)
    sw = rng.uniform(0.1, 2, 97).astype(np.float32)
    sw[::7] = 0
    old = rng.normal(size=10).astype(np.float32)
    new = (old + rng.normal(scale=3e-6, size=10)).astype(np.float32)
    denominator = float(sw.sum(dtype=np.float64))
    reg = np.r_[np.full(9, .013, dtype=np.float32), np.float32(0)]
    with mx.stream(mx.cpu):
        xd, yd, wd, before, after, penalty = map(mx.array, (X, y, sw, old, new, reg))
        logits = xd @ before[:-1] + before[-1]
        difference, step = _newton._represented_difference(
            xd, yd, wd, before, after, logits, mx.array(denominator, mx.float32), penalty,
        )
        mx.eval(difference, step)
    sign = 1 - 2 * y.astype(np.float64)
    oldlog = X.astype(np.float64) @ old[:-1].astype(np.float64) + float(old[-1])
    newlog = X.astype(np.float64) @ new[:-1].astype(np.float64) + float(new[-1])
    expected = np.sum(sw.astype(np.float64) * (
        np.logaddexp(0, sign * newlog) - np.logaddexp(0, sign * oldlog)
    )) / denominator
    expected += .5 * np.sum(reg.astype(np.float64) * (
        new.astype(np.float64) ** 2 - old.astype(np.float64) ** 2
    ))
    np.testing.assert_allclose(float(difference), expected, rtol=2e-5, atol=2e-12)
    np.testing.assert_array_equal(np.asarray(step), new - old)


def test_stable_loss_difference_handles_large_and_saturated_changes():
    X = np.array([[1], [-1], [0], [1e20]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    sw = np.array([1, 1, 2, 0], dtype=np.float32)
    old = np.array([80, 0], dtype=np.float32)
    new = np.array([30, -1], dtype=np.float32)
    with mx.stream(mx.cpu):
        xd, yd, wd, before, after = map(mx.array, (X, y, sw, old, new))
        logits = mx.where(wd > 0, xd @ before[:-1] + before[-1], 0)
        difference, _ = _newton._represented_difference(
            xd, yd, wd, before, after, logits, mx.array(4, mx.float32), mx.zeros(2),
        )
        mx.eval(difference)
    before_log = np.array([80, -80, 0], dtype=np.float64)
    after_log = np.array([29, -31, -1], dtype=np.float64)
    sign = 1 - 2 * y[:3]
    expected = np.sum(sw[:3] * (np.logaddexp(0, sign * after_log)
                      - np.logaddexp(0, sign * before_log))) / 4
    np.testing.assert_allclose(float(difference), expected, rtol=1e-6)


def test_private_newton_has_no_numpy_or_sklearn_import():
    source = inspect.getsource(_newton)
    assert "import numpy" not in source
    assert "from sklearn" not in source
    assert "import sklearn" not in source


@pytest.mark.parametrize("field,value", [
    ("tol", 0), ("l2", -1), ("denominator", float("inf")),
    ("max_iter", True), ("max_iter", 0), ("max_cg", 513),
    ("relative_gap_tol", float("nan")),
])
def test_invalid_solver_controls_fail_before_native_work(field, value):
    with mx.stream(mx.cpu):
        X = mx.zeros((4, 3), mx.float32)
        options = {"denominator": 4.0, "l2": .1, "tol": 1e-4, "max_iter": 20,
                   "max_cg": 128, "relative_gap_tol": 1e-5}
        options[field] = value
        with pytest.raises(ValueError):
            _newton._validate(X, X, mx.zeros(3), mx.zeros(4), mx.ones(4), **options)


def _require_native():
    if (not mx.metal.is_available() or _newton.solve_cg.__module__ != "mlxlearn_device._native_cg"):
        pytest.skip("requires native Metal CG")
    from mlxlearn_device._native_cg import metal_loop
    if metal_loop is None or not hasattr(metal_loop, "run_cg"):
        pytest.skip("requires native Metal CG extension")


def _weighted_problem():
    rng = np.random.default_rng(834)
    X = rng.integers(-16, 17, (1024, 65)).astype(np.float32) / np.float32(8)
    X[:, -1] = X[:, 1] + X[:, 32]
    y = (X[:, 0] + .2 * X[:, 1] + rng.normal(size=len(X)) > 0).astype(np.float32)
    sw = rng.uniform(.3, 2, len(X)).astype(np.float32)
    sw[::11] = 0
    return X, y, sw


@pytest.mark.gpu
def test_newton_weighted_objective_and_general_rank_match_independent_reference():
    _require_native()
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    X, y, sw = _weighted_problem()
    total = float(sw.sum(dtype=np.float64))
    with threadpool_limits(limits=1):
        reference = LogisticRegression(tol=1e-11, max_iter=2000).fit(
            X.astype(np.float64), y, sample_weight=sw.astype(np.float64),
        )
    with mx.stream(mx.gpu):
        xd, yd, wd = map(mx.array, (X, y, sw))
        center = mx.mean(xd, axis=0)
        result = _newton.minimize_newton(xd, xd - center, center, yd, wd, total, 1 / total,
                                        tol=2.5e-5, max_iter=30)
        theta, ch = np.asarray(result.theta, np.float64), np.asarray(center, np.float64)
    assert result.converged and result.stop_reason == "duality-gap"
    assert 0 < result.n_iter <= 30 and result.hvp_count > 0
    assert result.relative_gap <= 1e-5
    assert result.gradient_max_abs <= 2.5e-5
    logit = X.astype(np.float64) @ theta[:-1] + theta[-1] - theta[:-1] @ ch
    ref_logit = reference.decision_function(X.astype(np.float64))
    objective = (np.sum(sw * np.logaddexp(0, (1 - 2 * y) * logit))
                 + .5 * np.sum(theta[:-1] ** 2)) / total
    expected = (np.sum(sw * np.logaddexp(0, (1 - 2 * y) * ref_logit))
                + .5 * np.sum(reference.coef_ ** 2)) / total
    assert (objective - expected) / expected <= 1e-5
    null = abs(theta[-2] - theta[1] - theta[32]) / (np.sqrt(3) * np.linalg.norm(theta[:-1]))
    assert null < 2e-5


@pytest.mark.gpu
def test_newton_accepted_budget_and_resume_are_explicit():
    _require_native()
    X, y, sw = _weighted_problem()
    total = float(sw.sum(dtype=np.float64))
    with mx.stream(mx.gpu):
        xd, yd, wd = map(mx.array, (X, y, sw))
        center = mx.mean(xd, axis=0)
        centered = xd - center
        first = _newton.minimize_newton(xd, centered, center, yd, wd, total, 1 / total,
                                       tol=2.5e-5, max_iter=1)
        assert first.n_iter == 1 and not first.converged and first.stop_reason == "max_iter"
        resumed = _newton.minimize_newton(xd, centered, center, yd, wd, total, 1 / total,
                                         tol=2.5e-5, max_iter=29, initial=first.theta)
    assert resumed.converged
    assert first.n_iter + resumed.n_iter <= 30
