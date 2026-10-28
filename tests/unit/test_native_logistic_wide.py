"""Wide one-pass logistic gradients retain the accepted objective and tolerance."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn._solvers.logistic import _native_binary_objective
from mlxlearn._solvers.native_logistic import initial, minimize_binary, step
from mlxlearn._solvers.native_logistic_fused import forward_gradient

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _gpu_stream():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with use_device("gpu"):
        yield


@pytest.mark.parametrize(
    "width, intercept, n",
    [
        (width, intercept, 67)
        for width in [512, 1023, 1024, 1025, 2048, 4096]
        for intercept in [False, True]
    ]
    + [(1024, True, 16385), (2048, False, 16385), (4096, True, 16385)],
)
def test_speculative_gradient_matches_first_candidate(width, intercept, n):
    rng = np.random.default_rng(301)
    p = width + int(intercept)
    X = rng.normal(scale=0.2, size=(n, width)).astype(np.float32)
    direction = rng.normal(scale=0.05, size=p).astype(np.float32)
    raw = rng.normal(size=n).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    weights = np.geomspace(1e-3, 1e3, n).astype(np.float32)
    weights[::11] = 0
    denominator = float(weights.sum(dtype=np.float64))
    alpha = 2.0**-8  # The first candidate can belong to a continuation batch.
    stats = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, alpha, 0.0])
    projection, changes, gradient = forward_gradient(
        mx.array(X),
        mx.array(direction),
        mx.array(raw),
        mx.array(y),
        mx.array(weights),
        stats,
        mx.array(denominator),
        24,
    )
    mx.eval(projection, changes, gradient)
    candidate = raw.astype(np.float64) + alpha * np.asarray(projection, dtype=np.float64)
    residual = weights * (1 - 2 * y) * expit((1 - 2 * y) * candidate)
    expected = residual @ X / denominator
    if intercept:
        expected = np.append(expected, residual.sum() / denominator)
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=2e-5, atol=8e-8)
    assert np.isfinite(np.asarray(changes)).all()


def _single_step(X, y, inverse_scale, alpha_start=1.0, candidates=8):
    n, d = X.shape
    Xm, ym = mx.array(X), mx.array(y)
    weights, reg = mx.ones(n), mx.zeros(d)
    denominator = mx.array(float(n))
    gradient, _ = initial(Xm, ym, weights, d, reg, denominator)
    w = mx.zeros(d)
    history = mx.zeros((3, d))
    result = step(
        Xm,
        ym,
        weights,
        reg,
        denominator,
        mx.array(1e-8),
        mx.full(d, inverse_scale),
        mx.array(1e-4),
        mx.array(1e-10),
        candidates,
        mx.array(alpha_start),
        mx.zeros(d),
        w,
        gradient,
        w,
        gradient,
        history,
        mx.zeros_like(history),
        mx.zeros((3, 2)),
        mx.concatenate([mx.zeros(11), mx.array([np.log(2)])]),
        mx.zeros(n),
    )
    mx.eval(result)
    return result


@pytest.mark.parametrize(
    "inverse_scale, alpha, candidates, first",
    [
        (0.001, 1.0, 8, True),
        (0.02, 1.0, 8, False),
        (1.0, 2.0**-8, 24, True),
    ],
)
def test_step_uses_the_accepted_candidates_gradient(inverse_scale, alpha, candidates, first):
    # Rank-one curvature makes the accepted step known and forces backtracking.
    X = np.ones((259, 1024), dtype=np.float32)
    y = (np.arange(len(X)) % 2).astype(np.float32)
    state = _single_step(X, y, inverse_scale, alpha, candidates)
    w, gradient, metadata, raw = state[0], state[1], state[7], state[8]
    assert float(metadata[1]) == 0
    assert (float(metadata[8]) == alpha) is first
    assert float(metadata[8]) > 0
    residual = expit(np.asarray(raw, dtype=np.float64)) - y
    expected = residual @ X / len(X)
    # The rank-one fixture cancels residuals from both classes almost exactly.
    # The unchanged fallback reduction accumulates longer same-sign blocks,
    # losing a few ULPs relative to float64. Require it to match a fresh native
    # evaluation exactly. For the speculative path, the residuals being summed
    # are close to +/-0.5: two float32 ULPs at that magnitude bound the
    # observed cancellation error across GPU generations.
    if not first:
        direct = _native_binary_objective(mx.array(X), mx.array(y), None, float(len(X)), 0.0, False)
        _, direct_gradient = direct(w.reshape(1, -1))
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(direct_gradient)[0])
    np.testing.assert_allclose(
        np.asarray(gradient),
        expected,
        rtol=2e-4,
        atol=2 * float(np.spacing(np.float32(0.5))) if first else 8 * np.finfo(np.float32).eps,
    )
    if first and alpha == 2.0**-8:
        # The accepted candidate nearly cancels the initial gradient. This
        # must not pass by reusing the zero-parameter or unit-step gradient.
        assert abs(float(gradient[0]) - (0.5 - y).mean()) > 1e-3
        wrong_raw = -inverse_scale * (0.5 - y).mean() * X.shape[1]
        assert abs(float(gradient[0]) - (expit(wrong_raw) - y.mean())) > 1e-3
    if not first:
        original_gradient = (0.5 - y).mean()
        wrong_raw = -inverse_scale * original_gradient * X.shape[1]
        wrong_gradient = expit(wrong_raw) - y.mean()
        assert abs(float(gradient[0]) - wrong_gradient) > 1e-3
    assert np.isfinite(np.asarray(w)).all()


@pytest.mark.parametrize(
    "width, intercept",
    [
        (1023, True),
        (1024, False),
        (1025, True),
        (2048, False),
        (4096, True),
    ],
)
def test_wide_solver_preserves_float64_gradient_tolerance(width, intercept):
    rng = np.random.default_rng(718)
    n = 273
    X = rng.normal(scale=0.08, size=(n, width)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    weights = np.geomspace(1e-2, 1e2, n).astype(np.float32)
    weights[::13] = 0
    denominator = float(weights.sum(dtype=np.float64))
    l2 = 0.3
    Xm, ym, sw = mx.array(X), mx.array(y), mx.array(weights)
    fun = _native_binary_objective(Xm, ym, sw, denominator, l2, intercept)
    state = minimize_binary(
        Xm,
        ym,
        sw,
        denominator,
        l2,
        intercept,
        LBFGSConfig(max_iter=200, tol=1e-4),
        fun,
    )
    assert state.converged, state
    assert state.grad_max_abs <= 1e-4
    theta = np.asarray(state.x, dtype=np.float64)[0]
    logits = X.astype(np.float64) @ theta[:width]
    if intercept:
        logits += theta[-1]
    residual = weights * (expit(logits) - y)
    gradient = residual @ X / denominator + l2 * theta[:width]
    if intercept:
        gradient = np.append(gradient, residual.sum() / denominator)
    assert np.max(np.abs(gradient)) <= 1.01e-4
    expected = np.sum(
        weights * np.logaddexp(0, (1 - 2 * y) * logits)
    ) / denominator + 0.5 * l2 * np.sum(theta[:width] ** 2)
    assert state.objective == pytest.approx(expected, abs=2e-7)


def test_wide_scaled_weighted_problem_keeps_true_gradient_tolerance():
    rng = np.random.default_rng(392)
    n, width = 259, 1025
    X = rng.normal(size=(n, width)).astype(np.float32)
    X *= np.geomspace(0.1, 10.0, width).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    weights = np.geomspace(1e-3, 1e3, n).astype(np.float32)
    weights[::17] = 0
    denominator = float(weights.sum(dtype=np.float64))
    Xm, ym, sw = mx.array(X), mx.array(y), mx.array(weights)
    fun = _native_binary_objective(Xm, ym, sw, denominator, 1.0, False)
    state = minimize_binary(
        Xm,
        ym,
        sw,
        denominator,
        1.0,
        False,
        LBFGSConfig(max_iter=500, tol=1e-4),
        fun,
    )
    assert state.converged, state
    w = np.asarray(state.x, dtype=np.float64)[0]
    gradient = (weights * (expit(X.astype(np.float64) @ w) - y)) @ X / denominator + w
    assert np.max(np.abs(gradient)) <= 1.01e-4
