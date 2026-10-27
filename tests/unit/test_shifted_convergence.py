"""Stopping criteria must refer to the original, not centered, coordinates."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit, softmax

from mlxlearn._common.device import use_device
from mlxlearn._solvers import logistic
from mlxlearn._solvers.fused_lbfgs import _convergence_gradient_max, minimize
from mlxlearn._solvers.lbfgs import LBFGSConfig, LBFGSState
from mlxlearn._solvers.logistic import LogisticConfig, reference_objective, solve_logistic


@pytest.fixture(autouse=True)
def _cpu_stream():
    # These are optimizer and coordinate contracts, not performance tests.
    with use_device("cpu"):
        yield


@pytest.mark.parametrize("targets", [1, 3])
def test_gradient_norm_transforms_centered_binary_and_multiclass_gradients(targets):
    center = np.array([1000.0, -250.0, 0.5], dtype=np.float32)
    original = np.array([[0.01, -0.03, 0.04, -0.002]] * targets, dtype=np.float32)
    centered = original.copy()
    centered[:, :-1] -= centered[:, -1:] * center

    result = _convergence_gradient_max(mx.array(centered), mx.array(center))

    assert float(result) == pytest.approx(float(np.max(np.abs(original))), abs=2e-7)
    assert float(_convergence_gradient_max(mx.array(centered), None)) == pytest.approx(
        float(np.max(np.abs(centered))), abs=2e-7
    )


def _quadratic(theta):
    values = np.asarray(theta, dtype=np.float64)
    return float((values * values).sum() / 2), theta


def test_zero_budget_reports_original_coordinate_norm_without_false_convergence():
    theta = mx.array([[2e-5, 2e-5]], dtype=mx.float32)
    center = mx.array([1000.0], dtype=mx.float32)

    state = minimize(_quadratic, theta, LBFGSConfig(max_iter=0, tol=1e-4),
                     convergence_center=center)

    assert state.n_iter == 0
    assert not state.converged
    assert state.stop_reason == "max_iter"
    assert state.grad_max_abs == pytest.approx(0.02002, rel=1e-6)
    np.testing.assert_array_equal(np.asarray(state.x), np.asarray(theta))


def test_original_norm_continues_optimizing_using_centered_gradient():
    # Both centered components start below tol, but the original feature
    # derivative is about 0.02. A centered-coordinate step reaches the true
    # optimum immediately; using the transformed gradient as a direction does
    # not solve this quadratic in one step.
    theta = mx.array([[2e-5, 2e-5]], dtype=mx.float32)
    center = mx.array([1000.0], dtype=mx.float32)

    state = minimize(_quadratic, theta, LBFGSConfig(max_iter=4, tol=1e-4),
                     convergence_center=center)

    assert state.converged and state.stop_reason == "gtol"
    assert state.n_iter == 1
    assert state.grad_max_abs <= 1e-4
    np.testing.assert_array_equal(np.asarray(state.x), np.zeros((1, 2)))


def test_nonfinite_original_gradient_is_not_reported_as_converged():
    theta = mx.array([[0.0, 0.0]], dtype=mx.float32)
    center = mx.array([float("nan")], dtype=mx.float32)

    state = minimize(_quadratic, theta, LBFGSConfig(max_iter=0),
                     convergence_center=center)

    assert not state.converged
    assert state.stop_reason == "non-finite-objective"


def test_center_shape_must_match_features():
    with pytest.raises(ValueError, match="convergence_center"):
        minimize(_quadratic, mx.zeros((1, 3)), LBFGSConfig(max_iter=0),
                 convergence_center=mx.zeros((3,)))


@pytest.mark.parametrize("weighted", [False, True])
def test_stalled_shifted_fit_polishes_bias_and_certifies_original_gradient(monkeypatch, weighted):
    rng = np.random.default_rng(461)
    n, d = 1027, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X += np.float32(1000.0)
    weights = rng.uniform(0.25, 3.0, n) if weighted else None
    config = LogisticConfig(C=1.0, tol=1e-4, max_iter=300, device="cpu")

    original_polish = logistic._polish_binary_intercept
    monkeypatch.setattr(logistic, "_polish_binary_intercept", lambda *args: args[-1])
    before = solve_logistic(X, y, n_classes=2, sample_weight=weights, config=config)
    monkeypatch.setattr(logistic, "_polish_binary_intercept", original_polish)
    after = solve_logistic(X, y, n_classes=2, sample_weight=weights, config=config)

    assert not before.converged and before.stop_reason == "line-search-failure"
    assert after.converged and after.stop_reason == "gtol"
    assert after.n_iter == before.n_iter + 1 <= config.max_iter
    np.testing.assert_array_equal(after.coef, before.coef)
    raw = X.astype(np.float64) @ after.coef[0] + after.intercept[0]
    residual = expit(raw) - y
    denominator = float(n if weights is None else weights.sum())
    if weights is not None:
        residual *= weights
    gradient = residual @ X.astype(np.float64) / denominator + after.coef[0] / denominator
    gradient = np.append(gradient, residual.sum() / denominator)
    assert np.max(np.abs(gradient)) <= config.tol
    old_objective = reference_objective(
        before.coef, before.intercept, X, y, n_classes=2, C=1.0, sample_weight=weights
    )
    new_objective = reference_objective(
        after.coef, after.intercept, X, y, n_classes=2, C=1.0, sample_weight=weights
    )
    assert new_objective <= old_objective + 1e-12


def test_binary_polish_never_exceeds_an_exhausted_iteration_budget(monkeypatch):
    X = np.array([[1000.0], [1001.0], [999.0], [1002.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    calls = []
    original = logistic._bias_statistics

    def unexpected(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(logistic, "_bias_statistics", unexpected)
    result = solve_logistic(X, y, n_classes=2,
                            config=LogisticConfig(max_iter=0, device="cpu"))
    assert not result.converged and result.n_iter == 0
    assert not calls


@pytest.mark.parametrize("weighted", [False, True])
def test_shifted_multinomial_fit_certifies_original_gradient(monkeypatch, weighted):
    rng = np.random.default_rng(461)
    n, d = 1027, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    score = X[:, 0] + rng.normal(scale=0.5, size=n)
    y = np.digitize(score, [-0.5, 0.5])
    X += np.float32(1000.0)
    weights = rng.uniform(0.25, 3.0, n) if weighted else None
    config = LogisticConfig(C=1.0, tol=1e-4, max_iter=300, device="cpu")

    original_polish = logistic._polish_multinomial_intercepts
    monkeypatch.setattr(logistic, "_polish_multinomial_intercepts", lambda *args: args[-1])
    before = solve_logistic(X, y, n_classes=3, sample_weight=weights, config=config)
    monkeypatch.setattr(logistic, "_polish_multinomial_intercepts", original_polish)
    after = solve_logistic(X, y, n_classes=3, sample_weight=weights, config=config)

    assert after.converged and after.stop_reason == "gtol"
    assert after.n_iter <= config.max_iter
    if not weighted:
        assert not before.converged and before.stop_reason == "line-search-failure"
        assert after.n_iter == before.n_iter + 1
        np.testing.assert_array_equal(after.coef, before.coef)
    data = X.astype(np.float64)
    residual = softmax(data @ after.coef.T + after.intercept, axis=1)
    residual[np.arange(n), y] -= 1
    denominator = float(n if weights is None else weights.sum())
    if weights is not None:
        residual *= weights[:, None]
    feature_gradient = residual.T @ data / denominator + after.coef / denominator
    bias_gradient = residual.sum(axis=0) / denominator
    assert max(np.max(np.abs(feature_gradient)), np.max(np.abs(bias_gradient))) <= config.tol
    old_objective = reference_objective(
        before.coef, before.intercept, X, y, n_classes=3, C=1.0, sample_weight=weights
    )
    new_objective = reference_objective(
        after.coef, after.intercept, X, y, n_classes=3, C=1.0, sample_weight=weights
    )
    assert new_objective <= old_objective + 1e-12


def test_risk_trigger_skips_safe_center_but_checks_shifts_and_near_tolerance():
    X = np.zeros((3, 1), dtype=np.float32)
    coef = np.zeros((1, 1), dtype=np.float64)
    state = LBFGSState(mx.zeros((1, 2)), 0.5, 1e-5, 1, 2, True, "gtol")
    assert not logistic._needs_float64_certificate(state, np.array([0.1]), 1e-4, X, X, coef)
    assert logistic._needs_float64_certificate(state, np.array([1000.0]), 1e-4, X, X, coef)
    near = LBFGSState(state.x, state.objective, 9.95e-5, 1, 2, True, "gtol")
    assert logistic._needs_float64_certificate(near, np.array([0.1]), 1e-4, X, X, coef)
    assert logistic._needs_float64_certificate(state, np.array([0.1]), 0.0, X, X, coef)
    assert not logistic._needs_float64_certificate(state, None, 0.0, X, X, coef)


def test_sampled_float64_conversion_error_requests_certificate():
    represented = np.zeros((3, 1), dtype=np.float32)
    original = represented.astype(np.float64)
    original[-1, 0] = 1e-3
    coef = np.ones((1, 1), dtype=np.float64)
    state = LBFGSState(mx.zeros((1, 2)), 0.5, 1e-5, 1, 2, True, "gtol")
    assert logistic._needs_float64_certificate(
        state, np.array([0.0]), 1e-4, original, represented, coef
    )


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_float64_certificate_matches_independent_original_objective_and_gradient(n_classes, weighted):
    rng = np.random.default_rng(704)
    n, d = 103, 5
    X = rng.normal(size=(n, d)).astype(np.float64) + 1000.0
    y = rng.integers(0, n_classes, n)
    coef = rng.normal(scale=0.2, size=(1 if n_classes == 2 else n_classes, d))
    bias = -coef @ X.mean(axis=0)
    weights = rng.uniform(0.1, 3.0, n) if weighted else None
    if weights is not None:
        weights[::7] = 0.0
    denominator = float(n if weights is None else weights.sum())
    l2 = 1.0 / denominator
    objective, norm = logistic._original_float64_certificate(
        X, y, weights, denominator, l2, coef, bias, n_classes
    )
    raw = X @ coef.T + bias
    residual = (expit(raw[:, 0]) - y)[:, None] if n_classes == 2 else softmax(raw, axis=1)
    if n_classes != 2:
        residual[np.arange(n), y] -= 1
    if weights is not None:
        residual *= weights[:, None]
    gradient = residual.T @ X / denominator + l2 * coef
    bias_gradient = residual.sum(axis=0) / denominator
    assert norm == pytest.approx(max(np.max(np.abs(gradient)), np.max(np.abs(bias_gradient))), abs=1e-12)
    assert objective == pytest.approx(reference_objective(
        coef, bias, X, y, n_classes=n_classes, C=1.0, sample_weight=weights
    ), abs=1e-12)


def test_certificate_uses_original_float64_input_instead_of_cast_copy():
    rng = np.random.default_rng(461)
    n, d = 1027, 8
    represented = rng.normal(size=(n, d)).astype(np.float32)
    y = (represented[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    represented += np.float32(1000.0)
    original = represented.astype(np.float64)
    # This small discrepancy is lost in float32 at the shifted scale.
    original[:, 0] += np.linspace(-2e-5, 2e-5, n)
    config = LogisticConfig(C=1.0, tol=1e-4, max_iter=300, device="cpu")
    result = solve_logistic(represented, y, n_classes=2, config=config, certification_X=original)
    raw = original @ result.coef[0] + result.intercept[0]
    residual = expit(raw) - y
    grad = residual @ original / n + result.coef[0] / n
    exact = float(max(np.max(np.abs(grad)), abs(residual.mean())))
    assert result.grad_max_abs == pytest.approx(exact, abs=1e-10)
    assert result.converged == (exact <= config.tol)


def test_ordinary_fit_skips_expensive_float64_certificate(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("well-centered zero-gradient input should not run the host certificate")

    monkeypatch.setattr(logistic, "_original_float64_certificate", unexpected)
    X = np.zeros((8, 2), dtype=np.float32)
    result = solve_logistic(X, np.tile([0, 1], 4), n_classes=2,
                            config=LogisticConfig(tol=1e-4, max_iter=1, device="cpu"))
    assert result.converged


def test_wide_polish_cannot_bypass_the_native_progress_guard(monkeypatch):
    X = mx.zeros((2, 2048), dtype=mx.float32)
    state = LBFGSState(mx.zeros((1, 2049)), 0.5, 0.1, 2, 3, False, "line-search-failure")

    def unexpected(*args, **kwargs):
        pytest.fail("wide stalled fits cannot use bias-only convergence")

    monkeypatch.setattr(logistic, "_bias_statistics", unexpected)
    result = logistic._polish_binary_intercept(
        X, mx.zeros((2,)), None, 2.0, mx.zeros((2048,)),
        LogisticConfig(max_iter=10, device="cpu"), _quadratic, state,
    )
    assert result is state


def test_certification_input_shape_is_validated():
    X = np.zeros((8, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="certification_X"):
        solve_logistic(X, np.tile([0, 1], 4), n_classes=2,
                       config=LogisticConfig(device="cpu"),
                       certification_X=np.zeros((8, 3), dtype=np.float64))


@pytest.mark.parametrize("exact_norm,expected", [(0.2, False), (0.0, True)])
def test_zero_budget_is_certified_only_when_exact_gradient_meets_tolerance(
    monkeypatch, exact_norm, expected
):
    X = np.full((8, 2), 1000.0, dtype=np.float32)
    y = np.tile([0, 1], 4)
    monkeypatch.setattr(logistic, "_original_float64_certificate", lambda *args: (0.5, exact_norm))
    result = solve_logistic(X, y, n_classes=2,
                            config=LogisticConfig(tol=1e-4, max_iter=0, device="cpu"))
    assert result.n_iter == 0
    assert result.converged is expected
    assert result.grad_max_abs == exact_norm
    assert result.stop_reason == ("gtol" if expected else "max_iter")


def test_finite_host_certificate_cannot_hide_a_nonfinite_optimizer_failure(monkeypatch):
    X = np.full((8, 2), 1000.0, dtype=np.float32)
    y = np.tile([0, 1], 4)
    failure = LBFGSState(mx.zeros((1, 3)), float("nan"), float("nan"), 0, 1, False,
                         "non-finite-objective")
    monkeypatch.setattr(logistic, "minimize", lambda *args, **kwargs: failure)
    monkeypatch.setattr(logistic, "_original_float64_certificate", lambda *args: (0.5, 0.0))
    result = solve_logistic(X, y, n_classes=2,
                            config=LogisticConfig(tol=1e-4, max_iter=4, device="cpu"))
    assert not result.converged
    assert result.stop_reason == "non-finite-objective"


@pytest.mark.parametrize(
    "n_classes,offset,weighted", [(2, -1000.0, False), (3, 10000.0, True)]
)
def test_previously_false_convergence_stress_cases_are_certified(n_classes, offset, weighted):
    rng = np.random.default_rng(9)
    n, d = 259, 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    score = X[:, 0] + rng.normal(scale=0.5, size=n)
    y = (score > 0).astype(int) if n_classes == 2 else np.digitize(score, [-0.5, 0.5])
    X += np.float32(offset)
    weights = rng.uniform(0.1, 3.0, n) if weighted else None
    if weights is not None:
        weights[::7] = 0.0
    result = solve_logistic(
        X, y, n_classes=n_classes, sample_weight=weights,
        config=LogisticConfig(C=1.0, tol=1e-4, max_iter=200, device="cpu"),
    )

    data = X.astype(np.float64)
    raw = data @ result.coef.T + result.intercept
    residual = (expit(raw[:, 0]) - y)[:, None] if n_classes == 2 else softmax(raw, axis=1)
    if n_classes != 2:
        residual[np.arange(n), y] -= 1
    denominator = float(n if weights is None else weights.sum())
    if weights is not None:
        residual *= weights[:, None]
    gradient = residual.T @ data / denominator + result.coef / denominator
    bias_gradient = residual.sum(axis=0) / denominator
    actual = float(max(np.max(np.abs(gradient)), np.max(np.abs(bias_gradient))))
    assert result.grad_max_abs == pytest.approx(actual, abs=1e-10)
    if result.converged:
        assert result.stop_reason == "gtol" and actual <= 1e-4
    else:
        assert result.stop_reason in {"max_iter", "line-search-failure", "non-finite-objective"}


@pytest.mark.parametrize("accept_second", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_bias_polish_counts_rejected_and_accepted_evaluations(monkeypatch, accept_second, n_classes):
    n, d = 2, 1
    X = mx.zeros((n, d), dtype=mx.float32)
    center = mx.array([1000.0], dtype=mx.float32)
    targets = 1 if n_classes == 2 else n_classes
    state = LBFGSState(mx.zeros((targets, d + 1)), 0.5, 0.1, 1, 4, False,
                       "line-search-failure")
    calls = []

    def loss_difference(*args):
        calls.append(True)
        return mx.array([-1.0 if accept_second and len(calls) == 2 else 1.0])

    def direct(candidate):
        return 0.2, mx.zeros_like(candidate)

    config = LogisticConfig(tol=1e-4, max_iter=5, max_line_search=8, device="cpu")
    if n_classes == 2:
        monkeypatch.setattr(logistic, "_bias_statistics",
                            lambda *args: (mx.array([1.0]), mx.array([1.0])))
        monkeypatch.setattr(logistic, "_bias_loss_difference", loss_difference)
        result = logistic._polish_binary_intercept(
            X, mx.zeros((n,)), None, 2.0, center, config, direct, state
        )
    else:
        monkeypatch.setattr(logistic, "_multinomial_bias_statistics",
                            lambda *args: (mx.array([[1.0, -1.0, 0.0]]), mx.eye(3)))
        monkeypatch.setattr(logistic, "_multinomial_bias_loss_difference", loss_difference)
        result = logistic._polish_multinomial_intercepts(
            X, mx.zeros((n, 1), dtype=mx.int32), mx.zeros((n, 3)), None, 2.0,
            center, config, direct, state,
        )
    assert len(calls) == (2 if accept_second else 8)
    assert result.n_fev == state.n_fev + len(calls) + int(accept_second)
    assert result.n_iter == state.n_iter + int(accept_second)
    assert result.converged is accept_second
