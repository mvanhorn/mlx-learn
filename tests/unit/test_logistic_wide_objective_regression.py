"""Regression for the wide, aggregate-gradient logistic stopping failure."""

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
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import native_command, native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn.linear_model import LogisticRegression


def _wide_binary_fixture():
    """Reproduce the previously failing dense shape without a large float64 copy."""
    rows, features = 20_000, 4_096
    rng = np.random.default_rng(41)
    X = np.empty((rows, features), dtype=np.float32)
    signal = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        block = rng.normal(size=(stop - start, features))
        signal[start:stop] = block[:, 0]
        X[start:stop] = block
    y = (signal + rng.normal(scale=0.5, size=rows) > 0).astype(np.int32)
    return X, y


def _independent_score(model, X, y, C):
    """Score the original-coordinate objective and gradient in float64 blocks."""
    rows, features = X.shape
    coef = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    gradient = np.zeros(features, dtype=np.float64)
    bias_gradient = 0.0
    loss = 0.0
    predictions = np.empty(rows, dtype=bool)
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        block = np.asarray(X[start:stop], dtype=np.float64)
        logits = block @ coef + intercept
        labels = y[start:stop]
        residual = expit(logits) - labels
        gradient += block.T @ residual
        bias_gradient += float(residual.sum())
        loss += float(np.sum(np.logaddexp(0.0, logits) - labels * logits))
        predictions[start:stop] = logits >= 0.0
    gradient = gradient / rows + coef / (C * rows)  # Intercept is unpenalized.
    max_gradient = max(float(np.max(np.abs(gradient))), abs(bias_gradient / rows))
    objective = (loss + 0.5 * float(coef @ coef) / C) / rows
    return objective, max_gradient, predictions


@pytest.mark.gpu
@pytest.mark.parametrize("C", [1.0, 10.0])
def test_wide_default_tolerance_matches_sklearn_objective_without_overpenalizing_improvement(C):
    """The infinity-gradient test alone used to stop with a worse objective."""
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    X, y = _wide_binary_fixture()
    params = {"C": C, "fit_intercept": True, "tol": 1e-4, "max_iter": 200}

    with threadpool_limits(limits=1), tuning_context(force_backend="mlx"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            native = LogisticRegression(**params).fit(X, y)
        assert native._execution_backend_ == "mlx"
        assert not [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        with warnings.catch_warnings(record=True) as reference_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            reference = SkLogisticRegression(**params).fit(X, y)
        assert not [
            item for item in reference_warnings if issubclass(item.category, ConvergenceWarning)
        ]
        native_obj, native_grad, native_predictions = _independent_score(native, X, y, C)
        reference_obj, _, reference_predictions = _independent_score(reference, X, y, C)

    assert native_grad <= params["tol"]
    # This is deliberately one-sided: a lower objective is an improvement even
    # when it differs from sklearn's early-stopped objective by more than 1e-5.
    relative_increase = (native_obj - reference_obj) / max(abs(reference_obj), np.finfo(float).tiny)
    assert relative_increase <= 1e-5, (native_obj, reference_obj, relative_increase)
    assert np.mean(native_predictions == reference_predictions) >= 0.995


@pytest.mark.parametrize(
    ("last_difference", "previous_difference", "n_iter", "gradient", "should_converge"),
    [
        (1e-7, 1e-7, 2, 5e-5, True),
        (8e-7, 0.0, 2, 5e-5, True),
        (1e-7, float("nan"), 1, 5e-5, True),
        # These would pass the previous 1.0 × tolerance progress rule.
        (1e-6, 1e-6, 2, 5e-5, False),
        (3e-5, 0.0, 2, 5e-5, False),
        (1e-6, 1e-3, 2, 5e-5, False),
        (1e-3, 0.0, 2, 5e-5, False),
        (float("nan"), 0.0, 2, 5e-5, False),
        # No accepted step has a meaningful previous decrease, and an exact
        # zero gradient is already stationary.
        (float("nan"), 0.0, 0, 5e-5, True),
        (float("nan"), 0.0, 2, 0.0, True),
    ],
)
def test_native_direct_certification_checks_wide_objective_progress(
    monkeypatch, last_difference, previous_difference, n_iter, gradient, should_converge
):
    """A cached native success cannot bypass the direct progress check."""
    rows, features = 5, 2_048
    with use_device("cpu"):
        X = mx.ones((rows, features), dtype=mx.float32)
        y = mx.array([0, 1, 0, 1, 0], dtype=mx.float32)
        parts = mx.zeros((2, 1, features + 1), dtype=mx.float32)
        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "unused test source")

        def fake_run(*args):
            return np.zeros(features + 1, dtype=np.float32).tobytes(), n_iter, 1, 16, last_difference, previous_difference

        monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=fake_run))
        calls = []

        def direct(theta):
            calls.append(np.asarray(theta).copy())
            return 0.2, mx.full(theta.shape, gradient, dtype=mx.float32)

        state = native_command.run_native_loop(
            X, y, None, float(rows), 0.1, True, LBFGSConfig(tol=1e-4), direct,
            initial_parts=parts,
        )
    assert len(calls) == 1
    assert (state is not None) is should_converge
    if state is not None:
        assert state.converged and state.grad_max_abs <= 1e-4


def test_native_direct_certification_rechecks_shifted_coordinate_gradient(monkeypatch):
    """A converged centered gradient can still violate the original-coordinate tolerance."""
    rows, features = 5, 2_048
    with use_device("cpu"):
        X = mx.ones((rows, features), dtype=mx.float32)
        y = mx.array([0, 1, 0, 1, 0], dtype=mx.float32)
        parts = mx.zeros((2, 1, features + 1), dtype=mx.float32)
        center = mx.full((features,), 1_000.0, dtype=mx.float32)
        monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
        monkeypatch.setattr(native_command, "_source", lambda *args: "unused test source")
        monkeypatch.setattr(
            native_command, "_metal_loop",
            SimpleNamespace(run=lambda *args: (
                np.zeros(features + 1, dtype=np.float32).tobytes(), 2, 1, 16, 0.0, 0.0
            )),
        )

        def direct(theta):
            grad = mx.concatenate([mx.zeros((features,), dtype=mx.float32), mx.array([1e-6])])
            return 0.2, grad.reshape(theta.shape)

        state = native_command.run_native_loop(
            X, y, None, float(rows), 0.1, True, LBFGSConfig(tol=1e-4), direct,
            initial_parts=parts, convergence_center=center,
        )
    assert state is None


@pytest.mark.parametrize(
    ("difference", "direct_objective", "cached_gradient", "direct_gradient", "converged"),
    [
        (1e-7, 0.2, 5e-5, 5e-5, True),
        (1e-6, 0.2, 5e-5, 5e-5, False),
        (1e-3, 0.2, 5e-5, 5e-5, False),
        # The cached objective would accept this difference, but the direct
        # objective has a smaller scale and must make the final decision.
        (1.5e-5, 0.01, 5e-5, 5e-5, False),
        (1e-6, 0.2, 5e-5, 2e-4, False),
        # A cached zero must not bypass progress for a nonzero direct gradient.
        (1e-3, 0.2, 0.0, 5e-5, False),
    ],
)
def test_wide_fallback_checks_direct_progress_at_iteration_budget(
    monkeypatch, difference, direct_objective, cached_gradient, direct_gradient, converged
):
    """At max_iter a cached success cannot override direct certification."""
    rows, features = 3, 2_048
    with use_device("cpu"):
        X = mx.zeros((rows, features), dtype=mx.float32)
        y = mx.array([0, 1, 0], dtype=mx.float32)

        def initial(_X, _y, _sw, p, _reg, _denom):
            return mx.full((p,), cached_gradient, dtype=mx.float32), mx.ones((p,))

        def step(*args):
            w, g, oldw, oldg, hs, hy, hc, _meta, raw = args[12:]
            meta = np.zeros(12, dtype=np.float32)
            meta[0], meta[10], meta[11] = 1.0, difference, 0.2
            return w, g, oldw, oldg, hs, hy, hc, mx.array(meta), raw

        def direct(theta):
            return direct_objective, mx.full(theta.shape, direct_gradient, dtype=mx.float32)

        monkeypatch.setattr(native_logistic, "initial", initial)
        monkeypatch.setattr(native_logistic, "step", step)
        state = native_logistic.minimize_binary(
            X, y, None, float(rows), 0.1, True, LBFGSConfig(max_iter=1, tol=1e-4), direct
        )
    assert state.n_iter == 1
    assert state.converged is converged
    assert state.stop_reason == ("gtol" if converged else "max_iter")


@pytest.mark.parametrize(("gradient", "converged"), [(0.0, True), (5e-5, True), (2e-4, False)])
def test_wide_zero_budget_checks_direct_gradient_without_requiring_progress(gradient, converged):
    """A zero iteration budget must not schedule an artificial progress step."""
    rows, features = 3, 2_048
    with use_device("cpu"):
        X = mx.zeros((rows, features), dtype=mx.float32)
        y = mx.array([0, 1, 0], dtype=mx.float32)

        def direct(theta):
            return 0.2, mx.full(theta.shape, gradient, dtype=mx.float32)

        state = native_logistic.minimize_binary(
            X, y, None, float(rows), 0.1, True, LBFGSConfig(max_iter=0, tol=1e-4), direct
        )
    assert state.n_iter == 0
    assert state.converged is converged
    assert state.stop_reason == ("gtol" if converged else "max_iter")
