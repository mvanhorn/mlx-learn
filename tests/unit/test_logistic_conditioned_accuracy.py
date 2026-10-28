"""Regression cases for early stopping on scale-skewed logistic designs."""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers.lbfgs import LBFGSConfig
from mlxlearn._solvers.native_logistic import _conservative_config_for_skewed_curvature
from mlxlearn.linear_model import LogisticRegression


@pytest.mark.parametrize(
    ("low", "high", "l2", "expected"),
    [(1.0, 1.0, 0.0, 1e-4), (1.0, 101.0, 0.0, 5e-5),
     (0.0, 1.0, 0.0, 5e-5), (0.0, 1.0, 1.0, 1e-4)],
)
def test_internal_tolerance_uses_initial_feature_curvature(low, high, l2, expected):
    with use_device("cpu"):
        diagonal = mx.concatenate([mx.full((256,), low), mx.full((256,), high)])
        parts = mx.stack([mx.zeros((1, 513)), mx.concatenate([diagonal, mx.ones((1,))])[None]])
        result = _conservative_config_for_skewed_curvature(parts, 512, 1.0, l2, LBFGSConfig(tol=1e-4))
    assert result.tol == pytest.approx(expected)
    assert result.max_iter == LBFGSConfig().max_iter


def _weighted_scale_skew_fixture():
    rng = np.random.default_rng(83)
    n, d = 20_000, 1_024
    X = rng.standard_normal((n, d), dtype=np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    X[:, 512:] = X[:, :512] + 0.01 * X[:, 512:]
    X *= np.logspace(-1, 1, d, dtype=np.float32)
    weights = rng.uniform(0.25, 3.0, n)
    return X, y, weights


def _wide_scale_skew_fixture():
    # Match the previously reproduced 4,096-feature objective gap and draw
    # labels before altering the features, as the comparison benchmark does.
    rng = np.random.default_rng(41)
    n, d = 20_000, 4_096
    X64 = rng.normal(size=(n, d))
    y = (X64[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    X64[:, d // 2:] = X64[:, :d // 2] + 0.01 * X64[:, d // 2:]
    X64 *= np.logspace(-1, 1, d)
    return np.ascontiguousarray(X64, dtype=np.float32), y, None


def _objective_and_gradient(model, X, y, weights):
    """Independently score original-coordinate parameters in float64 blocks."""
    n, d = X.shape
    w = np.asarray(model.coef_[0], dtype=np.float64)
    b = float(model.intercept_[0])
    denominator = float(n if weights is None else np.sum(weights, dtype=np.float64))
    gradient = np.zeros(d, dtype=np.float64)
    bias_gradient = 0.0
    loss = 0.0
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        block = np.asarray(X[start:stop], dtype=np.float64)
        z = block @ w + b
        labels = y[start:stop]
        sample_weight = np.ones(stop - start) if weights is None else weights[start:stop]
        residual = sample_weight * (expit(z) - labels)
        gradient += block.T @ residual
        bias_gradient += float(residual.sum())
        loss += float(np.dot(sample_weight, np.logaddexp(0.0, z) - labels * z))
    gradient = gradient / denominator + w / denominator
    objective = loss / denominator + 0.5 * float(w @ w) / denominator
    return objective, max(float(np.max(np.abs(gradient))), abs(bias_gradient / denominator))


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", [_weighted_scale_skew_fixture, _wide_scale_skew_fixture])
def test_scale_skew_fit_matches_sklearn_objective_at_default_tolerance(fixture):
    if not gpu_available():
        pytest.skip("requires MLX GPU")
    X, y, weights = fixture()
    params = {"C": 1.0, "tol": 1e-4, "max_iter": 200}
    with threadpool_limits(limits=1), tuning_context(force_backend="mlx"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(**params).fit(X, y, sample_weight=weights)
        assert model._execution_backend_ == "mlx"
        assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            reference = SkLogisticRegression(**params).fit(X, y, sample_weight=weights)
        assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        objective, gradient = _objective_and_gradient(model, X, y, weights)
        reference_objective, _ = _objective_and_gradient(reference, X, y, weights)
    assert gradient <= params["tol"]
    assert model.n_iter_[0] <= params["max_iter"]
    assert (objective - reference_objective) / abs(reference_objective) <= 1e-5


def test_effective_tolerance_remains_authoritative_for_polish_and_certificate(monkeypatch):
    """A public-tolerance certificate must not promote a stricter failed fit."""
    from contextlib import contextmanager

    from mlxlearn._solvers import logistic, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSState

    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)
    seen = []

    def fake_minimize(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **kwargs):
        seen.append(("optimizer", config.tol))
        return LBFGSState(mx.zeros((1, X.shape[1] + 1)), 0.5, 7e-5, 1, 1, False, "max_iter")

    def fake_polish(X, y, sw, sw_sum, center, config, fun, state):
        seen.append(("polish", config.tol))
        return state

    monkeypatch.setattr(native_logistic, "minimize_binary", fake_minimize)
    monkeypatch.setattr(logistic, "_polish_binary_intercept", fake_polish)
    monkeypatch.setattr(logistic, "_needs_float64_certificate", lambda *args: True)
    monkeypatch.setattr(logistic, "_original_float64_certificate", lambda *args: (0.5, 7e-5))
    n, d = 3, 512
    X = np.zeros((n, d), dtype=np.float32)
    y = np.array([0, 1, 0])
    with use_device("cpu"):
        diagonal = mx.concatenate([mx.ones((256,)), mx.full((256,), 1000.0)])
        initial = mx.stack([mx.zeros((1, d + 1)), mx.concatenate([diagonal, mx.ones((1,))])[None]])
        result = logistic.solve_logistic(
            X, y, n_classes=2,
            config=logistic.LogisticConfig(device="gpu", max_iter=1, tol=1e-4),
            X_device=mx.zeros((n, d)), X_center=mx.zeros((d,)), initial_parts=initial,
        )
    assert seen == [("optimizer", 5e-5), ("polish", 5e-5)]
    assert result.required_tol == pytest.approx(5e-5)
    assert not result.converged and result.stop_reason == "max_iter"
    assert result.grad_max_abs == pytest.approx(7e-5)


@pytest.mark.parametrize("skewed", [False, True])
def test_hessian_sketch_is_wired_only_when_explicitly_enabled(monkeypatch, skewed):
    """The estimator path forwards exactly the device history it builds."""
    from contextlib import contextmanager

    from mlxlearn._solvers import logistic, native_command, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSState
    from mlxlearn_device import hessian_sketch

    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)
    monkeypatch.setattr(native_command, "_metal_loop", object())
    monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
    calls = []
    expected_history = object()

    def fake_build(X, sw, sw_sum, l2, fit_intercept, h0, **kwargs):
        calls.append(("build", kwargs["sketch_size"], kwargs["history_size"]))
        assert X.shape == (3, 512) and h0.shape == (513,)
        return expected_history

    def fake_minimize(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **kwargs):
        calls.append(("minimize", kwargs.get("initial_history"), config.tol))
        return LBFGSState(mx.zeros((1, 513)), 0.5, 0.0, 1, 1, True, "gtol")

    monkeypatch.setattr(hessian_sketch, "build_sketch_history", fake_build)
    monkeypatch.setattr(native_logistic, "minimize_binary", fake_minimize)
    X = np.zeros((3, 512), dtype=np.float32)
    y = np.array([0, 1, 0])
    with use_device("cpu"):
        if skewed:
            diagonal = mx.concatenate([mx.ones((256,)), mx.full((256,), 1000.0), mx.ones((1,))])
            parts = mx.stack([mx.ones((1, 513)), diagonal[None, :]])
        else:
            parts = mx.ones((2, 1, 513), dtype=mx.float32)
        common = {"X_device": mx.zeros((3, 512)), "X_center": mx.zeros((512,)), "initial_parts": parts}
        result = logistic.solve_logistic(
            X, y, n_classes=2,
            config=logistic.LogisticConfig(device="gpu", sketch_size=4), **common,
        )
    assert calls == [("build", 4, 10), ("minimize", expected_history, 5e-5)]
    assert result.converged and result.required_tol == pytest.approx(5e-5)


def test_zero_budget_sketch_does_not_build_history_or_tighten(monkeypatch):
    """A zero iteration budget only checks the initial point."""
    from contextlib import contextmanager

    from mlxlearn._solvers import logistic, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSState
    from mlxlearn_device import hessian_sketch

    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    seen = []
    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)

    def unexpected(*args, **kwargs):
        raise AssertionError("a zero budget must not create a sketch")

    def fake_minimize(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **kwargs):
        seen.append((config.tol, config.max_iter, kwargs.get("initial_history")))
        return LBFGSState(mx.zeros((1, 513)), 0.5, 2e-4, 0, 1, False, "max_iter")

    monkeypatch.setattr(hessian_sketch, "build_sketch_history", unexpected)
    monkeypatch.setattr(native_logistic, "minimize_binary", fake_minimize)
    monkeypatch.setattr(logistic, "_needs_float64_certificate", lambda *args: False)
    X = np.zeros((3, 512), dtype=np.float32)
    y = np.array([0, 1, 0])
    with use_device("cpu"):
        diagonal = mx.concatenate([mx.ones((256,)), mx.full((256,), 1000.0), mx.ones((1,))])
        parts = mx.stack([mx.ones((1, 513)), diagonal[None, :]])
        result = logistic.solve_logistic(
            X, y, n_classes=2,
            config=logistic.LogisticConfig(device="gpu", max_iter=0, tol=1e-4, sketch_size=4),
            X_device=mx.zeros((3, 512)), X_center=mx.zeros((512,)), initial_parts=parts,
        )
    assert seen == [(1e-4, 0, None)]
    assert result.n_iter == 0 and not result.converged
    assert result.required_tol == pytest.approx(1e-4)


@pytest.mark.gpu
def test_sketch_does_not_regress_wide_objective_against_unsketched_fit():
    """A different tolerance contour must not hide a worse wide objective."""
    if not gpu_available():
        pytest.skip("requires MLX GPU")
    rng = np.random.default_rng(41)
    n, d = 20_000, 4_096
    X = np.empty((n, d), dtype=np.float32)
    signal = np.empty(n, dtype=np.float64)
    for first in range(0, n, 256):
        last = min(first + 256, n)
        block = rng.normal(size=(last - first, d))
        signal[first:last] = block[:, 0]
        X[first:last] = block
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    params = {"C": 1.0, "tol": 1e-4, "max_iter": 200}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with tuning_context(force_backend="mlx", lbfgs_sketch_size=0):
            reference = LogisticRegression(**params).fit(X, y)
        with tuning_context(force_backend="mlx", lbfgs_sketch_size=4):
            sketched = LogisticRegression(**params).fit(X, y)
        baseline_objective, _ = _objective_and_gradient(reference, X, y, None)
        sketched_objective, sketched_gradient = _objective_and_gradient(sketched, X, y, None)
    assert sketched_gradient <= params["tol"]
    assert (sketched_objective - baseline_objective) / abs(baseline_objective) <= 1e-5


@pytest.mark.parametrize(
    ("features", "skewed", "sketch_size", "expected_native_loop"),
    [(1024, True, 0, True), (2048, False, 0, True), (2048, True, 0, False),
     (4096, True, 0, False), (2048, True, 4, True)],
)
def test_only_unseeded_skewed_wide_inputs_select_mlx_loop(
    monkeypatch, features, skewed, sketch_size, expected_native_loop
):
    """Choose the GPU route before spending any optimizer iterations."""
    from contextlib import contextmanager

    from mlxlearn._solvers import logistic, native_command, native_logistic
    from mlxlearn._solvers.lbfgs import LBFGSState
    from mlxlearn_device import hessian_sketch

    @contextmanager
    def pretend_gpu(_device):
        with use_device("cpu"):
            yield "gpu"

    monkeypatch.setattr(logistic, "use_device", pretend_gpu)
    monkeypatch.setattr(logistic, "_use_native_solver", lambda *_args: True)
    monkeypatch.setattr(logistic, "_needs_float64_certificate", lambda *args: False)
    monkeypatch.setattr(native_command, "_metal_loop", object())
    monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
    monkeypatch.setattr(hessian_sketch, "build_sketch_history", lambda *args, **kwargs: object())
    calls = []

    def fake_minimize(X, y, sw, sw_sum, l2, fit_intercept, config, fun, **kwargs):
        calls.append((kwargs.get("use_native_loop", True), config.tol))
        return LBFGSState(mx.zeros((1, features + 1)), 0.5, 0.0, 1, 1, True, "gtol")

    monkeypatch.setattr(native_logistic, "minimize_binary", fake_minimize)
    X = np.zeros((3, features), dtype=np.float32)
    y = np.array([0, 1, 0])
    with use_device("cpu"):
        diagonal = mx.concatenate([
            mx.ones((features // 2,)),
            mx.full((features - features // 2,), 1000.0 if skewed else 1.0),
            mx.ones((1,)),
        ])
        parts = mx.stack([mx.ones((1, features + 1)), diagonal[None, :]])
        result = logistic.solve_logistic(
            X, y, n_classes=2,
            config=logistic.LogisticConfig(device="gpu", max_iter=1, tol=1e-4,
                                           sketch_size=sketch_size),
            X_device=mx.zeros((3, features)), X_center=mx.zeros((features,)),
            initial_parts=parts,
        )
    assert calls[0][0] is expected_native_loop
    assert calls[0][1] == pytest.approx(5e-5 if skewed or sketch_size else 1e-4)
    assert result.converged and result.n_iter == 1
