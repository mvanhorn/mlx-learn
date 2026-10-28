"""Tests for the standalone, NumPy-free MLX-array logistic interface."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.device import gpu_available
from mlxlearn_device import DeviceConvergenceWarning, fit_logistic


def _fixture(kind: str):
    n, d = 4_096, 512
    rng = np.random.default_rng(41)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    weights = None
    C = 1.0
    ratio = None
    if kind == "weighted_shifted":
        X += np.float32(1_000.0)
        weights = rng.uniform(0.25, 3.0, size=n).astype(np.float32)
    elif kind == "rank_deficient":
        ratio = np.tile(np.array([2.0, 0.5], dtype=np.float32), d // 4)
        X[:, d // 2 :] = X[:, : d // 2] * ratio
        C = 1e6
    return X, y, weights, C, ratio


def _independent_score(model, X, y, weights, C):
    Xd = X.astype(np.float64)
    w = np.asarray(model.coef_[0], dtype=np.float64)
    center = np.asarray(model.center_, dtype=np.float64)
    intercept = float(model.centered_intercept_)
    logits = (Xd - center) @ w + intercept
    sw = np.ones(len(y), dtype=np.float64) if weights is None else weights.astype(np.float64)
    denominator = sw.sum()
    residual = (expit(logits) - y) * sw
    gradient = Xd.T @ residual / denominator + w / (C * denominator)
    gradient_bias = residual.sum() / denominator
    objective = np.dot(sw, np.logaddexp(0.0, logits) - y * logits) / denominator
    objective += 0.5 * (w @ w) / (C * denominator)
    return objective, max(float(np.max(np.abs(gradient))), abs(gradient_bias))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["normal", "weighted_shifted", "rank_deficient"])
def test_device_fit_preserves_input_and_matches_independent_accuracy(kind):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    X, y, weights, C, ratio = _fixture(kind)
    with mx.stream(mx.gpu), threadpool_limits(limits=1):
        Xm, ym = mx.array(X), mx.array(y)
        swm = None if weights is None else mx.array(weights)
        mx.eval(Xm, ym) if swm is None else mx.eval(Xm, ym, swm)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeviceConvergenceWarning)
            model = fit_logistic(Xm, ym, sample_weight=swm, C=C, sketch_size=4)
        assert model.converged and model.sketch_pairs_ > 0
        assert model.n_iter <= 200
        assert np.array_equal(np.asarray(Xm), X)
        assert np.array_equal(np.asarray(ym), y)
        if swm is not None:
            assert np.array_equal(np.asarray(swm), weights)
        objective, gradient = _independent_score(model, X, y, weights, C)
        assert gradient <= 1e-4
        assert abs(model.gradient_max_abs - gradient) <= 3e-8
        # Compare in centered coordinates so stock sklearn does not stop at a
        # poor uncentered solution on the large-offset fixture.
        center = np.asarray(model.center_, dtype=np.float64)
        reference = SkLogisticRegression(C=C, tol=1e-9, max_iter=1_000).fit(
            X.astype(np.float64) - center, y, sample_weight=weights
        )
        ref_coef = reference.coef_[0]
        ref_logits = (X.astype(np.float64) - center) @ ref_coef + reference.intercept_[0]
        sw = np.ones(len(y)) if weights is None else weights.astype(np.float64)
        ref_obj = np.dot(sw, np.logaddexp(0.0, ref_logits) - y * ref_logits) / sw.sum()
        ref_obj += 0.5 * (ref_coef @ ref_coef) / (C * sw.sum())
        assert (objective - ref_obj) / max(abs(ref_obj), np.finfo(float).tiny) <= 1e-5
        if ratio is not None:
            w = np.asarray(model.coef_[0], dtype=np.float64)
            null_ratio = np.linalg.norm(w[256:] - ratio * w[:256]) / np.linalg.norm(w)
            assert null_ratio < 1e-5
        predictions = model.predict(Xm)
        assert isinstance(predictions, mx.array) and predictions.shape == (len(y),)


@pytest.mark.gpu
def test_device_sketch_failure_retries_only_with_remaining_iteration_budget(monkeypatch):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import _logistic

    X, y, _, _, _ = _fixture("normal")
    calls = []
    native_run = _logistic.metal_loop.run

    def fail_first_sketch(*args):
        calls.append((args[8], args[-2]))
        if args[-2] > 0:
            return None, 1, 2, 8, 1.0, 1.0
        return native_run(*args)

    monkeypatch.setattr(_logistic.metal_loop, "run", fail_first_sketch)
    with mx.stream(mx.gpu):
        model = fit_logistic(mx.array(X), mx.array(y), max_iter=200, sketch_size=4)
    assert model.converged
    assert model.used_unsketched_fallback_
    assert len(calls) == 2 and calls[0][0] == 200 and calls[1] == (199, 0)
    assert model.n_iter <= 200


@pytest.mark.gpu
def test_device_fit_never_imports_numpy_or_sklearn_in_fresh_process():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    source = textwrap.dedent(
        """
        import builtins, sys
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'numpy' or name.startswith('numpy.') or name == 'sklearn' or name.startswith('sklearn.'):
                raise AssertionError('unexpected import: ' + name)
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        import mlx.core as mx
        from mlxlearn_device import fit_logistic
        with mx.stream(mx.gpu):
            mx.random.seed(41)
            X = mx.random.normal((2_048, 512))
            y = (X[:, 0] > 0).astype(mx.float32)
            model = fit_logistic(X, y, sketch_size=4)
            mx.eval(model.coef_, model.center_, model.centered_intercept_)
            assert model.converged
        assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), check=True, timeout=120)


@pytest.mark.gpu
def test_device_fit_reports_iteration_exhaustion():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    X, y, _, _, _ = _fixture("normal")
    with mx.stream(mx.gpu), pytest.warns(DeviceConvergenceWarning):
        model = fit_logistic(mx.array(X), mx.array(y), max_iter=1, sketch_size=0)
    assert not model.converged
    assert model.n_iter == 1
    assert model.stop_reason == "max_iter"


def test_device_fit_rejects_non_device_and_unsupported_inputs():
    with pytest.raises(TypeError, match="MLX array"):
        fit_logistic(np.zeros((1, 512), dtype=np.float32), mx.array([0]))
    with pytest.raises(ValueError, match="512"):
        fit_logistic(mx.zeros((2, 2), dtype=mx.float32), mx.array([0, 1]))
    with pytest.raises(ValueError, match="sketch_size"):
        fit_logistic(mx.zeros((2, 512), dtype=mx.float32), mx.array([0, 1]), sketch_size=11)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("status", "direct_norm", "recent_delta", "expected"),
    [
        # Passing the user's tolerance is insufficient when the solve was
        # required to meet the stricter effective tolerance.
        (0, 8e-5, 1e-6, False),
        (1, 8e-5, 1e-6, False),
        # A wide loss change can satisfy the user tolerance while failing the
        # stricter effective tolerance; cached success cannot override it.
        (0, 1e-6, 1e-5, False),
        (1, 1e-6, 1e-5, False),
        # The final accepted step can satisfy the strict condition at the
        # iteration boundary before a following GPU direction sets status=1.
        (0, 1e-6, 1e-6, False),
        (1, 1e-6, 1e-6, False),
        (0, 1e-6, 1e-8, True),
        (1, 1e-6, 1e-8, True),
    ],
)
def test_device_direct_check_uses_effective_tolerance(
    monkeypatch, status, direct_norm, recent_delta, expected
):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import _logistic

    def fake_run(*args):
        return None, 1, status, 8, recent_delta, 0.0

    def fake_score(*args):
        return mx.array(0.2), mx.array(direct_norm), mx.zeros(1), mx.zeros(1), mx.array(0.0)

    monkeypatch.setattr(_logistic.metal_loop, "run", fake_run)
    monkeypatch.setattr(_logistic, "score_original", fake_score)
    with mx.stream(mx.gpu), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeviceConvergenceWarning)
        X = mx.reshape(mx.arange(32 * 2_048, dtype=mx.float32), (32, 2_048)) / 65_536
        y = mx.arange(32, dtype=mx.int32) % 2
        model = fit_logistic(X, y, max_iter=1, sketch_size=0)
    assert model.converged is expected
    assert bool(caught) is not expected


def test_device_rejects_labels_or_weights_that_require_narrowing():
    X = mx.zeros((2, 512), dtype=mx.float32)
    with pytest.raises(TypeError, match="y must have a supported"):
        fit_logistic(X, mx.array([0.0 + 0j, 1.0 + 0j], dtype=mx.complex64))
    with pytest.raises(TypeError, match="sample_weight must have"):
        fit_logistic(X, mx.array([0, 1]), sample_weight=mx.array([1 + 0j, 1 + 0j]))


@pytest.mark.gpu
def test_device_rejects_unrepresentable_regularization_and_single_weighted_class():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    with mx.stream(mx.gpu):
        X, y = mx.zeros((2, 512), dtype=mx.float32), mx.array([0, 1])
        with pytest.raises(ValueError, match="normal float32"):
            fit_logistic(X, y, C=1e300)
        with pytest.raises(ValueError, match="positive sample weights"):
            fit_logistic(X, y, sample_weight=mx.array([1.0, 0.0]))


@pytest.mark.gpu
def test_compensated_tail_guard_covers_large_features_after_sigmoid_clamp():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device._compensated import score_original

    with mx.stream(mx.gpu):
        features = mx.concatenate([mx.array([[1e19], [-1e19]]), mx.zeros((2, 511))], axis=1)
        theta = mx.concatenate([mx.array([8.1e-18]), mx.zeros(512)])
        _, norm, _, _, tail = score_original(
            features, mx.zeros(512), theta, mx.array([1.0, 0.0]), mx.ones(2), 2.0, 0.0
        )
        mx.eval(norm, tail)
        # exp(-81) * 1e19 ~= 6.64e-17: returning zero without tracking this
        # error would be a false success at a tolerance of 1e-17.
        assert float(tail) >= 6.6e-17
        assert float(norm) + float(tail) > 1e-17


@pytest.mark.gpu
def test_polish_rejects_original_objective_increase(monkeypatch):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import _logistic

    calls = []

    def fake_score(*args):
        calls.append(True)
        if len(calls) == 1:
            # Positive bias gradient yields a negative Newton step.
            grad = mx.concatenate([mx.zeros(512), mx.array([0.25])])
            return mx.array(1.0), mx.array(1e-3), grad, mx.zeros(513), mx.array(0.0)
        return mx.array(2.0), mx.array(1e-8), mx.zeros(513), mx.zeros(513), mx.array(0.0)

    monkeypatch.setattr(_logistic, "score_original", fake_score)
    with mx.stream(mx.gpu):
        X = mx.zeros((4, 512))
        result = _logistic._polish_intercept(
            X, X, mx.zeros(512), mx.zeros(513), mx.array([0, 0, 0, 1]), mx.ones(4),
            4.0, 0.25, 0, 4, 2.5e-5, 1.0, 1e-3, 0.0,
        )
    assert len(calls) >= 2
    assert result[1] == 0 and not result[-1]


@pytest.mark.gpu
def test_failed_sketch_and_retry_keep_better_objective_and_budget(monkeypatch):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import _logistic

    calls = []

    def fake_run(*args):
        calls.append(args[8])
        return None, 1, 2, 8, 1.0, 1.0

    def fake_score(*args):
        obj = 0.3 if len(calls) == 1 else 0.4
        return mx.array(obj), mx.array(0.2), mx.zeros(513), mx.zeros(513), mx.array(0.0)

    monkeypatch.setattr(_logistic.metal_loop, "run", fake_run)
    monkeypatch.setattr(_logistic, "score_original", fake_score)
    with mx.stream(mx.gpu), pytest.warns(DeviceConvergenceWarning):
        X, y, _, _, _ = _fixture("normal")
        model = fit_logistic(mx.array(X), mx.array(y), max_iter=10, sketch_size=4)
    assert calls == [10, 9]
    assert model.used_unsketched_fallback_ and not model.converged and model.n_iter == 2
    assert abs(model.objective - 0.3) < 1e-7


def test_device_import_without_native_extension_is_lazy():
    source = textwrap.dedent(
        """
        import pathlib
        original = pathlib.Path.is_file
        def is_file(self):
            return False if self.name == '_metal_loop.abi3.so' else original(self)
        pathlib.Path.is_file = is_file
        import mlx.core as mx
        import mlxlearn_device
        X = mx.zeros((2, 512), mx.float32)
        try:
            mlxlearn_device.fit_logistic(X, mx.array([0, 1]))
        except RuntimeError as exc:
            assert 'Metal extension is unavailable' in str(exc)
        else:
            raise AssertionError('missing native extension was not reported')
        """
    )
    subprocess.run([sys.executable, "-c", source], env=os.environ.copy(), check=True, timeout=30)


@pytest.mark.gpu
def test_device_prediction_resolves_shifted_near_boundary_logits():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import DeviceLogisticModel

    X = np.full((2, 512), 1_000.0, dtype=np.float32)
    X[:, :2] = 1_000.0625
    X[0, 2] = 1_000.0625
    X[1, 2] = 999.9375
    coef = np.zeros(512, dtype=np.float32)
    coef[0], coef[1], coef[2] = 1.0, -1.0 + 2**-20, 1e-12
    centered_bias = -(2**-24)
    expected = ((X.astype(np.float64) - 1_000.0) @ coef.astype(np.float64) + centered_bias) >= 0
    assert expected.tolist() == [True, False]
    with mx.stream(mx.gpu):
        model = DeviceLogisticModel(
            coef_=mx.array(coef.reshape(1, -1)), center_=mx.full((512,), 1_000.0),
            centered_intercept_=mx.array(centered_bias), n_iter=0, converged=True,
            stop_reason="gtol", objective=0.0, gradient_max_abs=0.0, sketch_count=0,
            used_fallback=False, discarded_tail_bound=0.0,
        )
        predicted = np.asarray(model.predict(mx.array(X)))
    assert np.array_equal(predicted, expected.astype(np.int32))


@pytest.mark.gpu
def test_compensated_score_skips_zero_weight_rows_before_dot_product():
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device._compensated import score_original

    with mx.stream(mx.gpu):
        first = mx.concatenate([mx.array([1e38]), mx.zeros(511)])
        X = mx.stack([first, mx.zeros(512)])
        theta = mx.concatenate([mx.array([10.0]), mx.zeros(512)])
        objective, norm, gh, gl, tail = score_original(
            X, mx.zeros(512), theta, mx.array([0.0, 0.0]),
            mx.array([0.0, 1.0]), 1.0, 0.0,
        )
        mx.eval(objective, norm, gh, gl, tail)
    assert np.isfinite(float(objective))
    assert abs(float(objective) - np.log(2.0)) < 1e-6
    assert abs(float(norm) - 0.5) < 1e-6
    assert float(tail) == 0


@pytest.mark.parametrize("tol", [1e-40, 1e300])
def test_device_fit_rejects_unrepresentable_effective_tolerance_without_gpu(tol):
    with mx.stream(mx.cpu):
        X = mx.zeros((2, 512), dtype=mx.float32)
        y = mx.array([0, 1])
        with pytest.raises(ValueError, match="effective tolerance outside normal float32"):
            fit_logistic(X, y, tol=tol)

@pytest.mark.gpu
@pytest.mark.parametrize(
    ("recent", "polished_objective", "expected_calls", "expected_converged"),
    [(1e-5, 0.2, 0, False), (1e-8, 0.2, 1, True), (1e-8, 0.001, 1, False)],
)
def test_weighted_polish_cannot_bypass_progress(monkeypatch, recent, polished_objective, expected_calls, expected_converged):
    if not gpu_available():
        pytest.skip("requires a working MLX GPU and native Metal extension")
    from mlxlearn_device import _logistic

    calls = []
    monkeypatch.setattr(_logistic.metal_loop, "run", lambda *args: (None, 3, 2, 24, recent, recent))
    monkeypatch.setattr(_logistic, "score_original", lambda *args: (
        mx.array(0.2), mx.array(1e-3), mx.zeros(1), mx.zeros(1), mx.array(0.0)
    ))

    def polish(X, centered, center, output, y, sw, sw_sum, l2, iterations,
               budget, effective_tol, objective, norm, tail):
        calls.append(iterations)
        return output, iterations + 1, polished_objective, 1e-6, 0.0, True

    monkeypatch.setattr(_logistic, "_polish_intercept", polish)
    with mx.stream(mx.gpu), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeviceConvergenceWarning)
        X = mx.reshape(mx.arange(32 * 512, dtype=mx.float32), (32, 512)) / 16_384
        y = mx.arange(32, dtype=mx.int32) % 2
        weights = mx.where(y == 0, mx.array(0.5), mx.array(2.0))
        model = fit_logistic(X, y, sample_weight=weights, max_iter=8, sketch_size=0)
    assert len(calls) == expected_calls
    assert model.converged is expected_converged
    assert bool(caught) is not expected_converged
