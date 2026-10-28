"""Precision, tail and workspace checks for the coalesced device gradient."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import expit

from mlxlearn._common.device import gpu_available
from mlxlearn_device import _compensated as core


@pytest.mark.parametrize(
    ("n", "d", "shift"),
    [
        (1, 512, 0.0),
        (31, 513, 1000.0),
        (32, 1024, 0.0),
        (33, 1025, -1000.0),
        (255, 2048, 0.0),
        (256, 2049, 1000.0),
        (257, 4095, 0.0),
        (257, 4096, -1000.0),
    ],
)
@pytest.mark.gpu
def test_coalesced_backward_matches_float64_and_old_kernel_on_tails(n, d, shift):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(n + d)
    X = rng.normal(size=(n, d)).astype(np.float32)
    if shift:
        X += np.float32(shift)
    center = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
    theta = np.concatenate((rng.normal(scale=0.01, size=d).astype(np.float32), [0.125])).astype(
        np.float32
    )
    y = rng.integers(0, 2, size=n).astype(np.float32)
    weights = rng.uniform(0.1, 3.0, size=n).astype(np.float32)
    weights[::3] = 0
    weights[-1] = 1
    total = float(weights.astype(np.float64).sum())
    l2 = 1.0 / total
    input64 = X.astype(np.float64)
    coefficients = theta[:d].astype(np.float64)
    logits = (input64 - center.astype(np.float64)) @ coefficients + float(theta[d])
    residual = (expit(logits) - y) * weights / total
    expected_gradient = np.concatenate((input64.T @ residual + l2 * coefficients, [residual.sum()]))
    expected_objective = np.sum(
        (np.logaddexp(0, logits) - y * logits) * weights
    ) / total + 0.5 * l2 * (coefficients @ coefficients)

    with mx.stream(mx.gpu):
        Xm, cm, tm, ym, swm = (mx.array(v) for v in (X, center, theta, y, weights))
        denominator = mx.array(core._split(total), dtype=mx.float32)
        penalty = mx.array(core._split(l2), dtype=mx.float32)
        hi, lo, _, _ = core._forward(
            inputs=[Xm, cm, tm, ym, swm, denominator],
            template=[("N", n), ("D", d)],
            grid=(((n + 3) // 4) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n,)] * 4,
            output_dtypes=[mx.float32] * 4,
        )
        old_hi, old_lo = core._backward(
            inputs=[Xm, tm, hi, lo, denominator, penalty],
            template=[("N", n), ("D", d)],
            grid=((d + 1) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(d + 1,)] * 2,
            output_dtypes=[mx.float32] * 2,
        )
        new_hi, new_lo = core._coalesced_backward(Xm, tm, hi, lo, denominator, penalty)
        objective, norm, actual_hi, actual_lo, tail = core.score_original(
            Xm, cm, tm, ym, swm, total, l2
        )
        mx.eval(old_hi, old_lo, new_hi, new_lo, objective, norm, actual_hi, actual_lo, tail)
    old = np.asarray(old_hi, dtype=np.float64) + np.asarray(old_lo, dtype=np.float64)
    new = np.asarray(new_hi, dtype=np.float64) + np.asarray(new_lo, dtype=np.float64)
    actual = np.asarray(actual_hi, dtype=np.float64) + np.asarray(actual_lo, dtype=np.float64)
    np.testing.assert_allclose(new, expected_gradient, rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-8)
    np.testing.assert_array_equal(new, actual)
    assert float(objective) == pytest.approx(expected_objective, rel=2e-6, abs=2e-7)
    assert float(norm) == pytest.approx(
        float(np.max(np.abs(expected_gradient))), rel=2e-6, abs=2e-7
    )
    assert np.isfinite(float(tail))


def test_coalesced_workspace_grid_and_memory_limits():
    assert core._coalesced_workspace_supported(256 * 2048, 4095)
    assert not core._coalesced_workspace_supported(256 * 2048 + 1, 4095)
    assert core._coalesced_workspace_supported(256 * 65535, 0)
    assert not core._coalesced_workspace_supported(256 * 65535 + 1, 0)
    assert not core._coalesced_workspace_supported(0, 512)


@pytest.mark.gpu
def test_score_falls_back_to_original_kernel_when_workspace_is_disabled(monkeypatch):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    with mx.stream(mx.gpu):
        X = mx.array([[1.0, -2.0], [2.0, 0.5]], dtype=mx.float32)
        center = mx.array([0.25, -0.5], dtype=mx.float32)
        theta = mx.array([0.4, -0.3, 0.1], dtype=mx.float32)
        y = mx.array([0.0, 1.0], dtype=mx.float32)
        weights = mx.array([1.0, 2.0], dtype=mx.float32)
        expected = core.score_original(X, center, theta, y, weights, 3.0, 0.1)
        mx.eval(*expected)
        monkeypatch.setattr(core, "_coalesced_workspace_supported", lambda *_: False)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("workspace-limited path must use the original kernel")

        monkeypatch.setattr(core, "_coalesced_backward", fail_if_called)
        actual = core.score_original(X, center, theta, y, weights, 3.0, 0.1)
        mx.eval(*actual)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-6, atol=1e-7)
