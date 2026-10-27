"""Represented intercept steps retain their loss differences on Metal."""

import mlx.core as mx
import numpy as np
import pytest
from scipy.special import softmax

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import logistic


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [1e-8, 1e-6, 1e-5])
def test_multinomial_bias_loss_difference_retains_tiny_steps(monkeypatch, weighted, scale):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(1941)
    linear = rng.normal(size=(37, 3)).astype(np.float32)
    labels = (np.arange(len(linear)) % 3).astype(np.int32)
    old = np.zeros((1, 3), dtype=np.float32)
    new = np.array([[scale, -2 * scale, 3 * scale]], dtype=np.float32)
    weights = rng.uniform(.25, 3, len(linear)).astype(np.float32) if weighted else None
    delta = new.astype(np.float64) - old.astype(np.float64)
    probabilities = softmax(linear.astype(np.float64) + old.astype(np.float64), axis=1)
    changes = delta - delta[0, labels, None]
    expected = np.log1p(np.sum(probabilities * np.expm1(changes), axis=1))
    if weights is not None:
        expected *= weights.astype(np.float64)

    # Emulate an implementation where the general-purpose primitives lose
    # small arguments to rounding, as can occur on a different Metal compiler.
    monkeypatch.setattr(mx, "expm1", lambda value: mx.exp(value) - 1.0)
    monkeypatch.setattr(mx, "log1p", lambda value: mx.log(1.0 + value))
    with use_device("gpu"):
        observed = logistic._multinomial_bias_loss_difference(
            mx.array(linear), mx.array(old), mx.array(new), mx.array(labels[:, None]),
            None if weights is None else mx.array(weights),
        )
        mx.eval(observed)
        actual = np.asarray(observed, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-14)
    assert np.sign(actual.sum()) == np.sign(expected.sum())
