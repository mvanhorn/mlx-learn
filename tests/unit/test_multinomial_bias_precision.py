"""Independent reference for represented multinomial intercept updates."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers.logistic import _multinomial_bias_loss_difference


def _independent_parts(linear, old_bias, new_bias, codes, weights):
    """Float64 weighted loss changes for float32-represented inputs.

    Evaluate the loss *difference* directly; subtracting two total losses would
    destroy the very small improvements this test is designed to check.
    """
    raw = (linear + old_bias).astype(np.float32).astype(np.float64)
    raw -= raw.max(axis=1, keepdims=True)
    exponential = np.exp(raw)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    delta = (new_bias.astype(np.float64) - old_bias.astype(np.float64))[0]
    relative = delta[None, :] - delta[codes, None]
    loss_difference = np.log1p((probabilities * np.expm1(relative)).sum(axis=1))
    if weights is not None:
        loss_difference *= weights.astype(np.float64)
    # The device helper returns at most 1024 block sums.
    if len(linear) > 1024:
        chunk = math.ceil(len(linear) / 1024)
        loss_difference = np.pad(loss_difference, (0, chunk * 1024 - len(linear)))
        loss_difference = loss_difference.reshape(1024, chunk).sum(axis=1)
    return loss_difference


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("rows", [33, 1031])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scale", [1e-9, 1e-7, 1e-5, 0.02, 0.45])
def test_multinomial_bias_step_retains_weighted_loss_difference(device, rows, weighted, scale):
    if device == "gpu" and not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(715)
    linear = rng.normal(scale=2.0, size=(rows, 3)).astype(np.float32)
    codes = rng.integers(0, 3, rows, dtype=np.int32)
    old_bias = np.array([[0.0, 0.125, -0.25]], dtype=np.float32)
    change = np.array([[scale, -0.75 * scale, 0.3 * scale]], dtype=np.float32)
    new_bias = (old_bias + change).astype(np.float32)
    weights = rng.uniform(0.25, 3.0, rows).astype(np.float32) if weighted else None
    if weights is not None:
        weights[::9] = 0
    expected = _independent_parts(linear, old_bias, new_bias, codes, weights)

    with use_device(device):
        parts = _multinomial_bias_loss_difference(
            mx.array(linear), mx.array(old_bias), mx.array(new_bias),
            mx.array(codes.reshape(-1, 1)), None if weights is None else mx.array(weights),
        )
        mx.eval(parts)
        actual = np.asarray(parts, dtype=np.float64)

    # Each partial is one or two per-row float32 operations. The absolute
    # allowance scales with the represented step; an erased tiny update cannot
    # pass merely because it is small in absolute terms.
    represented_step = np.max(np.abs(new_bias.astype(np.float64) - old_bias.astype(np.float64)))
    maximum_weight = 1.0 if weights is None else float(np.max(weights))
    np.testing.assert_allclose(actual, expected, rtol=5e-6,
                               atol=max(5e-15, 8e-7 * represented_step * maximum_weight))
    meaningful = np.abs(expected) > 2e-5 * represented_step * maximum_weight
    assert meaningful.any()
    np.testing.assert_array_equal(np.sign(actual[meaningful]), np.sign(expected[meaningful]))
    if rows == 33 and weights is not None:
        np.testing.assert_array_equal(actual[weights == 0], 0.0)


def test_naive_float32_exp_and_log_erase_representable_tiny_steps():
    x = np.array([-1e-8, -1e-9, 1e-9, 1e-8], dtype=np.float32)
    naive_log = np.log(np.float32(1) + x)
    naive_exp = np.exp(x) - np.float32(1)
    assert np.all(naive_log == 0) and np.all(naive_exp == 0)
    assert np.all(np.log1p(x.astype(np.float64)) != 0)
    assert np.all(np.expm1(x.astype(np.float64)) != 0)
