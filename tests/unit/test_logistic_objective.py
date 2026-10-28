"""Precision regressions at the fused objective and line-search boundary."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers.logistic import _binary_objective, _multinomial_objective

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _gpu_stream():
    if not gpu_available():
        pytest.skip("requires MLX GPU")
    with use_device("gpu"):
        yield


@pytest.mark.parametrize("n_classes", [2, 3])
def test_unpenalized_large_finite_coefficients_do_not_square_to_nan(n_classes):
    X = mx.zeros((4, 1))
    if n_classes == 2:
        fun = _binary_objective(mx, X, mx.zeros(4), None, 4.0, 0.0, False, 1)
        theta = mx.full((1, 1), 1e20)
    else:
        fun = _multinomial_objective(
            mx, X, mx.zeros((4, 1), dtype=mx.int32), mx.array(np.eye(3)[[0] * 4]),
            None, 4.0, 0.0, False, 1,
        )
        theta = mx.full((3, 1), 1e20)
    value, gradient, slope = fun.value_gradient_slope(theta, mx.ones_like(theta))
    assert value == pytest.approx(np.log(n_classes), rel=1e-6)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(theta.shape))
    assert slope == 0.0


def test_binary_positive_tail_preserves_loss_gradient_and_slope():
    fun = _binary_objective(mx, mx.ones((1, 1)), mx.ones(1), None, 1.0, 0.0, False, 1)
    theta = mx.array([[20.0]])
    value, gradient, slope = fun.value_gradient_slope(theta, mx.ones_like(theta))
    assert value == pytest.approx(np.logaddexp(0.0, -20.0), rel=1e-5)
    expected = -1.0 / (1.0 + np.exp(20.0))
    assert float(gradient[0, 0]) == pytest.approx(expected, rel=1e-5)
    assert slope == pytest.approx(expected, rel=1e-5)


def test_multinomial_loss_and_gradient_ignore_common_logit_offset():
    fun = _multinomial_objective(
        mx, mx.ones((1, 1)), mx.array([[2]], dtype=mx.int32),
        mx.array([[0.0, 0.0, 1.0]]), None, 1.0, 0.0, False, 1,
    )
    theta = mx.array([[0.0], [1.0], [2.0]])
    value, gradient = fun(theta)
    shifted_value, shifted_gradient = fun(theta + 10000.0)
    assert shifted_value == pytest.approx(value, rel=1e-6)
    np.testing.assert_allclose(np.asarray(shifted_gradient), np.asarray(gradient), rtol=1e-6)
