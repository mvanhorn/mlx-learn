"""Numerical regression for the line search's small loss differences."""

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import native_command, native_logistic


def test_host_and_numpy_free_source_share_stable_log1p():
    from mlxlearn_device._native import source_for

    args = (257, 1024, 1025, 10, 8, 3, 4, 16, 1, False)
    host = native_command._source(*args)
    assert host == source_for(*args)
    assert "inline float log1p_stable(float x)" in host
    assert "log1p_stable(sigmoid_stable(u)*expm1_small(dz))" in host


@pytest.mark.gpu
@pytest.mark.parametrize("math_mode", ["safe", "fast"])
def test_log1p_stable_matches_float64_across_loss_domain(math_mode):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    kernel = mx.fast.metal_kernel(
        name=f"mlxlearn_test_log1p_stable_{math_mode}",
        input_names=["x"], output_names=["out"], header=native_logistic._REDUCE,
        source="uint i=thread_position_in_grid.x; if(i<N) out[i]=log1p_stable(x[i]);",
        compile_options={"math_mode": math_mode},
    )
    tiny = np.geomspace(1e-30, 1e-2, 101, dtype=np.float32)
    near_negative_one = np.array([-0.99999994, -0.99, -0.50000006], dtype=np.float32)
    x = np.concatenate((-tiny, np.array([0.0], dtype=np.float32), tiny,
                        np.linspace(-0.5, 1.0, 4097, dtype=np.float32), near_negative_one))
    with use_device("gpu"):
        (result,) = kernel(
            inputs=[mx.array(x)], template=[("N", x.size)],
            grid=((x.size + 255) // 256 * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[x.shape], output_dtypes=[mx.float32],
        )
        mx.eval(result)
        actual = np.asarray(result, dtype=np.float64)
    expected = np.log1p(x.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-37)
    assert np.all(actual[:tiny.size] < 0) and np.all(actual[tiny.size + 1:2 * tiny.size + 1] > 0)
