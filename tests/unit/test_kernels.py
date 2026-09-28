"""Direct tests of the SVM kernel module, against ``sklearn.metrics.pairwise``.

The estimator suite exercises these functions only through an SVC fit, where a
wrong kernel entry shows up — if at all — as a slightly different model. Here
each entry point is compared with scikit-learn's own kernel functions on both
MLX devices, including the cases the module docstring singles out: a negative
polynomial base raised to an odd power, ``degree=0``, and rbf on data far from
the origin.

Tolerances are float32 ones (``docs/fp32_policy.md``): every entry is computed in
float32 and compared with a float64 reference, relative to the scale of the
kernel's values.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)

from mlxlearn._common.device import gpu_available
from mlxlearn._kernels.kernels import (
    KernelSpec,
    _fused_rbf,
    _transform,
    build_kernel_rows,
    kernel_combine,
    kernel_diagonal,
    kernel_matrix,
)

BACKENDS = [
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device"),
    ),
    "cpu",
]

#: (spec kwargs, why it is here). gamma is deliberately non-round: a fused kernel
#: that captured it as a constant would round it differently from the eager path.
SPECS = [
    ({"kernel": "linear"}, "plain inner product"),
    ({"kernel": "rbf", "gamma": 1.0 / 3.0}, "the default kernel"),
    ({"kernel": "poly", "gamma": 0.37, "degree": 3, "coef0": -0.5}, "negative base, odd degree"),
    ({"kernel": "poly", "gamma": 0.37, "degree": 2, "coef0": 1.0}, "even degree"),
    ({"kernel": "sigmoid", "gamma": 0.11, "coef0": -0.3}, "indefinite kernel"),
]


def _reference(spec: KernelSpec, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``K(A, B)`` in float64 from scikit-learn, on the float32-rounded inputs."""
    A = A.astype(np.float32).astype(np.float64)
    B = B.astype(np.float32).astype(np.float64)
    if spec.kernel == "linear":
        return linear_kernel(A, B)
    if spec.kernel == "rbf":
        return rbf_kernel(A, B, gamma=spec.gamma)
    if spec.kernel == "sigmoid":
        return sigmoid_kernel(A, B, gamma=spec.gamma, coef0=spec.coef0)
    if spec.degree == 0:
        # scikit-learn's polynomial_kernel rejects degree=0; the value is 1.
        return np.ones((A.shape[0], B.shape[0]))
    return polynomial_kernel(A, B, degree=spec.degree, gamma=spec.gamma, coef0=spec.coef0)


def _assert_kernel_close(actual, expected):
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5 * scale)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(300, 12))
    B = rng.normal(size=(220, 12))
    return A, B


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_matrix_matches_sklearn(backend, kwargs, why, data):
    A, B = data
    spec = KernelSpec(**kwargs)
    _assert_kernel_close(kernel_matrix(spec, A, B, backend=backend), _reference(spec, A, B))
    _assert_kernel_close(kernel_matrix(spec, A, backend=backend), _reference(spec, A, A))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_rows_match_sklearn(backend, kwargs, why, data):
    """``KernelRows`` is the path every SVC subproblem above the Gram cutoff takes."""
    A, _ = data
    spec = KernelSpec(**kwargs)
    rows = build_kernel_rows(A, spec, backend=backend)
    expected = _reference(spec, A, A)
    picked = np.array([0, 7, 299, 150, 7])

    batch = rows.rows(picked)
    assert batch.shape == (picked.shape[0], A.shape[0])
    assert batch.dtype == np.float32
    _assert_kernel_close(batch, expected[picked])
    for index in (0, 150, 299):
        _assert_kernel_close(rows.row(index), expected[index])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_combine_matches_sklearn(backend, kwargs, why, data):
    A, B = data
    spec = KernelSpec(**kwargs)
    weights = np.random.default_rng(1).normal(size=(B.shape[0], 3))
    expected = _reference(spec, A, B) @ weights
    # Several tiles, so the pipelined loop and its last, short tile both run.
    actual = kernel_combine(spec, A, B, weights, backend=backend, block_rows=64)
    assert actual.dtype == np.float64
    scale = float(np.abs(_reference(spec, A, B)).max()) * np.abs(weights).sum(axis=0).max()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5 * max(scale, 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_degree_zero_is_the_constant_kernel(backend, data):
    A, B = data
    spec = KernelSpec(kernel="poly", gamma=0.5, degree=0, coef0=-2.0)
    np.testing.assert_array_equal(kernel_matrix(spec, A, B, backend=backend), 1.0)
    np.testing.assert_array_equal(build_kernel_rows(A, spec, backend=backend).rows([3, 4]), 1.0)
    np.testing.assert_array_equal(kernel_diagonal(spec, A), 1.0)


@pytest.mark.parametrize("kwargs, why", SPECS, ids=[why for _, why in SPECS])
def test_kernel_diagonal_matches_sklearn(kwargs, why, data):
    A, _ = data
    spec = KernelSpec(**kwargs)
    expected = np.diag(_reference(spec, A, A))
    np.testing.assert_allclose(kernel_diagonal(spec, A), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rbf_centering_rescues_data_far_from_the_origin(backend):
    """``‖a‖² + ‖b‖² − 2⟨a, b⟩`` at coordinate 1e3 cancels away float32's digits.

    Translating both sides by a common center is exact and brings the identity
    back to the variance scale — ``docs/fp32_policy.md`` §2.1.
    """
    rng = np.random.default_rng(3)
    X = 1e3 + rng.normal(size=(200, 6))
    spec = KernelSpec(kernel="rbf", gamma=0.2)
    center = X.mean(axis=0).astype(np.float32)
    expected = _reference(spec, X, X)
    _assert_kernel_close(kernel_matrix(spec, X, backend=backend, center=center), expected)
    rows = build_kernel_rows(X, spec, backend=backend, center=center)
    _assert_kernel_close(rows.rows(np.arange(20)), expected[:20])

    # Uncentered, the same computation is off by far more than float32 rounding.
    uncentered = kernel_matrix(spec, X, backend=backend)
    assert np.abs(uncentered - expected).max() > 1e-2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_a_row_does_not_depend_on_its_batch(backend, kernel):
    """Row values must not change with the rows fetched alongside them.

    The SMO solver prefetches rows in batches of varying size on the GPU; if a
    row's last bits depended on the batch, which rows happened to be prefetched
    would steer the solver's path. A lone row is therefore evaluated as a
    two-column product rather than a matrix-vector one. On the CPU device, where
    nothing is prefetched, the vectorized transcendental functions' scalar tail
    can still move an entry by an ulp, so there it is a float32-rounding claim.
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1500, 24))
    spec = KernelSpec(kernel=kernel, gamma=0.043, coef0=0.25, degree=3)
    rows = build_kernel_rows(X, spec, backend=backend)
    picked = np.array([5, 17, 1499, 250, 0, 777, 3, 1200, 42, 999, 64, 128, 256])
    together = rows.rows(picked)
    for size in (1, 2, 3, 8):
        for start in range(0, picked.shape[0], size):
            chunk = picked[start : start + size]
            if backend == "mlx":
                np.testing.assert_array_equal(rows.rows(chunk), together[start : start + size])
            else:
                np.testing.assert_allclose(
                    rows.rows(chunk), together[start : start + size], rtol=5e-7, atol=5e-7
                )


@pytest.mark.skipif(not gpu_available(), reason="needs an MLX GPU device")
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly", "sigmoid"])
def test_gpu_row_is_the_batched_row(kernel):
    """On the GPU ``row`` goes through ``rows``, so the two never disagree."""
    rng = np.random.default_rng(12)
    X = rng.normal(size=(900, 16))
    rows = build_kernel_rows(X, KernelSpec(kernel=kernel, gamma=0.07), backend="mlx")
    for index in (0, 450, 899):
        np.testing.assert_array_equal(rows.row(index), rows.rows([index, 1, 2])[0])


def test_fused_rbf_is_bit_identical_to_the_eager_transform():
    """gamma travels as an array argument, never as a traced constant.

    Captured as a Python float, MLX rounds it into the fused kernel differently
    from the eager path's float32 scalar for most values — 1/3 and the default
    ``"scale"`` among them — and every kernel entry moves by an ulp or more.
    """
    import mlx.core as mx

    if not gpu_available():
        pytest.skip("the fused kernel is only used on the GPU")
    rng = np.random.default_rng(5)
    A = rng.normal(size=(257, 19)).astype(np.float32)
    B = rng.normal(size=(301, 19)).astype(np.float32)
    for gamma in (1.0 / 3.0, 1.0 / (19 * float(A.var())), 0.05):
        spec = KernelSpec(kernel="rbf", gamma=gamma)
        with mx.stream(mx.gpu):
            left, right = mx.array(A), mx.array(B)
            dot = left @ right.T
            a_sq = mx.sum(left * left, axis=1)
            b_sq = mx.sum(right * right, axis=1)
            eager = _transform(mx, spec, dot, a_sq, b_sq)
            fused = _fused_rbf()(dot, a_sq, b_sq, mx.array(-gamma, dtype=mx.float32))
            mx.eval(eager, fused)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(eager))


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_combine_blocking_changes_only_rounding(backend, data):
    A, B = data
    spec = KernelSpec(kernel="rbf", gamma=0.08)
    weights = np.random.default_rng(2).normal(size=(B.shape[0], 6))
    whole = kernel_combine(spec, A, B, weights, backend=backend)
    for block_rows in (1, 7, 64, 299):
        blocked = kernel_combine(spec, A, B, weights, backend=backend, block_rows=block_rows)
        np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)
    # And a given blocking is deterministic.
    np.testing.assert_array_equal(
        kernel_combine(spec, A, B, weights, backend=backend, block_rows=64),
        kernel_combine(spec, A, B, weights, backend=backend, block_rows=64),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_owns_its_memory_and_validates_indices(backend):
    X = np.random.default_rng(4).normal(size=(50, 5)).astype(np.float32)
    rows = build_kernel_rows(X, KernelSpec(kernel="rbf", gamma=0.3), backend=backend)

    batch = rows.rows([1, 2, 3])
    assert batch.flags.owndata and batch.flags.c_contiguous
    assert rows.rows(np.array([], dtype=np.int64)).shape == (0, 50)

    # An MLX gather does not bounds-check, so these must be caught before it.
    with pytest.raises(IndexError):
        rows.rows([0, 50])
    with pytest.raises(IndexError):
        rows.rows([-1])
    with pytest.raises(IndexError):
        rows.row(50)
    with pytest.raises(TypeError):
        rows.rows(np.array([0.5]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_rows_does_not_alias_the_callers_array(backend):
    X = np.random.default_rng(6).normal(size=(40, 4)).astype(np.float32)
    rows = build_kernel_rows(X, KernelSpec(kernel="linear"), backend=backend)
    before = rows.rows([0, 1])
    X[:] = 0.0
    np.testing.assert_array_equal(rows.rows([0, 1]), before)
    assert not np.shares_memory(rows.X, X)


def test_prefetch_is_offered_only_where_a_call_is_expensive():
    X = np.zeros((10, 2), dtype=np.float32)
    assert build_kernel_rows(X, KernelSpec(kernel="linear"), backend="cpu").prefetch_rows == 0
    if gpu_available():
        gpu = build_kernel_rows(X, KernelSpec(kernel="linear"), backend="mlx")
        assert gpu.prefetch_rows > 0


# --------------------------------------------------------------- a host with no GPU


def test_mlx_work_on_a_gpu_less_host_runs_on_the_mlx_cpu(monkeypatch, data):
    """backend="mlx" names a fitted backend, not a device; inference_device() decides.

    Each kernel used to map "mlx" straight to the GPU, so a GPU-fitted model loaded
    where no GPU works raised DeviceError under device="auto" — the setting that
    asks for exactly this degradation. The answer is the CPU backend's, bit for bit:
    the same float32 code, the same (eager) tile transform, the same blocking.
    """
    from mlxlearn import config_context
    from mlxlearn._kernels.kernels import KernelBasis
    from mlxlearn.exceptions import DeviceError

    A, B = data
    spec = KernelSpec(kernel="rbf", gamma=0.08)
    weights = np.random.default_rng(3).normal(size=(B.shape[0], 4))
    expected = {
        "matrix": kernel_matrix(spec, A, B, backend="cpu"),
        "combine": kernel_combine(spec, A, B, weights, backend="cpu"),
        "rows": build_kernel_rows(B, spec, backend="cpu").rows([0, 5, 9]),
        "basis": KernelBasis(spec).combine(A, B, weights, backend="cpu"),
    }

    monkeypatch.setattr("mlxlearn._common.device.gpu_available", lambda: False)
    for device in ("auto", "cpu"):
        with config_context(device=device):
            rows = build_kernel_rows(B, spec, backend="mlx")
            assert rows.prefetch_rows == 0  # the CPU's policy, not the GPU's
            actual = {
                "matrix": kernel_matrix(spec, A, B, backend="mlx"),
                "combine": kernel_combine(spec, A, B, weights, backend="mlx"),
                "rows": rows.rows([0, 5, 9]),
                "basis": KernelBasis(spec).combine(A, B, weights, backend="mlx"),
            }
        for name, value in expected.items():
            np.testing.assert_array_equal(actual[name], value, err_msg=name)

    with config_context(device="gpu"):
        for call in (
            lambda: kernel_matrix(spec, A, B, backend="mlx"),
            lambda: kernel_combine(spec, A, B, weights, backend="mlx"),
            lambda: build_kernel_rows(B, spec, backend="mlx"),
            lambda: KernelBasis(spec).combine(A, B, weights, backend="mlx"),
        ):
            with pytest.raises(DeviceError, match="fitted on the MLX GPU"):
                call()
