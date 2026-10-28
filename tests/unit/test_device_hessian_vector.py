"""Boundary and numerical checks for the private device HVP component."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device._hessian_vector import (
    MetalLimits,
    logistic_curvature,
)
from mlxlearn_device._hessian_vector import (
    hessian_vector_plan as plan,
)
from mlxlearn_device._hessian_vector import (
    hessian_vector_product as hvp,
)
from mlxlearn_device._hessian_vector import (
    prepare_hessian_design as prepare_design,
)


def limits(**changes):
    values = {"available": True, "device_max_threads_per_group": 1024,
              "max_threadgroup_bytes": 32 * 1024, "max_buffer_bytes": 2**32,
              "pipeline_max_threads_per_group": 1024, "pipeline_static_threadgroup_bytes": 32 * 1024,
              "max_workspace_bytes": 64 * 1024 * 1024}
    return MetalLimits(**(values | changes))


def test_plan_bounds_workspace_and_dynamic_device_limits():
    result = plan(20_000, 4096, 4097, limits())
    assert result["fused"] and result["scratch_bytes"] == 32 * 1024
    assert result["partial_bytes"] == 157 * 4097 * 4
    assert result["grid_threads"] == 157 * 256
    assert plan(20_000, 4096, 4097, limits(max_threadgroup_bytes=32 * 1024 - 1))["reason"] == "threadgroup-memory-limit"
    assert plan(20_000, 4096, 4097, limits(pipeline_max_threads_per_group=255))["reason"] == "threadgroup-size-limit"
    assert plan(20_000, 4096, 4097, limits(max_buffer_bytes=2_000_000))["reason"] == "workspace-limit"
    assert plan(1_000_000, 4096, 4097, limits())["reason"] == "workspace-limit"
    assert plan(2**31, 4096, 4097, limits())["reason"] == "unsupported-dimensions"
    assert plan(20_000, 4097, 4098, limits())["reason"] == "unsupported-dimensions"
    assert plan(20_000, 512, 513, None)["reason"] == "metal-capabilities-unavailable"
    assert plan(20_000, 512, 513, limits(available=False))["reason"] == "metal-capabilities-unavailable"
    assert plan(20_000, 512, 513, limits(pipeline_max_threads_per_group=None))["reason"] == "pipeline-capabilities-unavailable"
    assert plan(20_000, 512, 513, limits(pipeline_static_threadgroup_bytes=None))["reason"] == "pipeline-capabilities-unavailable"
    assert plan(20_000, 4096, 4097, limits(pipeline_static_threadgroup_bytes=32 * 1024 - 1))["reason"] == "threadgroup-memory-limit"


def test_compiled_pipeline_cap_is_stricter_than_device_cap():
    device_threads = 1024
    compiled_pipeline_threads = 128
    verified = limits(device_max_threads_per_group=device_threads,
                      pipeline_max_threads_per_group=compiled_pipeline_threads)
    # The 4096-feature kernel needs 256 threads: device capability alone is
    # insufficient when the specialized pipeline supports fewer threads.
    assert plan(20_000, 4096, 4097, verified)["reason"] == "threadgroup-size-limit"


@pytest.mark.parametrize("field", [
    "device_max_threads_per_group", "max_threadgroup_bytes", "max_buffer_bytes",
    "pipeline_max_threads_per_group", "pipeline_static_threadgroup_bytes", "max_workspace_bytes",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, False, 0, -1, 1.5, "1024"])
def test_invalid_metal_capabilities_fail_closed(field, value):
    assert plan(20_000, 4096, 4097, limits(**{field: value}))["reason"] == "invalid-metal-capabilities"


@pytest.mark.parametrize("value", [float("nan"), 1, "true", None])
def test_invalid_metal_availability_fails_closed(value):
    assert plan(20_000, 4096, 4097, limits(available=value))["reason"] == "invalid-metal-capabilities"


@pytest.mark.parametrize("n,d,p", [(0, 4, 4), (3, 0, 0), (3, 4, 6)])
def test_plan_rejects_invalid_dimensions(n, d, p):
    with pytest.raises(ValueError):
        plan(n, d, p, limits())


@pytest.mark.parametrize("intercept", [False, True])
def test_cpu_fallback_handles_noncontiguous_and_no_intercept(intercept):
    with mx.stream(mx.cpu):
        # Transpose and slicing create a non-row-contiguous input view.
        raw = mx.reshape(mx.arange(48, dtype=mx.float32), (8, 6)).T[:, ::2]
        design = prepare_design(raw, device=mx.cpu)
        X = design.array
        curvature = mx.array([0.0, 0.2, 0.1, 0.4, 0.3, 0.0], dtype=mx.float32)
        d = X.shape[1]
        vector = mx.arange(d + int(intercept), dtype=mx.float32) * 0.1 + 0.2
        reg = mx.full((d + int(intercept),), 0.03, dtype=mx.float32)
        if intercept:
            reg = mx.concatenate([reg[:-1], mx.zeros((1,), mx.float32)])
        product, chosen = hvp(design, curvature, vector, reg, device=mx.cpu, limits=limits())
        mx.eval(product)
        host_X, host_c, host_v, host_reg = map(np.asarray, (X, curvature, vector, reg))
        projected = host_X.astype(np.float64) @ host_v[:d].astype(np.float64)
        if intercept:
            projected += host_v[d]
        residual = host_c.astype(np.float64) * projected
        expected = host_X.astype(np.float64).T @ residual
        if intercept:
            expected = np.concatenate([expected, [residual.sum()]])
        expected += host_reg.astype(np.float64) * host_v.astype(np.float64)
        assert not chosen["fused"]
        np.testing.assert_allclose(np.asarray(product), expected, rtol=3e-6, atol=2e-5)


def test_cpu_fallback_skips_zero_weight_overflow_and_rejects_bad_curvature():
    with mx.stream(mx.cpu):
        X = mx.array([[1e38, -1e38], [0.0, 0.0]], dtype=mx.float32)
        design = prepare_design(X, device=mx.cpu)
        vector = mx.array([10.0, -10.0, 2.0], dtype=mx.float32)
        reg = mx.zeros((3,), dtype=mx.float32)
        product, _ = hvp(design, mx.array([0.0, 1.0]), vector, reg, device=mx.cpu)
        mx.eval(product)
        assert np.isfinite(np.asarray(product)).all()
        for value in (-1.0, float("inf"), float("nan")):
            bad, _ = hvp(prepare_design(mx.ones((2, 2)), device=mx.cpu), mx.array([value, 1.0]), vector, reg, device=mx.cpu)
            mx.eval(bad)
            assert not np.isfinite(np.asarray(bad)).all()
        overflow, _ = hvp(design, mx.array([1.0, 0.0]), vector, reg, device=mx.cpu)
        mx.eval(overflow)
        assert not np.isfinite(np.asarray(overflow)).all()


def test_design_is_prepared_once_and_device_must_match(monkeypatch):
    calls = []
    original = mx.contiguous
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(mx, "contiguous", counted)
    with mx.stream(mx.cpu):
        raw = mx.reshape(mx.arange(24, dtype=mx.float32), (4, 6)).T
        design = prepare_design(raw, device=mx.cpu)
        curvature = mx.ones((6,), mx.float32)
        vector = mx.ones((4,), mx.float32)
        reg = mx.zeros((4,), mx.float32)
        for _ in range(2):
            product, _ = hvp(design, curvature, vector, reg)
            mx.eval(product)
        with pytest.raises(ValueError, match="prepared once"):
            hvp(raw, curvature, vector, reg)
        with pytest.raises(ValueError, match="match the prepared"):
            hvp(design, curvature, vector, reg, device=mx.gpu)
    assert len(calls) == 1


def test_stable_curvature_preserves_saturated_logits_and_zero_weights():
    with mx.stream(mx.cpu):
        logits = mx.array([40.0, -40.0, float("inf"), float("nan"), 1.0], mx.float32)
        weights = mx.array([1.0, 1.0, 0.0, 0.0, 0.0], mx.float32)
        curvature = logistic_curvature(logits, weights, 2.0, device=mx.cpu)
        mx.eval(curvature)
        result = np.asarray(curvature)
    assert np.isfinite(result).all()
    assert result[0] > 0 and result[1] > 0
    np.testing.assert_allclose(result[0], 0.5 / (2 + np.exp(40.0) + np.exp(-40.0)), rtol=1e-5)
    np.testing.assert_array_equal(result[2:], np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("logit,weight", [(float("inf"), 1.0), (float("nan"), 1.0), (1.0, -1.0), (1.0, float("inf"))])
def test_stable_curvature_rejects_invalid_active_rows(logit, weight):
    with mx.stream(mx.cpu):
        result = logistic_curvature(mx.array([logit], mx.float32), mx.array([weight], mx.float32), 1.0, device=mx.cpu)
        mx.eval(result)
        assert not np.isfinite(np.asarray(result)).all()


@pytest.mark.parametrize("denominator", [0.0, -1.0, float("inf"), float("nan"), 1e-45])
def test_stable_curvature_rejects_unrepresentable_denominator(denominator):
    with mx.stream(mx.cpu), pytest.raises(ValueError):
        logistic_curvature(mx.zeros(2), mx.ones(2), denominator, device=mx.cpu)


def test_import_does_not_require_numpy_or_sklearn():
    # Probe import in a clean interpreter; the scratch implementation itself
    # may only depend on MLX and the standard library.
    source_path = Path(__file__).resolve().parents[2] / "src"
    source = textwrap.dedent(f"""
        import builtins, sys
        root = {str(source_path)!r}
        sys.path.insert(0, root)
        orig = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'sklearn'):
                raise AssertionError(name)
            return orig(name, *args, **kwargs)
        builtins.__import__ = guarded
        import mlxlearn_device._hessian_vector
        assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", source], check=True, timeout=20)


@pytest.mark.gpu
@pytest.mark.parametrize("intercept", [False, True])
def test_metal_zero_weight_and_noncontiguous_match_float64(intercept):
    if not mx.metal.is_available():
        pytest.skip("requires MLX Metal GPU")
    with mx.stream(mx.gpu):
        raw = mx.reshape(mx.arange(80 * 65, dtype=mx.float32), (80, 65)).T[:, ::2] / 8192
        design = prepare_design(raw, device=mx.gpu)
        X = design.array
        curvature = mx.array(np.where(np.arange(65) % 7 == 0, 0, 0.001).astype(np.float32))
        d = X.shape[1]
        vector = mx.arange(d + int(intercept), dtype=mx.float32) / 100
        reg = mx.ones((d + int(intercept),), mx.float32) / 1000
        if intercept:
            reg = mx.concatenate([reg[:-1], mx.zeros((1,), mx.float32)])
        product, chosen = hvp(design, curvature, vector, reg, device=mx.gpu, limits=limits())
        mx.eval(product)
        hX, hc, hv, hr = (np.asarray(value).astype(np.float64) for value in (X, curvature, vector, reg))
        projection = hX @ hv[:d] + (hv[d] if intercept else 0)
        residual = hc * projection
        expected = hX.T @ residual
        if intercept:
            expected = np.concatenate([expected, [residual.sum()]])
        expected += hr * hv
        assert chosen["fused"]
        np.testing.assert_allclose(np.asarray(product), expected, rtol=7e-5, atol=1e-5)


@pytest.mark.gpu
def test_three_way_cross_block_nullspace_and_symmetric_action():
    if not mx.metal.is_available():
        pytest.skip("requires MLX Metal GPU")
    rng = np.random.default_rng(9201)
    # Integer values make col[511] = col[0] + col[256] exact in float32.
    design = rng.integers(-3, 4, (513, 513)).astype(np.float32)
    design[:, 511] = design[:, 0] + design[:, 256]
    null = np.zeros(514, dtype=np.float32)
    null[0], null[256], null[511] = 1.0, 1.0, -1.0
    assert np.array_equal(design.astype(np.float64) @ null[:513].astype(np.float64), np.zeros(513))
    with mx.stream(mx.gpu):
        prepared = prepare_design(mx.array(design), device=mx.gpu)
        curvature = mx.array(rng.uniform(0.0, 0.01, 513).astype(np.float32))
        regularization = mx.concatenate([mx.full((513,), 1e-4), mx.zeros((1,))])
        u = mx.array(rng.standard_normal(514, dtype=np.float32))
        v = mx.array(rng.standard_normal(514, dtype=np.float32))
        null_mx = mx.array(null)
        Hu, chosen = hvp(prepared, curvature, u, regularization, device=mx.gpu, limits=limits())
        Hv, _ = hvp(prepared, curvature, v, regularization, device=mx.gpu, limits=limits())
        Hnull, _ = hvp(prepared, curvature, null_mx, regularization, device=mx.gpu, limits=limits())
        mx.eval(Hu, Hv, Hnull)
        assert chosen["fused"]
        null_residual = float(mx.max(mx.abs(Hnull - regularization * null_mx)))
        projected_leak = abs(float(mx.sum(null_mx * (Hv - regularization * v))))
        lhs, rhs = float(mx.sum(u * Hv)), float(mx.sum(v * Hu))
        assert null_residual < 1e-4
        assert projected_leak < 1e-3
        assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), abs(rhs), 1.0)


@pytest.mark.gpu
def test_gpu_two_pass_capability_fallback_and_extreme_zero_weight():
    if not mx.metal.is_available():
        pytest.skip("requires MLX Metal GPU")
    with mx.stream(mx.gpu):
        X = mx.array([[1e38, -1e38], [0.0, 0.0]], mx.float32)
        prepared = prepare_design(X, device=mx.gpu)
        v = mx.array([10.0, -10.0, 2.0], mx.float32)
        reg = mx.zeros((3,), mx.float32)
        c = mx.array([0.0, 1.0], mx.float32)
        with pytest.raises(ValueError, match="match the prepared"):
            hvp(prepared, c, v, reg, device=mx.cpu)
        fused, chosen = hvp(prepared, c, v, reg, device=mx.gpu, limits=limits())
        fallback, fallback_plan = hvp(prepared, c, v, reg, device=mx.gpu, limits=limits(pipeline_max_threads_per_group=1))
        mx.eval(fused, fallback)
        assert chosen["fused"] and not fallback_plan["fused"]
        assert np.isfinite(np.asarray(fused)).all() and np.isfinite(np.asarray(fallback)).all()
        np.testing.assert_allclose(np.asarray(fused), np.asarray(fallback), rtol=0, atol=1e-6)
        overflow, _ = hvp(prepared, mx.array([1.0, 0.0], mx.float32), v, reg, device=mx.gpu, limits=limits())
        mx.eval(overflow)
        assert not np.isfinite(np.asarray(overflow)).all()
