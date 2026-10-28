"""Independent recurrence checks for the wide cooperative Metal direction."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn._common.device import gpu_available, use_device
from mlxlearn._solvers import native_command, native_logistic
from mlxlearn._solvers.lbfgs import LBFGSConfig


def test_native_and_device_generators_select_matching_direction_widths():
    from mlxlearn_device._native import source_for

    for d, width in ((1024, 32), (2049, 256), (4096, 256)):
        args = (257, d, d + 1, 10, 8, 3, 8, 8, 4, False)
        source = native_command._source(*args)
        assert source == source_for(*args)
        assert f"#define DIRECTION_THREADS {width}\n" in source
        assert "#define THREADS DIRECTION_THREADS\n" in source


@pytest.mark.parametrize("width", [32, 256])
@pytest.mark.parametrize("weighted", [False, True])
def test_device_generator_accepts_explicit_compatible_direction_width(width, weighted):
    from mlxlearn_device._native import source_for

    args = (257, 4096, 4097, 10, 8, 3, 8, 8, 4, weighted, width)
    source = source_for(*args)
    assert source == native_command._source(*args)
    assert f"#define DIRECTION_THREADS {width}\n" in source


@pytest.mark.parametrize("supports_cooperative,width", [(False, 32), (True, 256)])
def test_wide_host_launch_matches_optional_extension_capability(monkeypatch, supports_cooperative, width):
    sources = []

    def run(source, capsules, n, d, p, history, blocks, groups, max_iter, *args):
        sources.append(source)
        return np.zeros(p, np.float32).tobytes(), 2, 1, 16, 0., 0.

    extension = SimpleNamespace(run=run)
    if supports_cooperative:
        extension.run_cg = object()
    monkeypatch.setattr(native_command, "_metal_loop", extension)
    monkeypatch.setattr(native_command, "_supports_metal_dlpack", lambda: True)
    with use_device("cpu"):
        X = mx.ones((5, 4096), mx.float32)
        y = mx.array([0., 1., 0., 1., 0.], mx.float32)
        parts = mx.zeros((2, 1, 4097), mx.float32)
        result = native_command.run_native_loop(
            X, y, None, 5., .1, True, LBFGSConfig(max_iter=10),
            lambda theta: (.5, mx.zeros_like(theta)), initial_parts=parts,
        )
    assert result.converged
    assert len(sources) == 1
    assert f"#define DIRECTION_THREADS {width}\n" in sources[0]


@pytest.mark.parametrize("width", ["0", "7", "1024", "-32", "256 trailing", "256.0"])
def test_native_rejects_invalid_direction_launch_width(width):
    extension = native_command._metal_loop
    if extension is None:
        pytest.skip("requires the native extension")
    source = f"#define WEIGHTED 0\n#define DIRECTION_THREADS {width}\n"
    with pytest.raises(ValueError, match="direction threadgroup"):
        extension.run(source, (), 1, 1, 2, 1, 1, 1, 1, 1., 1e-4, 1e-4, 1e-12, (), b"")


def _run_direction(w, g, oldw, oldg, hs, hy, hc, *, count, head):
    p = len(w)
    d = p - 1
    width = 256 if d >= 2048 else 32
    meta = np.array([10, 0, count, head, -.001, 0, 0, 0, 1, 80, -.001, .3], np.float32)
    arrays = [
        w, g, oldw, oldg, hs, hy, hc, meta,
        np.concatenate([np.full(d, .0001, np.float32), np.zeros(1, np.float32)]),
        np.array(1e-12, np.float32), np.full(p, 4., np.float32),
        np.array(1e-10, np.float32), np.array(1., np.float32), np.zeros(d, np.float32),
    ]
    output = native_logistic._direction(
        inputs=[mx.array(a) for a in arrays],
        template=[("P", p), ("D", d), ("M", 10), ("WEIGHTED", 0), ("THREADS", width)],
        grid=(width, 1, 1), threadgroup=(width, 1, 1),
        output_shapes=[(p,), (10, p), (10, p), (10, 2), (12,)],
        output_dtypes=[mx.float32] * 5,
    )
    mx.eval(output)
    return [np.asarray(value).copy() for value in output]


def _reference(g, oldw, oldg, w, hs, hy, hc, count, head):
    hs, hy, hc = (value.astype(np.float64).copy() for value in (hs, hy, hc))
    s = w.astype(np.float64) - oldw.astype(np.float64)
    y = g.astype(np.float64) - oldg.astype(np.float64)
    sy, yy = s @ y, 4. * (y @ y)
    if sy > 1e-10 * max(yy, 1.) and yy > 0:
        hs[head], hy[head] = s, y
        hc[head] = [1. / sy, sy / yy]
        head, count = (head + 1) % 10, min(count + 1, 10)
    q = g.astype(np.float64).copy()
    alphas = []
    for a in range(count):
        h = (head + 10 - 1 - a) % 10
        alpha = (hs[h] @ q) * hc[h, 0]
        alphas.append(alpha)
        q -= alpha * hy[h]
    q *= 4. * (hc[(head + 9) % 10, 1] if count else 1.)
    for a in range(count):
        h = (head + 10 - count + a) % 10
        beta = (hy[h] @ q) * hc[h, 0]
        q += hs[h] * (alphas[count - 1 - a] - beta)
    return -q, count, head


@pytest.mark.gpu
@pytest.mark.parametrize("features", [2049, 4096])
@pytest.mark.parametrize("count,head", [(0, 0), (5, 3), (10, 7)])
@pytest.mark.parametrize("insert", [False, True])
def test_cooperative_direction_matches_float64_ring_reference(features, count, head, insert):
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(8103)
    p = features + 1
    hs = rng.normal(scale=.01, size=(10, p)).astype(np.float32)
    hy = hs * rng.uniform(.1, 2., size=p).astype(np.float32)
    hc = np.stack([1. / np.sum(hs * hy, axis=1),
                   np.sum(hs * hy, axis=1) / np.sum(4. * hy * hy, axis=1)], axis=1)
    w = rng.normal(scale=.01, size=p).astype(np.float32)
    g = rng.normal(scale=.01, size=p).astype(np.float32)
    oldw = w - hs[0] if insert else w
    oldg = g - hy[0] if insert else g
    expected, newcount, newhead = _reference(g, oldw, oldg, w, hs, hy, hc, count, head)
    with use_device("gpu"):
        actual, _, _, _, stats = _run_direction(w, g, oldw, oldg, hs, hy, hc,
                                               count=count, head=head)
    assert np.linalg.norm(actual - expected) <= 1e-6 * np.linalg.norm(expected)
    assert actual.astype(np.float64) @ g.astype(np.float64) < 0
    assert stats[1] == 0
    assert stats[2] == newcount
    assert stats[3] == newhead


@pytest.mark.gpu
def test_cooperative_direction_preserves_general_row_space_relation():
    if not gpu_available():
        pytest.skip("requires an MLX GPU")
    rng = np.random.default_rng(6208)
    p = 4097
    hs = rng.normal(scale=.01, size=(10, p)).astype(np.float32)
    hs[:, 4095] = hs[:, 1] + hs[:, 2047]
    hy = hs * np.arange(1, 11, dtype=np.float32)[:, None]
    hc = np.stack([1. / np.sum(hs * hy, axis=1),
                   np.sum(hs * hy, axis=1) / np.sum(4. * hy * hy, axis=1)], axis=1)
    g = rng.normal(scale=.01, size=p).astype(np.float32)
    g[4095] = g[1] + g[2047]
    w = np.zeros(p, np.float32)
    with use_device("gpu"):
        direction, _, _, _, _ = _run_direction(w, g, w, g, hs, hy, hc, count=10, head=4)
    error = abs(float(direction[4095]) - float(direction[1]) - float(direction[2047]))
    assert error / np.linalg.norm(direction[:4096]) < 2e-7
