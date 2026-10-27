"""CPU control-flow checks for certification and rejected Newton candidates."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlxlearn_device import _newton


@pytest.fixture
def cpu_newton(monkeypatch):
    """Exercise real logistic arithmetic with a float64 test-only linear solve."""
    original_stream = mx.stream
    monkeypatch.setattr(mx, "stream", lambda _device: original_stream(mx.cpu))
    calls = []

    def solve(X, curvature, gradient, reg, **_options):
        calls.append(1)
        A = np.column_stack((np.asarray(X, np.float64), np.ones(X.shape[0])))
        g = np.asarray(gradient, np.float64)
        H = A.T @ (np.asarray(curvature, np.float64)[:, None] * A) + np.diag(np.asarray(reg))
        result = -np.linalg.solve(H, g)
        return mx.array(result, mx.float32), (1, 1, 0.0, float(g @ g), 1.0)

    def original_score(X, center, theta, y, sw, total, l2):
        reg = mx.concatenate((mx.full((X.shape[1],), l2), mx.zeros((1,))))
        loss, gradient, _, _ = _newton._evaluate_fallback(X - center, y, sw, theta,
                                                        mx.array(total), reg)
        # This fixture uses zero centers, so the CPU gradient is also original-coordinate.
        return loss, mx.max(mx.abs(gradient)), gradient, mx.zeros_like(gradient), mx.array(0.0)

    monkeypatch.setattr(_newton, "solve_cg", solve)
    monkeypatch.setattr(_newton, "score_original", original_score)
    return calls


def _problem(labels=(0, 0, 1, 1)):
    with mx.stream(mx.cpu):
        return mx.zeros((4, 1)), mx.zeros((1,)), mx.array(labels, mx.float32), mx.ones((4,))


def test_invalid_dual_diagnostic_cannot_promote_stationary_gradient(cpu_newton, monkeypatch):
    X, center, y, weights = _problem()
    monkeypatch.setattr(_newton, "logistic_duality_gap", lambda *_a, **_kw: SimpleNamespace(
        valid=mx.array(False), relative_gap=mx.array(0.0),
    ))
    result = _newton.minimize_newton(X, X, center, y, weights, 4.0, 0.1, tol=1e-4, max_iter=2)
    assert not result.converged
    assert result.stop_reason == "non-descent-direction"
    assert result.n_iter == 0
    assert result.gradient_max_abs == 0


def test_failed_original_gradient_check_cannot_be_promoted(cpu_newton, monkeypatch):
    X, center, y, weights = _problem()
    monkeypatch.setattr(_newton, "score_original", lambda *_a: (
        mx.array(0.693147), mx.array(2e-4), mx.zeros((2,)), mx.zeros((2,)), mx.array(0.0),
    ))
    result = _newton.minimize_newton(X, X, center, y, weights, 4.0, 0.1, tol=1e-4, max_iter=2)
    assert not result.converged
    assert result.stop_reason == "non-descent-direction"
    assert result.n_iter == 0 and result.gradient_max_abs > 1e-4


def test_last_budgeted_update_receives_full_convergence_check(cpu_newton):
    # One Newton bias update moves 0 -> 1 toward the optimum log(3). This
    # deliberately requested loose gate is reached only after that update.
    X, center, y, weights = _problem((0, 1, 1, 1))
    result = _newton.minimize_newton(X, X, center, y, weights, 4.0, 0.1,
                                    tol=0.05, relative_gap_tol=0.01, max_iter=1)
    assert result.converged and result.stop_reason == "duality-gap"
    assert result.n_iter == 1 and len(cpu_newton) == 1
    assert result.gradient_max_abs < 0.05 * 0.95
    assert result.relative_gap <= 0.01


def test_exhausted_budget_is_not_promoted_by_final_gradient_score(cpu_newton):
    X, center, y, weights = _problem((0, 1, 1, 1))
    result = _newton.minimize_newton(X, X, center, y, weights, 4.0, 0.1,
                                    tol=1e-8, max_iter=1)
    assert not result.converged and result.stop_reason == "max_iter"
    assert result.n_iter == 1 and len(cpu_newton) == 1


def test_nonfinite_candidate_gradient_never_replaces_last_accepted_parameters(cpu_newton, monkeypatch):
    X, center, y, weights = _problem((0, 1, 1, 1))
    original_evaluate = _newton._evaluate_fallback
    evaluations = 0

    def broken_candidate(*args):
        nonlocal evaluations
        evaluations += 1
        loss, gradient, curvature, logits = original_evaluate(*args)
        if evaluations > 1:
            gradient = mx.full_like(gradient, float("nan"))
        return loss, gradient, curvature, logits

    monkeypatch.setattr(_newton, "_evaluate_fallback", broken_candidate)
    # The final independent certificate remains finite; it must not conceal
    # the candidate failure or turn the initial parameter into convergence.
    monkeypatch.setattr(_newton, "score_original", lambda *_a: (
        mx.array(0.693147), mx.array(0.0), mx.zeros((2,)), mx.zeros((2,)), mx.array(0.0),
    ))
    result = _newton.minimize_newton(X, X, center, y, weights, 4.0, 0.1, tol=1e-4, max_iter=2)
    assert not result.converged and result.stop_reason == "line-search-failure"
    assert result.n_iter == 0
    np.testing.assert_array_equal(np.asarray(result.theta), np.zeros(2))


def test_newton_and_duality_import_without_numpy_or_sklearn():
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoHostArrays(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == 'numpy' or name.startswith('numpy.') or name == 'sklearn' or name.startswith('sklearn.'):
                    raise AssertionError('forbidden host array dependency: ' + name)
        sys.meta_path.insert(0, NoHostArrays())
        import mlxlearn_device._duality
        import mlxlearn_device._native_cg
        import mlxlearn_device._newton
        assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", source], check=True, timeout=30)
