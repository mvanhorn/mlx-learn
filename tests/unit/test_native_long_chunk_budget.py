"""Long native command batches must respect the accepted-step budget."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import native_command
from mlxlearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def long_weighted_problem():
    """This fixture takes well over 49 accepted steps at the target tolerance."""
    n, d = 20_000, 1_024
    rng = np.random.default_rng(83)
    X = rng.standard_normal((n, d), dtype=np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(np.int32)
    X[:, d // 2:] = X[:, :d // 2] + np.float32(0.01) * X[:, d // 2:]
    X *= np.logspace(-1, 1, d, dtype=np.float32)
    weights = rng.uniform(0.25, 3.0, n)
    return X, y, weights


@pytest.mark.gpu
@pytest.mark.parametrize("budget", [32, 33, 34, 48, 49])
def test_long_native_batch_respects_budget_and_reports_failure(monkeypatch, long_weighted_problem, budget):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    X, y, weights = long_weighted_problem
    extension = native_command._metal_loop
    observed = []

    def run(*args):
        result = extension.run(*args)
        observed.append(result[1:])
        return result

    monkeypatch.setattr(native_command, "_metal_loop", SimpleNamespace(run=run))
    with warnings.catch_warnings(record=True) as caught, tuning_context(force_backend="mlx"):
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=1.0, tol=1e-4, max_iter=budget).fit(X, y, sample_weight=weights)
    assert model.execution_backend_ == "mlx"
    assert len(observed) == 1
    iterations, status, evaluations, *_ = observed[0]
    assert iterations == int(model.n_iter_[0]) == budget
    assert status == 0 and evaluations >= iterations
    assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
