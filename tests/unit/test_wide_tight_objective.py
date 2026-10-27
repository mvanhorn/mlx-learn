"""Wide binary fits must pass a tight independent float64 objective comparison."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.hessian_logistic import Case, dataset, independent_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn._common.config import tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import native_command
from mlxlearn.linear_model import LogisticRegression


@pytest.fixture(scope="module", params=[(41, 1.0), (83, 1.0), (41, 10.0)])
def wide_reference(request):
    seed, C = request.param
    case = Case("wide_4096", 20_000, 4_096, seed=seed, C=C)
    X, y, weights, ratio = dataset(case)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        center = X.mean(axis=0, dtype=np.float64)
        centered = X.astype(np.float64) - center
        model = SkLogisticRegression(C=C, tol=1e-10, max_iter=3_000).fit(centered, y)
        reference = SimpleNamespace(coef_=model.coef_, intercept_=model.intercept_ - model.coef_ @ center)
        score, predictions = independent_score(reference, X, y, weights, C, ratio)
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert score["gradient_max_abs_float64"] < 1e-7
    return case, X, y, weights, ratio, score, predictions


@pytest.mark.gpu
@pytest.mark.parametrize("sketch_size", [0, 4])
def test_wide_host_objective_matches_tight_reference(wide_reference, sketch_size):
    if not gpu_available() or native_command._metal_loop is None:
        pytest.skip("requires the native Metal extension and an MLX GPU")
    case, X, y, weights, ratio, reference, predictions = wide_reference
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tuning_context(force_backend="mlx", lbfgs_sketch_size=sketch_size):
            model = LogisticRegression(C=case.C, tol=1e-4, max_iter=200).fit(X, y)
        score, _ = independent_score(model, X, y, weights, case.C, ratio,
                                     (reference["objective_float64"], predictions))
    assert model._execution_backend_ == "mlx"
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert model.n_iter_[0] <= 200
    assert score["gradient_max_abs_float64"] <= 1e-4
    assert score["objective_relative_gap_to_sklearn"] <= 1e-5
    assert score["prediction_agreement_sklearn"] >= 0.995
