"""CPU checks for the general-rank fixture and benchmark failure accounting."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.rank_safe_logistic import (
    RankCase,
    audit_rank_fits,
    coefficient_null_ratio,
    observed_newton,
    rank_dataset,
)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_general_rank_fixture_is_exact_deterministic_and_crosses_blocks(scaled, weighted):
    case = RankCase("test", rows=71, features=65, scaled=scaled, weighted=weighted)
    X, y, sw, columns = rank_dataset(case)
    again = rank_dataset(case)
    a, b, last = columns
    np.testing.assert_array_equal(X, again[0])
    np.testing.assert_array_equal(y, again[1])
    assert X.dtype == np.float32 and X.flags.c_contiguous
    assert columns == (1, 31, 64)
    np.testing.assert_array_equal(X[:, last].astype(np.float64),
                                  X[:, a].astype(np.float64) + X[:, b].astype(np.float64))
    assert set(np.unique(y)) == {0, 1}
    if scaled:
        variance = X.astype(np.float64).var(axis=0)
        assert variance.max() / variance.min() > 100
    if weighted:
        assert sw.dtype == np.float32 and np.all(sw >= 0) and np.any(sw == 0)
        np.testing.assert_array_equal(sw, again[2])
    else:
        assert sw is None


def test_three_column_null_ratio_rejects_a_nonpairwise_artifact():
    coefficient = np.zeros((1, 9), dtype=np.float64)
    coefficient[0, [1, 3, 8]] = [-1, -1, 1]
    model = SimpleNamespace(coef_=coefficient)
    assert coefficient_null_ratio(model, (1, 3, 8)) == pytest.approx(1)
    coefficient[0, [1, 3, 8]] = [1, 1, 2]
    assert coefficient_null_ratio(model, (1, 3, 8)) == 0


def _valid_fit():
    score = {"objective_float64": .4, "gradient_max_abs_float64": 1e-6,
             "three_column_null_ratio": 1e-7, "objective_relative_gap_to_tight_reference": 1e-6,
             "prediction_agreement_tight_reference": .999}
    detail = {"backend": "mlx", "warnings": [], "solutions": [{"converged": True}],
              "newton_calls": [{"n_iter": 2, "hvp_count": 21}]}
    return score, detail


def test_rank_audit_does_not_accept_an_invalid_tight_reference():
    score, detail = _valid_fit()
    audit = audit_rank_fits("current", [score], [detail], tol=1e-4,
                           expected_newton=True, reference_valid=False)
    assert not audit["passed"]
    assert any("reference" in item["reason"] for item in audit["failures"])


@pytest.mark.parametrize("error", ["null", "warning", "missing_route", "objective"])
def test_rank_audit_keeps_every_quality_failure(error):
    score, detail = _valid_fit()
    if error == "null":
        score["three_column_null_ratio"] = .1
    elif error == "warning":
        detail["warnings"] = [{"category": "ConvergenceWarning", "message": "budget"}]
    elif error == "missing_route":
        detail["newton_calls"] = []
    else:
        score["objective_relative_gap_to_tight_reference"] = 1.1e-5
    audit = audit_rank_fits("current", [score], [detail], tol=1e-4,
                           expected_newton=True, reference_valid=True)
    assert not audit["passed"]


def test_rank_audit_requires_the_weak_regularization_route_to_avoid_newton():
    score, detail = _valid_fit()
    assert not audit_rank_fits("current", [score], [detail], tol=1e-4,
                              expected_newton=False, reference_valid=True)["passed"]
    detail["newton_calls"] = []
    assert audit_rank_fits("current", [score], [detail], tol=1e-4,
                          expected_newton=False, reference_valid=True)["passed"]
    assert observed_newton("device", {"newton_iterations": 2, "newton_hvp_count": 20})
    assert not observed_newton("device", {"newton_iterations": 0, "newton_hvp_count": 20})
