"""Independent scoring and failure accounting for the Hessian benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.hessian_logistic import Case, audit_case, dataset, independent_score
from scipy.special import expit


def test_independent_score_uses_original_coordinates_weights_and_penalty():
    X = np.array([[1.0, 2.0], [-1.0, -2.0], [2.0, 4.0]], dtype=np.float32)
    y = np.array([1, 0, 1], dtype=np.int32)
    weights = np.array([0.5, 2.0, 1.5])
    coef = np.array([[0.1, 0.2]])
    model = SimpleNamespace(coef_=coef, intercept_=np.array([0.3]))
    score, predictions = independent_score(model, X, y, weights, 2.0, np.array([2.0]))
    logits = X.astype(np.float64) @ coef[0] + 0.3
    residual = (expit(logits) - y) * weights
    gradient = X.T @ residual / weights.sum() + coef[0] / (2 * weights.sum())
    bias_gradient = residual.sum() / weights.sum()
    objective = (weights @ (np.logaddexp(0, logits) - y * logits)
                 + (coef[0] @ coef[0]) / 4) / weights.sum()
    np.testing.assert_allclose(score["objective_float64"], objective, rtol=1e-14)
    np.testing.assert_allclose(score["gradient_max_abs_float64"], max(abs(gradient).max(), abs(bias_gradient)))
    assert score["nullspace_coefficient_relative"] == 0.0
    np.testing.assert_array_equal(predictions, logits >= 0)


def test_wide_rank_fixture_exposes_exact_nontrivial_nullspace():
    case = Case("rank_4096", 8, 4_096, rank_deficient=True, C=1e6)
    X, y, weights, ratio = dataset(case)
    assert X.shape == (8, 4_096) and y.shape == (8,)
    assert weights is None and ratio.shape == (2_048,)
    np.testing.assert_array_equal(ratio[:4], np.array([2.0, 0.5, 2.0, 0.5], np.float32))
    np.testing.assert_array_equal(X[:, 2_048:], X[:, :2_048] * ratio)

    # Coefficients orthogonal to every data row must not pass the null-space audit.
    coef = np.concatenate([ratio, -np.ones_like(ratio)])[None, :]
    score, _ = independent_score(SimpleNamespace(coef_=coef, intercept_=np.zeros(1)), X, y, None, 1e6, ratio)
    assert score["nullspace_coefficient_relative"] > 1.0


@pytest.mark.parametrize("features", [6, 4_096])
def test_shifted_rank_fixture_preserves_the_known_exact_nullspace(features):
    case = Case("shifted_rank", 6, features, shifted=True, rank_deficient=True)
    X, _, _, ratio = dataset(case)
    half = features // 2
    assert ratio.shape == (half,)
    np.testing.assert_array_equal(X[:, half:], X[:, :half] * ratio)


def test_rank_fixture_rejects_an_odd_number_of_columns():
    with pytest.raises(ValueError, match="even feature count"):
        dataset(Case("odd_rank", 6, 5, rank_deficient=True))


def test_quality_audit_fails_false_convergence_objective_and_nullspace():
    case = Case("rank", 3, 2, rank_deficient=True)
    valid = {"objective_float64": 0.1, "prediction_agreement_sklearn": 1.0,
             "gradient_max_abs_float64": 1e-5, "objective_relative_gap_to_sklearn": 0.0,
             "nullspace_coefficient_relative": 0.0}
    invalid = {"objective_float64": 0.1, "prediction_agreement_sklearn": 1.0,
               "gradient_max_abs_float64": 2e-4, "objective_relative_gap_to_sklearn": 2e-5,
               "nullspace_coefficient_relative": 0.1}
    details = {"sklearn": [{"warnings": []}],
               "current": [{"backend": "mlx", "warnings": []}, {"backend": "mlx", "warnings": []}]}
    result = audit_case(case, {"sklearn": [valid], "current": [valid, invalid]}, details, 1e-4)
    assert not result["passed"]
    assert len(result["failures"]) == 3
    details["sklearn"][0]["warnings"] = [{"category": "ConvergenceWarning"}]
    result = audit_case(case, {"sklearn": [valid], "current": [valid, invalid]}, details, 1e-4)
    assert not result["sklearn_reference_gradient_and_warning_ok"]
    assert len(result["failures"]) == 3


def test_quality_audit_rejects_nonfinite_and_missing_sketch_execution():
    case = Case("wide", 3, 2)
    reference = {"objective_float64": 0.2, "prediction_agreement_sklearn": 1.0,
                 "gradient_max_abs_float64": 1e-5, "objective_relative_gap_to_sklearn": 0.0,
                 "nullspace_coefficient_relative": None}
    invalid = dict(reference, gradient_max_abs_float64=float("nan"))
    details = {"sklearn": [{"warnings": []}], "current": [{"backend": "mlx", "warnings": []}],
               "sketch": [{"backend": "mlx", "warnings": [], "sketch_calls": [], "native_loop_calls": []}]}
    result = audit_case(case, {"sklearn": [reference], "current": [reference], "sketch": [invalid]}, details, 1e-4)
    reasons = {item["reason"] for item in result["failures"]}
    assert "nonfinite gradient_max_abs_float64" in reasons
    assert "seeded native loop was not used" in reasons
