"""Accuracy gates for the paired weighted sklearn comparison."""

from copy import deepcopy

from benchmarks.weighted_tight_logistic import audit_variant


def test_audit_checks_every_fit_and_uses_the_tight_reference():
    good = {"gradient_max_abs_float64": 1e-7,
            "objective_relative_gap_to_sklearn": 1e-6,
            "prediction_agreement_sklearn": 1.0}
    detail = {"backend": "mlx", "warnings": [], "solutions": [{"converged": True}]}
    assert audit_variant("current", [good], [detail], 1e-4, 1e-5)["passed"]
    bad = deepcopy(good)
    bad["objective_relative_gap_to_sklearn"] = 2e-5
    audit = audit_variant("current", [good, bad], [detail, detail], 1e-4, 1e-5)
    assert not audit["passed"]
    assert audit["failures"] == [{"fit": 1, "reason": "objective exceeds tight reference limit"}]


def test_audit_rejects_inaccurate_or_unconverged_sklearn():
    good = {"gradient_max_abs_float64": 1e-5,
            "objective_relative_gap_to_sklearn": 0.0,
            "prediction_agreement_sklearn": 1.0}
    detail = {"backend": "sklearn", "warnings": [{"category": "ConvergenceWarning"}]}
    audit = audit_variant("sklearn", [good], [detail], 1e-6, 1e-5)
    assert not audit["passed"]
    assert {failure["reason"] for failure in audit["failures"]} == {
        "reported nonconvergence", "original gradient exceeds requested tolerance"
    }
