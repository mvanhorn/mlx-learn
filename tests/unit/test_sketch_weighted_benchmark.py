"""Ensure frozen-source and objective-regression benchmark checks are real."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from benchmarks.hessian_logistic import Case
from benchmarks.sketch_weighted import (
    CASES,
    EXTENDED_CASES,
    EXTENDED_TIGHT_CASES,
    TIGHT_CASES,
    audit_previous_objective,
    audit_previous_regression,
    benchmark_dataset,
    reference_sketch_module,
    selected_cases,
    tight_reference_usable,
)


def test_frozen_sketch_loads_verified_builder():
    root = Path(__file__).resolve().parents[2]
    module = reference_sketch_module(
        root / "benchmarks/reference/sketch-weighted-start-sources-2026-10-27.zip"
    )
    assert callable(module.build_sketch_history)
    assert callable(module.initial_metric_from_parts)


def test_previous_regression_checks_objective_and_frozen_route():
    good = {"objective_float64": 0.2, "gradient_max_abs_float64": 1e-5}
    worse = {"objective_float64": 0.200004, "gradient_max_abs_float64": 1e-5}
    scores = {"baseline": [deepcopy(good)], "baseline_sketch": [deepcopy(good)],
              "current": [deepcopy(good)], "sketch": [deepcopy(worse)]}
    detail = {"backend": "mlx", "warnings": [], "solutions": [{"converged": True}],
              "sketch_calls": [{"seeded": True, "pairs": 4}], "native_loop_calls": ["completed"]}
    details = {name: [deepcopy(detail)] for name in scores}
    audit = audit_previous_regression(scores, details, 1e-4)
    assert not audit["passed"]
    assert any(item["variant"] == "sketch" and "objective" in item["reason"] for item in audit["failures"])
    scores["sketch"] = [deepcopy(good)]
    details["baseline_sketch"][0]["native_loop_calls"] = ["fallback"]
    audit = audit_previous_regression(scores, details, 1e-4)
    assert not audit["passed"]
    assert any("native sketch" in item["reason"] for item in audit["failures"])


def test_previous_regression_rejects_an_earlier_invalid_reference_fit():
    good = {"objective_float64": 0.2, "gradient_max_abs_float64": 1e-5}
    invalid = {"objective_float64": 0.2, "gradient_max_abs_float64": float("nan")}
    detail = {"backend": "mlx", "warnings": [], "solutions": [{"converged": True}],
              "sketch_calls": [{"seeded": True, "pairs": 4}], "native_loop_calls": ["completed"]}
    scores = {"baseline": [invalid, good], "current": [good, good]}
    details = {name: [deepcopy(detail), deepcopy(detail)] for name in scores}
    audit = audit_previous_regression(scores, details, 1e-4)
    assert not audit["passed"]
    assert any(failure["variant"] == "baseline" and failure["fit"] == 0 for failure in audit["failures"])


def test_unbalanced_fixtures_cover_label_and_effective_weight_imbalance():
    def make(name):
        return benchmark_dataset(Case(name, 1_000, 16, seed=83, weighted=True))

    X, y, weights, _ = make("weighted_unbalanced_1024")
    assert X.shape == (1_000, 16)
    assert np.mean(y) == 0.1
    assert np.all(weights > 0)
    _, rare, _, _ = make("weighted_rare_1024")
    assert np.mean(rare) == 0.02
    _, balanced, effective_weights, _ = make("weighted_effective_unbalanced_1024")
    assert 0.4 < np.mean(balanced) < 0.6
    assert 0.02 < np.average(balanced, weights=effective_weights) < 0.08


def test_extended_wide_fixtures_are_explicit_and_require_tight_reference():
    assert selected_cases(None) == CASES
    assert not ({case.name for case in CASES} & EXTENDED_TIGHT_CASES)
    assert TIGHT_CASES == {"weighted_unbalanced_1024", "weighted_rare_1024",
                           "weighted_effective_unbalanced_1024"}
    names = {case.name for case in EXTENDED_CASES}
    assert names == {"weighted_ill_conditioned_4096", "shifted_4096",
                     "weighted_shifted_4096", "rank_deficient_4096"}
    assert EXTENDED_TIGHT_CASES == names
    assert [case.name for case in selected_cases(["rank_deficient_4096"])] == ["rank_deficient_4096"]
    for case in EXTENDED_CASES:
        assert case.features == 4_096 and case.rows == 20_000
    rank = next(case for case in EXTENDED_CASES if case.rank_deficient)
    assert rank.C == 1e6
    assert any(case.weighted and case.ill_conditioned for case in EXTENDED_CASES)
    assert any(case.weighted and case.shifted for case in EXTENDED_CASES)


def test_tight_reference_requires_stricter_independent_gradient_and_no_warning():
    score = {"objective_float64": 0.2, "gradient_max_abs_float64": 9e-8}
    assert tight_reference_usable(score, [], 1e-4)
    score["gradient_max_abs_float64"] = 2e-7
    assert not tight_reference_usable(score, [], 1e-4)
    score["gradient_max_abs_float64"] = 9e-8
    assert not tight_reference_usable(score, [{"category": "ConvergenceWarning"}], 1e-4)


def test_previous_objective_audit_preserves_known_inaccurate_baselines():
    scores = {"baseline": [{"objective_float64": 0.2}],
              "baseline_sketch": [{"objective_float64": 0.201}]}
    unavailable = audit_previous_objective(scores, None)
    assert unavailable["passed"] is None
    audit = audit_previous_objective(scores, 0.2)
    assert not audit["passed"]
    assert audit["fits"][0]["passed"]
    assert not audit["fits"][1]["passed"]
