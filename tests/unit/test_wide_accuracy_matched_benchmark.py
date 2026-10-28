"""Guard accuracy-matched benchmark audits against silent bad fits."""
from __future__ import annotations

import copy

import pytest
from benchmarks.sketch_weighted import balanced_order
from benchmarks.wide_accuracy_matched import audit_fits, reference_valid


@pytest.mark.parametrize("count", [4, 6])
def test_measured_order_balances_each_position_after_warmup(count):
    variants = tuple(range(count))
    positions = {variant: [0] * count for variant in variants}
    for repetition in range(3, 15):
        for position, variant in enumerate(balanced_order(variants, repetition)):
            positions[variant][position] += 1
    assert all(counts == [12 // count] * count for counts in positions.values())


def _score():
    return {"objective_float64": 0.5, "gradient_max_abs_float64": 1e-6,
            "objective_relative_gap_to_sklearn": 1e-6, "prediction_agreement_sklearn": 1.0}


def _detail():
    return {"backend": "mlx", "warnings": [], "solutions": [{"converged": True}],
            "sketch_calls": [{"seeded": True, "pairs": 4}], "native_loop_calls": ["completed"]}


def test_reference_requires_finite_gradient_and_warning_free_result():
    score = _score()
    assert reference_valid(score, [], 1e-5)
    assert not reference_valid(score, [{"category": "ConvergenceWarning"}], 1e-5)
    score["gradient_max_abs_float64"] = float("nan")
    assert not reference_valid(score, [], 1e-5)


def test_audit_checks_every_timed_fit_and_the_seeded_route():
    score, detail = _score(), _detail()
    assert audit_fits("sketch", [score], [detail], 1e-4, 1e-5)["passed"]
    later = copy.deepcopy(score)
    later["objective_relative_gap_to_sklearn"] = 2e-5
    assert not audit_fits("sketch", [score, later], [detail, detail], 1e-4, 1e-5)["passed"]
    wrong_route = copy.deepcopy(detail)
    wrong_route["sketch_calls"] = [{"seeded": False, "pairs": 0}]
    assert not audit_fits("sketch", [score], [wrong_route], 1e-4, 1e-5)["passed"]
    assert not audit_fits("sketch", [score], [], 1e-4, 1e-5)["passed"]
    warned = copy.deepcopy(detail)
    warned["warnings"] = [{"category": "ConvergenceWarning", "message": "limit"}]
    assert not audit_fits("current", [score], [warned], 1e-4, 1e-5)["passed"]


def test_device_requires_convergence_and_valid_sketch():
    score, detail = _score(), _detail()
    detail.update(backend="mlx_device", converged=True, sketch_pairs=4)
    assert audit_fits("device", [score], [detail], 1e-4, 1e-5)["passed"]
    detail["converged"] = False
    assert not audit_fits("device", [score], [detail], 1e-4, 1e-5)["passed"]
    detail.update(converged=True, sketch_pairs=0)
    assert not audit_fits("device", [score], [detail], 1e-4, 1e-5)["passed"]
