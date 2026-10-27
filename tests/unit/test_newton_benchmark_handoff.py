"""A partial native warm start must have observed, certified continuation."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from benchmarks import hessian_logistic, logistic_wide

from mlxlearn_device import _newton


def _handoff():
    return {
        "native_loop_calls": ["terminal_nonconvergence"],
        "native_loop_attempts": [{"status": 0, "native_n_iter": 32,
                                  "outcome": "terminal_nonconvergence"}],
        "newton_calls": [{"resumed": True, "converged": True, "stop_reason": "duality-gap",
                          "relative_duality_gap": 7e-6, "n_iter": 3, "budget": 168}],
    }


def test_certified_observed_native_newton_handoff_counts_as_sketch_execution():
    assert hessian_logistic._completed_sketch_or_handoff(_handoff())
    assert hessian_logistic._completed_sketch_or_handoff({"native_loop_calls": ["completed"]})


@pytest.mark.parametrize("field", ["native_loop_attempts", "newton_calls"])
def test_missing_handoff_evidence_is_not_inferred(field):
    details = _handoff()
    details.pop(field)
    assert not hessian_logistic._completed_sketch_or_handoff(details)


@pytest.mark.parametrize("field,value", [
    ("status", 1), ("status", 2), ("status", 3),
    ("native_n_iter", 0), ("native_n_iter", -1),
    ("outcome", "resumed"), ("outcome", "fallback"),
])
def test_failed_or_unused_native_attempt_cannot_qualify_as_warm_start(field, value):
    details = _handoff()
    details["native_loop_attempts"][0][field] = value
    assert not hessian_logistic._completed_sketch_or_handoff(details)


@pytest.mark.parametrize("field,value", [
    ("resumed", False), ("converged", False), ("stop_reason", "gtol"),
    ("relative_duality_gap", 1.00001e-5), ("relative_duality_gap", float("nan")),
    ("relative_duality_gap", float("inf")), ("n_iter", -1), ("n_iter", 169), ("budget", 2),
])
def test_failed_or_overbudget_newton_cannot_complete_handoff(field, value):
    details = _handoff()
    details["newton_calls"][0][field] = value
    assert not hessian_logistic._completed_sketch_or_handoff(details)


def test_one_success_does_not_conceal_another_failed_refinement():
    details = _handoff()
    bad = copy.deepcopy(details["newton_calls"][0])
    bad["converged"] = False
    details["newton_calls"].append(bad)
    assert not hessian_logistic._completed_sketch_or_handoff(details)


@pytest.mark.parametrize("converged", [True, False])
def test_observer_records_actual_newton_return_without_promoting_it(monkeypatch, converged):
    result = SimpleNamespace(n_iter=3, n_fev=12, converged=converged,
                             stop_reason="duality-gap" if converged else "line-search-failure",
                             objective=0.4, gradient_max_abs=2e-6,
                             relative_gap=4e-6, hvp_count=48)
    def original(*_a, **_kw):
        return result
    monkeypatch.setattr(_newton, "minimize_newton", original)
    estimator = SimpleNamespace(solve_logistic=lambda *_a, **_kw: result)
    solver = SimpleNamespace(minimize=lambda *_a, **_kw: result)
    native = SimpleNamespace(minimize_binary=lambda *_a, **_kw: result)
    with logistic_wide.observed_implementation(estimator, solver, native) as record:
        assert _newton.minimize_newton(max_iter=168, initial=object()) is result
        observed = record["newton_calls"]
        assert len(observed) == 1
        assert observed[0]["converged"] is converged
        assert observed[0]["n_iter"] == 3 and observed[0]["budget"] == 168
        assert observed[0]["resumed"]
        details = _handoff()
        details["newton_calls"] = observed
        assert hessian_logistic._completed_sketch_or_handoff(details) is converged
    assert _newton.minimize_newton is original
