"""The L-BFGS solver, tested directly rather than through an estimator.

``docs/development.md`` shapes every solver as "a frozen config and a callable in, a
frozen state out" precisely so that its numerics can be tested without constructing
an estimator. Most of what this module checks is therefore the *contract* the module
docstring of ``_solvers/lbfgs.py`` promises — which stop reason is reported when,
what ``n_iter`` counts, that a line-search failure returns instead of raising — on
synthetic objectives whose answers are known in closed form.

The pinned logistic trajectories at the bottom are the regression net for changes
to the solver's internals: on well-conditioned problems the iteration count must not
grow and the objective must not get worse.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from sklearn.datasets import make_classification

from mlxlearn._solvers.lbfgs import LBFGSConfig, LBFGSState, minimize
from mlxlearn._solvers.logistic import LogisticConfig, reference_objective, solve_logistic


def _quadratic(scale):
    """``f(x) = 0.5 * Σ scale_i x_i²``, minimized at the origin."""
    a = mx.array(np.asarray(scale, dtype=np.float32))

    def fun(x):
        return float(0.5 * mx.sum(a * x * x)), a * x

    return fun


def _rosenbrock(x):
    """The n-dimensional Rosenbrock function, minimized at ``x = 1``."""
    head, tail = x[:-1], x[1:]
    value = mx.sum(100.0 * (tail - head * head) ** 2 + (1.0 - head) ** 2)
    grad_head = -400.0 * head * (tail - head * head) - 2.0 * (1.0 - head)
    grad_tail = 200.0 * (tail - head * head)
    zero = mx.zeros((1,), dtype=x.dtype)
    grad = mx.concatenate([grad_head, zero]) + mx.concatenate([zero, grad_tail])
    return float(value), grad


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": -1},
        {"tol": -1e-4},
        {"tol": math.inf},
        {"tol": math.nan},
        {"history_size": 0},
        {"max_line_search": 0},
        {"c1": 0.0},
        {"c1": 0.95, "c2": 0.9},
        {"c2": 1.0},
        {"min_curvature": -1.0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LBFGSConfig(**kwargs)


# ------------------------------------------------------------------ stop reasons


def test_quadratic_converges_on_gtol():
    fun = _quadratic([1.0, 10.0, 100.0])
    state = minimize(fun, mx.array([3.0, -4.0, 0.5]), LBFGSConfig(tol=1e-6))

    assert isinstance(state, LBFGSState)
    assert state.converged and state.stop_reason == "gtol"
    assert state.grad_max_abs <= 1e-6
    assert state.n_fev >= state.n_iter + 1
    np.testing.assert_allclose(np.asarray(state.x), 0.0, atol=1e-6)


def test_the_parameter_keeps_its_shape_and_type():
    """The solver treats ``x`` as a vector space and never reshapes the caller's layout."""
    fun = _quadratic(np.arange(1.0, 7.0).reshape(2, 3))
    state = minimize(fun, mx.ones((2, 3)), LBFGSConfig(tol=1e-6))
    assert isinstance(state.x, mx.array)
    assert tuple(state.x.shape) == (2, 3)
    assert state.converged


def test_an_optimal_starting_point_takes_no_steps():
    """``n_iter`` counts accepted steps, so a converged start reports zero."""
    state = minimize(_quadratic([1.0, 2.0]), mx.zeros((2,)), LBFGSConfig())
    assert (state.n_iter, state.n_fev, state.stop_reason, state.converged) == (0, 1, "gtol", True)


def test_max_iter_zero_evaluates_once_and_does_not_claim_convergence():
    state = minimize(_quadratic([1.0, 2.0]), mx.array([1.0, 1.0]), LBFGSConfig(max_iter=0))
    assert (state.n_iter, state.n_fev, state.stop_reason, state.converged) == (
        0, 1, "max_iter", False,
    )
    assert state.objective == pytest.approx(1.5)


def test_max_iter_is_reported_when_the_budget_runs_out():
    fun = _quadratic(np.logspace(0, 4, 20))
    state = minimize(fun, mx.ones((20,)), LBFGSConfig(max_iter=2, tol=1e-12))
    assert state.stop_reason == "max_iter" and not state.converged
    assert state.n_iter == 2


def test_a_non_finite_start_is_a_stop_reason_not_an_exception():
    def fun(x):
        return math.inf, mx.zeros_like(x)

    state = minimize(fun, mx.array([1.0]), LBFGSConfig())
    assert state.stop_reason == "non-finite-objective"
    assert not state.converged and state.n_iter == 0


def test_an_inconsistent_gradient_ends_in_line_search_failure():
    """A gradient that points uphill can never meet sufficient decrease.

    That is the ancestor's clipped-objective/unclipped-gradient bug in miniature.
    It must come back as a stop reason with ``converged=False`` — not raise, and not
    take null steps until ``max_iter``.
    """
    base = _quadratic([1.0, 3.0])

    def fun(x):
        value, grad = base(x)
        return value, -grad

    state = minimize(fun, mx.array([2.0, -1.0]), LBFGSConfig(max_iter=50))
    assert state.stop_reason == "line-search-failure"
    assert not state.converged
    assert state.n_iter == 0
    np.testing.assert_array_equal(np.asarray(state.x), [2.0, -1.0])


def test_rosenbrock():
    """A classic non-convex valley. scipy's L-BFGS-B needs 43 iterations on this one."""
    x0 = mx.array([-1.2, 1.0, -1.2, 1.0])
    state = minimize(_rosenbrock, x0, LBFGSConfig(max_iter=500, tol=1e-5))
    assert state.converged, state.stop_reason
    np.testing.assert_allclose(np.asarray(state.x), 1.0, atol=1e-3)
    assert state.n_iter <= 100


# ------------------------------------------------------------------- line search


def test_a_trial_step_into_an_infinite_region_contracts():
    """``-log(x) - log(1 - x)``, infinite outside ``(0, 1)``.

    From ``x = 0.01`` the steepest-descent step overshoots past 1, where the
    objective is infinite. An infinite value carries no slope to interpolate on, so
    the search must contract toward the last finite point rather than give up.
    """

    def fun(x):
        v = np.asarray(x, dtype=np.float64)
        if np.any(v <= 0.0) or np.any(v >= 1.0):
            return math.inf, mx.zeros_like(x)
        value = float(np.sum(-np.log(v) - np.log1p(-v)))
        return value, mx.array((-1.0 / v + 1.0 / (1.0 - v)).astype(np.float32))

    state = minimize(fun, mx.array([0.01]), LBFGSConfig(tol=1e-5))
    assert state.converged, state.stop_reason
    np.testing.assert_allclose(np.asarray(state.x), 0.5, atol=1e-4)


def test_an_exhausted_budget_still_takes_the_best_sufficient_decrease_step():
    """Running out of evaluations is "could not certify this step", not "could not move".

    With a one-evaluation budget, the first trial of this shallow quadratic meets the
    Armijo condition but not the curvature condition. The search must accept it —
    the caller's curvature test protects the Hessian approximation — instead of
    reporting a failure.
    """
    fun = _quadratic([1e-4])
    config = LBFGSConfig(max_iter=1, max_line_search=1)
    state = minimize(fun, mx.array([100.0]), config)
    assert state.n_iter == 1
    assert state.objective < 0.5 * 1e-4 * 100.0**2


# --------------------------------------------------------- pinned logistic trajectories

#: ``(n_classes, C, fit_intercept) -> (n_iter, objective)`` measured with the solver as
#: shipped in 0.1.0a2, on ``make_classification(5000, 20, n_informative=12,
#: n_redundant=4, random_state=0)``. Identical on the GPU and CPU devices. Changes to
#: the solver's internals may reach the same point in fewer iterations or at a lower
#: objective, never the reverse.
_PINNED = {
    (2, 0.01, True): (16, 0.5293678939),
    (2, 1.0, True): (16, 0.5263766616),
    (2, 1.0, False): (17, 0.5294598614),
    (2, 100.0, True): (16, 0.5263454231),
    (5, 0.01, True): (41, 1.0993984138),
    (5, 1.0, True): (47, 1.0847425498),
    (5, 1.0, False): (30, 1.0866184770),
    (5, 100.0, True): (45, 1.0845856399),
}


@pytest.fixture(scope="module")
def _pinned_data():
    return {
        k: make_classification(
            n_samples=5000, n_features=20, n_informative=12, n_redundant=4,
            n_classes=k, random_state=0,
        )
        for k in (2, 5)
    }


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("key", sorted(_PINNED))
def test_logistic_trajectories_do_not_regress(key, device, _pinned_data):
    from mlxlearn._common.device import gpu_available

    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    n_classes, C, fit_intercept = key
    pinned_iter, pinned_objective = _PINNED[key]
    X, y = _pinned_data[n_classes]

    solution = solve_logistic(
        X, y, n_classes=n_classes,
        config=LogisticConfig(C=C, fit_intercept=fit_intercept, device=device),
    )
    objective = reference_objective(
        solution.coef, solution.intercept, X, y, n_classes=n_classes, C=C
    )

    assert solution.converged and solution.stop_reason == "gtol"
    assert solution.n_iter <= pinned_iter
    # 2e-9 absolute is float32 noise on an objective near 1, and well inside the
    # 1e-5 relative parity budget; an actual regression shows up orders larger.
    assert objective <= pinned_objective + 2e-9
