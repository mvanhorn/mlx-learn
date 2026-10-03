"""LogisticRegression's dense L-BFGS preconditioner: its gate, switch and bookkeeping.

What the plan in ``_solvers/logistic.py`` promises (``docs/fp32_policy.md`` §2.3): the
gate fires on correlated features and turns away uncorrelated data, nearly separable
classes and widths it is not built for; a fit it turns away, and every fit with the
switch off, is the diagonal path bit for bit; its host work runs once, while the
device computes the first evaluation, on one BLAS thread; and the fit's one event
records what it decided. Parity of the answers it produces is
``tests/parity/test_logistic_preconditioner.py``'s.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

import mlxlearn
from mlxlearn._common.config import Tuning, _tuning_from_env, tuning_context
from mlxlearn._common.device import gpu_available
from mlxlearn._solvers import logistic
from mlxlearn._solvers.logistic import LogisticConfig, solve_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.patching._estimators import LogisticRegression as PatchedLogisticRegression
from tests._helpers import join_or_fail, logistic_data

DEVICES = ["gpu", "cpu"]


def _device(device):
    if device == "gpu" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    return device


@pytest.fixture
def plans(monkeypatch):
    """Every ``_DensePlan`` the fits in a test build, in order."""
    built = []

    class Recorded(logistic._DensePlan):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(logistic, "_DensePlan", Recorded)
    return built


def _solve(X, y, n_classes, **config):
    return solve_logistic(X, y, n_classes=n_classes, config=LogisticConfig(**config))


# ---------------------------------------------------------------------- the gate


@pytest.mark.parametrize(
    "law, n, d, K, reason",
    [
        ("ar99", 20000, 16, 2, "correlated"),
        ("lowrank5", 20000, 32, 3, "correlated"),
        ("onehot", 20000, 64, 2, "correlated"),
        ("colscale", 20000, 16, 2, "correlated"),  # scales the diagonal leaves apart
        ("random", 20000, 32, 5, "correlated"),  # correlated features, no signal
        ("breast_cancer", 569, 30, 2, "correlated"),
        ("digits", 1797, 64, 3, "correlated"),
        ("iso", 20000, 32, 2, "uncorrelated"),
        ("blobs", 20000, 16, 3, "uncorrelated"),  # separable, but nothing to precondition
        ("blobsar", 20000, 16, 3, "separated"),
        ("digits", 1797, 64, 10, "separated"),
        ("digits", 20000, 64, 10, "separated"),
    ],
)
def test_the_gate_decides_by_data_law(law, n, d, K, reason, plans):
    X, y = logistic_data(law, n, d, K, seed=3)
    solution = _solve(X, y, K, max_iter=1, device="cpu")

    assert solution.preconditioner_reason == reason, plans[0].stats
    assert solution.preconditioner == ("dense" if reason == "correlated" else "diagonal")
    stats = plans[0].stats
    if reason == "uncorrelated":
        assert stats["correlation"] <= logistic._GATE_CORRELATION
        assert stats["condition"] <= logistic._GATE_CONDITION
        assert not stats["spread"] > logistic._GATE_SPREAD
        assert "separation" not in stats  # no model sample was taken
    elif reason == "separated":
        assert stats["separation"] >= logistic._SEPARATION_LIMIT
    else:
        assert stats["separation"] < logistic._SEPARATION_LIMIT


@pytest.mark.parametrize("lam", [2.0, 3.5, 8.0, 16.0])
def test_the_separation_is_the_largest_eigenvalue_of_within_over_between(lam, plans):
    """Two classes, means ``±Δ/2``, AR(0.9) noise ``W``: ``λ = ΔᵀW⁻¹Δ/4``, set exactly.

    The model reads it from 8 192 of 40 000 rows, against its own curvature, at
    ``C = inf`` so that no penalty discounts it.
    """
    rng = np.random.default_rng(0)
    n, d = 40000, 16
    lags = np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    W = 0.9**lags
    direction = rng.normal(size=d)
    delta = direction * math.sqrt(4.0 * lam / (direction @ np.linalg.solve(W, direction)))
    y = rng.integers(0, 2, size=n)
    noise = rng.multivariate_normal(np.zeros(d), W, size=n)
    X = noise + np.where(y[:, None] == 1, 0.5, -0.5) * delta

    _solve(X, y, 2, C=math.inf, max_iter=1, device="cpu")

    assert plans[0].stats["separation"] == pytest.approx(lam, rel=0.1)
    expected = "separated" if lam >= logistic._SEPARATION_LIMIT else "correlated"
    assert plans[0].reason == expected


def test_a_strong_penalty_reads_as_less_separable(plans):
    """The penalty enters through the model's curvature: a fit that cannot become
    confident is not turned away for the separation its data would allow (ten-class
    digits: λ ≈ 8 at C = 1, 5.4 at 1e-4, 1.9 at 1e-5)."""
    X, y = logistic_data("digits", 20000, 64, 10, seed=3)
    assert _solve(X, y, 10, C=1.0, max_iter=1, device="cpu").preconditioner_reason == "separated"
    assert _solve(X, y, 10, C=1e-5, max_iter=1, device="cpu").preconditioner_reason == "correlated"
    assert plans[1].stats["separation"] < plans[0].stats["separation"] / 2


@pytest.mark.parametrize("d", [1, 1025])
def test_widths_the_model_is_not_built_for_take_the_diagonal(d, plans):
    X, y = logistic_data("ar90", 2048, d, 2, seed=1)
    solution = _solve(X, y, 2, max_iter=2, device="cpu")
    assert (solution.preconditioner, solution.preconditioner_reason) == ("diagonal", "width")
    assert not plans


def test_a_non_finite_sample_is_declined_not_raised():
    X = np.ones((600, 4), dtype=np.float32)
    X[::7, 2] = np.inf
    plan = logistic._DensePlan(
        X, np.arange(600) % 2, None, n_classes=2, fit_intercept=True, l2=1e-3,
        diagonal=np.ones((1, 5)),
    )
    plan.run()
    assert plan.reason == "non-finite" and plan.resolve() is None


@pytest.mark.parametrize("n", [1, 2, 3, 40])
def test_tiny_and_degenerate_samples_do_not_break_the_plan(n):
    """A handful of rows, a constant column, a column equal to another: the plan
    decides something, and the fit runs."""
    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, 5))
    X[:, 1] = 3.0
    X[:, 4] = X[:, 0]
    y = np.arange(n) % 2 if n > 1 else np.array([1])
    solution = _solve(X, y, 2, device="cpu")
    assert solution.preconditioner_reason in {
        "correlated", "uncorrelated", "separated", "no-step"
    }
    assert np.all(np.isfinite(solution.coef))


def test_null_directions_get_neutral_curvature(plans):
    """Three one-hot blocks beside an intercept are three exact dependencies. Without a
    penalty the model's curvature along them is only its ridge, and its inverse would
    amplify the gradient's rounding there by 1/ridge (‖coef‖ 1 260 against 16.7 when
    it did); they get the model's mean curvature instead."""
    X, y = logistic_data("onehot", 10000, 68, 2, seed=5)
    dense = _solve(X, y, 2, C=math.inf, tol=1e-6, device="cpu")
    diagonal = _solve(X, y, 2, C=math.inf, tol=1e-6, device="cpu", dense_preconditioner=False)

    assert dense.preconditioner == "dense" and plans[0].stats["null_directions"] == 3
    assert np.linalg.norm(dense.coef) <= 1.5 * np.linalg.norm(diagonal.coef)
    assert dense.objective <= diagonal.objective * (1 + 1e-6)


# ----------------------------------------------------- the switch, bit for bit


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("decline", ["switch-off", "gate", "model"])
def test_a_declined_plan_is_the_diagonal_path_bit_for_bit(decline, device, monkeypatch):
    """The overlap, the sample and the gate change nothing a declined fit computes:
    whether the switch is off, the gate turns the fit away or the model is refused,
    every bit of the answer and every count is the diagonal path's."""
    device = _device(device)
    X, y = logistic_data("ar99", 20000, 24, 3, seed=11)
    off = _solve(X, y, 3, device=device, dense_preconditioner=False)
    if decline == "gate":
        monkeypatch.setattr(logistic._DensePlan, "_gate", lambda self: False)
    elif decline == "model":
        monkeypatch.setattr(logistic._DensePlan, "_model", lambda self: None)
    declined = _solve(X, y, 3, device=device, dense_preconditioner=decline != "switch-off")

    assert declined.preconditioner == off.preconditioner == "diagonal"
    np.testing.assert_array_equal(declined.coef, off.coef)
    np.testing.assert_array_equal(declined.intercept, off.intercept)
    assert (declined.n_iter, declined.n_fev, declined.objective, declined.grad_max_abs) == (
        off.n_iter, off.n_fev, off.objective, off.grad_max_abs
    )


def test_the_switch_off_takes_no_sample(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a plan was built with the switch off")

    monkeypatch.setattr(logistic, "_DensePlan", forbidden)
    monkeypatch.setattr(logistic, "_sample_moments", forbidden)
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    solution = _solve(X, y, 2, device="cpu", dense_preconditioner=False)
    assert solution.preconditioner_reason == "disabled"


@pytest.mark.parametrize("device", DEVICES)
def test_a_preconditioned_fit_is_deterministic(device):
    device = _device(device)
    X, y = logistic_data("lowrank20", 30000, 48, 4, seed=4)
    first, second = (_solve(X, y, 4, device=device, tol=1e-6) for _ in range(2))
    assert first.preconditioner == "dense"
    np.testing.assert_array_equal(first.coef, second.coef)
    np.testing.assert_array_equal(first.intercept, second.intercept)
    assert (first.n_fev, first.objective) == (second.n_fev, second.objective)


@pytest.mark.parametrize(
    "config, reason",
    [({"tol": 1e9}, "no-step"), ({"max_iter": 0}, "no-step")],
    ids=["optimal-start", "no-budget"],
)
def test_a_fit_that_takes_no_step_reports_the_diagonal(config, reason):
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    solution = _solve(X, y, 2, device="cpu", **config)
    assert (solution.n_iter, solution.preconditioner) == (0, "diagonal")
    assert solution.preconditioner_reason == reason


# ----------------------------------------------------- overlap and BLAS threads


class _RecordingMLX:
    def __init__(self, log):
        self.log = log

    def async_eval(self, *arrays):
        self.log.append("submit")

    def eval(self, *arrays):
        self.log.append("wait")


def test_the_host_work_runs_between_submitting_and_waiting_for_the_first_evaluation():
    log = []
    first = logistic._FirstEvaluation(lambda: log.append("host"))
    mx = _RecordingMLX(log)

    logistic._evaluate(mx, ("value", "grad"), first)
    logistic._evaluate(mx, ("value", "grad"), first)
    logistic._evaluate(mx, ("value", "grad"), None)

    assert log == ["submit", "host", "wait", "wait", "wait"]


def test_the_plan_runs_once_per_fit_before_it_is_resolved(monkeypatch):
    calls = []
    run, resolve = logistic._DensePlan.run, logistic._DensePlan.resolve
    monkeypatch.setattr(logistic._DensePlan, "run", lambda self: (calls.append("run"), run(self))[1])
    monkeypatch.setattr(
        logistic._DensePlan, "resolve", lambda self: (calls.append("resolve"), resolve(self))[1]
    )
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    assert _solve(X, y, 2, device="cpu").preconditioner == "dense"
    assert calls == ["run", "resolve"]


class _FakePools:
    """threadpoolctl's ``limit`` on a controller with a BLAS pool, recorded."""

    def __init__(self, log):
        self.log = log

    def limit(self, *, limits):
        log = self.log

        class Limit:
            def __enter__(self):
                assert logistic._BLAS_LIMIT_LOCK.locked()
                log.append(("limit", limits))

            def __exit__(self, *exc):
                log.append("restore")

        return Limit()


def test_the_plan_runs_on_one_blas_thread_under_the_lock(monkeypatch):
    log = []
    monkeypatch.setattr(logistic, "_blas_pools", lambda: _FakePools(log))
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    _solve(X, y, 2, device="cpu")
    assert log == [("limit", 1), "restore"]
    assert not logistic._BLAS_LIMIT_LOCK.locked()


def test_without_a_blas_pool_there_is_nothing_to_limit(monkeypatch):
    monkeypatch.setattr(logistic, "_blas_pools", lambda: None)
    with logistic._one_blas_thread():
        assert not logistic._BLAS_LIMIT_LOCK.locked()


def test_concurrent_fits_leave_the_blas_thread_count_as_they_found_it():
    """The limit is process-wide BLAS state; fits on several threads must not save and
    restore it out of order and leave the process on one thread. A no-op where the
    BLAS has no pool (Accelerate), and the test that matters on an OpenBLAS stack."""
    from threadpoolctl import threadpool_info

    before = [(lib["filepath"], lib["num_threads"]) for lib in threadpool_info()
              if lib["user_api"] == "blas"]
    X, y = logistic_data("ar99", 8000, 32, 3, seed=7)
    errors = []

    def fit():
        try:
            for _ in range(3):
                assert _solve(X, y, 3, device="cpu").preconditioner == "dense"
        except BaseException as exc:  # reported below, by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=fit, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    join_or_fail(threads)
    assert not errors, errors
    after = [(lib["filepath"], lib["num_threads"]) for lib in threadpool_info()
             if lib["user_api"] == "blas"]
    assert after == before


# ------------------------------------------------------- the estimator's record


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_the_fit_event_records_the_preconditioner_and_the_evaluations(backend):
    if backend == "mlx" and not gpu_available():
        pytest.skip("no usable MLX GPU")
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    with tuning_context(force_backend=backend):
        model = LogisticRegression().fit(X, y)
    events = [e for e in mlxlearn.get_backend_diagnostics() if e.operation == "fit"]
    assert events == [model.fit_event_]
    extra = model.fit_event_.extra
    assert (extra["preconditioner"], extra["preconditioner_reason"]) == ("dense", "correlated")
    assert extra["n_fev"] >= model.n_iter_[0] + 1
    assert "stop_reason" not in extra  # converged: nothing else to record


def test_the_patched_estimator_records_it_too():
    X, y = logistic_data("iso", 5000, 16, 3, seed=2)
    with tuning_context(force_backend="cpu"):
        model = PatchedLogisticRegression().fit(X, y)
    assert model.fit_event_.extra["preconditioner_reason"] == "uncorrelated"
    assert model.fit_event_.extra["preconditioner"] == "diagonal"


def test_tuning_turns_it_off_for_the_estimator():
    X, y = logistic_data("ar99", 5000, 16, 2, seed=2)
    with tuning_context(force_backend="cpu", lbfgs_dense_preconditioner=False):
        off = LogisticRegression().fit(X, y)
    with tuning_context(force_backend="cpu"):
        on = LogisticRegression().fit(X, y)
    assert off.fit_event_.extra["preconditioner_reason"] == "disabled"
    assert on.fit_event_.extra["n_fev"] < off.fit_event_.extra["n_fev"]


@pytest.mark.parametrize("raw, value", [("0", False), ("off", False), ("1", True), ("TRUE", True)])
def test_the_switch_reads_the_environment(monkeypatch, raw, value):
    monkeypatch.setenv("MLXLEARN_LBFGS_DENSE_PRECONDITIONER", raw)
    assert _tuning_from_env(Tuning()).lbfgs_dense_preconditioner is value


def test_a_malformed_switch_is_an_error(monkeypatch):
    monkeypatch.setenv("MLXLEARN_LBFGS_DENSE_PRECONDITIONER", "sometimes")
    with pytest.raises(ValueError, match="MLXLEARN_LBFGS_DENSE_PRECONDITIONER"):
        _tuning_from_env(Tuning())


@pytest.mark.parametrize("value", [1, "yes", None])
def test_the_switch_is_a_bool(value):
    with pytest.raises(TypeError, match="lbfgs_dense_preconditioner must be a bool"):
        Tuning(lbfgs_dense_preconditioner=value)


def test_the_switch_is_on_by_default():
    assert Tuning().lbfgs_dense_preconditioner is True
    assert LogisticConfig().dense_preconditioner is True
    assert Tuning(lbfgs_dense_preconditioner=np.bool_(False)).lbfgs_dense_preconditioner is False
