"""Reproduce the October 2026 investigation, without changing library code.

Run the solver and KNN probes:
    python -m benchmarks.investigate_performance --output bench/investigation.json
Run full estimator fits, hybrid optimization, and the SVC stage profile:
    python -m benchmarks.investigate_performance --section followup \
        --output bench/investigation-forced.json
Experimental replacements are local to this process and restored after fitting.
They are performance probes, not supported solver implementations. Run this
script by itself: its temporary function replacements are not thread-safe.
The label "current" selects the legacy optimizer with the current objective.
Historical JSON captures the objective before fusion. For an exact source-frozen
comparison against the original complete implementation, use logistic_gpu.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import subprocess
import time
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
from sklearn.svm import SVC as SkSVC
from threadpoolctl import threadpool_info, threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers import lbfgs, logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.neighbors import NearestNeighbors
from mlxlearn.svm import SVC
from mlxlearn.svm import _classes as svc_module

from .logistic_gpu import legacy_minimize


@contextlib.contextmanager
def legacy_solver():
    original = logistic.minimize
    logistic.minimize = legacy_minimize
    try:
        yield
    finally:
        logistic.minimize = original


def device_two_loop(mx, grad, s_history, y_history, rho_history):
    """Same two-loop equations, with device scalars instead of Python floats."""
    if not s_history:
        return -grad
    q = grad
    alphas = []
    for s, y, rho in zip(
        reversed(s_history), reversed(y_history), reversed(rho_history), strict=True
    ):
        alpha = rho * mx.sum(s * q)
        alphas.append(alpha)
        q = q - alpha * y
    gamma = 1.0 / (rho_history[-1] * mx.maximum(mx.sum(y_history[-1] ** 2), 1e-30))
    r = gamma * q
    for index, (s, y, rho) in enumerate(zip(s_history, y_history, rho_history, strict=True)):
        beta = rho * mx.sum(y * r)
        r = r + s * (alphas[-1 - index] - beta)
    return -r


@contextlib.contextmanager
def recursion(prototype):
    original = lbfgs._two_loop
    if prototype:
        lbfgs._two_loop = device_two_loop
    try:
        yield
    finally:
        lbfgs._two_loop = original


def timed(fn):
    mx.synchronize()
    start = time.perf_counter()
    result = fn()
    mx.synchronize()
    return time.perf_counter() - start, result


def summary(samples):
    return {
        "median_ms": statistics.median(samples) * 1000,
        "min_ms": min(samples) * 1000,
        "samples_s": samples,
    }


def logistic_cases(repeats):
    output = []
    for n, d in [(5000, 32), (20000, 128), (20000, 1024)]:
        rng = np.random.default_rng(41)
        X = rng.normal(size=(n, d))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        cfg = logistic.LogisticConfig(device="gpu", max_iter=200)

        def solve(prototype, X=X, y=y, cfg=cfg):
            with legacy_solver(), recursion(prototype):
                return logistic.solve_logistic(X, y, n_classes=2, config=cfg)

        fns = {
            "current": lambda: solve(False),
            "device_two_loop": lambda: solve(True),
            "sklearn": lambda X=X, y=y: SkLogisticRegression(max_iter=200).fit(X, y),
        }
        samples = {name: [] for name in fns}
        models = {}
        cold = {}
        for name, fn in fns.items():
            cold[name], _ = timed(fn)  # first call for this case, not process cold
        for rep in range(repeats):
            names = list(fns) if rep % 2 == 0 else list(reversed(fns))
            for name in names:
                elapsed, models[name] = timed(fns[name])
                samples[name].append(elapsed)
        result = {
            "shape": [n, d],
            "input_dtype": str(X.dtype),
            "first_case_calls_s": cold,
            "timings": {name: summary(s) for name, s in samples.items()},
        }
        quality = {}
        for name, model in models.items():
            if name == "sklearn":
                coef, intercept, n_iter = model.coef_, model.intercept_, int(model.n_iter_[0])
                stop, grad = None, None
            else:
                coef, intercept, n_iter = model.coef, model.intercept, model.n_iter
                stop, grad = model.stop_reason, model.grad_max_abs
            quality[name] = {
                "objective_float64": logistic.reference_objective(
                    coef, intercept, X, y, n_classes=2
                ),
                "n_iter": n_iter,
                "stop_reason": stop,
                "grad_max_abs_solver": grad,
                "prediction_agreement_sklearn": float(
                    np.mean(((X @ coef.T + intercept).ravel() > 0) == models["sklearn"].predict(X))
                ),
            }
        result["quality"] = quality
        output.append(result)
        print(
            "logistic",
            [n, d],
            {k: round(v["median_ms"], 3) for k, v in result["timings"].items()},
            quality,
            flush=True,
        )
    return output


def profile_logistic():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(20000, 128))
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    stats = {}
    originals = {}
    for name in ("_dot", "_max_abs", "_two_loop"):
        original = getattr(lbfgs, name)
        originals[name] = original
        stats[name] = {"calls": 0, "seconds_inclusive": 0.0}

        def wrapped(*args, _name=name, _fn=original):
            start = time.perf_counter()
            value = _fn(*args)
            stats[_name]["seconds_inclusive"] += time.perf_counter() - start
            stats[_name]["calls"] += 1
            return value

        setattr(lbfgs, name, wrapped)
    try:
        with legacy_solver():
            elapsed, result = timed(
                lambda: logistic.solve_logistic(
                    X, y, n_classes=2, config=logistic.LogisticConfig(device="gpu", max_iter=200)
                )
            )
    finally:
        for name, original in originals.items():
            setattr(lbfgs, name, original)
    return {
        "shape": list(X.shape),
        "fit_s": elapsed,
        "n_iter": result.n_iter,
        "nested_timings_do_not_sum": stats,
    }


def hybrid_minimize(fun, x0, config):
    """CPU L-BFGS-B direction/history, existing stable MLX loss and gradient."""
    from scipy.optimize import minimize

    shape = x0.shape

    def objective(x):
        value, gradient = fun(mx.array(x.reshape(shape), dtype=mx.float32))
        return value, np.asarray(gradient).astype(np.float64).ravel()

    result = minimize(
        objective,
        np.asarray(x0).astype(np.float64).ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": config.max_iter,
            "gtol": config.tol,
            "ftol": 0.0,
            "maxcor": config.history_size,
            "maxls": config.max_line_search,
        },
    )
    grad_max = float(np.abs(result.jac).max())
    return lbfgs.LBFGSState(
        x=mx.array(result.x.reshape(shape), dtype=mx.float32),
        objective=float(result.fun),
        grad_max_abs=grad_max,
        n_iter=int(result.nit),
        n_fev=int(result.nfev),
        converged=grad_max <= config.tol,
        stop_reason="gtol" if grad_max <= config.tol else str(result.message),
    )


def estimator_cases(repeats):
    output = []
    for n, d in [(5000, 32), (20000, 128), (20000, 1024)]:
        rng = np.random.default_rng(41)
        X = rng.normal(size=(n, d))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)

        def fit(variant, X=X, y=y):
            if variant == "sklearn":
                return SkLogisticRegression(max_iter=200).fit(X, y)
            old = logistic.minimize
            if variant == "hybrid":
                logistic.minimize = hybrid_minimize
            else:
                logistic.minimize = legacy_minimize
            try:
                with (
                    config_context(device="gpu"),
                    tuning_context(force_backend="mlx"),
                    recursion(variant == "device_two_loop"),
                ):
                    model = LogisticRegression(max_iter=200).fit(X, y)
                    assert model._execution_backend_ == "mlx"
                    return model
            finally:
                logistic.minimize = old

        names = ["current", "device_two_loop", "hybrid", "sklearn"]
        samples = {name: [] for name in names}
        models = {}
        with warnings.catch_warnings(record=True) as caught:
            for name in names:
                timed(lambda name=name: fit(name))
            for rep in range(repeats):
                for name in names if rep % 2 == 0 else list(reversed(names)):
                    elapsed, models[name] = timed(lambda name=name: fit(name))
                    samples[name].append(elapsed)
        result = {
            "shape": [n, d],
            "timings": {k: summary(v) for k, v in samples.items()},
            "warnings": [str(w.message) for w in caught],
            "quality": {},
        }
        baseline_pred = models["sklearn"].predict(X)
        for name, model in models.items():
            result["quality"][name] = {
                "n_iter": model.n_iter_.tolist(),
                "backend": getattr(model, "_execution_backend_", "sklearn"),
                "objective_float64": logistic.reference_objective(
                    model.coef_, model.intercept_, X, y, n_classes=2
                ),
                "prediction_agreement_sklearn": float(np.mean(model.predict(X) == baseline_pred)),
            }
        output.append(result)
        print(
            "estimator",
            [n, d],
            {k: round(v["median_ms"], 3) for k, v in result["timings"].items()},
            result["quality"],
            flush=True,
        )
    return output


def svc_cases(repeats):
    output = []
    for d in (32, 256):
        rng = np.random.default_rng(43)
        X = rng.normal(size=(4000, d))
        y = (X[:, 0] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
        samples = {"mlx": [], "sklearn": []}
        with config_context(device="gpu"), tuning_context(force_backend="mlx"):
            fns = {
                "mlx": lambda X=X, y=y: SVC().fit(X, y),
                "sklearn": lambda X=X, y=y: SkSVC().fit(X, y),
            }
            models = {}
            for fn in fns.values():
                timed(fn)
            for rep in range(repeats):
                names = list(fns) if rep % 2 == 0 else list(reversed(fns))
                for name in names:
                    elapsed, models[name] = timed(fns[name])
                    samples[name].append(elapsed)
            original_gram, original_solver = svc_module.kernel_matrix, svc_module.solve_c_svc
            stages = {}

            def gram(*args, _fn=original_gram, _stages=stages, **kwargs):
                elapsed, value = timed(lambda: _fn(*args, **kwargs))
                _stages["gram_s"] = elapsed
                return value

            def solver(*args, _fn=original_solver, _stages=stages, **kwargs):
                elapsed, value = timed(lambda: _fn(*args, **kwargs))
                _stages["solver_s"] = elapsed
                return value

            svc_module.kernel_matrix, svc_module.solve_c_svc = gram, solver
            try:
                elapsed, _ = timed(fns["mlx"])
                stages["total_profiled_s"] = elapsed
            finally:
                svc_module.kernel_matrix, svc_module.solve_c_svc = original_gram, original_solver
        assert models["mlx"]._execution_backend_ == "mlx"
        result = {
            "shape": list(X.shape),
            "timings": {k: summary(v) for k, v in samples.items()},
            "profile": stages,
            "mlx_backend": models["mlx"]._execution_backend_,
            "iterations": {k: m.n_iter_.tolist() for k, m in models.items()},
            "fit_status": {k: m.fit_status_ for k, m in models.items()},
            "mlx_solver_kkt": models["mlx"]._pair_kkt_violation_.tolist(),
            "prediction_agreement": float(
                np.mean(models["mlx"].predict(X) == models["sklearn"].predict(X))
            ),
        }
        output.append(result)
        print("svc", result, flush=True)
    return output


def neighbor_cases(repeats):
    output = []
    for n, d in [(8000, 32), (32000, 768)]:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, d))
        Q = rng.normal(size=(256, d))
        with config_context(device="gpu"):
            model = NearestNeighbors(n_neighbors=10).fit(X)
            sk_model = SkNearestNeighbors(n_neighbors=10, algorithm="brute").fit(X)
            for threads in (None, 1, 4):
                with threadpool_limits(limits=threads):
                    fns = {
                        "mlx": lambda model=model, Q=Q: model.kneighbors(Q),
                        "sklearn": lambda model=sk_model, Q=Q: model.kneighbors(Q),
                    }
                    samples = {name: [] for name in fns}
                    for fn in fns.values():
                        timed(fn)
                    models = {}
                    for rep in range(repeats):
                        names = list(fns) if rep % 2 == 0 else list(reversed(fns))
                        for name in names:
                            elapsed, models[name] = timed(fns[name])
                            samples[name].append(elapsed)
                    result = {
                        "shape": [n, d],
                        "queries": 256,
                        "threads": threads,
                        "timings": {k: summary(v) for k, v in samples.items()},
                        "indices_equal": bool(
                            np.array_equal(models["mlx"][1], models["sklearn"][1])
                        ),
                        "max_distance_error": float(
                            np.max(np.abs(models["mlx"][0] - models["sklearn"][0]))
                        ),
                    }
                    output.append(result)
                    print(
                        "neighbors",
                        [n, d],
                        threads,
                        {k: round(v["median_ms"], 3) for k, v in result["timings"].items()},
                        flush=True,
                    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/investigation.json"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--section", choices=("initial", "followup"), default="initial")
    args = parser.parse_args()
    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "device": mx.device_info(),
            "threadpools": threadpool_info(),
            "numpy_config": np.show_config(mode="dicts"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "repeats": args.repeats,
    }
    if args.section == "initial":
        payload["logistic"] = logistic_cases(args.repeats)
        payload["logistic_profile"] = profile_logistic()
        print("profile", payload["logistic_profile"], flush=True)
        payload["neighbors"] = neighbor_cases(args.repeats)
    else:
        payload["logistic_estimators"] = estimator_cases(args.repeats)
        payload["svc"] = svc_cases(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
