"""Complete fresh-fit comparison: previous fused MLX, native Metal, and sklearn.

    python -m benchmarks.logistic_native --dtype both --profile full \
        --output bench/logistic-native.json --report bench/logistic-native.md

Both MLX variants force the GPU backend. The prior variant disables only the
native solver selection in this standalone process; validation, preprocessing,
and the previous fused objective/optimizer remain shared. Mutable instrumentation
is restored after every fit. Do not run this benchmark in a threaded host.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import math
import subprocess
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers import logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module

from .logistic_gpu import (
    PROFILES as GPU_PROFILES,
)
from .logistic_gpu import (
    Case,
    make_data,
    quality,
    record_environment,
    summarize,
)

VARIANTS = ("prior_fused", "native_gpu", "sklearn")
PROFILES = {
    "smoke": GPU_PROFILES["smoke"],
    "original": GPU_PROFILES["full"],
    "full": GPU_PROFILES["full"]
    + (
        Case("binary_boundary", 20_000, 512),
        Case("binary_tall_crossover", 250_000, 32),
        Case("binary_tall", 500_000, 32),
        Case("binary_tall_wide", 500_000, 128),
    ),
}


def state_metadata(state):
    return {
        "n_iter": state.n_iter,
        "n_fev": getattr(state, "n_fev", None),
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "grad_max_abs": state.grad_max_abs,
        "objective": state.objective,
    }


@contextlib.contextmanager
def selected_solver(variant, native_module):
    """Observe the actual implementation and disable native only for baseline."""
    original_select = logistic._use_native_solver
    original_solve = estimator_module.solve_logistic
    original_fused = logistic.minimize
    original_native = native_module.minimize_binary
    records = {"selection": [], "solver_calls": [], "solutions": []}

    def selected(config, n_targets, n_features):
        enabled = (
            False if variant == "prior_fused" else original_select(config, n_targets, n_features)
        )
        records["selection"].append(
            {
                "native_selected": bool(enabled),
                "device": config.device,
                "n_targets": n_targets,
                "n_features": n_features,
            }
        )
        return enabled

    def recorded_solve(*args, **kwargs):
        solution = original_solve(*args, **kwargs)
        records["solutions"].append(state_metadata(solution))
        return solution

    def recorded_fused(*args, **kwargs):
        state = original_fused(*args, **kwargs)
        records["solver_calls"].append({"implementation": "fused_mlx", **state_metadata(state)})
        return state

    def recorded_native(*args, **kwargs):
        state = original_native(*args, **kwargs)
        records["solver_calls"].append({"implementation": "native_metal", **state_metadata(state)})
        return state

    logistic._use_native_solver = selected
    estimator_module.solve_logistic = recorded_solve
    logistic.minimize = recorded_fused
    native_module.minimize_binary = recorded_native
    try:
        yield records
    finally:
        logistic._use_native_solver = original_select
        estimator_module.solve_logistic = original_solve
        logistic.minimize = original_fused
        native_module.minimize_binary = original_native


def fit_once(variant, X, y, weights, *, max_iter, tol, native_module):
    cls = SkLogisticRegression if variant == "sklearn" else LogisticRegression
    forced = "" if variant == "sklearn" else "mlx"
    with (
        config_context(device="auto"),
        tuning_context(force_backend=forced),
        selected_solver(variant, native_module) as records,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        mx.synchronize()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            model = cls(C=1.0, tol=tol, max_iter=max_iter).fit(X, y, sample_weight=weights)
            mx.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            if gc_was_enabled:
                gc.enable()
    backend = getattr(model, "_execution_backend_", "sklearn")
    if forced:
        assert backend == "mlx", (variant, backend)
        assert len(records["solutions"]) == 1, (variant, records)
        assert len(records["solver_calls"]) >= 1, (variant, records)
        if variant == "prior_fused":
            assert all(
                state["implementation"] == "fused_mlx" for state in records["solver_calls"]
            ), records
    elif records["solutions"] or records["solver_calls"]:
        raise RuntimeError("Stock sklearn unexpectedly called an mlxlearn solver.")
    metadata = {
        "backend": backend,
        "implementation": (
            records["solver_calls"][-1]["implementation"] if records["solver_calls"] else "sklearn"
        ),
        "n_iter": model.n_iter_.tolist(),
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ],
        **records,
    }
    return elapsed, model, metadata


def run_case(case, *, seed, dtype, warmup, repeats, max_iter, tol, native_module):
    X, y, weights = make_data(case, seed, dtype=dtype)
    samples = {variant: [] for variant in VARIANTS}
    details = {variant: [] for variant in VARIANTS}
    first = {}
    models = {}
    for index in range(warmup):
        for variant in VARIANTS:
            elapsed, _, metadata = fit_once(
                variant, X, y, weights, max_iter=max_iter, tol=tol, native_module=native_module
            )
            if index == 0:
                first[variant] = {"seconds": elapsed, **metadata}
    orders = []
    for repeat in range(repeats):
        base = VARIANTS if repeat % 2 == 0 else VARIANTS[::-1]
        offset = (repeat // 2) % len(VARIANTS)
        order = base[offset:] + base[:offset]
        orders.append(order)
        for variant in order:
            elapsed, model, metadata = fit_once(
                variant, X, y, weights, max_iter=max_iter, tol=tol, native_module=native_module
            )
            samples[variant].append(elapsed)
            details[variant].append(metadata)
            models[variant] = model
    timings = {variant: summarize(values) for variant, values in samples.items()}
    scores = {
        name: quality(model, models["sklearn"], X, y, weights) for name, model in models.items()
    }
    for score in scores.values():
        score["float64_gradient_within_requested_tol"] = score["gradient_max_abs_float64"] <= tol
    return {
        "case": asdict(case),
        "seed": seed,
        "dtype": str(X.dtype),
        "parameters": {"C": 1.0, "tol": tol, "max_iter": max_iter, "fit_intercept": True},
        "first_case_calls_not_process_cold": first,
        "timed_orders": orders,
        "timings": timings,
        "fit_details": details,
        "quality_last_fit": scores,
        "native_speedup_over_prior_fused": timings["prior_fused"]["median_ms"]
        / timings["native_gpu"]["median_ms"],
        "native_speedup_over_sklearn": timings["sklearn"]["median_ms"]
        / timings["native_gpu"]["median_ms"],
    }


def write_report(path, payload):
    lines = [
        "# Native Metal logistic regression benchmark",
        "",
        f"Measured at {payload['measured_at_utc']}; {payload['repeats']} measured fresh fits "
        f"after {payload['warmup']} warmup fit(s) per variant, case and dtype.",
        "",
        "Complete estimator construction and fit, NumPy input/output, identical datasets and "
        "parameters. Both mlxlearn variants force GPU execution. The previous fused baseline "
        "disables only native solver selection; it shares current validation and preprocessing. "
        "Stock scikit-learn is unpatched. Compilation/warmup calls are reported separately and "
        "are not process-cold measurements. Every fit recomputes preprocessing and optimization; "
        "no converted arrays, fitted parameters or solver state are cached between fits.",
        "",
        "Multiclass uses the existing fused solver in both mlxlearn variants. The implementation "
        "column records which optimizer actually ran. Forced-GPU measurements do not describe "
        "the library's automatic CPU/GPU crossover selection.",
        "",
        "| Input | Case | Previous fused ms | New library ms | sklearn ms | Actual implementation | "
        "vs previous | vs sklearn |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for result in payload["results"]:
        t = result["timings"]
        case = result["case"]
        implementation = result["fit_details"]["native_gpu"][-1]["implementation"]
        lines.append(
            f"| {result['dtype']} | {case['name']} ({case['n_samples']:,} × "
            f"{case['n_features']:,}) | {t['prior_fused']['median_ms']:.3f} | "
            f"{t['native_gpu']['median_ms']:.3f} | {t['sklearn']['median_ms']:.3f} | "
            f"{implementation} | {result['native_speedup_over_prior_fused']:.2f}× | "
            f"{result['native_speedup_over_sklearn']:.2f}× |"
        )
    lines += [
        "",
        "Ratios use warmed medians. Raw samples, minimum/maximum/IQR, first calls, rotated "
        "execution orders and every fit's diagnostics are in the companion JSON. "
        "Labels are generated before the public dtype cast, so both panels use the same targets.",
        "",
        "| Input / case / variant | Iterations | Evaluations | Stop reason | Float64 objective | "
        "Relative objective gap vs sklearn | Float64 max gradient | Prediction agreement |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        for variant, score in result["quality_last_fit"].items():
            detail = result["fit_details"][variant][-1]
            state = detail["solver_calls"][-1] if detail["solver_calls"] else None
            reason = state["stop_reason"] if state else "sklearn (see JSON warnings)"
            evaluations = state["n_fev"] if state else "—"
            lines.append(
                f"| {result['dtype']} / {result['case']['name']} / {variant} | "
                f"{detail['n_iter'][0]} | {evaluations} | {reason} | "
                f"{score['objective_float64']:.10g} | "
                f"{score['objective_relative_gap_to_sklearn']:.3g} | "
                f"{score['gradient_max_abs_float64']:.3g} | "
                f"{score['prediction_agreement_sklearn']:.5f} |"
            )
    env = payload["environment"]
    lines += [
        "",
        "Objective and gradient are evaluated independently in float64 outside the timer. "
        "The JSON records whether each raw-feature gradient meets the requested tolerance; "
        "this can differ from a centered optimizer's gradient or sklearn's function-value "
        "stopping criterion.",
        "",
        f"Environment: {env.get('cpu', env['machine'])}; {env['platform']}; "
        f"MLX {env['mlx']}; sklearn {env['sklearn']}; NumPy {env['numpy']}; SciPy {env['scipy']}.",
        "",
        "Source hashes, thread-pool information, NumPy configuration, library tuning, and Git "
        "revision are saved in JSON. threadpoolctl can limit only recognized thread pools; "
        "Apple Accelerate may not be controlled by that setting.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--dtype", choices=("float32", "float64", "both"), default="both")
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--output", type=Path, default=Path("bench/logistic-native.json"))
    parser.add_argument("--report", type=Path, default=Path("bench/logistic-native.md"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least three measured repetitions and one warmup")
    if args.max_iter < 1 or args.tol < 0 or not math.isfinite(args.tol):
        parser.error("max-iter must be positive and tol must be finite and nonnegative")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if not hasattr(logistic, "_use_native_solver"):
        parser.error("install the native GPU implementation before running this benchmark")
    native_module = importlib.import_module("mlxlearn._solvers.native_logistic")
    cases = PROFILES[args.profile]
    if args.case:
        unknown = set(args.case) - {case.name for case in cases}
        if unknown:
            parser.error(f"cases not in selected profile: {sorted(unknown)}")
        cases = tuple(case for case in cases if case.name in args.case)
    dtypes = ("float64", "float32") if args.dtype == "both" else (args.dtype,)
    payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "requested_thread_limit": args.threads,
        "variants": VARIANTS,
        "input_dtypes": dtypes,
        "baseline": {
            "implementation": "previous fused MLX objective and L-BFGS",
            "selector_override": "logistic._use_native_solver returns False",
            "shared_boundary": "current estimator validation and solver preprocessing",
        },
        "results": [],
    }
    with threadpool_limits(limits=args.threads):
        payload["environment"] = record_environment()
        package_root = Path(logistic.__file__).resolve().parents[1]
        paths = (
            Path(native_module.__file__),
            Path(__file__),
            Path(estimator_module.__file__),
            package_root / "_common" / "validation.py",
            package_root / "_common" / "config.py",
            package_root / "_common" / "device.py",
        )
        payload["environment"]["source_sha256"].update(
            {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        )
        diff = subprocess.check_output(["git", "diff", "--", "src", "benchmarks"])
        payload["environment"]["tracked_working_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        for dtype in dtypes:
            for case in cases:
                print(
                    f"Running {dtype} / {case.name}: {case.n_samples:,} × {case.n_features:,}",
                    flush=True,
                )
                result = run_case(
                    case,
                    seed=args.seed,
                    dtype=dtype,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    native_module=native_module,
                )
                payload["results"].append(result)
                print(
                    {
                        name: round(timing["median_ms"], 3)
                        for name, timing in result["timings"].items()
                    },
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                write_report(args.report, payload)
    print(f"Saved {args.output} and {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
