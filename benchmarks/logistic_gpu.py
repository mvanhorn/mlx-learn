"""Compare complete fused-GPU, legacy MLX, and sklearn logistic fits.

Run after installing the working tree:
    python -m benchmarks.logistic_gpu --profile full \
        --output bench/logistic-gpu.json --report bench/logistic-gpu.md

The legacy solver is selected only in this standalone benchmark process, by
temporarily replacing the solver function consumed by the estimator. Run this
module by itself; those temporary replacements are not thread-safe. Production
code has no solver selector or mutable benchmark state.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import statistics
import subprocess
import sys
import time
import types
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import get_tuning, tuning_context
from mlxlearn._solvers import lbfgs, logistic, native_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module
from mlxlearn.patching._estimators import LogisticRegression as PatchedLogisticRegression

from ._harness import environment

VARIANTS = ("legacy_mlx", "fused_mlx", "sklearn", "patched_default")
BASELINE_REF = "fb79a1b11dad401e82808885884fc633da9c3e86"


@dataclass(frozen=True)
class Case:
    name: str
    n_samples: int
    n_features: int
    n_classes: int = 2
    weighted: bool = False
    ill_conditioned: bool = False


PROFILES = {
    "smoke": (
        Case("binary_small", 5_000, 32),
        Case("binary_medium", 20_000, 128),
        Case("binary_wide", 20_000, 1_024),
    ),
    "full": (
        Case("binary_small", 5_000, 32),
        Case("binary_medium", 20_000, 128),
        Case("binary_wide", 20_000, 1_024),
        Case("binary_large", 60_000, 1_024),
        Case("multiclass_weighted", 20_000, 128, n_classes=5, weighted=True),
        Case("binary_ill_conditioned", 5_000, 32, ill_conditioned=True),
    ),
}


def make_data(case: Case, seed: int, dtype: str = "float64"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(case.n_samples, case.n_features))
    if case.n_classes == 2:
        y = (X[:, 0] + rng.normal(scale=0.5, size=case.n_samples) > 0).astype(int)
    else:
        coefficients = rng.normal(size=(min(16, case.n_features), case.n_classes))
        logits = X[:, : coefficients.shape[0]] @ coefficients
        y = np.argmax(logits + rng.normal(scale=2.0, size=logits.shape), axis=1)
    if case.ill_conditioned:
        # Correlation and unequal feature scales stress the optimizer without
        # changing the target or introducing undefined/overflowing input.
        midpoint = case.n_features // 2
        X[:, midpoint:] = X[:, :midpoint] + 0.01 * X[:, midpoint:]
        X *= np.logspace(-1, 1, case.n_features)
    weights = rng.uniform(0.25, 3.0, case.n_samples) if case.weighted else None
    # Generate labels before the public input cast so dtype panels describe the
    # same statistical task rather than separate RNG/target realizations.
    return np.ascontiguousarray(X, dtype=dtype), y, weights


def legacy_minimize(fun, x0, config):
    """Execute the unchanged solver from before the GPU-fusion change."""
    return lbfgs.minimize(
        fun,
        x0,
        lbfgs.LBFGSConfig(
            max_iter=config.max_iter,
            tol=config.tol,
            history_size=config.history_size,
            max_line_search=config.max_line_search,
        ),
    )


@contextlib.contextmanager
def legacy_reference(ref: str):
    """Load the original complete loss/solve code, including its unfused objective."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        text=True,
    ).strip()
    relative_path = "src/mlxlearn/_solvers/logistic.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{relative_path}"])
    original_lbfgs = subprocess.check_output(
        ["git", "show", f"{revision}:src/mlxlearn/_solvers/lbfgs.py"]
    )
    if original_lbfgs != Path(lbfgs.__file__).read_bytes():
        raise RuntimeError(
            "The shared legacy lbfgs.py differs from the baseline revision; "
            "benchmarking it would not reproduce the original implementation."
        )
    name = f"mlxlearn._solvers._benchmark_logistic_{revision}"
    module = types.ModuleType(name)
    module.__package__ = "mlxlearn._solvers"
    module.__file__ = f"git:{revision}:{relative_path}"
    previous = sys.modules.get(name)
    sys.modules[name] = module  # dataclasses consult their defining module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        if module.minimize is not lbfgs.minimize:
            raise RuntimeError("The baseline must use the original MLX L-BFGS implementation.")
        yield (
            module,
            {
                "revision": revision,
                "logistic_path": relative_path,
                "logistic_sha256": hashlib.sha256(source).hexdigest(),
                "lbfgs_sha256": hashlib.sha256(original_lbfgs).hexdigest(),
                "boundary": "original solver and objective; shared current estimator validation",
                "device_handoff": (
                    "benchmark adapter drops X_device, X_center, and initial_parts; legacy solver "
                    "uploads and centers the raw input itself after the current estimator's "
                    "device preparation, which remains inside the timed fit"
                ),
            },
        )
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@contextlib.contextmanager
def selected_solver(variant: str, legacy_module):
    solver_module = legacy_module if variant == "legacy_mlx" else logistic
    original = solver_module.minimize
    original_native = native_logistic.minimize_binary
    original_select = logistic._use_native_solver
    original_estimator_solve = estimator_module.solve_logistic
    states = []

    def recorded(fun, x0, config, **kwargs):
        state = original(fun, x0, config, **kwargs)
        states.append(state)
        return state

    solver_module.minimize = recorded
    if variant == "fused_mlx":
        # Preserve this historical comparison after adding the native optimizer.
        logistic._use_native_solver = lambda *args: False

    def recorded_native(*args, **kwargs):
        state = original_native(*args, **kwargs)
        states.append(state)
        return state

    native_logistic.minimize_binary = recorded_native
    if variant == "legacy_mlx":

        def legacy_solve(*args, X_device=None, X_center=None, initial_parts=None, certification_X=None, **kwargs):
            # Keep the historical numerical source byte-for-byte intact. It
            # predates the estimator's optional prepared device-array handoff.
            # The legacy path uploads and centers raw input; the report states
            # that this can duplicate current estimator preparation.
            return legacy_module.solve_logistic(*args, **kwargs)

        estimator_module.solve_logistic = legacy_solve
    try:
        yield states
    finally:
        solver_module.minimize = original
        native_logistic.minimize_binary = original_native
        logistic._use_native_solver = original_select
        estimator_module.solve_logistic = original_estimator_solve


def fit_once(variant, X, y, weights, *, max_iter, tol, legacy_module):
    params = {"max_iter": max_iter, "tol": tol}
    cls = (
        SkLogisticRegression
        if variant == "sklearn"
        else PatchedLogisticRegression
        if variant == "patched_default"
        else LogisticRegression
    )
    forced = "mlx" if variant in ("legacy_mlx", "fused_mlx") else ""
    with (
        config_context(device="auto"),
        tuning_context(force_backend=forced),
        selected_solver(variant, legacy_module) as states,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        mx.synchronize()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            model = cls(**params).fit(X, y, sample_weight=weights)
            mx.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            if gc_was_enabled:
                gc.enable()
    backend = getattr(model, "_execution_backend_", "sklearn")
    if forced:
        assert backend == forced, (variant, backend)
        assert len(states) == 1, (variant, len(states))
    state = states[-1] if states else None
    metadata = {
        "backend": backend,
        "n_iter": model.n_iter_.tolist(),
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ],
        "solver": (
            {
                "n_fev": state.n_fev,
                "converged": state.converged,
                "stop_reason": state.stop_reason,
                "grad_max_abs": state.grad_max_abs,
                "objective": state.objective,
            }
            if state is not None
            else None
        ),
    }
    return elapsed, model, metadata


def summarize(samples):
    q1, q3 = np.percentile(samples, [25, 75])
    return {
        "median_ms": 1_000 * statistics.median(samples),
        "min_ms": 1_000 * min(samples),
        "max_ms": 1_000 * max(samples),
        "iqr_ms": 1_000 * float(q3 - q1),
        "samples_s": samples,
    }


def quality(model, reference, X, y, weights):
    """Evaluate objective and gradient in float64, outside the timed region."""
    codes = np.searchsorted(model.classes_, y)
    n_classes = len(model.classes_)
    W, b = model.coef_, model.intercept_
    sw_sum = len(X) if weights is None else float(weights.sum())
    raw = X @ W.T + b
    if n_classes == 2:
        residual = (expit(raw[:, 0]) - codes)[:, None]
    else:
        residual = softmax(raw, axis=1)
        residual[np.arange(len(X)), codes] -= 1.0
    if weights is not None:
        residual *= weights[:, None]
    grad_w = residual.T @ X / sw_sum + W / sw_sum
    grad_b = residual.sum(axis=0) / sw_sum
    objective = logistic.reference_objective(
        W, b, X, codes, n_classes=n_classes, sample_weight=weights
    )
    reference_value = logistic.reference_objective(
        reference.coef_,
        reference.intercept_,
        X,
        codes,
        n_classes=n_classes,
        sample_weight=weights,
    )
    return {
        "objective_float64": objective,
        "objective_relative_gap_to_sklearn": (objective - reference_value)
        / max(abs(reference_value), np.finfo(float).tiny),
        "gradient_max_abs_float64": float(max(np.abs(grad_w).max(), np.abs(grad_b).max())),
        "prediction_agreement_sklearn": float(np.mean(model.predict(X) == reference.predict(X))),
        "max_coefficient_difference_sklearn": float(np.abs(W - reference.coef_).max()),
    }


def run_case(case, *, seed, warmup, repeats, max_iter, tol, variants, legacy_module, dtype):
    X, y, weights = make_data(case, seed, dtype=dtype)
    samples = {variant: [] for variant in variants}
    details = {variant: [] for variant in variants}
    first = {}
    models = {}
    for index in range(warmup):
        for variant in variants:
            elapsed, _, metadata = fit_once(
                variant, X, y, weights, max_iter=max_iter, tol=tol, legacy_module=legacy_module
            )
            if index == 0:
                first[variant] = {"seconds": elapsed, **metadata}
    orders = []
    for repeat in range(repeats):
        # Rotate the first runner and reverse alternating rounds so the same
        # variant does not systematically pay for position/thermal effects.
        base = variants if repeat % 2 == 0 else variants[::-1]
        offset = (repeat // 2) % len(variants)
        order = list(base[offset:] + base[:offset])
        orders.append(order)
        for variant in order:
            elapsed, model, metadata = fit_once(
                variant, X, y, weights, max_iter=max_iter, tol=tol, legacy_module=legacy_module
            )
            samples[variant].append(elapsed)
            details[variant].append(metadata)
            models[variant] = model
    timings = {variant: summarize(values) for variant, values in samples.items()}
    return {
        "case": asdict(case),
        "seed": seed,
        "dtype": str(X.dtype),
        "parameters": {"C": 1.0, "tol": tol, "max_iter": max_iter, "fit_intercept": True},
        "first_case_calls_not_process_cold": first,
        "timed_orders": orders,
        "timings": timings,
        "fit_details": details,
        "quality_last_fit": {
            name: quality(model, models["sklearn"], X, y, weights) for name, model in models.items()
        },
        "fused_speedup_over_legacy": timings["legacy_mlx"]["median_ms"]
        / timings["fused_mlx"]["median_ms"],
        "fused_speedup_over_sklearn": timings["sklearn"]["median_ms"]
        / timings["fused_mlx"]["median_ms"],
    }


def record_environment():
    env = environment()
    env["threadpools"] = threadpool_info()
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()
    env["numpy_configuration"] = numpy_config.getvalue()
    env["tuning"] = asdict(get_tuning())
    env["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    paths = (
        Path(lbfgs.__file__),
        Path(logistic.__file__),
        Path(logistic.minimize.__code__.co_filename),
        Path(__file__),
    )
    env["source_sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    return env


def write_report(path, payload):
    lines = [
        "# Fused GPU logistic regression benchmark",
        "",
        f"Measured at {payload['measured_at_utc']}; {payload['repeats']} repetitions after "
        f"{payload['warmup']} warmup fit(s) per variant and case.",
        "",
        "Complete fresh estimator fits, NumPy input/output, identical data and parameters. "
        "Legacy and fused paths are forced to MLX. Patched-default timings, when included, "
        "retain production crossover dispatch. First calls are recorded separately and "
        "are not process-cold measurements. Quality scoring is outside the timers.",
        "",
        f"Public input dtype: `{payload['input_dtype']}`. Labels are generated in float64 "
        "before casting features so the float32 and float64 panels share the same task.",
        "",
        f"Legacy numerical code, including the unfused objective, is loaded from revision "
        f"`{payload['baseline']['revision']}`. Both variants share current estimator validation.",
        "Compatibility boundary: the legacy solver ignores X_device, X_center, and initial_parts, "
        "and uploads and centers the raw input itself. Current estimator preparation remains "
        "inside its timed fit, so this comparator can duplicate upload, centering, and initialization. "
        "Use benchmarks.logistic_wide for the authoritative complete "
        "frozen-package before/after comparison of the native optimizers.",
        "",
        "| Case | Legacy MLX ms | Fused MLX ms | sklearn ms | Patched ms / backend | "
        "Fused vs legacy | Fused vs sklearn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        t = result["timings"]
        patched = "—"
        if "patched_default" in t:
            backend = result["fit_details"]["patched_default"][-1]["backend"]
            patched = f"{t['patched_default']['median_ms']:.2f} / {backend}"
        lines.append(
            f"| {result['case']['name']} ({result['case']['n_samples']:,} × "
            f"{result['case']['n_features']:,}) | {t['legacy_mlx']['median_ms']:.2f} | "
            f"{t['fused_mlx']['median_ms']:.2f} | {t['sklearn']['median_ms']:.2f} | "
            f"{patched} | {result['fused_speedup_over_legacy']:.2f}× | "
            f"{result['fused_speedup_over_sklearn']:.2f}× |"
        )
    lines += [
        "",
        "The ratios above are ratios of warmed medians, not statistical significance claims. "
        "Compare convergence and independent objective values before interpreting a "
        "speedup on a difficult case.",
        "",
        "| Case / variant | Iterations | Stop reason | Float64 objective | "
        "Relative objective gap vs sklearn | Float64 max gradient | Prediction agreement |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        for variant, q in result["quality_last_fit"].items():
            detail = result["fit_details"][variant][-1]
            state = detail["solver"]
            reason = state["stop_reason"] if state else "sklearn (see warnings in JSON)"
            lines.append(
                f"| {result['case']['name']} / {variant} | {detail['n_iter'][0]} | {reason} | "
                f"{q['objective_float64']:.10g} | {q['objective_relative_gap_to_sklearn']:.3g} | "
                f"{q['gradient_max_abs_float64']:.3g} | "
                f"{q['prediction_agreement_sklearn']:.5f} |"
            )
    env = payload["environment"]
    lines += [
        "",
        f"Environment: {env.get('cpu', env['machine'])}; {env['platform']}; "
        f"MLX {env['mlx']}; sklearn {env['sklearn']}; NumPy {env['numpy']}; SciPy {env['scipy']}.",
        "",
        "Raw samples, per-fit warnings/backend/solver state, source hashes, thread pools, "
        "and NumPy build configuration are saved in the companion JSON. threadpoolctl "
        "limits affect only the pools it recognizes; Apple Accelerate may not be controlled.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--output", type=Path, default=Path("bench/logistic-gpu.json"))
    parser.add_argument("--report", type=Path, default=Path("bench/logistic-gpu.md"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip-patched", action="store_true")
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least three measured repetitions and one warmup")
    if args.max_iter < 1 or args.tol < 0 or not np.isfinite(args.tol):
        parser.error("max-iter must be positive and tol must be finite and nonnegative")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    variants = VARIANTS[:-1] if args.skip_patched else VARIANTS
    if logistic.minimize is lbfgs.minimize:
        parser.error(
            "production logistic still uses the legacy solver; install the GPU-fusion change first"
        )
    payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "requested_thread_limit": args.threads,
        "variants": variants,
        "input_dtype": args.dtype,
        "results": [],
    }
    with threadpool_limits(limits=args.threads), legacy_reference(args.baseline_ref) as legacy:
        legacy_module, payload["baseline"] = legacy
        payload["environment"] = record_environment()
        for case in PROFILES[args.profile]:
            print(f"Running {case.name}: {case.n_samples:,} × {case.n_features:,}", flush=True)
            result = run_case(
                case,
                seed=args.seed,
                warmup=args.warmup,
                repeats=args.repeats,
                max_iter=args.max_iter,
                tol=args.tol,
                variants=variants,
                legacy_module=legacy_module,
                dtype=args.dtype,
            )
            payload["results"].append(result)
            print(
                {name: round(timing["median_ms"], 3) for name, timing in result["timings"].items()},
                flush=True,
            )
            # Retain completed cases even if a later, expensive case fails.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, payload)
    print(f"Saved {args.output} and {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
