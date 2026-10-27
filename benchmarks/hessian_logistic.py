"""Audit complete logistic fits on wide, shifted, weighted and rank-deficient data.

The independent scores use original inputs and float64 arithmetic outside every
fit timer. All warnings and all measured samples are retained. GPU fits include
validation, preparation, sketch construction, convergence checking and export.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from mlxlearn import config_context
from mlxlearn._common.config import tuning_context
from mlxlearn._solvers import logistic, native_logistic
from mlxlearn.linear_model import LogisticRegression
from mlxlearn.linear_model import _logistic as estimator_module

from .logistic_gpu import Case as GpuCase
from .logistic_gpu import make_data, record_environment
from .logistic_wide import frozen_reference, observed_implementation


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    features: int
    seed: int = 41
    weighted: bool = False
    shifted: bool = False
    ill_conditioned: bool = False
    rank_deficient: bool = False
    C: float = 1.0


CASES = (
    Case("wide_1024", 20_000, 1_024),
    Case("wide_4096", 20_000, 4_096),
    Case("weighted_ill_1024", 20_000, 1_024, seed=83, weighted=True, ill_conditioned=True),
    Case("shifted_1024", 20_000, 1_024, shifted=True),
    Case("rank_deficient_1024", 20_000, 1_024, rank_deficient=True, C=1e6),
    Case("ill_conditioned_4096", 20_000, 4_096, ill_conditioned=True),
)


def _duplicate_columns(X):
    """Create an exactly known null space, including for shifted fixtures."""
    if X.shape[1] % 2:
        raise ValueError("rank-deficient fixture requires an even feature count")
    half = X.shape[1] // 2
    ratio = np.resize(np.array([2.0, 0.5], dtype=np.float32), half)
    X[:, half:] = X[:, :half] * ratio
    return ratio


def dataset(case):
    # Reuse the established wide fixture, including its pre-cast label RNG.
    if case.features == 4_096:
        X, y, weights = make_data(
            GpuCase(case.name, case.rows, case.features, weighted=case.weighted,
                    ill_conditioned=case.ill_conditioned), case.seed, dtype="float32"
        )
        if case.shifted:
            X += np.float32(1_000)
        ratio = _duplicate_columns(X) if case.rank_deficient else None
        return X, y, weights, ratio
    rng = np.random.default_rng(case.seed)
    X = rng.standard_normal((case.rows, case.features), dtype=np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=case.rows) > 0).astype(np.int32)
    ratio = None
    if case.ill_conditioned:
        half = case.features // 2
        X[:, half:] = X[:, :half] + np.float32(0.01) * X[:, half:]
        X *= np.logspace(-1, 1, case.features, dtype=np.float32)
    if case.shifted:
        X += np.float32(1_000)
    if case.rank_deficient:
        ratio = _duplicate_columns(X)
    weights = rng.uniform(0.25, 3.0, case.rows) if case.weighted else None
    return X, y, weights, ratio


def independent_score(model, X, y, weights, C, ratio, reference=None):
    """Score represented parameters on original X in float64 blocks."""
    if hasattr(model, "centered_intercept_"):
        coefficient = np.asarray(model.coef_, dtype=np.float64)
        center = np.asarray(model.center_, dtype=np.float64)
        bias = float(model.centered_intercept_) - float(coefficient[0] @ center)
        model = SimpleNamespace(coef_=coefficient, intercept_=np.array([bias]))
    rows, features = X.shape
    coef = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    gradient = np.zeros(features, dtype=np.float64)
    bias_gradient = loss = 0.0
    predictions = np.empty(rows, dtype=bool)
    for first in range(0, rows, 256):
        last = min(first + 256, rows)
        block = np.asarray(X[first:last], dtype=np.float64)
        labels = y[first:last]
        logits = block @ coef + intercept
        sample_weights = 1.0 if weights is None else weights[first:last]
        residual = (expit(logits) - labels) * sample_weights
        gradient += block.T @ residual
        bias_gradient += float(residual.sum())
        loss += float(np.sum((np.logaddexp(0.0, logits) - labels * logits) * sample_weights))
        predictions[first:last] = logits >= 0
    total = float(rows if weights is None else weights.sum(dtype=np.float64))
    gradient = gradient / total + coef / (C * total)
    objective = loss / total + 0.5 * float(coef @ coef) / (C * total)
    null = None
    if ratio is not None:
        half = features // 2
        null = float(np.linalg.norm(coef[half:] - ratio * coef[:half]) /
                     max(np.linalg.norm(coef), np.finfo(float).tiny))
    result = {"objective_float64": objective,
              "gradient_max_abs_float64": max(float(np.max(np.abs(gradient))), abs(bias_gradient / total)),
              "nullspace_coefficient_relative": null}
    if reference is not None:
        reference_obj, reference_predictions = reference
        result["objective_relative_gap_to_sklearn"] = (objective-reference_obj) / max(abs(reference_obj), np.finfo(float).tiny)
        result["prediction_agreement_sklearn"] = float(np.mean(predictions == reference_predictions))
    return result, predictions


@contextlib.contextmanager
def observed_sketch(native):
    original = native.minimize_binary
    calls = []

    def wrapper(*args, **kwargs):
        history = kwargs.get("initial_history")
        calls.append({"seeded": history is not None,
                      "pairs": int(history[3]) if history is not None else 0})
        return original(*args, **kwargs)

    native.minimize_binary = wrapper
    try:
        yield calls
    finally:
        native.minimize_binary = original


def fit_once(variant, X, y, weights, case, reference, tol, max_iter, device_data=None, sketch_size=4, device_sketch_size=4):
    if variant == "device":
        from mlxlearn_device import fit_logistic

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mx.synchronize()
            old_gc = gc.isenabled()
            gc.disable()
            try:
                start = time.perf_counter()
                model = fit_logistic(device_data[0], device_data[1], sample_weight=device_data[2],
                                     C=case.C, tol=tol, max_iter=max_iter, sketch_size=device_sketch_size)
                mx.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                if old_gc:
                    gc.enable()
        metadata = {"backend": "mlx_device", "n_iter": [model.n_iter],
                    "converged": model.converged, "stop_reason": model.stop_reason,
                    "gradient_max_abs": model.gradient_max_abs,
                    "sketch_pairs": getattr(model, "sketch_pairs_", None),
                    "used_unsketched_fallback": getattr(model, "used_unsketched_fallback_", None),
                    "newton_iterations": getattr(model, "newton_iterations", 0),
                    "newton_hvp_count": getattr(model, "newton_hvp_count", 0),
                    "relative_duality_gap": getattr(model, "relative_duality_gap", None),
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
        return model, elapsed, metadata
    if variant == "baseline":
        estimator, solver, native = reference.estimator, reference.solver, reference.native
        cls, settings, tuning = estimator.LogisticRegression, reference.config.config_context, reference.config.tuning_context
    elif variant in ("current", "sketch"):
        estimator, solver, native = estimator_module, logistic, native_logistic
        cls, settings, tuning = LogisticRegression, config_context, tuning_context
    else:
        cls, settings, tuning = SkLogisticRegression, config_context, tuning_context
        estimator, solver, native = estimator_module, logistic, native_logistic
    tuning_options = {"force_backend": "" if variant == "sklearn" else "mlx"}
    if variant == "sketch":
        tuning_options["lbfgs_sketch_size"] = sketch_size
    with (settings(device="auto"), tuning(**tuning_options), observed_sketch(native) as sketch_calls,
          observed_implementation(estimator, solver, native) as record,
          warnings.catch_warnings(record=True) as caught):
        warnings.simplefilter("always")
        mx.synchronize()
        old_gc = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            model = cls(C=case.C, tol=tol, max_iter=max_iter).fit(X, y, sample_weight=weights)
            mx.synchronize()
            elapsed = time.perf_counter() - start
        finally:
            if old_gc:
                gc.enable()
    metadata = {"backend": getattr(model, "_execution_backend_", "sklearn"),
                "n_iter": model.n_iter_.tolist(),
                "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                "sketch_calls": sketch_calls, **record}
    return model, elapsed, metadata


def _completed_sketch_or_handoff(detail):
    """Accept a completed native fit or an observed, certified Newton handoff."""
    if "completed" in detail.get("native_loop_calls", []):
        return True
    native_warm = any(
        attempt.get("status") == 0 and attempt.get("native_n_iter", 0) > 0
        and attempt.get("outcome") == "terminal_nonconvergence"
        for attempt in detail.get("native_loop_attempts", [])
    )
    refinement = detail.get("newton_calls", [])
    return bool(native_warm and refinement and all(
        call.get("resumed") and call.get("converged")
        and call.get("stop_reason") == "duality-gap"
        and math.isfinite(call.get("relative_duality_gap", float("inf")))
        and call["relative_duality_gap"] <= 1e-5
        and 0 <= call.get("n_iter", -1) <= call.get("budget", -1)
        for call in refinement
    ))


def audit_case(case, scores, details, tol, trusted_reference=None):
    """Record every quality failure; never discard a timed fit."""
    skscore = scores["sklearn"][-1]
    skwarn = any(w["category"] == "ConvergenceWarning" for w in details["sklearn"][-1]["warnings"])
    sklearn_usable = (math.isfinite(skscore["gradient_max_abs_float64"])
                      and math.isfinite(skscore["objective_float64"])
                      and skscore["gradient_max_abs_float64"] <= tol and not skwarn)
    trusted = trusted_reference is not None
    current_score = scores["current"][-1]
    current_valid = (math.isfinite(current_score["objective_float64"])
                     and math.isfinite(current_score["gradient_max_abs_float64"])
                     and current_score["gradient_max_abs_float64"] <= tol
                     and all(item.get("converged", True) for item in details["current"][-1].get("solutions", [])))
    failures = []
    if not sklearn_usable and not trusted:
        failures.append({"variant": "benchmark", "fit": None, "reason": "no usable objective reference"})
    for variant in ("current", "sketch", "device"):
        if variant not in scores:
            continue
        for i, (score, detail) in enumerate(zip(scores[variant], details[variant], strict=True)):
            metrics = ["objective_float64", "gradient_max_abs_float64", "prediction_agreement_sklearn"]
            if math.isfinite(skscore["objective_float64"]):
                metrics.append("objective_relative_gap_to_sklearn")
            for metric in metrics:
                if not math.isfinite(score[metric]):
                    failures.append({"variant": variant, "fit": i, "reason": f"nonfinite {metric}"})
            if case.rank_deficient and not math.isfinite(score["nullspace_coefficient_relative"]):
                failures.append({"variant": variant, "fit": i, "reason": "nonfinite null-space ratio"})
            if detail.get("backend") not in ("mlx", "mlx_device"):
                failures.append({"variant": variant, "fit": i, "reason": "unexpected backend"})
            if variant == "sketch" and (not any(call["seeded"] and call["pairs"] > 0 for call in detail["sketch_calls"])
                    or not _completed_sketch_or_handoff(detail)):
                failures.append({"variant": variant, "fit": i, "reason": "seeded native loop was not used"})
            if variant == "device" and not detail.get("sketch_pairs"):
                failures.append({"variant": variant, "fit": i, "reason": "device sketch had no valid pairs"})
            if (not detail.get("converged", True) or
                any(not item["converged"] for item in detail.get("solutions", [])) or
                any(w["category"] in ("ConvergenceWarning", "DeviceConvergenceWarning") for w in detail["warnings"])):
                failures.append({"variant": variant, "fit": i, "reason": "reported nonconvergence"})
            if score["gradient_max_abs_float64"] > tol:
                failures.append({"variant": variant, "fit": i, "reason": "original gradient exceeds tolerance"})
            gap = score.get("objective_relative_gap_to_trusted_reference") if trusted else score["objective_relative_gap_to_sklearn"]
            if (sklearn_usable or trusted) and (not math.isfinite(gap) or gap > 1e-5):
                failures.append({"variant": variant, "fit": i, "reason": "objective exceeds usable sklearn reference by more than 1e-5"})
            agreement = (score.get("prediction_agreement_trusted_reference") if trusted
                         else score["prediction_agreement_sklearn"])
            if (sklearn_usable or trusted) and (not math.isfinite(agreement) or agreement < 0.995):
                failures.append({"variant": variant, "fit": i, "reason": "prediction agreement below 99.5%"})
            if variant in ("sketch", "device") and current_valid:
                delta = (score["objective_float64"] - current_score["objective_float64"]) / max(abs(current_score["objective_float64"]), np.finfo(float).tiny)
                if not math.isfinite(delta) or delta > 1e-5:
                    failures.append({"variant": variant, "fit": i, "reason": "objective exceeds current solver by more than 1e-5"})
            if case.rank_deficient and score["nullspace_coefficient_relative"] > 1e-3:
                failures.append({"variant": variant, "fit": i, "reason": "null-space coefficient artifact"})
    return {"sklearn_reference_gradient_and_warning_ok": sklearn_usable,
            "trusted_centered_reference_used": trusted, "passed": not failures, "failures": failures}


def sources(extra=()):
    root = Path(__file__).resolve().parents[1]
    dirs = (root / "src/mlxlearn", root / "src/mlxlearn_device")
    paths = [p for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and (p.suffix in {".py", ".mm", ".h", ".hpp", ".so"} or p.name == "py.typed")]
    paths += [Path(__file__).resolve(), root / "benchmarks/logistic_wide.py", root / "benchmarks/logistic_gpu.py", *extra]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES])
    parser.add_argument("--device", action="store_true", help="also benchmark the resident MLX-array interface; upload measured separately")
    parser.add_argument("--sketch-size", type=int, default=4, help="pairs for opt-in sklearn-compatible GPU sketch")
    parser.add_argument("--device-sketch-size", type=int, default=4, help="pairs for resident-device API")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--reference-archive", type=Path, default=Path("benchmarks/reference/hessian-start-2026-10-27.zip"))
    parser.add_argument("--reference-extension", type=Path, default=Path("bench/hessian-start/_metal_loop.abi3.so"))
    parser.add_argument("--output", type=Path, default=Path("bench/hessian-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    extra = [args.reference_archive.resolve(), args.reference_archive.with_suffix(".manifest.json").resolve(), args.reference_extension.resolve()]
    before = sources(extra)
    result = {"method": "balanced complete fits; one-thread sklearn; independent original-input float64 score after timing",
              "parameters": vars(args) | {"reference_archive": str(args.reference_archive),
                                           "reference_extension": str(args.reference_extension), "output": str(args.output)},
              "environment": record_environment(), "source_integrity_before": before, "cases": []}
    with (frozen_reference(archive=args.reference_archive, extension=args.reference_extension) as ref,
          threadpool_limits(limits=1)):
        result["reference"] = ref.provenance
        for case in CASES:
            if args.case and case.name not in args.case:
                continue
            X, y, weights, ratio = dataset(case)
            device_data = None
            upload_ms = None
            if args.device:
                mx.synchronize()
                upload_start = time.perf_counter()
                device_data = (mx.array(X), mx.array(y), None if weights is None else mx.array(weights, dtype=mx.float32))
                mx.eval(*(value for value in device_data if value is not None))
                mx.synchronize()
                upload_ms = (time.perf_counter() - upload_start) * 1e3
            variants = ("baseline", "current", "sketch", "sklearn", "device") if args.device else ("baseline", "current", "sketch", "sklearn")
            samples = {key: [] for key in variants}
            details = {key: [] for key in variants}
            models = {key: [] for key in variants}
            for i in range(args.repeats + args.warmup):
                order = variants[i % len(variants):] + variants[:i % len(variants)]
                if i % 2:
                    order = order[::-1]
                for variant in order:
                    model, duration, metadata = fit_once(variant, X, y, weights, case, ref, args.tol, args.max_iter, device_data, args.sketch_size, args.device_sketch_size)
                    if i >= args.warmup:
                        samples[variant].append(duration * 1e3)
                        details[variant].append(metadata)
                        models[variant].append(model)
            skscore, skpred = independent_score(models["sklearn"][-1], X, y, weights, case.C, ratio)
            scored = {variant: [independent_score(model, X, y, weights, case.C, ratio,
                              (skscore["objective_float64"], skpred)) for model in group]
                      for variant, group in models.items()}
            scores = {variant: [item[0] for item in group] for variant, group in scored.items()}
            tight_info = None
            tight_objective = None
            sklearn_warned = any(w["category"] == "ConvergenceWarning" for w in details["sklearn"][-1]["warnings"])
            if (not math.isfinite(skscore["gradient_max_abs_float64"]) or
                skscore["gradient_max_abs_float64"] > args.tol or sklearn_warned):
                # This untimed float64 reference handles shifted inputs for which
                # sklearn on raw coordinates can claim success at the null model.
                center = X.mean(axis=0, dtype=np.float64)
                centered = np.asarray(X, dtype=np.float64) - center
                with warnings.catch_warnings(record=True) as tight_warnings:
                    warnings.simplefilter("always")
                    tight = SkLogisticRegression(C=case.C, tol=min(args.tol * 1e-4, 1e-9), max_iter=3_000).fit(centered, y, sample_weight=weights)
                tight_model = SimpleNamespace(coef_=tight.coef_, intercept_=tight.intercept_ - tight.coef_ @ center)
                tight_score, tight_predictions = independent_score(tight_model, X, y, weights, case.C, ratio)
                tight_info = {"score": tight_score, "n_iter": tight.n_iter_.tolist(),
                              "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in tight_warnings]}
                if (math.isfinite(tight_score["objective_float64"]) and
                    math.isfinite(tight_score["gradient_max_abs_float64"]) and
                    tight_score["gradient_max_abs_float64"] <= args.tol and
                    not any(w["category"] == "ConvergenceWarning" for w in tight_info["warnings"])):
                    tight_objective = tight_score["objective_float64"]
                    for group in scored.values():
                        for score, predictions in group:
                            score["objective_relative_gap_to_trusted_reference"] = (
                                score["objective_float64"] - tight_objective) / max(abs(tight_objective), np.finfo(float).tiny)
                            score["prediction_agreement_trusted_reference"] = float(np.mean(predictions == tight_predictions))
                del centered
            represented_scores = None
            if weights is not None and "device" in models:
                represented_weights = weights.astype(np.float32).astype(np.float64)
                represented_scores = [independent_score(model, X, y, represented_weights, case.C, ratio)[0]
                                      for model in models["device"]]
            row = {"case": asdict(case), "device_upload_ms": upload_ms, "median_ms": {v: statistics.median(s) for v, s in samples.items()},
                   "samples_ms": samples, "fit_details": details, "sklearn_reference_score": skscore,
                   "tight_centered_sklearn_reference": tight_info, "device_represented_weight_scores": represented_scores,
                   "quality_each_fit": scores, "quality_audit": audit_case(case, scores, details, args.tol, tight_objective)}
            result["cases"].append(row)
            print(case.name, {v: round(x, 3) for v, x in row["median_ms"].items()},
                  "grad", {v: round(scores[v][-1]["gradient_max_abs_float64"], 8) for v in variants}, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["source_integrity_after"] = sources(extra)
    result["source_integrity_pass"] = result["source_integrity_before"] == result["source_integrity_after"]
    result["quality_audit_pass"] = all(row["quality_audit"]["passed"] for row in result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["source_integrity_pass"] and result["quality_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
