"""Paired full-fit audit against frozen unseeded and four-pair solvers.

The frozen sketch is loaded from a source-verified archive and substituted only
for the frozen reference fit. No timed fit is discarded. Independent float64
checks are performed outside the timed region, including original weights,
objective, original-coordinate gradient, predictions and exact null space.
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
import types
import warnings
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from .hessian_logistic import (
    CASES as BASE_CASES,
)
from .hessian_logistic import (
    Case,
    audit_case,
    dataset,
    fit_once,
    independent_score,
    observed_sketch,
    sources,
)
from .logistic_gpu import record_environment
from .logistic_wide import frozen_reference, observed_implementation

CASES = BASE_CASES + (
    Case("weighted_1024", 20_000, 1_024, weighted=True),
    Case("weighted_4096", 20_000, 4_096, weighted=True),
    Case("weighted_unbalanced_1024", 20_000, 1_024, weighted=True),
    Case("weighted_rare_1024", 20_000, 1_024, weighted=True),
    Case("weighted_effective_unbalanced_1024", 20_000, 1_024, weighted=True),
)

EXTENDED_CASES = (
    Case("weighted_ill_conditioned_4096", 20_000, 4_096, weighted=True, ill_conditioned=True),
    Case("shifted_4096", 20_000, 4_096, shifted=True),
    Case("weighted_shifted_4096", 20_000, 4_096, weighted=True, shifted=True),
    Case("rank_deficient_4096", 20_000, 4_096, rank_deficient=True, C=1e6),
)

TIGHT_CASES = frozenset({"weighted_unbalanced_1024", "weighted_rare_1024",
                         "weighted_effective_unbalanced_1024"})
EXTENDED_TIGHT_CASES = frozenset(case.name for case in EXTENDED_CASES)


def balanced_order(variants, repetition):
    """Rotate variants so each occupies every timing position equally often."""
    offset = repetition % len(variants)
    return variants[offset:] + variants[:offset]


def selected_cases(requested):
    """Keep historical defaults while allowing explicit extended stress cases."""
    if not requested:
        return CASES
    return tuple(case for case in CASES + EXTENDED_CASES if case.name in requested)


def benchmark_dataset(case):
    """Add label and effective-weight imbalance without changing the input generator."""
    X, y, weights, ratio = dataset(case)
    if case.name == "weighted_unbalanced_1024":
        y = (X[:, 0] > np.quantile(X[:, 0], 0.90)).astype(np.int32)
    elif case.name == "weighted_rare_1024":
        y = (X[:, 0] > np.quantile(X[:, 0], 0.98)).astype(np.int32)
    elif case.name == "weighted_effective_unbalanced_1024":
        weights = weights * np.where(y == 1, 0.1, 2.0)
    return X, y, weights, ratio


def tight_reference_usable(score, caught, tol):
    """Require an independently small original-coordinate gradient and no warning."""
    limit = min(tol * 0.01, 1e-7)
    return (math.isfinite(score["objective_float64"]) and
            math.isfinite(score["gradient_max_abs_float64"]) and
            score["gradient_max_abs_float64"] <= limit and
            not any(w["category"] == "ConvergenceWarning" for w in caught))


def reference_sketch_module(archive):
    """Load only the frozen, SHA-verified GPU sketch helpers."""
    archive = Path(archive).resolve()
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive"]["sha256"]:
        raise RuntimeError("Frozen sketch source archive hash mismatch")
    entry = "src/mlxlearn_device/hessian_sketch.py"
    with zipfile.ZipFile(archive) as data:
        if data.namelist().count(entry) != 1:
            raise RuntimeError("Frozen sketch archive must contain exactly one helper")
        source = data.read(entry)
    if hashlib.sha256(source).hexdigest() != manifest["files"][entry]:
        raise RuntimeError("Frozen sketch helper hash mismatch")
    module = types.ModuleType("_mlxlearn_frozen_sketch")
    exec(compile(source, f"{archive}!/{entry}", "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def use_reference_sketch(reference):
    """Reference and live fits are sequential; restore shared names on exit."""
    import mlxlearn_device.hessian_sketch as live

    names = ("build_sketch_history", "initial_metric_from_parts")
    saved = {name: getattr(live, name) for name in names}
    try:
        for name in names:
            setattr(live, name, getattr(reference, name))
        yield
    finally:
        for name, value in saved.items():
            setattr(live, name, value)


def fit_previous_sketch(X, y, weights, case, reference, frozen_sketch, tol, max_iter, sketch_size):
    estimator, solver, native = reference.estimator, reference.solver, reference.native
    with (use_reference_sketch(frozen_sketch),
          reference.config.config_context(device="auto"),
          reference.config.tuning_context(force_backend="mlx", lbfgs_sketch_size=sketch_size),
          observed_sketch(native) as sketch_calls,
          observed_implementation(estimator, solver, native) as record,
          warnings.catch_warnings(record=True) as caught):
        warnings.simplefilter("always")
        mx.synchronize()
        enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            model = estimator.LogisticRegression(C=case.C, tol=tol, max_iter=max_iter).fit(
                X, y, sample_weight=weights
            )
            mx.synchronize()
            duration = time.perf_counter() - start
        finally:
            if enabled:
                gc.enable()
    metadata = {"backend": getattr(model, "_execution_backend_", "unknown"),
                "n_iter": model.n_iter_.tolist(),
                "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                "sketch_calls": sketch_calls, **record}
    return model, duration, metadata


def audit_previous_regression(scores, details, tol):
    """Compare paired fits only to gradient-valid frozen results; retain failures."""
    failures = []
    valid = {}
    for prior in ("baseline", "baseline_sketch"):
        if prior not in scores:
            continue
        valid[prior] = []
        for i, (score, detail) in enumerate(zip(scores[prior], details[prior], strict=True)):
            ok = (math.isfinite(score["objective_float64"]) and
                  math.isfinite(score["gradient_max_abs_float64"]) and
                  score["gradient_max_abs_float64"] <= tol and
                  detail.get("backend") == "mlx" and
                  not any(w["category"] == "ConvergenceWarning" for w in detail["warnings"]) and
                  all(state["converged"] for state in detail.get("solutions", [])))
            valid[prior].append(ok)
            if not ok:
                failures.append({"variant": prior, "fit": i, "reason": "previous fit fails GPU gradient or convergence checks"})
            if prior == "baseline_sketch":
                if not any(call["seeded"] and call["pairs"] > 0 for call in detail["sketch_calls"]):
                    failures.append({"variant": prior, "fit": i, "reason": "frozen sketch not exercised"})
                if "completed" not in detail.get("native_loop_calls", []):
                    failures.append({"variant": prior, "fit": i, "reason": "frozen native sketch did not finish"})
    comparisons = (("current", "baseline"), ("sketch", "baseline"),
                   ("sketch", "baseline_sketch"), ("device", "baseline"),
                   ("device", "baseline_sketch"))
    for candidate, prior in comparisons:
        if candidate not in scores or prior not in scores:
            continue
        if len(scores[candidate]) != len(scores[prior]):
            failures.append({"variant": candidate, "fit": None, "reason": "unpaired previous fits"})
            continue
        for i, (item, previous) in enumerate(zip(scores[candidate], scores[prior], strict=True)):
            if not valid[prior][i]:
                continue
            relative = (item["objective_float64"] - previous["objective_float64"]) / max(
                abs(previous["objective_float64"]), np.finfo(float).tiny
            )
            item[f"objective_relative_gap_to_{prior}"] = relative
            if not math.isfinite(relative) or relative > 1e-5:
                failures.append({"variant": candidate, "fit": i,
                                 "reason": f"objective > 1e-5 relative worse than {prior}"})
    return {"passed": not failures, "failures": failures}


def audit_previous_objective(scores, tight_objective):
    """Record prior objective gaps separately, including known inaccurate fits."""
    if tight_objective is None:
        return {"reference_available": False, "passed": None, "fits": []}
    fits = []
    for variant in ("baseline", "baseline_sketch"):
        for i, score in enumerate(scores.get(variant, [])):
            gap = (score["objective_float64"] - tight_objective) / max(
                abs(tight_objective), np.finfo(float).tiny
            )
            fits.append({"variant": variant, "fit": i, "relative_objective_gap": gap,
                         "passed": math.isfinite(gap) and gap <= 1e-5})
    return {"reference_available": True, "passed": all(fit["passed"] for fit in fits),
            "fits": fits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES + EXTENDED_CASES])
    parser.add_argument("--seed", type=int, help="override the recorded fixture seed")
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--tight-reference", action="store_true",
                        help="audit every case against an independently tightened centered sklearn fit")
    parser.add_argument("--input-dtype", choices=("float32", "float64"), default="float32",
                        help="float64 is an exact widening of the generated float32 fixture")
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--previous-sketch-size", type=int, default=4)
    parser.add_argument("--device-sketch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--reference-max-iter", type=int, default=3000,
                        help="accepted iteration budget for the independent tight reference only")
    parser.add_argument("--reference-tol", type=float,
                        help="optional tighter gradient tolerance for the independent reference only")
    parser.add_argument("--reference-solver", choices=("lbfgs", "newton-cg"), default="lbfgs",
                        help="untimed independent reference algorithm; timed stock sklearn stays lbfgs")
    parser.add_argument("--reference-archive", type=Path, default=Path("benchmarks/reference/sketch-weighted-start-2026-10-27.zip"))
    parser.add_argument("--reference-sources", type=Path, default=Path("benchmarks/reference/sketch-weighted-start-sources-2026-10-27.zip"))
    parser.add_argument("--reference-extension", type=Path, default=Path("bench/sketch-weighted-start/_metal_loop.abi3.so"))
    parser.add_argument("--output", type=Path, default=Path("bench/sketch-weighted-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    if args.reference_max_iter < 1 or (args.reference_tol is not None and
            (not math.isfinite(args.reference_tol) or args.reference_tol <= 0)):
        parser.error("reference iteration budget and tolerance must be positive")
    extra = [Path(__file__).resolve(), args.reference_archive.resolve(),
             args.reference_archive.with_suffix(".manifest.json").resolve(),
             args.reference_sources.resolve(), args.reference_sources.with_suffix(".manifest.json").resolve(),
             args.reference_extension.resolve()]
    before = sources(extra)
    frozen_sketch = reference_sketch_module(args.reference_sources)
    payload = {"method": "balanced complete fits; frozen four-pair sketch; one-thread sklearn; independent float64 scores after timing",
               "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "environment": record_environment(), "source_integrity_before": before, "cases": []}
    with (frozen_reference(archive=args.reference_archive, extension=args.reference_extension) as ref,
          threadpool_limits(limits=1)):
        payload["reference"] = ref.provenance
        for original_case in selected_cases(args.case):
            case = replace(original_case, seed=args.seed) if args.seed is not None else original_case
            X, y, weights, ratio = benchmark_dataset(case)
            if args.input_dtype == "float64":
                X = X.astype(np.float64)
            device_data, upload_ms = None, None
            if args.device:
                mx.synchronize()
                start = time.perf_counter()
                device_data = (mx.array(X, dtype=mx.float32), mx.array(y), None if weights is None else mx.array(weights, dtype=mx.float32))
                mx.eval(*(v for v in device_data if v is not None))
                mx.synchronize()
                upload_ms = 1e3 * (time.perf_counter() - start)
            variants = ("baseline", "baseline_sketch", "current", "sketch", "sklearn")
            if args.device:
                variants += ("device",)
            samples, details, models = ({v: [] for v in variants} for _ in range(3))
            for rep in range(args.warmup + args.repeats):
                order = balanced_order(variants, rep)
                for variant in order:
                    if variant == "baseline_sketch":
                        model, elapsed, meta = fit_previous_sketch(X, y, weights, case, ref, frozen_sketch,
                                                                    args.tol, args.max_iter, args.previous_sketch_size)
                    else:
                        model, elapsed, meta = fit_once(variant, X, y, weights, case, ref, args.tol,
                                                        args.max_iter, device_data, args.sketch_size, args.device_sketch_size)
                    if rep >= args.warmup:
                        samples[variant].append(elapsed * 1000)
                        if variant != "device":
                            digest = hashlib.sha256()
                            for parameter in (model.coef_, model.intercept_):
                                value = np.ascontiguousarray(parameter)
                                digest.update(str(value.dtype).encode())
                                digest.update(repr(value.shape).encode())
                                digest.update(value.tobytes())
                            meta["parameter_sha256"] = digest.hexdigest()
                        details[variant].append(meta)
                        models[variant].append(model)
            skscore, skpred = independent_score(models["sklearn"][-1], X, y, weights, case.C, ratio)
            scored = {v: [independent_score(m, X, y, weights, case.C, ratio,
                      (skscore["objective_float64"], skpred)) for m in group] for v, group in models.items()}
            scores = {v: [item[0] for item in group] for v, group in scored.items()}
            tight_info, tight_objective, tight_prediction = None, None, None
            skwarn = any(w["category"] == "ConvergenceWarning" for w in details["sklearn"][-1]["warnings"])
            require_tight = args.tight_reference or case.name in TIGHT_CASES | EXTENDED_TIGHT_CASES
            if require_tight or not math.isfinite(skscore["gradient_max_abs_float64"]) or skscore["gradient_max_abs_float64"] > args.tol or skwarn:
                center = X.mean(axis=0, dtype=np.float64)
                centered = np.asarray(X, dtype=np.float64) - center
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    tight = SkLogisticRegression(
                        C=case.C, tol=min(args.tol * 1e-4, 1e-9, args.reference_tol or 1e-9),
                        max_iter=args.reference_max_iter, solver=args.reference_solver,
                    ).fit(centered, y, sample_weight=weights)
                tight_model = SimpleNamespace(coef_=tight.coef_, intercept_=tight.intercept_ - tight.coef_ @ center)
                tight_score, tight_prediction = independent_score(tight_model, X, y, weights, case.C, ratio)
                tight_info = {"score": tight_score, "n_iter": tight.n_iter_.tolist(),
                              "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
                if tight_reference_usable(tight_score, tight_info["warnings"], args.tol):
                    tight_objective = tight_score["objective_float64"]
                    for group in scored.values():
                        for score, predictions in group:
                            score["objective_relative_gap_to_trusted_reference"] = (score["objective_float64"] - tight_objective) / max(abs(tight_objective), np.finfo(float).tiny)
                            score["prediction_agreement_trusted_reference"] = float(np.mean(predictions == tight_prediction))
                del centered
            device_prediction_audit = None
            if "device" in models:
                reference_predictions = tight_prediction if tight_objective is not None else skpred
                device_prediction_audit = []
                for i, model in enumerate(models["device"]):
                    actual = np.asarray(model.predict(device_data[0])).reshape(-1).astype(bool)
                    own_predictions = scored["device"][i][1]
                    device_prediction_audit.append({
                        "agreement_with_valid_reference": float(np.mean(actual == reference_predictions)),
                        "agreement_with_own_original_float64_model": float(np.mean(actual == own_predictions)),
                    })
            represented = None
            if weights is not None and "device" in models:
                represented_weights = weights.astype(np.float32).astype(np.float64)
                represented = [independent_score(m, X, y, represented_weights, case.C, ratio)[0] for m in models["device"]]
            row = {"case": asdict(case), "input_dtype": args.input_dtype, "device_upload_ms": upload_ms,
                   "median_ms": {v: statistics.median(s) for v, s in samples.items()},
                   "samples_ms": samples, "fit_details": details, "sklearn_reference_score": skscore,
                   "tight_centered_sklearn_reference": tight_info, "device_represented_weight_scores": represented,
                   "device_prediction_audit": device_prediction_audit,
                   "quality_each_fit": scores, "quality_audit": audit_case(case, scores, details, args.tol, tight_objective),
                   "previous_objective_audit": audit_previous_objective(scores, tight_objective),
                   "previous_regression_audit": audit_previous_regression(scores, details, args.tol)}
            if require_tight and tight_objective is None:
                row["quality_audit"]["passed"] = False
                row["quality_audit"]["failures"].append({"variant": "benchmark", "fit": None,
                    "reason": "required tighter reference failed its independent gradient or warning check"})
            if device_prediction_audit is not None:
                for i, check in enumerate(device_prediction_audit):
                    if (not math.isfinite(check["agreement_with_valid_reference"]) or
                        check["agreement_with_valid_reference"] < 0.995 or
                        not math.isfinite(check["agreement_with_own_original_float64_model"]) or
                        check["agreement_with_own_original_float64_model"] < 0.999):
                        row["quality_audit"]["passed"] = False
                        row["quality_audit"]["failures"].append({"variant": "device", "fit": i,
                            "reason": "actual device predictions disagree with audited reference"})
            # A trustworthy final stock reference must be representative of all
            # its timed fits; shifted cases may instead use the centered reference.
            if row["quality_audit"]["sklearn_reference_gradient_and_warning_ok"]:
                for i, (score, detail) in enumerate(zip(scores["sklearn"], details["sklearn"], strict=True)):
                    if (not math.isfinite(score["objective_float64"]) or
                        not math.isfinite(score["gradient_max_abs_float64"]) or
                        score["gradient_max_abs_float64"] > args.tol or
                        any(w["category"] == "ConvergenceWarning" for w in detail["warnings"])):
                        row["quality_audit"]["passed"] = False
                        row["quality_audit"]["failures"].append({"variant": "sklearn", "fit": i,
                            "reason": "stock reference accuracy varies across timed fits"})
            payload["cases"].append(row)
            print(case.name, {v: round(ms, 3) for v, ms in row["median_ms"].items()},
                  "quality", row["quality_audit"]["passed"], "previous", row["previous_regression_audit"]["passed"], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
    payload["source_integrity_after"] = sources(extra)
    payload["source_integrity_pass"] = before == payload["source_integrity_after"]
    payload["quality_audit_pass"] = all(c["quality_audit"]["passed"] and c["previous_regression_audit"]["passed"] for c in payload["cases"])
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["source_integrity_pass"] and payload["quality_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
