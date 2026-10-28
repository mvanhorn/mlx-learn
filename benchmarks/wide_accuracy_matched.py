"""Compare wide GPU fits and stock sklearn against a tight original-data score.

The timed sklearn and host GPU estimators receive the same contiguous float64
inputs, exactly widened from the generated float32 fixture. The device variant
receives the same represented values as resident float32 MLX arrays; its upload
is recorded separately. All scores use original inputs and float64 arithmetic
outside the fit timers. No measured fit or warning is discarded.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from .hessian_logistic import Case, dataset, fit_once, independent_score, sources
from .logistic_gpu import record_environment
from .sketch_weighted import balanced_order


def reference_valid(score, warnings_, gradient_limit):
    """Reject a reference that has not independently passed its gradient check."""
    return (math.isfinite(score["objective_float64"])
            and math.isfinite(score["gradient_max_abs_float64"])
            and score["gradient_max_abs_float64"] <= gradient_limit
            and not any(w["category"] == "ConvergenceWarning" for w in warnings_))


def audit_fits(variant, scores, details, tol, objective_limit):
    """Check each fit's quality and recorded execution path."""
    failures = []
    if len(scores) != len(details) or not scores:
        return {"passed": False, "failures": [{"fit": None, "reason": "missing or unpaired fitted results"}]}
    for index, (score, detail) in enumerate(zip(scores, details, strict=True)):
        if variant in ("current", "sketch") and detail.get("backend") != "mlx":
            failures.append({"fit": index, "reason": "host fit did not use the MLX backend"})
        if variant == "device" and detail.get("backend") != "mlx_device":
            failures.append({"fit": index, "reason": "unexpected device backend"})
        if variant == "sketch" and (
            not any(call["seeded"] and call["pairs"] > 0 for call in detail.get("sketch_calls", []))
            or "completed" not in detail.get("native_loop_calls", [])
        ):
            failures.append({"fit": index, "reason": "seeded native sketch was not exercised"})
        if variant == "device" and not detail.get("sketch_pairs"):
            failures.append({"fit": index, "reason": "device sketch had no valid pairs"})
        if (not detail.get("converged", True)
            or any(not state.get("converged", False) for state in detail.get("solutions", []))
            or any(w["category"] in ("ConvergenceWarning", "DeviceConvergenceWarning")
                   for w in detail["warnings"])):
            failures.append({"fit": index, "reason": "fit did not report convergence"})
        gradient = score["gradient_max_abs_float64"]
        if not math.isfinite(gradient) or gradient > tol:
            failures.append({"fit": index, "reason": "original-coordinate gradient exceeds tolerance"})
        gap = score["objective_relative_gap_to_sklearn"]
        if not math.isfinite(gap) or gap > objective_limit:
            failures.append({"fit": index, "reason": "objective exceeds tight reference limit"})
        agreement = score["prediction_agreement_sklearn"]
        if not math.isfinite(agreement) or agreement < 0.995:
            failures.append({"fit": index, "reason": "prediction agreement below 99.5%"})
    return {"passed": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", help="fixture seed; defaults to 41, 83, 2027")
    parser.add_argument("--C", type=float, action="append", help="regularization parameter; defaults to 1")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=4_096)
    parser.add_argument("--gpu-tol", type=float, default=1e-4)
    parser.add_argument("--sklearn-tol", type=float, default=1e-7)
    parser.add_argument("--gpu-max-iter", type=int, default=200)
    parser.add_argument("--sklearn-max-iter", type=int, default=2_000)
    parser.add_argument("--reference-max-iter", type=int, default=5_000)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reference-gradient-limit", type=float, default=1e-7)
    parser.add_argument("--objective-limit", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("bench/wide-accuracy-matched.json"))
    args = parser.parse_args()
    if (args.rows <= 0 or not 512 <= args.features <= 4096 or args.repeats <= 0 or args.warmup <= 0
        or args.gpu_max_iter <= 0 or args.sklearn_max_iter <= 0 or args.reference_max_iter <= 0
        or not 1 <= args.sketch_size <= 10
        or any(not math.isfinite(x) or x <= 0 for x in (
            args.gpu_tol, args.sklearn_tol, args.reference_gradient_limit, args.objective_limit,
            *(args.C or [1.0]),
        ))):
        parser.error("dimensions, budgets, tolerances and C must be valid positive values")

    extra = [Path(__file__).resolve(), Path(__file__).with_name("sketch_weighted.py").resolve()]
    before = sources(extra)
    result = {
        "method": "rotated complete fits on identical represented original inputs; one BLAS thread; untimed independent centered float64 reference",
        "parameters": vars(args) | {"output": str(args.output)},
        "environment": record_environment(), "source_integrity_before": before, "cases": [],
    }
    with threadpool_limits(limits=1):
        for C in args.C or [1.0]:
            for seed in args.seed or [41, 83, 2027]:
                case = Case("normal_wide", args.rows, args.features, seed=seed, C=C)
                X, y, weights, ratio = dataset(case)
                X = np.ascontiguousarray(X, dtype=np.float64)
                mx.synchronize()
                start = time.perf_counter()
                device = (mx.array(X, dtype=mx.float32), mx.array(y), None)
                mx.eval(device[0], device[1])
                mx.synchronize()
                upload_ms = (time.perf_counter() - start) * 1000
                variants = ("current", "sketch", "device", "sklearn")
                samples, details, models = ({v: [] for v in variants} for _ in range(3))
                for rep in range(args.warmup + args.repeats):
                    order = balanced_order(variants, rep)
                    for variant in order:
                        tol = args.sklearn_tol if variant == "sklearn" else args.gpu_tol
                        budget = args.sklearn_max_iter if variant == "sklearn" else args.gpu_max_iter
                        model, elapsed, metadata = fit_once(
                            variant, X, y, weights, case, None, tol, budget, device,
                            args.sketch_size, args.sketch_size,
                        )
                        if rep >= args.warmup:
                            samples[variant].append(elapsed * 1000)
                            details[variant].append(metadata)
                            models[variant].append(model)

                center = X.mean(axis=0, dtype=np.float64)
                centered = X - center
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reference = SkLogisticRegression(C=C, tol=1e-10, max_iter=args.reference_max_iter).fit(centered, y)
                ref_model = SimpleNamespace(coef_=reference.coef_,
                                            intercept_=reference.intercept_ - reference.coef_ @ center)
                ref_score, ref_predictions = independent_score(ref_model, X, y, None, C, ratio)
                ref_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                valid_ref = reference_valid(ref_score, ref_warnings, args.reference_gradient_limit)
                scored = {v: [independent_score(m, X, y, None, C, ratio,
                          (ref_score["objective_float64"], ref_predictions)) for m in group]
                          for v, group in models.items()}
                scores = {v: [item[0] for item in group] for v, group in scored.items()}
                audits = {v: audit_fits(v, scores[v], details[v],
                          args.sklearn_tol if v == "sklearn" else args.gpu_tol,
                          args.objective_limit) for v in variants}
                device_predictions = []
                for index, model in enumerate(models["device"]):
                    actual = np.asarray(model.predict(device[0])).reshape(-1).astype(bool)
                    device_predictions.append({
                        "agreement_with_reference": float(np.mean(actual == ref_predictions)),
                        "agreement_with_scored_model": float(np.mean(actual == scored["device"][index][1])),
                    })
                    if (device_predictions[-1]["agreement_with_reference"] < 0.995
                        or device_predictions[-1]["agreement_with_scored_model"] < 0.999):
                        audits["device"]["passed"] = False
                        audits["device"]["failures"].append({"fit": index, "reason": "actual device predictions failed audit"})
                row = {
                    "case": asdict(case), "input_dtype": "float64 exact widening of float32",
                    "device_upload_ms": upload_ms,
                    "median_ms": {v: statistics.median(s) for v, s in samples.items()},
                    "samples_ms": samples, "fit_details": details, "quality_each_fit": scores,
                    "device_prediction_audit": device_predictions,
                    "reference": {"score": ref_score, "warnings": ref_warnings,
                                  "n_iter": reference.n_iter_.tolist(), "valid": valid_ref},
                    "audit": audits, "passed": valid_ref and all(v["passed"] for v in audits.values()),
                }
                result["cases"].append(row)
                print("C", C, "seed", seed, {v: round(ms, 3) for v, ms in row["median_ms"].items()},
                      "quality", row["passed"], flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["source_integrity_after"] = sources(extra)
    result["source_integrity_pass"] = before == result["source_integrity_after"]
    result["quality_audit_pass"] = all(row["passed"] for row in result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["source_integrity_pass"] and result["quality_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
