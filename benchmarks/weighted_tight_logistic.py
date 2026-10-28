"""Compare complete weighted GPU fits with accuracy-matched stock sklearn.

Both timed fits receive the same contiguous float64 original inputs. Fixtures
start as float32, then are exactly widened before timing. The independent,
untimed reference uses centered float64 inputs and is evaluated in the original
coordinates. All samples, warnings, routes and quality results are retained.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from .hessian_logistic import Case, fit_once, independent_score, sources
from .logistic_gpu import record_environment
from .sketch_weighted import TIGHT_CASES, benchmark_dataset, tight_reference_usable


def audit_variant(variant, scores, details, tol, objective_limit):
    """Check every recorded fit against the independently verified reference."""
    failures = []
    for i, (score, detail) in enumerate(zip(scores, details, strict=True)):
        if variant == "current" and detail.get("backend") != "mlx":
            failures.append({"fit": i, "reason": "unexpected GPU route"})
        if any(w["category"] in ("ConvergenceWarning", "DeviceConvergenceWarning")
               for w in detail["warnings"]):
            failures.append({"fit": i, "reason": "reported nonconvergence"})
        if any(not item.get("converged", False) for item in detail.get("solutions", [])):
            failures.append({"fit": i, "reason": "solver did not report convergence"})
        gradient = score["gradient_max_abs_float64"]
        if not math.isfinite(gradient) or gradient > tol:
            failures.append({"fit": i, "reason": "original gradient exceeds requested tolerance"})
        gap = score["objective_relative_gap_to_sklearn"]
        if not math.isfinite(gap) or gap > objective_limit:
            failures.append({"fit": i, "reason": "objective exceeds tight reference limit"})
        agreement = score["prediction_agreement_sklearn"]
        if not math.isfinite(agreement) or agreement < 0.995:
            failures.append({"fit": i, "reason": "prediction agreement below 99.5%"})
    return {"passed": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(TIGHT_CASES))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--gpu-tol", type=float, default=1e-4)
    parser.add_argument("--sklearn-tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--objective-limit", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("bench/weighted-tight-logistic.json"))
    args = parser.parse_args()
    if (args.repeats < 1 or args.warmup < 1 or args.max_iter < 1 or
        args.gpu_tol <= 0 or args.sklearn_tol <= 0 or args.objective_limit <= 0):
        parser.error("repeats, warmup, iteration budget, tolerances and objective limit must be positive")
    extra = [Path(__file__).resolve(), Path("benchmarks/sketch_weighted.py").resolve()]
    before = sources(extra)
    result = {
        "method": "alternating complete fits on identical original float64 inputs, one BLAS thread; untimed centered float64 sklearn reference",
        "parameters": vars(args) | {"output": str(args.output)},
        "environment": record_environment(), "source_integrity_before": before, "cases": [],
    }
    with threadpool_limits(limits=1):
        for name in args.case or sorted(TIGHT_CASES):
            for seed in args.seed or [41, 83, 2027]:
                case = Case(name, 20_000, 1_024, seed=seed, weighted=True)
                X, y, weights, ratio = benchmark_dataset(case)
                X = np.ascontiguousarray(X, dtype=np.float64)
                samples = {variant: [] for variant in ("current", "sklearn")}
                details = {variant: [] for variant in samples}
                models = {variant: [] for variant in samples}
                for rep in range(args.warmup + args.repeats):
                    order = ("current", "sklearn") if rep % 2 == 0 else ("sklearn", "current")
                    for variant in order:
                        tol = args.gpu_tol if variant == "current" else args.sklearn_tol
                        model, elapsed, meta = fit_once(variant, X, y, weights, case, None, tol, args.max_iter)
                        if rep >= args.warmup:
                            samples[variant].append(elapsed * 1000)
                            details[variant].append(meta)
                            models[variant].append(model)

                center = X.mean(axis=0, dtype=np.float64)
                centered = X - center
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reference = SkLogisticRegression(C=case.C, tol=1e-10, max_iter=3000).fit(
                        centered, y, sample_weight=weights
                    )
                ref_model = SimpleNamespace(coef_=reference.coef_,
                                            intercept_=reference.intercept_ - reference.coef_ @ center)
                ref_score, ref_predictions = independent_score(ref_model, X, y, weights, case.C, ratio)
                ref_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                ref_valid = tight_reference_usable(ref_score, ref_warnings, args.gpu_tol)
                scores = {variant: [independent_score(model, X, y, weights, case.C, ratio,
                          (ref_score["objective_float64"], ref_predictions))[0] for model in group]
                          for variant, group in models.items()}
                audits = {variant: audit_variant(variant, scores[variant], details[variant],
                          args.gpu_tol if variant == "current" else args.sklearn_tol,
                          args.objective_limit) for variant in samples}
                row = {"case": asdict(case), "input_dtype": "float64 exact widening of generated float32",
                       "median_ms": {v: statistics.median(s) for v, s in samples.items()},
                       "samples_ms": samples, "fit_details": details, "quality_each_fit": scores,
                       "reference": {"score": ref_score, "warnings": ref_warnings,
                                     "n_iter": reference.n_iter_.tolist(), "valid": ref_valid},
                       "audit": audits, "passed": ref_valid and all(a["passed"] for a in audits.values())}
                result["cases"].append(row)
                print(name, seed, {v: round(ms, 3) for v, ms in row["median_ms"].items()},
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
