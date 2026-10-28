"""Check the wide correlated GPU solver at stronger and weaker regularization.

All timed GPU fits are retained, including warnings and failed convergence.
An untimed centered float64 sklearn Newton-CG fit supplies an independently
checked original-coordinate reference; an invalid reference fails the audit.
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

from .hessian_logistic import Case, audit_case, dataset, fit_once, independent_score, sources
from .logistic_gpu import record_environment
from .sketch_weighted import balanced_order, tight_reference_usable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--C", type=float, action="append", help="defaults to 0.1 and 10")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--reference-max-iter", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/rank-safe-regularization-controls-2026-10-27.json"))
    args = parser.parse_args()
    if (min(args.repeats, args.warmup, args.max_iter, args.reference_max_iter) < 1
        or any(not math.isfinite(v) or v <= 0 for v in [args.tol, *(args.C or [0.1, 10.0])])):
        parser.error("budgets, C, and tolerance must be positive and finite")
    extra = [Path(__file__).resolve(), Path(__file__).with_name("sketch_weighted.py").resolve()]
    before = sources(extra)
    result = {"method": "rotated full GPU fits, original float32 data, one BLAS thread; independent untimed centered float64 sklearn Newton-CG reference",
              "parameters": vars(args) | {"output": str(args.output)}, "environment": record_environment(),
              "source_integrity_before": before, "cases": []}
    variants = ("current", "sketch", "device")
    with threadpool_limits(limits=1):
        for C in args.C or [0.1, 10.0]:
            for weighted in (False, True):
                case = Case("weighted_ill_4096" if weighted else "ill_4096", 20_000, 4_096,
                            seed=args.seed, weighted=weighted, ill_conditioned=True, C=C)
                X, y, sw, ratio = dataset(case)
                mx.synchronize()
                start = time.perf_counter()
                device = (mx.array(X), mx.array(y), None if sw is None else mx.array(sw, mx.float32))
                mx.eval(*(value for value in device if value is not None))
                mx.synchronize()
                upload_ms = 1000 * (time.perf_counter() - start)
                samples, details, models = ({v: [] for v in variants} for _ in range(3))
                for repeat in range(args.warmup + args.repeats):
                    for variant in balanced_order(variants, repeat):
                        model, elapsed, metadata = fit_once(variant, X, y, sw, case, None,
                                                           args.tol, args.max_iter, device, 4, 4)
                        if repeat >= args.warmup:
                            samples[variant].append(1000 * elapsed)
                            details[variant].append(metadata)
                            models[variant].append(model)
                center = X.mean(axis=0, dtype=np.float64)
                centered = np.asarray(X, np.float64) - center
                reference_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reference = SkLogisticRegression(C=C, solver="newton-cg", tol=1e-10,
                                                       max_iter=args.reference_max_iter).fit(centered, y, sample_weight=sw)
                reference_ms = 1000 * (time.perf_counter() - reference_start)
                restored = SimpleNamespace(coef_=reference.coef_,
                                           intercept_=reference.intercept_ - reference.coef_ @ center)
                reference_score, reference_predictions = independent_score(restored, X, y, sw, C, ratio)
                caught_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                reference_valid = tight_reference_usable(reference_score, caught_warnings, args.tol)
                scored = {v: [independent_score(m, X, y, sw, C, ratio,
                            (reference_score["objective_float64"], reference_predictions)) for m in group]
                          for v, group in models.items()}
                scores = {v: [entry[0] for entry in group] for v, group in scored.items()}
                for group in scores.values():
                    for score in group:
                        score["objective_relative_gap_to_trusted_reference"] = score["objective_relative_gap_to_sklearn"]
                        score["prediction_agreement_trusted_reference"] = score["prediction_agreement_sklearn"]
                reference_metadata = {"warnings": caught_warnings, "n_iter": reference.n_iter_.tolist(),
                                      "solver": "newton-cg", "tol": 1e-10, "fit_ms": reference_ms,
                                      "valid": reference_valid, "score": reference_score}
                scores_for_audit = scores | {"sklearn": [reference_score]}
                details_for_audit = details | {"sklearn": [reference_metadata]}
                audit = audit_case(case, scores_for_audit, details_for_audit, args.tol,
                                   reference_score["objective_float64"] if reference_valid else None)
                if not reference_valid:
                    audit["passed"] = False
                    audit["failures"].append({"variant": "benchmark", "fit": None,
                                               "reason": "tight Newton-CG reference failed its independent gradient or warning check"})
                predictions = []
                for index, model in enumerate(models["device"]):
                    actual = np.asarray(model.predict(device[0])).reshape(-1).astype(bool)
                    check = {"agreement_with_reference": float(np.mean(actual == reference_predictions)),
                             "agreement_with_scored_model": float(np.mean(actual == scored["device"][index][1]))}
                    predictions.append(check)
                    if check["agreement_with_reference"] < .995 or check["agreement_with_scored_model"] < .999:
                        audit["passed"] = False
                        audit["failures"].append({"variant": "device", "fit": index,
                                                   "reason": "actual device predictions failed audit"})
                row = {"case": asdict(case), "samples_ms": samples,
                       "median_ms": {v: statistics.median(values) for v, values in samples.items()},
                       "device_upload_ms": upload_ms, "fit_details": details, "quality_each_fit": scores,
                       "reference": reference_metadata, "device_prediction_audit": predictions,
                       "quality_audit": audit, "passed": audit["passed"]}
                result["cases"].append(row)
                print(case.name, "C", C, {v: round(ms, 3) for v, ms in row["median_ms"].items()},
                      "quality", audit["passed"], "reference", reference_valid, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                del centered
    result["source_integrity_after"] = sources(extra)
    result["source_integrity_pass"] = before == result["source_integrity_after"]
    result["quality_audit_pass"] = all(case["passed"] for case in result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["source_integrity_pass"] and result["quality_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
