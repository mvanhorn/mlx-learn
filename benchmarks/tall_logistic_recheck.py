"""Repeat the three float32 datasets from the original native-solver table.

Reuse its exact seed, data generator, fit boundary, variant order and parameters.
Audit every measured fit in original coordinates in float64 after timing, with
an additional independently converged reference. No thread limit is imposed,
matching the original panel. Run alone, without concurrent GPU benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from mlxlearn._solvers import native_logistic

from . import logistic_native
from .hessian_logistic import independent_score, sources
from .logistic_gpu import make_data, record_environment, summarize

CASE_NAMES = ("binary_tall", "binary_tall_wide", "binary_large")


def fit_once(variant, X, y, weights):
    # The estimator now prepares native-only initial statistics through an
    # imported selector. Disable that binding too for the fused control; the
    # original harness predates this preparation stage. Retain shared input
    # validation, upload and centering, all inside the same complete-fit timer.
    estimator = logistic_native.estimator_module
    original_select = estimator._use_native_solver
    if variant == "prior_fused":
        estimator._use_native_solver = lambda *args, **kwargs: False
    try:
        return logistic_native.fit_once(
            variant, X, y, weights, max_iter=200, tol=1e-4, native_module=native_logistic
        )
    finally:
        estimator._use_native_solver = original_select


def run_case(case, *, repeats, warmup):
    X, y, weights = make_data(case, seed=41, dtype="float32")
    variants = logistic_native.VARIANTS
    samples, details, models = ({v: [] for v in variants} for _ in range(3))
    warm_calls, orders = [], []
    for _ in range(warmup):
        for variant in variants:
            elapsed, _, metadata = fit_once(variant, X, y, weights)
            warm_calls.append({"variant": variant, "seconds": elapsed, **metadata})
    for repeat in range(repeats):
        base = variants if repeat % 2 == 0 else variants[::-1]
        offset = (repeat // 2) % len(variants)
        order = base[offset:] + base[:offset]
        orders.append(order)
        for variant in order:
            elapsed, model, metadata = fit_once(variant, X, y, weights)
            samples[variant].append(elapsed)
            models[variant].append(model)
            details[variant].append(metadata)

    # All reference work and scoring occur after the final timed fit for a case.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        reference = SkLogisticRegression(C=1.0, tol=1e-10, max_iter=3000).fit(
            X.astype(np.float64), y
        )
        reference_ms = (time.perf_counter() - start) * 1000
    ref_score, ref_predictions = independent_score(reference, X, y, weights, 1.0, None)
    reference_ok = not caught and ref_score["gradient_max_abs_float64"] <= 1e-7
    ref_value = ref_score["objective_float64"]
    scores, failures = {}, []
    if not reference_ok:
        failures.append({"reason": "independent reference did not pass its own accuracy gate"})
    for variant in variants:
        scores[variant] = []
        for index, (model, detail) in enumerate(zip(models[variant], details[variant], strict=True)):
            score, predictions = independent_score(model, X, y, weights, 1.0, None)
            actual = model.predict(X).astype(bool)
            score.update(
                objective_relative_gap_to_tight_reference=(score["objective_float64"] - ref_value)
                / max(abs(ref_value), np.finfo(float).tiny),
                prediction_agreement_tight_reference=float(np.mean(predictions == ref_predictions)),
                actual_prediction_agreement_tight_reference=float(np.mean(actual == ref_predictions)),
                actual_prediction_agreement_scored_model=float(np.mean(actual == predictions)),
            )
            checks = {
                "objective": math.isfinite(score["objective_relative_gap_to_tight_reference"])
                and abs(score["objective_relative_gap_to_tight_reference"]) <= 1e-5,
                "gradient": score["gradient_max_abs_float64"] <= 1e-4,
                "predictions": score["actual_prediction_agreement_tight_reference"] >= .995
                and score["actual_prediction_agreement_scored_model"] >= .999,
                "convergence": not detail["warnings"]
                and all(s["converged"] for s in detail["solutions"]),
                "backend": detail["backend"] == ("sklearn" if variant == "sklearn" else "mlx"),
                "implementation": detail["implementation"] == {
                    "prior_fused": "fused_mlx", "native_gpu": "native_metal", "sklearn": "sklearn"
                }[variant],
            }
            score["checks"] = checks
            scores[variant].append(score)
            failures.extend({"variant": variant, "fit": index, "reason": name}
                            for name, passed in checks.items() if not passed)
    timings = {variant: summarize(values) for variant, values in samples.items()}
    return {
        "case": asdict(case), "seed": 41, "dtype": str(X.dtype),
        "input_sha256": {"X": hashlib.sha256(memoryview(X)).hexdigest(),
                         "y": hashlib.sha256(memoryview(y)).hexdigest()},
        "parameters": {"C": 1.0, "tol": 1e-4, "max_iter": 200, "fit_intercept": True},
        "warmup_calls_not_process_cold": warm_calls, "timed_orders": orders,
        "timings": timings, "fit_details": details, "quality_each_fit": scores,
        "tight_reference": {"quality": ref_score, "valid": reference_ok,
                            "n_iter": reference.n_iter_.tolist(), "fit_ms": reference_ms,
                            "parameters": {"C": 1.0, "tol": 1e-10, "max_iter": 3000},
                            "warnings": [{"category": w.category.__name__, "message": str(w.message)}
                                         for w in caught]},
        "native_speedup_over_sklearn": timings["sklearn"]["median_ms"] / timings["native_gpu"]["median_ms"],
        "native_speedup_over_prior_fused": timings["prior_fused"]["median_ms"] / timings["native_gpu"]["median_ms"],
        "quality_audit": {"passed": not failures, "failures": failures},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/tall-logistic-recheck-2026-10-27.json"))
    args = parser.parse_args()
    if min(args.repeats, args.warmup) < 1:
        parser.error("repeats and warmup must be positive")
    historical_path = Path(__file__).with_name("results") / "native-logistic-2026-10-26.json"
    historical = json.loads(historical_path.read_text())
    extra = [Path(__file__).resolve(), Path(logistic_native.__file__).resolve(), historical_path.resolve()]
    before = sources(extra)
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": __doc__, "repeats": args.repeats, "warmup": args.warmup,
        "requested_thread_limit": None, "environment": record_environment(),
        "source_integrity_before": before,
        "baseline_boundary": "prior_fused disables native selection in solver and estimator preparation, sharing current validation, upload and centering; not a frozen historical checkout",
        "historical_comparison": {"path": str(historical_path), "results": [
            r for r in historical["results"] if r["dtype"] == "float32" and r["case"]["name"] in CASE_NAMES
        ]},
        "results": [],
    }
    cases = {case.name: case for case in logistic_native.PROFILES["full"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in CASE_NAMES:
        row = run_case(cases[name], repeats=args.repeats, warmup=args.warmup)
        result["results"].append(row)
        print(name, {v: round(t["median_ms"], 3) for v, t in row["timings"].items()},
              "quality", row["quality_audit"]["passed"], flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["source_integrity_after"] = sources(extra)
    result["source_integrity_pass"] = before == result["source_integrity_after"]
    result["quality_audit_pass"] = all(r["quality_audit"]["passed"] for r in result["results"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["quality_audit_pass"] and result["source_integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
