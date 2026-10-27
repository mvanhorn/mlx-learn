"""Audit general rank dependencies across the L-BFGS and Newton GPU routes.

The exact relation X[:, last] = X[:, a] + X[:, b] crosses feature blocks and
cannot be checked by paired-column fixtures. Unscaled weakly regularized cases
exercise the existing L-BFGS route; scaled C=1 cases exercise the hybrid Newton
route. All fitted parameters are independently scored on original float32
values in float64. No timing, warning, or failed reference is discarded.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from threadpoolctl import threadpool_limits

from .hessian_logistic import Case, dataset, fit_once, independent_score, sources
from .logistic_gpu import record_environment
from .sketch_weighted import balanced_order, tight_reference_usable


@dataclass(frozen=True)
class RankCase:
    name: str
    rows: int = 20_000
    features: int = 4_096
    seed: int = 41
    weighted: bool = False
    scaled: bool = False
    C: float = 1e6


CASES = (
    RankCase("general_rank_weak_4096"),
    RankCase("general_rank_weak_weighted_4096", weighted=True),
    RankCase("general_rank_scaled_4096", scaled=True, C=1.0),
    RankCase("general_rank_scaled_weighted_4096", weighted=True, scaled=True, C=1.0),
)


def rank_dataset(case):
    """Generate a non-pairwise, exactly represented three-column dependency."""
    if case.rows < 2 or case.features < 5:
        raise ValueError("the fixture needs at least two rows and five features")
    base = Case(case.name, case.rows, case.features, seed=case.seed)
    X, _, _, _ = dataset(base)
    if case.scaled:
        X *= np.logspace(-1, 1, case.features, dtype=np.float32)
    rng = np.random.default_rng(case.seed + 231)
    a = rng.integers(-4096, 4097, case.rows, dtype=np.int32).astype(np.float32) / np.float32(2048)
    b = rng.integers(-4096, 4097, case.rows, dtype=np.int32).astype(np.float32) / np.float32(2048)
    columns = (1, case.features // 2 - 1, case.features - 1)
    X[:, columns[0]], X[:, columns[1]], X[:, columns[2]] = a, b, a + b
    y = (a + 0.3 * b + rng.normal(scale=2.0, size=case.rows) > 0).astype(np.int32)
    sw = None
    if case.weighted:
        sw = rng.uniform(0.25, 3.0, case.rows).astype(np.float32)
        sw[::17] = 0
    return X, y, sw, columns


def coefficient_null_ratio(model, columns):
    coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    a, b, last = columns
    denominator = math.sqrt(3.0) * max(float(np.linalg.norm(coefficient)), np.finfo(float).tiny)
    return float(abs(coefficient[last] - coefficient[a] - coefficient[b]) / denominator)


def observed_newton(variant, detail):
    if variant == "device":
        return detail.get("newton_iterations", 0) > 0 and detail.get("newton_hvp_count", 0) > 0
    return any(call.get("n_iter", 0) > 0 and call.get("hvp_count", 0) > 0
               for call in detail.get("newton_calls", []))


def audit_rank_fits(variant, scores, details, *, tol, expected_newton, reference_valid):
    """Check every recorded fit without making an invalid reference usable."""
    failures = []
    if not scores or len(scores) != len(details):
        return {"passed": False, "failures": [{"fit": None, "reason": "missing or unpaired fits"}]}
    if not reference_valid:
        failures.append({"fit": None, "reason": "tight reference is not independently usable"})
    for index, (score, detail) in enumerate(zip(scores, details, strict=True)):
        if variant != "sklearn" and detail.get("backend") not in ("mlx", "mlx_device"):
            failures.append({"fit": index, "reason": "unexpected execution backend"})
        if (not detail.get("converged", True)
            or any(not state.get("converged", False) for state in detail.get("solutions", []))
            or any(w["category"] in ("ConvergenceWarning", "DeviceConvergenceWarning")
                   for w in detail.get("warnings", []))):
            failures.append({"fit": index, "reason": "fit reports nonconvergence"})
        checks = (
            ("gradient_max_abs_float64", tol, "original-coordinate gradient exceeds tolerance"),
            ("three_column_null_ratio", 2e-5, "general null-space coefficient artifact"),
            ("objective_relative_gap_to_tight_reference", 1e-5, "objective exceeds tight reference gate"),
        )
        for key, limit, reason in checks:
            value = score.get(key, float("nan"))
            if not math.isfinite(value) or value > limit:
                failures.append({"fit": index, "reason": reason})
        if not math.isfinite(score["objective_float64"]):
            failures.append({"fit": index, "reason": "nonfinite objective"})
        agreement = score.get("prediction_agreement_tight_reference", float("nan"))
        if not math.isfinite(agreement) or agreement < .995:
            failures.append({"fit": index, "reason": "prediction disagreement with tight reference"})
        if variant != "sklearn" and observed_newton(variant, detail) != expected_newton:
            failures.append({"fit": index, "reason": "expected solver route was not exercised"})
        if variant == "sketch" and not any(c.get("seeded") and c.get("pairs", 0) > 0
                                             for c in detail.get("sketch_calls", [])):
            failures.append({"fit": index, "reason": "host sketch was not exercised"})
        if variant == "device" and not detail.get("sketch_pairs"):
            failures.append({"fit": index, "reason": "device sketch was not exercised"})
    return {"passed": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c.name for c in CASES])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--reference-max-iter", type=int, default=5000)
    parser.add_argument("--reference-solver", choices=("lbfgs", "newton-cg"), default="lbfgs")
    parser.add_argument("--output", type=Path, default=Path("bench/rank-safe-logistic.json"))
    args = parser.parse_args()
    if (args.rows < 4096 or min(args.repeats, args.warmup, args.max_iter, args.reference_max_iter) < 1
        or not math.isfinite(args.tol) or args.tol <= 0):
        parser.error("positive budgets and tolerance, and at least 4096 rows, are required")
    extra = [Path(__file__).resolve(), Path(__file__).with_name("sketch_weighted.py").resolve()]
    before = sources(extra)
    result = {"method": "rotated complete fits; one BLAS thread; original float32 inputs; general exact rank relation; untimed tight centered float64 reference",
              "parameters": vars(args) | {"output": str(args.output)},
              "environment": record_environment(), "source_integrity_before": before, "cases": []}
    variants = ("current", "sketch", "device", "sklearn")
    with threadpool_limits(limits=1):
        for template in CASES:
            if args.case and template.name not in args.case:
                continue
            case = replace(template, rows=args.rows, seed=args.seed)
            X, y, weights, columns = rank_dataset(case)
            X64 = np.asarray(X, dtype=np.float64)
            exact_relation = bool(np.all(X64[:, columns[2]] == X64[:, columns[0]] + X64[:, columns[1]]))
            if not exact_relation:
                raise RuntimeError("rank fixture lost its exact original-coordinate dependency")
            mx.synchronize()
            start = time.perf_counter()
            device = (mx.array(X), mx.array(y), None if weights is None else mx.array(weights))
            mx.eval(*(v for v in device if v is not None))
            mx.synchronize()
            upload_ms = (time.perf_counter() - start) * 1000
            samples, details, models = ({v: [] for v in variants} for _ in range(3))
            for rep in range(args.warmup + args.repeats):
                for variant in balanced_order(variants, rep):
                    model, elapsed, detail = fit_once(variant, X, y, weights, case, None,
                                                      args.tol, args.max_iter, device, 4, 4)
                    if rep >= args.warmup:
                        samples[variant].append(elapsed * 1000)
                        details[variant].append(detail)
                        models[variant].append(model)
            center = X64.mean(axis=0)
            reference_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reference = SkLogisticRegression(C=case.C, tol=1e-10, max_iter=args.reference_max_iter,
                                                 solver=args.reference_solver).fit(
                    X64 - center, y, sample_weight=weights,
                )
            reference_ms = (time.perf_counter() - reference_start) * 1000
            reference_model = SimpleNamespace(coef_=reference.coef_,
                                               intercept_=reference.intercept_ - reference.coef_ @ center)
            reference_score, reference_predictions = independent_score(reference_model, X, y, weights, case.C, None)
            reference_score["three_column_null_ratio"] = coefficient_null_ratio(reference_model, columns)
            reference_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
            reference_ok = (tight_reference_usable(reference_score, reference_warnings, args.tol)
                            and reference_score["three_column_null_ratio"] <= 2e-5)
            scored = {v: [independent_score(m, X, y, weights, case.C, None) for m in group]
                      for v, group in models.items()}
            scores = {v: [entry[0] for entry in group] for v, group in scored.items()}
            for variant in variants:
                for index, score in enumerate(scores[variant]):
                    score["three_column_null_ratio"] = coefficient_null_ratio(models[variant][index], columns)
                    score["objective_relative_gap_to_tight_reference"] = (
                        (score["objective_float64"] - reference_score["objective_float64"])
                        / max(abs(reference_score["objective_float64"]), np.finfo(float).tiny)
                    )
                    score["prediction_agreement_tight_reference"] = float(np.mean(scored[variant][index][1] == reference_predictions))
            audits = {v: audit_rank_fits(v, scores[v], details[v], tol=args.tol,
                                        expected_newton=case.scaled, reference_valid=reference_ok) for v in variants}
            prediction_audit = []
            for index, model in enumerate(models["device"]):
                actual = np.asarray(model.predict(device[0])).reshape(-1).astype(bool)
                entry = {"agreement_with_reference": float(np.mean(actual == reference_predictions)),
                         "agreement_with_scored_model": float(np.mean(actual == scored["device"][index][1]))}
                prediction_audit.append(entry)
                if entry["agreement_with_reference"] < .995 or entry["agreement_with_scored_model"] < .999:
                    audits["device"]["passed"] = False
                    audits["device"]["failures"].append({"fit": index, "reason": "actual device predictions failed audit"})
            row = {"case": asdict(case), "dependent_columns": columns, "exact_float64_relation": exact_relation,
                   "device_upload_ms": upload_ms, "samples_ms": samples,
                   "median_ms": {v: statistics.median(s) for v, s in samples.items()},
                   "fit_details": details, "quality_each_fit": scores, "device_prediction_audit": prediction_audit,
                   "reference": {"score": reference_score, "warnings": reference_warnings,
                                 "n_iter": reference.n_iter_.tolist(), "valid": reference_ok, "fit_ms": reference_ms,
                                 "solver": args.reference_solver},
                   "audits": audits,
                   "passed": reference_ok and all(audits[v]["passed"] for v in ("current", "sketch", "device"))}
            result["cases"].append(row)
            print(case.name, {v: round(t, 3) for v, t in row["median_ms"].items()},
                  "quality", row["passed"], "default_sklearn_quality", audits["sklearn"]["passed"], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["source_integrity_after"] = sources(extra)
    result["source_integrity_pass"] = before == result["source_integrity_after"]
    result["quality_audit_pass"] = all(c["passed"] for c in result["cases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["source_integrity_pass"] and result["quality_audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
